#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence

import requests

API_URL = "https://en.wikipedia.org/w/api.php"
PREFIX = "Wikipedia:Vital articles/Level/5"
USER_AGENT = "chain-of-hints-data-prep/0.1 (local research use)"
EXCLUDE_KEYWORDS = (
    "Article alerts",
    "Archive",
    "Candidates",
    "Removed articles",
)


def chunks(items: Sequence[str], size: int) -> Iterator[Sequence[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def request_json(session: requests.Session, params: Dict[str, object], pause_s: float) -> Dict[str, object]:
    max_attempts = 8
    for attempt in range(1, max_attempts + 1):
        try:
            response = session.get(API_URL, params=params, timeout=120)
        except requests.RequestException as exc:
            sleep_s = min(60.0, pause_s + 2 ** attempt)
            print(f"Request failed ({exc}). Sleeping {sleep_s:.1f}s before retry {attempt}/{max_attempts}.")
            time.sleep(sleep_s)
            continue

        if response.status_code == 429:
            retry_after = response.headers.get("retry-after")
            if retry_after and retry_after.isdigit():
                sleep_s = float(retry_after)
            else:
                sleep_s = min(60.0, pause_s + 2 ** attempt)
            print(f"Rate limited by Wikipedia API. Sleeping {sleep_s:.1f}s before retry {attempt}/{max_attempts}.")
            time.sleep(sleep_s)
            continue

        if response.status_code >= 500:
            sleep_s = min(60.0, pause_s + 2 ** attempt)
            print(f"Wikipedia API returned {response.status_code}. Sleeping {sleep_s:.1f}s before retry {attempt}/{max_attempts}.")
            time.sleep(sleep_s)
            continue

        response.raise_for_status()
        if pause_s:
            time.sleep(pause_s)
        return response.json()

    response.raise_for_status()
    raise RuntimeError("Unreachable")


def iter_subpages(session: requests.Session, pause_s: float) -> Iterator[str]:
    params: Dict[str, object] = {
        "action": "query",
        "list": "allpages",
        "apnamespace": 4,
        "apprefix": "Vital articles/Level/5/",
        "aplimit": "max",
        "format": "json",
    }
    while True:
        payload = request_json(session, params, pause_s)
        for page in payload["query"]["allpages"]:
            yield page["title"]
        if "continue" not in payload:
            break
        params.update(payload["continue"])


def keep_subpage(title: str) -> bool:
    if title == f"{PREFIX}/":
        return False
    return not any(keyword in title for keyword in EXCLUDE_KEYWORDS)


def topic_path(title: str) -> str:
    if not title.startswith(f"{PREFIX}/"):
        return ""
    return title[len(PREFIX) + 1 :]


def iter_list_page_links(session: requests.Session, title: str, pause_s: float) -> Iterator[str]:
    params: Dict[str, object] = {
        "action": "query",
        "prop": "links",
        "titles": title,
        "plnamespace": 0,
        "pllimit": "max",
        "format": "json",
    }
    while True:
        payload = request_json(session, params, pause_s)
        pages = payload["query"]["pages"]
        for page in pages.values():
            for link in page.get("links", []):
                yield link["title"]
        if "continue" not in payload:
            break
        params.update(payload["continue"])


def fetch_article_metadata(
    session: requests.Session,
    titles: Sequence[str],
    pause_s: float,
    batch_size: int,
    cache_path: Path,
    include_extracts: bool,
) -> Dict[str, Dict[str, str]]:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["title", "page_id", "wikidata_id", "url", "lead_summary"]
    cached_records: Dict[str, Dict[str, str]] = {}

    if cache_path.exists():
        with cache_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                cached_records[row["title"]] = row
        print(f"Loaded {len(cached_records)} cached metadata rows from {cache_path}")

    pending_titles = [title for title in titles if title not in cached_records]
    if not pending_titles:
        return cached_records

    write_header = not cache_path.exists()
    with cache_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for batch_index, batch in enumerate(chunks(list(pending_titles), batch_size), start=1):
            if batch_index % 100 == 0:
                print(f"Fetching metadata batch {batch_index} ({batch_index * batch_size} titles attempted)")

            batch_records: List[Dict[str, str]] = []
            props = "info|pageprops"
            if include_extracts:
                props += "|extracts"
            params = {
                "action": "query",
                "prop": props,
                "inprop": "url",
                "ppprop": "wikibase_item",
                "redirects": 1,
                "titles": "|".join(batch),
                "format": "json",
            }
            if include_extracts:
                params["exintro"] = 1
                params["explaintext"] = 1
            payload = request_json(session, params, pause_s)
            for page in payload["query"]["pages"].values():
                if "missing" in page:
                    continue
                extract = " ".join(page.get("extract", "").split())
                row = {
                    "title": page["title"],
                    "page_id": str(page["pageid"]),
                    "wikidata_id": page.get("pageprops", {}).get("wikibase_item", ""),
                    "url": page.get("fullurl", ""),
                    "lead_summary": extract,
                }
                batch_records.append(row)
                cached_records[row["title"]] = row

            if batch_records:
                writer.writerows(batch_records)
                handle.flush()

    return cached_records


def load_membership_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and preprocess Wikipedia Vital Articles Level 5.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/wikipedia"),
        help="Directory for processed CSV outputs.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw/wikipedia"),
        help="Directory for raw crawl artifacts.",
    )
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=0.1,
        help="Sleep interval between API calls.",
    )
    parser.add_argument(
        "--metadata-batch-size",
        type=int,
        default=50,
        help="Titles per metadata request.",
    )
    parser.add_argument(
        "--include-extracts",
        action="store_true",
        help="Fetch plaintext lead summaries in addition to IDs and URLs.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.raw_dir.mkdir(parents=True, exist_ok=True)

    session = build_session()
    membership_path = args.output_dir / "vital_articles_level5_membership.csv"
    if membership_path.exists():
        print(f"Reusing existing membership file: {membership_path}")
        membership_rows = load_membership_rows(membership_path)
        subpages = sorted({row["source_page"] for row in membership_rows})
    else:
        subpages = sorted(title for title in iter_subpages(session, args.pause_seconds) if keep_subpage(title))
        (args.raw_dir / "vital_articles_level5_subpages.json").write_text(
            json.dumps(subpages, indent=2),
            encoding="utf-8",
        )
        membership_rows = []
        for index, subpage in enumerate(subpages, start=1):
            current_path = topic_path(subpage)
            print(f"[{index}/{len(subpages)}] Parsing {subpage}")
            seen_titles = set()
            for article_title in iter_list_page_links(session, subpage, args.pause_seconds):
                if article_title in seen_titles:
                    continue
                seen_titles.add(article_title)
                membership_rows.append(
                    {
                        "source_page": subpage,
                        "topic_path": current_path,
                        "article_title": article_title,
                    }
                )

        with membership_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["source_page", "topic_path", "article_title"])
            writer.writeheader()
            writer.writerows(membership_rows)

    article_to_paths: Dict[str, set[str]] = defaultdict(set)
    article_to_pages: Dict[str, set[str]] = defaultdict(set)
    for row in membership_rows:
        article_to_paths[row["article_title"]].add(row["topic_path"])
        article_to_pages[row["article_title"]].add(row["source_page"])

    unique_titles = sorted(article_to_paths)
    print(f"Collected {len(unique_titles)} unique article titles across {len(subpages)} list pages.")
    metadata_by_title = fetch_article_metadata(
        session,
        unique_titles,
        args.pause_seconds,
        args.metadata_batch_size,
        args.raw_dir / (
            "vital_articles_level5_metadata_with_extracts_cache.csv"
            if args.include_extracts
            else "vital_articles_level5_metadata_cache.csv"
        ),
        args.include_extracts,
    )

    article_rows: List[Dict[str, str]] = []
    for title in unique_titles:
        info = metadata_by_title.get(title, {})
        article_rows.append(
            {
                "title": title,
                "page_id": info.get("page_id", ""),
                "wikidata_id": info.get("wikidata_id", ""),
                "url": info.get("url", ""),
                "topic_paths": " | ".join(sorted(article_to_paths[title])),
                "source_pages": " | ".join(sorted(article_to_pages[title])),
                "lead_summary": info.get("lead_summary", ""),
            }
        )

    articles_path = args.output_dir / "vital_articles_level5_articles.csv"
    with articles_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "title",
                "page_id",
                "wikidata_id",
                "url",
                "topic_paths",
                "source_pages",
                "lead_summary",
            ],
        )
        writer.writeheader()
        writer.writerows(article_rows)

    print(f"Wrote {membership_path}")
    print(f"Wrote {articles_path}")


if __name__ == "__main__":
    main()
