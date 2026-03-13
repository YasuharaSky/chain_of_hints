#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import gzip
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, List

import requests

USER_AGENT = "chain-of-hints-data-prep/0.1 (local research use)"
MESH_URL_TEMPLATE = "https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/desc{year}.gz"


def text_or_empty(element: ET.Element | None, path: str) -> str:
    if element is None:
        return ""
    node = element.find(path)
    if node is None or node.text is None:
        return ""
    return " ".join(node.text.split())


def ensure_download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        print(f"Reusing existing download: {destination}")
        return
    with requests.get(url, headers={"User-Agent": USER_AGENT}, stream=True, timeout=120) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            shutil.copyfileobj(response.raw, handle)
    print(f"Downloaded {destination}")


def parse_descriptor_records(xml_gz_path: Path) -> List[Dict[str, str]]:
    with gzip.open(xml_gz_path, "rb") as handle:
        tree = ET.parse(handle)
    root = tree.getroot()

    records: List[Dict[str, str]] = []
    for descriptor in root.findall("DescriptorRecord"):
        descriptor_ui = text_or_empty(descriptor, "DescriptorUI")
        descriptor_name = text_or_empty(descriptor, "DescriptorName/String")
        scope_note = text_or_empty(descriptor, "ConceptList/Concept/ScopeNote")
        annotation = text_or_empty(descriptor, "Annotation")
        tree_numbers = sorted(
            {
                node.text.strip()
                for node in descriptor.findall("TreeNumberList/TreeNumber")
                if node.text and node.text.strip()
            }
        )
        roots = sorted({tree_number[0] for tree_number in tree_numbers})
        entry_terms = sorted(
            {
                node.text.strip()
                for node in descriptor.findall("ConceptList/Concept/TermList/Term/String")
                if node.text and node.text.strip() and node.text.strip() != descriptor_name
            }
        )
        records.append(
            {
                "descriptor_ui": descriptor_ui,
                "descriptor_name": descriptor_name,
                "tree_numbers": "|".join(tree_numbers),
                "tree_roots": "|".join(roots),
                "scope_note": scope_note,
                "annotation": annotation,
                "entry_terms": "|".join(entry_terms),
            }
        )
    return records


def build_tree_edges(records: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    tree_to_descriptor: Dict[str, Dict[str, str]] = {}
    for record in records:
        for tree_number in filter(None, record["tree_numbers"].split("|")):
            tree_to_descriptor[tree_number] = record

    edges: List[Dict[str, str]] = []
    for tree_number, child_record in tree_to_descriptor.items():
        if "." not in tree_number:
            continue
        parent_tree_number = tree_number.rsplit(".", 1)[0]
        parent_record = tree_to_descriptor.get(parent_tree_number)
        if parent_record is None:
            continue
        edges.append(
            {
                "parent_tree_number": parent_tree_number,
                "parent_ui": parent_record["descriptor_ui"],
                "parent_name": parent_record["descriptor_name"],
                "child_tree_number": tree_number,
                "child_ui": child_record["descriptor_ui"],
                "child_name": child_record["descriptor_name"],
            }
        )
    return edges


def write_csv(path: Path, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and preprocess MeSH descriptor XML.")
    parser.add_argument("--year", type=int, default=2026, help="MeSH release year.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw/mesh"),
        help="Directory for raw downloads.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/mesh"),
        help="Directory for processed CSV outputs.",
    )
    args = parser.parse_args()

    url = MESH_URL_TEMPLATE.format(year=args.year)
    raw_path = args.raw_dir / f"desc{args.year}.gz"
    ensure_download(url, raw_path)

    records = parse_descriptor_records(raw_path)
    edges = build_tree_edges(records)

    descriptor_path = args.output_dir / f"mesh_descriptors_{args.year}.csv"
    edge_path = args.output_dir / f"mesh_tree_edges_{args.year}.csv"

    write_csv(
        descriptor_path,
        records,
        [
            "descriptor_ui",
            "descriptor_name",
            "tree_numbers",
            "tree_roots",
            "scope_note",
            "annotation",
            "entry_terms",
        ],
    )
    write_csv(
        edge_path,
        edges,
        [
            "parent_tree_number",
            "parent_ui",
            "parent_name",
            "child_tree_number",
            "child_ui",
            "child_name",
        ],
    )

    print(f"Wrote {descriptor_path}")
    print(f"Wrote {edge_path}")


if __name__ == "__main__":
    main()
