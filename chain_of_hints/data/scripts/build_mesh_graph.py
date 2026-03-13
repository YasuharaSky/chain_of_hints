#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, Iterator, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def parse_pipe_values(value: object) -> List[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    return [item for item in str(value).split("|") if item]


def get_depth3_ancestors(tree_numbers: Sequence[str]) -> List[str]:
    ancestors = set()
    for tree_number in tree_numbers:
        parts = tree_number.split(".")
        if len(parts) >= 3:
            ancestors.add(".".join(parts[:3]))
        elif parts:
            ancestors.add(tree_number)
    return sorted(ancestors)


def write_csv_gz(path: Path, rows: Iterable[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_nodes(descriptor_df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, row in descriptor_df.iterrows():
        tree_numbers = parse_pipe_values(row["tree_numbers"])
        tree_roots = sorted({tree_number[0] for tree_number in tree_numbers})
        scope_note = row["scope_note"] if isinstance(row["scope_note"], str) else ""
        text_for_embedding = scope_note.strip() if scope_note.strip() else row["descriptor_name"]
        depth3_ancestors = get_depth3_ancestors(tree_numbers)
        records.append(
            {
                "descriptor_ui": row["descriptor_ui"],
                "descriptor_name": row["descriptor_name"],
                "text_for_embedding": text_for_embedding,
                "tree_numbers": "|".join(tree_numbers),
                "tree_roots": "|".join(tree_roots),
                "primary_root": tree_roots[0] if tree_roots else "",
                "num_tree_roots": len(tree_roots),
                "depth3_ancestors": "|".join(depth3_ancestors),
                "num_depth3_ancestors": len(depth3_ancestors),
            }
        )
    return pd.DataFrame(records)


def iter_tree_edges(tree_edge_df: pd.DataFrame) -> Iterator[Dict[str, object]]:
    seen_pairs: Set[Tuple[str, str]] = set()
    for _, row in tree_edge_df.iterrows():
        source, target = sorted([row["parent_ui"], row["child_ui"]])
        pair = (source, target)
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        yield {
            "source_ui": source,
            "target_ui": target,
            "edge_type": "tree_parent_child",
            "support": 1,
            "detail": f"{row['parent_tree_number']}->{row['child_tree_number']}",
        }


def iter_shared_depth3_edges(nodes_df: pd.DataFrame) -> Iterator[Dict[str, object]]:
    ancestor_to_terms: Dict[str, Set[str]] = defaultdict(set)
    for _, row in nodes_df.iterrows():
        for ancestor in parse_pipe_values(row["depth3_ancestors"]):
            ancestor_to_terms[ancestor].add(row["descriptor_ui"])

    for ancestor, term_set in ancestor_to_terms.items():
        terms = sorted(term_set)
        if len(terms) < 2:
            continue
        for source_ui, target_ui in combinations(terms, 2):
            yield {
                "source_ui": source_ui,
                "target_ui": target_ui,
                "edge_type": "shared_depth3_ancestor",
                "support": 1,
                "detail": ancestor,
            }


def build_ancestor_to_terms(nodes_df: pd.DataFrame) -> Dict[str, Set[str]]:
    ancestor_to_terms: Dict[str, Set[str]] = defaultdict(set)
    for _, row in nodes_df.iterrows():
        for ancestor in parse_pipe_values(row["depth3_ancestors"]):
            ancestor_to_terms[ancestor].add(row["descriptor_ui"])
    return ancestor_to_terms


def build_multiroot_bridge_edges(nodes_df: pd.DataFrame) -> List[Dict[str, object]]:
    ancestor_to_terms = build_ancestor_to_terms(nodes_df)
    edge_support: DefaultDict[Tuple[str, str], int] = defaultdict(int)
    edge_details: DefaultDict[Tuple[str, str], List[str]] = defaultdict(list)

    multiroot_df = nodes_df.loc[nodes_df["num_tree_roots"] > 1].copy()
    for _, row in multiroot_df.iterrows():
        bridge_ui = row["descriptor_ui"]
        bridge_name = row["descriptor_name"]
        by_root: Dict[str, Set[str]] = defaultdict(set)
        for ancestor in parse_pipe_values(row["depth3_ancestors"]):
            by_root[ancestor[0]].update(ancestor_to_terms[ancestor])

        roots = sorted(by_root)
        if len(roots) < 2:
            continue

        for left_root, right_root in combinations(roots, 2):
            left_terms = by_root[left_root]
            right_terms = by_root[right_root]
            if not left_terms or not right_terms:
                continue

            bridge_detail = f"{bridge_ui}:{bridge_name}:{left_root}->{right_root}"
            for left_ui in left_terms:
                if left_ui == bridge_ui:
                    continue
                for right_ui in right_terms:
                    if right_ui == bridge_ui or left_ui == right_ui:
                        continue
                    source_ui, target_ui = sorted((left_ui, right_ui))
                    pair = (source_ui, target_ui)
                    edge_support[pair] += 1
                    if len(edge_details[pair]) < 3 and bridge_detail not in edge_details[pair]:
                        edge_details[pair].append(bridge_detail)

    rows = []
    for source_ui, target_ui in sorted(edge_support):
        rows.append(
            {
                "source_ui": source_ui,
                "target_ui": target_ui,
                "edge_type": "multiroot_bridge",
                "support": edge_support[(source_ui, target_ui)],
                "detail": " ; ".join(edge_details[(source_ui, target_ui)]),
            }
        )
    return rows


def build_knn_edges(
    embedding_npy: Path,
    metadata_csv: Path,
    k: int,
) -> List[Dict[str, object]]:
    if k <= 0:
        return []

    metadata_df = pd.read_csv(metadata_csv).fillna("")
    embeddings = np.load(embedding_npy)
    if len(metadata_df) != len(embeddings):
        raise ValueError("embedding metadata row count does not match embedding matrix row count")

    neighbor_count = min(k + 1, len(metadata_df))
    nn_model = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=neighbor_count, n_jobs=-1)
    nn_model.fit(embeddings)
    distances, indices = nn_model.kneighbors(embeddings, return_distance=True)

    edge_support: DefaultDict[Tuple[str, str], int] = defaultdict(int)
    edge_max_cosine: Dict[Tuple[str, str], float] = {}
    edge_min_rank: Dict[Tuple[str, str], int] = {}
    descriptor_uis = metadata_df["descriptor_ui"].tolist()

    for source_idx, source_ui in enumerate(descriptor_uis):
        for rank, (target_idx, distance) in enumerate(zip(indices[source_idx], distances[source_idx]), start=0):
            if target_idx == source_idx:
                continue
            if rank > k:
                break
            target_ui = descriptor_uis[target_idx]
            pair = tuple(sorted((source_ui, target_ui)))
            cosine = float(1.0 - distance)
            edge_support[pair] += 1
            edge_max_cosine[pair] = max(edge_max_cosine.get(pair, float("-inf")), cosine)
            edge_min_rank[pair] = min(edge_min_rank.get(pair, 10**9), rank)

    rows = []
    for source_ui, target_ui in sorted(edge_support):
        support = edge_support[(source_ui, target_ui)]
        max_cosine = edge_max_cosine[(source_ui, target_ui)]
        min_rank = edge_min_rank[(source_ui, target_ui)]
        rows.append(
            {
                "source_ui": source_ui,
                "target_ui": target_ui,
                "edge_type": "embedding_knn",
                "support": support,
                "detail": f"max_cosine={max_cosine:.6f};min_rank={min_rank};mutual={support > 1}",
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a denser MeSH graph from descriptor and tree-edge tables.")
    parser.add_argument(
        "--descriptor-csv",
        type=Path,
        default=Path("data/processed/mesh/mesh_descriptors_2026.csv"),
        help="Processed MeSH descriptor CSV.",
    )
    parser.add_argument(
        "--tree-edge-csv",
        type=Path,
        default=Path("data/processed/mesh/mesh_tree_edges_2026.csv"),
        help="Processed MeSH tree-edge CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/mesh"),
        help="Output directory for graph files.",
    )
    parser.add_argument(
        "--knn-embedding-npy",
        type=Path,
        default=None,
        help="Optional embedding matrix used to build a k-NN graph.",
    )
    parser.add_argument(
        "--knn-metadata-csv",
        type=Path,
        default=None,
        help="Metadata CSV aligned with the embedding matrix.",
    )
    parser.add_argument(
        "--knn-k",
        type=int,
        default=0,
        help="Number of nearest neighbors per node for optional embedding k-NN edges.",
    )
    parser.add_argument(
        "--knn-edge-output-csv-gz",
        type=Path,
        default=None,
        help="Optional output path for embedding k-NN edges.",
    )
    parser.add_argument(
        "--stats-json",
        type=Path,
        default=None,
        help="Optional custom graph stats output path.",
    )
    args = parser.parse_args()

    descriptor_df = pd.read_csv(args.descriptor_csv).fillna("")
    tree_edge_df = pd.read_csv(args.tree_edge_csv).fillna("")

    nodes_df = build_nodes(descriptor_df)
    nodes_path = args.output_dir / "mesh_graph_nodes_2026.csv"
    nodes_df.to_csv(nodes_path, index=False)

    tree_edges_path = args.output_dir / "mesh_graph_tree_edges_2026.csv.gz"
    shared_edges_path = args.output_dir / "mesh_graph_shared_depth3_edges_2026.csv.gz"
    multiroot_bridge_edges_path = args.output_dir / "mesh_graph_multiroot_bridge_edges_2026.csv.gz"
    knn_edges_path = args.knn_edge_output_csv_gz
    if args.knn_k > 0 and knn_edges_path is None:
        knn_edges_path = args.output_dir / f"mesh_graph_knn_k{args.knn_k}_edges_2026.csv.gz"

    write_csv_gz(
        tree_edges_path,
        iter_tree_edges(tree_edge_df),
        ["source_ui", "target_ui", "edge_type", "support", "detail"],
    )
    write_csv_gz(
        shared_edges_path,
        iter_shared_depth3_edges(nodes_df),
        ["source_ui", "target_ui", "edge_type", "support", "detail"],
    )
    multiroot_bridge_rows = build_multiroot_bridge_edges(nodes_df)
    write_csv_gz(
        multiroot_bridge_edges_path,
        multiroot_bridge_rows,
        ["source_ui", "target_ui", "edge_type", "support", "detail"],
    )
    knn_count = 0
    if args.knn_k > 0:
        if args.knn_embedding_npy is None or args.knn_metadata_csv is None:
            raise ValueError("knn_embedding_npy and knn_metadata_csv are required when knn_k > 0")
        knn_rows = build_knn_edges(args.knn_embedding_npy, args.knn_metadata_csv, args.knn_k)
        write_csv_gz(
            knn_edges_path,
            knn_rows,
            ["source_ui", "target_ui", "edge_type", "support", "detail"],
        )
        knn_count = len(knn_rows)

    shared_count = sum(1 for _ in gzip.open(shared_edges_path, "rt", encoding="utf-8")) - 1
    tree_count = sum(1 for _ in gzip.open(tree_edges_path, "rt", encoding="utf-8")) - 1
    multiroot_bridge_count = len(multiroot_bridge_rows)
    experiment_graph_components = [
        str(tree_edges_path),
        str(shared_edges_path),
        str(multiroot_bridge_edges_path),
    ]
    if args.knn_k > 0:
        experiment_graph_components = [str(tree_edges_path), str(knn_edges_path)]
    stats = {
        "num_nodes": int(len(nodes_df)),
        "num_tree_edges": int(tree_count),
        "num_shared_depth3_edges": int(shared_count),
        "num_multiroot_bridge_edges": int(multiroot_bridge_count),
        "num_knn_edges": int(knn_count),
        "num_multi_root_nodes": int((nodes_df["num_tree_roots"] > 1).sum()),
        "experiment_graph_components": experiment_graph_components,
    }
    stats_path = args.stats_json or (args.output_dir / "mesh_graph_stats_2026.json")
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"Wrote {nodes_path}")
    print(f"Wrote {tree_edges_path}")
    print(f"Wrote {shared_edges_path}")
    print(f"Wrote {multiroot_bridge_edges_path}")
    if args.knn_k > 0:
        print(f"Wrote {knn_edges_path}")
    print(f"Wrote {stats_path}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
