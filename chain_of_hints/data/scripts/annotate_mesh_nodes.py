#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Set

import pandas as pd


def parse_pipe_values(value: object) -> List[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    return [item for item in str(value).split("|") if item]


def compute_depth_stats(tree_numbers: List[str]) -> tuple[int, int, float]:
    if not tree_numbers:
        return 0, 0, 0.0
    depths = [tree_number.count(".") + 1 for tree_number in tree_numbers]
    return max(depths), min(depths), sum(depths) / len(depths)


def compute_graph_degrees(edge_paths: Sequence[Path]) -> Dict[str, int]:
    neighbors: Dict[str, Set[str]] = {}
    for edge_path in edge_paths:
        with gzip.open(edge_path, "rt", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                source_ui = row["source_ui"]
                target_ui = row["target_ui"]
                if source_ui == target_ui:
                    continue
                neighbors.setdefault(source_ui, set()).add(target_ui)
                neighbors.setdefault(target_ui, set()).add(source_ui)
    return {node_ui: len(node_neighbors) for node_ui, node_neighbors in neighbors.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate MeSH node table with depth and hub metrics.")
    parser.add_argument(
        "--node-csv",
        type=Path,
        default=Path("data/processed/mesh/mesh_graph_nodes_2026.csv"),
        help="Node CSV to update in place.",
    )
    parser.add_argument(
        "--edge-csv-gz",
        type=Path,
        nargs="+",
        default=[
            Path("data/processed/mesh/mesh_graph_tree_edges_2026.csv.gz"),
            Path("data/processed/mesh/mesh_graph_shared_depth3_edges_2026.csv.gz"),
            Path("data/processed/mesh/mesh_graph_multiroot_bridge_edges_2026.csv.gz"),
        ],
        help="Graph edge files used for degree and hub computation.",
    )
    parser.add_argument(
        "--hub-percent",
        type=float,
        default=0.01,
        help="Top degree fraction marked as hubs.",
    )
    parser.add_argument(
        "--stats-json",
        type=Path,
        default=Path("data/processed/mesh/mesh_node_annotation_stats_2026.json"),
        help="Where to write annotation stats.",
    )
    parser.add_argument(
        "--embedding-metadata-csv",
        type=Path,
        default=Path("data/processed/mesh/mesh_embeddings_BiomedNLP_PubMedBERT_base_uncased_abstract_fulltext_metadata.csv"),
        help="Embedding metadata CSV used for row-count validation.",
    )
    args = parser.parse_args()

    node_df = pd.read_csv(args.node_csv).fillna("")

    max_depths = []
    min_depths = []
    mean_depths = []
    for tree_numbers_raw in node_df["tree_numbers"]:
        tree_numbers = parse_pipe_values(tree_numbers_raw)
        max_depth, min_depth, mean_depth = compute_depth_stats(tree_numbers)
        max_depths.append(max_depth)
        min_depths.append(min_depth)
        mean_depths.append(mean_depth)

    degree_map = compute_graph_degrees(args.edge_csv_gz)
    node_df["max_depth"] = max_depths
    node_df["min_depth"] = min_depths
    node_df["mean_depth"] = mean_depths
    node_df["degree"] = node_df["descriptor_ui"].map(degree_map).fillna(0).astype(int)

    top_n = max(1, math.ceil(len(node_df) * args.hub_percent))
    hub_ranked = node_df.sort_values(["degree", "descriptor_ui"], ascending=[False, True]).copy()
    hub_uis = set(hub_ranked.head(top_n)["descriptor_ui"])
    node_df["is_hub"] = node_df["descriptor_ui"].isin(hub_uis)

    node_df.to_csv(args.node_csv, index=False)

    expected_columns = [
        "descriptor_ui",
        "descriptor_name",
        "tree_roots",
        "primary_root",
        "num_tree_roots",
        "depth3_ancestors",
        "text_for_embedding",
        "max_depth",
        "min_depth",
        "mean_depth",
        "degree",
        "is_hub",
    ]
    embedding_metadata_rows = None
    embedding_metadata_matches = None
    if args.embedding_metadata_csv.exists():
        embedding_metadata_rows = int(len(pd.read_csv(args.embedding_metadata_csv)))
        embedding_metadata_matches = embedding_metadata_rows == len(node_df)

    stats = {
        "num_nodes": int(len(node_df)),
        "hub_percent": args.hub_percent,
        "num_hubs": int(node_df["is_hub"].sum()),
        "hub_degree_min": int(node_df.loc[node_df["is_hub"], "degree"].min()),
        "hub_degree_max": int(node_df.loc[node_df["is_hub"], "degree"].max()),
        "max_depth_max": int(node_df["max_depth"].max()),
        "min_depth_min": int(node_df["min_depth"].min()),
        "mean_degree": float(node_df["degree"].mean()),
        "degree_graph_components": [str(path) for path in args.edge_csv_gz],
        "embedding_metadata_rows": embedding_metadata_rows,
        "embedding_metadata_matches_node_rows": embedding_metadata_matches,
        "validation_columns_present": {column: (column in node_df.columns) for column in expected_columns},
    }
    args.stats_json.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
