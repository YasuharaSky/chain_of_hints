#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import pandas as pd


DEFAULT_PAIR_SPECS = [
    {"category": "positive_control", "label": "swanson_fish_oils_raynaud", "source_name": "Fish Oils", "target_name": "Raynaud Disease"},
    {"category": "positive_control", "label": "swanson_magnesium_migraine", "source_name": "Magnesium", "target_name": "Migraine Disorders"},
    {"category": "same_class_baseline", "label": "hypertension_diabetes", "source_name": "Hypertension", "target_name": "Diabetes Mellitus"},
    {"category": "same_class_baseline", "label": "aspirin_ibuprofen", "source_name": "Aspirin", "target_name": "Ibuprofen"},
    {"category": "same_class_baseline", "label": "asthma_obesity", "source_name": "Asthma", "target_name": "Obesity"},
    {"category": "cross_class_remote", "label": "diet_neoplasms", "source_name": "Diet", "target_name": "Neoplasms"},
    {"category": "cross_class_remote", "label": "aspirin_alzheimer", "source_name": "Aspirin", "target_name": "Alzheimer Disease"},
    {"category": "cross_class_remote", "label": "sleep_inflammation", "source_name": "Sleep", "target_name": "Inflammation"},
    {"category": "cross_class_hot_topic", "label": "microbiota_depressive_disorder", "source_name": "Microbiota", "target_name": "Depressive Disorder"},
    {"category": "cross_class_hot_topic", "label": "exercise_depressive_disorder", "source_name": "Exercise", "target_name": "Depressive Disorder"},
    {"category": "cross_class_hot_topic", "label": "exercise_neoplasms", "source_name": "Exercise", "target_name": "Neoplasms"},
    {"category": "cross_class_hot_topic", "label": "curcumin_alzheimer", "source_name": "Curcumin", "target_name": "Alzheimer Disease"},
    {"category": "cross_class_hot_topic", "label": "meditation_inflammation", "source_name": "Meditation", "target_name": "Inflammation"},
    {"category": "cross_class_hot_topic", "label": "air_pollution_cognitive_dysfunction", "source_name": "Air Pollution", "target_name": "Cognitive Dysfunction"},
    {"category": "cross_class_hot_topic", "label": "probiotics_anxiety_disorders", "source_name": "Probiotics", "target_name": "Anxiety Disorders"},
]


def build_adjacency(edge_paths: Sequence[Path], blocked_nodes: Set[str]) -> Dict[str, Set[str]]:
    adjacency: Dict[str, Set[str]] = {}
    for edge_path in edge_paths:
        with gzip.open(edge_path, "rt", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                source_ui = row["source_ui"]
                target_ui = row["target_ui"]
                if source_ui in blocked_nodes or target_ui in blocked_nodes:
                    continue
                adjacency.setdefault(source_ui, set()).add(target_ui)
                adjacency.setdefault(target_ui, set()).add(source_ui)
    return adjacency


def shortest_path_with_limit(adjacency: Dict[str, Set[str]], source_ui: str, target_ui: str, max_edges: int) -> List[str] | None:
    if source_ui == target_ui:
        return [source_ui]
    queue = deque([(source_ui, [source_ui])])
    visited = {source_ui}
    while queue:
        node_ui, path = queue.popleft()
        if len(path) - 1 >= max_edges:
            continue
        for neighbor_ui in adjacency.get(node_ui, set()):
            if neighbor_ui in visited:
                continue
            next_path = path + [neighbor_ui]
            if neighbor_ui == target_ui:
                return next_path
            visited.add(neighbor_ui)
            queue.append((neighbor_ui, next_path))
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve and validate MeSH concept pairs for path search.")
    parser.add_argument(
        "--node-csv",
        type=Path,
        default=Path("data/processed/mesh/mesh_graph_nodes_2026.csv"),
        help="Annotated MeSH node CSV.",
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
        help="Graph edge files used for reachability validation.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/processed/mesh/mesh_selected_concept_pairs_2026.csv"),
        help="Where to write resolved pair metadata.",
    )
    parser.add_argument(
        "--config-json",
        type=Path,
        default=Path("data/processed/mesh/mesh_experiment_config_v1.json"),
        help="Where to write the experiment config.",
    )
    parser.add_argument(
        "--ready-output-csv",
        type=Path,
        default=Path("data/processed/mesh/mesh_ready_pairs_k4_2026.csv"),
        help="Where to write the subset of pairs that are ready on the current graph.",
    )
    parser.add_argument(
        "--max-edges",
        type=int,
        default=5,
        help="Maximum graph distance for reachability validation.",
    )
    parser.add_argument(
        "--k-intermediates",
        type=int,
        default=4,
        help="Number of intermediate nodes for the next path-search phase.",
    )
    args = parser.parse_args()

    node_df = pd.read_csv(args.node_csv).fillna("")
    blocked_nodes = set(node_df.loc[node_df["is_hub"] == True, "descriptor_ui"])
    name_to_rows = {
        descriptor_name: group.to_dict(orient="records")
        for descriptor_name, group in node_df.groupby("descriptor_name", sort=False)
    }
    ui_to_name = dict(zip(node_df["descriptor_ui"], node_df["descriptor_name"]))
    adjacency = build_adjacency(args.edge_csv_gz, blocked_nodes)

    resolved_rows = []
    for spec in DEFAULT_PAIR_SPECS:
        source_rows = name_to_rows.get(spec["source_name"], [])
        target_rows = name_to_rows.get(spec["target_name"], [])
        if len(source_rows) != 1 or len(target_rows) != 1:
            raise ValueError(f"Could not resolve unique descriptors for pair spec: {spec}")

        source_row = source_rows[0]
        target_row = target_rows[0]
        source_ui = source_row["descriptor_ui"]
        target_ui = target_row["descriptor_ui"]
        shortest_path = shortest_path_with_limit(adjacency, source_ui, target_ui, args.max_edges)

        resolved_rows.append(
            {
                "category": spec["category"],
                "label": spec["label"],
                "source_ui": source_ui,
                "source_name": source_row["descriptor_name"],
                "source_roots": source_row["tree_roots"],
                "target_ui": target_ui,
                "target_name": target_row["descriptor_name"],
                "target_roots": target_row["tree_roots"],
                "reachable_within_max_edges": shortest_path is not None,
                "ready_for_k4": shortest_path is not None,
                "status": (
                    "ready_on_current_graph"
                    if shortest_path is not None
                    else "requires_graph_upgrade_or_longer_k"
                ),
                "shortest_path_edges": (len(shortest_path) - 1) if shortest_path else -1,
                "shortest_path_ui": "|".join(shortest_path) if shortest_path else "",
                "shortest_path_name": " | ".join(ui_to_name[node_ui] for node_ui in shortest_path) if shortest_path else "",
            }
        )

    resolved_df = pd.DataFrame(resolved_rows)
    resolved_df.to_csv(args.output_csv, index=False)
    resolved_df.loc[resolved_df["ready_for_k4"] == True].to_csv(args.ready_output_csv, index=False)

    config = {
        "k_intermediates": args.k_intermediates,
        "max_edges_for_reachability_check": args.max_edges,
        "graph": "tree_plus_shared_depth3_plus_multiroot_bridge",
        "graph_components": [str(path) for path in args.edge_csv_gz],
        "exclude_hubs": True,
        "hub_percent": 0.01,
        "use_pubmed_cooccurrence_now": False,
        "pubmed_cooccurrence_decision": "still_recommend_next_for_semantics_not_connectivity",
        "pubmed_cooccurrence_reason": "The upgraded ontology graph now covers target k=4 pairs structurally, but PubMed PMI can still provide stronger semantic edges than hierarchy-derived links alone.",
        "recommended_next_graph_upgrade": "pubmed_pmi_graph",
        "pair_file": str(args.output_csv),
        "ready_pair_file": str(args.ready_output_csv),
    }
    args.config_json.write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.ready_output_csv}")
    print(f"Wrote {args.config_json}")
    print(resolved_df.to_string(index=False))


if __name__ == "__main__":
    main()
