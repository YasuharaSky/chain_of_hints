#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import pandas as pd


def parse_branch_caps(raw_value: str) -> List[int]:
    values = []
    for item in raw_value.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(int(item))
    if not values:
        raise ValueError("branch caps must contain at least one integer")
    return values


def load_node_metadata(node_csv: Path) -> tuple[pd.DataFrame, Dict[str, int], List[str], List[str], set[int]]:
    node_df = pd.read_csv(node_csv).fillna("")
    ui_to_idx = {descriptor_ui: idx for idx, descriptor_ui in enumerate(node_df["descriptor_ui"])}
    idx_to_ui = node_df["descriptor_ui"].tolist()
    idx_to_name = node_df["descriptor_name"].tolist()
    hub_indices = set(node_df.index[node_df["is_hub"] == True].tolist())
    return node_df, ui_to_idx, idx_to_ui, idx_to_name, hub_indices


def build_adjacency(edge_csv_gz_list: Sequence[Path], ui_to_idx: Dict[str, int], blocked_indices: set[int]) -> List[List[int]]:
    adjacency_sets = [set() for _ in range(len(ui_to_idx))]
    for edge_csv_gz in edge_csv_gz_list:
        with gzip.open(edge_csv_gz, "rt", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                source_idx = ui_to_idx.get(row["source_ui"])
                target_idx = ui_to_idx.get(row["target_ui"])
                if source_idx is None or target_idx is None:
                    continue
                if source_idx in blocked_indices or target_idx in blocked_indices or source_idx == target_idx:
                    continue
                adjacency_sets[source_idx].add(target_idx)
                adjacency_sets[target_idx].add(source_idx)

    return [sorted(neighbors) for neighbors in adjacency_sets]


def bfs_distances(adjacency: Sequence[Sequence[int]], target_idx: int, max_distance: int) -> Dict[int, int]:
    distances = {target_idx: 0}
    queue = deque([target_idx])
    while queue:
        node_idx = queue.popleft()
        distance = distances[node_idx]
        if distance >= max_distance:
            continue
        for neighbor_idx in adjacency[node_idx]:
            if neighbor_idx in distances:
                continue
            distances[neighbor_idx] = distance + 1
            queue.append(neighbor_idx)
    return distances


def candidate_order(
    node_idx: int,
    adjacency: Sequence[Sequence[int]],
    visited: set[int],
    target_idx: int,
    remaining_edges: int,
    distances_to_target: Dict[int, int],
    branch_cap: int,
    rng: random.Random,
) -> List[int]:
    buckets: Dict[int, List[int]] = defaultdict(list)
    for neighbor_idx in adjacency[node_idx]:
        if neighbor_idx in visited:
            continue
        if neighbor_idx == target_idx:
            if remaining_edges != 1:
                continue
            buckets[0].append(neighbor_idx)
            continue

        distance = distances_to_target.get(neighbor_idx)
        if distance is None or distance > remaining_edges - 1:
            continue
        buckets[distance].append(neighbor_idx)

    ordered: List[int] = []
    for distance in sorted(buckets):
        bucket = buckets[distance]
        rng.shuffle(bucket)
        ordered.extend(bucket)
        if branch_cap > 0 and len(ordered) >= branch_cap:
            break
    if branch_cap > 0:
        return ordered[:branch_cap]
    return ordered


def enumerate_exact_length_paths(
    adjacency: Sequence[Sequence[int]],
    source_idx: int,
    target_idx: int,
    path_edges: int,
    max_paths: int,
    branch_caps: Sequence[int],
    max_expansions: int,
    seed: int,
) -> tuple[List[List[int]], Dict[str, int]]:
    distances_to_target = bfs_distances(adjacency, target_idx, path_edges)
    if source_idx not in distances_to_target:
        return [], {"graph_distance_within_limit": -1, "expansions": 0, "rounds": 0}

    results: List[List[int]] = []
    seen_paths: set[tuple[int, ...]] = set()
    expansions = 0

    def dfs(
        node_idx: int,
        path: List[int],
        visited: set[int],
        branch_cap: int,
        rng: random.Random,
    ) -> None:
        nonlocal expansions
        if len(results) >= max_paths or expansions >= max_expansions:
            return

        used_edges = len(path) - 1
        remaining_edges = path_edges - used_edges
        if remaining_edges == 0:
            if node_idx == target_idx:
                path_key = tuple(path)
                if path_key not in seen_paths:
                    seen_paths.add(path_key)
                    results.append(path.copy())
            return

        for neighbor_idx in candidate_order(
            node_idx=node_idx,
            adjacency=adjacency,
            visited=visited,
            target_idx=target_idx,
            remaining_edges=remaining_edges,
            distances_to_target=distances_to_target,
            branch_cap=branch_cap,
            rng=rng,
        ):
            expansions += 1
            if expansions > max_expansions:
                return
            visited.add(neighbor_idx)
            path.append(neighbor_idx)
            dfs(neighbor_idx, path, visited, branch_cap, rng)
            path.pop()
            visited.remove(neighbor_idx)
            if len(results) >= max_paths or expansions >= max_expansions:
                return

    for round_idx, branch_cap in enumerate(branch_caps):
        if len(results) >= max_paths or expansions >= max_expansions:
            break
        rng = random.Random(seed + round_idx)
        dfs(source_idx, [source_idx], {source_idx}, branch_cap, rng)

    stats = {
        "graph_distance_within_limit": int(distances_to_target[source_idx]),
        "expansions": expansions,
        "rounds": len(branch_caps),
    }
    return results, stats


def path_names(path_indices: Iterable[int], idx_to_name: Sequence[str]) -> List[str]:
    return [idx_to_name[node_idx] for node_idx in path_indices]


def main() -> None:
    parser = argparse.ArgumentParser(description="Enumerate fixed-length MeSH concept paths for manual review and scoring.")
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
        help="Graph edge files used for path enumeration.",
    )
    parser.add_argument(
        "--pair-csv",
        type=Path,
        default=Path("data/processed/mesh/mesh_ready_pairs_k4_2026.csv"),
        help="Pair CSV to enumerate over.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/processed/mesh/candidate_paths.csv"),
        help="Where to write candidate paths.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("data/processed/mesh/candidate_paths_summary.json"),
        help="Where to write enumeration summary stats.",
    )
    parser.add_argument(
        "--k-intermediates",
        type=int,
        default=4,
        help="Number of intermediate nodes in each path.",
    )
    parser.add_argument(
        "--paths-per-pair",
        type=int,
        default=100,
        help="Maximum number of candidate paths to keep per pair.",
    )
    parser.add_argument(
        "--branch-caps",
        type=str,
        default="32,64,128,0",
        help="Comma-separated branch caps used across DFS rounds. Use 0 for an unbounded final pass.",
    )
    parser.add_argument(
        "--max-expansions-per-pair",
        type=int,
        default=250000,
        help="Safety cap on DFS expansions per pair.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=20260313,
        help="Base random seed for path enumeration.",
    )
    args = parser.parse_args()

    branch_caps = parse_branch_caps(args.branch_caps)
    path_edges = args.k_intermediates + 1

    node_df, ui_to_idx, idx_to_ui, idx_to_name, hub_indices = load_node_metadata(args.node_csv)
    adjacency = build_adjacency(args.edge_csv_gz, ui_to_idx, hub_indices)

    pair_df = pd.read_csv(args.pair_csv).fillna("")
    if "ready_for_k4" in pair_df.columns:
        pair_df = pair_df.loc[pair_df["ready_for_k4"] == True].copy()

    records = []
    pair_summaries = []
    for pair_rank, pair_row in enumerate(pair_df.to_dict(orient="records"), start=1):
        source_ui = pair_row["source_ui"]
        target_ui = pair_row["target_ui"]
        source_idx = ui_to_idx[source_ui]
        target_idx = ui_to_idx[target_ui]
        if source_idx in hub_indices or target_idx in hub_indices:
            raise ValueError(f"pair endpoint unexpectedly marked as hub: {source_ui} -> {target_ui}")

        paths, stats = enumerate_exact_length_paths(
            adjacency=adjacency,
            source_idx=source_idx,
            target_idx=target_idx,
            path_edges=path_edges,
            max_paths=args.paths_per_pair,
            branch_caps=branch_caps,
            max_expansions=args.max_expansions_per_pair,
            seed=args.random_seed + pair_rank * 1000,
        )

        for local_rank, path_indices in enumerate(paths, start=1):
            node_uis = [idx_to_ui[node_idx] for node_idx in path_indices]
            node_names = path_names(path_indices, idx_to_name)
            intermediate_uis = node_uis[1:-1]
            intermediate_names = node_names[1:-1]
            records.append(
                {
                    "category": pair_row["category"],
                    "label": pair_row["label"],
                    "source_ui": source_ui,
                    "source_name": pair_row["source_name"],
                    "target_ui": target_ui,
                    "target_name": pair_row["target_name"],
                    "path_rank_within_pair": local_rank,
                    "path_edge_count": path_edges,
                    "path_node_count": len(path_indices),
                    "path_ui": "|".join(node_uis),
                    "path_name": " | ".join(node_names),
                    "intermediate_uis": "|".join(intermediate_uis),
                    "intermediate_names": ", ".join(intermediate_names),
                    "m1_ui": intermediate_uis[0],
                    "m1_name": intermediate_names[0],
                    "m2_ui": intermediate_uis[1],
                    "m2_name": intermediate_names[1],
                    "m3_ui": intermediate_uis[2],
                    "m3_name": intermediate_names[2],
                    "m4_ui": intermediate_uis[3],
                    "m4_name": intermediate_names[3],
                }
            )

        pair_summaries.append(
            {
                "category": pair_row["category"],
                "label": pair_row["label"],
                "source_ui": source_ui,
                "source_name": pair_row["source_name"],
                "target_ui": target_ui,
                "target_name": pair_row["target_name"],
                "graph_distance_within_limit": stats["graph_distance_within_limit"],
                "num_candidate_paths": len(paths),
                "requested_paths": args.paths_per_pair,
                "complete": len(paths) >= args.paths_per_pair,
                "expansions": stats["expansions"],
                "rounds": stats["rounds"],
            }
        )

    candidate_df = pd.DataFrame(records)
    candidate_df.to_csv(args.output_csv, index=False)

    summary = {
        "pair_file": str(args.pair_csv),
        "graph": "tree_plus_shared_depth3_plus_multiroot_bridge",
        "graph_components": [str(path) for path in args.edge_csv_gz],
        "exclude_hubs": True,
        "k_intermediates": args.k_intermediates,
        "path_edges": path_edges,
        "paths_per_pair": args.paths_per_pair,
        "branch_caps": branch_caps,
        "max_expansions_per_pair": args.max_expansions_per_pair,
        "random_seed": args.random_seed,
        "num_pairs_requested": int(len(pair_df)),
        "num_pairs_with_candidates": int(sum(1 for item in pair_summaries if item["num_candidate_paths"] > 0)),
        "num_candidate_paths_total": int(len(candidate_df)),
        "pairs": pair_summaries,
    }
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote {args.output_csv} with {len(candidate_df)} paths")
    print(f"Wrote {args.summary_json}")
    print(pd.DataFrame(pair_summaries).to_string(index=False))


if __name__ == "__main__":
    main()
