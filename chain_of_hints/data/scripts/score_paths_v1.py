#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def load_normalized_embeddings(embedding_npy: Path) -> np.ndarray:
    embeddings = np.load(embedding_npy)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return embeddings / norms


def safe_cosine_with_vector(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(vector_b))
    if denominator <= 1e-12:
        return 0.0
    return float(np.dot(vector_a, vector_b / denominator))


def main() -> None:
    parser = argparse.ArgumentParser(description="Score fixed-length MeSH candidate paths with V1 creativity metrics.")
    parser.add_argument(
        "--candidate-csv",
        type=Path,
        default=Path("data/processed/mesh/candidate_paths.csv"),
        help="Candidate path CSV.",
    )
    parser.add_argument(
        "--node-csv",
        type=Path,
        default=Path("data/processed/mesh/mesh_graph_nodes_2026.csv"),
        help="Annotated MeSH node CSV.",
    )
    parser.add_argument(
        "--embedding-npy",
        type=Path,
        default=Path("data/processed/mesh/mesh_embeddings_BiomedNLP_PubMedBERT_base_uncased_abstract_fulltext.npy"),
        help="Embedding matrix .npy file.",
    )
    parser.add_argument(
        "--embedding-metadata-csv",
        type=Path,
        default=Path("data/processed/mesh/mesh_embeddings_BiomedNLP_PubMedBERT_base_uncased_abstract_fulltext_metadata.csv"),
        help="Embedding metadata CSV, used to map descriptor_ui to row index.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/processed/mesh/scored_paths.csv"),
        help="Where to write scored paths.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("data/processed/mesh/scored_paths_summary.json"),
        help="Where to write scoring summary stats.",
    )
    parser.add_argument(
        "--efficiency-threshold",
        type=float,
        default=0.1,
        help="Minimum Efficiency threshold.",
    )
    parser.add_argument(
        "--cv-threshold",
        type=float,
        default=2.0,
        help="Maximum CV threshold.",
    )
    parser.add_argument(
        "--min-cosine-threshold",
        type=float,
        default=0.05,
        help="Minimum adjacent-step cosine threshold.",
    )
    parser.add_argument(
        "--lambda-unique-categories",
        type=float,
        default=1.0,
        help="Weight for UniqueCategories in Creativity_V1.",
    )
    parser.add_argument(
        "--mu-depth-range",
        type=float,
        default=1.0,
        help="Weight for DepthRange in Creativity_V1.",
    )
    args = parser.parse_args()

    candidate_df = pd.read_csv(args.candidate_csv).fillna("")
    node_df = pd.read_csv(args.node_csv).fillna("")
    metadata_df = pd.read_csv(args.embedding_metadata_csv).fillna("")
    normalized_embeddings = load_normalized_embeddings(args.embedding_npy)

    if len(metadata_df) != len(normalized_embeddings):
        raise ValueError("embedding metadata row count does not match embedding matrix row count")

    ui_to_embedding_idx = {
        descriptor_ui: idx for idx, descriptor_ui in enumerate(metadata_df["descriptor_ui"].tolist())
    }
    node_lookup = node_df.set_index("descriptor_ui")[["descriptor_name", "primary_root", "max_depth"]].to_dict(orient="index")

    scored_records: List[Dict[str, object]] = []
    for row in candidate_df.to_dict(orient="records"):
        path_uis = str(row["path_ui"]).split("|")
        embedding_indices = [ui_to_embedding_idx[descriptor_ui] for descriptor_ui in path_uis]
        path_vectors = normalized_embeddings[embedding_indices]

        step_cosines = []
        step_distances = []
        for idx in range(len(path_vectors) - 1):
            cosine = float(np.dot(path_vectors[idx], path_vectors[idx + 1]))
            distance = 1.0 - cosine
            step_cosines.append(cosine)
            step_distances.append(distance)

        source_target_cosine = float(np.dot(path_vectors[0], path_vectors[-1]))
        source_target_distance = 1.0 - source_target_cosine
        total_path_length = float(sum(step_distances))
        efficiency = source_target_distance / total_path_length if total_path_length > 0 else 0.0

        mean_step_distance = float(np.mean(step_distances)) if step_distances else 0.0
        cv = float(np.std(step_distances) / mean_step_distance) if mean_step_distance > 0 else 0.0

        local_novelties = []
        for middle_idx in range(1, len(path_vectors) - 1):
            midpoint = (path_vectors[middle_idx - 1] + path_vectors[middle_idx + 1]) / 2.0
            midpoint_cosine = safe_cosine_with_vector(path_vectors[middle_idx], midpoint)
            local_novelties.append(1.0 - midpoint_cosine)
        local_novelty_sum = float(sum(local_novelties))

        primary_roots = []
        for descriptor_ui in path_uis:
            primary_root = str(node_lookup[descriptor_ui]["primary_root"]).strip()
            primary_roots.append(primary_root or "UNKNOWN")
        unique_categories = int(len(set(primary_roots)))

        middle_depths = [int(node_lookup[descriptor_ui]["max_depth"]) for descriptor_ui in path_uis[1:-1]]
        depth_range = int(max(middle_depths) - min(middle_depths)) if middle_depths else 0

        min_step_cosine = float(min(step_cosines)) if step_cosines else 0.0
        passes_efficiency = efficiency > args.efficiency_threshold
        passes_cv = cv < args.cv_threshold
        passes_min_cosine = min_step_cosine > args.min_cosine_threshold
        passes_constraints = passes_efficiency and passes_cv and passes_min_cosine

        failed_constraints = []
        if not passes_efficiency:
            failed_constraints.append("efficiency")
        if not passes_cv:
            failed_constraints.append("cv")
        if not passes_min_cosine:
            failed_constraints.append("min_cosine")

        creativity_v1_raw = (
            local_novelty_sum
            + args.lambda_unique_categories * unique_categories
            + args.mu_depth_range * depth_range
        )
        creativity_v1 = creativity_v1_raw if passes_constraints else np.nan

        scored_record = dict(row)
        scored_record.update(
            {
                "distance_metric": "cosine_distance",
                "source_target_cosine": source_target_cosine,
                "source_target_distance": source_target_distance,
                "total_path_length": total_path_length,
                "efficiency": efficiency,
                "cv": cv,
                "min_step_cosine": min_step_cosine,
                "unique_categories": unique_categories,
                "depth_range": depth_range,
                "root_sequence": "|".join(primary_roots),
                "creativity_v1_raw": creativity_v1_raw,
                "creativity_v1": creativity_v1,
                "passes_efficiency": passes_efficiency,
                "passes_cv": passes_cv,
                "passes_min_cosine": passes_min_cosine,
                "passes_constraints": passes_constraints,
                "failed_constraints": "|".join(failed_constraints),
            }
        )

        for step_idx, (cosine, distance) in enumerate(zip(step_cosines, step_distances), start=1):
            scored_record[f"step{step_idx}_cosine"] = cosine
            scored_record[f"step{step_idx}_distance"] = distance

        for novelty_idx, local_novelty in enumerate(local_novelties, start=1):
            scored_record[f"local_novelty_m{novelty_idx}"] = local_novelty

        scored_records.append(scored_record)

    scored_df = pd.DataFrame(scored_records)
    scored_df = scored_df.sort_values(
        ["label", "passes_constraints", "creativity_v1_raw", "efficiency"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)
    scored_df.to_csv(args.output_csv, index=False)

    summary = {
        "candidate_csv": str(args.candidate_csv),
        "num_candidate_paths": int(len(scored_df)),
        "num_pairs": int(scored_df["label"].nunique()) if len(scored_df) else 0,
        "num_paths_passing_constraints": int(scored_df["passes_constraints"].sum()) if len(scored_df) else 0,
        "constraint_thresholds": {
            "efficiency_gt": args.efficiency_threshold,
            "cv_lt": args.cv_threshold,
            "min_step_cosine_gt": args.min_cosine_threshold,
        },
        "creativity_weights": {
            "lambda_unique_categories": args.lambda_unique_categories,
            "mu_depth_range": args.mu_depth_range,
        },
        "distance_metric": "cosine_distance = 1 - cosine_similarity",
        "metric_definitions": {
            "efficiency": "source_target_distance / total_path_length",
            "cv": "std(step_distances) / mean(step_distances)",
            "local_novelty_sum": "sum_i 1 - cosine(M_i, midpoint(prev_i, next_i))",
            "unique_categories": "number of unique primary_root labels across S1..S2",
            "depth_range": "max(max_depth of M1..M4) - min(max_depth of M1..M4)",
        },
        "per_pair_counts": (
            scored_df.groupby("label")["passes_constraints"]
            .agg(total_paths="count", passing_paths="sum")
            .reset_index()
            .to_dict(orient="records")
            if len(scored_df)
            else []
        ),
    }
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote {args.output_csv} with {len(scored_df)} scored paths")
    print(f"Wrote {args.summary_json}")
    if len(scored_df):
        print(
            scored_df.groupby("label")["passes_constraints"]
            .agg(total_paths="count", passing_paths="sum")
            .reset_index()
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
