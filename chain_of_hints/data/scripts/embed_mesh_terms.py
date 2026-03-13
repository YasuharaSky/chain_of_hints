#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked = last_hidden_state * mask
    denom = mask.sum(dim=1).clamp(min=1e-9)
    return masked.sum(dim=1) / denom


def model_slug(model_name: str) -> str:
    return model_name.split("/")[-1].replace("-", "_")


def default_output_prefix(model_name: str, text_column: str, pooling: str) -> str:
    slug = model_slug(model_name)
    if text_column == "text_for_embedding" and pooling == "mean":
        return f"mesh_embeddings_{slug}"
    text_slug = text_column.replace("-", "_")
    return f"mesh_embeddings_{slug}_{text_slug}_{pooling}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed MeSH descriptor texts with a biomedical encoder.")
    parser.add_argument(
        "--node-csv",
        type=Path,
        default=Path("data/processed/mesh/mesh_graph_nodes_2026.csv"),
        help="Node table with text_for_embedding.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/mesh"),
        help="Output directory for embedding artifacts.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        help="Hugging Face model identifier.",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="text_for_embedding",
        help="Node CSV column to encode, for example text_for_embedding or descriptor_name.",
    )
    parser.add_argument(
        "--pooling",
        type=str,
        choices=["mean", "cls"],
        default="mean",
        help="Pooling strategy over token representations.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="",
        help="Optional custom output prefix without extension.",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Encoding batch size.")
    parser.add_argument("--max-length", type=int, default=256, help="Tokenizer max length.")
    args = parser.parse_args()

    import torch
    from transformers import AutoModel, AutoTokenizer

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    node_df = pd.read_csv(args.node_csv).fillna("")
    if args.text_column not in node_df.columns:
        raise ValueError(f"text column {args.text_column!r} not found in {args.node_csv}")
    texts = node_df[args.text_column].astype(str).tolist()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name)
    model.to(device)
    model.eval()

    all_embeddings = []
    for start in range(0, len(texts), args.batch_size):
        batch_texts = texts[start : start + args.batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}

        with torch.inference_mode():
            outputs = model(**encoded)
            if args.pooling == "cls":
                pooled = outputs.last_hidden_state[:, 0, :]
            else:
                pooled = mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        all_embeddings.append(pooled.cpu().numpy().astype(np.float32))

        if (start // args.batch_size + 1) % 50 == 0:
            print(f"Encoded {min(start + args.batch_size, len(texts))}/{len(texts)} terms")

    embeddings = np.vstack(all_embeddings)

    output_prefix = args.output_prefix.strip() or default_output_prefix(args.model_name, args.text_column, args.pooling)
    embedding_path = args.output_dir / f"{output_prefix}.npy"
    metadata_path = args.output_dir / f"{output_prefix}_metadata.csv"
    config_path = args.output_dir / f"{output_prefix}_config.json"

    np.save(embedding_path, embeddings)
    metadata_df = node_df[
        [
            "descriptor_ui",
            "descriptor_name",
            "tree_roots",
            "primary_root",
            "num_tree_roots",
            "depth3_ancestors",
        ]
    ].copy()
    metadata_df["embedding_text"] = texts
    metadata_df["embedding_text_column"] = args.text_column
    metadata_df.to_csv(metadata_path, index=False)
    config_path.write_text(
        json.dumps(
            {
                "model_name": args.model_name,
                "text_column": args.text_column,
                "pooling": args.pooling,
                "output_prefix": output_prefix,
                "batch_size": args.batch_size,
                "max_length": args.max_length,
                "device": str(device),
                "num_terms": int(len(node_df)),
                "embedding_dim": int(embeddings.shape[1]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Wrote {embedding_path}")
    print(f"Wrote {metadata_path}")
    print(f"Wrote {config_path}")


if __name__ == "__main__":
    main()
