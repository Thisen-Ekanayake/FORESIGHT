#!/usr/bin/env python
"""Compute cosine similarity between two images' embeddings (from embed_image.py output).

By default, each image's per-token embeddings are mean-pooled into a single
vector and the cosine similarity between the two pooled vectors is reported.

Use --token-level to additionally compute the full token-by-token pairwise
cosine similarity matrix and dump it to a JSON file.

Usage:
    /ml/HelaBERT_Analysis/.venv/bin/python cosine_similarity.py \
        --embeddings1 embeddings_1.json --embeddings2 embeddings_2.json

    /ml/HelaBERT_Analysis/.venv/bin/python cosine_similarity.py \
        --embeddings1 embeddings_1.json --embeddings2 embeddings_2.json \
        --token-level --output pairwise_similarity.json
"""

import argparse
import json

import torch


def load_embeddings(path: str) -> torch.Tensor:
    with open(path) as f:
        data = json.load(f)
    return torch.tensor(data["embeddings"], dtype=torch.float32)


def main():
    parser = argparse.ArgumentParser(description="Cosine similarity between two image embedding files")
    parser.add_argument("--embeddings1", required=True, help="Path to first embeddings JSON")
    parser.add_argument("--embeddings2", required=True, help="Path to second embeddings JSON")
    parser.add_argument("--token-level", action="store_true", help="Also compute full token-by-token similarity matrix")
    parser.add_argument("--output", default="pairwise_similarity.json", help="Where to save the token-level matrix")
    args = parser.parse_args()

    emb1 = load_embeddings(args.embeddings1)
    emb2 = load_embeddings(args.embeddings2)

    pooled1 = emb1.mean(dim=0)
    pooled2 = emb2.mean(dim=0)
    pooled_sim = torch.nn.functional.cosine_similarity(pooled1, pooled2, dim=0).item()

    print(f"Mean-pooled cosine similarity: {pooled_sim:.6f}")

    if args.token_level:
        emb1_norm = torch.nn.functional.normalize(emb1, dim=1)
        emb2_norm = torch.nn.functional.normalize(emb2, dim=1)
        sim_matrix = emb1_norm @ emb2_norm.T  # (num_tokens_1, num_tokens_2)

        best1to2 = sim_matrix.max(dim=1)
        best2to1 = sim_matrix.max(dim=0)

        print(f"Token-level similarity matrix shape: {tuple(sim_matrix.shape)}")
        print(f"Average of best-match similarity (1->2): {best1to2.values.mean().item():.6f}")
        print(f"Average of best-match similarity (2->1): {best2to1.values.mean().item():.6f}")

        result = {
            "pooled_cosine_similarity": pooled_sim,
            "similarity_matrix": sim_matrix.tolist(),
            "best_match_1_to_2": best1to2.values.tolist(),
            "best_match_2_to_1": best2to1.values.tolist(),
        }
        with open(args.output, "w") as f:
            json.dump(result, f)
        print(f"Saved token-level similarity matrix to {args.output}")


if __name__ == "__main__":
    main()
