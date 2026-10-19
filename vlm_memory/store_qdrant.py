#!/usr/bin/env python
"""Store image embeddings (from embed_image.py output) into Qdrant.

By default, each image's per-token embeddings are mean-pooled into a single
vector and stored as one point per image. Use --token-level to instead store
one point per patch token (useful for finer-grained / spatial search).

Usage:
    /ml/HelaBERT_Analysis/.venv/bin/python store_qdrant.py \
        --embeddings embeddings_1.json embeddings_2.json \
        --collection image_embeddings

    /ml/HelaBERT_Analysis/.venv/bin/python store_qdrant.py \
        --embeddings embeddings_1.json --token-level --collection image_tokens
"""

import argparse
import json
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


def load_embeddings(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def mean_pool(embeddings: list[list[float]]) -> list[float]:
    n = len(embeddings)
    dim = len(embeddings[0])
    pooled = [0.0] * dim
    for vec in embeddings:
        for i, v in enumerate(vec):
            pooled[i] += v
    return [v / n for v in pooled]


def main():
    parser = argparse.ArgumentParser(description="Store image embeddings into Qdrant")
    parser.add_argument("--embeddings", nargs="+", required=True, help="One or more embed_image.py output JSON files")
    parser.add_argument("--collection", default="image_embeddings", help="Qdrant collection name")
    parser.add_argument("--host", default="localhost", help="Qdrant host")
    parser.add_argument("--port", type=int, default=6333, help="Qdrant HTTP port")
    parser.add_argument("--token-level", action="store_true", help="Store one point per patch token instead of one mean-pooled point per image")
    parser.add_argument("--recreate", action="store_true", help="Drop and recreate the collection before inserting")
    args = parser.parse_args()

    client = QdrantClient(host=args.host, port=args.port)

    vector_size = None
    points = []

    for path in args.embeddings:
        data = load_embeddings(path)
        image_name = data["image"]
        embedding_size = data["embedding_size"]
        vector_size = vector_size or embedding_size

        if args.token_level:
            for token_idx, vec in enumerate(data["embeddings"]):
                points.append(
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vec,
                        payload={"image": image_name, "token_index": token_idx, "source_file": path},
                    )
                )
        else:
            pooled = mean_pool(data["embeddings"])
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=pooled,
                    payload={"image": image_name, "num_tokens": data["num_tokens"], "source_file": path},
                )
            )

    if args.recreate or not client.collection_exists(args.collection):
        client.recreate_collection(
            collection_name=args.collection,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

    client.upsert(collection_name=args.collection, points=points)

    print(f"Stored {len(points)} point(s) of dimension {vector_size} in collection '{args.collection}'")


if __name__ == "__main__":
    main()
