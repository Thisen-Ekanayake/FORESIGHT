#!/usr/bin/env python
"""Embed image(s) with Qwen3-VL and store them in Qdrant, one point per image.

Usage:
    /ml/HelaBERT_Analysis/.venv/bin/python ingest_image.py --images img1.jpg img2.jpg --collection image_embeddings
"""

import argparse
import os
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from qwen_vl import get_image_embeds, load_model, mean_pool


def main():
    parser = argparse.ArgumentParser(description="Embed images with Qwen3-VL and store in Qdrant")
    parser.add_argument("--images", nargs="+", required=True, help="One or more image paths/URLs")
    parser.add_argument("--collection", default="image_embeddings", help="Qdrant collection name")
    parser.add_argument("--host", default="localhost", help="Qdrant host")
    parser.add_argument("--port", type=int, default=6333, help="Qdrant HTTP port")
    args = parser.parse_args()

    model, processor = load_model()

    client = QdrantClient(host=args.host, port=args.port)

    points = []
    vector_size = None
    for image_path in args.images:
        abs_path = image_path if image_path.startswith("http") else os.path.abspath(image_path)
        embeds = get_image_embeds(model, processor, abs_path)
        vector = mean_pool(embeds)
        vector_size = vector_size or len(vector)
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={"image": abs_path, "num_tokens": embeds.shape[0]},
            )
        )
        print(f"Embedded {abs_path}: {embeds.shape[0]} tokens")

    if not client.collection_exists(args.collection):
        client.create_collection(
            collection_name=args.collection,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

    client.upsert(collection_name=args.collection, points=points)
    print(f"Stored {len(points)} image(s) in collection '{args.collection}'")


if __name__ == "__main__":
    main()
