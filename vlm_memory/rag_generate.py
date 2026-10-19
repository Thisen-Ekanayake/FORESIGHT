#!/usr/bin/env python
"""RAG-style VLM pipeline:

  1. Embed a new query image with Qwen3-VL.
  2. Search Qdrant for the most similar previously-ingested image
     (see ingest_image.py) by cosine similarity on the mean-pooled embedding.
  3. Feed BOTH images (query + retrieved match) into Qwen3-VL as a single
     multi-image chat turn. The model internally embeds each image into its
     own token sequence and concatenates them (query image tokens followed
     by matched image tokens) into one input sequence, then generates text
     conditioned on both.

Usage:
    /ml/HelaBERT_Analysis/.venv/bin/python rag_generate.py \
        --image new_image.jpg \
        --collection image_embeddings \
        --prompt "Compare these two images and describe the differences."
"""

import argparse
import os
import time

import torch
from qdrant_client import QdrantClient

from qwen_vl import get_image_embeds, load_model, mean_pool

DEFAULT_PROMPT = (
    "The first image is the new query image. The second image is the most "
    "similar reference image retrieved from the database. Combine the two "
    "and give instructions for an autonomously navitaging robot," 
    "what you see as a combination of both images to reveal environment details,"
)


def main():
    parser = argparse.ArgumentParser(description="Retrieve nearest match from Qdrant and generate text conditioned on both images")
    parser.add_argument("--image", required=True, help="Path/URL to the new query image")
    parser.add_argument("--collection", default="image_embeddings", help="Qdrant collection to search")
    parser.add_argument("--host", default="localhost", help="Qdrant host")
    parser.add_argument("--port", type=int, default=6333, help="Qdrant HTTP port")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Text prompt for generation")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    query_path = args.image if args.image.startswith("http") else os.path.abspath(args.image)

    timings = {}

    t0 = time.perf_counter()
    model, processor = load_model()
    timings["load_model"] = time.perf_counter() - t0

    # 1. Embed the query image
    t0 = time.perf_counter()
    query_embeds = get_image_embeds(model, processor, query_path)
    query_vector = mean_pool(query_embeds)
    timings["embed_query_image"] = time.perf_counter() - t0

    # 2. Search Qdrant for the closest previously-ingested image
    t0 = time.perf_counter()
    client = QdrantClient(host=args.host, port=args.port)
    hits = client.query_points(
        collection_name=args.collection,
        query=query_vector,
        limit=1,
    ).points
    timings["qdrant_search"] = time.perf_counter() - t0

    if not hits:
        raise RuntimeError(f"No matches found in Qdrant collection '{args.collection}'. Ingest some images first with ingest_image.py.")

    top_hit = hits[0]
    matched_path = top_hit.payload["image"]
    print(f"Top match: {matched_path} (cosine similarity={top_hit.score:.4f})")

    # 3. Build a multi-image chat turn: query image + retrieved match.
    # Qwen3-VL embeds each image separately and concatenates their token
    # sequences into the language model's input in the order given here.
    t0 = time.perf_counter()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": query_path},
                {"type": "image", "image": matched_path},
                {"type": "text", "text": args.prompt},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    timings["build_multi_image_prompt"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    timings["generate"] = time.perf_counter() - t0

    generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    print("\n--- Generated text ---")
    print(output_text)

    print("\n--- Timing (seconds) ---")
    for step, elapsed in timings.items():
        print(f"{step:28s} {elapsed:8.3f}")
    print(f"{'total':28s} {sum(timings.values()):8.3f}")


if __name__ == "__main__":
    main()
