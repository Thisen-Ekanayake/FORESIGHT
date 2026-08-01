# Compact VLM comparison for milestone 5 (scene reasoning)

Research done 2026-07-21 to pick a VLM for `src/foresight/perception/vlm/`. This is the model that reasons about the RGB scene at the slow planning loop (0.5 to 2 Hz), alongside the depth model from milestone 4. Constraint: it has to share an 8 GB laptop GPU (RTX 4060) with that depth model, so VRAM budget matters as much as raw capability.

Note on timing: my training knowledge cuts off in January 2026 and this is a fast-moving space, so this survey leans on a live web search done today rather than memory. Worth a quick re-check before the paper's camera-ready if there is a long gap between now and then.

## Candidates

| Model | Params (active) | Approx VRAM, quantized | License | Notes |
|---|---|---|---|---|
| Gemma 4 E2B | ~2.3B effective | ~1.5 GB (4 bit QAT) | Apache 2.0 | Google, native vision, quantization aware training so 4 bit quality stays close to bf16, elastic MatFormer architecture built for on device use, full `transformers` support |
| Gemma 4 E4B | ~4.5B effective | ~5 GB (4 bit QAT) | Apache 2.0 | Same family as above, one size up, stronger reasoning, still fits our card with room for the depth model |
| Qwen3-VL-2B-Instruct | 2B dense | ~2 to 3 GB (4 bit) | Apache 2.0 | Alibaba, strong OCR and document understanding, hour long video understanding with timestamp localization, part of the current leading open weight VLM family |
| Qwen3.5 (0.8B to 4B multimodal) | 0.8B to 4B | well under 2 GB at 0.8B | Apache 2.0 | Newer and smaller than Qwen3-VL, supports a thinking and non thinking mode, good option if latency matters more than raw reasoning depth |
| SmolVLM2-2.2B-Instruct | 2.2B | ~5.2 GB (measured, video inference) | Apache 2.0 | Hugging Face's dedicated compact VLM line, learns spatial layout well via dedicated positional tokens, easiest to fine tune later if we ever need to |
| Moondream (2B classic) | ~1.9B | ~1 to 2 GB (4 bit checkpoint available) | Moondream Model License (free for research, personal, and most commercial use, not a standard OSI license) | Purpose built for edge devices, has a built in "point to object" grounding capability that could map directly to navigation queries like "point to the doorway" |
| Moondream 3.1 | 9B total, 2B active (MoE) | ~19 GB measured | Moondream Model License | Newer flagship Moondream, frontier level reasoning claimed, but the MoE keeps all experts resident in VRAM so the memory footprint is too large for our card even though only 2B are active per token |
| PaliGemma 2 (3B) | 3B | ~4 to 6 GB | Gemma license (custom, commercial use allowed subject to a Prohibited Use Policy) | Very strong at object localization and referring expression tasks, but the license is more restrictive than Apache 2.0 and there is now a same size Apache licensed alternative in Gemma 4 E4B |
| Ministral 3 (3.4B) | 3.4B LM + 0.4B vision encoder | ~8 GB at FP8, less quantized further | Apache 2.0 | Mistral's edge focused release, single GPU friendly, less of a track record for spatial or navigation style tasks in the sources found |

## Recommendation

**Primary: Gemma 4 E4B.** Apache 2.0, native vision support, and the quantization aware training means the 4 bit checkpoint (about 5 GB) does not lose much quality compared to full precision, which matters because we are quantizing out of necessity, not choice. It leaves enough headroom on the 8 GB card once the small depth model (well under 1 GB) is also loaded. Full `transformers` support means it slots into the same stack we already used for the depth model in milestone 4.

**Fallback if VRAM is still tight: Gemma 4 E2B.** Same family, same license, same integration path, just smaller (about 1.5 GB in 4 bit) with a modest reasoning quality drop.

**Alternate if we want stronger OCR, document, or video reasoning specifically: Qwen3-VL-2B-Instruct.** Also Apache 2.0, also fits comfortably, and is currently described as the leading open weight VLM family. Worth a look if Gemma 4's scene reasoning turns out to be weaker on cluttered or text heavy HM3D rooms.

**Not recommended as primary:** PaliGemma 2, because Gemma 4 E4B now covers similar ground under a fully permissive license instead of the restricted Gemma license. Moondream 3.1, because its VRAM footprint is too large for this hardware even though only part of it is active per token. Moondream 2 (classic) stays interesting as a specialized fallback if we later need explicit point based object grounding rather than free form reasoning.

## Open question for the team

Do we want a model that reasons in free text (Gemma 4, Qwen3-VL) or one with a built in grounding primitive like Moondream's "point" capability that maps more directly onto "which direction is clear"? Free text reasoning is more flexible for the action fusion controller described in the proposal, but grounding primitives could make the reactive safety layer's job easier. Flagging this rather than deciding it unilaterally since it shapes how milestone 6 (action fusion) gets built.

## Sources

- [Ultimate Guide, The Best Small LLMs For Edge Devices In 2026](https://www.siliconflow.com/articles/en/best-small-llms-for-edge-devices)
- [Best Open-Source Vision Language Models of 2026](https://www.labellerr.com/blog/top-open-source-vision-language-models/)
- [Multimodal AI: The Best Open-Source Vision Language Models in 2026](https://www.bentoml.com/blog/multimodal-ai-a-guide-to-open-source-vision-language-models)
- [Top 10 Vision Language Models in 2026, DataCamp](https://www.datacamp.com/blog/top-vision-language-models)
- [Best Vision-Language Models (VLMs) in 2026, Tested and Ranked, Mixpeek](https://mixpeek.com/curated-lists/best-vision-language-models)
- [GitHub, QwenLM/Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)
- [Qwen3.5 small models: Everything you need to know](https://artificialanalysis.ai/articles/qwen3-5-small-models)
- [Best Qwen Models Ranked: Which to Run Locally (Mid-2026), InsiderLLM](https://insiderllm.com/guides/qwen-models-guide/)
- [Moondream Blog, Model Release category](https://moondream.ai/blog/categories/model%20release)
- [Moondream 3 Preview: Frontier-level reasoning at a blazing speed](https://moondream.ai/blog/moondream-3-preview)
- [moondream/moondream3.1-9B-A2B, Hugging Face](https://huggingface.co/moondream/moondream3.1-9B-A2B)
- [Install Moondream 3 Locally](https://sonusahani.com/blogs/moondream-3)
- [PaliGemma 2: A Family of Versatile VLMs for Transfer](https://arxiv.org/pdf/2412.03555)
- [SmolVLM: Redefining small and efficient multimodal models](https://arxiv.org/html/2504.05299v1)
- [HuggingFaceTB/SmolVLM2-2.2B-Instruct, Hugging Face](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct)
- [SmolVLM2: Overview, Resources and Licensing, Roboflow](https://playground.roboflow.com/models/hugging-face/smolvlm2)
- [Gemma 4 Guide: E2B, E4B, 26B MoE and 31B Open Weights (2026)](https://codersera.com/blog/gemma-4-complete-guide-2026/)
- [A Visual Guide to Gemma 4, Maarten Grootendorst](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-gemma-4)
- [Gemma 4 with quantization-aware training, Google blog](https://blog.google/innovation-and-ai/technology/developers-tools/quantization-aware-training-gemma-4/)
- [Gemma 4 QAT, Unsloth Documentation](https://unsloth.ai/docs/models/gemma-4/qat)
- [Gemma 4 model overview, Google AI for Developers](https://ai.google.dev/gemma/docs/core)
- [Ministral 3 (3B) vs Qwen3.5-2B Comparison, llm-stats.com](https://llm-stats.com/models/compare/ministral-3-3b-base-2512-vs-qwen3.5-2b)
- [Open-Weight License Landscape 2026, Presenc AI](https://presenc.ai/research/open-weight-license-landscape-2026)
