# AeroRL Agent Instructions

This repository is built for high-velocity AI coding agents (Claude 4 Sonnet, Cursor, GPT-4o, etc.).

**Rules every agent must follow:**
1. Never change the public API in aerorl/__init__.py unless you bump the major version.
2. All kernels must be in `aerorl/kernels/` as pure Triton Python (no raw CUDA unless absolutely required for IPC).
3. Every new feature must ship with a benchmark delta vs. vanilla verl/TRL on Qwen2.5-VL-7B.
4. Vision token masking is mandatory for any loss function — use the `vision_mask` tensor passed from the VLM processor.
5. Zero-copy KV must survive `torch.compile` and FSDP2.
6. All benchmarks must run on a single RTX PRO 6000 96 GB card and report exact VRAM delta and throughput.

**Preferred workflow:**
- Agent 1 (Architect): Update README + plan new feature
- Agent 2 (Kernel): Implement Triton kernel + test
- Agent 3 (Benchmark): Run on 96 GB card, update tables, create PR

Current priority queue:
1. Finish quantized ref model + DAPO filter fusion
2. Add InternVL2 and Phi-3-Vision processor support
3. Publish wheel to PyPI

When in doubt: make it faster and lower-VRAM than verl on VLMs. Ship numbers, not promises.