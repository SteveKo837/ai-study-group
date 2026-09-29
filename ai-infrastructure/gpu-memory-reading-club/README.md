# GPU Memory Reading Club

**Author:** ChesterHsieh

**Date:** 2026-09-29

**Domain:** AI Infrastructure

## Overview

A four-part reading club (in Traditional Chinese) for ML engineers who write
PyTorch but have not worked with CUDA or computer architecture. It starts from a
single GPU's memory hierarchy and ends at an inference cluster of ~100 GPUs, using
two mental models throughout: the **roofline** (arithmetic intensity vs. ridge
point) and the **memory hierarchy**. The storyline: decode is memory-bound, because
every generated token moves the full weights and KV cache through memory.

All materials are static web pages. Open [`index.html`](./index.html) to start.

| Part | Topic | Slides | Speaker notes |
|---|---|---|---|
| 1 | Hardware × Transformer | [Slides](./slides/full-series.html) | [Notes](./notes/view.html?doc=full-series) |
| 2 | Transformer × GPU: from one card to many | [Slides](./slides/class2-transformer-gpu.html) | [Notes](./notes/view.html?doc=class2-transformer-gpu) |
| 3 | Single-node inference engines: SGLang × vLLM | [Slides](./slides/class3-engine-single-node.html) | [Notes](./notes/view.html?doc=class3-engine-single-node) |
| 4 | Finale: from models to racks | [Slides](./slides/class4-models-to-racks.html) | [Notes](./notes/view.html?doc=class4-models-to-racks) |

## Materials

- [Site entry point](./index.html)
- [Review quiz](./quiz/index.html) (8 questions per part, with targeted hints)
- Interactive explainers: [GPU](./interactive/gpu-map.html), [Transformer](./interactive/transformer-map.html), [Parallelism](./interactive/parallelism-map.html), [Interconnect](./interactive/interconnect-map.html), [Serving](./interactive/serving-map.html), [Rack journey](./interactive/rack-journey-map.html)
- [Glossary](./notes/view.html?doc=glossary)

The speaker notes are Markdown files in [`notes/`](./notes/) rendered in the
browser by `notes/view.html`, which loads them with `fetch`. To view them locally,
serve the folder over HTTP (`python3 -m http.server`) rather than opening the
file directly; on GitHub Pages this works without any setup.

## Source

Only the web version is included here. The slide generator scripts, the five
reproducible PyTorch demos, and the full project history live in
[ChesterHsieh/Always_try_to_learn](https://github.com/ChesterHsieh/Always_try_to_learn/tree/main/gpu-memory-reading-club).

## References

- [SGLang](https://github.com/sgl-project/sglang)
- [vLLM](https://github.com/vllm-project/vllm)
- [DeepSeek-V3/R1 Inference System Overview](https://github.com/deepseek-ai/open-infra-index)
