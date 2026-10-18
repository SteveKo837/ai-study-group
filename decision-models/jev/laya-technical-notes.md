# Laya technical deep-dive notes

Research date: 2026-10-13

Purpose: evidence ledger for slides 11–17 of the JEV deck.

These notes describe **Laya**, an open implementation of a Jev-compatible typed-decision workload. They do not describe TypeSafe Jev’s undisclosed architecture.

## Source snapshot

- Laya GitHub commit: `c73908d96ff73c0062c340a11bf7c54ec7d8bc4d`
- `typed-decisions` dataset commit: `e4502d04c0ba18f151188d9756ed838981b49389`
- Model repository files reviewed: `rl_common.py`, `rl_agent_api.py`, `rl_agent_config.json`
- Training files reviewed: `laya/train.py`, `notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb`, and the MPS wrapper

Primary links:

- https://huggingface.co/convaiinnovations/laya
- https://github.com/NandhaKishorM/laya
- https://huggingface.co/datasets/LocalLLaMA/typed-decisions
- https://huggingface.co/convaiinnovations/laya-typed-decisions

## Code-confirmed inference path

1. `RLAgent._to_internal` keeps string instructions unchanged and normalizes a list-form choice criterion into an insertion-ordered mapping. `build_sequence` then renders the instruction exactly as `{qtype} question: {instructions}` and builds `[CLS] rendered-instruction [SEP] [MASK] option ... [SEP] state [SEP]`.
2. Each option is tokenized with a leading space and capped at 48 text tokens. The code records `len(ids)` immediately before appending each `[MASK]`-prefixed option; `marker_pos` therefore contains sequence position indices, not token IDs or embeddings.
3. Non-string state is serialized with `json.dumps(state, ensure_ascii=False)`. The English API defaults to right truncation for state, preserving the beginning when the final sequence exceeds its remaining budget. Special token IDs are inserted manually while text fragments use `add_special_tokens=False`.
4. The English config points to `answerdotai/ModernBERT-large`, sets `head_layers` to 2, `max_len` to 512, and `head_max_len` to 192.
5. `DecisionModel.forward` runs the bidirectional encoder, adds a learned question-type embedding, and applies two additional Transformer encoder layers to the complete `[B, L, D]` sequence.
6. Only after those two head layers finish does `torch.gather()` extract the contextual representation at every option marker, producing `[B, K, D]` from `[B, L, D]`. The gather is differentiable.
7. A shared LayerNorm–Linear–GELU–Linear scorer produces one scalar logit per option.
8. `RLAgent.system_one` divides logits by a type/cardinality temperature and applies softmax. `choice`, `score`, and `noul` then decode the same distribution differently.
9. The act/escalation head is separate. It reads pooled `[CLS]` plus detached top-1 probability, top-1 margin, normalized entropy, and option count. Its code path does not by itself establish calibrated correctness.

The model card reports approximately 421M total parameters: a 395M ModernBERT-large backbone plus the decision head. That count is author-documented rather than independently recomputed here.

## `typed-decisions` provenance

The public dataset documents four synthetic workflows:

| Workflow | Train cases | Test cases |
|---|---:|---:|
| Agent Trace Observability | 300 | 100 |
| Customer Service | 300 | 100 |
| Invoice Processing | 300 | 100 |
| Security Incidents | 300 | 100 |

Each case contains five typed decisions. The `all` configuration therefore provides 1,200 training cases / 6,000 decisions and 400 test cases / 2,000 decisions.

The dataset card says each case begins with independently sampled latent factors such as topic, tone, severity, or discrepancy type. A model renders the scenario, then a roughly 4B-class teacher labels the case three times at temperature 0.7. The released gold distribution is the mean of those three teacher distributions.

Evidence boundary: this is a public domain fine-tuning and evaluation dataset. It is **not** the complete pretraining or post-training corpus used for the original Laya checkpoint. The public materials reviewed do not fully establish that original data recipe.

## RLCD as implemented by the current notebook

`laya.train.rlcd_loss` is a hybrid objective:

- Four zero-mean Gaussian perturbations are added to detached logits.
- Sigma is annealed from 0.4 to 0.1 across four notebook epochs.
- Each perturbed distribution receives a proper-rule reward: log score + 0.75 × spherical score, with a ranked-probability penalty for ordinal `score` questions.
- Reward minus the four-sample mean is normalized and used as a REINFORCE-style advantage over the Gaussian log density.
- The function then **adds full soft-label cross-entropy** on the unperturbed logits.

Therefore the fine-tuning notebook is not an RL-only comparison. It combines a policy-gradient term with direct teacher-distribution distillation.

The notebook computes `logits, act` but constructs loss only from `logits`. The act/escalation head’s own weights receive no direct loss in this typed-decisions fine-tune.

The current notebook holds out 10% of items, capped at 400, before training and fits temperatures afterward. The hosted `laya-typed-decisions` checkpoint card states that its published checkpoint was created before this correction: inherited per-option temperatures can shadow newly fitted per-type temperatures, so confidence should be refit on held-out domain data.

## Reported benchmark and correct interpretation

Developer-reported accuracy on the 400-case / 2,000-decision test split:

| Model | Accuracy |
|---|---:|
| Laya base English checkpoint | 0.362 |
| Per-question majority baseline | 0.461 |
| Laya typed-decisions checkpoint | 0.766 |

The result supports this claim: domain fine-tuning substantially improved argmax teacher-label agreement on the four workflows.

It does not isolate the cause of the improvement. A controlled study still needs the same backbone, data, seeds, and training budget across:

1. hard-label supervised cross-entropy;
2. soft-label cross-entropy / distillation;
3. RLCD without auxiliary cross-entropy;
4. RLCD plus post-hoc temperature scaling.

Accuracy should be reported with NLL, Brier score, ECE, latency, selective risk, and out-of-distribution generalization. Agreement with teacher-derived gold is not equivalent to factual correctness.

## Claims not fully verified

- The complete data mixture and training recipe used for the original Laya checkpoint.
- Whether the current reproduction notebook exactly matches the historical run that produced the hosted fine-tuned weights; current notebook settings and hosted checkpoint metadata differ.
- The identity and exact prompts of the dataset’s roughly 4B-class teacher from the public dataset card alone.
- Any controlled evidence that RLCD is superior to soft-label supervised distillation on this benchmark.
- Calibration stability outside the fitted domains, option-count buckets, and held-out sample.
- Architectural or training equivalence between Laya and TypeSafe Jev.
