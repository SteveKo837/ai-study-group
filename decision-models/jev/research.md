# JEV research summary — current through 2026-10-13

This summary separates evidence layers so that a public API, reasonable inference, and open-source implementation details are not collapsed into “Jev’s official architecture.”

## 1. Officially confirmed by TypeSafe

- Jev is TypeSafe’s first System One model. It takes `state` plus typed questions and returns program-consumable typed values and probability distributions; it does not generate free-form text.
- The official primitives are `Choice`, `Score`, and `Noul`: bounded-set selection, an ordinal scale, and a yes/no probability.
- Questions in one request are evaluated independently and in parallel against the same state. TypeSafe says adding questions has little latency impact.
- TypeSafe expands RLCD as **Reinforcement Learning for Calibrated Decisions**. The public description says the model is trained to output decisions and calibrated probabilities rather than text.
- TypeSafe defines calibration at the population level: among outcomes assigned 0.8 under similar conditions, roughly 80% should be correct over time. It is not a guarantee about one case.
- Official `jev-1.13.0` specifications: text input; 64k tokens per request; 32k for `state` plus the longest question; US$0.042 per million input tokens; output tokens are free.
- Official Jev 1.13 jagged edges include literal readings, numbers and arithmetic, date comparison, indirection, irrelevant long context, adversarial content, conflicting instructions or criteria, choice-option order, and unsuitability for generation.
- TypeSafe’s CEO wrote on Hacker News that the architecture remains “close to the chest.” The exact backbone, parameter count, full dataset, loss, RLCD recipe, calibration procedure, and serving stack are therefore undisclosed.

Primary sources:

- https://typesafe.ai/blog/introducing-system-one-models-and-jev
- https://docs.typesafe.ai/introduction
- https://docs.typesafe.ai/introduction/machine-learning-primer
- https://docs.typesafe.ai/confidence
- https://docs.typesafe.ai/models
- https://docs.typesafe.ai/model-jaggedness/jev-1.13
- https://news.ycombinator.com/item?id=49717558

## 2. Reasonable inference from the API and observed behavior

- The public interface supports a latency hypothesis: TypeSafe says Jev does not generate answer text, emits a tiny fixed result, and evaluates questions independently and in parallel. Actual model compute, hardware, batching, scheduler, and serving topology are undisclosed.
- No internal block diagram should be presented as Jev architecture. Public information cannot establish whether Jev is encoder-only, a decoder backbone with a readout, a dual encoder, a cross-encoder, or something else.
- Small output does not guarantee the lowest end-to-end latency. Network, queueing, serialization, cold starts, and service location can dominate some requests.
- Typed output guarantees a value that is valid under the schema; it does not guarantee semantic correctness. “Cannot hallucinate” should not be interpreted as “cannot make a wrong decision.”
- Router, ranker, filter, or verifier are candidate system roles to benchmark. For high-risk use, the deck recommends separate deterministic validation, policy enforcement, and fallback; this is engineering guidance, not a demonstrated Jev safety property.

## 3. Architectures disclosed by open-source projects—not by Jev

| Project | Verified category | Public implementation details | Key limitation |
|---|---|---|---|
| Laya | Open decision-model family + SDK/runtime | English checkpoint uses ModernBERT-large plus a custom decision head, 421M total; also a 322M multilingual model and typed-decisions checkpoint; its own RLCD and temperature fitting | Weak base zero-shot result on typed-decisions; routing or domain fine-tuning advised; limited context and option budget |
| Ollaya | Runtime / model registry / Jev-compatible API | Rust daemon serving ONNX and llama.cpp models through `/v1/systemone` | It is not a model; capability depends on the backend |
| Jeff | Open decision-model family + LoRA adapters | Qwen3.5 0.8B/2B and Gemma 4 E2B variants; single-forward readout, temperature calibration, domain adapters | Small models do not perform multi-step reasoning; high domain accuracy comes from adapter fine-tuning |
| AutoJev-27B (`denis-pplx`) | Open-weight model + training/server | Qwen3.8-27B full-weight SFT; 73k curated examples; cross-entropy plus separate scalar-temperature calibration | Training corpus not fully included; author benchmarks; not Jev’s architecture; large-GPU requirement |
| CLM | Contrastive decision model / training method / server | Frozen Qwen3-8B state and action encoders with 20M projection heads; InfoNCE; candidate embeddings can be cached | Primarily a choice/ranking design; its dual-encoder structure cannot be attributed to Jev |
| Clef-Flash | Open multimodal decision model | Qwen3.5-9B plus joint schema head; LoRA/post-training; Jev/SystemOne API compatible | Released 2026-10-01; comparisons are primarily author or Cloudflare reports |
| ModernCE | General NLI cross-encoder baseline | ModernBERT NLI cross-encoder for candidate-wise entailment/ranking | Not a Jev clone and does not implement the full native typed-decision contract |

Primary sources:

- https://huggingface.co/convaiinnovations/laya
- https://github.com/NandhaKishorM/laya
- https://github.com/ollaya-dev/ollaya
- https://github.com/firelex/jeff
- https://github.com/denis-pplx/autojev
- https://huggingface.co/denis-pplx/autojev-27b
- https://github.com/Contrastive-LM/CLM
- https://huggingface.co/Cloudflare/clef-flash
- https://blog.cloudflare.com/clef-decision-models/
- https://huggingface.co/dleemiller/ModernCE-large-nli

Naming ambiguity: `thinkany-ai/autojev` is an agent model-routing desktop gateway, not the AutoJev-27B model above.

### Laya code and fine-tuning deep dive — rechecked 2026-10-13

The new slides use the Laya GitHub repository at commit `c73908d96ff73c0062c340a11bf7c54ec7d8bc4d`, the `typed-decisions` dataset repository at commit `e4502d04c0ba18f151188d9756ed838981b49389`, and the current Hugging Face model files.

Code-confirmed details:

- `build_sequence` renders one sequence row per question: `[CLS]` + question type/instructions + `[SEP]` + one `[MASK]` marker per option + `[SEP]` + serialized state + `[SEP]`.
- The English config points to ModernBERT-large and sets a two-layer Transformer decision head. The model card reports 395M backbone parameters and approximately 421M total.
- The forward pass gathers contextual representations at option markers with `torch.gather`, producing `[B,K,D]` from `[B,L,D]`, then uses a shared MLP to emit `[B,K]` logits.
- A separate act/escalation head consumes pooled `[CLS]` plus detached distribution features: top probability, top-two margin, normalized entropy, and option count.
- The inference API divides logits by fitted type/cardinality temperatures and normalizes them with softmax. No answer tokens are generated autoregressively.

Dataset-confirmed details:

- `typed-decisions` has four workflows, each with 300 training and 100 test cases. Five questions per case produce 6,000 training and 2,000 test decisions.
- The card says latent factors are sampled first, a model renders each scenario, and a roughly 4B-class teacher labels it three times at temperature 0.7. The mean distribution becomes soft gold.
- The dataset explicitly says scores measure teacher agreement rather than factual correctness. It documents a duplicate-invoice example the teacher missed.
- This public dataset is a domain fine-tuning experiment. It does not disclose the original Laya checkpoint’s complete data recipe.

Notebook-confirmed details:

- The current two-T4 notebook calls `laya.train.rlcd_loss`: four Gaussian logit perturbations, a group-mean baseline, and a REINFORCE-style term using log + spherical reward with ranked-probability score for ordinal questions.
- The same loss function adds full soft-label cross-entropy. The executable objective is therefore hybrid RLCD + teacher-distribution distillation, despite notebook prose that calls it “pure policy gradients.”
- The notebook computes act/escalation logits but does not include them in the fine-tuning loss.
- The current notebook holds out 10%, capped at 400 items, before training and fits temperatures post hoc. The hosted fine-tuned checkpoint card warns that its historical artifact retained inherited per-option overrides and should be recalibrated.

Developer-reported typed-decisions accuracy is 0.362 for base English Laya, 0.461 for the per-question majority baseline, and 0.766 for the fine-tuned checkpoint on 2,000 test decisions. This demonstrates specialization on the benchmark. It does not isolate RLCD from soft-label distillation because no controlled objective ablation is published.

Additional primary sources:

- https://huggingface.co/convaiinnovations/laya/blob/main/rl_common.py
- https://huggingface.co/convaiinnovations/laya/blob/main/rl_agent_api.py
- https://huggingface.co/convaiinnovations/laya/blob/main/rl_agent_config.json
- https://github.com/NandhaKishorM/laya/blob/main/laya/train.py
- https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb
- https://huggingface.co/datasets/LocalLLaMA/typed-decisions
- https://huggingface.co/convaiinnovations/laya-typed-decisions

## 4. Local applied benchmark: one-clerk convenience store

- A local discrete-event simulation compared one clerk controlled by rule-based logic, Gemma 4, and Laya under live simulated time: customers continued waiting while model inference ran.
- The final experiment evaluated nine strategies across ten matched seeds (42–51), for 90 episodes total. Each episode ran 300 simulated seconds with a maximum of 15 customers and the same customer schedules across strategies.
- The best Laya strategy used a 0.1-second decision tick with WAIT removed. It averaged 63.3 ms inference latency, NT$1,280.9 revenue, 12.9 served customers, and 2.8 abandoned customers.
- The best Gemma strategy used a 0.5-second decision tick with WAIT removed. It averaged 464.2 ms inference latency, NT$1,194.9 revenue, 12.4 served customers, and 3.3 abandoned customers.
- Against the best tested Gemma policy on matched scenarios, the best tested Laya policy produced 7.2% more average revenue, served 4.0% more customers, and had 15.2% fewer abandonments. It won revenue on 6/10 seeds with 2 ties and had fewer abandonments on 6/10 seeds with 3 ties. Because the winning Laya and Gemma configurations used different tick intervals, this is a best-system comparison, not a model-only causal estimate.
- The rule-based baseline remained strong at NT$1,246.3 revenue, 12.7 served, and 2.9 abandoned. Laya's small average advantage over rules was inconsistent across seeds, so the evidence supports Laya over the slower generative controller—not a blanket claim that it beats deterministic policies.
- Action-space design mattered: the Laya 1-second strategy with WAIT available applied only 30.8% of decisions; removing WAIT produced 100% applied decisions. The practical pattern is a bounded model policy plus deterministic validation and fallback.
- This is engineering evidence from a warmed local setup, not a production forecast. Laya MPS inference was measured on the main thread because background-thread Metal execution crashed in the local environment.

Local sources:

- [`report.md`](./assets/convenience-store-benchmark/report.md)
- [`summary.csv`](./assets/convenience-store-benchmark/summary.csv)
- [`summary.json`](./assets/convenience-store-benchmark/summary.json)

## 5. Unverified community or marketing narratives

- The `$31,680` trading loss is available through an archived showcase displaying the author as “Daily AI Tracker.” The accessible copy lacks an auditable account, wallet, full trade log, starting capital, leverage, and strategy. The amount is not an audited loss and cannot be attributed to Jev alone. Evidence reviewed for this audit did not support the deck’s previous `@MoonGotchi` attribution; “overnight” is also absent from the archived wording.
- A separate Reddit paper-trading experiment is more transparent: US$100k simulated capital, three CME micro futures, a decision every 30 seconds, about 24 hours, 731 trades, and a US$3,150 loss. Roughly US$1,650 was fees, leaving an approximately US$1,500 pre-fee loss. It remains a one-day, self-reported paper trade that depends heavily on one prompt and strategy.
- TypeSafe’s 40–200×, 193.6× faster, 444.6× cheaper, and “frontier intelligence” language comes from official workflow evaluations under specific comparison conditions. TypeSafe also says these are high-end gains and that its internal capabilities team created the workflows. The deck labels them as `Official claim`, not a universal law.
- Latency, accuracy, and ECE on open model cards are usually author-run measurements with different hardware, data, prompts, and samples. Cross-project conclusions require the same harness, data, and candidate sets.

Primary sources:

- https://jev.guide/ja/showcases/building-an-autonomous-trading-bot-with-jev-2101897628510494812
- https://www.reddit.com/r/ai_trading/comments/1wkq4lt/i_let_jev_ai_trade_every_30_seconds_for_more_than/
- https://github.com/jarrodwatts/jev-trader
- https://github.com/aowang-ai/jev-trade

## 6. Independent preprints: scoped warnings, not universal estimates

- **Beyond Calibration** (arXiv:2609.33209, submitted 2026-09-27) evaluates 160 items and 480 negation pairs. It reports that Jev complementary probabilities miss summing to one by 0.064 on average. Its comparison is a Qwen3.8-27B first-token readout; it is not a matched study of all decision-model designs.
- **JevOut** (arXiv:2609.30243, revised 2026-10-08) evaluates four systems on seven datasets. Its iterative natural-context attack redirects 61.4–73.2% of initially correct decisions within up to 64 accepted target evaluations. These are attack-search success rates under the paper’s protocol, not ordinary-use error rates.
- **JEV-as-a-Judge** (arXiv:2609.26550) studies an accept/escalate cascade. It supports evaluation of selective routing patterns but does not establish universal judge reliability.
- All three are recent preprints and should be labeled as such; none is a peer-reviewed, comprehensive independent validation of Jev.

Sources:

- https://arxiv.org/abs/2609.33209
- https://arxiv.org/abs/2609.30243
- https://arxiv.org/abs/2609.26550

## Jev details that must remain unknown

- Exact backbone and whether it is encoder-only
- Parameter count
- Complete pretraining and post-training dataset
- Complete loss and RLCD algorithm
- Calibration fitting procedure and per-domain results
- Candidate interaction and decision-head implementation
- Production serving, batching, and hardware topology
