# JEV deck technical and factual audit

Audit date: 2026-10-13

Scope: all 28 slides in `index.html`, including the six-slide Laya source-code deep dive, presenter notes, diagrams, tables, captions, local benchmark links, and source footers.

Classification used in this report:

- **VERIFIED** — supported by a primary source or directly reproducible local evidence.
- **REPORTED** — stated by a vendor, project author, or community author but not independently verified.
- **INFERRED** — a reasonable technical interpretation or engineering recommendation.
- **SPECULATIVE** — a plausible hypothesis with insufficient evidence.
- **UNSUPPORTED** — previously presented without adequate evidence.
- **INCORRECT** — contradicted by the best available evidence.

## 1. Verdict

**Deliverable after correction — 23/25.** The revised deck has a clear causal spine and now separates TypeSafe's public contract, TypeSafe's own claims, independent preprints, open projects' distinct implementations, local benchmark evidence, and unverified trading anecdotes.

The most important finding is negative but useful: the public record does **not** disclose Jev's backbone, output head, complete training algorithm, calibration procedure, dataset, or serving topology. Laya, Clef, CLM, AutoJev, and Jeff demonstrate that several known architectures can implement a Jev-compatible contract. They do not establish that Jev's architecture was reproduced or that any implementation has matched broad zero-shot, calibration, OOD, or long-tail capability.

The original deck was directionally skeptical but still made four materially misleading moves: it inferred an internal architecture from the API, described the head and training pattern as reproducible, framed a differently tuned local comparison as “Laya beat Gemma,” and attributed an unaudited $31,680 claim to an unsupported X handle. Those statements have been corrected in place.

## 2. Spine Reconstruction

Main causal spine:

> Bounded typed output changes cost and interface, but does not establish correctness, architectural novelty, or safe autonomy.

The title chain now moves through a non-interchangeable causal sequence:

1. Opening contradiction: fast typed decisions, rapid compatible alternatives, and high-risk misuse.
2. Separate mechanism, evidence, and trust.
3. State the conclusion early: skipping decode does not remove decision risk.
4. Define the bounded-decision workload.
5. Draw the public-contract / private-architecture boundary.
6. Explain only the speed mechanism that the public evidence supports.
7–8. Separate useful productization from undisclosed architecture and RLCD.
9–11. Explain interface compatibility and show that open projects use different architectures.
12–17. Tear down Laya’s inference path, source code, public fine-tuning data, soft labels, hybrid RLCD objective, and benchmark confound.
18–20. Define calibration, a bounded routing role, and a fair evaluation protocol.
21. Show a scoped local systems benchmark with its causal limit.
22–25. Move from known failure modes to trading anecdotes and a recommended safety boundary.
26–27. State the evidence-weighted verdict and compress it into a reusable answer.
28. Convert the argument into audience-specific engineering questions.

This is a causal spine rather than a feature list. Removing slides 5–9 would destroy the architecture-evidence boundary; removing slides 12–17 would sever the open implementation from its code, data, and objective; removing slides 18–22 would sever confidence from evaluation; removing slides 23–25 would turn the risk conclusion into assertion rather than explanation.

## 3. Hook Chain

| Stage | Assessment | Evidence in deck |
|---|---|---|
| Cold open | Works, but is conceptual rather than scene-led | Slide 1 asks why bounded decisions produced speed claims, compatible implementations, and a viral loss story. |
| Hold | Strong | Slides 2–3 promise separate answers for mechanism, evidence, and trust and state a falsifiable conclusion early. |
| First payoff | Strong | Slides 4–6 explain the observable workload shift while explicitly leaving the internal network unknown. |
| Second payoff | Strong | Slides 9–17 resolve why alternatives appeared, then audit one implementation from marker tokens through supervision and objective. |
| Application payoff | Strong | Slide 21 connects inference delay to simulated customer outcomes, with matched demand seeds and an explicit best-system caveat. |
| Risk payoff | Strong | Slides 22–25 connect typed-output limitations to scoped preprints, unaudited anecdotes, and a recommended control boundary. |
| Close | Strong | Slides 26–28 return to the opening distinction and give the audience operational questions. |

## 4. Dimension Scores

| Dimension | Score | Reason |
|---|---:|---|
| S1 Narrative spine | 5/5 | One explicit thesis drives mechanism → evidence boundary → evaluation → risk → judgment. The title chain is readable as a standalone argument. |
| S2 Suspense and information release | 4/5 | The opening contradiction and two direct-answer slides sustain curiosity. Deduction: slide 1 opens with a question rather than a concrete incident or image. Original: “What is JEV?” Better future rewrite: open on one clerk waiting 464 ms while simulated customers continue to queue, then ask why bounded decisions change system outcomes. |
| S3 Specificity and anti-buzzword discipline | 5/5 | Claims are labeled as reported, inferred, local, preprint, or unknown. Marketing terms are quoted, scoped, or qualified rather than treated as evidence. |
| S4 Visual necessity | 4/5 | The bounded-output comparison, opaque architecture block, calibration diagram, simulator cards, risk chain, and safety stack encode relationships that prose alone would obscure. Deduction: slide 10 remains a dense reference table. Original: eight ecosystem rows on one slide. Better future rewrite: move the full table to an appendix and keep a model/runtime/gateway classification map in the live talk. |
| S5 Information gain | 5/5 | Each core slide adds a new mechanism, boundary, dataset, failure mode, or operational implication; no sequence can be freely reordered without weakening the argument. |
| **Total** | **23/25** | **Deliverable** under the deck-audit rubric. |

## 5. Required Corrections

All high-severity corrections below were applied to `index.html`. The “Original Claim” column records the pre-audit wording or implication; it is not the current slide text.

| Slide | Original Claim | Classification | Evidence | Action Taken |
|---:|---|---|---|---|
| 1 | Jev “was replicated” quickly. | **UNSUPPORTED** as an architectural claim | TypeSafe's CEO says the architecture remains “close to the chest”; open projects document different designs. | Reframed as “compatible alternatives appeared”; removed the implied architectural reproduction. |
| 1 | The $31,680 story appeared “within weeks.” | **UNSUPPORTED** | No reviewed source established that timing. | Removed the timing claim; retained the story only as a later, explicitly unaudited anecdote. |
| 2 | Speed, replication, and trust can be evaluated together. | **INFERRED** framing | These are distinct systems, reproducibility, and statistical questions. | Preserved as an organizing framework and defined “replication” as interface/data evidence, not cloned internals. |
| 3 | “The API, head, and training pattern are reproducible.” | **INCORRECT** | The API is public; Jev's head and complete training recipe are not. Laya, Clef, CLM, AutoJev, and Jeff use different disclosed approaches. | Replaced with “Only the contract has been copied” and an explicit non-equivalence statement. |
| 3 | Typed output implies correctness or eliminates hallucination. | **INCORRECT** | TypeSafe documents option-order, arithmetic, date, adversarial-content, and other decision failures. | Changed to “Valid output ≠ correct decision”; distinguished schema validity from semantic correctness. |
| 4 | Jev “scores known answers directly.” | **INFERRED** | The API returns probabilities over declared answers, but the internal scoring computation is undisclosed. | Rewrote to the observable API behavior and labeled the displayed probabilities illustrative. |
| 4 | A generative LLM always needs unconstrained JSON parsing. | **UNSUPPORTED** | Constrained decoding can guarantee syntactic structure while still decoding tokens. | Added constrained-decoding nuance and kept the comparison at the workload level. |
| 5 | Jev contains a semantic encoder, candidate-interaction block, and decision head. | **UNSUPPORTED** | No public architecture diagram, parameter count, head, or loss is available. | Replaced the invented middle pipeline with one opaque “Undisclosed Jev computation” block and listed unknowns. |
| 5 | Laya's ModernBERT design reveals Jev's architecture. | **INCORRECT** | Laya's model card documents Laya, not Jev. | Added an explicit warning not to project Laya's backbone or `[MASK]` option markers onto Jev. |
| 6 | Jev is fast because it is an encoder/small model/avoids KV cache. | **UNSUPPORTED** | TypeSafe documents no answer-text generation and parallel questions, but not model size, backbone, hardware, batching, or serving topology. | Limited the explanation to removal of output-length-dependent decoding; labeled the remaining path unknown. |
| 6 | Vendor latency proves a universal speed ranking. | **UNSUPPORTED** | TypeSafe's workflow evaluations are vendor-run and workload-specific; cross-system hardware and serving conditions differ. | Added matched-hardware, batching, region, input, and success-criteria requirements. |
| 7 | Jev's novelty is a known combination of encoder, interaction, and head. | **UNSUPPORTED** | Only the product contract and vendor claim of a new architecture are public. | Changed the title and diagram to separate established alternative methods from undisclosed Jev internals. |
| 8 | RLCD's mechanics can be inferred from Laya or Clef. | **INCORRECT** | TypeSafe publishes the expansion and high-level objective, not reward construction, optimizer, baseline, full loss, data mix, or ablations. | Kept RLCD as a **REPORTED** vendor method and identified Laya/Clef recipes as their own. |
| 9 | “The shell is easy to copy. The API, head, and training pattern are reproducible.” | **INCORRECT** | Wire behavior is observable; the Jev head and training algorithm are not. | Rebuilt the entire slide into public interface, alternative implementations, and unestablished capability equivalence. |
| 9 | Jev has proven broad zero-shot coverage, robust calibration, and long-tail reliability that clones struggle to copy. | **UNSUPPORTED** | No matched independent evidence establishes these broad Jev capabilities. | Reclassified them as evaluation targets, not established Jev properties. |
| 10 | Every item in the ecosystem table is a Jev model. | **INCORRECT** | Ollaya is a runtime/registry; ModernCE is an NLI baseline; `thinkany-ai/autojev` is a gateway, while `denis-pplx/autojev-27b` is a model. | Corrected categories, naming collision, weights status, architectures, and limitations using original repos/model cards. |
| 10 | AutoJev-27B is trivially downloadable/open in the same sense as smaller projects. | **REPORTED** | The model card lists Apache-2.0 weights but notes authentication/private access may be required; BF16 weights are about 49 GiB. | Marked weights “Published*” and added the access and size caveat. |
| 11 | Laya reproduces Jev's internals. | **INCORRECT** | Laya documents a ModernBERT-large backbone and custom two-layer decision head; Jev's internals are undisclosed. | Reframed as an evidence boundary: inspectable Laya implementation details are not a disclosure of Jev. |
| 11 | Laya's 512-token value is a hard universal limit. | **INCORRECT** | It is the English default context; settings can be increased with added cost. | Changed “hard limit” to default context/option budgets and added the configurability caveat. |
| 11 | Laya's ECE/accuracy numbers are independently comparable to Jev. | **UNSUPPORTED** | They are author-run and use Laya's benchmark/tuned split; no matched Jev run is reported there. | Kept numbers as self-reported within-project evidence only. |
| 12 | Raw request JSON is passed directly to ModernBERT. | **INCORRECT** | `_to_internal` and `build_sequence` render the question, options, marker tokens, and serialized state into an explicit sequence before tokenization and collation. | Replaced the generic pipeline with a concrete convenience-store request, exact rendering rules, tensor contract, and truncation limits. |
| 12 | Laya architecture reveals Jev architecture. | **INCORRECT** | TypeSafe has not disclosed Jev internals. | Repeated the Laya-only boundary in the slide title, diagram, and presenter notes. |
| 13 | Marker states are gathered before the Decision Transformer. | **INCORRECT** | The full `[B,L,D]` sequence passes through both additional Transformer layers before `torch.gather` selects option-marker states. | Made “decision head first → gather second” the slide’s title, ordering callout, and highlighted flow transition. |
| 13 | Options are generated or mapped through a fixed classifier head. | **INCORRECT** | `build_sequence` inserts one marker per request-time option; `torch.gather` extracts those marker embeddings and one shared scorer emits logits. | Showed the complete B–G shape flow, shared scorer, fictional softmax output, and separate act-head branch. |
| 14 | `typed-decisions` is Laya's full training corpus. | **INCORRECT** | The dataset card and notebook document a four-workflow domain fine-tuning experiment only. | Added workflow counts, data-generation pipeline, and a prominent provenance boundary. |
| 15 | Soft teacher distributions are ground truth. | **INCORRECT** | The dataset says scores measure agreement with the teacher; it documents a teacher error on a duplicate invoice. | Labeled all example values illustrative and showed both information gain and inherited error. |
| 16 | The current typed-decisions notebook uses pure RLCD / pure policy gradients. | **INCORRECT** | `laya.train.rlcd_loss` returns a REINFORCE-style term **plus** `soft_ce_loss`. | Retitled the slide around the hybrid objective and contrasted notebook prose with executable code. |
| 16 | The notebook fine-tunes the act/escalation head. | **UNSUPPORTED** | It computes `logits, act` but forms loss only from `logits`; the act head receives no direct objective. | Added the caveat to visible content and notes. |
| 17 | 0.766 proves RLCD is superior to supervised fine-tuning. | **UNSUPPORTED** | Domain supervision and objective changed together; no matched hard-CE, soft-CE, RLCD-only, and calibration ablation is published. | Reframed the result as specialization and provided a controlled ablation design. |
| 18 | A model's 0.8 output means one case has an 80% chance of success/profit. | **INCORRECT** | Calibration is a population-frequency statement under matching conditions; TypeSafe's Choice confidence is also a concentration score, not a case guarantee. | Added bucket-level interpretation and trading label/horizon/cost constraints. |
| 19 | Jev is a safety gate for tool calling. | **UNSUPPORTED** | TypeSafe publishes routing examples, but no evidence establishes Jev as a validated safety mechanism. | Relabeled the architecture “Recommended pattern · inference” and Jev as a candidate routing signal behind policy and validation. |
| 20 | Accuracy plus mean latency is a fair benchmark. | **INCORRECT** | Calibration, selective risk, robustness, tail latency, concurrency, and failure rate can change operational outcomes. | Expanded the matrix and required matched datasets, candidates, hardware conditions, repeated runs, and confidence intervals. |
| 21 | “At sub-second timing, Laya beat Gemma.” | **UNSUPPORTED** | As a model-only conclusion, this exceeds the local evidence: best Laya used a 0.1 s tick; best Gemma used 0.5 s; both removed WAIT. Ten matched seeds do not isolate architecture. | Retitled as best Laya policy vs best Gemma policy; added tick-policy and causal-limit disclosures. |
| 21 | Removing WAIT lifted application 30.8% → 100%. | **VERIFIED** | The local summary supports the comparison only when the tick is fixed at 1 s and the model remains Laya. | Added the controlled comparison condition directly on the slide. |
| 21 | Local revenue and abandonment differences forecast production. | **UNSUPPORTED** | The setup is a warmed local discrete-event simulator with ten matched demand seeds and one clerk. | Labeled all values local engineering evidence, not a production forecast. |
| 22 | Probability incoherence and context-attack rates generalize to ordinary Jev use. | **UNSUPPORTED** | Both sources are recent arXiv preprints with scoped protocols: 160 items in one study; adaptive attack search across four systems/seven datasets in the other. | Changed badges to “Independent preprint,” added sample/protocol numbers, and warned against treating them as population error rates. |
| 22 | Typed output eliminates hallucination. | **INCORRECT** | Typed output prevents schema-invalid free text; it does not prevent wrong bounded choices. | Changed conclusion to “may reduce format hallucination; does not eliminate decision error.” |
| 23 | The $31,680 loss was posted by `@MoonGotchi`. | **UNSUPPORTED** | The accessible archived showcase displays “Daily AI Tracker”; no reviewed evidence tied the claim to `@MoonGotchi`. | Removed the handle and X-account link; cited the archived copy and documented the attribution gap. |
| 23 | The $31,680 was an “overnight” audited live loss. | **UNSUPPORTED** | The archived wording says “so far” and supplies no account, wallet, full trade log, starting capital, leverage, or live-status proof. | Labeled it an unaudited self-report and removed “overnight.” |
| 23 | The ~$3,150 Reddit loss proves Jev fails at trading. | **UNSUPPORTED** | It is one self-reported, roughly 24-hour paper simulation with 731 trades and about $1,650 in fees. | Preserved the details but explicitly rejected model-wide causal attribution. |
| 24 | Low latency necessarily causes losses. | **INCORRECT** | Latency can amplify a real edge or amplify friction; the sign depends on net expectancy after costs. | Made “no established edge” an explicit condition and changed the endpoint to negative net expectancy. |
| 25 | The shown stack is the proven safe Jev trading architecture. | **SPECULATIVE** if stated as fact | The cited sources show routing and implementations, not a safety guarantee. | Relabeled as a recommended architecture/inference and stated that controls reduce risk but cannot guarantee safe trading. |
| 26 | Jev is definitively a new model category or definitively old technology. | **UNSUPPORTED** | The architecture and RLCD recipe are undisclosed; useful contract-level differentiation is observable. | Replaced binary judgment with: useful contract; architecture-level novelty and broad superiority unproven. |
| 26 | “Cannot hallucinate” means “cannot be wrong.” | **INCORRECT** | Official limitations and preprints document wrong typed decisions. | Limited the defensible interpretation to schema-invalid output and highlighted branding risk. |
| 27 | Open replications prove Jev capability is easy to copy. | **UNSUPPORTED** | Compatible behavior exists, but no matched evidence demonstrates architectural or broad capability equivalence. | Rewrote the 30-second answer around contract reproducibility and unknown internals/capability. |
| 28 | Confidence can safely own an irreversible decision. | **UNSUPPORTED** | Confidence is not expected utility or a risk control. | Kept this as an explicit discussion challenge and ended on error cost, fallback, and verification. |

### Unsupported or incorrect claims removed

- Jev's internal semantic encoder, candidate-interaction block, and decision head.
- The claim that Jev's output head and training pattern are reproducible from public information.
- Any implication that Laya, Clef, CLM, AutoJev, or Jeff reproduces Jev's architecture.
- Any implication that broad zero-shot generalization, cross-domain calibration, or long-tail reliability has been independently established for Jev.
- Encoder-only, small-model, or KV-cache explanations for Jev's latency.
- Direct comparability of vendor or model-card latency/accuracy numbers collected on different hardware and workloads.
- The unsupported `@MoonGotchi` attribution and “overnight” wording for the $31,680 anecdote.
- The unqualified statement that Laya beat Gemma as a model-only effect.
- The claim that low latency itself causes trading losses.

### Major corrections and why the original wording was misleading

1. **Interface compatibility vs. architectural replication:** “replica” sounded as if hidden internals had been recovered. The evidence supports wire-format compatibility and alternative implementations only.
2. **Reported mechanism vs. invented internals:** a bounded response does not reveal the network that generated it. The old pipeline converted inference into fact.
3. **Vendor benchmarks vs. independent evidence:** TypeSafe's speed and calibration language is valuable product documentation, but not a matched independent benchmark.
4. **Local best-policy comparison vs. model effect:** different tick intervals are part of each system's strategy. The result supports a tuned system choice, not an isolated architecture claim.
5. **Anecdote vs. incident:** the trading figures are self-reports with different evidentiary quality. Neither is an audited incident or a causal test of Jev.

## 6. Source Gaps + Known Limits

### Claims that remain uncertain

- Jev backbone, parameter count, candidate interaction, output head, loss, calibration fitting, training data, RLCD implementation, batching, hardware, and serving topology.
- Whether TypeSafe's claimed “new architecture” is technically novel; no complete paper, ablation, or reproducible recipe is public.
- Broad zero-shot superiority over strong classifiers, cross-encoders, and constrained LLMs under one independent harness.
- Calibration stability by domain, option count, time period, class prevalence, and distribution shift.
- None-of-the-above/OOD rejection quality and long-tail reliability.
- The authenticity, live status, strategy, and actual P&L behind the $31,680 social-media claim.
- Production external validity of the convenience-store simulator; it covers one clerk, ten matched seeds, 300 simulated seconds, warmed local inference, and hand-selected policies.
- Peer-reviewed status and external replication of the 2026 Jev preprints.
- The complete pretraining and base-training corpus behind the original Laya checkpoint; `typed-decisions` documents only one later domain fine-tuning experiment.
- The identity of the approximately 4B-parameter teacher, the exact generation prompts, and the full data-generation runtime used for `typed-decisions`.
- A matched ablation that separates hard-label supervision, soft-label distillation, the RLCD policy-gradient term, and post-hoc temperature fitting.
- Exact historical provenance of the hosted typed-decisions checkpoint relative to the current notebook, which now contains calibration-split safeguards absent from the checkpoint card's reported artifact.

### Primary and original sources used

- TypeSafe: [announcement](https://typesafe.ai/blog/introducing-system-one-models-and-jev), [introduction](https://docs.typesafe.ai/introduction), [models](https://docs.typesafe.ai/models), [primitives](https://docs.typesafe.ai/primitives), [confidence](https://docs.typesafe.ai/confidence), [RLCD primer](https://docs.typesafe.ai/introduction/machine-learning-primer), [Jev 1.13 jaggedness](https://docs.typesafe.ai/model-jaggedness/jev-1.13), [parallel questions](https://docs.typesafe.ai/cookbooks/parallel_questions), [function calling](https://docs.typesafe.ai/cookbooks/function_calling), and [confidence routing](https://docs.typesafe.ai/patterns/confidence-routing).
- TypeSafe CEO discussion: [Hacker News launch thread](https://news.ycombinator.com/item?id=49717558).
- Open implementations: [Laya model card](https://huggingface.co/convaiinnovations/laya), [Laya repository](https://github.com/NandhaKishorM/laya), [Ollaya](https://github.com/ollaya-dev/ollaya), [Jeff](https://github.com/firelex/jeff), [AutoJev-27B](https://github.com/denis-pplx/autojev), [CLM](https://github.com/Contrastive-LM/CLM), [Clef-Flash](https://huggingface.co/Cloudflare/clef-flash), [Cloudflare Clef article](https://blog.cloudflare.com/clef-decision-models/), and [ModernCE](https://huggingface.co/dleemiller/ModernCE-large-nli).
- Independent preprints: [Beyond Calibration](https://arxiv.org/abs/2609.33209), [JevOut](https://arxiv.org/abs/2609.30243), and [JEV-as-a-Judge](https://arxiv.org/abs/2609.26550).
- Trading materials: [$31,680 archived showcase](https://jev.guide/ja/showcases/building-an-autonomous-trading-bot-with-jev-2101897628510494812), [one-day Reddit paper simulation](https://www.reddit.com/r/ai_trading/comments/1wkq4lt/i_let_jev_ai_trade_every_30_seconds_for_more_than/), [jev-trader](https://github.com/jarrodwatts/jev-trader), and [jev-trade](https://github.com/aowang-ai/jev-trade).
- Local simulator evidence: [`report.md`](./assets/convenience-store-benchmark/report.md), [`summary.csv`](./assets/convenience-store-benchmark/summary.csv), and [`summary.json`](./assets/convenience-store-benchmark/summary.json).

### Questions likely to be challenged in Q&A

1. What evidence shows Jev's latency comes from model architecture rather than hardware, batching, or serving design?
2. If the architecture is private, what exactly has the open-source community reproduced?
3. Where is the matched independent benchmark against rules, XGBoost, cross-encoders, local classifiers, and constrained LLMs?
4. Is TypeSafe's “confidence” a calibrated probability, a concentration score, or both, and on which domains was it validated?
5. How does Jev behave when the correct option is absent or the state is adversarial?
6. Why compare the best Laya and Gemma policies if their tick intervals differ? What would a same-policy ablation show?
7. Are the trading losses live, audited, and causally attributable to Jev?
8. What falsifiable result would demonstrate architectural novelty rather than good product packaging?
9. Which decisions in our own system are genuinely bounded, and what is the cost of a confident error?
10. What fallback coverage and selective-risk curve is acceptable before deployment?

### Validation performed

- Parsed all 28 HTML slides after editing: two semantic HTML tables, 28 presenter-note blocks, 28 source footers, sequential slide numbers, and no duplicate IDs.
- Verified that all visible slide content, controls, metadata, alternative text, sources, and print content remain English. CJK text is intentionally confined to the 28 hidden presenter-note blocks, each marked <code>lang="zh-Hant"</code>.
- Rendered the refactored Laya slides at both 1920×1080 and 1440×900. A browser layout probe checked every slide at both viewports and found no descendant outside its slide boundary, with positive separation between content and source footers.
- Browser-tested Home, End, right-arrow navigation, previous/next controls, the 28-card overview, Escape-to-close, Traditional Chinese presenter notes, and the progress counter.
- Confirmed that the notes panel injects nonempty Traditional Chinese notes and preserves `lang="zh-Hant"`. Longer notes remain available through the panel's existing scrolling behavior.
- Exercised the fullscreen control and keyboard path. The current automated Chrome context denies the Fullscreen API by policy; the unchanged handler catches that denial without breaking navigation. The same handler was previously validated in an interactive Chrome session.
- Opened the revised forward-pass slide in a fresh tab and confirmed that the browser console contained no warnings or errors during normal deck use.
- Exported print mode through Chrome and rendered the resulting 28-page, 16:9 PDF. A full contact-sheet review and full-size checks of slides 12–13 found no clipping; the hidden Chinese presenter notes were absent from those PDF pages.
- Checked internal narrative consistency between `index.html`, `outline.md`, `research.md`, and `laya-technical-notes.md`.
- Kept diagrams as editable HTML/CSS and preserved the existing visual and interaction system. Source footers distinguish vendor documentation, author-reported model-card results, independent preprints, local evidence, and unaudited community self-reports.
