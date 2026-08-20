---
title: "DAG patching: literature pass and source matrix"
type: research-review
status: complete
date: 2026-08-16
---

# DAG patching — literature pass and source matrix

**Role of this file.** This is the evidence record: what each source did, and
what each source rules out. It does not govern the paper's claims. The thesis,
the contribution list and the wording rules live in
[`2026-08-16-dag-literature-and-claim-boundary.md`](2026-08-16-dag-literature-and-claim-boundary.md),
and where the two disagree, that note wins. Verdict, ranked-survivor and
terminology sections that previously duplicated it have been removed from here.

Scope of the check: the experiment described in [`PAPER_STRATEGY_DAG.md`](../../PAPER_STRATEGY_DAG.md)
and [`results/dag_patching/e3_ladder/README.md`](../../results/dag_patching/e3_ladder/README.md) —
`DeepSeek-R1-Distill-Qwen-1.5B` prompted with a generated single-digit arithmetic
program whose dependency graph is known by construction; residual-stream
activation patching at layer 13 writing a donor trace's state at the clean
trace's own token positions; three digits kept distinct per row (**clean**,
**implied**, **raw**); the headline being 555/603 implied-digit installs at one
written step against 0/432 at two or more, at overlapping token distances, plus a
within-item 144/144-vs-0/144 dissociation and an omission arm.

**`RELATED_WORK.md` does not cover any of this.** I read it end to end: its
declared scope is MATH-500 selective prediction with `rmd_tail_q20` on 7–8B
models, its eleven "nearest neighbours" are all uncertainty-quantification papers,
and the string "patching" does not appear in it in the interventional sense. It
is the RMD paper's literature pass and shares no citation with this one. The
§7 write-up debt item "the DAG literature pass" is what this file discharges.

Every paper below was resolved by fetching its primary source. Rows are marked
**[full text]** where I read the paper's own HTML/PDF body, **[abstract]** where I
could only reach the arXiv/anthology abstract page, and **[listing]** where I only
resolved title/authors/venue through a search index and could not open the paper
itself. No blog summary, thread, or explainer is cited as evidence for what a
paper does.

---

## 1. Novelty matrix

Columns as commissioned. `not reported` means the paper does not state it; it is
not a guess about what they might have done.

### 1a. Closest overall

| Paper | Task | Model | Intervention type | Unit patched | Semantic outcome measured | Graph ground truth | Controls | What remains different here |
|---|---|---|---|---|---|---|---|---|
| [Shih, Winnicki, Darve, *Do Models Read What They Write? Causal Registers in Scratchpad Reasoning*](https://arxiv.org/abs/2606.29522) ([HTML](https://arxiv.org/html/2606.29522)) **[full text]** | Two eight-state finite transition systems (Q₈, D₈); each state is a visible coordinate plus an order-sensitive phase bit, updated by a known row-dependent rule | Qwen2.5-Coder-7B in three conditions — pretrained base, final-answer-only LoRA, running-state LoRA (r=16, α=32), identical move sequences and final states — replicated across model families | Rank-16 projection edit into a phase-bit subspace estimated from class-mean differences on a calibration split | Residual stream (`resid_pre`) at layer 12, at the current-state token position; **the printed token and the visible coordinate are unchanged by construction** | Agreement with the rule-consistent next phase bit, and counterfactual-update selectivity against the original branch: 80% (Q₈) and 91% (D₈) for the running-state model, pretrained and final-answer controls near baseline | Yes — the transition rule is known, so the edit has a single correct downstream consequence | **Move-swap** (+0.57/+0.68): edited state fixed, upcoming move varied, ruling out a fixed answer bias. **Conflicting continuation** ("computed not copied", +0.59/+0.81): a real occurrence of the counterfactual state injected from a context whose future disagrees with the current move. Plus random-rank and orthogonal-complement patches (≈0.02), full-residual edits, matched visible-coordinate edges, restoration tests | **The nearest precedent, and it forecloses three claims**: first causal intervention on written state, first exact scratchpad counterfactual, first computation-versus-copy control. What remains different is granularity and scope. Their copy competitor is another context's *continuation*, read off a contrast between two rule-defined branches; literal token copying is never scored as its own outcome category, and their edit leaves the written token fixed, so no digit on the page could be copied. Ours scores three digits distinct by construction per row. Their positive result needs task-specific running-state supervision — pretrained controls stay near baseline — while ours is on a pretrained reasoning-tuned checkpoint. They do not systematically vary written steps or token distance; branch persistence to k=4 on generated prefixes is secondary reporting, not a design variable. |
| [Kudo et al., *LLMs Faithfully and Iteratively Compute Answers During CoT*](https://aclanthology.org/2026.findings-eacl.59/) ([arXiv:2412.01113](https://arxiv.org/abs/2412.01113)) **[full text, HTML v2]** | Generated single-digit +/− assignment chains (`A=1+B, B=2+3; A=?`), five levels graded by `#Step`, `#Stack`, `#Dist.` | Qwen2.5-7B/14B/32B, Qwen2.5-Math-7B, Yi1.5-9B/34B, Llama3.1-8B, Llama3.2-3B, Mistral-Nemo-12B — all pretrained | Activation patching with a counterfactual input; also linear probes across layers/positions | Residual hidden states `H⁺ = {h_{t,l} : l ≥ 12, −5 ≤ t ≤ −4}` — **input/equation region only, before CoT generation begins** | "Success rate" = generated answer equals the **counterfactual gold answer**; "unchanged rate" = equals the original answer | Yes — the equation set defines the dependency structure and both gold answers are known in closed form | Low-probe-accuracy region `H⁻` as a negative control (never changes the output); distractor equations; ten models | They never patch inside the generated CoT; **no step-distance analysis of the patching effect** (verified against the paper: absent); **no copy competitor** — nothing distinguishes "carried the counterfactual value" from "reproduced a written digit"; no omission/read-vs-recompute arm. Their headline claim (the model computes sub-answers on the fly) is about *when a value first becomes available* during generation; ours is about *what determines the answer once a value is written*. Not a contradiction, but close enough in the title that the manuscript must distinguish the two in as many words. |
| [Ameisen, Lindsey, Pearce et al., *Circuit Tracing* / *On the Biology of a Large Language Model*](https://transformer-circuits.pub/2025/attribution-graphs/biology.html) (Anthropic, 27 Mar 2025) **[full text]** | Open-ended prompts, incl. two-step factual recall ("the capital of the state containing Dallas") and 36+59 addition | Claude 3.5 Haiku (production model) | Feature-level intervention on a cross-layer-transcoder replacement model; "constrained patching" over layer ranges | Feature activations (Texas cluster inhibited, California cluster activated) | The generated answer token: swapping the intermediate yields "Sacramento", "Atlanta", "Victoria", "Beijing" | Attribution graph is *recovered*, not given; the Dallas→Texas→Austin chain is the recovered graph | Graph-vs-intervention agreement checks; acknowledged limits on attention and global circuits | **This is the paper that kills "we show a patched intermediate is transformed, not copied" as a first.** But their intermediate is latent, not written, so there is no raw digit at the patched position and the copy confound cannot arise; there is no ladder past one intermediate; and the unit is a transcoder feature, not a residual state at a matched token position. |
| [Brinkmann, Sheshadri, Levoso, Swoboda, Bartelt, *A Mechanistic Analysis of a Transformer Trained on a Symbolic Multi-Step Reasoning Task*](https://aclanthology.org/2024.findings-acl.242/) ([arXiv:2402.11917](https://arxiv.org/abs/2402.11917), HTML v3) **[full text]** | Path-finding on a generated binary tree: shuffled edge list + goal + root, predict the path node by node | 6-layer decoder-only transformer, 1 head/layer, d=128, 1.2M params, **trained from scratch** on 150k generated trees | Linear probes; activation patching (resampling ablation); causal scrubbing; attention knockout; register-token patching after layer 4 | Residual stream at specific token positions, incl. "register tokens"; attention weights | Whether the model still emits the correct next path node; probe F1; causal-scrubbing performance recovered | The tree is generated and given in the prompt, so the dependency structure is known — but it is not used as an interventional reference; mechanisms are reverse-engineered inductively | Probes at 1.00 F1; causal scrubbing recovering ≈100%; attention knockouts | **This is the paper that kills "intermediate results are stored at token positions" as a finding.** It also owns "depth-bounded" — but their bound is *L−1 layers*, an architectural limit on a toy model, not a step boundary inside a written CoT of a pretrained model. Their mechanism is explicitly a copy (`[A][B]...[B] → [A]`); no arithmetic is applied to a patched value, so propagate-vs-copy is answered "copy" by construction rather than measured. |
| [Garcia, *The Last Word Often Wins: A Format Confound in Chain-of-Thought Corruption Studies*](https://arxiv.org/abs/2605.10799) (May 2026) **[abstract]** | GSM8K/MATH-style CoT corruption with and without an explicit terminal answer line | Qwen2.5-3B/7B plus five open-weight families at 3–7B | Text corruption of the CoT suffix (not activation patching) | Written tokens — specifically the final answer line | Whether the model follows the corrupted/wrong answer | No | Ablating only the terminal answer line while preserving all reasoning steps; scale sweep | **The most dangerous paper for our framing, not for our result.** It argues that corruption sensitivity "tracks the location of explicit answer text, not a fixed computational depth" — i.e. exactly the reading a reviewer will impose on a depth effect. Our token-distance banding (D2) and within-item pairing (D7) mitigate it — they show the boundary is not explained by placement alone — but they do not dispose of it: step count and site role stay bundled, the arms are not an exhaustive placement sweep, and their setting is text corruption rather than activation patching, so the mapping between the designs is itself an assumption. Cite it and state both the mitigation and the residual. |

### 1b. Activation patching / causal tracing methodology

| Paper | Task | Model | Intervention type | Unit patched | Semantic outcome measured | Graph ground truth | Controls | What remains different here |
|---|---|---|---|---|---|---|---|---|
| [Vig et al., *Investigating Gender Bias in Language Models Using Causal Mediation Analysis*](https://arxiv.org/abs/2004.12265) **[abstract]** | Gendered pronoun/occupation continuation | Pretrained transformer LMs (GPT-2 family) | Causal mediation analysis: direct vs indirect effect | Neurons and attention heads as mediators | Change in the gendered continuation probability | No | Decomposition into direct/indirect; sparsity and synergy analyses | No multi-step task, no written intermediates, no counterfactual answer in closed form |
| [Meng, Bau, Andonian, Belinkov, *Locating and Editing Factual Associations in GPT* (ROME)](https://arxiv.org/abs/2202.05262) **[abstract]** | Factual recall (zsRE, CounterFact) | GPT-2 XL, GPT-J | Causal tracing: corrupt subject tokens, restore clean activations | Hidden states at (layer, token) | Recovery of the original factual prediction | No | Corrupted-run baseline; severed-MLP/attention traces | Single-hop recall, no arithmetic, no step ladder, no copy competitor |
| [Wang, Variengien, Conmy, Shlegeris, Steinhardt, *Interpretability in the Wild* (IOI)](https://arxiv.org/abs/2211.00593) **[abstract]** | Indirect object identification | GPT-2 small | Path patching / causal interventions | Attention heads and the paths between them | Logit difference between the two candidate names | No — the circuit is discovered | Faithfulness, completeness, minimality | Single-step task; the "graph" is the discovered circuit, not the task's dependency structure |
| [Goldowsky-Dill, MacLeod, Sato, Arora, *Localizing Model Behavior with Path Patching*](https://arxiv.org/abs/2304.05969) **[abstract]** | Induction; a GPT-2 behavior | 2-layer attention-only; GPT-2 | Path patching, defined as a testable hypothesis class about localization to paths | Specific paths between components | Behavior-specific loss/logit metrics | No | The paper is itself the formalization of the control | Defines the term we must *not* misuse: we patch a residual state and let all downstream paths see it, which is plain activation patching |
| [Zhang & Nanda, *Towards Best Practices of Activation Patching: Metrics and Methods*](https://arxiv.org/abs/2309.16042) ([HTML](https://arxiv.org/html/2309.16042)) **[full text]** | IOI, factual recall | GPT-2, Pythia | Systematic comparison of patching hyperparameters | Residual stream / heads / MLPs | Probability, logit difference, KL divergence, compared head to head | No | Gaussian-noise vs symmetric-token-replacement corruption; denoising vs noising | Directly load-bearing for us as a *methods citation*, not a competitor: they recommend logit difference over probability ("may fail to detect negative model components", §6) and STR over Gaussian noise because GN "puts the model off distribution". Our donor is an STR-style in-distribution counterfactual and our margin statistic is a logit difference — say both. |
| [Heimersheim & Nanda, *How to use and interpret activation patching*](https://arxiv.org/abs/2404.15255) **[abstract]** | Tutorial | — | — | — | — | — | — | The vocabulary reference: it treats *activation patching*, *causal tracing*, *interchange intervention* and *resample ablation* as names for the same family, and separates *denoising* from *noising*. Cite it for terminology, not for a result. |
| [Syed, Rager, Conmy, *Attribution Patching Outperforms Automated Circuit Discovery*](https://arxiv.org/abs/2310.10348) **[abstract]** | Circuit recovery across several tasks | GPT-2 family | Edge attribution patching: linear approximation to patching, 2 forward + 1 backward pass | Edges of the computational subgraph | AUC of circuit recovery against a reference circuit | Reference circuits, not task graphs | Comparison against ACDC | Approximation method for scale; we do exact patching at a handful of sites and are not doing circuit discovery |
| [Fernandez-Boullon & Olivieri, *Patch-Effect Graph Kernels for LLM Interpretability*](https://arxiv.org/abs/2605.06480) (May 2026) **[abstract]** | IOI and variants | GPT-2 small | Activation patching profiles turned into graphs | Model components | Classification accuracy over graph features | No — explicitly compares against prompt-only and raw patch-effect controls | Prompt-only and raw patch-effect controls | Graph-ML over patching profiles; no task dependency graph, no arithmetic, no written CoT |

### 1c. Causal abstraction / interchange interventions — the most dangerous family for "known graph as ground truth"

| Paper | Task | Model | Intervention type | Unit patched | Semantic outcome measured | Graph ground truth | Controls | What remains different here |
|---|---|---|---|---|---|---|---|---|
| [Geiger, Lu, Icard, Potts, *Causal Abstractions of Neural Networks*](https://arxiv.org/abs/2106.02997) (NeurIPS 2021) **[abstract]** | MQNLI (multiply-quantified NLI) | BERT-based model and a weaker baseline, both trained on the task | **Interchange intervention**: swap the representation aligned with a high-level variable between two inputs | Neural representations aligned to variables of a tree-structured natural-logic causal model | Whether the output matches the high-level model's counterfactual prediction | **Yes — an explicit high-level causal model is the reference.** This is the origin of the move | Baseline model that fails to show the structure | This is the paper that owns "use a known computation as an interventional reference". Ours differs only in that the reference is *generated per item* and the counterfactual answer is closed-form arithmetic; that is a convenience, not a novelty |
| [Geiger, Wu, Potts, Icard, Goodman, *Finding Alignments Between Interpretable Causal Variables and Distributed Neural Representations* (DAS)](https://arxiv.org/abs/2303.02536) **[abstract]** | Alignment search | Various | Interchange intervention in a learned rotated basis, found by gradient descent | Subspaces of the residual stream | Interchange intervention accuracy against the high-level model | Yes | Comparison against brute-force alignment search | Learns *where* the variable lives; we fix the site by construction. If a reviewer asks "why layer 13, why that position", DAS is the method they have in mind |
| [Wu, Geiger, Potts, Goodman, *Interpretability at Scale: Identifying Causal Mechanisms in Alpaca* (Boundless DAS)](https://arxiv.org/abs/2305.08809) (NeurIPS 2023) **[abstract]** | A simple numerical reasoning problem (price tagging) | Alpaca 7B — **pretrained, instruction-tuned, off the shelf** | Boundless DAS interchange interventions | Learned subspaces | Interchange intervention accuracy ≈94% | Yes — a hypothesized causal model with two boolean variables | Robustness to input and instruction changes | **This is the row that kills "first causal-abstraction-style intervention on a pretrained model doing numeric reasoning".** But the causal model has two boolean variables, not a multi-step arithmetic chain, and there is no written intermediate anywhere |
| [Geiger et al., *Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability*](https://arxiv.org/abs/2301.04709) **[abstract]** | Theory | — | Unifies "activation and path patching, causal mediation analysis, causal scrubbing, causal tracing, circuit analysis, ... distributed alignment search, and steering" | — | — | — | — | The vocabulary authority. Cite it for what "interchange intervention" and "causal abstraction" mean, and to make clear we are *not* claiming an abstraction |
| [Chan, Garriga-Alonso, Goldowsky-Dill, Greenblatt, Nitishinskaya, Radhakrishnan, Shlegeris, Thomas, *Causal Scrubbing*](https://www.lesswrong.com/posts/JvZhhzycHu2Yd57RN/causal-scrubbing-a-method-for-rigorously-testing) (Redwood Research; read via the [GreaterWrong mirror](https://www.greaterwrong.com/posts/JvZhhzycHu2Yd57RN/causal-scrubbing-a-method-for-rigorously-testing)) **[full text of the post]** | Paren balance checker; induction on filtered OpenWebText | 2-layer attention-only transformer; a paren-balancer | **Resampling ablation** on a treeified model, under a hypothesis `h = (G, I, c)` where `c` is an injective graph homomorphism from the interpretation graph `I` into the model graph `G` | Every activation the hypothesis says is unimportant, plus activations from semantically equivalent inputs | "Performance recovered": the fraction of the model's loss recovered, normalized between random-label and original performance | The interpretation graph is a *human-proposed hypothesis*, not a ground-truth task computation | The method is the control; it is deliberately conservative | **The closest prior art for "does the model's computation respect a stated graph" — and it answers a different question.** Scrubbing asks how much of the loss survives ablating everything a hypothesis calls irrelevant. We ask whether one site's state determines one digit. Do not describe our method as scrubbing |

### 1d. Multi-hop, latent vs verbalized reasoning

| Paper | Task | Model | Intervention type | Unit patched | Semantic outcome measured | Graph ground truth | Controls | What remains different here |
|---|---|---|---|---|---|---|---|---|
| [Biran, Gottesman, Yang, Geva, Globerson, *Hopping Too Late*](https://arxiv.org/abs/2406.12775) (EMNLP 2024) **[abstract]** | Two-hop factual queries ("the spouse of the performer of Imagine") | Pretrained LLMs | **Back-patching**: patch a hidden representation from a later layer back into an earlier layer | Hidden representation at the last position | Whether the correct final answer is generated; up to 66% of previously incorrect cases fixed | No — the two-hop structure is known but no counterfactual answer is computed | Layer sweep across the back-patch source/target | **The paper a reviewer will cite as "a depth result already exists".** Its depth is *layers*, ours is *written steps*, and its intervention direction is within one forward pass rather than donor-to-recipient. The bridge entity is latent, never written |
| [Yang, Gribovskaya, Kassner, Geva, Riedel, *Do Large Language Models Latently Perform Multi-Hop Reasoning?*](https://arxiv.org/abs/2402.16837) (ACL 2024) **[listing]** | Two-hop prompts with a latent bridge entity | Pretrained LLMs of several scales | Entity-recall and consistency measurements, not patching | — (representation-level analyses) | Evidence of a latent reasoning pathway, present for >80% of prompts of some relation types | No | Relation-type stratification, scale sweep | No intervention on a residual state; the intermediate is latent by design, which is the format our §6b G1 gate is trying to reach and cannot reach on this checkpoint |
| [Li, Jiang, Xie, Song, Lian, Wei, *Understanding and Patching Compositional Reasoning in LLMs* (CREME)](https://arxiv.org/abs/2402.14328) **[listing]** | Two-hop compositional queries | Pretrained LLMs | Logit lens plus intervention; then a weight edit of MHSA modules | Implicit reasoning result in middle layers; MHSA modules | Whether the composed answer is produced | No | Localization by layer/module | Their target is *repair*; ours is measurement. Still, "the implicit intermediate causally shapes the explicit answer" is theirs |
| [Ghandeharioun et al., *Patchscopes*](https://arxiv.org/abs/2401.06102) ([HTML v3](https://arxiv.org/html/2401.06102v3)) **[full text of the multi-hop section]** | Two-hop queries where ω₁ = σ₂ | Pretrained LLMs | Patch a representation into a *different, constructed* inspection prompt | Representation of the intermediate answer ω₁, patched into the position where σ₂ would be | Whether ω₂ appears in the generation: vanilla 19.57%, CoT 35.71%, Patchscope 50% | No | Vanilla and CoT baselines | **They patch an intermediate value and the model completes the next hop from it** — so "the downstream computation uses the patched intermediate" is theirs. But the outcome is accuracy improvement, not a three-way clean/implied/raw readout, and they do not test copying |
| [Khandelwal & Pavlick, *How Do Language Models Compose Functions?*](https://arxiv.org/abs/2510.01685) **[listing]** | Two-hop factual recall as g(f(x)) | Pretrained LLMs | Logit lens on residual activations | — | Presence/absence of a detectable intermediate signature | No | Contrast between compositional and direct mechanisms | Establishes that a compositional route and a shortcut route coexist; no patched-value transformation test |
| [Wang, Yue, Su, Sun, *Grokked Transformers are Implicit Reasoners*](https://arxiv.org/abs/2405.15071) (NeurIPS 2024) **[listing]** | Composition and comparison over synthetic facts | Transformers trained from scratch; plus GPT-4-Turbo/Gemini-1.5-Pro comparisons | Training-dynamics and mechanistic analysis | — | Systematic OOD generalization for comparison, failure for composition | The synthetic fact set is generated | Held-out OOD splits | Owns "composition has a structural limit in transformers" as a training-and-generalization claim, not an intervention claim |
| [Liang & Pan, *Do Latent-CoT Models Think Step-by-Step?*](https://arxiv.org/abs/2602.00449) (Feb 2026) **[listing]** | Strictly sequential polynomial-iteration tasks, two- and three-hop | CODI (continuous-thought distillation) | Logit lens, linear probes, attention analysis, activation patching | Latent-thought positions | Whether the bridge states are decodable and routed to the readout | The hop chain is known by construction | Hop-length sweep (2, 3, longer) | **Closest to our ladder in shape**: they report that at longer hop lengths the model does not execute a full latent rollout and concentrates on late intermediates. That is a step-count-dependent finding — but on a *latent* CoT model, with decodability rather than a three-way counterfactual outcome |
| [Cywiński, Bussmann, Conmy, Engels, Nanda, Rajamanoharan, *Can we interpret latent reasoning using current mechanistic interpretability techniques?*](https://www.alignmentforum.org/posts/YGAimivLxycZcqRFR/can-we-interpret-latent-reasoning-using-current) (22 Dec 2025) **[full text of the post]** | A three-step word problem (X+Y, ×Z, sum) | CODI, self-distilled from Llama-3.2-1B-Instruct | Overwrite the latent vector at a position with vectors averaged from other prompts | Latent thought vectors | Whether the model outputs "the expected answer" under same-intermediate vs different-intermediate vs random patches | The three-step chain is known by construction | **Same-intermediate patch as a null** — structurally our null-flip gate | The null design is theirs. They do not report a step-distance analysis, and they do not separate the implied answer from a copy of the patched value |

### 1e. CoT faithfulness and written-intermediate reliance — where D3/D4 live

| Paper | Task | Model | Intervention type | Unit patched | Semantic outcome measured | Graph ground truth | Controls | What remains different here |
|---|---|---|---|---|---|---|---|---|
| [Lanham et al., *Measuring Faithfulness in Chain-of-Thought Reasoning*](https://arxiv.org/abs/2307.13702) **[abstract]** | Multiple reasoning benchmarks | Anthropic models across a size range | **Text** interventions only: early answering, adding mistakes, paraphrasing, filler tokens | Written CoT tokens | Whether the final answer changes | No | Multiple perturbation types compared; size sweep | The canonical "does the model use its CoT" paper, and it is entirely text-side. It does not hold token count, position and notation fixed, and it has no decoy arm separating "notation unreadable" from "value missing" |
| [Turpin, Michael, Perez, Bowman, *Language Models Don't Always Say What They Think*](https://arxiv.org/abs/2305.04388) (NeurIPS 2023) **[listing]** | Multiple-choice reasoning with biasing features | GPT-3.5/Claude-class models | Input biasing (e.g. answer always "(A)") | Prompt features | Whether the CoT mentions the true cause of the answer | No | Bias/no-bias contrast | Establishes CoT can misdescribe the cause. Different failure mode from ours: we ask whether the written value is *read*, not whether the prose is honest |
| [Pfau, Merrill, Bowman, *Let's Think Dot by Dot*](https://arxiv.org/abs/2404.15758) (COLM 2024) **[listing]** | Two hard algorithmic tasks | Transformers trained with dense supervision | Replace CoT with meaningless filler tokens | Written tokens (replaced with "…") | Task accuracy | No | Filler vs no-intermediate-token vs full CoT | The reason "the written token matters" cannot be assumed: extra tokens buy computation independent of their content. Our omission arm holds token count fixed precisely because of this result |
| [Brauer, Verdun, Marks, *Reading Between the Dots*](https://arxiv.org/abs/2607.03502) (Jul 2026) **[abstract]** | Fact retrieval, parallel numeric composition, string manipulation, in-context computation | DeepSeek V3, Kimi K2 | Unsupervised decoding of hidden states; **KV-cache transplants at filler positions** | KV entries at filler token positions | Whether outputs swap between examples | No | Unsupervised pipeline reported at 80–95% recovery of intermediate values | A transplant at a filler position causally swaps outputs — the filler-token analogue of our chain edit. No step ladder, no copy competitor, and the intermediate is unwritten by construction |
| [Sprague et al., *To CoT or not to CoT?*](https://arxiv.org/abs/2409.12183) **[listing]** | 20 datasets, meta-analysis over 100+ papers | 14 models | Prompt-level ablation of CoT | — | Accuracy delta from CoT | No | Meta-analysis | Background for "why arithmetic is the right place to look", not a competitor |
| [Zhang, Lin, Rajmohan, Zhang, *From Reasoning to Answer*](https://arxiv.org/abs/2509.23676) (Sep 2025) **[abstract]** | Diverse domains | **Three distilled DeepSeek-R1 models** | Activation patching to test dependence of answer tokens on reasoning activations | Reasoning-token activations | Whether the final answer changes | No | Attention-head tracking alongside patching | **Same model family as ours.** Establishes "perturbing key reasoning tokens reliably alters final answers" — so that sentence is not available to us as a discovery. No graph, no counterfactual answer, no step ladder |
| [Mehrafarin, Parekh, Konstas, *When Chain-of-Thought Fails, the Solution Hides in the Hidden States*](https://arxiv.org/abs/2604.23351) (Apr 2026) **[abstract]** | GSM8K | Several pretrained LLMs | Activation patching: transfer token-level hidden states from a CoT run into a **direct-answer** run for the same question | Token-level hidden states, swept across tokens and layers | Final-answer accuracy | No | Correct vs incorrect CoT runs; token-type stratification (verbs/entities vs mathematical tokens) | Cross-condition transplant to *improve* accuracy, not a counterfactual donor to *redirect* the answer. No implied/raw separation. Their finding that mathematical tokens "encode answer-proximal content that rarely succeeds" is a nearby but distinct observation about where task information sits |
| [Zhao, Koishekenov, Yang, Murray, Cancedda, *Verifying Chain-of-Thought Reasoning via Its Computational Graph* (CRV)](https://arxiv.org/abs/2510.09312) (ICLR 2026 oral) **[abstract]** | Multi-domain CoT | Pretrained LLMs with a transcoder replacement model | Attribution graphs per CoT step; targeted feature interventions | Transcoder features | Whether a reasoning step is correct; whether an intervention corrects a faulty step | The attribution graph is recovered, not given | Domain stratification; causal check on the structural signature | Owns "treat a CoT step's attribution graph as an execution trace". Reinforces that "recover the computation graph" is unavailable to us |
| [Garcia, *The Last Word Often Wins*](https://arxiv.org/abs/2605.10799) — see §1a | | | | | | | | |

### 1f. Synthetic arithmetic, symbolic tasks, and value binding

| Paper | Task | Model | Intervention type | Unit patched | Semantic outcome measured | Graph ground truth | Controls | What remains different here |
|---|---|---|---|---|---|---|---|---|
| [Ye, Xu, Li, Allen-Zhu, *Physics of Language Models: Part 2.1, Grade-School Math and the Hidden Reasoning Process*](https://arxiv.org/abs/2407.20311) (ICLR 2025) **[abstract only — see §4]** | iGSM: generated grade-school math problems | GPT2-style models pretrained on iGSM | Probing of the "hidden (mental) reasoning process" | — | Whether the model knows which parameters are necessary before generating them | **Yes — iGSM is generated from a dependency graph** ([generator code](https://github.com/facebookresearch/iGSM)) | Controlled generation of problem structure | **The row that kills "a generated dependency graph is our novelty".** I could not open the methods, so I do not claim what interventions they do or do not run — but the generated-graph-as-ground-truth move is unambiguously theirs, and the model is trained on the task |
| [Nikankin, Reusch, Mueller, Belinkov, *Arithmetic Without Algorithms*](https://arxiv.org/abs/2410.21272) (ICLR 2025) **[listing]** | Basic arithmetic | Several pretrained LLMs | Causal circuit analysis; neuron-level ablation | Neurons and circuit components | Arithmetic answer correctness | No | Circuit faithfulness checks | Says LLM arithmetic is a bag of heuristics rather than an algorithm — relevant background for why "one remaining affine step applied to a patched value" (D9) is a nontrivial observation, and a reason not to over-read it |
| [Feng & Steinhardt, *How do Language Models Bind Entities in Context?*](https://arxiv.org/abs/2310.17191) (ICLR 2024) **[listing]** | Binding attributes to entities in context | Pythia and LLaMA families | Causal interventions on activations | Binding-ID vectors attached to entity and attribute representations | Whether the model reports the correct attribute for an entity | The binding structure is known by construction of the prompt | Cross-model replication; subspace structure analysis | Owns "value bound to a name" at the representational level. It binds an attribute; we bind a computed digit that must then be transformed |
| [Prakash, Shaham, Haklay, Belinkov, Bau, *Fine-Tuning Enhances Existing Mechanisms: A Case Study on Entity Tracking*](https://arxiv.org/abs/2402.14811) (ICLR 2024) **[listing]** | Entity tracking (boxes task) | Llama-7B and math-fine-tuned variants | DCM; CMAP (cross-model activation patching) | Attention-head outputs across models | Whether the tracked value is reported correctly | Box/value assignment known by construction | Original vs fine-tuned circuit comparison | Tracks a *stated* value; no arithmetic transformation of a patched value, no step ladder |
| [Prakash et al., *Language Models use Lookbacks to Track Beliefs*](https://arxiv.org/abs/2505.14685) **[listing]** | Belief/state tracking | Pretrained LLMs | Causal interventions on ordering-ID subspaces | Low-rank subspaces of state-token residual streams | Whether the correct state is reported | Triples known by construction | Subspace ablations | The "lookback" mechanism — retrieve information when needed — is the mechanism-level statement nearest to our D3. It is about pointers to written content, which is arguably what our omission arm probes; cite it rather than describe the phenomenon as new |
| [Sharma, Dawes, Raval, *Dissociating Decodability and Causal Use in Bracket-Sequence Transformers*](https://arxiv.org/abs/2604.22128) (Apr 2026) **[abstract]** | Dyck languages | Transformers trained from scratch | Attention masking; residual-subspace ablation | Attention to the true stack top; residual subspaces | Long-distance bracket accuracy | Stack structure known by construction | Decodability vs causal-use contrast | **The general form of our D5**: a signal can be present and decodable without being causally used. Cite it when reporting that the depth-2 patch reaches the read position (median TV 0.0877) and does nothing |

---

## 2. Per-family reading — what each family takes, what it leaves

### 2.1 Activation patching / causal tracing

**Takes:** the method, the vocabulary, and any claim of methodological novelty about
patching itself. Causal tracing ([Meng et al.](https://arxiv.org/abs/2202.05262)),
path patching ([Goldowsky-Dill et al.](https://arxiv.org/abs/2304.05969);
[Wang et al.](https://arxiv.org/abs/2211.00593)), causal mediation
([Vig et al.](https://arxiv.org/abs/2004.12265)) and attribution patching
([Syed et al.](https://arxiv.org/abs/2310.10348)) together mean the manuscript
cannot present residual-stream patching, token-position sweeps, layer sweeps or
null controls as contributions.

**Leaves:** two specific methodological points that this family has *not* made.

The first is the propagate-vs-copy gap. [Zhang & Nanda](https://arxiv.org/html/2309.16042)
is the field's metric-hygiene paper, and its warnings are about probability
saturating, negative components being missed, and Gaussian noise pushing the model
off distribution. Nowhere does it — or Heimersheim & Nanda's tutorial — raise the
case where the counterfactual value and the literal token at the patched position
are both plausible readings of "the answer moved toward the donor". In a
single-hop factual-recall setting the two coincide, which is why the gap has
stayed invisible; in an arithmetic chain they come apart, and `v3_distinct` was
built to force them apart. **This is the methods point to address to this
family**, with Zhang & Nanda cited as the nearest prior treatment of metric
choice, not as the paper that already said it. Note the scope limit: the gap
bites only where the patched position carries a value the readout could emit
verbatim, which is false across most of this literature. And it is a point about
granularity rather than priority — [Shih et al.](https://arxiv.org/abs/2606.29522)
run a computation-versus-copy control, just not one that scores a literal written
token as a competing outcome.

The second is the fixed-count quorum point (§3.10). It is a statistics
observation, not a literature-novel one, and it should be reported as an
experience report rather than a result.

Two things we must borrow rather than invent. Our donor is a symmetric-token-
replacement-style counterfactual — a different item value, in distribution — which
is exactly what Zhang & Nanda recommend; say so. And our primary registered
outcome (implied digit uniquely on top) is an argmax criterion, which is the class
of metric they warn about. The `delta_toward − delta_toward_raw` log-odds margin
*is* a logit-difference metric, and the paper should lead with the fact that both
are reported, pre-empting the obvious reviewer question.

### 2.2 Causal abstraction / interchange interventions

This is the family the review feared, and the fear was justified — but it takes a
different thing than expected. It does not take the step boundary. It takes
**"we use a known computation as the reference for our interventions"**, which is
the defining move of [Geiger et al. 2021](https://arxiv.org/abs/2106.02997) and
has been carried onto pretrained instruction-tuned models by
[Boundless DAS on Alpaca 7B](https://arxiv.org/abs/2305.08809) with an
interchange-intervention accuracy of about 94%. It also takes the outcome
statistic: our "implied digit uniquely on top at layer 13, among items whose clean
answer was alone on top" *is* an interchange-intervention accuracy for a single
aligned variable, and calling it that is both accurate and strategically better
than inventing a name.

[Causal scrubbing](https://www.greaterwrong.com/posts/JvZhhzycHu2Yd57RN/causal-scrubbing-a-method-for-rigorously-testing)
is the closest prior art for "does the model's computation respect the stated
graph", as §5 of the strategy doc guessed — and reading it closes rather than
opens the threat. Scrubbing takes a hypothesis `h = (G, I, c)` with `c` an
injective homomorphism from an interpretation graph into the model graph,
resamples every activation the hypothesis calls unimportant, and reports
*performance recovered* (72% for the paren balancer, 86% for the refined induction
hypothesis). That is an aggregate loss-recovery number over a whole hypothesis. We
report a per-item digit outcome at two named sites. Different question, different
statistic; the danger is purely terminological, and the wording rules in the
[claim-boundary note](2026-08-16-dag-literature-and-claim-boundary.md) handle it.

What this family leaves: it has no analogue of the copy competitor, and — because
its causal models are small (two boolean variables for Alpaca, a natural-logic
tree for MQNLI) — it has no ladder in the number of *written* steps between the
intervened variable and the readout.

### 2.3 Multi-hop and latent-vs-verbalized reasoning

The commissioning question here was: **who has already patched an intermediate
value and measured whether the downstream computation transforms it or copies it?**

The honest answer, corrected after reading [Shih et
al.](https://arxiv.org/abs/2606.29522): several groups have done the first half,
and one has done the second in a different form. Shih et al.'s
conflicting-continuation control asks whether the model recomputes from the
edited state or carries over another context's continuation, and answers it
directly (80%/90% follow the current move). What none of them does is score a
literal token written at the patched position as a competing outcome alongside
the transformed value — which is a difference in granularity and in what the
task makes possible, not a gap in the idea of controlling for copying.

- [Patchscopes](https://arxiv.org/html/2401.06102v3) patches the representation of
  ω₁ into the σ₂ position and the model completes the second hop, raising accuracy
  from 19.57% (vanilla) and 35.71% (CoT) to 50%. The model plainly does something
  with the patched value beyond emitting it — but the reported outcome is accuracy,
  and no copy competitor is measured.
- [Anthropic's biology paper](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)
  swaps Texas features for California features and gets "Sacramento", not
  "California". That is the cleanest published demonstration that a patched
  intermediate is *transformed*. It is also the reason our D9 cannot be written as
  a first. The escape is narrow but real: their intermediate is latent, so no
  literal token at the patched position could have been copied, and the control is
  never needed.
- [Cywiński et al.](https://www.alignmentforum.org/posts/YGAimivLxycZcqRFR/can-we-interpret-latent-reasoning-using-current)
  patch latent thought vectors in CODI on a three-step word problem, with a
  same-intermediate patch as a null — structurally our null-flip gate, published
  eight months earlier on a different model class.

On the boundary itself, the family's depth results are about **layers**, not steps.
[Biran et al.](https://arxiv.org/abs/2406.12775) find the bridge entity resolved
early and the second hop executed late, and fix up to 66% of failures by
back-patching across layers. [Liang & Pan](https://arxiv.org/abs/2602.00449) come
closest to a step-count result — CODI stops executing a full latent rollout at
longer hop lengths and concentrates on late intermediates — but on a latent-CoT
model with decodability as the outcome.

**I searched deliberately for a written-step boundary in patching and did not find
one.** Search terms tried: "patching depth", "step boundary activation patching",
"intermediate result stored in token position", "counterfactual answer arithmetic
patching", "carried vs copied patched value", "one step ancestor patching no
effect", plus family-specific variants. Nothing returned a paper that varies the
number of written steps between a patched site and the answer while controlling
token distance. That absence is the single most valuable thing in this review, and
it should be stated in the paper as an absence, in the same style
`RELATED_WORK.md` uses for the RMD thread.

### 2.4 CoT faithfulness and written-intermediate reliance (D3/D4)

This family takes the *question* and leaves the *control*.

"Does the model use its written CoT" is [Lanham et al.](https://arxiv.org/abs/2307.13702)'s
paper, with early answering, added mistakes, paraphrasing and filler tokens.
"CoT can misdescribe the real cause" is [Turpin et al.](https://arxiv.org/abs/2305.04388).
"Extra tokens buy computation independent of their content" is
[Pfau et al.](https://arxiv.org/abs/2404.15758) — and this is the result that makes
our omission arm's fixed-token-count design mandatory rather than fastidious.
"Perturbing key reasoning tokens reliably alters final answers in distilled
DeepSeek-R1 models" is [Zhang et al.](https://arxiv.org/abs/2509.23676), on our own
model family.

What is left is the *pairing*: an omission that holds token count, position and
notation fixed, plus a decoy arm that separates "the notation became unreadable"
from "the value is missing" (D4: `--omit decoy` arms indistinguishable from written
arms, 5/5 clean at p(target) 0.997/0.999). I found no paper running that pair. But
n = 5 is n = 5, and this leg cannot carry weight it does not have.

The nearest precedent in this family — missed on the first pass and added here —
is [Shih, Winnicki & Darve](https://arxiv.org/abs/2606.29522). They edit the
internal representation of a *written* state while holding the visible scratchpad
text fixed, with a known transition rule supplying the single correct downstream
consequence, and they run both a move-swap control and a "computed, not copied"
conflicting-continuation control. This is the closest published design to ours
and it forecloses the three strongest priority claims available here (first
causal intervention on written state, first exact scratchpad counterfactual,
first computation-versus-copy control). Two differences are real and should be
stated without inflation. Their positive result requires task-specific
running-state supervision — the pretrained and final-answer-only controls stay
near baseline — whereas ours is on a pretrained reasoning-tuned checkpoint with
no such training. And their state is a single order-sensitive bit edited in a
rank-16 subspace with the printed token fixed, so no written digit exists to be
copied and no arithmetic is applied to the edited value; ours transplants a whole
residual state at its native position and requires the recipient to apply its own
remaining arithmetic. The persistence contrast (their register survives several
updates, our exact control stops after one) is consistent with a supervision
explanation but does not establish one: task, intervention, and outcome measure
differ too.

The threat in this family is [Garcia's format confound](https://arxiv.org/abs/2605.10799):
corruption sensitivity tracking the location of explicit answer text rather than
computational depth, with suffix sensitivity collapsing ~19× when the terminal
answer line is removed. Our design mitigates it — the token-distance banding
(one step at 46–60 tokens still lands 86%; two steps at 16–30 tokens lands 0%) and
the within-item pairing show the boundary is not explained by placement alone.
They do not dispose of it: step count and site role remain bundled, the arms are
not an exhaustive sweep over placements, and Garcia's setting is text corruption
rather than activation patching, so treating his confound and ours as the same
variable is an assumption the manuscript makes rather than tests. Cite the paper,
state the mitigation, and state the residual — "you measured answer placement" is
the review comment this design most invites, and a claim of having settled it
invites it twice.

The second point of pressure is on the framing, not the result:
[Kudo et al.](https://aclanthology.org/2026.findings-eacl.59/)
title their paper *LLMs Faithfully and Iteratively Compute Answers During CoT* and
report that sub-answers are computed on the fly during generation. Our D3 says the
model reads the written intermediate rather than recomputing it. These are not
formally contradictory — theirs is about *when* a value is first available, ours is
about *what determines the answer once the value is on the page* — but a reviewer
will read them as contradictory unless the paper distinguishes them in as many
words.

### 2.5 Synthetic arithmetic, symbolic tasks, and binding

[iGSM](https://arxiv.org/abs/2407.20311) settles the "generated dependency graph"
question against us. [Brinkmann et al.](https://arxiv.org/abs/2402.11917) settle
"intermediate results are stored in selected token positions" against us — that is
their abstract's own sentence, and their register-token patching after layer 4 is
the experiment behind it. [Feng & Steinhardt](https://arxiv.org/abs/2310.17191) and
[Prakash et al.](https://arxiv.org/abs/2402.14811) settle "a value is bound to a
name and later retrieved". [Prakash et al.'s lookback work](https://arxiv.org/abs/2505.14685)
gives the mechanism vocabulary — retrieve information when it becomes necessary —
that D3 is an instance of.

What this family leaves is thin but real: none of it patches a *donor's* residual
state into a recipient trace and asks whether the recipient applies its own
remaining arithmetic to the donor's value. [Nikankin et al.](https://arxiv.org/abs/2410.21272)'s
"bag of heuristics" finding is worth citing next to D9 in both directions: it makes
"the readout applied one affine step to a patched digit" a non-obvious observation,
and it is also a reason to state D9 as a behavioural fact about a readout rather
than as evidence of an arithmetic algorithm.

[Sharma et al.](https://arxiv.org/abs/2604.22128) give D5 its general form and its
citation: decodable does not mean causally used.

### 2.6 Attribution graphs and circuit tracing

[Circuit Tracing](https://transformer-circuits.pub/2025/attribution-graphs/methods.html)
and [On the Biology of a Large Language Model](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)
(Anthropic, 27 Mar 2025), together with [CRV](https://arxiv.org/abs/2510.09312),
are why "recover the computation graph" is not available at any strength. They
recover graph-shaped internal computation from a production model, validate it by
feature intervention, and — in the multi-step reasoning section — demonstrate the
Dallas→Texas→Austin chain by swapping the intermediate and reading the transformed
output.

This family also supplies the negative-space argument the paper should make
instead: attribution graphs are recovered *per prompt* from features, whereas our
graph is *given* per item and our intervention is at a residual state at a named
token position. Those are complementary, and saying so is a better positioning
sentence than any claim of priority.

---

## 3. What each source rules out

One claim per entry, with the work that forecloses it and why. These are findings
about the literature, not drafting rules — the enforceable wording lives in the
[claim-boundary note](2026-08-16-dag-literature-and-claim-boundary.md), which
carries a condensed version of this list.

1. **"We recover the model's causal DAG"** / "we reconstruct the computation
   graph" — [Circuit Tracing](https://transformer-circuits.pub/2025/attribution-graphs/methods.html)
   and [CRV](https://arxiv.org/abs/2510.09312). Already ruled out by §1 of the
   strategy doc; the citation makes it enforceable.
2. **"The first interpretability study to use a known ground-truth computation
   graph as the reference for interventions"** —
   [Geiger et al. 2021](https://arxiv.org/abs/2106.02997) (MQNLI natural-logic
   causal model), [Boundless DAS](https://arxiv.org/abs/2305.08809) (Alpaca 7B).
3. **"The first to generate a dependency graph and derive exact counterfactual
   answers from it"** — [iGSM](https://arxiv.org/abs/2407.20311);
   [Kudo et al.](https://aclanthology.org/2026.findings-eacl.59/) construct the
   counterfactual by editing one equation term and score against the counterfactual
   gold answer.
4. **"The first activation-patching study on synthetic multi-step arithmetic in a
   pretrained (non-toy) LLM"** — [Kudo et al.](https://aclanthology.org/2026.findings-eacl.59/),
   ten pretrained models.
5. **"We show that intermediate results are stored at token positions"** —
   [Brinkmann et al.](https://arxiv.org/abs/2402.11917), stated in their abstract.
6. **"The first demonstration that a patched intermediate value is transformed by
   the downstream computation rather than emitted"** — Anthropic's
   [Texas→California→Sacramento swap](https://transformer-circuits.pub/2025/attribution-graphs/biology.html);
   also [Patchscopes](https://arxiv.org/html/2401.06102v3)'s multi-hop patch.
7. **"The first control separating computation from copying of a patched
   value"** — [Shih, Winnicki & Darve](https://arxiv.org/abs/2606.29522), whose
   conflicting-continuation test is exactly this control in a different form.
   What survives is narrower: they never score a literal written token as a
   competing outcome, because their edit leaves the printed token fixed and
   their state is a single bit. Claim granularity, not the control.
8. **"The first to patch inside a chain of thought and measure the final answer"** —
   [Mehrafarin et al.](https://arxiv.org/abs/2604.23351);
   [Zhang et al.](https://arxiv.org/abs/2509.23676) on distilled DeepSeek-R1 models.
9. **"We show the model relies on its written chain of thought"** —
   [Lanham et al.](https://arxiv.org/abs/2307.13702);
   [Zhang et al.](https://arxiv.org/abs/2509.23676).
10. **"The first depth limit on multi-hop interventions"** —
   [Biran et al.](https://arxiv.org/abs/2406.12775) (layer depth);
   [Brinkmann et al.](https://arxiv.org/abs/2402.11917) (a depth-bounded mechanism,
   bounded by layer count); [Liang & Pan](https://arxiv.org/abs/2602.00449) (hop
   length in latent CoT).
11. **"The depth result is about graph depth"** — retracted internally on
    2026-08-15 (§3.3), and independently exposed by
    [Garcia's format confound](https://arxiv.org/abs/2605.10799).
12. **"We use causal scrubbing / path patching"** — we use neither.
    [Causal scrubbing](https://www.greaterwrong.com/posts/JvZhhzycHu2Yd57RN/causal-scrubbing-a-method-for-rigorously-testing)
    is resampling ablation under a hypothesis `(G, I, c)` scored by performance
    recovered; [path patching](https://arxiv.org/abs/2304.05969) restricts the
    intervention to specified paths. We overwrite a residual state and let every
    downstream path see it.
13. **"We give a causal abstraction of the task"** / "the model implements the
    DAG" — [Geiger et al.](https://arxiv.org/abs/2301.04709). Abstraction requires
    an alignment for every variable and interchange accuracy across the full
    intervention set. We test one variable at a time, and at ≥2 steps we get zero.
14. **"We introduce the null-patch control"** —
    [Cywiński et al.](https://www.alignmentforum.org/posts/YGAimivLxycZcqRFR/can-we-interpret-latent-reasoning-using-current)
    use a same-intermediate patch for exactly this purpose.
15. **"A signal that is present but not used is a new phenomenon"** —
    [Sharma et al.](https://arxiv.org/abs/2604.22128).
16. **"Every patching paper that reports a directional statistic is wrong"** — the
    strategy doc's §4 bullet is right that they are *exposed*, but the exposure
    is conditional on a written competitor value existing at the patched
    position, which is false for most of this literature (factual recall, IOI,
    binding). Write "exposed
    whenever the patched position carries a value the readout could emit
    verbatim", not "exposed".

---

## 4. Verification status and open questions

- **OpenReview was unreachable.** `https://openreview.net/forum?id=RmuXDtjDhG`
  returned a browser-verification page on every attempt, so I could not confirm
  that this record is the Geiger et al. causal-abstraction paper. I substituted
  [arXiv:2301.04709](https://arxiv.org/abs/2301.04709), whose title, author list
  and abstract match the description in the commissioning brief. **The OpenReview
  identity is unconfirmed.**
- **LessWrong's own page truncated.** The direct
  [LessWrong URL](https://www.lesswrong.com/posts/JvZhhzycHu2Yd57RN/causal-scrubbing-a-method-for-rigorously-testing)
  returned navigation chrome and a truncation marker. I read the post body through
  the [GreaterWrong mirror](https://www.greaterwrong.com/posts/JvZhhzycHu2Yd57RN/causal-scrubbing-a-method-for-rigorously-testing)
  of the same post. This is a mirror of the primary post, not a secondary summary,
  but it is worth re-checking the canonical URL before citing exact figures (72%
  paren balancer, 86% induction).
- **The ACL Anthology PDF for Brinkmann et al. would not render** (returned raw
  compressed PDF streams). I read [arXiv HTML v3](https://arxiv.org/html/2402.11917v3)
  instead. Section numbers quoted above are from the arXiv version and may differ
  from the anthology version.
- **[iGSM / Physics of Language Models Part 2.1](https://arxiv.org/abs/2407.20311)
  is abstract-only here.** Neither the arXiv abstract page nor the
  [project landing page](https://physics.allen-zhu.com/part-2-grade-school-math/part-2-1)
  yielded methods detail, and the full text sits behind SSRN. I therefore state
  only what is unambiguous — that iGSM is generated and the model is trained on it
  — and make no claim about whether they run interventions. **Read the SSRN PDF
  before citing anything more specific.**
- **Rows marked [listing]** (Turpin, Pfau, Nikankin, Feng & Steinhardt, Prakash ×2,
  Yang et al., Li et al./CREME, Khandelwal & Pavlick, Wang et al. grokking,
  Sprague et al.) were resolved to a real arXiv/anthology record with matching
  title, authors and venue through a search index, but I did not open the papers
  themselves in this pass. Their matrix cells are drawn from abstract-level
  descriptions and should be re-verified against full text before any of them is
  cited for a specific number.
- **The `2604.*`, `2605.*`, `2606.*`, `2607.*` and `2602.*` identifiers were each
  verified by fetching the arXiv abstract page** and finding a coherent title,
  author list and abstract: 2604.23351, 2605.10799, 2605.06480, 2604.22128,
  2607.03502, 2602.00449, 2509.23676, 2510.09312, 2510.01685. None appears
  fabricated. I dropped nothing as unresolvable. Added 2026-08-19: 2606.29522,
  verified from the arXiv abstract page and the full HTML.
- **A miss, corrected on 2026-08-19.** This pass did not find
  [Shih, Winnicki & Darve, arXiv:2606.29522](https://arxiv.org/abs/2606.29522),
  the nearest precedent in the literature. It was already recorded in
  [`2026-08-16-dag-literature-and-claim-boundary.md`](2026-08-16-dag-literature-and-claim-boundary.md),
  which this pass did not read before running. It has since been read in full
  text and folded into §1a, §2.3, §2.4 and §3. **Read the claim-boundary note
  before extending this one.** The likely cause of the miss is vocabulary: they
  say "scratchpad", "causal register" and "state tracking" where every search
  term used here said "chain of thought", "patching" and "intermediate".
- **Searches that returned nothing, as amended.** No paper found that (a) varies
  the number of *written* steps between a patched site and the readout while
  controlling token distance; (b) scores a literal token written at the patched
  position as a competing outcome alongside the transformed value; (c) patches two
  sites in the same trace and pairs them within item; (d) runs a length- and
  position-matched omission of a written intermediate with a decoy control.
  **(b) is the amended form.** As originally written — "records a copy competitor
  alongside a directional patching outcome" — it was wrong: Shih et al.'s
  conflicting-continuation control is exactly that, in the form of a contrast
  between two rule-defined branches. Absence of evidence is weak evidence here —
  (c) and (d) are the kind of control that lives in an appendix — and the Shih
  miss shows the search vocabulary was narrower than the phenomenon.
- **Not searched, and worth a pass before submission:** the state-tracking
  vocabulary that hid Shih et al. from this pass — "scratchpad", "causal
  register", "state tracking", "working memory", "transition system" — run
  against the same questions §2.3 asks, since the search terms used here were
  all chain-of-thought-and-patching terms; non-English venues; the 2026 workshop
  proceedings (NeurIPS/ICLR interpretability workshops), where a step-boundary
  result is most likely to be sitting unindexed; and the code repositories of
  Kudo et al. and Brinkmann et al., which may contain unreported step-distance
  analyses.
- **Re-check before submission:** [Garcia 2605.10799](https://arxiv.org/abs/2605.10799)
  and [Mehrafarin et al. 2604.23351](https://arxiv.org/abs/2604.23351) are recent
  single-version preprints and may be revised; and Kudo et al. now has a
  camera-ready at [Findings of EACL 2026](https://aclanthology.org/2026.findings-eacl.59/)
  whose section numbering differs from the arXiv v2 I read.
