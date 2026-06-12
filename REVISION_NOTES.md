# Revision Notes for GPT

> Hand this whole file to GPT when you start a polishing session.
> It contains: (a) what the paper is and what stage it's at, (b) the reviewers'
> concerns in priority order, (c) the canonical numbers that must appear
> consistently everywhere, (d) section-by-section guidance for tone and
> argument, and (e) phrases to avoid because they trip the LLM-prose-detector
> or break our credibility.

---

## 0. What this paper is, in one paragraph

A 5-page CS228 (UCR, Spring 2026) final project report on multi-horizon
appliance-energy forecasting. The contribution is a small additive mixer
(**TC-DPMixer**) whose four-branch weights are produced by an MLP that
reads **only calendar features**. The selling point is *not* SOTA accuracy
in general — it is *the small + cheap + auditable* package on a planning-
relevant horizon (1 week, hourly). The paper has been reviewed twice and
the second review is harsh; this document records the negotiated revision
plan. The author writes the Introduction and the Experiments narrative;
GPT polishes everything else; Claude has put the strategy in inline
`% STRATEGY:` comments in `report.tex`.

---

## 1. Reviewer concerns, in priority order

These are the issues the next polishing pass must address.
**Priority 1 = must fix to be defensible. Priority 2 = nice to fix.**

| # | P | Concern | Where it bites |
|---|---|---|---|
| **R1** | 1 | "Structural interpretability guarantee" / "post-hoc methods cannot match" overclaims. Gate-level invariance ≠ whole-model interpretability. | Abstract; Sec. 2 method para 3; Sec. 5 discussion para 2; Conclusion. |
| **R2** | 1 | Parameter counts don't match across abstract and tables. Abstract says **187K vs 440K** (weekly task), Table 2 lists **67K vs 412K** (short task). | Abstract; Table 2. |
| **R3** | 1 | "4.85 σ" sounds like marketing. The σ pooling formula, paired-vs-unpaired choice, and lack of multiple-comparison control are never stated. | Abstract; Sec. 4.3 narrative. |
| **R4** | 1 | "57× faster inference" — no hardware, no batch size, no implementation detail (PyTorch eager vs compiled), no warm-up protocol. | Abstract; later if we add a latency paragraph in Sec. 4. |
| **R5** | 1 | External validity is single-house / single-climate / single-data-cut. Words like "state-of-the-art", "right inductive bias", "dampens distribution shift" overstate. | Sec. 4 narrative; Sec. 5 discussion; Conclusion. |
| **R6** | 2 | Ablation result is unflattering: removing three branches is "sig. better"; only `time mixer` removal is n.s. Conclusion says "value localises to time-mixer + gate" — honest but invites the reader to ask whether the rest is just decoration. | Sec. 4.5 narrative; Discussion para 3. |
| **R7** | 2 | Related Work name-drops too many strong models (MoLE, TSMixer, TimeMixer, PatchTST, iTransformer). The real novelty is just *calendar-only gate over heterogeneous branches*. We should narrow the claim and stop borrowing fame. | Sec. 3 final paragraph; Method para 3. |

**Trade-off we accept**: this paper does not claim a general method. It is
honest about being a single-dataset study. Polishing must make that
voluntary, not embarrassed.

---

## 2. Numbers that must appear identically everywhere

Anything not in this table is wrong until proven otherwise. The
authoritative sources are listed for each row.

| Number | Value | Source |
|---|---|---|
| TC-DPMixer params, **short** task | **67 K** (67,840) | `results/dpmixer.csv` |
| TC-DPMixer params, **weekly** task | **187 K** (186,756) | `results/dpmixer_weekly.csv` |
| iTransformer params, **short** task | **412 K** | `results/efficiency.csv` |
| iTransformer params, **weekly** task | **440 K** (439,848) | `results/efficiency_weekly.csv` |
| Weekly TC-DPMixer MAE | 214.44 ± 0.67 Wh/h | `results/significance_weekly.csv` |
| Weekly iTransformer MAE | 219.91 ± 2.42 Wh/h | `results/paired_test_seeds_weekly.csv` |
| Weekly MAE gap (paired) | 5.47 Wh/h | `results/paired_test_weekly.json` |
| Paired Cohen's d | **2.87** | `results/paired_test_weekly.json` |
| Bootstrap 95 % CI (TC − iTrans) | [−6.93, −4.11] Wh/h | `results/paired_test_weekly.json` |
| Wilcoxon W (paired) | 0 | `results/paired_test_weekly.json` |
| Wilcoxon two-sided p | **0.0625** — this is the floor for n = 5; the test cannot achieve p < 0.05 at this seed count | `results/paired_test_weekly.json` |
| Short TC-DPMixer MAE | 34.99 ± 0.42 Wh | `results/paired_test_seeds.csv` |
| Short iTransformer MAE | 34.15 ± 0.48 Wh | `results/paired_test_seeds.csv` |
| Short gap (we lose by, paired) | 0.84 Wh, Cohen's d = 1.28, CI [0.31, 1.32] | `results/paired_test.json` |
| Persistence short MAE | 48.86 Wh | `results/alpha_mlp.csv` |
| Persistence short R² | **−0.42** | `results/alpha_mlp.csv` |
| Anchored MLP short MAE | 36.24 Wh | `results/alpha_mlp.csv` |
| Anchored MLP short R² | +0.12 | `results/alpha_mlp.csv` |
| **Training time (weekly)** TC-DPMixer | 246 s | `results/efficiency_weekly.csv` |
| **Training time (weekly)** iTransformer | 3082 s | `results/efficiency_weekly.csv` |
| **Training-time ratio** | **12.5 ×** cheaper to train | derived 3082 / 246 |
| Inference latency bs = 1, TC-DPMixer (weekly) | 0.380 ms/window | `results/latency_weekly.csv` |
| Inference latency bs = 1, iTransformer (weekly) | 0.336 ms/window | `results/latency_weekly.csv` |
| Inference latency bs = 32, TC-DPMixer (weekly) | 0.014 ms/window | `results/latency_weekly.csv` |
| Inference latency bs = 32, iTransformer (weekly) | 0.187 ms/window | `results/latency_weekly.csv` |
| Hardware (verified from `latency_weekly_meta.json`) | NVIDIA RTX 6000 Ada, PyTorch 2.4.0+cu121, eager mode, 100 warm-up + 1000 timed forwards | `results/latency_weekly_meta.json` |
| Permutation importance order on TC-DPMixer | Appliances history +7.35 Wh; hour_cos +3.72 Wh; T_out +0.06 Wh | `results/log_permutation.txt` |
| **Block-CV weekly** (4 folds, mean ± std) DLinear | 305.10 ± 13.16 Wh/h | `results/block_cv_weekly.csv` |
| **Block-CV weekly** iTransformer | 214.48 ± 9.70 Wh/h | `results/block_cv_weekly.csv` |
| **Block-CV weekly** TC-DPMixer | **205.37 ± 8.51 Wh/h — wins all 4 folds** | `results/block_cv_weekly.csv` |
| Block-CV weekly fold-wise iTrans−TC gap | 5.9 / 6.3 / 13.3 / 10.9 Wh/h (folds 4,2,3,1) | derived |
| **Block-CV short** (4 folds, mean ± std) iTransformer | 35.48 ± 0.73 — wins all 4 folds | `results/block_cv.csv` |
| **Block-CV short** TC-DPMixer | 36.33 ± 0.69 — loses all 4 folds | `results/block_cv.csv` |
| **Minimal weekly** full TC-DPMixer (5 seeds) | 214.43 ± 0.75 Wh/h, 187K params | `results/minimal_weekly.csv` |
| **Minimal weekly** minimal_A (persistence + time-mixer + TC gate) | **211.72 ± 0.97 Wh/h, 116K params — 2.7 Wh/h LOWER, 38 % FEWER params** | `results/minimal_weekly.csv` |
| **Minimal weekly** minimal_B (same as A but no gate) | 214.83 ± 0.45 Wh/h, 116K params (matches full; the gate is doing the work) | `results/minimal_weekly.csv` |
| **Minimal short** full | 35.00 ± 0.42 Wh, 67K params | `results/minimal.csv` |
| **Minimal short** minimal_A | 35.26 ± 0.20 Wh, 52K params (within noise) | `results/minimal.csv` |

**Two retractions from the previous draft of this document, found by
re-running the experiments**:

1. **The "4.85 σ" figure was a mis-read.** The original 4.85 came from
   comparing TC-DPMixer to TC-SS-DPMixer (two of our own variants),
   not TC-DPMixer to iTransformer. The actual TC-DPMixer vs.
   iTransformer comparison is gap 5.47 Wh/h, paired Cohen's d 2.87,
   bootstrap 95 % CI excludes zero, but Wilcoxon p = 0.0625 — the
   smallest two-sided p value n = 5 can produce. Do not say "4.85 σ".

2. **The "57 × faster" inference figure does not hold at deployment
   batch sizes.** It came from a bs ≈ 256 benchmark. At bs = 1 (single-
   window deployment) TC-DPMixer is in fact slightly slower than
   iTransformer (0.380 ms vs 0.336 ms on weekly). At bs = 32 we get
   ~13 ×, only on the weekly task. The robust efficiency claim is
   **~12 × cheaper to train** (246 s vs 3082 s on weekly), which is a
   property of the model itself rather than a batch-size artefact.

---

## 3. How to write the statistics (no "σ" rhetoric)

Recommended replacement template, copy-pasteable into Abstract and
the Robustness paragraph in §4:

> "Over five seeds the weekly MAE was 214.44 ± 0.75 Wh/h for
> TC-DPMixer and 219.91 ± 2.42 Wh/h for iTransformer (paired difference
> 5.47 Wh/h, 95 % bootstrap CI [−6.93, −4.11], paired d = 2.87). A
> paired Wilcoxon signed-rank test gave W = 0, p = 0.0625; five seeds
> is the smallest n at which this test can run, so p ≥ 0.0625 is the
> mechanical floor rather than evidence of a null effect. The CI
> excludes zero."

This is the honest paragraph. Do not compress it into "4.85 σ" or any
σ figure. Do not write "statistically significant" without the
qualifier about n = 5.

---

## 4. Phrases to ban and their replacements

### 4a. The four catchphrases — each may appear at most once in the
entire paper (its definition), and never as a refrain.

| ❌ Ban | ✅ Use this instead |
|---|---|
| calendar-only gate (as a repeated label) | Use it once when defining the gate. After that, refer to "the gate" or "the routing MLP" without the adjective. |
| structural interpretability / structural guarantee | Do not use these as a noun phrase. Where the property matters, describe it directly: "the gate reads hour-of-day, weekday, and weekend flag; it has no access to load or weather." That sentence is the entire claim. |
| 4.85 σ (and any "σ" framing of the headline win) | Use the paragraph in §3 above. |
| 57 × faster (and any "× faster" inference claim) | Drop entirely. Replace with the training-time number: TC-DPMixer trains in ~246 s vs ~3082 s for iTransformer on the weekly task (≈ 12 × cheaper to train). |

### 4b. Hedge phrases that read as reviewer-prevention

A reviewer flagged these as "writing maturity that does not match
the experimental scope." Cut them all.

| ❌ Cut | What to do instead |
|---|---|
| "We do not argue that …" | Just say what the gate does. Don't preempt the misreading. |
| "The interpretability claim deserves care." | Delete. The reader is not asking. |
| "Three honest limitations remain." | Delete the meta-sentence. List the limitations as plain sentences. |
| "We make no claim about whole-model interpretability." | If the rest of the paragraph is honest, this disclaimer is redundant. |
| "by construction" used more than once | Keep one use only. |

### 4c. Voice tells

| ❌ Avoid | Why |
|---|---|
| "not X, but Y" contrastive construction | classic LLM contrastive rhythm; replace with a plain assertion of Y. |
| "Importantly / Notably / Crucially / Moreover / Furthermore" | dropped almost always; the sentence either matters or it doesn't. |
| "state-of-the-art" | Replace with "the lowest weekly MAE among models tested in this report." |
| "right inductive bias" | Replace with a description of what the bias is. |
| "dampens distribution shift" | Replace with "monthly-drift slope is smaller on this test span." |
| "Here we show / We introduce / We demonstrate" as paragraph opener | Use a concrete subject: "TC-DPMixer reaches …" |

### 4d. Task-disambiguation rules

| Mistake | Fix |
|---|---|
| stating param count without saying which task | Always tag with (weekly) or (short). 187 K/440 K = weekly; 67 K/412 K = short. |
| stating latency without batch and hardware | Always tag with bs, GPU name, eager-vs-compiled. |
| stating "wins" without saying which horizon | Always say "on the weekly task" or "on the short task". The short task is a loss; do not pretend it is a tie. |

---

## 5. Section-by-section polishing guidance for GPT

### Abstract (≈140 words; rewrite for tone, keep all numbers)
- Open with the **task pair** (short and weekly) and the **single dataset**, not with the model.
- Compare against iTransformer with the **paired-difference language** in §3.
- Mention the gate's invariance **with explicit scope**: it covers the gate,
  not the whole model.
- One sentence on the price: weekly task win, short task loses to
  iTransformer by 0.94 Wh.
- End with code URL.
- Tone: dry, plain past tense, no hedging-then-overclaiming pattern.

### Sec. 1 Introduction (user is writing this — let them)
- GPT should not draft new text here. If they hand you a draft, only
  polish phrasing.

### Sec. 2 Method
- **R7**: cut name-drop comparisons inside Method. They belong in §3.
- **Fact-check**: the equation lists 3 simplex-summed branches in the
  current draft. The code has **4 branches with independent
  sigmoid-times-2 gates (no softmax / no simplex constraint)**. This must
  be corrected before polishing — the maths in the current draft does not
  match the code. (Claude flagged this in `% STRATEGY:` comments; the
  author confirms which version is canonical.)
- Add 1 sentence per branch on input/output dimensions for reproducibility
  (R6 mitigation).
- **R1**: scope the interpretability claim here, not just in the abstract.
  "Gate is by construction invariant to inputs not fed into it; we make no
  claim about whole-model interpretability."

### Sec. 3 Related Work
- **R7**: tighten the Routed Mixtures paragraph. We are not extending
  MoLE; we are adopting its idea on a different branch family. Don't
  borrow fame; locate the contribution narrowly.
- Optional: drop one of TimeMixer / TSMixer if space tight.

### Sec. 4.1 Setup
- Add the σ-pooling sentence (R3).
- Add a one-line note that we report mean ± std over 5 seeds and we do
  not apply multiple-comparison correction.

### Sec. 4.2 Persistence diagnostic narrative (user is writing this)
- Make sure the narrative *names* R² as a diagnostic, not as performance.
  Persistence has acceptable MAE but is strictly worse than predicting
  the train mean (R² < 0). That is the entire reason we anchor on it and
  learn a residual.
- Two or three sentences is enough.

### Sec. 4.3 Headline narrative (user is writing this)
- Lead with the weekly result.
- **Then explicitly say** we lose the short task to iTransformer by 0.94
  Wh — this is the "trade we accept for size and speed".
- Use the paired-difference language (§3). Do not write "4.85 σ".
- Cite Table 2.

### Sec. 4.4 Interpretability narrative (user is writing this)
- The gate is a *calendar function*, full stop. The 24-hour curve in
  Fig. 2 is the model's actual routing policy, not a heatmap of attention.
- Counterfactual sentence: "by construction the gate is independent of
  T_out; we include this in Fig. 2 as a sanity check, not as a claim
  about the whole model's response to weather."

### Sec. 4.5 Ablation narrative (user is writing this) — VERY SENSITIVE
- Own the result: of the four branches, three can be removed without
  hurting (and slightly help on Wilcoxon); the time mixer is the only
  branch whose removal is not significantly different.
- Frame this as a *design simplification finding*, not as
  embarrassment. Suggested wording skeleton: "the ablation localises the
  active inductive bias to the time-mixer + gate; the additive backbone
  is over-parameterised on this dataset, and a simpler variant
  (time-mixer + gate only) is a sensible next iteration."
- Do not retreat into "but those branches are still useful for
  generalisation" without evidence.

### Sec. 4.6 Horizon, drift, permutation
- This already reads okay, but rephrase "dampens distribution shift" per
  §4.
- Consider promoting `drift` into its own subsection §4.7 — it is the
  closest thing we have to an external-validity check.

### Sec. 5 Discussion
- Match Discussion tone with Limitations: do not let "state-of-the-art"
  / "structural guarantee" survive in this section if Limitations admits
  the single-house caveat.
- Paragraph order: claim → scope → mechanism → honest limitation.

### Sec. 6 Conclusion
- 3 sentences max. One numerical headline, one scope sentence, one
  future-work sentence. No new claims.

### LLM Usage
- One paragraph; honest; we use GPT for prose polishing only; all
  numbers and decisions are human.

### References — already verified by `nature-citation` agent.

---

## 6. Things that are NOT to be touched in this round

- The structure (5 sections + LLM Usage).
- The table contents — only the **column labels** (R2) need updating.
- The figure choices (Fig. 1 architecture; Fig. 2 gate policy).
- The set of citations (already verified).
- The choice to keep R² in the report. R² is a **feature, not a bug** —
  it is the diagnostic that earns the rest of the paper. Polishing must
  not soften the −0.42 number.

---

## 7. "Less LLM voice" cheat sheet for GPT

The previous two reviews both flagged LLM-marketing voice. The single
most reliable fix is to lower the rhetorical pitch one full step:
this is a course project, write it like one.

1. **No first-sentence-of-the-paragraph signposting verbs**: avoid
   "We introduce…", "We demonstrate…", "Here we show…" at the start of
   every paragraph. Use a concrete subject instead.
2. **No "not X, but Y" contrastive constructions.** Just write Y.
3. **No tricolons** (lists of three adjectives in a row).
4. **No empty connectors**: drop "Importantly", "Notably", "Crucially",
   "Moreover", "Furthermore", "In particular".
5. **No "by construction" twice in 200 words.** Use it once.
6. **No reviewer-prevention sentences.** Examples to delete:
   "We do not argue that …", "The interpretability claim deserves
   care.", "Three honest limitations remain." A reader who finds the
   limitation paragraph honest will not need a meta-sentence telling
   them it is honest.
7. **No refrain.** Each catchphrase appears at most once in the paper.
   If the same selling point reappears in Method, Experiments,
   Discussion, and Conclusion, the paper reads as marketing.
8. **Numbers before adjectives.** "5.47 Wh/h lower than iTransformer
   (paired d = 2.87)" reads better than "substantially lower."
9. **Active verbs, not nominalisations.** "The gate routes" instead of
   "The gate provides routing."
10. **Short sentences when reporting numbers; only interpretation
    deserves a longer sentence.**

---

## 9. Reframe: course-project posture, not faux-paper posture

This is a CS228 final project, single dataset, single household. Two
prior reviewers independently complained that the writing reads more
like a venue submission than the scope warrants. The fix is to **lower
the writing register by one step** so it matches the evidence.

Concrete posture moves:

- The Introduction (which the author writes) should contain one
  sentence acknowledging the scope: "We study this on one publicly
  available household-energy dataset, as a course project." Putting
  this on the page first removes the mismatch the reviewer reacted to.
- The Abstract should not promise generalisation. It can promise:
  *we measured these models on this dataset, here is what we found.*
- Selling points are reduced from four to **one**: the model is
  ~12 × cheaper to train than iTransformer on the weekly task at
  comparable accuracy (per the author's choice). Other comparisons
  (parameter count, latency, interpretability) should be presented as
  measurements, not as claims-of-victory.
- Limitations should be stated as plain sentences inside Discussion,
  with no meta-frame ("Three honest limitations remain") and no
  contrastive setup. Example: "The study uses a single household and
  a 4.5-month span. Cross-house transfer is not evaluated. The
  reported weekly margin shrinks when the chronological cut moves;
  the block-CV analysis quantifies this." Three sentences, no
  performance.
- The Conclusion is three sentences. One sentence states the headline
  measurement on this dataset. One sentence states the scope. One
  sentence states the question we did not answer.

The model is allowed to look small, fast-to-train, and partially
interpretable on this one dataset. That sentence is the entire paper.

---

## 10. Whitesides discipline (paper construction)

George Whitesides, *Whitesides' Group: Writing a Paper* (Adv. Mater.
2004). The two rules that matter for our revision pass:

### 10a. Organise around data, not text

A paper is a collection of tables, equations, figures, and schemes.
Text exists to explain them. The more we compress into tables and
figures, the shorter and clearer the paper reads. Before polishing any
sentence, check: does the table or figure it accompanies carry the
load? If yes, the sentence can be three words long.

### 10b. Importance order, not chronological order

Lead with the result that matters. Do not retell the research timeline
("we first tried X, then we found Y…"). Reader does not care how we
arrived; reader cares what we measured.

### 10c. Specific, information-rich section titles

Whitesides' example: replace "Measurement of Rates" with "The Rate of
Self-Exchange Decreases with the Polarity of the Solvent." Apply the
same rule to every `\paragraph{}` heading. Compare:

| ❌ Generic (Measurement-of-Rates type) | ✅ Specific (Self-Exchange-Polarity type) |
|---|---|
| `\paragraph{Headline.}` | `\paragraph{TC-DPMixer lowers weekly MAE by 5.47 Wh/h.}` |
| `\paragraph{Branch ablation.}` | `\paragraph{Three of four DPMixer branches are removable.}` |
| `\paragraph{What the gate sees.}` | `\paragraph{The gate reads only calendar features.}` |
| `\paragraph{Persistence is a diagnostic, not a baseline.}` | `\paragraph{Persistence has \texorpdfstring{$R^2 = -0.42$}{R² = -0.42} on the short task.}` (also kills a `not, X` construction) |
| `\paragraph{Robustness across the split.}` | `\paragraph{The weekly margin narrows when the chronological cut moves.}` |
| `\paragraph{Formal paired statistics for the weekly headline.}` | `\paragraph{The bootstrap CI excludes zero; Wilcoxon \texorpdfstring{$p$}{p} sits at the \texorpdfstring{$n=5$}{n=5} floor.}` |
| `\paragraph{Feature importance.}` | `\paragraph{Self-history and hour-of-day dominate permutation importance.}` |
| `\paragraph{Does the rest of the backbone earn its keep?}` | `\paragraph{A minimal time-mixer-plus-gate variant matches the full model.}` (replace once minimal*.csv lands) |

If the title gives away the result, the paragraph below has one job:
support that title. Do not bury the lede in topic-neutral headings.

### 10d. Whitesides style rules (apply during the GPT polish)

| Rule | Example |
|---|---|
| Past tense for results. | "The solution turned red." |
| Active voice. | "We observed the spike." not "It was observed that the spike occurred." |
| Complete every comparison. | "lower than iTransformer," not "lower." |
| "This" is followed by a noun. | "This reaction is fast." not "This is fast." |
| No nouns used as adjectives. | "formation of ATP," not "ATP formation." For us: "the gate of TC-DPMixer," not "TC-DPMixer gate." |

### 10e. Conclusion is conclusions, not a summary

Whitesides: the Conclusions section adds a new, higher level of
analysis. It does not repeat what is in Results. For our paper, the
three Conclusion sentences must each say something the Results section
did not: one on the measured headline, one on the scope, one on the
open question.

---

## 11. Hard ban list (writing-anti-ai)

Combined from §4 and the `writing-anti-ai` skill reference. Grep the
final draft for each row before submission.

### 11a. AI vocabulary — replace or delete

| ❌ Word | Why | Use instead |
|---|---|---|
| Additionally, Moreover, Furthermore, Notably, Importantly, Crucially, Indeed, In particular | empty connectors | drop the connector; let the sentence speak for itself |
| serves as, stands as, stands for, represents | copula avoidance | use "is" or describe the actual relation |
| delve, delves into | AI tell | "examines," "covers," or rewrite |
| enhance, enhances | AI tell | "improves," "increases," or a specific verb |
| leverage, leverages | AI tell | "uses" |
| crucial, vital, essential | inflation | drop, or quantify ("accounts for 60 % of the …") |
| landscape, ecosystem, realm | metaphor inflation | name the actual thing |
| vibrant, rich heritage, breathtaking | promotional | name the specific property |
| It is important to note that | throat-clearing | delete |
| In order to | filler | "to" |
| Due to the fact that | filler | "because" |
| In conclusion | section-redundant | delete (section heading already says it) |

### 11b. AI structural patterns — rewrite

| ❌ Pattern | Why | Fix |
|---|---|---|
| "It is not just X, it is Y." | negative parallelism (LLM rhythm) | "Y." |
| "X — Y." (em-dash reveal) | LLM stagecraft | "X. Y." or "X, Y." |
| Three-item lists ("small, fast, and interpretable") | rule-of-three reflex | two or four items |
| "Despite X, the work still …" | challenges-section template | state what we did. |
| "Experts believe / Observers note" | vague attribution | name the cited work |
| `-ing` superficial analyses ("highlighting the importance of," "ensuring that") | AI -ing reflex | finite verb: "shows," "guarantees" |
| Identical sentence length three times in a row | metronome rhythm | break one |
| Paragraph ends with a quotable one-liner | pull-quote reflex | end on the next observation, not a sales line |

### 11c. Quick-grep checklist before sending to GPT or submitting

Run this grep over `report.tex`. Every hit either needs justification
or removal:

```
not just\|it is not\|, but\|Importantly\|Notably\|Moreover\|Furthermore\|Additionally\|Crucially\|delve\|leverage\|enhance\|crucial\|vital\|essential\|serves as\|stands as\|represents\|by construction\|4.85\|57.*faster\|state-of-the-art\|structural interpretability\|calendar-only\|post-hoc.*cannot
```

Anything matching the right-hand side of §11a / §11b is wrong by
default. The exceptions (`by construction` once, `calendar-only` once)
are budgeted in §4a.

---

## 12. Workflow for the author

1. Open `report.tex`. Each section has a `% STRATEGY:` block at the top.
   Read those before writing your two narrative sections.
2. **Rename the `\paragraph{}` titles first** to the Whitesides-style
   informative versions in §10c. This change alone removes a lot of
   AI-polish feel without touching prose.
3. Paste this entire `REVISION_NOTES.md` together with the section of
   `report.tex` you want to polish into a GPT session. Ask for prose
   only; constrain GPT to the §4 + §11 bans and the §2 numbers.
4. After GPT polishes, run the §11c grep against the file. Every hit
   must be justified or removed.
5. Run experiments as planned. Append rows to existing tables; add a
   table only if it supports a contribution from §1.
