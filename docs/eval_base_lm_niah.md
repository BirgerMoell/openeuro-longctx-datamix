# Base-LM NIAH Evaluation

**Model:** `birgermoell/oellm-9b-yarn-multilingual-32k` (iter_0001000)  
**Eval:** Forced-choice log-likelihood scoring — no instruction following required  
**Job:** SLURM 18603871 · LUMI (CSC) · 2026-05-13  
**Scripts:** `scripts/eval_base_lm_niah.py` · `lumi/slurm/eval_base_lm_niah.sbatch`

---

## Why this eval

The standard NIAH benchmark asks the model to generate an answer in a specific format (e.g. `<Answer>...</Answer>`). That requires instruction tuning — a base model will simply continue the text rather than output the expected format, and will score 0% regardless of whether it can actually find the needle.

This eval removes that dependency entirely. Instead of asking the model to generate an answer, we ask: *which of these four candidates does the model assign the highest probability to?* This works directly on a base LM, before any fine-tuning.

---

## How it works

### 1. Context generation

For each trial we generate a document in the target language containing dozens of key→value "magic number" facts:

```
Voici un ensemble de faits.

Le nombre magique spécial pour « river » est : 3827461.
Le nombre magique spécial pour « forest » est : 9041823.
Le nombre magique spécial pour « castle » est : 5830182.
Le nombre magique spécial pour « candle » est : 6174029.
...   [hundreds more pairs filling the context]
Le nombre magique spécial pour « apple » est : 7319420.   ← query needle at 50% depth
...   [more pairs]
```

The **needle depth** controls where in the document the query fact is placed:
- `0%` = very beginning of the fact list
- `50%` = middle
- `100%` = end

Context length is controlled in tokens: 2 048 / 4 096 / 8 192 / 16 384 / 32 768.

### 2. Completion prefix

After the context, we append the question as a plain base-LM completion stub — no instruction format, no answer tags:

```
Le nombre magique spécial pour « apple » est :
```

The model sees this as a sentence to complete, exactly as it would any text in its training distribution.

### 3. Candidates

We present 4 candidate completions:

| Index | Value | Where it appears in context |
|---|---|---|
| ✓ true | `7319420` | attached to "apple" |
| distractor 1 | `3827461` | attached to "river" |
| distractor 2 | `9041823` | attached to "forest" |
| distractor 3 | `5830182` | attached to "castle" |

**All four values appear somewhere in the context**, attached to different keys. This is the critical design choice: the model cannot rely on having seen any of these 7-digit numbers in pre-training data, and it cannot rely on frequency priors ("which number sounds more likely?"). It must **read the context and bind the correct value to the query key**.

### 4. Scoring

For each candidate `C` we compute the sum of log-probabilities the model assigns to its tokens given the full `[context + prefix]`:

```
score(C) = Σⱼ log P(tokenⱼ | context + prefix + C[0:j])
```

This is a single forward pass through the model per candidate — no sampling, no generation, fully deterministic.

The candidate with the highest score is the model's prediction:

```
prediction = argmax_C score(C)
```

### 5. Accuracy

A trial is **correct** if `prediction == true candidate`. Accuracy is reported per cell in the `context_length × needle_depth` grid.

Random-chance baseline for a 4-way forced choice: **25%**.

---

## Eval grid

| Dimension | Values |
|---|---|
| Context lengths | 2 048 · 4 096 · 8 192 · 16 384 · 32 768 tokens |
| Needle depths | 0% · 25% · 50% · 75% · 100% |
| Languages | fr · fi · cs · nl |
| Trials per cell | 10 |
| Total scored predictions | 1 000 (main grid) + controls |

### Language templates

Each language uses its own needle template so the eval is in-distribution for the multilingual training data:

| Lang | Needle template |
|---|---|
| fr | `Le nombre magique spécial pour « {key} » est : {value}.` |
| fi | `Sanan "{key}" erityinen taikuusnumero on: {value}.` |
| cs | `Speciální magické číslo pro „{key}" je: {value}.` |
| nl | `Het speciale magische getal voor "{key}" is: {value}.` |

---

## Controls

Three controls run at the end of each language to validate the eval design:

### `no_context`
The prefix is presented **without any context document** — just the bare question:

```
Le nombre magique spécial pour « apple » est :
```

The model has no way to retrieve the correct value. Expected accuracy: ~25% (random). If this is much higher, it would mean the model has memorised the specific 7-digit values, which would invalidate the main eval. If it is ~25%, the main eval results are genuine.

### `shuffled`
All key→value bindings in the context are **rotated by one position**: the query key now maps to a different value, and every distractor key maps to its neighbour's value. The true candidate changes accordingly.

A model that reads the context should follow the rotated binding. If accuracy stays high (model ignores context and predicts by prior), that is evidence that the main-grid results are not actually from context reading.

### `short_ctx`
A tiny 256-token context (just a few needle pairs). This is the easy case: the answer is immediately accessible in a very short window. Expected accuracy: ~100%. Confirms that the scoring code works and the model can solve the task at short context before we report long-context numbers.

---

## Results

> **SLURM job 18604459 — 2026-05-13**  
> FR, FI, CS: 280/280 trials complete. NL: 229/280 (2K–16K + 32K depth=0%/25%/50% before 3 h timeout; controls not reached).

### Accuracy by language × context length (main grid, all depths averaged)

| lang |   2048 |   4096 |   8192 |  16384 |  32768 |
|------|-------:|-------:|-------:|-------:|-------:|
| fr   |   1.00 |   1.00 |   1.00 |   1.00 |   0.84 |
| fi   |   1.00 |   1.00 |   1.00 |   1.00 |   0.86 |
| cs   |   0.94 |   1.00 |   1.00 |   1.00 |   0.80 |
| nl   |   0.98 |   1.00 |   1.00 |   1.00 |  0.73† |

*† NL 32K average over 3 completed depths (0%, 25%, 50%) only.*

### Full accuracy grid — FR

| ctx \ depth |  0% | 25% | 50% | 75% | 100% |
|-------------|----:|----:|----:|----:|-----:|
|   2 048     | 1.00| 1.00| 1.00| 1.00|  1.00|
|   4 096     | 1.00| 1.00| 1.00| 1.00|  1.00|
|   8 192     | 1.00| 1.00| 1.00| 1.00|  1.00|
|  16 384     | 1.00| 1.00| 1.00| 1.00|  1.00|
|  32 768     | **0.20**| 1.00| 1.00| 1.00|  1.00|

### Full accuracy grid — FI

| ctx \ depth |  0% | 25% | 50% | 75% | 100% |
|-------------|----:|----:|----:|----:|-----:|
|   2 048     | 1.00| 1.00| 1.00| 1.00|  1.00|
|   4 096     | 1.00| 1.00| 1.00| 1.00|  1.00|
|   8 192     | 1.00| 1.00| 1.00| 1.00|  1.00|
|  16 384     | 1.00| 1.00| 1.00| 1.00|  1.00|
|  32 768     | **0.30**| 1.00| 1.00| 1.00|  1.00|

### Full accuracy grid — CS

| ctx \ depth |  0% | 25% | 50% | 75% | 100% |
|-------------|----:|----:|----:|----:|-----:|
|   2 048     | 1.00| 0.80| 0.90| 1.00|  1.00|
|   4 096     | 1.00| 1.00| 1.00| 1.00|  1.00|
|   8 192     | 1.00| 1.00| 1.00| 1.00|  1.00|
|  16 384     | 1.00| 1.00| 1.00| 1.00|  1.00|
|  32 768     | **0.00**| 1.00| 1.00| 1.00|  1.00|

### Full accuracy grid — NL (partial at 32K)

| ctx \ depth |  0% | 25% | 50% | 75% | 100% |
|-------------|----:|----:|----:|----:|-----:|
|   2 048     | 1.00| 0.90| 1.00| 1.00|  1.00|
|   4 096     | 1.00| 1.00| 1.00| 1.00|  1.00|
|   8 192     | 1.00| 1.00| 1.00| 1.00|  1.00|
|  16 384     | 1.00| 1.00| 1.00| 1.00|  1.00|
|  32 768     | **0.20**| 1.00| 1.00|  —  |   —  |

### Controls

| lang | no_context | shuffled | short_ctx |
|------|----------:|--------:|----------:|
| fr   |      0.00 |    1.00 |      1.00 |
| fi   |      0.30 |    1.00 |      1.00 |
| cs   |      0.10 |    0.90 |      1.00 |
| nl   |         — |       — |         — |

*NL controls not reached before 3 h timeout.*

---

## Interpretation

**2K–16K: near-perfect across all four languages.** Every context length from 2K to 16K achieves ≥94% accuracy (most cells 100%), at every needle depth, in all four languages. There is no "lost in the middle" effect — the model retrieves equally well whether the needle is at the start, middle, or end of the document.

**32K: one specific failure — needle at depth=0% ("lost at the beginning").** The only cells that fall below 100% at 32K are those where the needle is placed at position 0 — the very first fact in the document, farthest from the query. Accuracy at this cell:

| lang | 32K depth=0% |
|------|-------------:|
| fr   |         0.20 |
| fi   |         0.30 |
| cs   |         0.00 |
| nl   |         0.20 |

All other 32K depths (25%–100%) score 1.00 in all languages. The model fails only when the retrieval distance is at its absolute maximum (~32 768 tokens between needle and query).

**Effective reliable retrieval range: ~24K tokens.** At depth=25% with ctx=32K, the needle is ~24K tokens before the query — and retrieval is perfect. The failure threshold lies between 24K and 32K tokens of separation.

**Controls confirm genuine context reading:**
- `no_context` ≈ 0–30%: near chance, confirming the model cannot guess values from priors alone.
- `shuffled` ≈ 90–100%: model correctly follows the rotated bindings, confirming it is reading and using the context.
- `short_ctx` = 100%: scoring code is correct; the task is solvable at short range.

The FI `no_context` score of 0.30 (vs 0.25 chance) is within noise for 10 trials. The CS `shuffled` score of 0.90 suggests 1 trial in 10 where the model ignored the rotated binding — also within noise.

---

## Known limitations

- **10 trials per cell** is low for a statistical claim; accuracy estimates have ±~10% uncertainty at this sample size. A follow-up run with 50 trials per cell would give tighter bounds.
- **English keys**: The fact keys ("river", "forest", "apple", ...) are English words even though the sentence templates are in the target language. This reflects the structure of the YaRN training data but could be made fully multilingual in a future version.
- **Single fact type**: All needles are numeric key→value pairs. A more demanding version would use heterogeneous fact types (dates, names, quantities) to test more general retrieval.
- **No length normalisation on scores**: We use raw sum-of-log-probs, not per-token average. Candidates are all 7-digit integers so their token lengths are comparable (typically 7–9 tokens), making this a minor issue.

---

## Reproducing

```bash
# On LUMI — pull latest and submit
git -C /scratch/project_462000963/bmoell/openeuro-longctx-datamix pull
sbatch lumi/slurm/eval_base_lm_niah.sbatch

# All 8 training languages, skip 2K:
sbatch --export=ALL,\
LANGUAGES="bg cs da et fi fr hr nl",\
CTX_LENGTHS="4096 8192 16384 32768",\
TRIALS=20 \
lumi/slurm/eval_base_lm_niah.sbatch
```
