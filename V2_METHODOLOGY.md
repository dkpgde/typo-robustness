# Typo Robustness v2 — authoritative methodology

This document defines the protocol used for final v2 results. The exploratory
`v2_*.ipynb` notebooks are retained for inspection, but final reported results
must be produced by `run_v2_final.py` and `compare_v2_authoritative.py`.

## Research questions

**RQ1.** Can observable tokenization instability predict when coarse
tokenization will fail, allowing a compact model to adapt representation
granularity to each input?

**RQ2.** Do the same tokenization trade-offs, instability signals, and adaptive
routing gains persist when the frozen CNN backbone is replaced by one
prespecified compact TCN?

The intended contribution is adaptive representation / adaptive compute, not a
larger tokenizer or architecture benchmark.

## Frozen expert experiment

The v1 expert protocol remains unchanged:

- AG News descriptions only;
- balanced 12,000 train / 2,000 validation / 2,000 final test examples;
- nine training seeds;
- word, BPE-500, BPE-1,000, BPE-2,000, BPE-5,000 and character tokenization;
- nested typo levels from 0% to 50%;
- the same corruption generator and statistical philosophy;
- one fixed CNN and one fixed TCN configuration.

No router decision changes expert training.

## Leakage-safe router calibration

The router is **not fitted on final test examples or on duplicate copies of
those examples under different model seeds**.

The frozen v1 experiment consumes the first 3,500 AG News training rows per
class. Router calibration takes the next 500 otherwise-unused rows per class,
for 2,000 calibration texts total. Expert models are evaluated on corrupted
versions of those calibration texts, and those predictions define the router
oracle targets.

The final 2,000-example test set remains untouched until final router
evaluation.

Model seed is never a router feature. Multiple expert seeds provide repeated
estimates of expert behavior on the calibration examples, but no final-test
text appears in router fitting.

## Tokenization-instability features

Deployable router features include:

- raw text length;
- whitespace word count;
- character count;
- word OOV fraction;
- digit and punctuation fractions;
- BPE token count;
- BPE tokens per word;
- BPE tokens per character;
- fraction of words split into 2+ BPE pieces;
- fraction of words split into 3+ BPE pieces;
- maximum and mean pieces per word;
- normalized sequence length.

For analysis only, paired clean/corrupted inputs additionally receive:

- `delta_fragmentation`;
- `delta_sequence_length`.

These clean-counterpart deltas are explicitly excluded from `ROUTER_FEATURES`.

## Mechanism analysis

Before interpreting adaptive routing, instability is tested against three
sample-level outcomes while retaining the model seed in every join:

1. word-model classification error;
2. word-model confidence degradation relative to the clean counterpart;
3. the word-minus-character predictive-loss gap.

This avoids the prior many-seeds-per-`(sample_id, corruption_level)` indexing
error.

## Router experts and targets

Only three experts are routed between:

- word;
- BPE-500;
- character.

For expert `m`, sample `i` and trade-off weight `lambda`, the oracle minimizes:

`cross_entropy(i, m) + lambda * measured_end_to_end_CPU_cost(m)`.

The full predefined lambda sweep is retained. A separate learned router is fit
for **every lambda**, rather than training only one midpoint policy.

Router families:

- logistic regression (primary learned router);
- shallow decision tree;
- gradient boosting;
- interpretable fragmentation-threshold router;
- oracle.

The threshold router is tuned only on calibration data and uses two thresholds
to permit word -> BPE-500 -> character routing.

## Correct routing regret

Regret is evaluated against the same lambda-penalized objective that defines
the oracle:

`chosen_loss + lambda * chosen_cost - oracle_objective`.

It is not measured against the loss-only best expert when lambda is nonzero.

## Required fixed and simple baselines

The final frontier contains at least:

- word;
- BPE-500;
- BPE-5000;
- character;
- BPE-500 with BPE dropout;
- fragmentation-threshold routing;
- learned routing;
- oracle routing.

BPE dropout therefore competes directly with adaptive routing rather than
appearing only in a separate degradation plot.

## True OOD corruption holdout

The controlled OOD test uses `substitution` as the prespecified held-out
corruption family.

For the OOD router:

- calibration corruptions use swap, deletion and insertion only;
- substitution is excluded completely from OOD router fitting/tuning;
- evaluation uses substitution-only corrupted final-test examples.

This is a real corruption-family holdout. Merely training on the full mixed
corruption distribution and evaluating on substitution-only examples is not
called a holdout.

## CPU and memory accounting

The key frontier x-axis is **mean end-to-end CPU latency per input**, not active
sequence length.

Fixed-expert latency includes:

1. tokenization/vectorization;
2. model inference.

Adaptive latency includes:

1. instability-feature extraction, including BPE fragmentation work;
2. router inference;
3. the selected expert's end-to-end latency.

The exported frontier also retains active sequence length and active parameter
count as secondary efficiency measures.

Resident cost is reported separately from active compute. The adaptive resident
size includes all three stored expert models plus the serialized router.
Inference peak RAM is measured with a process-level monitor for fixed experts
and the primary adaptive policy.

## Relative degradation and statistical analysis

Macro-F1 is retained per seed and corruption level. Clean-relative Macro-F1
degradation is computed per seed before aggregation.

The predefined learned-logistic lambda family is compared with the best fixed
representation using:

- paired seed-level differences in mean nonzero-corruption relative degradation;
- 95% confidence intervals;
- Cohen's dz;
- exact paired sign-flip p-values;
- Holm correction across the lambda family.

A representative midpoint logistic policy is additionally compared with the
three router experts using paired McNemar tests with Holm correction on
identical final-test rows and all nine seeds.

## Failure analysis

For both CNN and TCN, the primary learned policy exports per-example failure
cases and counts using the shared taxonomy, including:

- missed fragmentation / under-routing;
- rare-word lookalikes;
- unnecessary character routing;
- character-model failure.

This makes the CNN-vs-TCN failure comparison real rather than attempting to
load a TCN artifact that was never produced.

## TCN replication

The TCN remains one frozen compact architecture. There is no TCN architecture
sweep. It repeats the same:

- six fixed tokenizers;
- instability diagnostics;
- three-expert routing study;
- lambda sweep;
- BPE-dropout baseline;
- true OOD holdout;
- CPU/memory accounting;
- statistical procedure;
- failure analysis.

## Cross-backbone comparison

`compare_v2_authoritative.py` fixes the two prior analysis errors:

1. clean Macro-F1 is averaged across seeds rather than taking the first seed;
2. granularity is ordered correctly from coarse to fine:
   word -> BPE-5000 -> BPE-2000 -> BPE-1000 -> BPE-500 -> character.

CNN and TCN are compared on qualitative conclusions and matched quantities,
not treated as an architecture leaderboard. The central figure is a two-panel
Macro-F1 versus measured end-to-end CPU-latency frontier.

## Execution

```bash
python run_v2_final.py --backbone cnn
python run_v2_final.py --backbone tcn
python compare_v2_authoritative.py
```

Do not report v2 results until both authoritative runs complete and their run
manifests contain `"protocol": "v2_authoritative_leakage_safe"`.
