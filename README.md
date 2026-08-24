# Typo Robustness — Adaptive Tokenization Granularity

This repository studies how tokenization granularity affects text-classification robustness under typographical noise, and whether a compact system can **adapt representation granularity per input** when finer tokenization is worth its additional compute.

The project has two stages:

1. **v1 — fixed tokenizers:** controlled comparison of word, byte-level BPE and character tokenization under the same compact CNN.
2. **v2 — adaptive tokenization:** leakage-safe routing between word, BPE-500 and character experts, plus a matched replication with one frozen compact TCN backbone.

The v2 contribution is the **robustness–compute trade-off**, not higher absolute accuracy or a larger model benchmark.

## Research questions

**RQ1 — CNN:** Can observable tokenization instability predict when coarse tokenization will fail, allowing a compact router to adapt representation granularity to each input?

**RQ2 — TCN:** Do the same tokenization trade-offs, instability signals and adaptive-routing gains persist when the CNN is replaced by one prespecified compact TCN?

## v1 baseline

The frozen baseline uses AG News descriptions only, with balanced splits of 12,000 training, 2,000 validation and 2,000 test examples.

Six representations are compared under the same CNN concept:

- word tokenization;
- BPE vocabularies of 500, 1,000, 2,000 and 5,000;
- character tokenization.

Each model is trained with nine seeds. Evaluation uses nested corruption levels from 0% to 50% of words. A corrupted word receives an adjacent-character swap, QWERTY-neighbor substitution, deletion or insertion. Resource use and paired statistical tests are recorded.

The committed v1 run `20260820_052447` shows the motivating trade-off:

| Model | Clean Macro-F1 | Macro-F1 at 50% | Mean relative degradation |
| --- | ---: | ---: | ---: |
| Word | **85.10%** | 75.80% | 4.72% |
| BPE-500 | 80.80% | 73.84% | 3.69% |
| BPE-1,000 | 81.47% | 72.13% | 5.39% |
| BPE-2,000 | 82.76% | 72.82% | 5.25% |
| BPE-5,000 | 83.96% | 74.64% | 5.02% |
| Character | 80.87% | **76.34%** | **2.74%** |

Word tokenization performs best on clean text; character tokenization is most robust but produces much longer sequences; BPE-500 is a useful intermediate point.

## v2 protocol

Final v2 results must be produced by:

```bash
python run_v2_final.py --backbone cnn
python run_v2_final.py --backbone tcn
python compare_v2_authoritative.py
```

The exploratory `v2_*.ipynb` notebooks are retained for inspection but are **not authoritative for final reported numbers**.

The complete protocol is documented in [`V2_METHODOLOGY.md`](V2_METHODOLOGY.md).

### Leakage-safe router calibration

The router is never fitted on final-test examples. Instead, the frozen v1 expert experiment remains unchanged and router calibration uses the next 500 otherwise-unused AG News training rows per class after the first 3,500 rows per class consumed by the v1 train/validation pool.

This yields 2,000 disjoint calibration texts. Expert predictions on corrupted calibration examples define routing targets. Model seed is not a router feature.

### Instability diagnostics

Deployable features include:

- word OOV rate;
- BPE tokens per word and per character;
- fractions of words split into 2+ and 3+ pieces;
- maximum/mean pieces per word;
- sequence-length statistics;
- text, word and character length;
- punctuation and digit ratios.

For analysis only, paired clean/corrupted examples also receive `delta_fragmentation` and `delta_sequence_length`. Clean-counterpart information is never passed to a deployed router.

Instability is tested against:

1. word-model classification error;
2. confidence degradation relative to the clean counterpart;
3. the word-minus-character predictive-loss gap.

### Adaptive routing

The router chooses among:

- **word**;
- **BPE-500**;
- **character**.

For sample `i`, expert `m` and trade-off weight `lambda`, the oracle minimizes:

```text
cross_entropy(i, m) + lambda * measured_end_to_end_CPU_cost(m)
```

A separate router is trained for every predefined `lambda` value. Implemented router families are logistic regression, shallow decision tree and gradient boosting, plus a calibrated fragmentation-threshold router and the oracle.

Routing regret is evaluated against the same lambda-penalized oracle objective.

### Required baselines

The final frontier contains:

- word;
- BPE-500;
- BPE-5000;
- character;
- BPE-500 with BPE dropout;
- fragmentation-threshold routing;
- learned routing;
- oracle routing.

### True OOD holdout

`substitution` is the prespecified corruption-family holdout. The OOD router is calibrated only on swap, deletion and insertion corruptions, then evaluated on substitution-only corruptions of the final test set.

### Efficiency accounting

The key v2 figure is:

```text
Macro-F1 vs measured end-to-end CPU latency per input
```

Fixed-expert latency includes tokenization/vectorization and model inference. Adaptive latency additionally includes instability-feature extraction and router inference.

The repository also records:

- active sequence length;
- active parameter count;
- resident system size;
- inference peak RAM;
- training time and training RAM;
- routing allocation;
- routing regret.

### Statistical analysis

Clean-relative Macro-F1 degradation is calculated per seed before aggregation. Learned logistic policies across the lambda sweep are compared with the best fixed representation using paired seed-level differences, 95% confidence intervals, Cohen's dz, exact paired sign-flip tests and Holm correction.

A representative midpoint policy is additionally evaluated with paired McNemar tests and Holm correction on identical final-test rows and all nine seeds.

## TCN replication

The TCN branch uses one fixed compact same-padded dilated residual architecture with dilation schedule `(1, 2, 4)`. It repeats the same six fixed tokenizers, diagnostics, adaptive-routing protocol, BPE-dropout baseline, OOD holdout, efficiency accounting, statistics and failure analysis.

There is no TCN architecture sweep. RQ2 asks whether the **mechanism replicates across backbones**, not whether TCN beats CNN in absolute performance.

## Cross-backbone analysis

`compare_v2_authoritative.py` compares matched qualitative conclusions and effect sizes and produces the two-panel CPU frontier.

Granularity is ordered correctly from coarse to fine:

```text
word -> BPE-5000 -> BPE-2000 -> BPE-1000 -> BPE-500 -> character
```

Clean Macro-F1 is averaged across seeds rather than taking the first seed.

## Repository layout

```text
Typo_tokenization.ipynb          Frozen v1 experiment
common.py                        Shared frozen-v1 + v2 model/corruption machinery
v2_methodology.py                Leakage-safe adaptive-routing/evaluation utilities
run_v2_final.py                  Authoritative final CNN/TCN execution pipeline
compare_v2_authoritative.py      Corrected CNN-vs-TCN comparison
V2_METHODOLOGY.md                Detailed v2 protocol and methodological invariants
v2_rq1_cnn_adaptive.ipynb        Exploratory RQ1 notebook
v2_rq2_tcn_adaptive.ipynb        Exploratory RQ2 notebook
v2_comparison_cnn_vs_tcn.ipynb   Exploratory comparison notebook
results/                         Metrics, predictions, statistics and manifests
artifacts/                       Models, tokenizers, configs and split artifacts
figures/                         Generated figures
```

## Replication

Use Python 3.12 and install dependencies with:

```bash
python -m pip install -r requirements.txt
```

Download the AG News Classification Dataset and place `train.csv` and `test.csv` in `dataset/`.

Run v1 with:

```bash
jupyter lab Typo_tokenization.ipynb
```

Run final v2 with the three commands shown above. Do not report v2 results until both run manifests contain:

```json
{"protocol": "v2_authoritative_leakage_safe"}
```

## Technologies

- Python 3.12
- TensorFlow / Keras
- Hugging Face Tokenizers
- NumPy and pandas
- SciPy, scikit-learn and statsmodels
- Matplotlib
- psutil

## License

MIT — see [`LICENSE`](LICENSE).

## References

1. Aman Anand Rai. *AG News Classification Dataset*. Kaggle, Version 2.
2. Yekun Chai, Yewei Fang, Qiwei Peng, and Xuhong Li. 2024. *Tokenization Falling Short: On Subword Robustness in Large Language Models*. Findings of EMNLP 2024, 1582–1599.
