# Typo Robustness: Word vs Character Tokenization

This project is a controlled text-classification experiment comparing word-level and character-level tokenization under typographical noise.

**Research question:** When the model architecture and training data are held broadly constant, does character-level tokenization preserve classification performance better than word-level tokenization as the typo rate increases?

The experiment was inspired by Chai et al.'s *Tokenization Falling Short: On Subword Robustness in Large Language Models*, which examines how tokenization makes language models sensitive to typographical and formatting variations.

## Experiment

The task is four-class news-topic classification on the AG News dataset. The notebook uses article descriptions only and creates balanced splits of 12,000 training, 2,000 validation, and 2,000 test examples.

Two 1D CNN classifiers use the same overall architecture - embedding, spatial dropout, convolution, global max pooling, dropout, and dense layers - but different input representations:

- **Word model:** whitespace tokenization, a vocabulary capped at 5,000 tokens, and sequences of 69 tokens.
- **Character model:** character tokenization, a vocabulary capped at 200 tokens, and sequences of 443 tokens.

Sequence lengths defined to cover the full length of 99% of training samples.

Each model is trained with three seeds. At evaluation time, 0%, 5%, 10%, 15%, or 20% of words receive one randomly selected typo: an adjacent-character swap, a QWERTY-neighbor substitution, a deletion, or an insertion (each with equal probability). The notebook reports accuracy, macro precision, macro F1, Matthews correlation coefficient, log loss, resource use, and McNemar tests with Holm correction.

## Results

The results are unsurprising: the character model is more robust to increasing typo rates, while the word model remains better in absolute terms throughout the experiment.

| Words corrupted | Word accuracy | Character accuracy | Word macro F1 | Character macro F1 |
| ---: | ---: | ---: | ---: | ---: |
| 0% | 85.32% | 80.97% | 85.27% | 80.89% |
| 5% | 84.97% | 80.60% | 84.91% | 80.52% |
| 10% | 84.28% | 80.35% | 84.22% | 80.28% |
| 15% | 83.67% | 79.97% | 83.61% | 79.89% |
| 20% | 82.37% | 79.70% | 82.30% | 79.64% |

Values are means across three training seeds. From clean text to 20% corruption, macro F1 falls by **2.97 percentage points** for the word model and **1.25 points** for the character model.

This is not yet a capacity-matched comparison. The word model has 189,124 parameters versus 30,372 for the character model - **6.23× more** - simply because of its larger embedding vocabulary, and its saved training curves show strong overfitting. The larger clean-performance baseline therefore should not be read as evidence that word tokenization is inherently better.

## Replication

1. Clone the repository and enter its directory.
2. Use Python 3.12 to create and activate a virtual environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Install the Python dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

4. Download the [AG News Classification Dataset from Kaggle](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset/data). Place `train.csv` and `test.csv` in `dataset/`.
5. Create the output directories if they are not already present:

   ```powershell
   New-Item -ItemType Directory -Force dataset, results/histories, artifacts/model_summaries, artifacts/models, figures | Out-Null
   ```

6. Start Jupyter and run every cell in the main notebook:

   ```powershell
   jupyter lab Typo_tokenization.ipynb
   ```

The run is deterministic for the configured seeds and writes timestamped metrics, predictions, model files, vocabularies, training histories, and figures to `results/`, `artifacts/`, and `figures/`. CPU execution is intended in light of the tiny size of the models.

## Technologies

- Python 3.12 and Jupyter
- TensorFlow and Keras
- NumPy and pandas
- scikit-learn and statsmodels
- Matplotlib and Seaborn
- psutil

## Project layout

```text
Typo_tokenization.ipynb  Main experiment and executed analysis
dataset/                 AG News train/test CSV files (not committed)
results/                 Metrics, predictions, tests, and histories
artifacts/               Configs, vocabularies, model summaries, and models (not committed)
figures/                 Generated plots
```

## Future work

- Extend the corruption range to a 30% typo rate.
- Allow multiple typos within a single word.
- Reduce the size of the word model to combat overfitting and try to equalize the compute

## License

This project is licensed under the [MIT License](LICENSE).

## References

1. Aman Anand Rai. *[AG News Classification Dataset](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset/data).* Kaggle, Version 2.
2. Yekun Chai, Yewei Fang, Qiwei Peng, and Xuhong Li. 2024. *[Tokenization Falling Short: On Subword Robustness in Large Language Models](https://aclanthology.org/2024.findings-emnlp.86/).* Findings of the Association for Computational Linguistics: EMNLP 2024, 1582–1599. Association for Computational Linguistics. [doi:10.18653/v1/2024.findings-emnlp.86](https://doi.org/10.18653/v1/2024.findings-emnlp.86).
