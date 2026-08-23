"""Shared machinery for Typo Robustness v2 — Adaptive Tokenization Granularity.

Everything in this module is either:

1. Ported verbatim (or with a documented, behavior-preserving tweak) from the
   frozen v1 notebook ``Typo_tokenization.ipynb`` so that the baseline
   experimental conditions (split, seeds, corruption generator, nested test
   sets, metrics, statistics) stay identical; or
2. New v2 code: tokenization-instability diagnostics, the oracle/routing
   target, lightweight routers, the BPE-dropout baseline, the compact TCN
   backbone, OOD corruption holdouts, and efficiency accounting.

Nothing here executes an experiment on import. Notebooks drive all runs.
"""

from __future__ import annotations

import html
import json
import os
import string
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration (frozen v1 constants + new v2 constants)
# ---------------------------------------------------------------------------

DATASET_PATH = "./dataset"
TRAIN_CSV = os.path.join(DATASET_PATH, "train.csv")
TEST_CSV = os.path.join(DATASET_PATH, "test.csv")
RESULTS_DIR = "./results"
HISTORIES_DIR = os.path.join(RESULTS_DIR, "histories")
ARTIFACTS_DIR = "./artifacts"
MODEL_SUMMARIES_DIR = os.path.join(ARTIFACTS_DIR, "model_summaries")
MODELS_DIR = os.path.join(ARTIFACTS_DIR, "models")
FIGURES_DIR = "./figures"

TRAINING_SEEDS = [123, 132, 213, 231, 321, 312, 111, 222, 333]  # v1
CORRUPTION_SEED = 0  # v1
SPLIT_SEED = 1  # v1

CORRUPTIONS = ["swap", "substitution", "deletion", "insertion"]  # v1
CORRUPTION_LEVELS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]  # v1
DOUBLE_CORRUPTION_PERCENT = 7.5  # v1

STANDARDIZATION = "lower_and_strip_punctuation"

WORD_MAX_TOKENS = 5000
WORD_SEQUENCE_LENGTH = 69
WORD_EMBEDDING_DIM = 32

BPE_VOCAB_SIZES = [500, 1000, 2000, 5000]
BPE_MIN_FREQUENCY = 2
BPE_SEQUENCE_LENGTHS = {500: 206, 1000: 157, 2000: 126, 5000: 99}
BPE_EMBEDDING_DIM = 32

CHAR_MAX_TOKENS = 200
CHAR_SEQUENCE_LENGTH = 443
CHAR_EMBEDDING_DIM = 32

CONV_FILTERS = 128
KERNEL_SIZE = 5
DENSE_UNITS = 64

HIDDEN_ACTIVATION = "relu"
OUTPUT_ACTIVATION = "softmax"

DROPOUT_RATE = 0.2

NUM_CLASSES = 4

LEARNING_RATE = 1e-3
ALPHA_LR = 0.01  # cosine-decay floor (v1 constant ALPHA)
DECAY_STEPS = 2820

EPOCHS = 50
PATIENCE = 5
PATIENCE_MIN_DELTA = 0.005
BATCH_SIZE = 64

BPE_MODEL_NAMES = [f"bpe_{vocab_size}" for vocab_size in BPE_VOCAB_SIZES]
MODEL_ORDER = ["word", *BPE_MODEL_NAMES, "char"]
MODEL_PAIRS = list(combinations(MODEL_ORDER, 2))

# Expert subset every router study uses (plan §6). Do not widen without cause.
ROUTER_EXPERTS = ["word", "bpe_500", "char"]

EXPERT_SEQUENCE_LENGTHS = {
    "word": WORD_SEQUENCE_LENGTH,
    **{f"bpe_{v}": BPE_SEQUENCE_LENGTHS[v] for v in BPE_VOCAB_SIZES},
    "char": CHAR_SEQUENCE_LENGTH,
}

METRICS = {
    "accuracy": "Accuracy",
    "f1_macro": "Macro F1",
    "mcc": "MCC",
    "log_loss": "Log loss",
}

# ---------------------------------------------------------------------------
# New v2 configuration (prespecified once, then frozen — plan §§10, 12)
# ---------------------------------------------------------------------------

TCN_CONFIG = {
    "embedding_dim": 32,      # match the CNN embedding width
    "channels": 64,           # residual-block width
    "kernel_size": 5,
    "dilations": (1, 2, 4),   # small fixed dilation schedule
    "dropout_rate": DROPOUT_RATE,
    "dense_units": DENSE_UNITS,
}

# BPE dropout probability (Provilkov et al.-style subword regularization).
BPE_DROPOUT_PROB = 0.1

# Lambda sweep for the performance-compute trade-off (plan §7).
LAMBDA_SWEEP = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]


def new_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dirs() -> None:
    for directory in (
        RESULTS_DIR,
        HISTORIES_DIR,
        ARTIFACTS_DIR,
        MODEL_SUMMARIES_DIR,
        MODELS_DIR,
        FIGURES_DIR,
    ):
        os.makedirs(directory, exist_ok=True)


# ---------------------------------------------------------------------------
# QWERTY neighbor table (verbatim from v1 cell 6)
# ---------------------------------------------------------------------------

QWERTY_NEIGHBORS = {
    "`": ["1", "q"],
    "~": ["!", "Q"],
    "1": ["`", "2", "q", "w"],
    "!": ["~", "@", "Q", "W"],
    "2": ["1", "3", "w", "e"],
    "@": ["!", "#", "W", "E"],
    "3": ["2", "4", "e", "r"],
    "#": ["@", "$", "E", "R"],
    "4": ["3", "5", "r", "t"],
    "$": ["#", "%", "R", "T"],
    "5": ["4", "6", "t", "y"],
    "%": ["$", "^", "T", "Y"],
    "6": ["5", "7", "y", "u"],
    "^": ["%", "&", "Y", "U"],
    "7": ["6", "8", "u", "i"],
    "&": ["^", "*", "U", "I"],
    "8": ["7", "9", "i", "o"],
    "*": ["&", "(", "I", "O"],
    "9": ["8", "0", "o", "p"],
    "(": ["*", ")", "O", "P"],
    "0": ["9", "-", "p", "["],
    ")": ["(", "_", "P", "{"],
    "-": ["0", "=", "[", "]"],
    "_": [")", "+", "{", "}"],
    "=": ["-", "]", "["],
    "+": ["_", "}", "|"],
    "q": ["`", "1", "w", "a"],
    "Q": ["~", "!", "W", "A"],
    "w": ["1", "2", "q", "e", "a", "s"],
    "W": ["!", "@", "Q", "E", "A", "S"],
    "e": ["2", "3", "w", "r", "s", "d"],
    "E": ["@", "#", "W", "R", "S", "D"],
    "r": ["3", "4", "e", "t", "d", "f"],
    "R": ["#", "$", "E", "T", "D", "F"],
    "t": ["4", "5", "r", "y", "f", "g"],
    "T": ["$", "%", "R", "Y", "F", "G"],
    "y": ["5", "6", "t", "u", "g", "h"],
    "Y": ["%", "^", "T", "U", "G", "H"],
    "u": ["6", "7", "y", "i", "h", "j"],
    "U": ["^", "&", "Y", "I", "H", "J"],
    "i": ["7", "8", "u", "o", "j", "k"],
    "I": ["&", "*", "U", "O", "J", "K"],
    "o": ["8", "9", "i", "p", "k", "l"],
    "O": ["*", "(", "I", "P", "K", "L"],
    "p": ["9", "0", "o", "[", "l", ";"],
    "P": ["(", ")", "O", "{", "L", ":"],
    "[": ["0", "-", "p", "]", ";", "'"],
    "{": [")", "_", "P", "}", ':', '"'],
    "]": ["-", "=", "[", "\\", "'"],
    "}": ["_", "+", "{", "|", '"'],
    "\\": ["=", "]"],
    "|": ["+", "}"],
    "a": ["q", "w", "s", "z"],
    "A": ["Q", "W", "S", "Z"],
    "s": ["w", "e", "a", "d", "z", "x"],
    "S": ["W", "E", "A", "D", "Z", "X"],
    "d": ["e", "r", "s", "f", "x", "c"],
    "D": ["E", "R", "S", "F", "X", "C"],
    "f": ["r", "t", "d", "g", "c", "v"],
    "F": ["R", "T", "D", "G", "C", "V"],
    "g": ["t", "y", "f", "h", "v", "b"],
    "G": ["T", "Y", "F", "H", "V", "B"],
    "h": ["y", "u", "g", "j", "b", "n"],
    "H": ["Y", "U", "G", "J", "B", "N"],
    "j": ["u", "i", "h", "k", "n", "m"],
    "J": ["U", "I", "H", "K", "N", "M"],
    "k": ["i", "o", "j", "l", "m", ","],
    "K": ["I", "O", "J", "L", "M", "<"],
    "l": ["o", "p", "k", ";", ",", "."],
    "L": ["O", "P", "K", ":", "<", ">"],
    ";": ["p", "[", "l", "'", ".", "/"],
    ":": ["P", "{", "L", '"', ">", "?"],
    "'": ["[", "]", ";", "/"],
    '"': ["{", "}", ":", "?"],
    "z": ["a", "s", "x"],
    "Z": ["A", "S", "X"],
    "x": ["s", "d", "z", "c"],
    "X": ["S", "D", "Z", "C"],
    "c": ["d", "f", "x", "v"],
    "C": ["D", "F", "X", "V"],
    "v": ["f", "g", "c", "b"],
    "V": ["F", "G", "C", "B"],
    "b": ["g", "h", "v", "n"],
    "B": ["G", "H", "V", "N"],
    "n": ["h", "j", "b", "m"],
    "N": ["H", "J", "B", "M"],
    "m": ["j", "k", "n", ","],
    "M": ["J", "K", "N", "<"],
    ",": ["k", "l", "m", "."],
    "<": ["K", "L", "M", ">"],
    ".": ["l", ";", ",", "/"],
    ">": ["L", ":", "<", "?"],
    "/": [";", "'", "."],
    "?": [':', '"', ">"],
}


# ---------------------------------------------------------------------------
# Data loading and splitting (ported from v1 cells 12–17)
# ---------------------------------------------------------------------------


def load_train_val_set() -> pd.DataFrame:
    """Balanced 14,000-example pool (3,500 per class), cleaned exactly as v1."""
    train_val_set = pd.read_csv(TRAIN_CSV).groupby("Class Index", sort=False).head(3500)
    train_val_set = train_val_set.drop(columns=["Title"])
    train_val_set["Description"] = train_val_set["Description"].map(html.unescape)
    # NOTE (behavior-preserving port): v1 replaced '\\' with ' ' here but with ''
    # in the test frame below. Kept deliberately different, as in v1.
    train_val_set["Description"] = (
        train_val_set["Description"]
        .str.replace("\\", " ")
        .str.replace("quot;", "")
        .str.replace("--", "-")
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    return train_val_set


def load_test_set() -> pd.DataFrame:
    """Balanced 2,000-example test set (500 per class), cleaned exactly as v1."""
    test_df = (
        pd.read_csv(TEST_CSV)
        .groupby("Class Index", sort=False)
        .head(500)
        .drop(columns=["Title"])
    )
    test_df["Description"] = test_df["Description"].map(html.unescape)
    test_df["Description"] = (
        test_df["Description"]
        .str.replace("\\", "")
        .str.replace("quot;", "")
        .str.replace("--", "-")
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    return test_df


def make_train_val_split(train_val_set: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.model_selection import train_test_split

    # v1: 14,000 x 1/7 -> 2,000 validation examples, stratified, SPLIT_SEED.
    return train_test_split(
        train_val_set,
        test_size=1 / 7,
        random_state=SPLIT_SEED,
        stratify=train_val_set["Class Index"],
    )


def save_split_ids(train_df: pd.DataFrame, val_df: pd.DataFrame, run_id: str) -> None:
    split_df = pd.concat(
        [
            pd.DataFrame(
                {
                    "sample_id": train_df.index,
                    "split": "train",
                    "class_index": train_df["Class Index"].to_numpy(),
                }
            ),
            pd.DataFrame(
                {
                    "sample_id": val_df.index,
                    "split": "validation",
                    "class_index": val_df["Class Index"].to_numpy(),
                }
            ),
        ],
        ignore_index=True,
    )
    split_df.to_csv(os.path.join(ARTIFACTS_DIR, f"split_ids_{run_id}.csv"), index=False)


# ---------------------------------------------------------------------------
# Corruption generator (ported verbatim from v1 cell 20)
# ---------------------------------------------------------------------------


def apply_typo(chars: list[str], i: int, rng: np.random.Generator) -> None:
    corruption = rng.choice(CORRUPTIONS)

    if corruption == "swap":
        neighbors = [
            j for j in (i - 1, i + 1) if 0 <= j < len(chars) and chars[j] != chars[i]
        ]
        if neighbors:
            j = int(rng.choice(neighbors))
            chars[i], chars[j] = chars[j], chars[i]
            return
        corruption = "insertion"

    replacements = QWERTY_NEIGHBORS.get(chars[i], ["x"])
    if corruption == "substitution":
        chars[i] = rng.choice(replacements)
    elif corruption == "deletion":
        del chars[i]
    else:
        chars.insert(i + 1, rng.choice(replacements))


def corrupt_word(word: str, rng: np.random.Generator) -> tuple[str, int]:
    positions = [int(rng.integers(len(word)))]

    if len(word) >= 5 and rng.random() < DOUBLE_CORRUPTION_PERCENT / 100:
        candidates = [i for i in range(len(word)) if abs(i - positions[0]) > 1]
        positions.append(int(rng.choice(candidates)))

    chars = list(word)
    for i in sorted(positions, reverse=True):
        apply_typo(chars, i, rng)

    corrupted_word = "".join(chars)
    if corrupted_word == word:
        corrupted_word += "x"
    return corrupted_word, len(positions)


def build_nested_test_sets(
    source_df: pd.DataFrame,
    corruption_levels: list[int],
    corruption_seed: int,
    allowed_corruptions: Sequence[str] | None = None,
) -> tuple[dict[int, pd.DataFrame], pd.DataFrame]:
    """Nested typo test sets, ported verbatim from v1 cell 20.

    The only new argument is ``allowed_corruptions`` (plan §11 Option A): it
    restricts which corruption families are applied and is used ONLY for the
    OOD corruption-type-holdout condition. With the default ``None`` the
    generated sets are bit-identical to v1 given the same seed.
    """
    active = list(allowed_corruptions) if allowed_corruptions else list(CORRUPTIONS)
    if not set(active) <= set(CORRUPTIONS):
        raise ValueError(f"allowed_corruptions must be a subset of {CORRUPTIONS}")
    if "swap" in active and "insertion" not in active:
        # apply_typo falls back from swap to insertion when no neighbor differs;
        # forbid ambiguous configurations so holdouts stay interpretable.
        raise ValueError("swap requires insertion to be allowed (fallback path).")

    rng = np.random.default_rng(corruption_seed)
    rows = [description.split(" ") for description in source_df["Description"]]
    word_positions = [
        (row_i, word_i)
        for row_i, words in enumerate(rows)
        for word_i, word in enumerate(words)
        if word
    ]

    corruption_order = rng.permutation(len(word_positions))
    corrupted_words = []
    typo_counts = []

    def _apply_typo(chars: list[str], i: int) -> None:
        corruption = rng.choice(active)
        if corruption == "swap":
            neighbors = [
                j for j in (i - 1, i + 1) if 0 <= j < len(chars) and chars[j] != chars[i]
            ]
            if neighbors:
                j = int(rng.choice(neighbors))
                chars[i], chars[j] = chars[j], chars[i]
                return
            corruption = "insertion"
        replacements = QWERTY_NEIGHBORS.get(chars[i], ["x"])
        if corruption == "substitution":
            chars[i] = rng.choice(replacements)
        elif corruption == "deletion":
            del chars[i]
        else:
            chars.insert(i + 1, rng.choice(replacements))

    def _corrupt_word(word: str) -> tuple[str, int]:
        positions = [int(rng.integers(len(word)))]
        if len(word) >= 5 and rng.random() < DOUBLE_CORRUPTION_PERCENT / 100:
            candidates = [i for i in range(len(word)) if abs(i - positions[0]) > 1]
            positions.append(int(rng.choice(candidates)))
        chars = list(word)
        for i in sorted(positions, reverse=True):
            _apply_typo(chars, i)
        corrupted_word = "".join(chars)
        if corrupted_word == word:
            corrupted_word += "x"
        return corrupted_word, len(positions)

    for row_i, word_i in word_positions:
        corrupted_word, typo_count = _corrupt_word(rows[row_i][word_i])
        corrupted_words.append(corrupted_word)
        typo_counts.append(typo_count)

    test_sets: dict[int, pd.DataFrame] = {}
    summary_rows = []
    total_words = len(word_positions)

    for corruption_level in corruption_levels:
        target_count = int(np.floor(total_words * corruption_level / 100 + 0.5))
        selected = corruption_order[:target_count]
        corrupted_rows = [words.copy() for words in rows]
        for flat_i in selected:
            row_i, word_i = word_positions[flat_i]
            corrupted_rows[row_i][word_i] = corrupted_words[flat_i]

        test_sets[corruption_level] = source_df.assign(
            Description=[" ".join(words) for words in corrupted_rows]
        )
        summary_rows.append(
            {
                "corruption_level": corruption_level,
                "total_words": total_words,
                "corrupted_words": target_count,
                "realized_corruption_rate": target_count / total_words,
                "double_corrupted_words": sum(typo_counts[i] == 2 for i in selected),
            }
        )

    return test_sets, pd.DataFrame(summary_rows)


# ---------------------------------------------------------------------------
# Tokenizers and vectorizers (ported from v1 cells 24/26)
# ---------------------------------------------------------------------------

BPE_PUNCTUATION_TABLE = str.maketrans("", "", string.punctuation)


def standardize_for_bpe(text: str) -> str:
    """Match Keras' lower_and_strip_punctuation standardization (v1)."""
    return text.lower().translate(BPE_PUNCTUATION_TABLE)


def standardize_words(text: str) -> list[str]:
    """Whitespace tokens after lower_and_strip_punctuation, plain Python."""
    return str(text).lower().translate(BPE_PUNCTUATION_TABLE).split()


def pad_truncate_rows(rows: list[list[int]], length: int, pad_id: int) -> list[list[int]]:
    return [(row[:length] + [pad_id] * length)[:length] for row in rows]


@dataclass
class TokenizationSuite:
    """Vectorizers/tokenizers for one run.

    ``vectorize(name, texts)`` reproduces exactly what the v1 vectorizers
    produced for expert ``name`` ('word' | 'bpe_<vocab>' | 'char'): Keras
    TextVectorization layers for word/char (same adapt, same standardization,
    same fixed output_sequence_length) and the HF byte-level BPE tokenizer
    with v1's padding/truncation settings.
    """

    word_vectorizer: object
    char_vectorizer: object
    bpe_tokenizers: dict[int, object]
    bpe_train_texts: list[str]

    @property
    def sequence_lengths(self) -> dict[str, int]:
        return dict(EXPERT_SEQUENCE_LENGTHS)

    def vectorize(self, name: str, texts: Sequence[str]):
        import tensorflow as tf

        texts = [str(text) for text in texts]
        if name == "word":
            return self.word_vectorizer(np.asarray(texts))
        if name == "char":
            return self.char_vectorizer(np.asarray(texts))
        if name.startswith("bpe_"):
            vocab_size = int(name.split("_")[1])
            tokenizer = self.bpe_tokenizers[vocab_size]
            standardized = [standardize_for_bpe(text) for text in texts]
            encodings = tokenizer.encode_batch(standardized)
            return tf.convert_to_tensor([enc.ids for enc in encodings], dtype=tf.int32)
        raise KeyError(f"Unknown model name: {name}")


def train_bpe_tokenizer(vocab_size: int, bpe_train_texts: list[str]):
    from tokenizers import Tokenizer
    from tokenizers import models as tokenizer_models
    from tokenizers import pre_tokenizers as tokenizer_pre_tokenizers
    from tokenizers import trainers as tokenizer_trainers

    tokenizer = Tokenizer(tokenizer_models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = tokenizer_pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = tokenizer_trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=BPE_MIN_FREQUENCY,
        special_tokens=["[PAD]", "[UNK]"],
        initial_alphabet=tokenizer_pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tokenizer.train_from_iterator(bpe_train_texts, trainer=trainer, length=len(bpe_train_texts))
    return tokenizer


def unpadded_bpe_clone(tokenizer):
    """A clone of ``tokenizer`` with padding/truncation disabled.

    Needed for feature extraction and per-word piece counting: the production
    tokenizers pad every encoding to the fixed sequence length, which would
    corrupt piece-per-word statistics.
    """
    from tokenizers import Tokenizer

    clone = Tokenizer.from_str(tokenizer.to_str())
    clone.no_padding()
    clone.no_truncation()
    return clone


def build_tokenization_suite(train_texts: Sequence[str]) -> TokenizationSuite:
    """Train/adapt every tokenizer on the training split only, as in v1."""
    from tensorflow.keras.layers import TextVectorization

    train_array = np.asarray([str(text) for text in train_texts])
    bpe_train_texts = [standardize_for_bpe(text) for text in train_array]

    bpe_tokenizers = {
        vocab_size: train_bpe_tokenizer(vocab_size, bpe_train_texts)
        for vocab_size in BPE_VOCAB_SIZES
    }

    for vocab_size, tokenizer in bpe_tokenizers.items():
        sequence_length = BPE_SEQUENCE_LENGTHS[vocab_size]
        tokenizer.enable_padding(
            length=sequence_length,
            pad_id=tokenizer.token_to_id("[PAD]"),
            pad_token="[PAD]",
        )
        tokenizer.enable_truncation(max_length=sequence_length)

    word_vectorizer = TextVectorization(
        max_tokens=WORD_MAX_TOKENS,
        standardize=STANDARDIZATION,
        split="whitespace",
        output_mode="int",
        output_sequence_length=WORD_SEQUENCE_LENGTH,
    )
    word_vectorizer.adapt(train_array)

    char_vectorizer = TextVectorization(
        max_tokens=CHAR_MAX_TOKENS,
        standardize=STANDARDIZATION,
        split="character",
        output_mode="int",
        output_sequence_length=CHAR_SEQUENCE_LENGTH,
    )
    char_vectorizer.adapt(train_array)

    return TokenizationSuite(
        word_vectorizer=word_vectorizer,
        char_vectorizer=char_vectorizer,
        bpe_tokenizers=bpe_tokenizers,
        bpe_train_texts=bpe_train_texts,
    )


def save_artifacts(suite: TokenizationSuite, run_id: str) -> None:
    """Save vocabularies and BPE tokenizer JSONs exactly as v1 did (cell 25)."""
    pd.DataFrame(
        {
            "token_id": range(len(suite.word_vectorizer.get_vocabulary())),
            "token": suite.word_vectorizer.get_vocabulary(),
        }
    ).to_csv(os.path.join(ARTIFACTS_DIR, f"word_vocabulary_{run_id}.csv"), index=False)

    for vocab_size, tokenizer in suite.bpe_tokenizers.items():
        vocabulary = sorted(tokenizer.get_vocab().items(), key=lambda item: item[1])
        pd.DataFrame(
            {
                "token_id": [token_id for _, token_id in vocabulary],
                "token": [token for token, _ in vocabulary],
            }
        ).to_csv(
            os.path.join(ARTIFACTS_DIR, f"bpe_{vocab_size}_vocabulary_{run_id}.csv"),
            index=False,
        )
        tokenizer.save(os.path.join(ARTIFACTS_DIR, f"bpe_{vocab_size}_tokenizer_{run_id}.json"))

    pd.DataFrame(
        {
            "token_id": range(len(suite.char_vectorizer.get_vocabulary())),
            "token": suite.char_vectorizer.get_vocabulary(),
        }
    ).to_csv(os.path.join(ARTIFACTS_DIR, f"char_vocabulary_{run_id}.csv"), index=False)


# ---------------------------------------------------------------------------
# BPE dropout baseline (new — plan §10)
# ---------------------------------------------------------------------------


def make_bpe_dropout_tokenizer(tokenizer, dropout_prob: float = BPE_DROPOUT_PROB):
    """Clone a trained byte-level BPE tokenizer with BPE-dropout enabled.

    Uses the HuggingFace tokenizers BPE model's native ``dropout`` option
    (Provilkov et al., 2019): during encoding, merges are dropped with
    probability ``dropout_prob``, so words are sometimes split into smaller
    pieces (down to characters). Training-time-only; evaluation always uses
    the clean tokenizer. Configuration frozen at prob=0.1 (plan §10).
    """
    from tokenizers import Tokenizer
    from tokenizers import models as tokenizer_models

    with tempfile.TemporaryDirectory() as tmp_dir:
        tokenizer.model.save(tmp_dir)
        with open(os.path.join(tmp_dir, "vocab.json"), encoding="utf-8") as f:
            vocab = json.load(f)
        merges = []
        with open(os.path.join(tmp_dir, "merges.txt"), encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line or line.startswith("#version"):
                    continue
                parts = line.split(" ")
                if len(parts) == 2:
                    merges.append((parts[0], parts[1]))

    dropout_model = tokenizer_models.BPE(
        vocab=vocab,
        merges=merges,
        unk_token="[UNK]",
        dropout=dropout_prob,
        fuse_unk=False,
        byte_fallback=False,
    )
    clone = Tokenizer.from_str(tokenizer.to_str())
    clone.model = dropout_model
    return clone


class BPEDropoutTrainingSequence:
    """Per-epoch re-encoded training data for the BPE-dropout baseline.

    Wraps the BPE-500 inputs so that every epoch sees freshly dropped merges.
    Pass its ``__getitem__``/``__len__`` to ``model.fit(x=a_sequence_object)``
    (Keras 3 accepts any object implementing the sequence protocol together
    with ``y`` supplied identically).

    Implemented as a plain class rather than subclassing
    ``tf.keras.utils.Sequence`` so that common.py stays importable without a
    live TensorFlow; notebooks register it via
    ``tensorflow.keras.utils.Sequence.register(BPEDropoutTrainingSequence)``
    if Keras insists on the type.
    """

    def __init__(self, texts: Sequence[str], labels: np.ndarray, dropout_tokenizer, batch_size: int = BATCH_SIZE):
        import tensorflow as tf

        self.tf = tf
        self.texts = [standardize_for_bpe(str(t)) for t in texts]
        self.labels = np.asarray(labels, dtype=np.int32)
        self.dropout_tokenizer = dropout_tokenizer
        self.batch_size = batch_size
        self.pad_id = dropout_tokenizer.token_to_id("[PAD]")
        self.width = EXPERT_SEQUENCE_LENGTHS["bpe_500"]

    def __len__(self) -> int:
        return int(np.ceil(len(self.texts) / self.batch_size))

    def __getitem__(self, idx: int):
        start = idx * self.batch_size
        batch_texts = self.texts[start : start + self.batch_size]
        encodings = self.dropout_tokenizer.encode_batch(batch_texts)
        rows = pad_truncate_rows([list(enc.ids) for enc in encodings], self.width, self.pad_id)
        x = self.tf.convert_to_tensor(rows, dtype=self.tf.int32)
        y = self.tf.convert_to_tensor(self.labels[start : start + self.batch_size])
        return x, y


# ---------------------------------------------------------------------------
# Instability features (new — plan §4)
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    "text_length",
    "word_count",
    "character_count",
    "word_oov_fraction",
    "digit_fraction",
    "punct_fraction",
    "bpe_token_count",
    "bpe_tokens_per_word",
    "bpe_tokens_per_character",
    "fraction_words_split_2plus",
    "fraction_words_split_3plus",
    "max_pieces_per_word",
    "mean_pieces_per_word",
    "sequence_length",
    "normalized_sequence_length",
]


class WordVocabularyIndex:
    """Word-level vocabulary lookup for OOV computation.

    Built from the Keras TextVectorization vocabulary so that OOV matches
    exactly what the word expert can represent: index 0 is reserved ([PAD]),
    1 is [UNK]/OOV, indices >= 2 are learned tokens.
    """

    def __init__(self, vocabulary: Sequence[str]):
        self.known_words = {token for token in vocabulary[2:] if token}

    def oov_fraction(self, tokens: Sequence[str]) -> float:
        if not tokens:
            return 0.0
        hits = sum(1 for token in tokens if token in self.known_words)
        return 1.0 - hits / len(tokens)


def make_word_vocab_index(suite: TokenizationSuite) -> WordVocabularyIndex:
    return WordVocabularyIndex(suite.word_vectorizer.get_vocabulary())


def compute_instability_features(
    texts: Sequence[str],
    suite: TokenizationSuite,
    word_vocab: WordVocabularyIndex,
    bpe_vocab_size: int = 500,
) -> pd.DataFrame:
    """Cheap inference-time features for every input (plan §4).

    Uses only information available in the raw (possibly corrupted) text — no
    clean counterpart, per plan §4. Piece-per-word statistics come from an
    unpadded tokenizer clone so padding never contaminates the counts.
    """
    texts = [str(text) for text in texts]
    bpe_tokenizer = suite.bpe_tokenizers[bpe_vocab_size]
    scratch = unpadded_bpe_clone(bpe_tokenizer)
    seq_len = EXPERT_SEQUENCE_LENGTHS[f"bpe_{bpe_vocab_size}"]

    standardized = [standardize_for_bpe(t) for t in texts]
    encodings = scratch.encode_batch(standardized)

    rows = []
    for text, encoding in zip(texts, encodings):
        raw = str(text)
        lowered = raw.lower()
        words = standardize_words(raw)
        word_count = len(words)
        n_chars_stripped = len(lowered.translate(BPE_PUNCTUATION_TABLE))
        n_raw_chars = max(len(lowered), 1)

        pieces_per_word: list[int] = []
        if word_count:
            word_encodings = scratch.encode_batch([standardize_for_bpe(w) for w in words])
            pieces_per_word = [len(enc.ids) for enc in word_encodings]

        bpe_token_count = len(encoding.ids)
        split_2plus = sum(1 for p in pieces_per_word if p >= 2)
        split_3plus = sum(1 for p in pieces_per_word if p >= 3)
        digits = sum(ch.isdigit() for ch in lowered)

        rows.append(
            {
                "text_length": len(raw),
                "word_count": word_count,
                "character_count": n_chars_stripped,
                "word_oov_fraction": word_vocab.oov_fraction(words),
                "digit_fraction": digits / n_raw_chars,
                "punct_fraction": sum(not ch.isalnum() and not ch.isspace() for ch in lowered) / n_raw_chars,
                "bpe_token_count": bpe_token_count,
                "bpe_tokens_per_word": bpe_token_count / word_count if word_count else 0.0,
                "bpe_tokens_per_character": bpe_token_count / n_chars_stripped if n_chars_stripped else 0.0,
                "fraction_words_split_2plus": split_2plus / word_count if word_count else 0.0,
                "fraction_words_split_3plus": split_3plus / word_count if word_count else 0.0,
                "max_pieces_per_word": max(pieces_per_word) if pieces_per_word else 0,
                "mean_pieces_per_word": float(np.mean(pieces_per_word)) if pieces_per_word else 0.0,
                "sequence_length": bpe_token_count,
                "normalized_sequence_length": bpe_token_count / seq_len,
            }
        )

    return pd.DataFrame(rows, columns=FEATURE_NAMES)


# ---------------------------------------------------------------------------
# Model builders (CNN ported from v1 cells 29–31; TCN new — plan §12)
# ---------------------------------------------------------------------------


def make_optimizer():
    import keras

    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=LEARNING_RATE,
        decay_steps=DECAY_STEPS,
        alpha=ALPHA_LR,
    )
    return keras.optimizers.Adam(learning_rate=lr_schedule)


def compile_model(model):
    model.compile(
        optimizer=make_optimizer(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_cnn(sequence_length: int, vocabulary_size: int, embedding_dim: int = WORD_EMBEDDING_DIM):
    """The exact v1 architecture (cells 30–31), unchanged."""
    import keras
    from keras import layers

    model = keras.Sequential(
        [
            keras.Input(shape=(sequence_length,), dtype="int32"),
            layers.Embedding(input_dim=vocabulary_size, output_dim=embedding_dim),
            layers.SpatialDropout1D(DROPOUT_RATE),
            layers.Conv1D(
                filters=CONV_FILTERS,
                kernel_size=KERNEL_SIZE,
                activation=HIDDEN_ACTIVATION,
            ),
            layers.GlobalMaxPooling1D(),
            layers.Dropout(DROPOUT_RATE),
            layers.Dense(units=DENSE_UNITS, activation=HIDDEN_ACTIVATION),
            layers.Dense(units=NUM_CLASSES, activation=OUTPUT_ACTIVATION),
        ]
    )
    return compile_model(model)


def build_tcn(sequence_length: int, vocabulary_size: int):
    """Compact same-padded dilated TCN — prespecified once (plan §12), frozen.

    Design notes (fixed before any TCN run):
      * same-padded Conv1D keeps T_out == T_in, so the identical
        GlobalMaxPooling1D head concept used by the CNN applies unchanged;
      * residual blocks with the small fixed dilation schedule (1, 2, 4);
      * pre-activation blocks: LayerNorm -> GELU -> Conv1D -> Dropout -> Conv1D;
      * block width 64 with embeddings of dim 32 keeps the parameter count in
        roughly the same small-model regime as the CNN;
      * NO architecture sweep anywhere in RQ2 (plan §12).
    """
    import keras
    from keras import layers

    cfg = TCN_CONFIG
    inputs = keras.Input(shape=(sequence_length,), dtype="int32")
    x = layers.Embedding(input_dim=vocabulary_size, output_dim=cfg["embedding_dim"])(inputs)
    x = layers.SpatialDropout1D(cfg["dropout_rate"])(x)

    for dilation in cfg["dilations"]:
        block_input = x
        y = layers.LayerNormalization()(x)
        y = layers.Activation("gelu")(y)
        y = layers.Conv1D(
            filters=cfg["channels"],
            kernel_size=cfg["kernel_size"],
            dilation_rate=dilation,
            padding="same",
        )(y)
        y = layers.Dropout(cfg["dropout_rate"])(y)
        y = layers.Conv1D(
            filters=int(block_input.shape[-1]),
            kernel_size=cfg["kernel_size"],
            dilation_rate=dilation,
            padding="same",
        )(y)
        x = layers.Add()([block_input, y])

    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dropout(cfg["dropout_rate"])(x)
    x = layers.Dense(units=cfg["dense_units"], activation=HIDDEN_ACTIVATION)(x)
    outputs = layers.Dense(units=NUM_CLASSES, activation=OUTPUT_ACTIVATION)(x)
    model = keras.Model(inputs, outputs, name=f"tcn_seq{sequence_length}")
    return compile_model(model)


def build_experiment_builders(backbone: str, suite: TokenizationSuite) -> dict[str, Callable[[], object]]:
    """Builders for all six fixed-tokenizer models under one backbone.

    Vocabulary sizes mirror v1 exactly: len(word vocab), tokenizer vocab size,
    len(char vocab).
    """
    if backbone == "cnn":

        def builder(name: str):
            if name == "word":
                return build_cnn(WORD_SEQUENCE_LENGTH, len(suite.word_vectorizer.get_vocabulary()), WORD_EMBEDDING_DIM)
            if name == "char":
                return build_cnn(CHAR_SEQUENCE_LENGTH, len(suite.char_vectorizer.get_vocabulary()), CHAR_EMBEDDING_DIM)
            vocab_size = int(name.split("_")[1])
            return build_cnn(BPE_SEQUENCE_LENGTHS[vocab_size], suite.bpe_tokenizers[vocab_size].get_vocab_size(), BPE_EMBEDDING_DIM)

    elif backbone == "tcn":

        def builder(name: str):
            if name == "word":
                return build_tcn(WORD_SEQUENCE_LENGTH, len(suite.word_vectorizer.get_vocabulary()))
            if name == "char":
                return build_tcn(CHAR_SEQUENCE_LENGTH, len(suite.char_vectorizer.get_vocabulary()))
            vocab_size = int(name.split("_")[1])
            return build_tcn(BPE_SEQUENCE_LENGTHS[vocab_size], suite.bpe_tokenizers[vocab_size].get_vocab_size())

    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    return {name: (lambda n=name: builder(n)) for name in MODEL_ORDER}


def save_model_summary(model, model_name: str, run_id: str) -> str:
    summary_path = os.path.join(MODEL_SUMMARIES_DIR, f"{model_name}_{run_id}.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        model.summary(print_fn=lambda line: f.write(line + "\n"))
    return summary_path


# ---------------------------------------------------------------------------
# Training loop (ported from v1 cells 36–37, 41)
# ---------------------------------------------------------------------------


class PeakRAMMonitor:
    def __init__(self, sample_interval: float = 0.1):
        import psutil

        self.process = psutil.Process(os.getpid())
        self.sample_interval = sample_interval
        self.initial_bytes = 0
        self.peak_bytes = 0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self):
        while not self._stop_event.is_set():
            rss = self.process.memory_info().rss
            self.peak_bytes = max(self.peak_bytes, rss)
            time.sleep(self.sample_interval)

    def start(self):
        self.initial_bytes = self.process.memory_info().rss
        self.peak_bytes = self.initial_bytes
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        assert self._thread is not None
        self._thread.join()
        self.peak_bytes = max(self.peak_bytes, self.process.memory_info().rss)
        return {
            "initial_ram_mb": self.initial_bytes / 1024**2,
            "peak_ram_mb": self.peak_bytes / 1024**2,
            "ram_increase_mb": (self.peak_bytes - self.initial_bytes) / 1024**2,
        }


def enable_determinism() -> None:
    import tensorflow as tf

    tf.config.experimental.enable_op_determinism()


def train_model(model_builder, train_X, train_Y, val_X, val_Y, seed: int, batch_size: int = BATCH_SIZE):
    """Train one model with v1's exact recipe.

    ``train_X`` may be an array/tensor (then ``train_Y`` is required and
    ``batch_size`` applies) or a batch-yielding sequence object such as
    ``BPEDropoutTrainingSequence`` (then pass ``train_Y=None`` and
    ``batch_size=1`` — Keras forbids an explicit batch_size alongside a
    sequence; the sequence's own batching governs).
    """
    import gc

    import keras

    keras.backend.clear_session()
    gc.collect()
    keras.utils.set_random_seed(seed)

    model = model_builder()

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        min_delta=PATIENCE_MIN_DELTA,
        restore_best_weights=True,
    )

    fit_kwargs = dict(
        validation_data=(val_X, val_Y),
        epochs=EPOCHS,
        callbacks=[early_stopping],
        verbose=1,
    )
    if train_Y is None:  # batch-yielding sequence input
        assert batch_size == 1 or batch_size is None
        fit_kwargs["x"] = train_X
    else:
        fit_kwargs["x"] = train_X
        fit_kwargs["y"] = train_Y
        fit_kwargs["batch_size"] = batch_size

    ram_monitor = PeakRAMMonitor()
    ram_monitor.start()
    start_time = time.perf_counter()
    try:
        history = model.fit(**fit_kwargs)
    finally:
        training_time = time.perf_counter() - start_time
        ram_stats = ram_monitor.stop()

    stats = {
        "seed": seed,
        "epochs": len(history.history["loss"]),
        "restored_epoch": int(early_stopping.best_epoch + 1),
        "training_time_s": training_time,
        "training_time_per_epoch_s": training_time / len(history.history["loss"]),
        **ram_stats,
    }
    return model, history, stats


def run_fixed_comparison(
    backbone: str,
    suite: TokenizationSuite,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_sets: dict[int, pd.DataFrame],
    run_id: str,
    seeds: Sequence[int] = TRAINING_SEEDS,
) -> dict[str, pd.DataFrame]:
    """Full six-model fixed comparison for one backbone (v1 cell 41, modularized).

    Writes the same artifact families as v1 (metrics, resource_usage,
    predictions, histories, .keras models), namespaced by run_id. Returns the
    in-memory DataFrames so downstream v2 phases can reuse them.
    """
    import gc

    import keras

    builders = build_experiment_builders(backbone, suite)
    metric_rows: list[dict] = []
    resource_rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []

    train_Y = (train_df["Class Index"] - 1).to_numpy()
    val_Y = (val_df["Class Index"] - 1).to_numpy()

    train_X_cache = {
        name: suite.vectorize(name, train_df["Description"].to_numpy()) for name in MODEL_ORDER
    }
    val_X_cache = {
        name: suite.vectorize(name, val_df["Description"].to_numpy()) for name in MODEL_ORDER
    }

    for model_name in MODEL_ORDER:
        for seed in seeds:
            print(f"\n[{backbone}] Training {model_name}, seed={seed}")
            model, history, stats = train_model(
                model_builder=builders[model_name],
                train_X=train_X_cache[model_name],
                train_Y=train_Y,
                val_X=val_X_cache[model_name],
                val_Y=val_Y,
                seed=seed,
            )
            resource_rows.append(
                {
                    "model": model_name,
                    "seed": seed,
                    "n_parameters": model.count_params(),
                    "epochs_trained": stats["epochs"],
                    "restored_epoch": stats["restored_epoch"],
                    "training_time_s": stats["training_time_s"],
                    "training_time_per_epoch_s": stats["training_time_per_epoch_s"],
                    "initial_ram_mb": stats["initial_ram_mb"],
                    "peak_ram_mb": stats["peak_ram_mb"],
                    "ram_increase_mb": stats["ram_increase_mb"],
                }
            )
            history_df = pd.DataFrame(history.history)
            history_df.insert(0, "epoch", np.arange(1, len(history_df) + 1))
            history_df.to_csv(
                os.path.join(HISTORIES_DIR, f"{backbone}_{model_name}_seed{seed}_{run_id}.csv"),
                index=False,
            )

            evaluate_model(
                model=model,
                model_name=model_name,
                seed=seed,
                suite=suite,
                test_sets=test_sets,
                metric_rows=metric_rows,
                prediction_frames=prediction_frames,
            )

            pd.DataFrame(metric_rows).to_csv(
                os.path.join(RESULTS_DIR, f"{backbone}_metrics_{run_id}.csv"), index=False
            )
            pd.DataFrame(resource_rows).to_csv(
                os.path.join(RESULTS_DIR, f"{backbone}_resource_usage_{run_id}.csv"), index=False
            )
            pd.concat(prediction_frames, ignore_index=True).to_csv(
                os.path.join(RESULTS_DIR, f"{backbone}_predictions_{run_id}.csv"), index=False
            )
            model.save(os.path.join(MODELS_DIR, f"{backbone}_{model_name}_seed{seed}_{run_id}.keras"))

            del model, history
            keras.backend.clear_session()
            gc.collect()

    return {
        "metrics": pd.DataFrame(metric_rows),
        "resource_usage": pd.DataFrame(resource_rows),
        "predictions": pd.concat(prediction_frames, ignore_index=True),
    }


# ---------------------------------------------------------------------------
# Evaluation (ported from v1 cell 39)
# ---------------------------------------------------------------------------


def evaluate_model(
    model,
    model_name: str,
    seed: int,
    suite: TokenizationSuite,
    test_sets: dict[int, pd.DataFrame],
    metric_rows: list | None = None,
    prediction_frames: list | None = None,
    system_label: str | None = None,
):
    """Evaluate one trained model over the nested test sets (v1 cell 39).

    ``system_label`` lets routed/baseline systems reuse this path under a
    custom name in the predictions frame (defaults to model_name).
    """
    from sklearn.metrics import (
        accuracy_score,
        log_loss,
        matthews_corrcoef,
        precision_recall_fscore_support,
    )

    metric_rows = [] if metric_rows is None else metric_rows
    prediction_frames = [] if prediction_frames is None else prediction_frames
    label = system_label or model_name

    first_test_set = next(iter(test_sets.values()))
    warmup_X = suite.vectorize(model_name, first_test_set["Description"].iloc[:BATCH_SIZE].to_numpy())
    _ = model.predict(warmup_X, batch_size=BATCH_SIZE, verbose=0)

    for corruption_level, test_set in test_sets.items():
        test_X = suite.vectorize(model_name, test_set["Description"].to_numpy())
        y_true = (test_set["Class Index"].to_numpy() - 1).astype(int)

        start_time = time.perf_counter()
        y_prob = model.predict(test_X, batch_size=BATCH_SIZE, verbose=0)
        inference_time = time.perf_counter() - start_time

        y_pred = np.argmax(y_prob, axis=1)
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )

        metric_rows.append(
            {
                "model": label,
                "seed": seed,
                "corruption_level": corruption_level,
                "n_samples": len(y_true),
                "accuracy": accuracy_score(y_true, y_pred),
                "precision_macro": precision_macro,
                "f1_macro": f1_macro,
                "mcc": matthews_corrcoef(y_true, y_pred),
                "log_loss": log_loss(y_true, y_prob),
                "inference_time_s": inference_time,
                "inference_time_per_sample_ms": inference_time / len(y_true) * 1000,
            }
        )

        prediction_df = pd.DataFrame(
            {
                "sample_id": test_set.index.to_numpy(),
                "model": label,     # v1-compatible key
                "system": label,    # v2 key (routers/baselines join the same frame)
                "seed": seed,
                "corruption_level": corruption_level,
                "true_class": y_true + 1,
                "predicted_class": y_pred + 1,
                "correct": (y_true == y_pred).astype(int),
            }
        )
        for class_id in range(NUM_CLASSES):
            prediction_df[f"prob_class_{class_id + 1}"] = y_prob[:, class_id]
        prediction_frames.append(prediction_df)

    return metric_rows, prediction_frames


# ---------------------------------------------------------------------------
# Statistics (ported/generalized from v1 cells 52–74)
# ---------------------------------------------------------------------------


def mcnemar_holm_table(
    predictions_df: pd.DataFrame,
    system_order: Sequence[str],
    seeds: Sequence[int],
    corruption_levels: Sequence[int],
    csv_name: str | None = None,
) -> pd.DataFrame:
    """Paired McNemar tests with Holm correction across named systems.

    ``predictions_df`` needs columns: sample_id, system, seed,
    corruption_level, correct. Generalized from v1 cell 62 (which keyed on
    'model'); v2 systems (routers, BPE dropout) join the same framework.
    """
    from statsmodels.stats.contingency_tables import mcnemar
    from statsmodels.stats.multitest import multipletests

    pairs = list(combinations(system_order, 2))
    mcnemar_rows = []
    for seed in seeds:
        for corruption_level in corruption_levels:
            paired_df = predictions_df[
                (predictions_df["seed"] == seed)
                & (predictions_df["corruption_level"] == corruption_level)
            ].pivot(index="sample_id", columns="system", values="correct")
            missing = [s for s in system_order if s not in paired_df.columns]
            if missing:
                raise KeyError(f"Systems absent from predictions frame: {missing}")
            for system_a, system_b in pairs:
                a_correct = paired_df[system_a].astype(bool)
                b_correct = paired_df[system_b].astype(bool)
                both_correct = (a_correct & b_correct).sum()
                a_only = (a_correct & ~b_correct).sum()
                b_only = (~a_correct & b_correct).sum()
                both_wrong = (~a_correct & ~b_correct).sum()
                result = mcnemar([[both_correct, a_only], [b_only, both_wrong]], exact=True)
                mcnemar_rows.append(
                    {
                        "seed": seed,
                        "corruption_level": corruption_level,
                        "model_a": system_a,
                        "model_b": system_b,
                        "n_samples": len(paired_df),
                        "both_correct": both_correct,
                        "model_a_only_correct": a_only,
                        "model_b_only_correct": b_only,
                        "both_wrong": both_wrong,
                        "accuracy_delta_a_minus_b": (a_only - b_only) / len(paired_df),
                        "p_raw": result.pvalue,
                    }
                )

    mcnemar_df = pd.DataFrame(mcnemar_rows)
    reject, p_holm, _, _ = multipletests(mcnemar_df["p_raw"], alpha=0.05, method="holm")
    mcnemar_df["p_holm"] = p_holm
    mcnemar_df["significant_holm_0.05"] = reject
    if csv_name:
        mcnemar_df.to_csv(os.path.join(RESULTS_DIR, csv_name), index=False)
    return mcnemar_df


def sign_flip_permutation_pvalue(paired_deltas: np.ndarray) -> float:
    """Two-sided exact sign-flip permutation test (v1 cell 64 logic)."""
    paired_deltas = np.asarray(paired_deltas, dtype=float)
    n_seeds = len(paired_deltas)
    n_permutations = 2**n_seeds
    sign_bits = (np.arange(n_permutations)[:, None] >> np.arange(n_seeds)[None, :]) & 1
    signs = np.where(sign_bits == 0, -1.0, 1.0)
    permuted_mean_deltas = (signs * paired_deltas).mean(axis=1)
    observed_mean_delta = paired_deltas.mean()
    return float(np.mean(np.abs(permuted_mean_deltas) >= np.abs(observed_mean_delta) - 1e-12))


def relative_f1_degradation_analysis(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Per-(model, seed, level) absolute/relative degradation (v1 cell 48)."""
    analysis_df = metrics_df.melt(
        id_vars=["model", "seed", "corruption_level"],
        value_vars=list(METRICS),
        var_name="metric",
        value_name="performance",
    )
    clean_df = analysis_df.loc[
        analysis_df["corruption_level"].eq(0),
        ["model", "seed", "metric", "performance"],
    ].rename(columns={"performance": "clean_performance"})
    analysis_df = analysis_df.merge(clean_df, on=["model", "seed", "metric"], how="left")

    analysis_df["abs_degradation"] = np.where(
        analysis_df["metric"].eq("log_loss"),
        analysis_df["performance"] - analysis_df["clean_performance"],
        analysis_df["clean_performance"] - analysis_df["performance"],
    )
    analysis_df["rel_degradation"] = np.nan
    higher_is_better = analysis_df["metric"].isin(["accuracy", "f1_macro"])
    analysis_df.loc[higher_is_better, "rel_degradation"] = (
        analysis_df.loc[higher_is_better, "clean_performance"]
        - analysis_df.loc[higher_is_better, "performance"]
    ) / analysis_df.loc[higher_is_better, "clean_performance"]
    log_loss_mask = analysis_df["metric"].eq("log_loss")
    analysis_df.loc[log_loss_mask, "rel_degradation"] = (
        analysis_df.loc[log_loss_mask, "performance"]
        - analysis_df.loc[log_loss_mask, "clean_performance"]
    ) / analysis_df.loc[log_loss_mask, "clean_performance"]
    return analysis_df


def mean_relative_f1_degradation_by_seed(
    metrics_df: pd.DataFrame, model_order: Sequence[str]
) -> pd.DataFrame:
    """Mean relative Macro-F1 degradation over corrupted levels, per seed (v1 cell 52)."""
    analysis_df = relative_f1_degradation_analysis(metrics_df)
    mean_df = (
        analysis_df.loc[
            analysis_df["metric"].eq("f1_macro") & analysis_df["corruption_level"].gt(0)
        ]
        .groupby(["model", "seed"], as_index=False)
        .agg(
            mean_abs_degradation=("abs_degradation", "mean"),
            mean_rel_degradation=("rel_degradation", "mean"),
        )
    )
    wide = mean_df.pivot(index="seed", columns="model", values="mean_rel_degradation")
    return wide.dropna(subset=list(model_order)).reset_index()


def paired_seed_comparison(
    by_seed_df: pd.DataFrame,
    model_pairs: Sequence[tuple[str, str]],
    family_name: str,
) -> pd.DataFrame:
    """Exact sign-flip permutation tests + Holm correction within a family."""
    from statsmodels.stats.multitest import multipletests

    rows = []
    for model_a, model_b in model_pairs:
        deltas = (by_seed_df[model_a] - by_seed_df[model_b]).to_numpy()
        rows.append(
            {
                "test": "Exact paired sign-flip permutation test",
                "family": family_name,
                "metric": "Mean relative Macro-F1 degradation",
                "model_a": model_a,
                "model_b": model_b,
                "n_seeds": len(deltas),
                "observed_mean_delta_a_minus_b": deltas.mean(),
                "n_exact_permutations": 2 ** len(deltas),
                "p_raw": sign_flip_permutation_pvalue(deltas),
            }
        )
    family_df = pd.DataFrame(rows)
    reject, p_holm, _, _ = multipletests(family_df["p_raw"], alpha=0.05, method="holm")
    family_df["p_holm_family"] = p_holm
    family_df["significant_holm_0.05"] = reject
    return family_df


# ---------------------------------------------------------------------------
# Predictions alignment, oracle target, and routing evaluation (plan §§7–8)
# ---------------------------------------------------------------------------


def build_expert_probability_table(
    predictions_long: pd.DataFrame,
    experts: Sequence[str] = ROUTER_EXPERTS,
) -> pd.DataFrame:
    """One row per (sample_id, seed, corruption_level): true_class + probs per expert.

    Input: long predictions frame as produced by :func:`evaluate_model` /
    v1's pipeline, with columns sample_id, model, seed, corruption_level,
    true_class, prob_class_1..4. Experts must share identical keys (guaranteed
    by construction: everyone evaluates the same nested test sets).
    """
    indexed_frames = []
    for expert in experts:
        sub = predictions_long[predictions_long["model"].eq(expert)]
        if sub.empty:
            raise KeyError(f"No predictions found for expert '{expert}'.")
        prob_cols = [f"prob_class_{c}" for c in range(1, NUM_CLASSES + 1)]
        sub = sub[["sample_id", "seed", "corruption_level", "true_class", *prob_cols]].copy()
        sub = sub.rename(columns={c: f"{c}__{expert}" for c in prob_cols})
        sub = sub.rename(columns={"true_class": f"true_class__{expert}"})
        sub = sub.set_index(["sample_id", "seed", "corruption_level"])
        indexed_frames.append(sub)

    merged = pd.concat(indexed_frames, axis=1, join="inner")

    # Alignment sanity: every expert must agree on true_class per key.
    reference = merged[f"true_class__{experts[0]}"]
    for expert in experts[1:]:
        col = f"true_class__{expert}"
        if not (merged[col] == reference).all():
            raise AssertionError(f"Expert '{expert}' disagrees on true_class — misaligned keys.")

    merged = merged.rename(columns={f"true_class__{experts[0]}": "true_class"})
    for expert in experts[1:]:
        merged = merged.drop(columns=[f"true_class__{expert}"])
    return merged.reset_index()


def expert_losses(prob_table: pd.DataFrame, experts: Sequence[str] = ROUTER_EXPERTS) -> np.ndarray:
    """Per-example cross-entropy matrix, shape (n_examples, n_experts)."""
    y_true_idx = prob_table["true_class"].to_numpy() - 1
    take = np.arange(len(prob_table))
    columns = []
    for expert in experts:
        probs = prob_table[[f"prob_class_{c}__{expert}" for c in range(1, NUM_CLASSES + 1)]].to_numpy()
        columns.append(-np.log(np.clip(probs[take, y_true_idx], 1e-12, None)))
    return np.column_stack(columns)


def oracle_routing_target(
    prob_table: pd.DataFrame,
    cost_per_sample_ms: dict[str, float],
    lambdas: Sequence[float] = LAMBDA_SWEEP,
    experts: Sequence[str] = ROUTER_EXPERTS,
) -> pd.DataFrame:
    """Per-example oracle expert for each lambda (plan §7).

    Oracle: argmin_m [ cross_entropy(y_i, p_im) + lambda * C_m ], where C_m is
    the measured per-sample CPU inference cost. Sweeping lambda traces the
    Pareto frontier instead of committing to one arbitrary trade-off.
    """
    losses = expert_losses(prob_table, experts)  # (n, E)
    costs = np.array([cost_per_sample_ms[e] for e in experts])
    key_cols = ["sample_id", "seed", "corruption_level"]

    frames = []
    for lam in lambdas:
        objective = losses + lam * costs[None, :]
        choices = np.argmin(objective, axis=1)
        best_loss_only = np.argmin(losses, axis=1)
        take = np.arange(len(prob_table))
        frames.append(
            pd.DataFrame(
                {
                    **{col: prob_table[col].to_numpy() for col in key_cols},
                    "lambda": lam,
                    "oracle_expert": [experts[i] for i in choices],
                    "oracle_loss": objective.min(axis=1),  # includes lambda*cost (selection criterion)
                    "best_expert_loss_only": [experts[i] for i in best_loss_only],
                    "best_loss_only_value": losses[take, best_loss_only],  # pure CE, no cost term
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def route_probabilities(
    prob_table: pd.DataFrame,
    routing: pd.Series | np.ndarray | list[str],
    experts: Sequence[str] = ROUTER_EXPERTS,
) -> pd.DataFrame:
    """Assemble per-example outcomes under a given routing decision.

    Returns one row per example: keys, true/predicted class, correct flag,
    per-example log loss under the chosen expert, chosen expert, and active
    sequence length (the compute proxy actually paid at inference).
    """
    routing = np.asarray(routing, dtype=object)
    unknown = set(routing) - set(experts)
    if unknown:
        raise ValueError(f"Routing decisions outside the expert set: {unknown}")

    y_true_idx = prob_table["true_class"].to_numpy() - 1
    take = np.arange(len(prob_table))
    chosen_probs = np.zeros((len(prob_table), NUM_CLASSES))
    for expert in experts:
        mask = routing == expert
        if not mask.any():
            continue
        chosen_probs[mask] = prob_table.loc[mask, [f"prob_class_{c}__{expert}" for c in range(1, NUM_CLASSES + 1)]].to_numpy()

    predicted = chosen_probs.argmax(axis=1) + 1
    out = prob_table[["sample_id", "seed", "corruption_level"]].copy()
    out["true_class"] = prob_table["true_class"].to_numpy()
    out["chosen_expert"] = routing
    out["predicted_class"] = predicted
    out["correct"] = (predicted == out["true_class"]).astype(int)
    out["log_loss_contrib"] = -np.log(np.clip(chosen_probs[take, y_true_idx], 1e-12, None))
    out["active_sequence_length"] = [EXPERT_SEQUENCE_LENGTHS[e] for e in routing]
    return out


def summarize_routed_run(routed_df: pd.DataFrame) -> dict:
    """Aggregate one routed evaluation into frontier coordinates (plan §8)."""
    from sklearn.metrics import f1_score

    return {
        "macro_f1": float(
            f1_score(routed_df["true_class"], routed_df["predicted_class"], average="macro", zero_division=0)
        ),
        "mean_log_loss": float(routed_df["log_loss_contrib"].mean()),
        "mean_active_sequence_length": float(routed_df["active_sequence_length"].mean()),
        **{
            f"fraction_{expert}": float(routed_df["chosen_expert"].eq(expert).mean())
            for expert in ROUTER_EXPERTS
        },
    }


def routing_regret(routed_df: pd.DataFrame, oracle_target_df: pd.DataFrame, lam: float) -> pd.DataFrame:
    """Per-example regret of a router against the lambda-oracle (plan §8).

    Regret = pure cross-entropy(chosen) − pure cross-entropy(oracle choice at
    this lambda). The cost term λ·C_m is a *selection* criterion only — it
    decides which expert the oracle picks, but regret itself is measured on
    predictive loss so that routers choosing cheaper experts are not penalized
    twice.
    """
    oracle_at_lam = oracle_target_df[oracle_target_df["lambda"].eq(lam)]
    merged = routed_df.merge(
        oracle_at_lam[["sample_id", "seed", "corruption_level", "oracle_expert", "best_expert_loss_only", "best_loss_only_value"]],
        on=["sample_id", "seed", "corruption_level"],
        how="left",
        validate="one_to_one",
    )
    merged["regret"] = merged["log_loss_contrib"] - merged["best_loss_only_value"]
    merged["is_optimal_choice"] = merged["chosen_expert"].eq(merged["best_expert_loss_only"])
    return merged


# ---------------------------------------------------------------------------
# Routers (plan §6: logistic regression, shallow tree, GBT if needed)
# ---------------------------------------------------------------------------

ROUTER_FEATURES = [
    "text_length",
    "word_count",
    "character_count",
    "word_oov_fraction",
    "digit_fraction",
    "punct_fraction",
    "bpe_token_count",
    "bpe_tokens_per_word",
    "bpe_tokens_per_character",
    "fraction_words_split_2plus",
    "fraction_words_split_3plus",
    "max_pieces_per_word",
    "mean_pieces_per_word",
    "normalized_sequence_length",
]


def fragmentation_threshold_router(
    features_df: pd.DataFrame,
    thresholds: Sequence[float],
    experts: Sequence[str] = ("word", "char"),
) -> pd.Series:
    """Simple interpretable router: escalating fragmentation => finer expert.

    Routes to experts[0] below thresholds[0], experts[1] between
    thresholds[0] and thresholds[1], and experts[-1] at or above
    thresholds[-1]. Sweep the thresholds to trace its frontier point(s);
    the default single-threshold variant is thresholds=(t,) with two experts.
    """
    signal = features_df["fraction_words_split_2plus"].to_numpy()
    if len(thresholds) != len(experts) - 1:
        raise ValueError("Provide exactly len(experts)-1 ascending thresholds.")
    if list(thresholds) != sorted(thresholds):
        raise ValueError("Thresholds must be ascending.")
    decisions = np.full(len(features_df), experts[-1], dtype=object)
    lower_bound = -np.inf
    for expert, threshold in zip(experts, thresholds):
        band = (signal > lower_bound) & (signal < threshold)
        decisions[band] = expert
        lower_bound = threshold
    return pd.Series(decisions, index=features_df.index)


def fit_learned_router(
    features_df: pd.DataFrame,
    target_labels: Sequence[str],
    router_type: str = "logistic_regression",
    random_state: int = 0,
):
    """Train a lightweight router mapping instability features -> expert.

    router_type: 'logistic_regression' | 'decision_tree' | 'gradient_boosting'.
    No neural router at this stage (plan §6).
    """
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier

    X = features_df[ROUTER_FEATURES].to_numpy()
    y = np.asarray(target_labels, dtype=object)

    if router_type == "logistic_regression":
        estimator = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, random_state=random_state)),
            ]
        )
    elif router_type == "decision_tree":
        estimator = DecisionTreeClassifier(
            max_depth=4, min_samples_leaf=20, random_state=random_state
        )
    elif router_type == "gradient_boosting":
        estimator = GradientBoostingClassifier(random_state=random_state)
    elif router_type == "calibrated_logistic":
        base = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, random_state=random_state)),
            ]
        )
        estimator = CalibratedClassifierCV(base, method="sigmoid", cv=5)
    else:
        raise ValueError(f"Unknown router_type: {router_type}")

    estimator.fit(X, y)
    return estimator


def router_predict(estimator, features_df: pd.DataFrame) -> np.ndarray:
    return np.asarray(estimator.predict(features_df[ROUTER_FEATURES].to_numpy()), dtype=object)


def router_expected_loss_decision(
    calibrated_estimators: dict[str, object],
    features_df: pd.DataFrame,
    cost_per_sample_ms: dict[str, float],
    lam: float,
    experts: Sequence[str] = ROUTER_EXPERTS,
) -> np.ndarray:
    """Decision-theoretic routing from per-expert calibrated probabilities.

    For each expert model m we have a classifier estimating P(expert m is the
    right choice). Combining those with measured costs gives an expected-cost
    decision rule. Provided for completeness; the primary learned router is
    the direct label-based one in :func:`fit_learned_router`.
    """
    prob_matrix = np.column_stack(
        [calibrated_estimators[e].predict_proba(features_df[ROUTER_FEATURES].to_numpy())[:, 1] for e in experts]
    )
    prob_matrix = prob_matrix / prob_matrix.sum(axis=1, keepdims=True)
    costs = np.array([cost_per_sample_ms[e] for e in experts])
    expected_cost = (1.0 - prob_matrix) + lam * costs[None, :]
    return np.asarray([experts[i] for i in expected_cost.argmin(axis=1)], dtype=object)


# ---------------------------------------------------------------------------
# Efficiency accounting (new — plan §14)
# ---------------------------------------------------------------------------


def measure_cpu_latency_ms_per_sample(
    model,
    suite: TokenizationSuite,
    model_name: str,
    sample_texts: Sequence[str],
    n_repeat: int = 5,
) -> float:
    """Median single-sample CPU latency in milliseconds (batch size 1).

    Routing compares per-input costs, so batch-of-one latency is the relevant
    quantity — stricter than v1's throughput-style timing. Uses a small
    calibration sample; deterministic ops recommended before measuring.
    """
    sample_texts = list(sample_texts)[:64]
    timings = []
    for text in sample_texts:
        X = suite.vectorize(model_name, [text])
        for _ in range(2):  # warm-up per shape (identical shapes here, cheap)
            _ = model.predict(X, batch_size=1, verbose=0)
        start = time.perf_counter()
        for _ in range(n_repeat):
            _ = model.predict(X, batch_size=1, verbose=0)
        timings.append((time.perf_counter() - start) / n_repeat * 1000)
    return float(np.median(timings))


def resident_model_size_mb(path: str) -> float:
    return os.path.getsize(path) / 1024**2


def efficiency_table(
    resource_usage_df: pd.DataFrame,
    latency_ms: dict[tuple[str, int], float],
    model_sizes_mb: dict[tuple[str, int], float] | None = None,
) -> pd.DataFrame:
    """Combine v1-style resource rows with new per-sample latency (plan §14)."""
    df = resource_usage_df.copy()
    df["latency_ms_per_sample"] = [latency_ms.get((m, s), np.nan) for m, s in zip(df["model"], df["seed"])]
    if model_sizes_mb:
        df["resident_size_mb"] = [model_sizes_mb.get((m, s), np.nan) for m, s in zip(df["model"], df["seed"])]
    return df


# ---------------------------------------------------------------------------
# Failure analysis (new — plan §15)
# ---------------------------------------------------------------------------

FAILURE_MODES = {
    "missed_fragmentation": "fragmentation features miss corruption",
    "rare_word_lookalike": "normal rare words resemble corrupted words",
    "false_route_clean_text": "router chose char but word was sufficient",
    "under_route_corrupted": "router chose word but char was much better",
    "char_model_failed": "character model itself failed",
}


def label_failure_modes(per_example_df: pd.DataFrame) -> pd.DataFrame:
    """Heuristic failure-mode labels where routing regret is high.

    Expected input: per-example rows with columns sample_id, seed,
    corruption_level, chosen_expert, correct_chosen, correct_word,
    correct_char, regret, plus instability features merged on the keys.
    Two target cases from plan §15:
      * router chose word but character much better;
      * router chose character but word sufficient.
    """
    df = per_example_df.copy()

    def classify(row):
        if row["chosen_expert"] == "char" and row.get("correct_word", 0) == 1:
            return FAILURE_MODES["false_route_clean_text"]
        if row["chosen_expert"] == "word" and row.get("correct_word", 0) == 0 and row.get("correct_char", 0) == 1:
            if row.get("fraction_words_split_2plus", 0.0) < 0.1:
                return FAILURE_MODES["missed_fragmentation"]
            if row.get("word_oov_fraction", 0.0) > 0.5 and row.get("word_oov_fraction", 0.0) == row.get("_oov_baseline", row.get("word_oov_fraction", 0.0)):
                return FAILURE_MODES["rare_word_lookalike"]
            return FAILURE_MODES["under_route_corrupted"]
        if row["chosen_expert"] == "char" and row.get("correct_char", 0) == 0:
            return FAILURE_MODES["char_model_failed"]
        return ""

    df["failure_mode"] = df.apply(classify, axis=1)
    return df


# ---------------------------------------------------------------------------
# Misc shared utilities
# ---------------------------------------------------------------------------


def save_json(payload: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def save_figure(fig, figure_name: str, run_id: str) -> str:
    figure_path = os.path.join(FIGURES_DIR, f"{figure_name}_{run_id}.png")
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    return figure_path


def config_snapshot(run_id: str, backbone: str, extra: dict | None = None) -> dict:
    config = {
        "run_id": run_id,
        "backbone": backbone,
        "training_seeds": list(TRAINING_SEEDS),
        "corruption_seed": CORRUPTION_SEED,
        "split_seed": SPLIT_SEED,
        "double_corruption_percent": DOUBLE_CORRUPTION_PERCENT,
        "corruption_levels": list(CORRUPTION_LEVELS),
        "standardization": STANDARDIZATION,
        "word_max_tokens": WORD_MAX_TOKENS,
        "word_sequence_length": WORD_SEQUENCE_LENGTH,
        "word_embedding_dim": WORD_EMBEDDING_DIM,
        "bpe_vocab_sizes": list(BPE_VOCAB_SIZES),
        "bpe_min_frequency": BPE_MIN_FREQUENCY,
        "bpe_sequence_lengths": BPE_SEQUENCE_LENGTHS,
        "bpe_embedding_dim": BPE_EMBEDDING_DIM,
        "char_max_tokens": CHAR_MAX_TOKENS,
        "char_sequence_length": CHAR_SEQUENCE_LENGTH,
        "char_embedding_dim": CHAR_EMBEDDING_DIM,
        "conv_filters": CONV_FILTERS,
        "kernel_size": KERNEL_SIZE,
        "dense_units": DENSE_UNITS,
        "hidden_activation": HIDDEN_ACTIVATION,
        "output_activation": OUTPUT_ACTIVATION,
        "dropout_rate": DROPOUT_RATE,
        "num_classes": NUM_CLASSES,
        "learning_rate": LEARNING_RATE,
        "alpha": ALPHA_LR,
        "decay_steps": DECAY_STEPS,
        "epochs": EPOCHS,
        "patience": PATIENCE,
        "batch_size": BATCH_SIZE,
        "router_experts": list(ROUTER_EXPERTS),
        "lambda_sweep": list(LAMBDA_SWEEP),
        "tcn_config": {**TCN_CONFIG, "dilations": list(TCN_CONFIG["dilations"])},
        "bpe_dropout_prob": BPE_DROPOUT_PROB,
        "router_features": list(ROUTER_FEATURES),
    }
    if extra:
        config.update(extra)
    return config
