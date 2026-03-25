from collections import Counter
from typing import Tuple

from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer

import numpy as np

from hate_speech_xai.config import MODEL_NAME, MAX_LENGTH, TRUNCATION, PADDING, PREPROCESSED_DATA_DIR

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def get_majority_label(annotators_labels: list) -> int:
	"""We agree on the most common label the annotators assigned to the post as the
	"ground truth" label for this post.
	If all labels are different, the first one in the list is chosen as the majority label.
	"""
	return Counter(annotators_labels).most_common(1)[0][0]

def aggregate_rationales(annotators_rationales: list, label: int, temperature: float = 5.0) -> np.ndarray:
	"""Aggregate annotator rationales for one post into a ground truth attention distribution.
	Following Mathew et al. (2021):
	- For hate/offensive posts: average the rationale vectors, then apply temperature-scaled softmax
	- For normal posts: use a uniform distribution (1/sentence_length)
	"""
	if not annotators_rationales or len(annotators_rationales) == 0:
		return np.array([])

	# We have to pad rationales to the same length in case of mismatched lengths (present in the data!)
	max_length = max(len(rat) for rat in annotators_rationales)
	padded = [list(rat) + [0] * (max_length - len(rat)) for rat in annotators_rationales]

	# Normal posts get uniform attention
	if label == 1:
		return np.full(max_length, 1.0 / max_length)

	avg_rationale = np.array(padded, dtype=float).mean(axis=0)

	# Temperature-scaled softmax
	scaled = avg_rationale * temperature # different from the authors, we use a static temperature
	exp = np.exp(scaled - np.max(scaled))  # subtract max for numerical stability
	return exp / exp.sum()

def get_post_as_str(tokens: list) -> str:
	return " ".join(tokens)

def preprocess_post(post: dict) -> dict:
	"""Performs preprocessing steps: majority label, aggregated rationales, joined text."""
	label = get_majority_label(post["annotators"]["label"])
	rationale = aggregate_rationales(post["rationales"], label)
	text = get_post_as_str(post["post_tokens"])

	return {
		"text": text,
		"label": label,
		"rationale": rationale.tolist() if len(rationale) > 0 else [],
	}


def tokenize_post(post: dict) -> dict:
	"""Tokenize preprocessed post and align rationales with subword tokens."""
	encoding = tokenizer(
		post["text"],
		truncation=TRUNCATION,
		padding=PADDING,
		max_length=MAX_LENGTH
	)

	rationale = post["rationale"]
	word_ids = encoding.word_ids()
	token_rationale = []

	for word_id in word_ids:
		if word_id is None or word_id >= len(rationale):
			token_rationale.append(0.0)
		else:
			token_rationale.append(float(rationale[word_id]))

	encoding["labels"] = post["label"]
	encoding["rationale_mask"] = token_rationale

	return encoding


_RAW_COLUMNS_TO_REMOVE = ["id", "annotators", "rationales", "post_tokens"]
_PREPROCESSED_COLUMNS_TO_REMOVE = ["text", "rationale"]


def preprocess_dataset(train: Dataset, val: Dataset, test: Dataset) -> Tuple[Dataset, Dataset, Dataset]:
	if PREPROCESSED_DATA_DIR.exists():
		print("Loading already existing preprocessed dataset from disk...")
		return load_preprocessed_dataset()

	train = train.map(preprocess_post, remove_columns=_RAW_COLUMNS_TO_REMOVE)
	val = val.map(preprocess_post, remove_columns=_RAW_COLUMNS_TO_REMOVE)
	test = test.map(preprocess_post, remove_columns=_RAW_COLUMNS_TO_REMOVE)

	PREPROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
	train.save_to_disk(str(PREPROCESSED_DATA_DIR / "train"))
	val.save_to_disk(str(PREPROCESSED_DATA_DIR / "val"))
	test.save_to_disk(str(PREPROCESSED_DATA_DIR / "test"))

	return train, val, test


def tokenize_dataset(train: Dataset, val: Dataset) -> Tuple[Dataset, Dataset]:
	"""Tokenizes preprocessed dataset for training."""
	train = train.map(tokenize_post, remove_columns=_PREPROCESSED_COLUMNS_TO_REMOVE)
	val = val.map(tokenize_post, remove_columns=_PREPROCESSED_COLUMNS_TO_REMOVE)

	return train, val


def load_preprocessed_dataset() -> Tuple[Dataset, Dataset, Dataset]:
	if not PREPROCESSED_DATA_DIR.exists():
		raise FileNotFoundError(
			f"Preprocessed dataset not found at {PREPROCESSED_DATA_DIR}. "
			"Run run_preprocessing.py first."
		)

	train = load_from_disk(str(PREPROCESSED_DATA_DIR / "train"))
	val = load_from_disk(str(PREPROCESSED_DATA_DIR / "val"))
	test = load_from_disk(str(PREPROCESSED_DATA_DIR / "test"))

	return train, val, test