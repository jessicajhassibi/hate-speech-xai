from collections import Counter
from typing import Tuple

from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer

import numpy as np

from hate_speech_xai.config import MODEL_NAME, MAX_LENGTH, TRUNCATION, PADDING, PREPROCESSED_DATA_DIR

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def get_majority_label(annotators_labels: list):
	"""We agree on the most common label the annotators assigned to the post as the
	"ground truth" label for this post.
	If all labels are different, the first one in the list is chosen as the majority label.
	"""
	return Counter(annotators_labels).most_common(1)[0][0]

def aggregate_rationales(annotators_rationales: list):
	"""We take the union of all rationales across annotators as the "ground truth" rationale for this post.
	Assuming that if any annotator marked a token, it is really potentially important.
	"""
	if not annotators_rationales or len(annotators_rationales) == 0:
		return np.array([])

	# Find the maximum length among all rationales (in case annotators marked different lengths)
	max_length = max(len(rat) for rat in annotators_rationales)

	# Pad all rationales to the same length with zeros
	padded_rationales = []
	for rat in annotators_rationales:
		padded = list(rat) + [0] * (max_length - len(rat))
		padded_rationales.append(padded)

	return np.array(padded_rationales).max(axis=0)

def reconstruct_post(tokens: list):
	"""Reconstruct the original post text from the token list."""
	return " ".join(tokens)


def preprocess_post(post: dict):
	"""Annotation-level preprocessing: majority label, aggregated rationales, joined text."""
	label = get_majority_label(post["annotators"]["label"])
	rationale = aggregate_rationales(post["rationales"])
	text = reconstruct_post(post["post_tokens"])

	return {
		"text": text,
		"label": label,
		"rationale": rationale.tolist() if len(rationale) > 0 else [],
	}


def tokenize_post(post: dict):
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
			token_rationale.append(0)
		else:
			token_rationale.append(int(rationale[word_id]))

	encoding["labels"] = post["label"]
	encoding["rationale_mask"] = token_rationale

	return encoding


_RAW_COLUMNS_TO_REMOVE = ["id", "annotators", "rationales", "post_tokens"]
_PREPROCESSED_COLUMNS_TO_REMOVE = ["text", "rationale"]


def preprocess_dataset(train: Dataset, val: Dataset, test: Dataset) -> Tuple[Dataset, Dataset, Dataset]:
	"""Annotation-level preprocessing on raw dataset splits."""
	train = train.map(preprocess_post, remove_columns=_RAW_COLUMNS_TO_REMOVE)
	val = val.map(preprocess_post, remove_columns=_RAW_COLUMNS_TO_REMOVE)
	test = test.map(preprocess_post, remove_columns=_RAW_COLUMNS_TO_REMOVE)

	return train, val, test


def tokenize_dataset(train: Dataset, val: Dataset) -> Tuple[Dataset, Dataset]:
	"""Tokenize preprocessed dataset for training."""
	train = train.map(tokenize_post, remove_columns=_PREPROCESSED_COLUMNS_TO_REMOVE)
	val = val.map(tokenize_post, remove_columns=_PREPROCESSED_COLUMNS_TO_REMOVE)

	return train, val


def load_preprocessed_dataset() -> Tuple[Dataset, Dataset, Dataset]:
	"""Load the saved preprocessed dataset splits."""
	if not PREPROCESSED_DATA_DIR.exists():
		raise FileNotFoundError(
			f"Preprocessed dataset not found at {PREPROCESSED_DATA_DIR}. "
			"Run run_preprocessing.py first."
		)

	train = load_from_disk(str(PREPROCESSED_DATA_DIR / "train"))
	val = load_from_disk(str(PREPROCESSED_DATA_DIR / "val"))
	test = load_from_disk(str(PREPROCESSED_DATA_DIR / "test"))

	return train, val, test