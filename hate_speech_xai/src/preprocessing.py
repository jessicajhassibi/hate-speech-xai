from collections import Counter
from typing import Tuple

from datasets import Dataset
from transformers import AutoTokenizer

import numpy as np

from hate_speech_xai.src.config import MODEL_NAME, MAX_LENGTH, TRUNCATION, PADDING
from hate_speech_xai.src.load_hatexplain import load_hatexplain_dataset

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

def tokenize_text(text: str):
	"""BERT uses subword tokenization, so we reconstruct the text before tokenizing."""
	encoding = tokenizer(
		text,
		truncation=TRUNCATION,
		padding=PADDING,
		max_length=MAX_LENGTH
	)
	return encoding


def preprocess_post(post: dict):
	# Majority label
	label = get_majority_label(post["annotators"]["label"])

	# Aggregate rationales
	rationales = aggregate_rationales(post["rationales"])

	# Reconstruct text
	text = reconstruct_post(post["post_tokens"])

	# Tokenize
	encoding = tokenize_text(text)

	# Align rationales with subwords
	word_ids = encoding.word_ids()
	token_rationale = []

	for word_id in word_ids:
		if word_id is None or word_id >= len(rationales):
			token_rationale.append(0)
		else:
			token_rationale.append(int(rationales[word_id]))

	encoding["label"] = label
	encoding["rationale_mask"] = token_rationale

	return encoding


def preprocess_dataset() -> Tuple[Dataset, Dataset, Dataset]:
	train, val, test = load_hatexplain_dataset()

	train = train.map(preprocess_post)
	val = val.map(preprocess_post)
	test = test.map(preprocess_post)

	return train, val, test