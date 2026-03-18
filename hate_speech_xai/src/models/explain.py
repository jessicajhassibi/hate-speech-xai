import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from hate_speech_xai.config import SAVED_MODELS_DIR


def load_model_for_explanation(source=SAVED_MODELS_DIR):
	tokenizer = AutoTokenizer.from_pretrained(source)
	model = AutoModelForSequenceClassification.from_pretrained(
		source, output_attentions=True
	)
	model.eval()
	return tokenizer, model


def _subword_importance_to_word_importance(importance, word_ids):
	"""For a word, it takes the max importance across its subword tokens.
	Skips special tokens ([CLS], [SEP], [PAD]) (word_id=None).
	"""
	word_importance = {}
	for idx, word_id in enumerate(word_ids):
		if word_id is None:
			continue
		if word_id not in word_importance:
			word_importance[word_id] = importance[idx]
		else:
			word_importance[word_id] = max(word_importance[word_id], importance[idx])

	num_words = max(word_importance.keys()) + 1 if word_importance else 0
	result = np.zeros(num_words)
	for word_id, imp in word_importance.items():
		result[word_id] = imp
	return result


def explain_attention(text: str, source=SAVED_MODELS_DIR):
	"""Computes word-level importance using attention weights.
	"""
	tokenizer, model = load_model_for_explanation(source)
	inputs = tokenizer(text, return_tensors="pt", truncation=True)

	with torch.no_grad():
		outputs = model(**inputs)

	last_attention = outputs.attentions[-1]
	cls_attention = last_attention[0, :, 0, :].mean(dim=0).numpy()

	word_ids = inputs.word_ids()
	word_importance = _subword_importance_to_word_importance(cls_attention, word_ids)

	if word_importance.max() > 0:
		word_importance = word_importance / word_importance.max()

	return word_importance


# Possible explanation methods
EXPLANATION_METHODS = {
	"Attention (Last Layer)": explain_attention,
}