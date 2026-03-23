import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from captum.attr import IntegratedGradients

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


def explain_integrated_gradients(text: str, source=SAVED_MODELS_DIR):
	"""Compute word-level importance using Integrated Gradients on the
	embedding layer. Attributes importance to each token embedding,
	then sums across the embedding dimension to get per-token scores.
	"""

	tokenizer, model = load_model_for_explanation(source)
	inputs = tokenizer(text, return_tensors="pt", truncation=True)
	attention_mask = inputs["attention_mask"]

	# Get embeddings for the input
	input_embeds = model.bert.embeddings(inputs["input_ids"])
	baseline = torch.zeros_like(input_embeds)

	# Predicted class to explain
	with torch.no_grad():
		predicted_class = model(**inputs).logits.argmax(dim=-1).item()

	def forward_from_embeddings(embeddings):
		outputs = model(inputs_embeds=embeddings, attention_mask=attention_mask)
		return outputs.logits

	ig = IntegratedGradients(forward_from_embeddings)

	attributions = ig.attribute(
		inputs=input_embeds,
		baselines=baseline,
		target=predicted_class,
		n_steps=50,
	)

	# Sum across embedding dim, take absolute value -> per-token importance
	token_importance = attributions.sum(dim=-1).abs().squeeze(0).detach().numpy()

	word_ids = inputs.word_ids()
	word_importance = _subword_importance_to_word_importance(token_importance, word_ids)

	if word_importance.max() > 0:
		word_importance = word_importance / word_importance.max()

	return word_importance


# Possible explanation methods
EXPLANATION_METHODS = {
	"Attention (Last Layer)": explain_attention,
	"Integrated Gradients": explain_integrated_gradients,
}