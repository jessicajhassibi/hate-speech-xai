import shap
import torch
import numpy as np

from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline, TextClassificationPipeline
from captum.attr import IntegratedGradients

from hate_speech_xai.config import SAVED_MODELS_DIR

_model_cache: dict[str, tuple[AutoTokenizer, AutoModelForSequenceClassification]] = {}


def load_model_for_explanation(source: Path = SAVED_MODELS_DIR) -> tuple[AutoTokenizer, AutoModelForSequenceClassification]:
	"""Loads our fine-tuned tokenizer and model with output_attentions set to True.
	The result is cached so subsequent calls with the same source reuse the loaded model.
	"""
	key = str(source)
	if key not in _model_cache:
		tokenizer = AutoTokenizer.from_pretrained(source)
		model = AutoModelForSequenceClassification.from_pretrained(
			source, output_attentions=True
		)
		model.eval()
		_model_cache[key] = (tokenizer, model)
	return _model_cache[key]


def _subword_importance_to_word_importance(importance: np.ndarray, word_ids: list[int | None]) -> np.ndarray:
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


def explain_attention(text: str, source: Path = SAVED_MODELS_DIR) -> np.ndarray:
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


def explain_integrated_gradients(text: str, source: Path = SAVED_MODELS_DIR) -> np.ndarray:
	"""Compute word-level importance using Integrated Gradients on the
	embedding layer. Attributes importance to each token embedding,
	then sums across the embedding dimension to get per-token scores.
	"""
	tokenizer, model = load_model_for_explanation(source)
	inputs = tokenizer(text, return_tensors="pt", truncation=True)
	attention_mask = inputs["attention_mask"]
	input_embeds = model.bert.embeddings(inputs["input_ids"])
	baseline = torch.zeros_like(input_embeds)

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
		word_importance = word_importance / word_importance.max() # normalize to value between 0 and 1

	return word_importance


def explain_shap(text: str, source: Path = SAVED_MODELS_DIR) -> np.ndarray:
	"""Computes word-level importance using SHAP's Partition Explainer with a HuggingFace text-classification pipeline
	wrapped around our model.
	"""
	tokenizer, model = load_model_for_explanation(source)
	inputs = tokenizer(text, return_tensors="pt", truncation=True)

	with torch.no_grad():
		predicted_class = model(**inputs).logits.argmax(dim=-1).item()

	pipe: TextClassificationPipeline = pipeline("text-classification", model=model, tokenizer=tokenizer,
					return_all_scores=True, truncation=True, device="cpu") # device="cpu" to keep the model on cpu, pipeline from Huggingface otherwise would have moved it to MPS on Mac
	explainer = shap.Explainer(pipe) # Use Partition Explainer with a text masker: masks tokens and computes Shapley values
	shap_values = explainer([text])

	# shap_values.values have the shape (1, n_tokens, n_classes) for one importance value per token and class
	token_importance = np.abs(shap_values.values[0, :, predicted_class]) # take abs of SHAP values for predicted class

	word_ids = inputs.word_ids()
	if len(token_importance) == len(word_ids):
		word_importance = _subword_importance_to_word_importance(token_importance, word_ids)
	else:
		word_importance = token_importance

	if word_importance.max() > 0:
		word_importance = word_importance / word_importance.max()

	return word_importance


# Possible explanation methods
EXPLANATION_METHODS = {
	"Attention (Last Layer)": explain_attention,
	"Integrated Gradients": explain_integrated_gradients,
	# "SHAP": explain_shap,
	# as time was not sufficient and evaluating SHAP took very long, SHAP was dropped
}