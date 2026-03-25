from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from hate_speech_xai.config import SAVED_MODELS_DIR


def load_model(source: Path = SAVED_MODELS_DIR) -> tuple[AutoTokenizer, AutoModelForSequenceClassification]:
	tokenizer = AutoTokenizer.from_pretrained(source)
	model = AutoModelForSequenceClassification.from_pretrained(source)
	return tokenizer, model


def predict_label(post: str) -> int:
	tokenizer, model = load_model()
	inputs = tokenizer(post, return_tensors="pt", truncation=True)
	with torch.no_grad():
		logits = model(**inputs).logits
	predicted_class_id = logits.argmax().item()
	return predicted_class_id