import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from hate_speech_xai.src.config import SAVED_MODELS_DIR, MODEL_NAME


def load_model(source=SAVED_MODELS_DIR):
	tokenizer = AutoTokenizer.from_pretrained(source)
	model = AutoModelForSequenceClassification.from_pretrained(source)
	return tokenizer, model


def predict(post: str):
	tokenizer, model = load_model()
	inputs = tokenizer(post, return_tensors="pt", truncation=True)
	with torch.no_grad():
		outputs = model(**inputs)
	pred = torch.argmax(outputs.logits, dim=1)
	print(pred)