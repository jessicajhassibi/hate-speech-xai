.PHONY: install preprocess train evaluate evaluate-xai test app

install:
	pip install -r requirements.txt
	pip install -e .

preprocess:
	python -m hate_speech_xai.scripts.run_preprocessing

train:
	python -m hate_speech_xai.scripts.run_training

evaluate:
	python -m hate_speech_xai.scripts.run_evaluation

evaluate-xai:
	python -m hate_speech_xai.scripts.run_xai_evaluation

test:
	python -m pytest hate_speech_xai/tests/

app:
	streamlit run hate_speech_xai/app/app.py
