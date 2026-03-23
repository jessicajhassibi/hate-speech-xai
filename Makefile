.PHONY: install preprocess train evaluate evaluate-xai app

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

app:
	streamlit run hate_speech_xai/app/app.py
