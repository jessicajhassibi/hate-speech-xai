.PHONY: install app

install:
	 pip install -r requirements.txt
	 pip install -e .

app:
	streamlit run hate_speech_xai/app/app.py
