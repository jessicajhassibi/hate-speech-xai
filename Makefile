.PHONY: install app

install:
	 pip install -r requirements.txt
	 pip install -e .

app:
	streamlit run student_performance_xai/app/app.py
