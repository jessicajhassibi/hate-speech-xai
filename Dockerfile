FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY setup.py .
COPY hate_speech_xai/ hate_speech_xai/

RUN pip install --no-cache-dir -e .

# run the Streamlit app at http://localhost:8501
EXPOSE 8501
CMD ["streamlit", "run", "hate_speech_xai/app/app.py", "--server.address", "0.0.0.0"]