FROM python:3.10-slim

WORKDIR /app

COPY fastapi_app/ /app/

COPY models/vectorizer.pkl /app/models/vectorizer.pkl

RUN pip install -r requirements.txt

RUN python -m nltk.downloader stopwords wordnet

EXPOSE 5000

# Local run
# CMD ["python", "main.py"]

# Production run
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "main:app", "-k", "uvicorn.workers.UvicornWorker"]
