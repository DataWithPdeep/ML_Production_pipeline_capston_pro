FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY fastapi_app/ /app/fastapi_app/

# Copy ML models
COPY models/ /app/models/

# Download NLTK data
RUN python -m nltk.downloader stopwords wordnet

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "fastapi_app.app:app", "-k", "uvicorn.workers.UvicornWorker"]
