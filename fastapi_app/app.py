from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import mlflow
import pickle
import os
import pandas as pd
import time
import dagshub
import re
import string

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# ----------------------------
# Text Preprocessing Functions
# ----------------------------

def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text]
    return " ".join(text)

def remove_stop_words(text):
    stop_words = set(stopwords.words("english"))
    text = [word for word in str(text).split() if word not in stop_words]
    return " ".join(text)

def removing_numbers(text):
    return ''.join([char for char in text if not char.isdigit()])

def lower_case(text):
    return " ".join([word.lower() for word in text.split()])

def removing_punctuations(text):
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub('\s+', ' ', text).strip()
    return text

def removing_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub('', text)

def normalize_text(text):
    text = lower_case(text)
    text = remove_stop_words(text)
    text = removing_numbers(text)
    text = removing_punctuations(text)
    text = removing_urls(text)
    text = lemmatization(text)
    return text

# ----------------------------
# MLflow Setup
# ----------------------------

# Set up DagsHub credentials for MLflow tracking
dagshub_token = os.getenv("CAPSTONE_TEST")

print("Token Exists:", bool(dagshub_token))
print("Token Length:", len(dagshub_token) if dagshub_token else 0)

dagshub.auth.add_app_token(dagshub_token)

dagshub.init(
    repo_owner="DataWithPdeep",
    repo_name="ML_Production_pipeline_capston_pro",
    mlflow=True
)

print("Tracking URI:", mlflow.get_tracking_uri())
model_name = "my_model"

def get_latest_model_version(model_name):
    client = mlflow.MlflowClient()

    versions = client.search_model_versions(
        f"name='{model_name}'"
    )

    if not versions:
        return None

    latest = max(versions, key=lambda v: int(v.version))
    return latest.version

model_version = get_latest_model_version(model_name)

model_uri = f"models:/{model_name}/{model_version}"
model = mlflow.pyfunc.load_model(model_uri)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS_DIR = os.path.join(BASE_DIR, "..", "models")

MODEL_PATH = os.path.join(MODELS_DIR, "model.pkl")

VECTORIZER_PATH = os.path.join(MODELS_DIR, "vectorizer.pkl")

print("Model Path :", MODEL_PATH)
print("Vectorizer Path :", VECTORIZER_PATH)

print("Model Exists :", os.path.exists(MODEL_PATH))
print("Vectorizer Exists :", os.path.exists(VECTORIZER_PATH))
# ----------------------------
# FastAPI App
# ----------------------------

app = FastAPI(title="Fake News Detection")

templates = Jinja2Templates(directory="fastapi_app/templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,          
        name="index.html",        
        context={"result": None}  
    )

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    text: str = Form(...)
):
    cleaned = normalize_text(text)
    features = vectorizer.transform([cleaned])
    features_df = pd.DataFrame(
        features.toarray(),
        columns=[str(i) for i in range(features.shape[1])]
    )
    result = model.predict(features_df)
    prediction = result[0]

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"result": prediction}
    )