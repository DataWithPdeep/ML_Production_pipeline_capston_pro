import unittest
from unittest.mock import patch, MagicMock
import os


# ─────────────────────────────────────────────
# Sabse pehle env var set karo — import se pehle
# ─────────────────────────────────────────────
os.environ["CAPSTONE_TEST"] = "fake_token_for_testing"


# ─────────────────────────────────────────────
# Ab sab kuch mock karo BEFORE app import
# ─────────────────────────────────────────────
mock_model = MagicMock()
mock_model.predict.return_value = ["Positive"]

mock_vectorizer = MagicMock()
mock_vectorizer.transform.return_value.toarray.return_value = [[0] * 10]
mock_vectorizer.transform.return_value.shape = (1, 10)

with patch("mlflow.set_tracking_uri"), \
     patch("mlflow.MlflowClient") as mock_client_cls, \
     patch("mlflow.pyfunc.load_model", return_value=mock_model), \
     patch("pickle.load", return_value=mock_vectorizer):

    # MlflowClient mock
    mock_client = MagicMock()
    mock_client.get_latest_versions.return_value = [MagicMock(version="1")]
    mock_client_cls.return_value = mock_client

    # Ab app import karo — yahan module-level code chalega
    from fastapi_app.app import app


# ─────────────────────────────────────────────
# Template directory fix — CI ke liye absolute path
# ─────────────────────────────────────────────
from fastapi.templating import Jinja2Templates
import fastapi_app.app as app_module

templates_dir = os.path.join(os.path.dirname(__file__), "..", "fastapi_app", "templates")
app_module.templates = Jinja2Templates(directory=os.path.abspath(templates_dir))


# ─────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────
from fastapi.testclient import TestClient


class FastAPIAppTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_home_page(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("<title>Sentiment Analysis</title>", response.text)

    def test_predict_page(self):
        with patch.object(app_module, "model", mock_model), \
             patch.object(app_module, "vectorizer", mock_vectorizer):

            response = self.client.post("/predict", data={"text": "I love this!"})
            self.assertEqual(response.status_code, 200)
            self.assertTrue(
                "Positive" in response.text or "Negative" in response.text,
                "Response should contain either 'Positive' or 'Negative'"
            )


if __name__ == "__main__":
    unittest.main()