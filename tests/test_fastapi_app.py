import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from fastapi.testclient import TestClient


# ─────────────────────────────────────────────
# MLflow aur vectorizer ko mock karo
# CAPSTONE_TEST env var bhi set karo
# ─────────────────────────────────────────────

@patch.dict("os.environ", {"CAPSTONE_TEST": "fake_token_for_testing"})
@patch("fastapi_app.app.mlflow.set_tracking_uri")
@patch("fastapi_app.app.get_latest_model_version", return_value="1")
@patch("fastapi_app.app.mlflow.pyfunc.load_model")
@patch("fastapi_app.app.pickle.load")
class FastAPIAppTests(unittest.TestCase):

    def _make_client(self, mock_pickle, mock_load_model, mock_version, mock_uri):
        """Mock model aur vectorizer setup karke TestClient banao."""
        # Vectorizer mock: transform() → sparse-like array
        mock_vectorizer = MagicMock()
        mock_vectorizer.transform.return_value.toarray.return_value = [[0] * 10]
        mock_vectorizer.transform.return_value.shape = (1, 10)
        mock_pickle.return_value = mock_vectorizer

        # Model mock: predict() → ["Positive"]
        mock_model = MagicMock()
        mock_model.predict.return_value = ["Positive"]
        mock_load_model.return_value = mock_model

        return TestClient(app)

    @classmethod
    def setUpClass(cls):
        # App ko yahan import karo taaki patches pehle lag jayein
        from fastapi_app.app import app
        cls.app = app

    def test_home_page(self, mock_pickle, mock_load_model, mock_version, mock_uri):
        mock_vectorizer = MagicMock()
        mock_pickle.return_value = mock_vectorizer
        mock_load_model.return_value = MagicMock()

        with TestClient(self.app) as client:
            response = client.get("/")
            self.assertEqual(response.status_code, 200)
            self.assertIn("<title>Sentiment Analysis</title>", response.text)

    def test_predict_page(self, mock_pickle, mock_load_model, mock_version, mock_uri):
        # Vectorizer mock
        mock_vectorizer = MagicMock()
        mock_vectorizer.transform.return_value.toarray.return_value = [[0] * 10]
        mock_vectorizer.transform.return_value.shape = (1, 10)
        mock_pickle.return_value = mock_vectorizer

        # Model mock
        mock_model = MagicMock()
        mock_model.predict.return_value = ["Positive"]
        mock_load_model.return_value = mock_model

        with TestClient(self.app) as client:
            response = client.post("/predict", data={"text": "I love this!"})
            self.assertEqual(response.status_code, 200)
            self.assertTrue(
                "Positive" in response.text or "Negative" in response.text,
                "Response should contain either 'Positive' or 'Negative'"
            )


if __name__ == "__main__":
    unittest.main()