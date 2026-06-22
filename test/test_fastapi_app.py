import unittest
from fastapi.testclient import TestClient
from fastapi_app.app import app

class FastAPIAppTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_home_page(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("<title>Sentiment Analysis</title>", response.text)

    def test_predict_page(self):
        response = self.client.post(
            "/predict",
            data={"text": "I love this!"}
        )

        self.assertEqual(response.status_code, 200)

        self.assertTrue(
            "Positive" in response.text or "Negative" in response.text,
            "Response should contain either 'Positive' or 'Negative'"
        )

if __name__ == "__main__":
    unittest.main()