"""
API tests. These mock the model so they run fast and don't require a trained
checkpoint to be present -- useful for CI.
"""

from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_health_endpoint_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()


@patch("src.api.main._model")
@patch("src.api.main._tokenizer")
def test_predict_returns_valid_shape(mock_tokenizer, mock_model):
    # Arrange: fake a 2-class model output
    import torch

    mock_tokenizer.return_value = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
    }
    mock_output = MagicMock()
    mock_output.logits = torch.tensor([[0.2, 1.5]])
    mock_model.return_value = mock_output

    with patch("src.api.main._label_names", ["CLASS_A", "CLASS_B"]):
        response = client.post("/predict", json={"text": "sample clinical sentence"})

    assert response.status_code == 200
    body = response.json()
    assert "label" in body
    assert "confidence" in body
    assert "all_scores" in body
    assert 0.0 <= body["confidence"] <= 1.0


def test_predict_rejects_missing_text_field():
    response = client.post("/predict", json={})
    assert response.status_code == 422
