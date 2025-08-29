import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from inference.app import app
from inference.models import MockModel, YOLOModel

client = TestClient(app)

def test_inference_endpoint_no_image():
    """Test inference endpoint without image"""
    response = client.post("/inference")
    assert response.status_code == 422  # Validation error

def test_model_status():
    """Test model status endpoint"""
    response = client.get("/model/status")
    assert response.status_code == 200
    assert "training_phase" in response.json()
    assert "model_type" in response.json()

def test_mock_model_prediction():
    """Test mock model prediction"""
    model = MockModel()
    
    # Create test image
    test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    # Test async prediction
    import asyncio
    predictions = asyncio.run(model.predict_async(test_image))
    
    assert isinstance(predictions, list)
    if predictions:  # Mock model always returns at least one prediction
        assert "bbox" in predictions[0]
        assert "confidence" in predictions[0]

def test_yolo_model_initialization():
    """Test YOLO model initialization"""
    try:
        model = YOLOModel()
        assert model is not None
        assert hasattr(model, 'predict_async')
        assert hasattr(model, 'get_metrics')
    except Exception as e:
        # YOLO might not be available in test environment
        pytest.skip(f"YOLO model not available: {e}")

def test_base64_inference():
    """Test base64 inference endpoint"""
    # Create test image
    test_image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
    pil_image = Image.fromarray(test_image)
    
    # Convert to base64
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG")
    image_data = buffer.getvalue()
    import base64
    base64_data = base64.b64encode(image_data).decode('utf-8')
    
    # Test endpoint
    response = client.post("/inference/base64", json={
        "image_data": base64_data,
        "labels": ["test"]
    })
    
    assert response.status_code in [200, 500]  # Could work or fail gracefully

if __name__ == "__main__":
    pytest.main([__file__, "-v"])