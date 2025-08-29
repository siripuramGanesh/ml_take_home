import json
from pathlib import Path

import pytest

from client.client import MLClient, SyncMLClient


def test_client_initialization():
    """Test client initialization"""
    client = MLClient()
    assert client.inference_url == "http://localhost:8000"
    assert client.collector_url == "http://localhost:8001"
    
    custom_client = MLClient("http://example.com:9000", "http://example.com:9001")
    assert custom_client.inference_url == "http://example.com:9000"
    assert custom_client.collector_url == "http://example.com:9001"

def test_sync_client():
    """Test synchronous client"""
    client = SyncMLClient()
    assert client.client is not None

def test_bbox_template_creation():
    """Test bounding box template creation"""
    # Create a test bbox template
    template = {
        "person": [[50, 50, 150, 150]],
        "car": [[200, 200, 300, 250]]
    }
    
    # Test JSON serialization
    json_str = json.dumps(template)
    parsed = json.loads(json_str)
    
    assert parsed["person"] == [[50, 50, 150, 150]]
    assert parsed["car"] == [[200, 200, 300, 250]]

def test_dataset_stats_endpoint():
    """Test dataset stats endpoint (mock test)"""
    # This would test the actual endpoint if servers were running
    # For now, just test the client method exists
    client = MLClient()
    assert hasattr(client, 'get_dataset_stats')

def test_model_status_endpoint():
    """Test model status endpoint (mock test)"""
    client = MLClient()
    assert hasattr(client, 'get_model_status')

@pytest.mark.asyncio
async def test_async_context_manager():
    """Test async context manager"""
    async with MLClient() as client:
        assert client is not None
        assert hasattr(client, 'infer')
        assert hasattr(client, 'collect')

def test_cli_import():
    """Test that CLI can be imported"""
    from client.cli import cli
    assert cli is not None
    assert hasattr(cli, 'command')

if __name__ == "__main__":
    pytest.main([__file__, "-v"])