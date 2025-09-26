import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from api.main import app

client = TestClient(app)

@pytest.fixture
def mock_azl_generator():
    """Mock the AZLGenerator class"""
    with patch("agent.azl.generator.AZLGenerator") as mock:
        instance = mock.return_value
        instance.propose_examples.return_value = [
            {
                "question": "Test question 1?",
                "answer": "Test answer 1",
                "options": ["Wrong 1", "Wrong 2", "Wrong 3"]
            },
            {
                "question": "Test question 2?",
                "answer": "Test answer 2",
                "options": ["Wrong A", "Wrong B", "Wrong C"]
            }
        ]
        yield instance

@pytest.fixture
def mock_azl_validators():
    """Mock the AZLValidators class"""
    with patch("agent.services.validators.AZLValidators") as mock:
        instance = mock.return_value
        
        # Mock validation methods
        instance.validate_length.return_value = {"passed": True, "message": "Length is acceptable"}
        instance.validate_no_duplicates.return_value = {"passed": True, "message": "No duplicates found"}
        instance.validate_consistency.return_value = {"passed": True, "message": "Example is consistent"}
        instance.validate_roundtrip.return_value = {"passed": True, "message": "Roundtrip validation passed"}
        instance.validate_no_toxicity.return_value = {"passed": True, "message": "No toxicity detected"}
        
        yield instance

@pytest.fixture
def mock_teaching_service():
    """Mock the TeachingService class"""
    with patch("agent.services.teaching.TeachingService") as mock:
        instance = mock.return_value
        instance.store_synthetic_example.return_value = True
        yield instance

def test_azl_propose_endpoint(mock_azl_generator):
    """Test the AZL propose endpoint"""
    with patch("api.routes.AZLGenerator", return_value=mock_azl_generator):
        response = client.post(
            "/azl/propose",
            json={"topic": "test topic", "count": 2}
        )
        assert response.status_code == 200
        data = response.json()
        assert "examples" in data
        assert "proposal_id" in data
        assert len(data["examples"]) == 2
        mock_azl_generator.propose_examples.assert_called_once_with("test topic", 2)

def test_azl_validate_endpoint(mock_azl_validators):
    """Test the AZL validate endpoint"""
    with patch("api.routes.AZLValidators", return_value=mock_azl_validators):
        response = client.post(
            "/azl/validate",
            json={"proposal_id": "test-id", "example_idx": 0}
        )
        assert response.status_code == 200
        data = response.json()
        assert "validation_results" in data
        assert "passed" in data
        assert data["passed"] == True
        
        # Check that all validation methods were called
        mock_azl_validators.validate_length.assert_called_once()
        mock_azl_validators.validate_no_duplicates.assert_called_once()
        mock_azl_validators.validate_consistency.assert_called_once()
        mock_azl_validators.validate_roundtrip.assert_called_once()
        mock_azl_validators.validate_no_toxicity.assert_called_once()

def test_azl_accept_endpoint(mock_teaching_service):
    """Test the AZL accept endpoint"""
    with patch("api.deps.get_teaching_service", return_value=mock_teaching_service):
        response = client.post(
            "/azl/accept",
            json={
                "proposal_id": "test-id", 
                "example_idx": 0, 
                "accepted": True,
                "message": "Looks good"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "success" in data
        assert data["success"] == True
        mock_teaching_service.store_synthetic_example.assert_called_once()
