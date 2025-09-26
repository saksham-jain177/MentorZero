import pytest
from unittest.mock import patch, MagicMock

from agent.services.validators import AZLValidators

@pytest.fixture
def mock_llm():
    """Mock the LLM client"""
    mock = MagicMock()
    mock.send.return_value = {"text": "True"}
    return mock

@pytest.fixture
def validators(mock_llm):
    """Create an instance of AZLValidators with a mock LLM"""
    return AZLValidators(mock_llm)

@pytest.fixture
def example():
    """Create a test example"""
    return {
        "question": "What is the capital of France?",
        "answer": "The capital of France is Paris.",
        "options": ["London", "Berlin", "Madrid"]
    }

@pytest.mark.asyncio
async def test_validate_length(validators, example):
    """Test the length validator"""
    # Test with a valid example
    result = await validators.validate_length(example)
    assert result["passed"] == True
    
    # Test with an example that's too short
    short_example = {
        "question": "Q?",
        "answer": "A.",
        "options": ["B", "C", "D"]
    }
    result = await validators.validate_length(short_example)
    assert result["passed"] == False
    
    # Test with an example that's too long
    long_question = "Q?" * 500
    long_example = {
        "question": long_question,
        "answer": "A.",
        "options": ["B", "C", "D"]
    }
    result = await validators.validate_length(long_example)
    assert result["passed"] == False

@pytest.mark.asyncio
async def test_validate_no_duplicates(validators, example):
    """Test the duplicate validator"""
    # Mock the LLM response for duplicate check
    validators.llm.send.return_value = {"text": "False"}
    
    # Test with a unique example
    result = await validators.validate_no_duplicates(example)
    assert result["passed"] == True
    
    # Test with a duplicate example
    validators.llm.send.return_value = {"text": "True"}
    result = await validators.validate_no_duplicates(example)
    assert result["passed"] == False

@pytest.mark.asyncio
async def test_validate_consistency(validators, example):
    """Test the consistency validator"""
    # Mock the LLM response for consistency check
    validators.llm.send.return_value = {"text": "True"}
    
    # Test with a consistent example
    result = await validators.validate_consistency(example)
    assert result["passed"] == True
    
    # Test with an inconsistent example
    validators.llm.send.return_value = {"text": "False"}
    result = await validators.validate_consistency(example)
    assert result["passed"] == False

@pytest.mark.asyncio
async def test_validate_roundtrip(validators, example):
    """Test the roundtrip validator"""
    # Mock the LLM response for roundtrip check
    validators.llm.send.return_value = {"text": "True"}
    
    # Test with a valid roundtrip example
    result = await validators.validate_roundtrip(example)
    assert result["passed"] == True
    
    # Test with an invalid roundtrip example
    validators.llm.send.return_value = {"text": "False"}
    result = await validators.validate_roundtrip(example)
    assert result["passed"] == False

@pytest.mark.asyncio
async def test_validate_no_toxicity(validators, example):
    """Test the toxicity validator"""
    # Mock the LLM response for toxicity check
    validators.llm.send.return_value = {"text": "False"}
    
    # Test with a non-toxic example
    result = await validators.validate_no_toxicity(example)
    assert result["passed"] == True
    
    # Test with a toxic example
    validators.llm.send.return_value = {"text": "True"}
    result = await validators.validate_no_toxicity(example)
    assert result["passed"] == False
