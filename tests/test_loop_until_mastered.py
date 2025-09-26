import pytest
from unittest.mock import patch, MagicMock
import json

from agent.services.teaching import TeachingService

@pytest.fixture
def mock_llm():
    """Mock the LLM client"""
    mock = MagicMock()
    mock.send.return_value = {"text": "Mocked response"}
    return mock

@pytest.fixture
def teaching_service(mock_llm):
    """Create an instance of TeachingService with a mock LLM"""
    return TeachingService(mock_llm)

@pytest.mark.asyncio
async def test_initialize_mastery_tracking(teaching_service):
    """Test initializing mastery tracking"""
    session_id = "test-session"
    user_id = "test-user"
    topic = "test-topic"
    
    teaching_service._initialize_mastery_tracking(session_id, user_id, topic)
    
    # Check that tracking was initialized
    assert session_id in teaching_service._mastery_tracking
    tracking = teaching_service._mastery_tracking[session_id]
    
    # Verify tracking fields
    assert tracking["user_id"] == user_id
    assert tracking["topic"] == topic
    assert tracking["attempts"] == 0
    assert tracking["correct_answers"] == 0
    assert tracking["current_streak"] == 0
    assert tracking["mastery_level"] == 0
    assert tracking["current_difficulty"] == "beginner"
    assert tracking["current_strategy"] in teaching_service.STRATEGIES

@pytest.mark.asyncio
async def test_update_mastery_tracking_correct(teaching_service):
    """Test updating mastery tracking with correct answers"""
    session_id = "test-session"
    user_id = "test-user"
    topic = "test-topic"
    
    # Initialize tracking
    teaching_service._initialize_mastery_tracking(session_id, user_id, topic)
    
    # Update with correct answer
    teaching_service._update_mastery_tracking(session_id, True)
    
    # Check tracking was updated
    tracking = teaching_service._mastery_tracking[session_id]
    assert tracking["attempts"] == 1
    assert tracking["correct_answers"] == 1
    assert tracking["current_streak"] == 1
    assert tracking["mastery_level"] > 0  # Should increase
    
    # Update with another correct answer
    teaching_service._update_mastery_tracking(session_id, True)
    
    # Check tracking was updated again
    tracking = teaching_service._mastery_tracking[session_id]
    assert tracking["attempts"] == 2
    assert tracking["correct_answers"] == 2
    assert tracking["current_streak"] == 2
    assert tracking["mastery_level"] > 0.5  # Should increase more

@pytest.mark.asyncio
async def test_update_mastery_tracking_incorrect(teaching_service):
    """Test updating mastery tracking with incorrect answers"""
    session_id = "test-session"
    user_id = "test-user"
    topic = "test-topic"
    
    # Initialize tracking
    teaching_service._initialize_mastery_tracking(session_id, user_id, topic)
    
    # Set initial mastery level
    teaching_service._mastery_tracking[session_id]["mastery_level"] = 5.0
    
    # Update with incorrect answer
    teaching_service._update_mastery_tracking(session_id, False)
    
    # Check tracking was updated
    tracking = teaching_service._mastery_tracking[session_id]
    assert tracking["attempts"] == 1
    assert tracking["correct_answers"] == 0
    assert tracking["current_streak"] == 0
    assert tracking["mastery_level"] < 5.0  # Should decrease
    assert tracking["retry_count"] == 1

@pytest.mark.asyncio
async def test_strategy_switching(teaching_service):
    """Test strategy switching after multiple failures"""
    session_id = "test-session"
    user_id = "test-user"
    topic = "test-topic"
    
    # Initialize tracking
    teaching_service._initialize_mastery_tracking(session_id, user_id, topic)
    
    # Get initial strategy
    initial_strategy = teaching_service._mastery_tracking[session_id]["current_strategy"]
    
    # Update with multiple incorrect answers to trigger strategy change
    # We need to patch datetime.now to simulate time passing
    with patch('agent.services.teaching.datetime') as mock_datetime:
        # Mock datetime to return a time in the future
        mock_now = MagicMock()
        mock_now.return_value = teaching_service._mastery_tracking[session_id]["last_strategy_change"]
        mock_datetime.now = mock_now
        
        # First incorrect answer
        teaching_service._update_mastery_tracking(session_id, False)
        
        # Second incorrect answer
        teaching_service._update_mastery_tracking(session_id, False)
        
        # Set time to be after cooldown
        from datetime import timedelta
        mock_now.return_value = teaching_service._mastery_tracking[session_id]["last_strategy_change"] + timedelta(minutes=10)
        mock_datetime.now = mock_now
        
        # Third incorrect answer should trigger strategy change
        teaching_service._update_mastery_tracking(session_id, False)
    
    # Check that strategy was changed
    new_strategy = teaching_service._mastery_tracking[session_id]["current_strategy"]
    assert new_strategy != initial_strategy

@pytest.mark.asyncio
async def test_difficulty_progression(teaching_service):
    """Test difficulty progression based on mastery level"""
    session_id = "test-session"
    user_id = "test-user"
    topic = "test-topic"
    
    # Initialize tracking
    teaching_service._initialize_mastery_tracking(session_id, user_id, topic)
    
    # Initial difficulty should be beginner
    assert teaching_service._mastery_tracking[session_id]["current_difficulty"] == "beginner"
    
    # Set mastery level to trigger intermediate difficulty
    teaching_service._mastery_tracking[session_id]["mastery_level"] = 3.0
    teaching_service._update_mastery_tracking(session_id, True)
    assert teaching_service._mastery_tracking[session_id]["current_difficulty"] == "intermediate"
    
    # Set mastery level to trigger advanced difficulty
    teaching_service._mastery_tracking[session_id]["mastery_level"] = 6.0
    teaching_service._update_mastery_tracking(session_id, True)
    assert teaching_service._mastery_tracking[session_id]["current_difficulty"] == "advanced"
    
    # Set mastery level to trigger expert difficulty
    teaching_service._mastery_tracking[session_id]["mastery_level"] = 9.0
    teaching_service._update_mastery_tracking(session_id, True)
    assert teaching_service._mastery_tracking[session_id]["current_difficulty"] == "expert"

@pytest.mark.asyncio
async def test_generate_feedback_with_tracking(teaching_service, mock_llm):
    """Test that generate_feedback updates mastery tracking"""
    session_id = "test-session"
    user_id = "test-user"
    topic = "test-topic"
    
    # Initialize tracking
    teaching_service._initialize_mastery_tracking(session_id, user_id, topic)
    teaching_service._sessions[session_id] = {"user_id": user_id, "topic": topic, "mode": "quiz"}
    
    # Mock the LLM response to indicate correct answer
    mock_llm.send.return_value = {"text": "Feedback text\nCORRECT: true"}
    
    # Generate feedback
    feedback, is_correct = await teaching_service.generate_feedback(
        mock_llm,
        session_id,
        "Test answer",
        None
    )
    
    # Check that feedback was generated and tracking was updated
    assert feedback == "Feedback text"
    assert is_correct == True
    assert teaching_service._mastery_tracking[session_id]["attempts"] == 1
    assert teaching_service._mastery_tracking[session_id]["correct_answers"] == 1

@pytest.mark.asyncio
async def test_generate_next_item_with_strategy(teaching_service, mock_llm):
    """Test that generate_next_item uses the current strategy"""
    session_id = "test-session"
    user_id = "test-user"
    topic = "test-topic"
    
    # Initialize tracking with a specific strategy
    teaching_service._initialize_mastery_tracking(session_id, user_id, topic)
    teaching_service._mastery_tracking[session_id]["current_strategy"] = "socratic_dialogue"
    teaching_service._sessions[session_id] = {"user_id": user_id, "topic": topic, "mode": "quiz"}
    
    # Generate next item
    await teaching_service.generate_next_item(
        mock_llm,
        session_id,
        None
    )
    
    # Check that the LLM was called with the correct strategy in the prompt
    call_args = mock_llm.send.call_args[1]
    assert "socratic_dialogue" in call_args.get("system_prompt", "")
