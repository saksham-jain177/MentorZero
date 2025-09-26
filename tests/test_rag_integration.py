import pytest
from unittest.mock import patch, MagicMock

from agent.services.teaching import TeachingService
from agent.vectorstore.faiss_store import FAISSStore
from agent.embeddings.service import EmbeddingService

@pytest.fixture
def mock_llm():
    """Mock the LLM client"""
    mock = MagicMock()
    mock.send.return_value = {"text": "Mocked response"}
    return mock

@pytest.fixture
def mock_embedding_service():
    """Mock the embedding service"""
    mock = MagicMock()
    mock.embed_query.return_value = [0.1] * 384  # Mock embedding vector
    return mock

@pytest.fixture
def mock_faiss_store():
    """Mock the FAISS store"""
    mock = MagicMock()
    mock.similarity_search_with_score.return_value = [
        (
            MagicMock(
                page_content="Relevant context for the query",
                metadata={"source": "test-document"}
            ),
            0.8
        )
    ]
    return mock

@pytest.fixture
def teaching_service(mock_llm):
    """Create an instance of TeachingService with a mock LLM"""
    return TeachingService(mock_llm)

@pytest.mark.asyncio
async def test_process_text(teaching_service):
    """Test processing text for RAG"""
    session_id = "test-session"
    text = "This is a test document.\n\nIt has multiple paragraphs.\n\nEach paragraph provides context."
    
    # Process the text
    chunks = teaching_service.process_text(session_id, text)
    
    # Check that the text was stored
    assert session_id in teaching_service._contexts
    assert teaching_service._contexts[session_id] == text
    
    # Check that the number of chunks is correct
    assert chunks == 3  # 3 paragraphs

@pytest.mark.asyncio
async def test_get_context_for_topic(teaching_service):
    """Test getting context for a topic"""
    session_id = "test-session"
    text = "This is context about binary search."
    
    # Store context
    teaching_service._contexts[session_id] = text
    
    # Get context
    context = teaching_service.get_context_for_topic(session_id, "binary search")
    
    # Check that the context was retrieved
    assert context == text

@pytest.mark.asyncio
async def test_generate_explanation_with_context(teaching_service, mock_llm):
    """Test generating explanation with RAG context"""
    session_id = "test-session"
    user_id = "test-user"
    topic = "binary search"
    context = "Binary search is an efficient algorithm for finding an item in a sorted list."
    
    # Store context
    teaching_service._contexts[session_id] = context
    
    # Generate explanation
    await teaching_service.generate_explanation(
        mock_llm,
        user_id,
        topic,
        "explain",
        session_id,
        context
    )
    
    # Check that the LLM was called with the context
    call_args = mock_llm.send.call_args[1]
    assert "Relevant context" in call_args.get("prompt", "")
    assert "Binary search" in call_args.get("prompt", "")

@pytest.mark.asyncio
async def test_generate_feedback_with_context(teaching_service, mock_llm):
    """Test generating feedback with RAG context"""
    session_id = "test-session"
    user_id = "test-user"
    topic = "binary search"
    context = "Binary search requires a sorted array."
    user_answer = "Binary search works on any array."
    
    # Initialize session
    teaching_service._initialize_mastery_tracking(session_id, user_id, topic)
    teaching_service._sessions[session_id] = {"user_id": user_id, "topic": topic, "mode": "quiz"}
    
    # Store context
    teaching_service._contexts[session_id] = context
    
    # Mock LLM response
    mock_llm.send.return_value = {"text": "Feedback with context\nCORRECT: false"}
    
    # Generate feedback
    feedback, is_correct = await teaching_service.generate_feedback(
        mock_llm,
        session_id,
        user_answer,
        context
    )
    
    # Check that the LLM was called with the context
    call_args = mock_llm.send.call_args[1]
    assert "Relevant context" in call_args.get("prompt", "")
    assert "Binary search requires a sorted array" in call_args.get("prompt", "")
    
    # Check the feedback
    assert feedback == "Feedback with context"
    assert is_correct == False

@pytest.mark.asyncio
async def test_generate_next_item_with_context(teaching_service, mock_llm):
    """Test generating next item with RAG context"""
    session_id = "test-session"
    user_id = "test-user"
    topic = "binary search"
    context = "Binary search has O(log n) time complexity."
    
    # Initialize session
    teaching_service._initialize_mastery_tracking(session_id, user_id, topic)
    teaching_service._sessions[session_id] = {"user_id": user_id, "topic": topic, "mode": "quiz"}
    
    # Store context
    teaching_service._contexts[session_id] = context
    
    # Generate next item
    await teaching_service.generate_next_item(
        mock_llm,
        session_id,
        context
    )
    
    # Check that the LLM was called with the context
    call_args = mock_llm.send.call_args[1]
    assert "Relevant context" in call_args.get("prompt", "")
    assert "Binary search has O(log n)" in call_args.get("prompt", "")
