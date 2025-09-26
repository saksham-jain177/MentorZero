import os
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from api.main import app
from agent.voice.stt import STT_AVAILABLE
from agent.voice.tts import TTS_AVAILABLE

client = TestClient(app)

# Skip all tests if voice modules are not available
pytestmark = pytest.mark.skipif(
    not (STT_AVAILABLE and TTS_AVAILABLE),
    reason="Voice modules (STT or TTS) not available"
)

@pytest.fixture
def mock_audio_file():
    """Create a mock audio file for testing"""
    return MagicMock(
        filename="test_audio.wav",
        file=MagicMock(read=lambda: b"mock audio data")
    )

@pytest.fixture
def mock_whisper_stt():
    """Mock the WhisperSTT class"""
    with patch("agent.voice.stt.WhisperSTT") as mock:
        instance = mock.return_value
        instance.transcribe.return_value = "mock transcription"
        yield instance

@pytest.fixture
def mock_chatterbox_tts():
    """Mock the ChatterboxTTS class"""
    with patch("agent.voice.tts.ChatterboxTTS") as mock:
        instance = mock.return_value
        instance.synthesize.return_value = b"mock audio data"
        yield instance

def test_voice_health_endpoint():
    """Test the voice health endpoint"""
    response = client.get("/voice_health")
    assert response.status_code == 200
    data = response.json()
    assert "stt_available" in data
    assert "tts_available" in data
    assert data["stt_available"] == STT_AVAILABLE
    assert data["tts_available"] == TTS_AVAILABLE

@pytest.mark.skipif(not STT_AVAILABLE, reason="STT not available")
def test_stt_endpoint(mock_audio_file, mock_whisper_stt):
    """Test the STT endpoint"""
    with patch("api.routes.WhisperSTT", return_value=mock_whisper_stt):
        response = client.post(
            "/stt",
            files={"audio_file": ("test_audio.wav", b"mock audio data", "audio/wav")},
            data={"language": "en"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "text" in data
        assert data["text"] == "mock transcription"
        mock_whisper_stt.transcribe.assert_called_once()

@pytest.mark.skipif(not TTS_AVAILABLE, reason="TTS not available")
def test_tts_endpoint(mock_chatterbox_tts):
    """Test the TTS endpoint"""
    with patch("api.routes.ChatterboxTTS", return_value=mock_chatterbox_tts):
        response = client.post(
            "/tts",
            json={"text": "Hello, world!", "voice": "default"}
        )
        assert response.status_code == 200
        assert response.content == b"mock audio data"
        mock_chatterbox_tts.synthesize.assert_called_once_with("Hello, world!")
