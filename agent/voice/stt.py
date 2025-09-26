"""
Speech-to-Text service using Whisper from local-talking-llm
"""
import os
import tempfile
from pathlib import Path
from typing import Optional, Union

from agent.config import get_settings

# Check if Whisper is available
try:
    import whisper
    STT_AVAILABLE = True
except ImportError:
    STT_AVAILABLE = False


class WhisperSTT:
    """Speech-to-Text service using Whisper"""
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the STT service
        
        Args:
            model_name: Whisper model name, defaults to settings or "tiny"
        """
        self.settings = get_settings()
        self.model_name = model_name or self.settings.whisper_model or "tiny"
        self.device = self.settings.tts_device or "cpu"
        self.model = None
        
    def load_model(self):
        """
        Load the Whisper model on demand
        
        Returns:
            The loaded model
        """
        if not STT_AVAILABLE:
            raise ImportError(
                "Whisper is not installed. Install it with: pip install openai-whisper"
            )
            
        if self.model is None:
            self.model = whisper.load_model(self.model_name, device=self.device)
        return self.model
    
    async def transcribe(self, audio_file: Union[str, Path, bytes], language: str = "en") -> str:
        """
        Transcribe audio to text
        
        Args:
            audio_file: Path to audio file or bytes
            language: Language code
            
        Returns:
            Transcribed text
        """
        if not STT_AVAILABLE:
            return self.transcribe_fallback(audio_file)
            
        try:
            model = self.load_model()
            
            # Handle bytes input
            if isinstance(audio_file, bytes):
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp:
                    temp.write(audio_file)
                    temp_path = temp.name
                
                try:
                    result = model.transcribe(temp_path, language=language)
                    return result["text"].strip()
                finally:
                    # Clean up temp file
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
            else:
                # Handle file path
                result = model.transcribe(str(audio_file), language=language)
                return result["text"].strip()
                
        except Exception as e:
            print(f"STT Error: {str(e)}")
            return ""
            
    def transcribe_fallback(self, audio_file: Union[str, Path, bytes]) -> str:
        """
        Fallback transcription when Whisper is not available
        
        Args:
            audio_file: Path to audio file or bytes
            
        Returns:
            Message about Whisper not being available
        """
        return "Speech-to-text is not available. Please install Whisper with: pip install openai-whisper"