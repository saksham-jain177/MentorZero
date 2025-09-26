"""
Text-to-Speech service using ChatterBox from local-talking-llm
"""
import os
import json
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, Union

from agent.config import get_settings

# Check if ChatterBox is available
try:
    from chatterbox import Chatterbox
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False


class ChatterboxTTS:
    """Text-to-Speech service using ChatterBox"""
    
    def __init__(self, settings_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize the TTS service
        
        Args:
            settings_dict: ChatterBox settings, defaults to settings from env
        """
        self.settings = get_settings()
        
        # Parse TTS settings from env or use provided dict
        if settings_dict:
            self.tts_settings = settings_dict
        else:
            try:
                self.tts_settings = json.loads(self.settings.tts_settings or "{}")
            except (json.JSONDecodeError, TypeError):
                self.tts_settings = {}
                
        # Set defaults if not provided
        if not self.tts_settings.get("model"):
            self.tts_settings["model"] = "facebook/mms-tts-eng"
        
        self.device = self.settings.tts_device or "cpu"
        self.model = None
        
    def load_model(self):
        """
        Load the ChatterBox model on demand
        
        Returns:
            The loaded model
        """
        if not TTS_AVAILABLE:
            raise ImportError(
                "ChatterBox is not installed. Install it from https://github.com/vndee/local-talking-llm"
            )
            
        if self.model is None:
            self.model = Chatterbox(
                model=self.tts_settings.get("model"),
                device=self.device
            )
        return self.model
    
    async def generate(self, text: str, voice: str = "default") -> str:
        """
        Generate audio from text and return the path
        
        Args:
            text: Text to synthesize
            voice: Voice ID (not used in current implementation)
            
        Returns:
            Path to the generated audio file
        """
        if not TTS_AVAILABLE:
            return ""
            
        try:
            model = self.load_model()
            
            # Create temp file with unique name
            temp_dir = tempfile.gettempdir()
            temp_file = f"tts_{hash(text) & 0xFFFFFFFF}.wav"
            temp_path = os.path.join(temp_dir, temp_file)
            
            # Generate audio
            model.tts(
                text=text,
                output_path=temp_path,
                **self.tts_settings
            )
            
            return temp_path
                
        except Exception as e:
            print(f"TTS Error: {str(e)}")
            return ""
    
    async def synthesize(self, text: str) -> bytes:
        """
        Synthesize text to audio
        
        Args:
            text: Text to synthesize
            
        Returns:
            Audio bytes
        """
        if not TTS_AVAILABLE:
            return self.synthesize_fallback(text)
            
        try:
            model = self.load_model()
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp:
                temp_path = temp.name
            
            try:
                # Generate audio
                model.tts(
                    text=text,
                    output_path=temp_path,
                    **self.tts_settings
                )
                
                # Read audio bytes
                with open(temp_path, "rb") as f:
                    audio_bytes = f.read()
                    
                return audio_bytes
            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    
        except Exception as e:
            print(f"TTS Error: {str(e)}")
            return b""
            
    def synthesize_fallback(self, text: str) -> bytes:
        """
        Fallback synthesis when ChatterBox is not available
        
        Args:
            text: Text to synthesize
            
        Returns:
            Empty bytes
        """
        print("TTS is not available. Please install ChatterBox.")
        return b""