"""
Enhanced error handling utilities for MentorZero.
"""
from typing import Dict, Any, Optional
import traceback
import time
from datetime import datetime


class ErrorContext:
    """Context information for errors."""
    
    def __init__(
        self, 
        operation: str, 
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        topic: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.operation = operation
        self.user_id = user_id
        self.session_id = session_id
        self.topic = topic
        self.metadata = metadata or {}
        self.timestamp = time.time()


class MentorZeroError(Exception):
    """Base exception for MentorZero with enhanced context."""
    
    def __init__(
        self, 
        message: str, 
        context: Optional[ErrorContext] = None,
        recovery_suggestions: Optional[list[str]] = None,
        user_friendly_message: Optional[str] = None
    ):
        super().__init__(message)
        self.context = context
        self.recovery_suggestions = recovery_suggestions or []
        self.user_friendly_message = user_friendly_message or message


class AZLError(MentorZeroError):
    """AZL-specific errors."""
    pass


class RAGError(MentorZeroError):
    """RAG-specific errors."""
    pass


class LLMError(MentorZeroError):
    """LLM communication errors."""
    pass


class ErrorHandler:
    """Enhanced error handler with detailed feedback and recovery suggestions."""
    
    @staticmethod
    def handle_llm_error(
        error: Exception, 
        context: ErrorContext,
        model_name: str = "unknown"
    ) -> Dict[str, Any]:
        """Handle LLM-related errors with specific recovery suggestions."""
        
        error_type = type(error).__name__
        error_msg = str(error)
        
        # Common LLM error patterns and suggestions
        recovery_suggestions = []
        user_message = "AI model communication failed"
        
        if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
            recovery_suggestions = [
                "Try using a faster model (e.g., llama3.1:8b instead of gpt-oss:20b)",
                "Increase timeout settings in configuration",
                "Check if Ollama server is responsive",
                "Simplify the query to reduce processing time"
            ]
            user_message = "AI model took too long to respond. Try a simpler query or check your model settings."
            
        elif "connection" in error_msg.lower() or "refused" in error_msg.lower():
            recovery_suggestions = [
                "Ensure Ollama server is running on http://localhost:11434",
                "Check if the specified model is downloaded: ollama list",
                "Restart Ollama service if needed",
                "Verify network connectivity to Ollama server"
            ]
            user_message = "Cannot connect to AI model. Please check if Ollama is running."
            
        elif "model" in error_msg.lower() and "not found" in error_msg.lower():
            recovery_suggestions = [
                f"Download the missing model: ollama pull {model_name}",
                "Check available models: ollama list",
                "Use a different model that's already downloaded",
                "Update model name in configuration"
            ]
            user_message = f"AI model '{model_name}' not found. Please download it first."
            
        elif "json" in error_msg.lower() or "parse" in error_msg.lower():
            recovery_suggestions = [
                "Retry the operation - sometimes models produce invalid output",
                "Use a more reliable model for structured output",
                "Check if the prompt is clear and specific",
                "Enable fallback parsing in configuration"
            ]
            user_message = "AI model produced invalid output. Please try again."
            
        else:
            recovery_suggestions = [
                "Retry the operation",
                "Check Ollama server logs for details",
                "Try a different model if available",
                "Contact support if problem persists"
            ]
        
        return {
            "error_type": "llm_error",
            "error_class": error_type,
            "message": error_msg,
            "user_message": user_message,
            "recovery_suggestions": recovery_suggestions,
            "context": {
                "operation": context.operation,
                "model": model_name,
                "timestamp": datetime.fromtimestamp(context.timestamp).isoformat(),
                "metadata": context.metadata
            },
            "debug_info": {
                "traceback": traceback.format_exc(),
                "user_id": context.user_id,
                "session_id": context.session_id,
                "topic": context.topic
            }
        }
    
    @staticmethod
    def handle_azl_error(
        error: Exception, 
        context: ErrorContext,
        phase: str = "unknown"
    ) -> Dict[str, Any]:
        """Handle AZL-specific errors."""
        
        error_msg = str(error)
        recovery_suggestions = []
        user_message = "AutoLearn process failed"
        
        if phase == "generation":
            recovery_suggestions = [
                "Try a more specific topic",
                "Reduce the number of examples to generate",
                "Check if the model is suitable for educational content",
                "Verify model has sufficient knowledge about the topic"
            ]
            user_message = "Failed to generate learning examples. Try a more specific topic."
            
        elif phase == "evaluation":
            recovery_suggestions = [
                "Lower the quality threshold temporarily",
                "Use a faster judge model",
                "Check if examples are too complex",
                "Increase maximum attempts"
            ]
            user_message = "Failed to evaluate learning examples. Consider adjusting quality settings."
            
        elif phase == "regeneration":
            recovery_suggestions = [
                "Increase maximum regeneration attempts",
                "Use a more creative model (higher temperature)",
                "Simplify the topic or make it more specific",
                "Check if the original examples were valid"
            ]
            user_message = "Failed to improve learning examples. Try a different approach."
        
        return {
            "error_type": "azl_error",
            "error_class": type(error).__name__,
            "message": error_msg,
            "user_message": user_message,
            "recovery_suggestions": recovery_suggestions,
            "context": {
                "operation": context.operation,
                "phase": phase,
                "timestamp": datetime.fromtimestamp(context.timestamp).isoformat(),
                "metadata": context.metadata
            },
            "debug_info": {
                "traceback": traceback.format_exc(),
                "user_id": context.user_id,
                "session_id": context.session_id,
                "topic": context.topic
            }
        }
    
    @staticmethod
    def handle_rag_error(
        error: Exception, 
        context: ErrorContext,
        phase: str = "unknown"
    ) -> Dict[str, Any]:
        """Handle RAG-specific errors."""
        
        error_msg = str(error)
        recovery_suggestions = []
        user_message = "Document processing failed"
        
        if phase == "ingestion":
            recovery_suggestions = [
                "Check if the document format is supported",
                "Try uploading smaller chunks of text",
                "Verify the text encoding is correct",
                "Ensure sufficient disk space for indexing"
            ]
            user_message = "Failed to process document. Try uploading smaller text chunks."
            
        elif phase == "retrieval":
            recovery_suggestions = [
                "Try a more specific search query",
                "Check if relevant documents have been uploaded",
                "Verify the vector index is not corrupted",
                "Try rebuilding the search index"
            ]
            user_message = "Failed to find relevant information. Try a more specific query."
            
        elif phase == "embedding":
            recovery_suggestions = [
                "Check if embedding model is available",
                "Try using the fallback hash embeddings",
                "Restart the embedding service",
                "Verify model files are not corrupted"
            ]
            user_message = "Failed to process text for search. Please try again."
        
        return {
            "error_type": "rag_error",
            "error_class": type(error).__name__,
            "message": error_msg,
            "user_message": user_message,
            "recovery_suggestions": recovery_suggestions,
            "context": {
                "operation": context.operation,
                "phase": phase,
                "timestamp": datetime.fromtimestamp(context.timestamp).isoformat(),
                "metadata": context.metadata
            },
            "debug_info": {
                "traceback": traceback.format_exc(),
                "user_id": context.user_id,
                "session_id": context.session_id,
                "topic": context.topic
            }
        }
    
    @staticmethod
    def handle_generic_error(
        error: Exception, 
        context: ErrorContext
    ) -> Dict[str, Any]:
        """Handle generic errors with basic recovery suggestions."""
        
        return {
            "error_type": "generic_error",
            "error_class": type(error).__name__,
            "message": str(error),
            "user_message": "An unexpected error occurred. Please try again.",
            "recovery_suggestions": [
                "Try the operation again",
                "Check your input for any issues",
                "Contact support if the problem persists"
            ],
            "context": {
                "operation": context.operation,
                "timestamp": datetime.fromtimestamp(context.timestamp).isoformat(),
                "metadata": context.metadata
            },
            "debug_info": {
                "traceback": traceback.format_exc(),
                "user_id": context.user_id,
                "session_id": context.session_id,
                "topic": context.topic
            }
        }