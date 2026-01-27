"""
Whisper transcription service for real audio processing.
Uses OpenAI Whisper API for accurate speech-to-text transcription.
"""

import os
import logging
from typing import Optional, Dict, Any
from pathlib import Path

import openai
from ..utils.result import Result, Ok, Err, ProcessingError
from ..config.settings import settings

logger = logging.getLogger(__name__)


class WhisperTranscriber:
    """OpenAI Whisper API client for audio transcription."""

    def __init__(self):
        """Initialize Whisper transcriber with API key."""
        try:
            api_key = settings.openai_api_key
            if not api_key:
                raise ValueError("OPENAI_API_KEY not configured in settings")

            self.client = openai.OpenAI(
                api_key=api_key,
                timeout=120.0  # 2-minute timeout for large files
            )
            self.is_enabled = True
            logger.info("WhisperTranscriber initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize WhisperTranscriber: {e}")
            self.client = None
            self.is_enabled = False

    async def transcribe_audio(
        self,
        audio_file_path: str,
        language: Optional[str] = "en"
    ) -> Result[Dict[str, Any], ProcessingError]:
        """
        Transcribe audio file using OpenAI Whisper API.

        Args:
            audio_file_path: Path to the audio file (M4A, MP3, WAV, etc.)
            language: Language code (e.g., "en", "es", "fr") or None for auto-detect

        Returns:
            Result containing transcription data or error
        """
        if not self.is_enabled:
            return Err(ProcessingError("Whisper transcriber not available"))

        audio_path = Path(audio_file_path)
        if not audio_path.exists():
            return Err(ProcessingError(f"Audio file not found: {audio_file_path}"))

        try:
            logger.info(f"Starting Whisper transcription: {audio_path.name}")

            # Get file size for logging
            file_size_mb = audio_path.stat().st_size / (1024 * 1024)
            logger.info(f"Audio file size: {file_size_mb:.2f} MB")

            # Open and transcribe audio file
            with open(audio_path, "rb") as audio_file:
                # Call Whisper API with verbose JSON response
                transcript_response = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="verbose_json",
                    language=language,  # None for auto-detect, "en" for English
                    temperature=0.0  # Deterministic output
                )

            # Extract transcription data
            transcript_text = transcript_response.text
            transcript_language = getattr(transcript_response, 'language', language or 'en')
            transcript_duration = getattr(transcript_response, 'duration', None)
            segments = getattr(transcript_response, 'segments', [])

            # Log results
            logger.info(f"Transcription completed successfully")
            logger.info(f"  Text length: {len(transcript_text)} characters")
            logger.info(f"  Language: {transcript_language}")
            logger.info(f"  Duration: {transcript_duration} seconds" if transcript_duration else "  Duration: Unknown")
            logger.info(f"  Segments: {len(segments)}")

            # Return structured result
            result_data = {
                "text": transcript_text,
                "language": transcript_language,
                "duration_seconds": transcript_duration,
                "segments": segments,
                "word_count": len(transcript_text.split()) if transcript_text else 0,
                "file_size_mb": file_size_mb,
                "model": "whisper-1"
            }

            return Ok(result_data)

        except openai.APIError as e:
            logger.error(f"Whisper API error: {e}")
            return Err(ProcessingError(f"Whisper API error: {str(e)}"))

        except openai.RateLimitError as e:
            logger.error(f"Whisper rate limit exceeded: {e}")
            return Err(ProcessingError(f"Whisper rate limit exceeded. Please try again later."))

        except Exception as e:
            logger.error(f"Unexpected error during transcription: {e}")
            return Err(ProcessingError(f"Transcription failed: {str(e)}"))

    def test_connection(self) -> Result[bool, ProcessingError]:
        """
        Test the OpenAI API connection.

        Returns:
            Result indicating if the connection is working
        """
        if not self.is_enabled:
            return Err(ProcessingError("Whisper transcriber not initialized"))

        try:
            # Test with a simple API call
            models = self.client.models.list()
            whisper_available = any(model.id == "whisper-1" for model in models.data)

            if whisper_available:
                logger.info("Whisper API connection test successful")
                return Ok(True)
            else:
                return Err(ProcessingError("Whisper-1 model not available"))

        except Exception as e:
            logger.error(f"Whisper API connection test failed: {e}")
            return Err(ProcessingError(f"API connection failed: {str(e)}"))

    def estimate_cost(self, duration_minutes: float) -> float:
        """
        Estimate the cost for transcribing audio of given duration.

        Args:
            duration_minutes: Audio duration in minutes

        Returns:
            Estimated cost in USD
        """
        # OpenAI Whisper pricing: $0.006 per minute
        cost_per_minute = 0.006
        return duration_minutes * cost_per_minute

    def get_supported_formats(self) -> list[str]:
        """
        Get list of supported audio formats.

        Returns:
            List of supported file extensions
        """
        return [
            "mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm",
            "flac", "3gp", "aac", "ogg", "oga", "opus"
        ]

    def validate_audio_file(self, file_path: str) -> Result[bool, ProcessingError]:
        """
        Validate that the audio file is supported and accessible.

        Args:
            file_path: Path to the audio file

        Returns:
            Result indicating if the file is valid
        """
        audio_path = Path(file_path)

        # Check if file exists
        if not audio_path.exists():
            return Err(ProcessingError(f"Audio file does not exist: {file_path}"))

        # Check file extension
        file_extension = audio_path.suffix.lower().lstrip('.')
        supported_formats = self.get_supported_formats()

        if file_extension not in supported_formats:
            return Err(ProcessingError(
                f"Unsupported audio format: {file_extension}. "
                f"Supported formats: {', '.join(supported_formats)}"
            ))

        # Check file size (Whisper has 25MB limit)
        max_file_size_mb = 25
        file_size_mb = audio_path.stat().st_size / (1024 * 1024)

        if file_size_mb > max_file_size_mb:
            return Err(ProcessingError(
                f"Audio file too large: {file_size_mb:.1f}MB. "
                f"Maximum size: {max_file_size_mb}MB"
            ))

        logger.info(f"Audio file validation passed: {audio_path.name} ({file_size_mb:.2f}MB)")
        return Ok(True)


# Global instance
whisper_transcriber = WhisperTranscriber()