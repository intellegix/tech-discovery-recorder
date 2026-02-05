"""
Minimal Claude service for testing without external dependencies.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List

from config.settings import settings
from utils.result import Result, Ok, Err, ProcessingError, ExternalServiceError
from models.schemas import StructuredNotes, ProcessingTypeEnum


logger = logging.getLogger(__name__)


class ClaudeService:
    """
    Minimal Claude service for testing without external dependencies.
    """

    def __init__(self):
        self.is_demo_mode = True  # Always demo mode for testing
        logger.info("ClaudeService initialized in demo mode for testing")

    async def transcribe_audio(self, audio_content: bytes, audio_format: str) -> Result[str, ProcessingError]:
        """Generate demo transcript."""
        try:
            logger.info(f"Demo mode: generating mock transcript for {audio_format}")
            await asyncio.sleep(1)  # Simulate processing time
            return Ok(self._generate_demo_transcript(audio_content))
        except Exception as e:
            logger.error(f"Demo transcription failed: {str(e)}")
            return Err(ProcessingError(f"Demo transcription failed: {str(e)}"))

    async def extract_structured_notes(
        self,
        transcript: str,
        context: Optional[str],
        participants: List[str],
        processing_type: ProcessingTypeEnum,
        custom_prompt: Optional[str] = None
    ) -> Result[StructuredNotes, ProcessingError]:
        """Extract structured notes in demo mode."""
        try:
            logger.info(f"Demo mode: extracting structured notes with type {processing_type}")
            await asyncio.sleep(2)  # Simulate processing time

            # Generate demo structured notes
            if processing_type == ProcessingTypeEnum.CUSTOM and custom_prompt:
                notes = StructuredNotes(
                    action_items=[f"Custom action based on: {custom_prompt[:50]}..."],
                    decisions=[f"Decision extracted using custom prompt"],
                    architecture_details=[f"Architecture detail from custom analysis"],
                    follow_ups=[f"Follow-up from custom processing"],
                    participants_summary={p: f"Contributed via custom analysis" for p in participants[:3]},
                    technical_specs=[f"Technical spec from custom prompt"],
                    risks_concerns=[f"Risk identified in custom analysis"]
                )
            else:
                notes = self._generate_demo_structured_notes(participants, context)

            return Ok(notes)
        except Exception as e:
            logger.error(f"Demo structured notes extraction failed: {str(e)}")
            return Err(ProcessingError(f"Demo structured notes extraction failed: {str(e)}"))

    async def process_recording(
        self,
        audio_content: bytes,
        audio_format: str,
        context: Optional[str],
        participants: List[str],
        processing_type: ProcessingTypeEnum,
        custom_prompt: Optional[str] = None
    ) -> Result[Dict[str, Any], ProcessingError]:
        """Complete processing pipeline in demo mode."""
        try:
            processing_start = datetime.now()
            logger.info(f"Demo mode: starting complete processing pipeline for {processing_type}")

            # Step 1: Transcribe audio
            transcript_result = await self.transcribe_audio(audio_content, audio_format)
            if transcript_result.is_err():
                return transcript_result

            transcript = transcript_result.unwrap()

            # Step 2: Extract structured notes
            notes_result = await self.extract_structured_notes(
                transcript, context, participants, processing_type, custom_prompt
            )
            if notes_result.is_err():
                return notes_result

            structured_notes = notes_result.unwrap()
            processing_end = datetime.now()

            # Compile results
            result = {
                "transcript": transcript,
                "structured_notes": structured_notes,
                "processing_metadata": {
                    "claude_model": "demo-mode",
                    "processing_type": processing_type,
                    "processing_duration_seconds": (processing_end - processing_start).total_seconds(),
                    "transcript_length": len(transcript),
                    "processed_at": processing_end.isoformat(),
                    "demo_mode": True
                }
            }

            logger.info(f"Demo processing pipeline completed in {result['processing_metadata']['processing_duration_seconds']:.2f}s")
            return Ok(result)

        except Exception as e:
            logger.error(f"Demo processing pipeline failed: {str(e)}")
            return Err(ProcessingError(f"Demo processing pipeline failed: {str(e)}"))

    async def health_check(self) -> Result[Dict[str, Any], ExternalServiceError]:
        """Demo health check."""
        return Ok({
            "claude_api_healthy": True,
            "model": "demo-mode",
            "response_received": True,
            "tested_at": datetime.now().isoformat(),
            "demo_mode": True
        })

    def _generate_demo_transcript(self, audio_content: bytes) -> str:
        """Generate demo transcript based on audio file size."""
        file_size_kb = len(audio_content) / 1024
        duration_estimate = max(10, int(file_size_kb / 16))  # Rough estimate

        return f"""[Demo Transcript - {duration_estimate}s recording]

Austin: Good morning everyone, thanks for joining our technical discovery call. I'm recording this for documentation purposes and we'll use our AI transcription system to generate structured notes. Does everyone consent to recording?

Client Lead: Yes, that works for me.

Technical Lead: Sounds good, I consent as well.

Austin: Perfect. So today we're discussing the integration architecture for the new system. Can you walk us through the current technical setup?

Technical Lead: Sure, we currently have a microservices architecture with REST APIs. The main challenge we're facing is handling data synchronization between services.

Client Lead: We also need to ensure high availability and scalability for our growing user base.

Austin: That makes sense. What about your current database architecture? Are there any performance concerns?

Technical Lead: We're using PostgreSQL for transactional data and Redis for caching. We might need to consider sharding if we scale beyond our current projections.

Austin: Good points. Let's document these requirements and create a roadmap for the integration.

[End of transcript]"""

    def _generate_demo_structured_notes(self, participants: List[str], context: Optional[str]) -> StructuredNotes:
        """Generate demo structured notes."""
        context_items = []
        if context:
            context_items = [f"Address requirement: {context[:50]}..."]

        return StructuredNotes(
            action_items=[
                "Document current microservices architecture and API specifications",
                "Analyze data synchronization requirements between services",
                "Evaluate database sharding strategy for scalability",
                "Create integration roadmap with timeline and milestones"
            ] + context_items,
            decisions=[
                "Continue with PostgreSQL for transactional data storage",
                "Maintain Redis caching layer for performance optimization",
                "Implement microservices integration with REST APIs",
                "Plan for horizontal scaling with database sharding"
            ],
            architecture_details=[
                "Current microservices architecture with REST API communication",
                "PostgreSQL database for transactional data storage",
                "Redis caching layer for performance optimization",
                "Data synchronization challenges between distributed services",
                "High availability and scalability requirements for growth"
            ],
            follow_ups=[
                "Research database sharding implementation options",
                "Design data synchronization strategy between microservices",
                "Create detailed integration timeline and resource requirements",
                "Evaluate monitoring and observability tools for distributed system"
            ],
            participants_summary={
                participants[0] if participants else "Lead": "Facilitated discussion and documentation requirements",
                participants[1] if len(participants) > 1 else "Technical Lead": "Provided technical architecture context and scaling concerns",
                participants[2] if len(participants) > 2 else "Client Lead": "Highlighted business requirements for availability and growth"
            },
            technical_specs=[
                "Microservices architecture with REST API endpoints",
                "PostgreSQL database for ACID transactional requirements",
                "Redis caching for sub-second response time optimization",
                "Horizontal scaling preparation for database sharding",
                "High availability architecture for business continuity"
            ],
            risks_concerns=[
                "Data consistency challenges in distributed microservices architecture",
                "Database performance bottlenecks at scale without proper sharding",
                "Service coordination complexity with multiple API dependencies",
                "Monitoring and debugging complexity in distributed system"
            ]
        )


# Global service instance
claude_service = ClaudeService()