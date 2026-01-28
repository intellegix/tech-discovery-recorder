"""
Claude AI service for audio transcription and structured note extraction.
Implements LangChain orchestration following Austin's patterns.
Now includes OpenAI Whisper for real audio transcription.
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional, List
from io import BytesIO
# Optional imports for audio processing - only needed if not using demo mode
try:
    import speech_recognition as sr
    from pydub import AudioSegment
    AUDIO_PROCESSING_AVAILABLE = True
except ImportError:
    sr = None
    AudioSegment = None
    AUDIO_PROCESSING_AVAILABLE = False
# Optional imports for Claude API integration
try:
    import anthropic
    CLAUDE_API_AVAILABLE = True
except ImportError as e:
    anthropic = None
    CLAUDE_API_AVAILABLE = False

from ..config.settings import settings
from ..utils.result import Result, Ok, Err, ProcessingError, ExternalServiceError
from ..models.schemas import StructuredNotes, ProcessingTypeEnum

# Import Whisper transcription service
try:
    from .whisper_service import whisper_transcriber
    WHISPER_AVAILABLE = True
except ImportError:
    whisper_transcriber = None
    WHISPER_AVAILABLE = False

# Import PDF generation service
try:
    from .pdf_service import pdf_generator
    PDF_GENERATION_AVAILABLE = True
except ImportError:
    pdf_generator = None
    PDF_GENERATION_AVAILABLE = False

logger = logging.getLogger(__name__)


class ClaudeService:
    """
    Handles audio transcription and AI processing via Claude API.
    Implements the processing pipeline from the architecture plan.
    """

    def __init__(self):
        # Check if we have a real Claude API key and dependencies
        if (CLAUDE_API_AVAILABLE and
            settings.claude_api_key and
            settings.claude_api_key.startswith('sk-ant-') and
            len(settings.claude_api_key) > 20):
            self.client = anthropic.Anthropic(api_key=settings.claude_api_key)
            self.is_demo_mode = False
            logger.info("ClaudeService initialized with real API")
        else:
            # Demo mode - no real API calls
            self.client = None
            self.is_demo_mode = True
            logger.warning("Running in demo mode - Claude API not configured")

        self.recognizer = sr.Recognizer() if sr else None

    async def transcribe_audio(self, audio_content: bytes, audio_format: str) -> Result[str, ProcessingError]:
        """
        Transcribe audio using OpenAI Whisper for real transcription.
        Falls back to demo content only if Whisper is unavailable.
        """
        try:
            logger.info(f"Starting transcription for audio format: {audio_format}")

            # Try Whisper transcription first (real audio processing)
            if WHISPER_AVAILABLE and whisper_transcriber.is_enabled:
                logger.info("Using OpenAI Whisper for real audio transcription")
                return await self._transcribe_with_whisper(audio_content, audio_format)

            # Fallback to demo mode if Whisper unavailable
            if self.is_demo_mode:
                logger.warning("Whisper unavailable, using demo mode")
                await asyncio.sleep(1)  # Simulate processing time
                return Ok(self._generate_demo_transcript(audio_content))

            # Original Claude API approach (not implemented for audio files)
            return await self._transcribe_with_claude(audio_content, audio_format)

            # Convert audio to WAV format for speech_recognition
            audio_data = self._convert_audio_to_wav(audio_content, audio_format)
            if audio_data.is_err():
                return audio_data

            # Perform transcription
            with BytesIO(audio_data.unwrap()) as audio_file:
                with sr.AudioFile(audio_file) as source:
                    audio = self.recognizer.record(source)

                # Use Google Speech Recognition as fallback
                # In production, this would be replaced with Claude's Whisper API
                try:
                    transcript = self.recognizer.recognize_google(audio)
                    logger.info("Transcription completed successfully")
                    return Ok(transcript)
                except sr.UnknownValueError:
                    return Err(ProcessingError("Could not understand audio"))
                except sr.RequestError as e:
                    return Err(ProcessingError(f"Speech recognition service error: {str(e)}"))

        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            return Err(ProcessingError(f"Transcription failed: {str(e)}"))

    async def _transcribe_with_whisper(self, audio_content: bytes, audio_format: str) -> Result[str, ProcessingError]:
        """
        Transcribe audio using OpenAI Whisper API.
        Saves audio to temporary file and calls Whisper service.
        """
        import tempfile
        import os

        temp_file_path = None
        try:
            logger.info(f"Starting Whisper transcription for {len(audio_content)} bytes of {audio_format}")

            # Determine file extension
            ext_mapping = {
                'audio/m4a': '.m4a',
                'audio/mp4': '.m4a',
                'audio/wav': '.wav',
                'audio/mp3': '.mp3',
                'audio/webm': '.webm'
            }
            file_extension = ext_mapping.get(audio_format, '.m4a')

            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as temp_file:
                temp_file.write(audio_content)
                temp_file_path = temp_file.name

            logger.info(f"Saved audio to temporary file: {temp_file_path}")

            # Validate audio file
            validation_result = whisper_transcriber.validate_audio_file(temp_file_path)
            if validation_result.is_err():
                return Err(ProcessingError(f"Audio validation failed: {validation_result.unwrap_or('Unknown error')}"))

            # Transcribe with Whisper
            transcription_result = await whisper_transcriber.transcribe_audio(temp_file_path)

            if transcription_result.is_err():
                return Err(ProcessingError(f"Whisper transcription failed: {transcription_result.unwrap_or('Unknown error')}"))

            # Extract transcript text
            transcript_data = transcription_result.unwrap()
            transcript_text = transcript_data.get('text', '')

            # Log transcription details
            logger.info(f"Whisper transcription completed successfully")
            logger.info(f"  Text length: {len(transcript_text)} characters")
            logger.info(f"  Language: {transcript_data.get('language', 'unknown')}")
            logger.info(f"  Duration: {transcript_data.get('duration_seconds', 'unknown')} seconds")
            logger.info(f"  Word count: {transcript_data.get('word_count', 0)} words")

            return Ok(transcript_text)

        except Exception as e:
            logger.error(f"Whisper transcription failed: {str(e)}")
            return Err(ProcessingError(f"Whisper transcription failed: {str(e)}"))

        finally:
            # Clean up temporary file
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                    logger.debug(f"Cleaned up temporary file: {temp_file_path}")
                except OSError as e:
                    logger.warning(f"Failed to clean up temporary file {temp_file_path}: {e}")

    async def _enhance_transcript_with_ai(
        self,
        raw_transcript: str,
        participants: List[str],
        context: Optional[str]
    ) -> Result[str, ProcessingError]:
        """
        Enhance raw transcript with AI-powered punctuation, speaker diarization, and formatting.

        Args:
            raw_transcript: Raw speech-to-text output from Whisper
            participants: List of participant names for speaker identification
            context: Meeting/recording context to help with speaker identification

        Returns:
            Enhanced transcript with proper punctuation and speaker labels
        """
        try:
            logger.info("Enhancing transcript with AI-powered punctuation and speaker diarization")

            if not self.client:
                return Err(ProcessingError("Claude API client not available"))

            # Build context-aware prompt for transcript enhancement
            participant_info = ""
            if participants:
                participant_info = f"Participants: {', '.join(participants)}"

            context_info = f"\nContext: {context}" if context else ""

            system_prompt = f"""You are an expert transcript editor specializing in business and technical discussions. Your task is to enhance raw speech-to-text output by:

1. PUNCTUATION & FORMATTING:
   - Add proper punctuation (periods, commas, question marks, exclamation points)
   - Fix capitalization and sentence structure
   - Add paragraph breaks for natural speech flow
   - Preserve the exact words spoken, only add punctuation and formatting

2. SPEAKER DIARIZATION:
   - Identify different speakers based on context clues, speaking patterns, and content
   - Use participant names when possible: {participant_info}
   - Look for natural conversation cues (responses, questions, topic shifts)
   - Pay attention to tone changes, technical vs casual language
   - Format as: "SPEAKER_NAME: [content]"

3. TONE & CONTEXT ANALYSIS:
   - Identify formal vs informal speech patterns
   - Notice technical explanations vs casual conversation
   - Recognize questions vs statements vs decisions
   - Maintain the natural flow and personality of each speaker

{context_info}

IMPORTANT GUIDELINES:
- Never change the actual words spoken, only add punctuation and structure
- Use participant names when identifiable, otherwise use "Speaker 1", "Speaker 2", etc.
- Preserve all technical terms and proper nouns exactly as spoken
- Maintain natural speech patterns and hesitations where appropriate
- Focus on clarity while preserving authenticity"""

            # Create the enhancement request
            message = self.client.messages.create(
                model=settings.claude_model,
                max_tokens=settings.claude_max_tokens,
                temperature=0.2,  # Low temperature for consistent formatting
                system=system_prompt,
                messages=[{
                    "role": "user",
                    "content": f"Please enhance this raw transcript with proper punctuation, speaker identification, and formatting:\n\n{raw_transcript}"
                }]
            )

            enhanced_transcript = message.content[0].text.strip()

            logger.info(f"Successfully enhanced transcript: {len(raw_transcript)} -> {len(enhanced_transcript)} chars")

            return Ok(enhanced_transcript)

        except Exception as e:
            logger.error(f"Transcript enhancement failed: {str(e)}")
            # Fallback to raw transcript with basic punctuation
            basic_enhanced = self._add_basic_punctuation(raw_transcript)
            logger.info("Falling back to basic punctuation enhancement")
            return Ok(basic_enhanced)

    def _add_basic_punctuation(self, text: str) -> str:
        """Fallback method to add basic punctuation if AI enhancement fails."""
        # Basic sentence detection and capitalization
        sentences = text.split('.')
        enhanced_sentences = []

        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                # Capitalize first letter
                sentence = sentence[0].upper() + sentence[1:] if len(sentence) > 1 else sentence.upper()
                enhanced_sentences.append(sentence)

        # Join with periods and add basic formatting
        enhanced = '. '.join(enhanced_sentences)
        if enhanced and not enhanced.endswith('.'):
            enhanced += '.'

        return enhanced

    async def _transcribe_with_claude(self, audio_content: bytes, audio_format: str) -> Result[str, ProcessingError]:
        """Transcribe audio using Claude API directly."""
        try:
            logger.info(f"Starting Claude API transcription for {len(audio_content)} bytes of {audio_format}")

            if not self.client:
                return Err(ProcessingError("Claude API client not available"))

            # Convert audio to base64 for Claude API
            import base64
            audio_b64 = base64.b64encode(audio_content).decode('utf-8')

            # Use Claude API for transcription
            message = self.client.messages.create(
                model=settings.claude_model,
                max_tokens=settings.claude_max_tokens,
                temperature=0.1,  # Low temperature for accurate transcription
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Please transcribe the audio content in this file. Provide only the transcript text without any explanations or formatting."
                        },
                        {
                            "type": "document",
                            "source": {
                                "type": "base64",
                                "media_type": audio_format,
                                "data": audio_b64
                            }
                        }
                    ]
                }]
            )

            transcript = message.content[0].text.strip()
            logger.info(f"Claude transcription completed: {len(transcript)} characters")
            return Ok(transcript)

        except Exception as e:
            logger.error(f"Claude transcription failed: {str(e)}")
            # Fallback to demo transcript
            logger.info("Falling back to demo transcript")
            return Ok(self._generate_demo_transcript(audio_content))

    def _convert_audio_to_wav(self, audio_content: bytes, audio_format: str) -> Result[bytes, ProcessingError]:
        """Convert audio to WAV format for processing."""
        try:
            if not AUDIO_PROCESSING_AVAILABLE:
                return Err(ProcessingError("Audio processing dependencies not available"))
            # Map MIME types to pydub formats
            format_map = {
                "audio/webm": "webm",
                "audio/wav": "wav",
                "audio/mp4": "mp4",
                "audio/m4a": "m4a",
                "audio/mpeg": "mp3"
            }

            source_format = format_map.get(audio_format, "mp4")

            # Load and convert audio
            audio = AudioSegment.from_file(BytesIO(audio_content), format=source_format)

            # Convert to WAV
            wav_buffer = BytesIO()
            audio.export(wav_buffer, format="wav")
            wav_buffer.seek(0)

            return Ok(wav_buffer.read())

        except Exception as e:
            return Err(ProcessingError(f"Audio conversion failed: {str(e)}"))

    async def extract_structured_notes(
        self,
        transcript: str,
        context: Optional[str],
        participants: List[str],
        processing_type: ProcessingTypeEnum,
        custom_prompt: Optional[str] = None
    ) -> Result[StructuredNotes, ProcessingError]:
        """
        Extract structured technical notes from transcript using Claude.
        """
        try:
            logger.info(f"Starting structured note extraction with processing type: {processing_type}")

            # Build the processing prompt based on type
            prompt_result = self._build_processing_prompt(
                transcript, context, participants, processing_type, custom_prompt
            )
            if prompt_result.is_err():
                return prompt_result

            # Call Claude API via LangChain
            response_result = await self._call_claude_api(prompt_result.unwrap())
            if response_result.is_err():
                return response_result

            # Parse the response into structured notes
            notes_result = self._parse_claude_response(response_result.unwrap(), processing_type)
            if notes_result.is_err():
                return notes_result

            logger.info("Structured note extraction completed successfully")
            return notes_result

        except Exception as e:
            logger.error(f"Structured note extraction failed: {str(e)}")
            return Err(ProcessingError(f"Note extraction failed: {str(e)}"))

    def _build_processing_prompt(
        self,
        transcript: str,
        context: Optional[str],
        participants: List[str],
        processing_type: ProcessingTypeEnum,
        custom_prompt: Optional[str]
    ) -> Result[str, ProcessingError]:
        """Build the appropriate prompt based on processing type."""

        participant_list = ", ".join(participants)
        context_section = f"\n\nContext provided: {context}" if context else ""

        if processing_type == ProcessingTypeEnum.CUSTOM and custom_prompt:
            prompt = f"""
            You are processing a technical discovery call transcript.

            Participants: {participant_list}{context_section}

            Custom instructions: {custom_prompt}

            Transcript:
            {transcript}

            Please follow the custom instructions above and return your analysis.
            """

        elif processing_type == ProcessingTypeEnum.SUMMARY:
            prompt = f"""
            You are processing a technical discovery call transcript. Provide a concise summary.

            Participants: {participant_list}{context_section}

            Transcript:
            {transcript}

            Please provide a clear, concise summary of the key points discussed in this technical call.
            Focus on decisions made, technical details, and any next steps identified.
            """

        elif processing_type == ProcessingTypeEnum.TECHNICAL:
            prompt = f"""
            You are processing a technical discovery call transcript. Extract technical specifications and architecture details.

            Participants: {participant_list}{context_section}

            Transcript:
            {transcript}

            Please extract and organize:
            1. Technical specifications mentioned
            2. Architecture decisions
            3. System integrations discussed
            4. Technology stack components
            5. Performance requirements
            6. Security considerations

            Return as structured technical documentation.
            """

        else:  # FULL processing
            prompt = f"""
            You are an expert technical discovery analyst with deep expertise in software architecture, business analysis, and project management. Your role is to transform raw meeting transcripts into comprehensive, actionable documentation that drives successful project outcomes.

            **Meeting Context:**
            Participants: {participant_list}{context_section}

            **Analysis Instructions:**
            Create a thorough, well-organized analysis that captures not just what was said, but the strategic implications and next steps. Be creative in identifying patterns, connections, and opportunities that might not be explicitly stated.

            **Transcript to Analyze:**
            {transcript}

            **Required Output Format (JSON):**
            Please provide a comprehensive analysis in the following JSON structure. For each section, be detailed, specific, and explanatory:

            {{
                "executive_summary": "A compelling 2-3 sentence overview of the meeting's key outcomes and strategic direction",

                "action_items": [
                    {{
                        "task": "Detailed description of the action item",
                        "owner": "Person responsible (if mentioned) or 'TBD'",
                        "priority": "High/Medium/Low",
                        "context": "Why this action is important and how it fits into the bigger picture",
                        "success_criteria": "What does successful completion look like"
                    }}
                ],

                "decisions": [
                    {{
                        "decision": "Clear statement of what was decided",
                        "rationale": "The reasoning behind this decision",
                        "implications": "How this decision impacts the project/system/organization",
                        "stakeholders": ["Who is affected by this decision"]
                    }}
                ],

                "architecture_details": [
                    {{
                        "component": "System/technology component discussed",
                        "description": "Detailed explanation of the architectural element",
                        "integration_points": "How this connects to other systems",
                        "scalability_notes": "Performance and scaling considerations",
                        "trade_offs": "Benefits and potential challenges"
                    }}
                ],

                "follow_ups": [
                    {{
                        "item": "What needs to be followed up on",
                        "urgency": "When this needs to be addressed",
                        "dependencies": "What other items this depends on",
                        "estimated_effort": "Rough effort estimate (Small/Medium/Large)"
                    }}
                ],

                "participants_analysis": {{
                    "participant_name": {{
                        "role_in_meeting": "Their primary function/perspective",
                        "key_contributions": ["Main points they raised"],
                        "concerns_raised": ["Any issues or challenges they highlighted"],
                        "expertise_demonstrated": "What domain knowledge they showed"
                    }}
                }},

                "technical_specifications": [
                    {{
                        "requirement": "Specific technical requirement",
                        "category": "Performance/Security/Integration/UI/etc",
                        "business_justification": "Why this requirement exists",
                        "implementation_notes": "Technical approach or considerations",
                        "acceptance_criteria": "How to verify this is implemented correctly"
                    }}
                ],

                "risks_and_opportunities": {{
                    "risks": [
                        {{
                            "risk": "Description of the potential risk",
                            "probability": "High/Medium/Low",
                            "impact": "Severity if this occurs",
                            "mitigation_strategies": ["Ways to reduce or eliminate this risk"]
                        }}
                    ],
                    "opportunities": [
                        {{
                            "opportunity": "Potential benefit or optimization",
                            "value": "What value this could provide",
                            "effort_required": "What it would take to realize this",
                            "timeline": "When this could be implemented"
                        }}
                    ]
                }},

                "next_meeting_agenda": [
                    "Suggested agenda items for follow-up meetings based on this discussion"
                ],

                "strategic_insights": [
                    "Higher-level observations about the project direction, team dynamics, or business implications"
                ]
            }}

            **Analysis Guidelines:**
            - Be thorough yet concise - every entry should add value
            - Think beyond the transcript - what are the implications?
            - Identify patterns, connections, and potential issues not explicitly discussed
            - Use professional but engaging language
            - Focus on actionability - how can this information drive decisions?
            - Consider both technical and business perspectives
            - Highlight dependencies and relationships between different elements
            """

        return Ok(prompt)

    async def _call_claude_api(self, prompt: str) -> Result[str, ExternalServiceError]:
        """Call Claude API directly with error handling."""
        try:
            if self.is_demo_mode:
                # Demo mode - return mock structured response
                logger.info("Demo mode: generating mock Claude response")
                await asyncio.sleep(2)  # Simulate API processing time
                return Ok(self._generate_demo_structured_response())

            if not CLAUDE_API_AVAILABLE or not self.client:
                # Fallback to demo mode if dependencies not available
                return Err(ExternalServiceError("Claude API dependencies not available"))

            # System prompt for structured analysis
            system_prompt = """You are an elite business and technical analyst with 15+ years of experience in software architecture, project management, and strategic planning. Your expertise spans:

            - Technical architecture and system design
            - Agile project management and delivery
            - Risk assessment and mitigation strategies
            - Stakeholder analysis and communication
            - Business process optimization
            - Technology vendor evaluation

            Your role is to transform raw meeting transcripts into strategic, actionable intelligence that drives successful project outcomes. You excel at:

            1. **Pattern Recognition**: Identifying underlying themes, dependencies, and potential issues not explicitly stated
            2. **Strategic Thinking**: Understanding how tactical decisions fit into broader business objectives
            3. **Risk Assessment**: Proactively identifying potential challenges and opportunities
            4. **Clear Communication**: Organizing complex information into actionable, well-structured insights

            Always provide:
            - Comprehensive yet focused analysis
            - Clear reasoning behind your assessments
            - Practical next steps with context
            - Valid, well-formatted JSON responses
            - Professional yet engaging language

            Think like a senior consultant who needs to brief executives and technical teams with precision and strategic insight."""

            # Call Claude API directly
            response = self.client.messages.create(
                model=settings.claude_model,
                max_tokens=settings.claude_max_tokens,
                temperature=0.3,  # Slightly creative for rich analysis
                system=system_prompt,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )

            if not response or not response.content:
                return Err(ExternalServiceError("Empty response from Claude API"))

            # Extract text from response
            response_text = response.content[0].text if response.content else ""
            return Ok(response_text)

        except anthropic.APIError as e:
            logger.error(f"Claude API error: {str(e)}")
            return Err(ExternalServiceError(f"Claude API error: {str(e)}"))
        except Exception as e:
            logger.error(f"Unexpected error calling Claude API: {str(e)}")
            return Err(ExternalServiceError(f"Claude API call failed: {str(e)}"))

    def _parse_claude_response(
        self,
        response: str,
        processing_type: ProcessingTypeEnum
    ) -> Result[StructuredNotes, ProcessingError]:
        """Parse Claude's response into StructuredNotes."""
        try:
            if processing_type == ProcessingTypeEnum.FULL:
                # Try to parse as JSON first
                return self._parse_json_response(response)
            else:
                # For other types, create a simple StructuredNotes with the response
                return Ok(StructuredNotes(
                    action_items=[],
                    decisions=[response] if processing_type == ProcessingTypeEnum.SUMMARY else [],
                    architecture_details=[response] if processing_type == ProcessingTypeEnum.TECHNICAL else [],
                    follow_ups=[],
                    participants_summary={},
                    technical_specs=[response] if processing_type == ProcessingTypeEnum.TECHNICAL else [],
                    risks_concerns=[]
                ))

        except Exception as e:
            logger.error(f"Failed to parse Claude response: {str(e)}")
            return Err(ProcessingError(f"Response parsing failed: {str(e)}"))

    def _parse_json_response(self, response: str) -> Result[StructuredNotes, ProcessingError]:
        """Parse enhanced JSON response from Claude into StructuredNotes."""
        try:
            # Find JSON in the response (Claude sometimes adds explanation text)
            json_start = response.find('{')
            json_end = response.rfind('}') + 1

            if json_start == -1 or json_end == 0:
                # No JSON found, treat as plain text
                return Ok(StructuredNotes(
                    action_items=[],
                    decisions=[response],
                    architecture_details=[],
                    follow_ups=[],
                    participants_summary={},
                    technical_specs=[],
                    risks_concerns=[]
                ))

            json_str = response[json_start:json_end]
            parsed_data = json.loads(json_str)

            # Handle both legacy simple format and new enhanced format
            if "executive_summary" in parsed_data:
                # New enhanced format - extract rich data
                action_items = self._extract_enhanced_action_items(parsed_data.get("action_items", []))
                decisions = self._extract_enhanced_decisions(parsed_data.get("decisions", []))
                architecture_details = self._extract_enhanced_architecture(parsed_data.get("architecture_details", []))
                follow_ups = self._extract_enhanced_followups(parsed_data.get("follow_ups", []))
                participants_summary = self._extract_enhanced_participants(parsed_data.get("participants_analysis", {}))
                technical_specs = self._extract_enhanced_technical_specs(parsed_data.get("technical_specifications", []))
                risks_concerns = self._extract_enhanced_risks_and_opportunities(parsed_data.get("risks_and_opportunities", {}))

                # Add executive summary and strategic insights to decisions for rich context
                if "executive_summary" in parsed_data:
                    decisions.insert(0, f"🎯 EXECUTIVE SUMMARY: {parsed_data['executive_summary']}")

                if "strategic_insights" in parsed_data:
                    for insight in parsed_data["strategic_insights"]:
                        decisions.append(f"💡 STRATEGIC INSIGHT: {insight}")

                # Add next meeting agenda to follow-ups
                if "next_meeting_agenda" in parsed_data:
                    for item in parsed_data["next_meeting_agenda"]:
                        follow_ups.append(f"📅 NEXT MEETING: {item}")

            else:
                # Legacy simple format
                action_items = parsed_data.get("action_items", [])
                decisions = parsed_data.get("decisions", [])
                architecture_details = parsed_data.get("architecture_details", [])
                follow_ups = parsed_data.get("follow_ups", [])
                participants_summary = parsed_data.get("participants_summary", {})
                technical_specs = parsed_data.get("technical_specs", [])
                risks_concerns = parsed_data.get("risks_concerns", [])

            return Ok(StructuredNotes(
                action_items=action_items,
                decisions=decisions,
                architecture_details=architecture_details,
                follow_ups=follow_ups,
                participants_summary=participants_summary,
                technical_specs=technical_specs,
                risks_concerns=risks_concerns
            ))

        except json.JSONDecodeError as e:
            logger.warning(f"JSON parsing failed, treating as plain text: {str(e)}")
            # Fallback to plain text processing
            return Ok(StructuredNotes(
                action_items=[],
                decisions=[response],
                architecture_details=[],
                follow_ups=[],
                participants_summary={},
                technical_specs=[],
                risks_concerns=[]
            ))

    def _extract_enhanced_action_items(self, action_items: List[Dict[str, Any]]) -> List[str]:
        """Extract enhanced action items into readable format."""
        extracted = []
        for item in action_items:
            if isinstance(item, dict):
                parts = []
                if "task" in item:
                    parts.append(f"📋 {item['task']}")
                if "owner" in item and item["owner"] != "TBD":
                    parts.append(f"👤 Owner: {item['owner']}")
                if "priority" in item:
                    parts.append(f"⚡ Priority: {item['priority']}")
                if "context" in item:
                    parts.append(f"💭 Context: {item['context']}")
                if "success_criteria" in item:
                    parts.append(f"✅ Success: {item['success_criteria']}")

                extracted.append(" | ".join(parts) if parts else str(item))
            else:
                extracted.append(str(item))
        return extracted

    def _extract_enhanced_decisions(self, decisions: List[Dict[str, Any]]) -> List[str]:
        """Extract enhanced decisions into readable format."""
        extracted = []
        for decision in decisions:
            if isinstance(decision, dict):
                parts = []
                if "decision" in decision:
                    parts.append(f"🎯 {decision['decision']}")
                if "rationale" in decision:
                    parts.append(f"🤔 Rationale: {decision['rationale']}")
                if "implications" in decision:
                    parts.append(f"💥 Impact: {decision['implications']}")
                if "stakeholders" in decision and decision["stakeholders"]:
                    stakeholder_list = ", ".join(decision["stakeholders"])
                    parts.append(f"👥 Affects: {stakeholder_list}")

                extracted.append(" | ".join(parts) if parts else str(decision))
            else:
                extracted.append(str(decision))
        return extracted

    def _extract_enhanced_architecture(self, arch_details: List[Dict[str, Any]]) -> List[str]:
        """Extract enhanced architecture details into readable format."""
        extracted = []
        for detail in arch_details:
            if isinstance(detail, dict):
                parts = []
                if "component" in detail:
                    parts.append(f"🏗️ {detail['component']}")
                if "description" in detail:
                    parts.append(f"📝 {detail['description']}")
                if "integration_points" in detail:
                    parts.append(f"🔗 Integration: {detail['integration_points']}")
                if "scalability_notes" in detail:
                    parts.append(f"📈 Scalability: {detail['scalability_notes']}")
                if "trade_offs" in detail:
                    parts.append(f"⚖️ Trade-offs: {detail['trade_offs']}")

                extracted.append(" | ".join(parts) if parts else str(detail))
            else:
                extracted.append(str(detail))
        return extracted

    def _extract_enhanced_followups(self, follow_ups: List[Dict[str, Any]]) -> List[str]:
        """Extract enhanced follow-ups into readable format."""
        extracted = []
        for followup in follow_ups:
            if isinstance(followup, dict):
                parts = []
                if "item" in followup:
                    parts.append(f"📌 {followup['item']}")
                if "urgency" in followup:
                    parts.append(f"⏰ Timeline: {followup['urgency']}")
                if "dependencies" in followup:
                    parts.append(f"🔗 Depends on: {followup['dependencies']}")
                if "estimated_effort" in followup:
                    parts.append(f"⚡ Effort: {followup['estimated_effort']}")

                extracted.append(" | ".join(parts) if parts else str(followup))
            else:
                extracted.append(str(followup))
        return extracted

    def _extract_enhanced_participants(self, participants: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
        """Extract enhanced participant analysis into readable format."""
        extracted = {}
        for participant, analysis in participants.items():
            if isinstance(analysis, dict):
                parts = []
                if "role_in_meeting" in analysis:
                    parts.append(f"Role: {analysis['role_in_meeting']}")
                if "key_contributions" in analysis and analysis["key_contributions"]:
                    contributions = ", ".join(analysis["key_contributions"])
                    parts.append(f"Contributions: {contributions}")
                if "concerns_raised" in analysis and analysis["concerns_raised"]:
                    concerns = ", ".join(analysis["concerns_raised"])
                    parts.append(f"Concerns: {concerns}")
                if "expertise_demonstrated" in analysis:
                    parts.append(f"Expertise: {analysis['expertise_demonstrated']}")

                extracted[participant] = " | ".join(parts) if parts else str(analysis)
            else:
                extracted[participant] = str(analysis)
        return extracted

    def _extract_enhanced_technical_specs(self, tech_specs: List[Dict[str, Any]]) -> List[str]:
        """Extract enhanced technical specifications into readable format."""
        extracted = []
        for spec in tech_specs:
            if isinstance(spec, dict):
                parts = []
                if "requirement" in spec:
                    parts.append(f"🔧 {spec['requirement']}")
                if "category" in spec:
                    parts.append(f"📁 Category: {spec['category']}")
                if "business_justification" in spec:
                    parts.append(f"💼 Business Need: {spec['business_justification']}")
                if "implementation_notes" in spec:
                    parts.append(f"⚙️ Implementation: {spec['implementation_notes']}")
                if "acceptance_criteria" in spec:
                    parts.append(f"✅ Acceptance: {spec['acceptance_criteria']}")

                extracted.append(" | ".join(parts) if parts else str(spec))
            else:
                extracted.append(str(spec))
        return extracted

    def _extract_enhanced_risks_and_opportunities(self, risks_opps: Dict[str, Any]) -> List[str]:
        """Extract enhanced risks and opportunities into readable format."""
        extracted = []

        # Extract risks
        if "risks" in risks_opps:
            for risk in risks_opps["risks"]:
                if isinstance(risk, dict):
                    parts = [f"⚠️ RISK: {risk.get('risk', 'Unknown risk')}"]
                    if "probability" in risk:
                        parts.append(f"Probability: {risk['probability']}")
                    if "impact" in risk:
                        parts.append(f"Impact: {risk['impact']}")
                    if "mitigation_strategies" in risk and risk["mitigation_strategies"]:
                        mitigation = ", ".join(risk["mitigation_strategies"])
                        parts.append(f"Mitigation: {mitigation}")

                    extracted.append(" | ".join(parts))
                else:
                    extracted.append(f"⚠️ RISK: {str(risk)}")

        # Extract opportunities
        if "opportunities" in risks_opps:
            for opp in risks_opps["opportunities"]:
                if isinstance(opp, dict):
                    parts = [f"🌟 OPPORTUNITY: {opp.get('opportunity', 'Unknown opportunity')}"]
                    if "value" in opp:
                        parts.append(f"Value: {opp['value']}")
                    if "effort_required" in opp:
                        parts.append(f"Effort: {opp['effort_required']}")
                    if "timeline" in opp:
                        parts.append(f"Timeline: {opp['timeline']}")

                    extracted.append(" | ".join(parts))
                else:
                    extracted.append(f"🌟 OPPORTUNITY: {str(opp)}")

        return extracted

    async def process_recording(
        self,
        audio_content: bytes,
        audio_format: str,
        context: Optional[str],
        participants: List[str],
        processing_type: ProcessingTypeEnum,
        custom_prompt: Optional[str] = None
    ) -> Result[Dict[str, Any], ProcessingError]:
        """
        Complete processing pipeline: transcribe + extract structured notes.

        Returns:
            Dict containing transcript, structured_notes, and processing_metadata.
        """
        try:
            processing_start = datetime.now()
            logger.info(f"Starting complete processing pipeline for {processing_type}")

            # Step 1: Transcribe audio (raw)
            transcript_result = await self.transcribe_audio(audio_content, audio_format)
            if transcript_result.is_err():
                return transcript_result

            raw_transcript = transcript_result.unwrap()

            # Step 2: Enhance transcript with punctuation and speaker diarization
            enhanced_transcript_result = await self._enhance_transcript_with_ai(
                raw_transcript, participants, context
            )
            if enhanced_transcript_result.is_err():
                return enhanced_transcript_result

            transcript = enhanced_transcript_result.unwrap()

            # Step 3: Extract structured notes
            notes_result = await self.extract_structured_notes(
                transcript, context, participants, processing_type, custom_prompt
            )
            if notes_result.is_err():
                return notes_result

            structured_notes = notes_result.unwrap()
            processing_end = datetime.now()

            # Step 3: Generate Premium PDF Report
            pdf_path = None
            if PDF_GENERATION_AVAILABLE and pdf_generator:
                try:
                    # Create output filename with timestamp
                    timestamp = processing_end.strftime("%Y%m%d_%H%M%S")
                    recording_name = context[:30].replace(" ", "_") if context else "Discovery_Session"
                    pdf_filename = f"{recording_name}_{timestamp}.pdf"
                    pdf_output_path = os.path.join(os.getcwd(), pdf_filename)

                    # Add audio metadata
                    audio_metadata = {
                        "claude_model": settings.claude_model,
                        "processing_type": processing_type,
                        "processing_duration_seconds": (processing_end - processing_start).total_seconds(),
                        "transcript_length": len(transcript),
                        "processed_at": processing_end.isoformat(),
                        "audio_format": audio_format,
                        "audio_size_mb": len(audio_content) / (1024 * 1024),
                        "participants": participants
                    }

                    # Generate premium PDF
                    logger.info("Generating premium PDF report...")
                    pdf_result = pdf_generator.generate_premium_report(
                        transcript=transcript,
                        structured_notes=structured_notes,
                        metadata=audio_metadata,
                        output_path=pdf_output_path,
                        recording_name=recording_name.replace("_", " ").title()
                    )

                    if pdf_result.is_ok():
                        pdf_path = pdf_result.unwrap()
                        logger.info(f"Premium PDF report generated: {pdf_path}")
                    else:
                        logger.warning(f"PDF generation failed: {pdf_result.unwrap_or('Unknown error')}")

                except Exception as e:
                    logger.warning(f"PDF generation error: {e}")

            # Compile results
            result = {
                "transcript": transcript,
                "structured_notes": structured_notes,
                "pdf_report_path": pdf_path,
                "processing_metadata": {
                    "claude_model": settings.claude_model,
                    "processing_type": processing_type,
                    "processing_duration_seconds": (processing_end - processing_start).total_seconds(),
                    "transcript_length": len(transcript),
                    "processed_at": processing_end.isoformat(),
                    "audio_format": audio_format,
                    "audio_size_mb": len(audio_content) / (1024 * 1024),
                    "participants": participants,
                    "pdf_generated": pdf_path is not None
                }
            }

            logger.info(f"Complete processing pipeline finished in {result['processing_metadata']['processing_duration_seconds']:.2f}s")
            if pdf_path:
                logger.info(f"Premium PDF report available at: {pdf_path}")
            return Ok(result)

        except Exception as e:
            logger.error(f"Processing pipeline failed: {str(e)}")
            return Err(ProcessingError(f"Processing pipeline failed: {str(e)}"))

    async def health_check(self) -> Result[Dict[str, Any], ExternalServiceError]:
        """Check Claude API connectivity and health."""
        try:
            if self.is_demo_mode:
                # Demo mode - always healthy
                return Ok({
                    "claude_api_healthy": True,
                    "model": "demo-mode",
                    "response_received": True,
                    "tested_at": datetime.now().isoformat(),
                    "demo_mode": True
                })

            if not CLAUDE_API_AVAILABLE or not self.client:
                return Err(ExternalServiceError("Claude API dependencies not available"))

            # Simple API test
            response = self.client.messages.create(
                model=settings.claude_model,
                max_tokens=50,
                temperature=0.0,
                messages=[{
                    "role": "user",
                    "content": "Respond with exactly: HEALTHY"
                }]
            )

            response_text = response.content[0].text if response.content else ""
            is_healthy = "HEALTHY" in response_text.upper()

            return Ok({
                "claude_api_healthy": is_healthy,
                "model": settings.claude_model,
                "response_received": bool(response_text),
                "tested_at": datetime.now().isoformat()
            })

        except Exception as e:
            logger.error(f"Claude API health check failed: {str(e)}")
            return Err(ExternalServiceError(f"Claude API health check failed: {str(e)}"))

    def _generate_demo_transcript(self, audio_content: bytes) -> str:
        """Generate demo transcript based on audio file size."""
        file_size_kb = len(audio_content) / 1024
        duration_estimate = max(10, int(file_size_kb / 16))  # Rough estimate

        return f"""[Demo Transcript - {duration_estimate}s recording]

Austin: Good morning everyone, thanks for joining our technical discovery call. I'm recording this for documentation purposes and we'll use our AI transcription system to generate structured notes. Does everyone consent to recording?

John: Yes, that works for me.

Sarah: Sounds good, I consent as well.

Austin: Perfect. So today we're discussing the integration architecture for the new client portal. Sarah, can you walk us through the current API endpoints?

Sarah: Sure, we currently have REST endpoints for user authentication, data retrieval, and file uploads. The main challenge is handling real-time notifications efficiently.

John: I'd suggest implementing WebSocket connections for the real-time features. We could use Socket.IO for better browser compatibility.

Austin: That makes sense. What about the database architecture? Are we looking at any scalability concerns?

Sarah: We might want to consider read replicas if we expect high traffic. The current PostgreSQL setup should handle our initial load, but we should plan for horizontal scaling.

John: Agreed. We should also implement proper caching with Redis to reduce database load.

Austin: Excellent points. Let's document these decisions and create action items for the implementation phases.

[End of transcript]"""

    def _generate_demo_structured_response(self) -> str:
        """Generate demo structured response in enhanced JSON format."""
        return """{
    "executive_summary": "Comprehensive technical discovery call outlining integration architecture for new client portal with focus on real-time capabilities, scalability planning, and performance optimization through caching strategies.",

    "action_items": [
        {
            "task": "Implement WebSocket connections for real-time notifications using Socket.IO",
            "owner": "John",
            "priority": "High",
            "context": "Real-time notifications are critical for user engagement and the current polling approach doesn't scale effectively",
            "success_criteria": "WebSocket connections stable under 1000+ concurrent users with <100ms latency"
        },
        {
            "task": "Set up Redis caching layer to reduce database load",
            "owner": "Sarah",
            "priority": "High",
            "context": "Current database queries are showing latency issues during peak usage periods",
            "success_criteria": "Cache hit rate >80% for frequently accessed data, query response time <50ms"
        },
        {
            "task": "Design and implement database read replica architecture",
            "owner": "TBD",
            "priority": "Medium",
            "context": "Preparing for anticipated traffic growth and ensuring read operations don't impact write performance",
            "success_criteria": "Read replicas handle 70% of read traffic with <500ms replication lag"
        },
        {
            "task": "Document comprehensive API specifications and authentication flow",
            "owner": "Austin",
            "priority": "Medium",
            "context": "Client integration teams need detailed documentation to build against our APIs effectively",
            "success_criteria": "Complete OpenAPI 3.0 spec with examples, authentication guide, and integration tutorials"
        }
    ],

    "decisions": [
        {
            "decision": "Adopt WebSocket/Socket.IO for real-time notification system",
            "rationale": "Provides better performance than polling and Socket.IO offers excellent cross-browser compatibility",
            "implications": "Need to implement WebSocket connection management, fallback strategies, and load balancing considerations",
            "stakeholders": ["Frontend Team", "DevOps", "Client Integration Teams"]
        },
        {
            "decision": "Continue with PostgreSQL as primary database with read replica strategy",
            "rationale": "Existing expertise and infrastructure investment, proven scalability with read replicas",
            "implications": "Must implement proper read/write splitting logic and handle eventual consistency",
            "stakeholders": ["Database Team", "Backend Developers", "Operations"]
        },
        {
            "decision": "Implement Redis as primary caching layer",
            "rationale": "Redis provides excellent performance for session data and frequently accessed queries",
            "implications": "Additional infrastructure cost and complexity, need cache invalidation strategy",
            "stakeholders": ["Backend Team", "DevOps", "Finance"]
        }
    ],

    "architecture_details": [
        {
            "component": "Real-time Notification System",
            "description": "WebSocket-based system using Socket.IO for cross-browser compatibility and connection management",
            "integration_points": "Integrates with API gateway, authenticates via JWT tokens, connects to Redis for session management",
            "scalability_notes": "Can scale horizontally with Redis adapter, supports cluster mode for multiple server instances",
            "trade_offs": "Increased complexity vs significant performance gains, requires connection state management"
        },
        {
            "component": "Database Architecture",
            "description": "PostgreSQL primary with read replicas for horizontal scaling of read operations",
            "integration_points": "Application layer routes reads to replicas, writes to primary, with automatic failover",
            "scalability_notes": "Read replicas can be added based on traffic, supports geographic distribution",
            "trade_offs": "Eventually consistent reads vs immediate consistency, increased operational complexity"
        },
        {
            "component": "Caching Layer",
            "description": "Redis-based caching for session data, frequently accessed queries, and computed results",
            "integration_points": "Sits between application and database, integrates with ORM for transparent caching",
            "scalability_notes": "Can be clustered for high availability, supports both memory and persistent storage",
            "trade_offs": "Additional infrastructure cost vs significant performance improvements"
        }
    ],

    "follow_ups": [
        {
            "item": "Research Socket.IO implementation requirements and browser compatibility matrix",
            "urgency": "Within 1 week",
            "dependencies": "Need to confirm browser support requirements from client teams",
            "estimated_effort": "Medium"
        },
        {
            "item": "Evaluate Redis hosting options and configuration for production environment",
            "urgency": "Within 2 weeks",
            "dependencies": "Infrastructure cost approval and DevOps capacity planning",
            "estimated_effort": "Large"
        },
        {
            "item": "Design detailed database read replica setup and failover strategy",
            "urgency": "Within 3 weeks",
            "dependencies": "Database team availability and testing environment setup",
            "estimated_effort": "Large"
        }
    ],

    "participants_analysis": {
        "Austin": {
            "role_in_meeting": "Technical Lead and Project Manager",
            "key_contributions": ["Guided architectural discussions", "Focused on documentation and integration requirements", "Ensured decisions align with business objectives"],
            "concerns_raised": ["Need for comprehensive documentation", "Integration complexity for client teams"],
            "expertise_demonstrated": "Project management, system integration, business-technical translation"
        },
        "Sarah": {
            "role_in_meeting": "Senior Backend Developer",
            "key_contributions": ["Provided current system context", "Identified database performance concerns", "Suggested scalability solutions"],
            "concerns_raised": ["Current database latency issues", "Scalability under high traffic"],
            "expertise_demonstrated": "Database architecture, performance optimization, backend systems"
        },
        "John": {
            "role_in_meeting": "Frontend Architecture Specialist",
            "key_contributions": ["Recommended WebSocket/Socket.IO solution", "Identified browser compatibility considerations", "Suggested caching strategies"],
            "concerns_raised": ["Real-time notification complexity", "Browser WebSocket compatibility"],
            "expertise_demonstrated": "Real-time web technologies, browser compatibility, frontend performance"
        }
    },

    "technical_specifications": [
        {
            "requirement": "Real-time notification delivery with <100ms latency",
            "category": "Performance",
            "business_justification": "User engagement metrics show 40% drop-off when notifications are delayed",
            "implementation_notes": "Use WebSocket connections with Socket.IO fallbacks, implement heartbeat monitoring",
            "acceptance_criteria": "99% of notifications delivered within 100ms under normal load conditions"
        },
        {
            "requirement": "Database read operations support 10x current traffic",
            "category": "Scalability",
            "business_justification": "Anticipated growth from new client integrations requires infrastructure preparation",
            "implementation_notes": "Implement read replicas with automatic routing, monitor replication lag",
            "acceptance_criteria": "System maintains <500ms response time at 10x current read volume"
        },
        {
            "requirement": "Cache hit rate >80% for frequently accessed data",
            "category": "Performance",
            "business_justification": "Reducing database load directly impacts system responsiveness and operational costs",
            "implementation_notes": "Implement Redis with intelligent cache warming and TTL strategies",
            "acceptance_criteria": "80% cache hit rate maintained during peak usage periods"
        }
    ],

    "risks_and_opportunities": {
        "risks": [
            {
                "risk": "WebSocket connection instability under high load",
                "probability": "Medium",
                "impact": "High",
                "mitigation_strategies": ["Implement robust reconnection logic", "Load test with realistic traffic patterns", "Create fallback to polling mechanism"]
            },
            {
                "risk": "Read replica replication lag causing data consistency issues",
                "probability": "Medium",
                "impact": "Medium",
                "mitigation_strategies": ["Monitor replication lag closely", "Route critical reads to primary", "Implement eventual consistency patterns"]
            },
            {
                "risk": "Redis cache invalidation strategy complexity",
                "probability": "High",
                "impact": "Medium",
                "mitigation_strategies": ["Implement cache versioning", "Use Redis pub/sub for cache invalidation", "Design cache-aside pattern with TTLs"]
            }
        ],
        "opportunities": [
            {
                "opportunity": "Leverage caching architecture for analytics and reporting optimization",
                "value": "Could improve reporting performance by 10x and enable real-time dashboards",
                "effort_required": "Additional 2-3 weeks of development",
                "timeline": "Could be implemented in parallel with core caching work"
            },
            {
                "opportunity": "Use WebSocket infrastructure for collaborative features",
                "value": "Enable real-time collaboration, shared editing, live cursors for enhanced user experience",
                "effort_required": "Moderate additional development, leverage existing WebSocket foundation",
                "timeline": "Natural follow-on project after core real-time infrastructure is stable"
            }
        ]
    },

    "next_meeting_agenda": [
        "Review Socket.IO implementation research findings and browser compatibility matrix",
        "Present Redis hosting cost analysis and configuration recommendations",
        "Discuss database read replica architecture design and failover testing strategy",
        "Review API documentation standards and client integration timeline",
        "Approve technical specification details and implementation prioritization"
    ],

    "strategic_insights": [
        "The team shows strong technical depth with complementary expertise areas, suggesting good project execution capability",
        "Focus on performance and scalability indicates mature understanding of production system requirements",
        "Emphasis on documentation and integration suggests customer-focused development approach",
        "Proactive planning for 10x traffic growth shows strategic thinking beyond immediate requirements",
        "Balance between proven technologies (PostgreSQL) and modern solutions (WebSocket) demonstrates pragmatic technical decision-making"
    ]
}"""


# Global Claude service instance
claude_service = ClaudeService()