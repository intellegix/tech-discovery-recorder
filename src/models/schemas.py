"""
Pydantic schemas for API request/response validation.
Follows Austin's patterns with strict type validation and error handling.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List, Union
from pydantic import BaseModel, Field, validator, ConfigDict
from enum import Enum


class ProcessingStatusEnum(str, Enum):
    """Processing status for recordings."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"


class ProcessingTypeEnum(str, Enum):
    """Types of processing available."""
    FULL = "full"  # Full transcription + summary + action items
    SUMMARY = "summary"  # Summary only (faster)
    TECHNICAL = "technical"  # Technical spec extraction
    CUSTOM = "custom"  # Custom prompt


# Base schemas with common configuration
class BaseSchema(BaseModel):
    """Base schema with common configuration."""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        use_enum_values=True
    )


# Request schemas
class ConsentProof(BaseSchema):
    """Consent tracking data for legal compliance."""
    verbal_consent: bool = Field(description="All parties consented verbally")
    documented_consent: bool = Field(default=False, description="Written consent in contract/email")
    purpose_explained: bool = Field(description="Purpose explained to participants")
    consent_statement: str = Field(description="Exact consent statement used")
    timestamp: datetime = Field(description="When consent was obtained")

    @validator('consent_statement')
    def validate_consent_statement(cls, v: str) -> str:
        """Ensure consent statement is not empty."""
        if not v.strip():
            raise ValueError("Consent statement cannot be empty")
        return v.strip()


class RecordingCreate(BaseSchema):
    """Schema for creating a new recording."""
    name: str = Field(min_length=1, max_length=255, description="Recording name/title")
    participants: List[str] = Field(min_items=1, description="List of participant names")
    context: Optional[str] = Field(default=None, max_length=2000, description="Technical context")
    consent_proof: ConsentProof = Field(description="Consent verification data")
    audio_format: str = Field(description="Audio file format (MIME type)")
    audio_file_size: int = Field(gt=0, description="Audio file size in bytes")

    @validator('participants')
    def validate_participants(cls, v: List[str]) -> List[str]:
        """Clean and validate participant names."""
        cleaned = [p.strip() for p in v if p.strip()]
        if not cleaned:
            raise ValueError("At least one participant name is required")
        return cleaned

    @validator('name')
    def validate_name(cls, v: str) -> str:
        """Validate recording name."""
        if not v.strip():
            raise ValueError("Recording name cannot be empty")
        return v.strip()


class ProcessingRequest(BaseSchema):
    """Schema for requesting AI processing of a recording."""
    recording_id: str = Field(description="UUID of recording to process")
    processing_type: ProcessingTypeEnum = Field(description="Type of processing to perform")
    custom_prompt: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="Custom prompt for processing (required if type=custom)"
    )

    @validator('custom_prompt')
    def validate_custom_prompt(cls, v: Optional[str], values: Dict[str, Any]) -> Optional[str]:
        """Validate custom prompt when processing_type is custom."""
        processing_type = values.get('processing_type')
        if processing_type == ProcessingTypeEnum.CUSTOM and not v:
            raise ValueError("Custom prompt is required when processing_type is 'custom'")
        return v.strip() if v else None


# Response schemas
class ConsentProofResponse(BaseSchema):
    """Response schema for consent proof."""
    verbal_consent: bool
    documented_consent: bool
    purpose_explained: bool
    consent_statement: str
    timestamp: datetime


class StructuredNotes(BaseSchema):
    """Schema for structured notes extracted from transcript."""
    action_items: List[str] = Field(default_factory=list, description="Action items identified")
    decisions: List[str] = Field(default_factory=list, description="Key decisions made")
    architecture_details: List[str] = Field(default_factory=list, description="Technical architecture notes")
    follow_ups: List[str] = Field(default_factory=list, description="Follow-up tasks")
    participants_summary: Dict[str, str] = Field(default_factory=dict, description="Per-participant summary")
    technical_specs: List[str] = Field(default_factory=list, description="Technical specifications")
    risks_concerns: List[str] = Field(default_factory=list, description="Identified risks or concerns")


class RecordingResponse(BaseSchema):
    """Response schema for recording data."""
    id: str
    name: str
    participants: List[str]
    context: Optional[str]
    consent_proof: ConsentProofResponse
    processing_status: ProcessingStatusEnum
    audio_file_size: Optional[int]
    audio_format: Optional[str]
    transcript: Optional[str]
    structured_notes: Optional[StructuredNotes]
    processing_error: Optional[str]
    created_at: datetime
    updated_at: datetime
    retention_expires_at: Optional[datetime]
    processing_completed_at: Optional[datetime]

    @validator('transcript')
    def redact_sensitive_transcript(cls, v: Optional[str]) -> Optional[str]:
        """Basic redaction for sensitive information in transcript."""
        if not v:
            return v

        # Simple redaction patterns - in production, use more sophisticated NLP
        redacted = v
        sensitive_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN pattern
            r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',  # Credit card pattern
        ]

        for pattern in sensitive_patterns:
            import re
            redacted = re.sub(pattern, '[REDACTED]', redacted)

        return redacted


class RecordingListResponse(BaseSchema):
    """Response schema for recording list."""
    recordings: List[RecordingResponse]
    total: int
    page: int
    per_page: int


class ProcessingStatusResponse(BaseSchema):
    """Response schema for processing status."""
    recording_id: str
    status: ProcessingStatusEnum
    processing_started_at: Optional[datetime]
    processing_completed_at: Optional[datetime]
    error: Optional[str]
    claude_model_used: Optional[str]


class ExportData(BaseSchema):
    """Schema for exported recording data."""
    recording: RecordingResponse
    export_timestamp: datetime
    export_format: str = Field(default="json", description="Export format")
    includes_audio: bool = Field(default=False, description="Whether audio file is included")


# Authentication schemas
class TokenResponse(BaseSchema):
    """JWT token response."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds


class UserCreate(BaseSchema):
    """Simple user creation for session management."""
    user_id: Optional[str] = Field(default=None, description="Optional user ID")


# Error schemas
class ErrorDetail(BaseSchema):
    """Error detail schema."""
    type: str = Field(description="Error type")
    message: str = Field(description="Human-readable error message")
    field: Optional[str] = Field(default=None, description="Field that caused the error")


class ErrorResponse(BaseSchema):
    """Standard error response schema."""
    error: str = Field(description="Error category")
    message: str = Field(description="Error message")
    details: Optional[List[ErrorDetail]] = Field(default=None, description="Detailed error information")
    request_id: Optional[str] = Field(default=None, description="Request tracking ID")


# Health and status schemas
class HealthResponse(BaseSchema):
    """API health check response."""
    status: str = Field(default="healthy")
    timestamp: datetime
    version: str
    database_connected: bool
    claude_api_connected: bool
    storage_available: bool


class StatsResponse(BaseSchema):
    """API usage statistics."""
    total_recordings: int
    pending_processing: int
    completed_processing: int
    failed_processing: int
    total_storage_mb: float
    avg_processing_time_minutes: Optional[float]