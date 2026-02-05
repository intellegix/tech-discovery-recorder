"""
Database models for Tech Discovery Recorder.
Implements audit logging and compliance features as per the plan.
"""

import uuid
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from sqlalchemy import (
    Column, String, DateTime, Boolean, Text, Integer,
    JSON, ForeignKey, LargeBinary, Enum as SQLEnum
)
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, Mapped
from sqlalchemy.sql import func
import enum


Base = declarative_base()


class ProcessingStatus(str, enum.Enum):
    """Recording processing status enum."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"


class AuditAction(str, enum.Enum):
    """Audit log action types."""
    RECORDING_CREATED = "recording_created"
    PROCESSING_STARTED = "processing_started"
    PROCESSING_COMPLETED = "processing_completed"
    PROCESSING_FAILED = "processing_failed"
    RECORDING_EXPORTED = "recording_exported"
    RECORDING_DELETED = "recording_deleted"
    CONSENT_VERIFIED = "consent_verified"
    RETENTION_APPLIED = "retention_applied"


def generate_uuid() -> str:
    """Generate UUID string for database IDs."""
    return str(uuid.uuid4())

# Database compatibility helpers
def get_uuid_column():
    """Get UUID column type based on database."""
    from config.settings import settings
    if settings.is_sqlite:
        return Column(String(36), primary_key=True, default=generate_uuid)
    else:
        return Column(String(36), primary_key=True, default=generate_uuid)

def get_array_column():
    """Get array column type based on database."""
    from config.settings import settings
    if settings.is_sqlite:
        return Column(JSON, nullable=False)  # Store as JSON array in SQLite
    else:
        return Column(ARRAY(String), nullable=False)


class Recording(Base):
    """
    Core recording model with consent tracking and compliance features.
    Immutable consent proof for legal defensibility.
    """
    __tablename__ = "recordings"

    id: Mapped[str] = Column(String(36), primary_key=True, default=generate_uuid)
    user_id: Mapped[str] = Column(String(36), nullable=False, index=True)

    # Recording metadata
    name: Mapped[str] = Column(String(255), nullable=False)
    participants: Mapped[List[str]] = Column(JSON, nullable=False)  # Store as JSON array
    context: Mapped[Optional[str]] = Column(Text, nullable=True)

    # Consent tracking (immutable for legal compliance)
    consent_proof: Mapped[Dict[str, Any]] = Column(JSON, nullable=False)
    consent_timestamp: Mapped[datetime] = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc)
    )

    # File storage
    audio_file_path: Mapped[Optional[str]] = Column(String(500), nullable=True)
    audio_file_size: Mapped[Optional[int]] = Column(Integer, nullable=True)  # bytes
    audio_format: Mapped[Optional[str]] = Column(String(50), nullable=True)

    # AI processing results
    transcript: Mapped[Optional[str]] = Column(Text, nullable=True)
    structured_notes: Mapped[Optional[Dict[str, Any]]] = Column(JSON, nullable=True)
    processing_status: Mapped[ProcessingStatus] = Column(
        SQLEnum(ProcessingStatus),
        default=ProcessingStatus.PENDING,
        nullable=False
    )
    processing_error: Mapped[Optional[str]] = Column(Text, nullable=True)

    # Compliance and retention
    created_at: Mapped[datetime] = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now()
    )
    updated_at: Mapped[datetime] = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc)
    )
    retention_expires_at: Mapped[Optional[datetime]] = Column(DateTime(timezone=True), nullable=True)
    deleted_at: Mapped[Optional[datetime]] = Column(DateTime(timezone=True), nullable=True)

    # Processing metadata
    processing_started_at: Mapped[Optional[datetime]] = Column(DateTime(timezone=True), nullable=True)
    processing_completed_at: Mapped[Optional[datetime]] = Column(DateTime(timezone=True), nullable=True)
    claude_model_used: Mapped[Optional[str]] = Column(String(100), nullable=True)

    # Relationships
    audit_logs: Mapped[List["AuditLog"]] = relationship("AuditLog", back_populates="recording")

    def __repr__(self) -> str:
        return f"<Recording(id={self.id}, name='{self.name}', status={self.processing_status})>"

    @property
    def is_expired(self) -> bool:
        """Check if recording has passed retention period."""
        if not self.retention_expires_at:
            return False
        return datetime.now(timezone.utc) > self.retention_expires_at

    @property
    def is_processed(self) -> bool:
        """Check if recording has been successfully processed."""
        return self.processing_status == ProcessingStatus.COMPLETED

    @property
    def duration_minutes(self) -> Optional[float]:
        """Calculate processing duration in minutes."""
        if not self.processing_started_at or not self.processing_completed_at:
            return None
        delta = self.processing_completed_at - self.processing_started_at
        return delta.total_seconds() / 60


class AuditLog(Base):
    """
    Immutable audit trail for compliance and legal defensibility.
    Tracks all operations on recordings with timestamp and metadata.
    """
    __tablename__ = "audit_log"

    id: Mapped[str] = Column(String(36), primary_key=True, default=generate_uuid)
    recording_id: Mapped[str] = Column(
        String(36),
        ForeignKey("recordings.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    # Audit details
    action: Mapped[AuditAction] = Column(SQLEnum(AuditAction), nullable=False)
    user_id: Mapped[str] = Column(String(36), nullable=False, index=True)

    # Context and metadata (JSON for flexibility)
    audit_metadata: Mapped[Optional[Dict[str, Any]]] = Column(JSON, nullable=True)
    ip_address: Mapped[Optional[str]] = Column(String(45), nullable=True)  # IPv6 compatible
    user_agent: Mapped[Optional[str]] = Column(Text, nullable=True)

    # Immutable timestamp
    timestamp: Mapped[datetime] = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now()
    )

    # Relationships
    recording: Mapped["Recording"] = relationship("Recording", back_populates="audit_logs")

    def __repr__(self) -> str:
        return f"<AuditLog(id={self.id}, action={self.action}, timestamp={self.timestamp})>"


class UserSession(Base):
    """
    Simple user session tracking for JWT token management.
    Not a full user system - just session management.
    """
    __tablename__ = "user_sessions"

    id: Mapped[str] = Column(String(36), primary_key=True, default=generate_uuid)
    user_id: Mapped[str] = Column(String(36), nullable=False, index=True)

    # Session metadata
    session_token_hash: Mapped[str] = Column(String(128), nullable=False, unique=True)
    created_at: Mapped[datetime] = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc)
    )
    expires_at: Mapped[datetime] = Column(DateTime(timezone=True), nullable=False)
    last_used_at: Mapped[datetime] = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc)
    )

    # Client info
    ip_address: Mapped[Optional[str]] = Column(String(45), nullable=True)
    user_agent: Mapped[Optional[str]] = Column(Text, nullable=True)

    # Session status
    is_active: Mapped[bool] = Column(Boolean, default=True, nullable=False)
    revoked_at: Mapped[Optional[datetime]] = Column(DateTime(timezone=True), nullable=True)

    def __repr__(self) -> str:
        return f"<UserSession(id={self.id}, user_id={self.user_id}, active={self.is_active})>"

    @property
    def is_expired(self) -> bool:
        """Check if session has expired."""
        return datetime.now(timezone.utc) > self.expires_at

    @property
    def is_valid(self) -> bool:
        """Check if session is valid (active and not expired)."""
        return self.is_active and not self.is_expired and not self.revoked_at