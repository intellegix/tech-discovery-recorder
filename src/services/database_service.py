"""
Database service for recording management and audit logging.
Implements async database operations following Austin's patterns.
"""

import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import selectinload
from sqlalchemy import select, update, delete, func, and_, or_
from sqlalchemy.exc import SQLAlchemyError
import logging

from config.settings import settings
from models.database import Base, Recording, AuditLog, UserSession, ProcessingStatus, AuditAction
from models.schemas import (
    RecordingCreate, RecordingResponse, ConsentProofResponse, StructuredNotes
)
from utils.result import Result, Ok, Err, RecordingError, ValidationError


logger = logging.getLogger(__name__)


class DatabaseService:
    """
    Handles all database operations for recordings, audit logging, and sessions.
    Implements async patterns and proper error handling.
    """

    def __init__(self):
        # Create async engine
        database_url = settings.get_database_url()

        # Handle different database types
        if "postgresql://" in database_url:
            database_url = database_url.replace("postgresql://", "postgresql+psycopg://")

        engine_kwargs = {
            "echo": settings.debug,  # Log SQL in debug mode
        }

        # SQLite specific configuration
        if settings.is_sqlite:
            engine_kwargs.update({
                "connect_args": {"check_same_thread": False}
            })
        else:
            # PostgreSQL specific configuration
            engine_kwargs.update({
                "pool_pre_ping": True,  # Verify connections before use
                "pool_recycle": 300  # Recycle connections after 5 minutes
            })

        self.engine = create_async_engine(database_url, **engine_kwargs)
        self.async_session = async_sessionmaker(
            bind=self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        logger.info("Database service initialized")

    async def create_tables(self) -> Result[None, Exception]:
        """Create database tables if they don't exist."""
        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables created/verified")
            return Ok(None)
        except Exception as e:
            logger.error(f"Failed to create database tables: {str(e)}")
            return Err(e)

    async def create_recording(
        self,
        user_id: str,
        recording_data: RecordingCreate,
        storage_metadata: Dict[str, Any]
    ) -> Result[Recording, RecordingError]:
        """
        Create a new recording with consent proof and storage info.
        """
        try:
            async with self.async_session() as session:
                # Calculate retention expiry
                retention_expires_at = datetime.now(timezone.utc) + timedelta(days=settings.retention_days)

                # Create recording instance
                recording = Recording(
                    user_id=user_id,
                    name=recording_data.name,
                    participants=recording_data.participants,
                    context=recording_data.context,
                    consent_proof={
                        "verbal_consent": recording_data.consent_proof.verbal_consent,
                        "documented_consent": recording_data.consent_proof.documented_consent,
                        "purpose_explained": recording_data.consent_proof.purpose_explained,
                        "consent_statement": recording_data.consent_proof.consent_statement,
                        "timestamp": recording_data.consent_proof.timestamp.isoformat(),
                        "ip_address": storage_metadata.get("ip_address"),  # Add if available
                    },
                    audio_file_path=storage_metadata["file_path"],
                    audio_file_size=storage_metadata["file_size"],
                    audio_format=recording_data.audio_format,
                    retention_expires_at=retention_expires_at,
                    processing_status=ProcessingStatus.PENDING
                )

                session.add(recording)
                await session.flush()  # Get the ID

                # Create audit log entry
                audit_log = AuditLog(
                    recording_id=recording.id,
                    action=AuditAction.RECORDING_CREATED,
                    user_id=user_id,
                    audit_metadata={
                        "file_size": storage_metadata["file_size"],
                        "audio_format": recording_data.audio_format,
                        "participant_count": len(recording_data.participants),
                        "consent_verified": True,
                        "storage_type": storage_metadata.get("storage_type", "local")
                    }
                )
                session.add(audit_log)

                await session.commit()
                logger.info(f"Created recording {recording.id} for user {user_id}")
                return Ok(recording)

        except SQLAlchemyError as e:
            logger.error(f"Database error creating recording: {str(e)}")
            return Err(RecordingError(f"Failed to create recording: {str(e)}"))
        except Exception as e:
            logger.error(f"Unexpected error creating recording: {str(e)}")
            return Err(RecordingError(f"Failed to create recording: {str(e)}"))

    async def get_recording(self, recording_id: str, user_id: str) -> Result[Recording, RecordingError]:
        """Get a recording by ID, ensuring user ownership."""
        try:
            async with self.async_session() as session:
                stmt = select(Recording).where(
                    and_(
                        Recording.id == recording_id,
                        Recording.user_id == user_id,
                        Recording.deleted_at.is_(None)
                    )
                )
                result = await session.execute(stmt)
                recording = result.scalar_one_or_none()

                if not recording:
                    return Err(RecordingError(f"Recording {recording_id} not found or access denied"))

                return Ok(recording)

        except SQLAlchemyError as e:
            logger.error(f"Database error getting recording {recording_id}: {str(e)}")
            return Err(RecordingError(f"Failed to get recording: {str(e)}"))

    async def list_recordings(
        self,
        user_id: str,
        limit: int = 50,
        offset: int = 0,
        status_filter: Optional[ProcessingStatus] = None
    ) -> Result[tuple[List[Recording], int], RecordingError]:
        """
        List user's recordings with pagination and optional status filtering.
        Returns (recordings, total_count).
        """
        try:
            async with self.async_session() as session:
                # Build base query
                base_stmt = select(Recording).where(
                    and_(
                        Recording.user_id == user_id,
                        Recording.deleted_at.is_(None)
                    )
                )

                # Apply status filter
                if status_filter:
                    base_stmt = base_stmt.where(Recording.processing_status == status_filter)

                # Get total count
                count_stmt = select(func.count()).select_from(base_stmt.subquery())
                total_result = await session.execute(count_stmt)
                total_count = total_result.scalar()

                # Get paginated results
                stmt = base_stmt.order_by(Recording.created_at.desc()).limit(limit).offset(offset)
                result = await session.execute(stmt)
                recordings = result.scalars().all()

                logger.info(f"Retrieved {len(recordings)} recordings for user {user_id}")
                return Ok((list(recordings), total_count))

        except SQLAlchemyError as e:
            logger.error(f"Database error listing recordings for user {user_id}: {str(e)}")
            return Err(RecordingError(f"Failed to list recordings: {str(e)}"))

    async def update_processing_status(
        self,
        recording_id: str,
        user_id: str,
        status: ProcessingStatus,
        error_message: Optional[str] = None
    ) -> Result[Recording, RecordingError]:
        """Update recording processing status with audit logging."""
        try:
            async with self.async_session() as session:
                # Get the recording
                recording_result = await self.get_recording(recording_id, user_id)
                if recording_result.is_err():
                    return recording_result

                recording = recording_result.unwrap()

                # Update status and timestamps
                now = datetime.now(timezone.utc)
                update_data = {"processing_status": status, "updated_at": now}

                if status == ProcessingStatus.PROCESSING:
                    update_data["processing_started_at"] = now
                elif status == ProcessingStatus.COMPLETED:
                    update_data["processing_completed_at"] = now
                elif status == ProcessingStatus.FAILED:
                    update_data["processing_error"] = error_message

                # Update recording
                stmt = update(Recording).where(Recording.id == recording_id).values(**update_data)
                await session.execute(stmt)

                # Create audit log
                audit_action = {
                    ProcessingStatus.PROCESSING: AuditAction.PROCESSING_STARTED,
                    ProcessingStatus.COMPLETED: AuditAction.PROCESSING_COMPLETED,
                    ProcessingStatus.FAILED: AuditAction.PROCESSING_FAILED
                }.get(status, AuditAction.PROCESSING_STARTED)

                audit_log = AuditLog(
                    recording_id=recording_id,
                    action=audit_action,
                    user_id=user_id,
                    audit_metadata={
                        "new_status": status,
                        "error_message": error_message,
                        "processing_duration": None  # Will be calculated if completed
                    }
                )
                session.add(audit_log)

                await session.commit()

                # Refresh recording object
                await session.refresh(recording)
                logger.info(f"Updated recording {recording_id} status to {status}")
                return Ok(recording)

        except SQLAlchemyError as e:
            logger.error(f"Database error updating recording status {recording_id}: {str(e)}")
            return Err(RecordingError(f"Failed to update recording status: {str(e)}"))

    async def save_processing_results(
        self,
        recording_id: str,
        user_id: str,
        transcript: str,
        structured_notes: StructuredNotes,
        processing_metadata: Dict[str, Any]
    ) -> Result[Recording, RecordingError]:
        """Save AI processing results to the recording."""
        try:
            async with self.async_session() as session:
                # Update recording with results
                now = datetime.now(timezone.utc)
                stmt = update(Recording).where(
                    and_(Recording.id == recording_id, Recording.user_id == user_id)
                ).values(
                    transcript=transcript,
                    structured_notes=structured_notes.model_dump(),
                    processing_status=ProcessingStatus.COMPLETED,
                    processing_completed_at=now,
                    claude_model_used=processing_metadata.get("claude_model"),
                    updated_at=now
                )

                result = await session.execute(stmt)
                if result.rowcount == 0:
                    return Err(RecordingError(f"Recording {recording_id} not found or access denied"))

                # Create audit log
                audit_log = AuditLog(
                    recording_id=recording_id,
                    action=AuditAction.PROCESSING_COMPLETED,
                    user_id=user_id,
                    audit_metadata={
                        "transcript_length": len(transcript),
                        "processing_duration_seconds": processing_metadata.get("processing_duration_seconds"),
                        "claude_model": processing_metadata.get("claude_model"),
                        "action_items_count": len(structured_notes.action_items),
                        "decisions_count": len(structured_notes.decisions)
                    }
                )
                session.add(audit_log)
                await session.commit()

                # Return updated recording
                recording_result = await self.get_recording(recording_id, user_id)
                if recording_result.is_ok():
                    logger.info(f"Saved processing results for recording {recording_id}")

                return recording_result

        except SQLAlchemyError as e:
            logger.error(f"Database error saving processing results {recording_id}: {str(e)}")
            return Err(RecordingError(f"Failed to save processing results: {str(e)}"))

    async def delete_recording(self, recording_id: str, user_id: str) -> Result[None, RecordingError]:
        """Soft delete a recording with audit logging."""
        try:
            async with self.async_session() as session:
                now = datetime.now(timezone.utc)

                # Soft delete the recording
                stmt = update(Recording).where(
                    and_(
                        Recording.id == recording_id,
                        Recording.user_id == user_id,
                        Recording.deleted_at.is_(None)
                    )
                ).values(deleted_at=now, updated_at=now)

                result = await session.execute(stmt)
                if result.rowcount == 0:
                    return Err(RecordingError(f"Recording {recording_id} not found or access denied"))

                # Create audit log
                audit_log = AuditLog(
                    recording_id=recording_id,
                    action=AuditAction.RECORDING_DELETED,
                    user_id=user_id,
                    audit_metadata={"deleted_at": now.isoformat()}
                )
                session.add(audit_log)
                await session.commit()

                logger.info(f"Deleted recording {recording_id} for user {user_id}")
                return Ok(None)

        except SQLAlchemyError as e:
            logger.error(f"Database error deleting recording {recording_id}: {str(e)}")
            return Err(RecordingError(f"Failed to delete recording: {str(e)}"))

    async def get_expired_recordings(self) -> Result[List[Recording], RecordingError]:
        """Get recordings that have passed their retention period."""
        try:
            async with self.async_session() as session:
                now = datetime.now(timezone.utc)
                stmt = select(Recording).where(
                    and_(
                        Recording.retention_expires_at <= now,
                        Recording.deleted_at.is_(None),
                        Recording.audio_file_path.isnot(None)  # Has audio file to clean up
                    )
                )

                result = await session.execute(stmt)
                expired_recordings = result.scalars().all()

                logger.info(f"Found {len(expired_recordings)} expired recordings for cleanup")
                return Ok(list(expired_recordings))

        except SQLAlchemyError as e:
            logger.error(f"Database error getting expired recordings: {str(e)}")
            return Err(RecordingError(f"Failed to get expired recordings: {str(e)}"))

    async def apply_retention_policy(self, recording_id: str) -> Result[None, RecordingError]:
        """Apply retention policy - remove audio file but keep metadata."""
        try:
            async with self.async_session() as session:
                now = datetime.now(timezone.utc)

                # Update recording to remove audio file reference
                stmt = update(Recording).where(Recording.id == recording_id).values(
                    audio_file_path=None,  # Remove file path
                    processing_status=ProcessingStatus.EXPIRED,
                    updated_at=now
                )

                await session.execute(stmt)

                # Create audit log
                audit_log = AuditLog(
                    recording_id=recording_id,
                    action=AuditAction.RETENTION_APPLIED,
                    user_id="system",  # System action
                    audit_metadata={
                        "applied_at": now.isoformat(),
                        "retention_policy": f"{settings.retention_days} days"
                    }
                )
                session.add(audit_log)
                await session.commit()

                logger.info(f"Applied retention policy to recording {recording_id}")
                return Ok(None)

        except SQLAlchemyError as e:
            logger.error(f"Database error applying retention policy {recording_id}: {str(e)}")
            return Err(RecordingError(f"Failed to apply retention policy: {str(e)}"))

    async def get_audit_logs(
        self,
        recording_id: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 100
    ) -> Result[List[AuditLog], RecordingError]:
        """Get audit logs with optional filtering."""
        try:
            async with self.async_session() as session:
                stmt = select(AuditLog).options(selectinload(AuditLog.recording))

                # Apply filters
                filters = []
                if recording_id:
                    filters.append(AuditLog.recording_id == recording_id)
                if user_id:
                    filters.append(AuditLog.user_id == user_id)

                if filters:
                    stmt = stmt.where(and_(*filters))

                # Order by timestamp and limit
                stmt = stmt.order_by(AuditLog.timestamp.desc()).limit(limit)

                result = await session.execute(stmt)
                audit_logs = result.scalars().all()

                return Ok(list(audit_logs))

        except SQLAlchemyError as e:
            logger.error(f"Database error getting audit logs: {str(e)}")
            return Err(RecordingError(f"Failed to get audit logs: {str(e)}"))

    async def get_stats(self, user_id: Optional[str] = None) -> Result[Dict[str, Any], RecordingError]:
        """Get database statistics."""
        try:
            async with self.async_session() as session:
                base_filter = Recording.deleted_at.is_(None)
                if user_id:
                    base_filter = and_(base_filter, Recording.user_id == user_id)

                # Count recordings by status
                stats = {}
                for status in ProcessingStatus:
                    stmt = select(func.count()).where(
                        and_(base_filter, Recording.processing_status == status)
                    )
                    result = await session.execute(stmt)
                    stats[f"{status}_count"] = result.scalar()

                # Total file size
                stmt = select(func.sum(Recording.audio_file_size)).where(
                    and_(base_filter, Recording.audio_file_size.isnot(None))
                )
                result = await session.execute(stmt)
                total_size_bytes = result.scalar() or 0
                stats["total_storage_mb"] = total_size_bytes / (1024 * 1024)

                # Average processing time
                stmt = select(
                    func.avg(
                        func.extract('epoch', Recording.processing_completed_at - Recording.processing_started_at) / 60
                    )
                ).where(
                    and_(
                        base_filter,
                        Recording.processing_status == ProcessingStatus.COMPLETED,
                        Recording.processing_started_at.isnot(None),
                        Recording.processing_completed_at.isnot(None)
                    )
                )
                result = await session.execute(stmt)
                avg_processing_minutes = result.scalar()
                stats["avg_processing_time_minutes"] = float(avg_processing_minutes) if avg_processing_minutes else None

                return Ok(stats)

        except SQLAlchemyError as e:
            logger.error(f"Database error getting stats: {str(e)}")
            return Err(RecordingError(f"Failed to get stats: {str(e)}"))

    async def health_check(self) -> Result[bool, Exception]:
        """Check database connectivity."""
        try:
            async with self.async_session() as session:
                await session.execute(select(1))
            return Ok(True)
        except Exception as e:
            logger.error(f"Database health check failed: {str(e)}")
            return Err(e)


# Global database service instance
database_service = DatabaseService()