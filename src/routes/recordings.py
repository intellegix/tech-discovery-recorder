"""
API routes for recording management.
Implements the main recording lifecycle endpoints.
"""

from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse
import logging

from models.schemas import (
    RecordingCreate, RecordingResponse, RecordingListResponse,
    ProcessingRequest, ProcessingStatusResponse, ExportData,
    ConsentProof, ProcessingTypeEnum, ErrorResponse
)
from services.database_service import database_service
from services.storage_service import storage_service
from services.claude_service import claude_service
from utils.result import Result, Ok, Err, RecordingError, ProcessingError
from config.settings import settings


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/recordings", tags=["recordings"])


# TODO: Implement proper JWT authentication
# For now, using a simple user_id parameter
async def get_current_user_id() -> str:
    """Get current user ID from JWT token. Simplified for now."""
    return "demo-user-123"  # In production, extract from JWT


@router.post(
    "/",
    response_model=RecordingResponse,
    status_code=201,
    summary="Create new recording",
    description="Upload audio file with metadata and consent proof"
)
async def create_recording(
    background_tasks: BackgroundTasks,
    audio_file: UploadFile = File(..., description="Audio file (WebM, WAV, MP4, M4A)"),
    name: str = Form(..., description="Recording name"),
    participants: str = Form(..., description="Comma-separated participant names"),
    context: Optional[str] = Form(None, description="Technical context"),
    consent_verbal: bool = Form(..., description="Verbal consent obtained"),
    consent_documented: bool = Form(False, description="Written consent obtained"),
    consent_purpose: bool = Form(..., description="Purpose explained to participants"),
    consent_statement: str = Form(..., description="Consent statement used"),
    user_id: str = Depends(get_current_user_id)
) -> RecordingResponse:
    """
    Create a new recording with audio upload and consent tracking.
    """
    try:
        logger.info(f"Creating recording '{name}' for user {user_id}")

        # Validate audio file
        if not audio_file.content_type or audio_file.content_type not in settings.supported_formats:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported audio format. Supported: {settings.supported_formats}"
            )

        # Read audio content
        audio_content = await audio_file.read()
        file_size_mb = len(audio_content) / (1024 * 1024)

        if file_size_mb > settings.max_file_size_mb:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {settings.max_file_size_mb}MB"
            )

        # Parse participants
        participant_list = [p.strip() for p in participants.split(",") if p.strip()]
        if not participant_list:
            raise HTTPException(status_code=400, detail="At least one participant is required")

        # Validate consent
        if not consent_verbal:
            raise HTTPException(
                status_code=400,
                detail="Verbal consent is required for all recording creation"
            )

        # Create consent proof
        consent_proof = ConsentProof(
            verbal_consent=consent_verbal,
            documented_consent=consent_documented,
            purpose_explained=consent_purpose,
            consent_statement=consent_statement,
            timestamp=datetime.now()
        )

        # Create recording data model
        recording_data = RecordingCreate(
            name=name,
            participants=participant_list,
            context=context,
            consent_proof=consent_proof,
            audio_format=audio_file.content_type,
            audio_file_size=len(audio_content)
        )

        # Store audio file
        storage_result = await storage_service.store_audio_file(
            recording_id="temp",  # Will be updated after recording creation
            audio_content=audio_content,
            content_type=audio_file.content_type
        )

        if storage_result.is_err():
            logger.error(f"Audio storage failed: {storage_result.unwrap_or('Unknown error')}")
            raise HTTPException(status_code=500, detail="Failed to store audio file")

        storage_metadata = storage_result.unwrap()

        # Create recording in database
        recording_result = await database_service.create_recording(
            user_id=user_id,
            recording_data=recording_data,
            storage_metadata=storage_metadata
        )

        if recording_result.is_err():
            logger.error(f"Database recording creation failed: {recording_result.unwrap_or('Unknown error')}")
            # TODO: Clean up stored audio file on failure
            raise HTTPException(status_code=500, detail="Failed to create recording")

        recording = recording_result.unwrap()

        # Convert to response model
        response = _recording_to_response(recording)

        logger.info(f"Successfully created recording {recording.id}")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error creating recording: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get(
    "/",
    response_model=RecordingListResponse,
    summary="List recordings",
    description="Get paginated list of user's recordings"
)
async def list_recordings(
    page: int = 1,
    per_page: int = 20,
    status: Optional[str] = None,
    user_id: str = Depends(get_current_user_id)
) -> RecordingListResponse:
    """List user's recordings with optional status filtering."""
    try:
        if per_page > 100:
            raise HTTPException(status_code=400, detail="per_page cannot exceed 100")

        offset = (page - 1) * per_page

        # Parse status filter
        status_filter = None
        if status:
            try:
                from models.database import ProcessingStatus
                status_filter = ProcessingStatus(status)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid status: {status}")

        # Get recordings from database
        recordings_result = await database_service.list_recordings(
            user_id=user_id,
            limit=per_page,
            offset=offset,
            status_filter=status_filter
        )

        if recordings_result.is_err():
            logger.error(f"Failed to list recordings: {recordings_result.unwrap_or('Unknown error')}")
            raise HTTPException(status_code=500, detail="Failed to retrieve recordings")

        recordings, total_count = recordings_result.unwrap()

        # Convert to response models
        recording_responses = [_recording_to_response(rec) for rec in recordings]

        return RecordingListResponse(
            recordings=recording_responses,
            total=total_count,
            page=page,
            per_page=per_page
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error listing recordings: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get(
    "/{recording_id}",
    response_model=RecordingResponse,
    summary="Get recording",
    description="Get recording details by ID"
)
async def get_recording(
    recording_id: str,
    user_id: str = Depends(get_current_user_id)
) -> RecordingResponse:
    """Get a specific recording by ID."""
    try:
        recording_result = await database_service.get_recording(recording_id, user_id)

        if recording_result.is_err():
            error = recording_result.unwrap_or("Unknown error")
            if "not found" in str(error).lower():
                raise HTTPException(status_code=404, detail="Recording not found")
            raise HTTPException(status_code=500, detail="Failed to retrieve recording")

        recording = recording_result.unwrap()
        return _recording_to_response(recording)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error getting recording {recording_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post(
    "/{recording_id}/process",
    response_model=ProcessingStatusResponse,
    summary="Process recording",
    description="Start AI processing of recording (transcription + note extraction)"
)
async def process_recording(
    recording_id: str,
    background_tasks: BackgroundTasks,
    processing_type: ProcessingTypeEnum = ProcessingTypeEnum.FULL,
    custom_prompt: Optional[str] = None,
    user_id: str = Depends(get_current_user_id)
) -> ProcessingStatusResponse:
    """
    Start AI processing of a recording.
    Processing runs in background to avoid timeout issues.
    """
    try:
        # Validate custom prompt requirement
        if processing_type == ProcessingTypeEnum.CUSTOM and not custom_prompt:
            raise HTTPException(
                status_code=400,
                detail="custom_prompt is required when processing_type is 'custom'"
            )

        # Get recording
        recording_result = await database_service.get_recording(recording_id, user_id)
        if recording_result.is_err():
            error = recording_result.unwrap_or("Unknown error")
            if "not found" in str(error).lower():
                raise HTTPException(status_code=404, detail="Recording not found")
            raise HTTPException(status_code=500, detail="Failed to retrieve recording")

        recording = recording_result.unwrap()

        # Check if already processing or completed
        from models.database import ProcessingStatus
        if recording.processing_status == ProcessingStatus.PROCESSING:
            raise HTTPException(status_code=409, detail="Recording is already being processed")

        if recording.processing_status == ProcessingStatus.COMPLETED:
            raise HTTPException(status_code=409, detail="Recording has already been processed")

        # Check if audio file exists
        if not recording.audio_file_path:
            raise HTTPException(status_code=400, detail="No audio file available for processing")

        # Start processing in background
        background_tasks.add_task(
            _process_recording_background,
            recording_id,
            user_id,
            processing_type,
            custom_prompt
        )

        # Update status to processing
        await database_service.update_processing_status(
            recording_id, user_id, ProcessingStatus.PROCESSING
        )

        return ProcessingStatusResponse(
            recording_id=recording_id,
            status=ProcessingStatus.PROCESSING,
            processing_started_at=datetime.now(),
            processing_completed_at=None,
            error=None,
            claude_model_used=settings.claude_model
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing recording {recording_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get(
    "/{recording_id}/status",
    response_model=ProcessingStatusResponse,
    summary="Get processing status",
    description="Check the processing status of a recording"
)
async def get_processing_status(
    recording_id: str,
    user_id: str = Depends(get_current_user_id)
) -> ProcessingStatusResponse:
    """Get the current processing status of a recording."""
    try:
        recording_result = await database_service.get_recording(recording_id, user_id)
        if recording_result.is_err():
            error = recording_result.unwrap_or("Unknown error")
            if "not found" in str(error).lower():
                raise HTTPException(status_code=404, detail="Recording not found")
            raise HTTPException(status_code=500, detail="Failed to retrieve recording")

        recording = recording_result.unwrap()

        return ProcessingStatusResponse(
            recording_id=recording_id,
            status=recording.processing_status,
            processing_started_at=recording.processing_started_at,
            processing_completed_at=recording.processing_completed_at,
            error=recording.processing_error,
            claude_model_used=recording.claude_model_used
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error getting processing status {recording_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete(
    "/{recording_id}",
    status_code=204,
    summary="Delete recording",
    description="Soft delete a recording and its associated files"
)
async def delete_recording(
    recording_id: str,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user_id)
):
    """Delete a recording (soft delete with audit trail)."""
    try:
        # Get recording to check if it exists and get file path
        recording_result = await database_service.get_recording(recording_id, user_id)
        if recording_result.is_err():
            error = recording_result.unwrap_or("Unknown error")
            if "not found" in str(error).lower():
                raise HTTPException(status_code=404, detail="Recording not found")
            raise HTTPException(status_code=500, detail="Failed to retrieve recording")

        recording = recording_result.unwrap()

        # Soft delete in database
        delete_result = await database_service.delete_recording(recording_id, user_id)
        if delete_result.is_err():
            logger.error(f"Failed to delete recording {recording_id}: {delete_result.unwrap_or('Unknown error')}")
            raise HTTPException(status_code=500, detail="Failed to delete recording")

        # Schedule audio file deletion in background
        if recording.audio_file_path:
            background_tasks.add_task(
                storage_service.delete_audio_file,
                recording.audio_file_path
            )

        logger.info(f"Successfully deleted recording {recording_id}")
        return JSONResponse(status_code=204, content=None)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error deleting recording {recording_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


async def _process_recording_background(
    recording_id: str,
    user_id: str,
    processing_type: ProcessingTypeEnum,
    custom_prompt: Optional[str]
):
    """Background task for processing recording with Claude."""
    from models.database import ProcessingStatus
    try:
        logger.info(f"Starting background processing for recording {recording_id}")

        # Get recording and audio file
        recording_result = await database_service.get_recording(recording_id, user_id)
        if recording_result.is_err():
            await database_service.update_processing_status(
                recording_id, user_id, ProcessingStatus.FAILED,
                "Failed to retrieve recording for processing"
            )
            return

        recording = recording_result.unwrap()

        # Get audio file content
        audio_result = await storage_service.retrieve_audio_file(recording.audio_file_path)
        if audio_result.is_err():
            await database_service.update_processing_status(
                recording_id, user_id, ProcessingStatus.FAILED,
                f"Failed to retrieve audio file: {audio_result.unwrap_or('Unknown error')}"
            )
            return

        audio_content = audio_result.unwrap()

        # Process with Claude
        processing_result = await claude_service.process_recording(
            audio_content=audio_content,
            audio_format=recording.audio_format,
            context=recording.context,
            participants=recording.participants,
            processing_type=processing_type,
            custom_prompt=custom_prompt
        )

        if processing_result.is_err():
            error_msg = str(processing_result.unwrap_or("Unknown processing error"))
            await database_service.update_processing_status(
                recording_id, user_id, ProcessingStatus.FAILED, error_msg
            )
            logger.error(f"Claude processing failed for recording {recording_id}: {error_msg}")
            return

        # Save results
        processing_data = processing_result.unwrap()
        save_result = await database_service.save_processing_results(
            recording_id=recording_id,
            user_id=user_id,
            transcript=processing_data["transcript"],
            structured_notes=processing_data["structured_notes"],
            processing_metadata=processing_data["processing_metadata"]
        )

        if save_result.is_err():
            await database_service.update_processing_status(
                recording_id, user_id, ProcessingStatus.FAILED,
                f"Failed to save processing results: {save_result.unwrap_or('Unknown error')}"
            )
            return

        logger.info(f"Successfully completed background processing for recording {recording_id}")

    except Exception as e:
        logger.error(f"Unexpected error in background processing for recording {recording_id}: {str(e)}")
        await database_service.update_processing_status(
            recording_id, user_id, ProcessingStatus.FAILED,
            f"Unexpected processing error: {str(e)}"
        )


def _recording_to_response(recording) -> RecordingResponse:
    """Convert database Recording model to response schema."""
    from models.schemas import ConsentProofResponse, StructuredNotes

    # Convert consent proof
    consent_data = recording.consent_proof
    consent_response = ConsentProofResponse(
        verbal_consent=consent_data.get("verbal_consent", False),
        documented_consent=consent_data.get("documented_consent", False),
        purpose_explained=consent_data.get("purpose_explained", False),
        consent_statement=consent_data.get("consent_statement", ""),
        timestamp=datetime.fromisoformat(consent_data.get("timestamp", recording.consent_timestamp.isoformat()))
    )

    # Convert structured notes
    structured_notes = None
    if recording.structured_notes:
        structured_notes = StructuredNotes(**recording.structured_notes)

    return RecordingResponse(
        id=recording.id,
        name=recording.name,
        participants=recording.participants,
        context=recording.context,
        consent_proof=consent_response,
        processing_status=recording.processing_status,
        audio_file_size=recording.audio_file_size,
        audio_format=recording.audio_format,
        transcript=recording.transcript,
        structured_notes=structured_notes,
        processing_error=recording.processing_error,
        created_at=recording.created_at,
        updated_at=recording.updated_at,
        retention_expires_at=recording.retention_expires_at,
        processing_completed_at=recording.processing_completed_at
    )