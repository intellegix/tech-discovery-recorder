"""
PDF export API routes for generating premium reports.
"""

import os
import logging
from typing import Optional
from pathlib import Path
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import FileResponse, JSONResponse

from services.database_service import database_service
from services.pdf_service import pdf_generator
from utils.result import Result, Ok, Err
from config.settings import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/recordings", tags=["pdf-export"])


# TODO: Implement proper JWT authentication
async def get_current_user_id() -> str:
    """Get current user ID from JWT token. Simplified for now."""
    return "demo-user-123"  # In production, extract from JWT


@router.post(
    "/{recording_id}/export-pdf",
    summary="Generate premium PDF report",
    description="Generate a professional PDF report from processed recording"
)
async def generate_pdf_report(
    recording_id: str,
    user_id: str = Depends(get_current_user_id)
):
    """
    Generate a premium PDF report for a processed recording.

    Returns the path to the generated PDF file.
    """
    try:
        logger.info(f"Generating PDF report for recording {recording_id}")

        # Get recording from database
        recording_result = await database_service.get_recording(recording_id, user_id)
        if recording_result.is_err():
            error = recording_result.unwrap_or("Unknown error")
            if "not found" in str(error).lower():
                raise HTTPException(status_code=404, detail="Recording not found")
            raise HTTPException(status_code=500, detail="Failed to retrieve recording")

        recording = recording_result.unwrap()

        # Check if processing is complete
        from models.database import ProcessingStatus
        if recording.processing_status != ProcessingStatus.COMPLETED:
            raise HTTPException(
                status_code=400,
                detail=f"Recording is not processed yet. Status: {recording.processing_status}"
            )

        # Check if we have transcript and structured notes
        if not recording.transcript:
            raise HTTPException(status_code=400, detail="No transcript available for PDF generation")

        if not recording.structured_notes:
            raise HTTPException(status_code=400, detail="No structured analysis available for PDF generation")

        # Create PDF filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = recording.name.replace(" ", "_").replace("/", "-").replace("\\", "-")[:50]
        pdf_filename = f"{safe_name}_{timestamp}.pdf"

        # Create PDFs directory if it doesn't exist
        pdf_dir = Path("generated_pdfs")
        pdf_dir.mkdir(exist_ok=True)
        pdf_path = pdf_dir / pdf_filename

        # Prepare metadata
        metadata = {
            "recording_id": recording.id,
            "claude_model": recording.claude_model_used or settings.claude_model,
            "processing_completed_at": recording.processing_completed_at.isoformat() if recording.processing_completed_at else None,
            "transcript_length": len(recording.transcript) if recording.transcript else 0,
            "participants": recording.participants,
            "audio_format": recording.audio_format,
            "audio_file_size": recording.audio_file_size,
            "created_at": recording.created_at.isoformat(),
        }

        # Convert database structured notes to StructuredNotes object
        from models.schemas import StructuredNotes
        try:
            if isinstance(recording.structured_notes, dict):
                structured_notes = StructuredNotes(**recording.structured_notes)
            else:
                # Handle case where it's already a StructuredNotes object
                structured_notes = recording.structured_notes
        except Exception as e:
            logger.error(f"Failed to parse structured notes: {e}")
            raise HTTPException(status_code=500, detail="Invalid structured notes format")

        # Generate PDF
        pdf_result = pdf_generator.generate_premium_report(
            transcript=recording.transcript,
            structured_notes=structured_notes,
            metadata=metadata,
            output_path=str(pdf_path),
            recording_name=recording.name
        )

        if pdf_result.is_err():
            error_msg = pdf_result.unwrap_or("Unknown error")
            logger.error(f"PDF generation failed: {error_msg}")
            raise HTTPException(status_code=500, detail=f"PDF generation failed: {error_msg}")

        generated_path = pdf_result.unwrap()
        logger.info(f"PDF report generated successfully: {generated_path}")

        return JSONResponse({
            "success": True,
            "pdf_path": str(pdf_path),
            "filename": pdf_filename,
            "recording_id": recording_id,
            "generated_at": datetime.now().isoformat()
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error generating PDF report: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get(
    "/{recording_id}/download-pdf",
    summary="Download PDF report",
    description="Download the generated PDF report for a recording"
)
async def download_pdf_report(
    recording_id: str,
    user_id: str = Depends(get_current_user_id)
):
    """
    Download the generated PDF report for a recording.
    """
    try:
        # Verify user has access to this recording
        recording_result = await database_service.get_recording(recording_id, user_id)
        if recording_result.is_err():
            error = recording_result.unwrap_or("Unknown error")
            if "not found" in str(error).lower():
                raise HTTPException(status_code=404, detail="Recording not found")
            raise HTTPException(status_code=500, detail="Failed to retrieve recording")

        recording = recording_result.unwrap()

        # Look for the most recent PDF for this recording
        pdf_dir = Path("generated_pdfs")
        if not pdf_dir.exists():
            raise HTTPException(status_code=404, detail="No PDF reports found")

        # Find PDF files for this recording
        safe_name = recording.name.replace(" ", "_").replace("/", "-").replace("\\", "-")[:50]
        pdf_pattern = f"{safe_name}_*.pdf"

        pdf_files = list(pdf_dir.glob(pdf_pattern))
        if not pdf_files:
            # Try finding by recording ID in filename
            pdf_files = [f for f in pdf_dir.glob("*.pdf") if recording_id in f.name]

        if not pdf_files:
            raise HTTPException(
                status_code=404,
                detail="PDF report not found. Please generate a PDF report first."
            )

        # Get the most recent PDF file
        latest_pdf = max(pdf_files, key=lambda f: f.stat().st_mtime)

        if not latest_pdf.exists():
            raise HTTPException(status_code=404, detail="PDF file not found on disk")

        # Return the PDF file
        return FileResponse(
            path=str(latest_pdf),
            filename=f"tech-discovery-{recording.name.replace(' ', '-')}-{recording_id}.pdf",
            media_type="application/pdf"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error downloading PDF: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")