"""
Admin and health check routes.
Provides system monitoring and administrative functionality.
"""

from datetime import datetime
from typing import Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
import logging

from models.schemas import HealthResponse, StatsResponse, ErrorResponse
from services.database_service import database_service
from services.storage_service import storage_service
from services.claude_service import claude_service
from config.settings import settings


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["admin"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check system health and component status"
)
async def health_check() -> HealthResponse:
    """
    Comprehensive health check for all system components.
    """
    try:
        logger.info("Performing health check")

        # Check database connectivity
        db_health = await database_service.health_check()
        database_connected = db_health.is_ok()

        # Check Claude API connectivity
        claude_health = await claude_service.health_check()
        claude_api_connected = claude_health.is_ok()

        # Check storage availability
        storage_stats = await storage_service.get_storage_stats()
        storage_available = storage_stats.is_ok()

        # Determine overall health status
        all_healthy = database_connected and claude_api_connected and storage_available
        status = "healthy" if all_healthy else "degraded"

        health_response = HealthResponse(
            status=status,
            timestamp=datetime.now(),
            version=settings.app_version,
            database_connected=database_connected,
            claude_api_connected=claude_api_connected,
            storage_available=storage_available
        )

        if not all_healthy:
            logger.warning(f"Health check shows degraded status: {health_response}")
        else:
            logger.info("Health check passed")

        return health_response

    except Exception as e:
        logger.error(f"Health check failed with exception: {str(e)}")
        # Return degraded status instead of failing
        return HealthResponse(
            status="error",
            timestamp=datetime.now(),
            version=settings.app_version,
            database_connected=False,
            claude_api_connected=False,
            storage_available=False
        )


@router.get(
    "/stats",
    response_model=StatsResponse,
    summary="System statistics",
    description="Get system usage statistics and metrics"
)
async def get_stats() -> StatsResponse:
    """
    Get system-wide statistics for monitoring and analytics.
    """
    try:
        logger.info("Retrieving system statistics")

        # Get database stats
        db_stats_result = await database_service.get_stats()
        if db_stats_result.is_err():
            logger.error(f"Failed to get database stats: {db_stats_result.unwrap_or('Unknown error')}")
            raise HTTPException(status_code=500, detail="Failed to retrieve database statistics")

        db_stats = db_stats_result.unwrap()

        # Get storage stats
        storage_stats_result = await storage_service.get_storage_stats()
        storage_stats = storage_stats_result.unwrap_or({})

        # Compile statistics
        total_recordings = sum([
            db_stats.get("pending_count", 0),
            db_stats.get("processing_count", 0),
            db_stats.get("completed_count", 0),
            db_stats.get("failed_count", 0),
            db_stats.get("expired_count", 0)
        ])

        stats_response = StatsResponse(
            total_recordings=total_recordings,
            pending_processing=db_stats.get("pending_count", 0),
            completed_processing=db_stats.get("completed_count", 0),
            failed_processing=db_stats.get("failed_count", 0),
            total_storage_mb=db_stats.get("total_storage_mb", 0.0),
            avg_processing_time_minutes=db_stats.get("avg_processing_time_minutes")
        )

        logger.info(f"Retrieved statistics: {total_recordings} total recordings")
        return stats_response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error getting stats: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve statistics")


@router.post(
    "/admin/retention-cleanup",
    summary="Run retention cleanup",
    description="Manually trigger retention policy cleanup (admin only)"
)
async def run_retention_cleanup(background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """
    Manually trigger retention policy cleanup.
    In production, this would require admin authentication.
    """
    try:
        logger.info("Starting manual retention policy cleanup")

        # Get expired recordings
        expired_result = await database_service.get_expired_recordings()
        if expired_result.is_err():
            logger.error(f"Failed to get expired recordings: {expired_result.unwrap_or('Unknown error')}")
            raise HTTPException(status_code=500, detail="Failed to get expired recordings")

        expired_recordings = expired_result.unwrap()
        expired_count = len(expired_recordings)

        if expired_count == 0:
            logger.info("No expired recordings found for cleanup")
            return {
                "message": "No expired recordings found",
                "expired_count": 0,
                "cleanup_started": False
            }

        # Extract file paths for storage cleanup
        file_paths = [rec.audio_file_path for rec in expired_recordings if rec.audio_file_path]

        # Start background cleanup task
        background_tasks.add_task(_cleanup_expired_recordings, expired_recordings, file_paths)

        logger.info(f"Started cleanup of {expired_count} expired recordings")
        return {
            "message": f"Started cleanup of {expired_count} expired recordings",
            "expired_count": expired_count,
            "files_to_cleanup": len(file_paths),
            "cleanup_started": True
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in retention cleanup: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to start retention cleanup")


@router.get(
    "/admin/audit-logs",
    summary="Get audit logs",
    description="Retrieve audit logs for compliance monitoring (admin only)"
)
async def get_audit_logs(
    recording_id: str = None,
    user_id: str = None,
    limit: int = 100
) -> Dict[str, Any]:
    """
    Get audit logs with optional filtering.
    In production, this would require admin authentication.
    """
    try:
        if limit > 1000:
            raise HTTPException(status_code=400, detail="Limit cannot exceed 1000")

        logger.info(f"Retrieving audit logs (limit: {limit})")

        audit_logs_result = await database_service.get_audit_logs(
            recording_id=recording_id,
            user_id=user_id,
            limit=limit
        )

        if audit_logs_result.is_err():
            logger.error(f"Failed to get audit logs: {audit_logs_result.unwrap_or('Unknown error')}")
            raise HTTPException(status_code=500, detail="Failed to retrieve audit logs")

        audit_logs = audit_logs_result.unwrap()

        # Convert to serializable format
        logs_data = []
        for log in audit_logs:
            logs_data.append({
                "id": log.id,
                "recording_id": log.recording_id,
                "action": log.action,
                "user_id": log.user_id,
                "metadata": log.audit_metadata,
                "ip_address": log.ip_address,
                "user_agent": log.user_agent,
                "timestamp": log.timestamp.isoformat(),
                "recording_name": log.recording.name if log.recording else None
            })

        logger.info(f"Retrieved {len(logs_data)} audit log entries")
        return {
            "audit_logs": logs_data,
            "total_entries": len(logs_data),
            "filters_applied": {
                "recording_id": recording_id,
                "user_id": user_id,
                "limit": limit
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error getting audit logs: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve audit logs")


@router.get(
    "/admin/storage-info",
    summary="Get storage information",
    description="Get detailed storage usage information (admin only)"
)
async def get_storage_info() -> Dict[str, Any]:
    """
    Get detailed storage information and statistics.
    In production, this would require admin authentication.
    """
    try:
        logger.info("Retrieving storage information")

        # Get storage statistics
        storage_stats_result = await storage_service.get_storage_stats()
        if storage_stats_result.is_err():
            logger.error(f"Failed to get storage stats: {storage_stats_result.unwrap_or('Unknown error')}")
            raise HTTPException(status_code=500, detail="Failed to retrieve storage information")

        storage_stats = storage_stats_result.unwrap()

        # Add configuration information
        storage_info = {
            "statistics": storage_stats,
            "configuration": {
                "storage_type": "s3" if settings.use_s3_storage else "local",
                "max_file_size_mb": settings.max_file_size_mb,
                "retention_days": settings.retention_days,
                "supported_formats": settings.supported_formats
            },
            "s3_config": {
                "enabled": settings.use_s3_storage,
                "bucket": settings.s3_bucket if settings.use_s3_storage else None,
                "region": settings.aws_region if settings.use_s3_storage else None
            } if settings.use_s3_storage else None,
            "local_config": {
                "storage_path": settings.local_storage_path,
                "available_space_gb": storage_stats.get("available_space_bytes", 0) / (1024**3) if storage_stats.get("available_space_bytes") else None
            } if not settings.use_s3_storage else None
        }

        logger.info(f"Storage info retrieved: {storage_stats.get('total_files', 0)} files, {storage_stats.get('total_size_bytes', 0)} bytes")
        return storage_info

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error getting storage info: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve storage information")


async def _cleanup_expired_recordings(expired_recordings: List, file_paths: List[str]):
    """Background task for cleaning up expired recordings."""
    try:
        logger.info(f"Starting cleanup of {len(expired_recordings)} expired recordings")

        # Clean up storage files
        if file_paths:
            cleanup_result = await storage_service.cleanup_expired_files(file_paths)
            if cleanup_result.is_err():
                logger.error(f"Storage cleanup failed: {cleanup_result.unwrap_or('Unknown error')}")
            else:
                cleanup_stats = cleanup_result.unwrap()
                logger.info(f"Storage cleanup completed: {cleanup_stats}")

        # Apply retention policy to each recording (removes file references)
        success_count = 0
        for recording in expired_recordings:
            retention_result = await database_service.apply_retention_policy(recording.id)
            if retention_result.is_ok():
                success_count += 1
            else:
                logger.error(f"Failed to apply retention policy to recording {recording.id}: {retention_result.unwrap_or('Unknown error')}")

        logger.info(f"Retention cleanup completed: {success_count}/{len(expired_recordings)} recordings processed")

    except Exception as e:
        logger.error(f"Unexpected error in retention cleanup task: {str(e)}")


# Add a simple status endpoint for load balancers
@router.get(
    "/status",
    summary="Simple status check",
    description="Lightweight status check for load balancers"
)
async def status_check() -> Dict[str, str]:
    """Simple status endpoint for load balancer health checks."""
    return {
        "status": "ok",
        "service": settings.app_name,
        "version": settings.app_version,
        "timestamp": datetime.now().isoformat()
    }