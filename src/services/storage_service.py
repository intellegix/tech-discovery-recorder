"""
Storage service for audio files with S3 and local storage support.
Implements encryption and follows Austin's async patterns.
"""

import os
import aiofiles
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, BinaryIO, Union
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
import logging

from ..config.settings import settings
from ..utils.result import Result, Ok, Err, StorageError, wrap_result


logger = logging.getLogger(__name__)


class StorageService:
    """
    Handles audio file storage with support for both local and S3 storage.
    Provides encryption, integrity checking, and retention management.
    """

    def __init__(self):
        self.use_s3 = settings.use_s3_storage
        self.local_path = Path(settings.local_storage_path)
        self._s3_client: Optional[boto3.client] = None
        self._ensure_local_storage()

    def _ensure_local_storage(self) -> None:
        """Ensure local storage directory exists."""
        self.local_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Local storage directory ensured: {self.local_path}")

    @property
    def s3_client(self) -> boto3.client:
        """Lazy initialization of S3 client."""
        if self._s3_client is None:
            if not self.use_s3:
                raise StorageError("S3 storage not configured")

            try:
                self._s3_client = boto3.client(
                    's3',
                    aws_access_key_id=settings.aws_access_key_id,
                    aws_secret_access_key=settings.aws_secret_access_key,
                    region_name=settings.aws_region
                )
                logger.info("S3 client initialized successfully")
            except (ClientError, NoCredentialsError) as e:
                raise StorageError(f"Failed to initialize S3 client: {str(e)}")

        return self._s3_client

    def _generate_file_path(self, recording_id: str, file_extension: str) -> str:
        """
        Generate storage path for audio file.
        Uses date-based partitioning for better organization.
        """
        date_prefix = datetime.now().strftime("%Y/%m/%d")
        filename = f"{recording_id}.{file_extension.lstrip('.')}"
        return f"recordings/{date_prefix}/{filename}"

    def _calculate_file_hash(self, content: bytes) -> str:
        """Calculate SHA-256 hash for file integrity."""
        return hashlib.sha256(content).hexdigest()

    async def store_audio_file(
        self,
        recording_id: str,
        audio_content: bytes,
        content_type: str
    ) -> Result[dict, StorageError]:
        """
        Store audio file with encryption and integrity checking.

        Returns:
            Result containing storage metadata (path, hash, size) or error.
        """
        try:
            # Determine file extension from content type
            extension_map = {
                "audio/webm": "webm",
                "audio/wav": "wav",
                "audio/mp4": "mp4",
                "audio/m4a": "m4a",
                "audio/mpeg": "mp3"
            }
            extension = extension_map.get(content_type, "audio")

            # Generate file path and calculate integrity hash
            file_path = self._generate_file_path(recording_id, extension)
            file_hash = self._calculate_file_hash(audio_content)
            file_size = len(audio_content)

            # Validate file size
            max_size_bytes = settings.max_file_size_mb * 1024 * 1024
            if file_size > max_size_bytes:
                return Err(StorageError(f"File size {file_size} exceeds maximum {max_size_bytes} bytes"))

            # Store the file
            if self.use_s3:
                storage_result = await self._store_s3(file_path, audio_content, content_type)
            else:
                storage_result = await self._store_local(file_path, audio_content)

            if storage_result.is_err():
                return storage_result

            metadata = {
                "file_path": file_path,
                "file_hash": file_hash,
                "file_size": file_size,
                "content_type": content_type,
                "storage_type": "s3" if self.use_s3 else "local",
                "created_at": datetime.now().isoformat()
            }

            logger.info(f"Stored audio file for recording {recording_id}: {file_path}")
            return Ok(metadata)

        except Exception as e:
            logger.error(f"Failed to store audio file for recording {recording_id}: {str(e)}")
            return Err(StorageError(f"Storage operation failed: {str(e)}"))

    async def _store_s3(self, file_path: str, content: bytes, content_type: str) -> Result[None, StorageError]:
        """Store file in S3 with encryption."""
        try:
            self.s3_client.put_object(
                Bucket=settings.s3_bucket,
                Key=file_path,
                Body=content,
                ContentType=content_type,
                ServerSideEncryption='AES256',  # Encryption at rest
                Metadata={
                    'uploaded_at': datetime.now().isoformat(),
                    'service': 'tech-discovery-recorder'
                }
            )
            return Ok(None)
        except ClientError as e:
            return Err(StorageError(f"S3 upload failed: {str(e)}"))

    async def _store_local(self, file_path: str, content: bytes) -> Result[None, StorageError]:
        """Store file locally with directory creation."""
        try:
            full_path = self.local_path / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)

            async with aiofiles.open(full_path, 'wb') as f:
                await f.write(content)

            return Ok(None)
        except Exception as e:
            return Err(StorageError(f"Local storage failed: {str(e)}"))

    async def retrieve_audio_file(self, file_path: str) -> Result[bytes, StorageError]:
        """
        Retrieve audio file content.
        """
        try:
            if self.use_s3:
                content_result = await self._retrieve_s3(file_path)
            else:
                content_result = await self._retrieve_local(file_path)

            if content_result.is_err():
                return content_result

            logger.info(f"Retrieved audio file: {file_path}")
            return content_result

        except Exception as e:
            logger.error(f"Failed to retrieve audio file {file_path}: {str(e)}")
            return Err(StorageError(f"Retrieval operation failed: {str(e)}"))

    async def _retrieve_s3(self, file_path: str) -> Result[bytes, StorageError]:
        """Retrieve file from S3."""
        try:
            response = self.s3_client.get_object(Bucket=settings.s3_bucket, Key=file_path)
            return Ok(response['Body'].read())
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                return Err(StorageError(f"File not found: {file_path}"))
            return Err(StorageError(f"S3 download failed: {str(e)}"))

    async def _retrieve_local(self, file_path: str) -> Result[bytes, StorageError]:
        """Retrieve file from local storage."""
        try:
            full_path = self.local_path / file_path
            if not full_path.exists():
                return Err(StorageError(f"File not found: {file_path}"))

            async with aiofiles.open(full_path, 'rb') as f:
                content = await f.read()
            return Ok(content)
        except Exception as e:
            return Err(StorageError(f"Local retrieval failed: {str(e)}"))

    async def delete_audio_file(self, file_path: str) -> Result[None, StorageError]:
        """
        Delete audio file (for retention policy enforcement).
        """
        try:
            if self.use_s3:
                delete_result = await self._delete_s3(file_path)
            else:
                delete_result = await self._delete_local(file_path)

            if delete_result.is_ok():
                logger.info(f"Deleted audio file: {file_path}")

            return delete_result

        except Exception as e:
            logger.error(f"Failed to delete audio file {file_path}: {str(e)}")
            return Err(StorageError(f"Deletion operation failed: {str(e)}"))

    async def _delete_s3(self, file_path: str) -> Result[None, StorageError]:
        """Delete file from S3."""
        try:
            self.s3_client.delete_object(Bucket=settings.s3_bucket, Key=file_path)
            return Ok(None)
        except ClientError as e:
            return Err(StorageError(f"S3 deletion failed: {str(e)}"))

    async def _delete_local(self, file_path: str) -> Result[None, StorageError]:
        """Delete file from local storage."""
        try:
            full_path = self.local_path / file_path
            if full_path.exists():
                full_path.unlink()
            return Ok(None)
        except Exception as e:
            return Err(StorageError(f"Local deletion failed: {str(e)}"))

    async def verify_file_integrity(self, file_path: str, expected_hash: str) -> Result[bool, StorageError]:
        """
        Verify file integrity by comparing hashes.
        """
        content_result = await self.retrieve_audio_file(file_path)
        if content_result.is_err():
            return content_result.map_err(lambda e: StorageError(f"Cannot verify integrity: {str(e)}"))

        actual_hash = self._calculate_file_hash(content_result.unwrap())
        return Ok(actual_hash == expected_hash)

    async def cleanup_expired_files(self, expired_file_paths: list[str]) -> Result[dict, StorageError]:
        """
        Clean up expired files for retention policy enforcement.

        Returns:
            Result containing cleanup statistics.
        """
        deleted_count = 0
        failed_count = 0
        errors = []

        for file_path in expired_file_paths:
            delete_result = await self.delete_audio_file(file_path)
            if delete_result.is_ok():
                deleted_count += 1
            else:
                failed_count += 1
                errors.append(f"{file_path}: {str(delete_result.unwrap_or('Unknown error'))}")

        stats = {
            "deleted_count": deleted_count,
            "failed_count": failed_count,
            "errors": errors
        }

        logger.info(f"Cleanup completed: {deleted_count} deleted, {failed_count} failed")
        return Ok(stats)

    async def get_storage_stats(self) -> Result[dict, StorageError]:
        """Get storage usage statistics."""
        try:
            if self.use_s3:
                # For S3, we'd need to list objects and sum sizes
                # This is a simplified version
                return Ok({
                    "storage_type": "s3",
                    "total_files": 0,  # Would require listing all objects
                    "total_size_bytes": 0,
                    "available_space_bytes": None  # S3 has virtually unlimited space
                })
            else:
                # For local storage, calculate directory size
                total_size = 0
                file_count = 0

                for file_path in self.local_path.rglob('*'):
                    if file_path.is_file():
                        total_size += file_path.stat().st_size
                        file_count += 1

                # Get available space
                statvfs = os.statvfs(self.local_path)
                available_space = statvfs.f_frsize * statvfs.f_bavail

                return Ok({
                    "storage_type": "local",
                    "total_files": file_count,
                    "total_size_bytes": total_size,
                    "available_space_bytes": available_space,
                    "storage_path": str(self.local_path)
                })

        except Exception as e:
            return Err(StorageError(f"Failed to get storage stats: {str(e)}"))


# Global storage service instance
storage_service = StorageService()