"""
Authentication service for JWT token management.
Implements secure session handling following Austin's patterns.
"""

import uuid
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any
from jose import JWTError, jwt
from passlib.context import CryptContext
import logging

from config.settings import settings
from models.database import UserSession
from services.database_service import database_service
from utils.result import Result, Ok, Err, AuthenticationError


logger = logging.getLogger(__name__)


class AuthService:
    """
    Handles JWT token generation, validation, and session management.
    Implements simplified auth for the demo - in production would integrate with
    proper user management system.
    """

    def __init__(self):
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.algorithm = settings.jwt_algorithm
        self.secret_key = settings.jwt_secret
        self.access_token_expire_minutes = settings.access_token_expire_minutes

    async def create_access_token(
        self,
        user_id: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> Result[Dict[str, Any], AuthenticationError]:
        """
        Create JWT access token and session record.

        Returns:
            Dict with access_token, token_type, expires_in, user_id
        """
        try:
            logger.info(f"Creating access token for user {user_id}")

            # Generate session data
            session_id = str(uuid.uuid4())
            now = datetime.now(timezone.utc)
            expires_at = now + timedelta(minutes=self.access_token_expire_minutes)

            # Create JWT payload
            token_data = {
                "user_id": user_id,
                "session_id": session_id,
                "iat": now,
                "exp": expires_at,
                "sub": user_id,  # Subject (standard JWT claim)
                "type": "access_token"
            }

            # Generate JWT token
            access_token = jwt.encode(
                token_data,
                self.secret_key,
                algorithm=self.algorithm
            )

            # Hash the token for storage (security best practice)
            token_hash = self._hash_token(access_token)

            # Create session record in database
            async with database_service.async_session() as session:
                user_session = UserSession(
                    user_id=user_id,
                    session_token_hash=token_hash,
                    expires_at=expires_at,
                    ip_address=ip_address,
                    user_agent=user_agent,
                    is_active=True
                )

                session.add(user_session)
                await session.commit()

            logger.info(f"Created access token for user {user_id}, session {session_id}")

            return Ok({
                "access_token": access_token,
                "token_type": "bearer",
                "expires_in": self.access_token_expire_minutes * 60,  # seconds
                "user_id": user_id,
                "session_id": session_id
            })

        except Exception as e:
            logger.error(f"Failed to create access token for user {user_id}: {str(e)}")
            return Err(AuthenticationError(f"Token creation failed: {str(e)}"))

    async def validate_token(self, token: str) -> Result[Dict[str, Any], AuthenticationError]:
        """
        Validate JWT token and return user information.

        Returns:
            Dict with user_id, session_id, and other token data
        """
        try:
            # Decode JWT token
            try:
                payload = jwt.decode(
                    token,
                    self.secret_key,
                    algorithms=[self.algorithm]
                )
            except JWTError as e:
                logger.warning(f"JWT validation failed: {str(e)}")
                return Err(AuthenticationError("Invalid token"))

            # Extract token data
            user_id = payload.get("user_id")
            session_id = payload.get("session_id")

            if not user_id or not session_id:
                return Err(AuthenticationError("Invalid token payload"))

            # Verify session in database
            token_hash = self._hash_token(token)
            session_result = await self._get_valid_session(token_hash)

            if session_result.is_err():
                return session_result

            session_record = session_result.unwrap()

            # Update last used timestamp
            await self._update_session_last_used(session_record.id)

            logger.debug(f"Token validated for user {user_id}, session {session_id}")

            return Ok({
                "user_id": user_id,
                "session_id": session_id,
                "session_record": session_record,
                "token_payload": payload
            })

        except Exception as e:
            logger.error(f"Unexpected error validating token: {str(e)}")
            return Err(AuthenticationError(f"Token validation failed: {str(e)}"))

    async def revoke_token(self, token: str) -> Result[None, AuthenticationError]:
        """Revoke a specific token/session."""
        try:
            token_hash = self._hash_token(token)

            async with database_service.async_session() as session:
                # Find and revoke the session
                from sqlalchemy import select, update
                stmt = update(UserSession).where(
                    UserSession.session_token_hash == token_hash
                ).values(
                    is_active=False,
                    revoked_at=datetime.now(timezone.utc)
                )

                result = await session.execute(stmt)
                await session.commit()

                if result.rowcount == 0:
                    return Err(AuthenticationError("Token not found or already revoked"))

            logger.info(f"Token revoked: {token_hash[:8]}...")
            return Ok(None)

        except Exception as e:
            logger.error(f"Failed to revoke token: {str(e)}")
            return Err(AuthenticationError(f"Token revocation failed: {str(e)}"))

    async def revoke_all_user_sessions(self, user_id: str) -> Result[int, AuthenticationError]:
        """Revoke all active sessions for a user."""
        try:
            async with database_service.async_session() as session:
                from sqlalchemy import update
                now = datetime.now(timezone.utc)

                stmt = update(UserSession).where(
                    UserSession.user_id == user_id,
                    UserSession.is_active == True
                ).values(
                    is_active=False,
                    revoked_at=now
                )

                result = await session.execute(stmt)
                await session.commit()

                revoked_count = result.rowcount
                logger.info(f"Revoked {revoked_count} sessions for user {user_id}")
                return Ok(revoked_count)

        except Exception as e:
            logger.error(f"Failed to revoke user sessions for {user_id}: {str(e)}")
            return Err(AuthenticationError(f"Session revocation failed: {str(e)}"))

    async def cleanup_expired_sessions(self) -> Result[int, AuthenticationError]:
        """Clean up expired sessions (maintenance task)."""
        try:
            async with database_service.async_session() as session:
                from sqlalchemy import delete
                now = datetime.now(timezone.utc)

                # Delete sessions that have been expired for more than 24 hours
                cleanup_threshold = now - timedelta(hours=24)

                stmt = delete(UserSession).where(
                    UserSession.expires_at < cleanup_threshold
                )

                result = await session.execute(stmt)
                await session.commit()

                cleaned_count = result.rowcount
                logger.info(f"Cleaned up {cleaned_count} expired sessions")
                return Ok(cleaned_count)

        except Exception as e:
            logger.error(f"Failed to cleanup expired sessions: {str(e)}")
            return Err(AuthenticationError(f"Session cleanup failed: {str(e)}"))

    async def get_user_sessions(self, user_id: str) -> Result[list, AuthenticationError]:
        """Get active sessions for a user."""
        try:
            async with database_service.async_session() as session:
                from sqlalchemy import select

                stmt = select(UserSession).where(
                    UserSession.user_id == user_id,
                    UserSession.is_active == True
                ).order_by(UserSession.last_used_at.desc())

                result = await session.execute(stmt)
                sessions = result.scalars().all()

                # Convert to serializable format
                session_data = []
                for sess in sessions:
                    session_data.append({
                        "session_id": sess.id,
                        "created_at": sess.created_at.isoformat(),
                        "last_used_at": sess.last_used_at.isoformat(),
                        "expires_at": sess.expires_at.isoformat(),
                        "ip_address": sess.ip_address,
                        "user_agent": sess.user_agent,
                        "is_expired": sess.is_expired
                    })

                return Ok(session_data)

        except Exception as e:
            logger.error(f"Failed to get user sessions for {user_id}: {str(e)}")
            return Err(AuthenticationError(f"Failed to get user sessions: {str(e)}"))

    def _hash_token(self, token: str) -> str:
        """Hash token for secure storage."""
        return hashlib.sha256(token.encode()).hexdigest()

    async def _get_valid_session(self, token_hash: str) -> Result[UserSession, AuthenticationError]:
        """Get valid session by token hash."""
        try:
            async with database_service.async_session() as session:
                from sqlalchemy import select

                stmt = select(UserSession).where(
                    UserSession.session_token_hash == token_hash
                )

                result = await session.execute(stmt)
                session_record = result.scalar_one_or_none()

                if not session_record:
                    return Err(AuthenticationError("Session not found"))

                if not session_record.is_valid:
                    return Err(AuthenticationError("Session expired or revoked"))

                return Ok(session_record)

        except Exception as e:
            return Err(AuthenticationError(f"Session validation failed: {str(e)}"))

    async def _update_session_last_used(self, session_id: str) -> None:
        """Update session last used timestamp."""
        try:
            async with database_service.async_session() as session:
                from sqlalchemy import update

                stmt = update(UserSession).where(
                    UserSession.id == session_id
                ).values(last_used_at=datetime.now(timezone.utc))

                await session.execute(stmt)
                await session.commit()

        except Exception as e:
            logger.warning(f"Failed to update session last_used timestamp: {str(e)}")


# Global auth service instance
auth_service = AuthService()


# FastAPI dependency for authentication
async def get_current_user(authorization: Optional[str] = None) -> Result[str, AuthenticationError]:
    """
    FastAPI dependency to extract and validate JWT token.
    Returns user_id if valid, error otherwise.
    """
    if not authorization or not authorization.startswith("Bearer "):
        return Err(AuthenticationError("Missing or invalid authorization header"))

    token = authorization.replace("Bearer ", "")
    validation_result = await auth_service.validate_token(token)

    if validation_result.is_err():
        return validation_result

    token_data = validation_result.unwrap()
    return Ok(token_data["user_id"])


# Demo user creation for simplified auth
async def create_demo_user(user_id: Optional[str] = None) -> str:
    """
    Create a demo user ID for simplified authentication.
    In production, this would integrate with proper user management.
    """
    if not user_id:
        user_id = f"user-{str(uuid.uuid4())[:8]}"

    logger.info(f"Created demo user: {user_id}")
    return user_id