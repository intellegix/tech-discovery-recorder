"""
Main FastAPI application for Tech Discovery Recorder.
Implements the backend architecture from the plan.
"""

import logging
from contextlib import asynccontextmanager
from typing import Dict, Any
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from .config.settings import settings
from .services.database_service import database_service
from .routes import recordings, admin
from .utils.result import Result


# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    logger.info("🚀 Starting Tech Discovery Recorder backend")

    # Initialize database tables
    logger.info("Initializing database...")
    db_init_result = await database_service.create_tables()
    if db_init_result.is_err():
        logger.error(f"Failed to initialize database: {db_init_result.unwrap_or('Unknown error')}")
        raise Exception("Database initialization failed")

    logger.info("✅ Database initialized successfully")
    logger.info(f"🎯 Server ready on port 8000 (debug={settings.debug})")

    yield  # Server is running

    logger.info("🛑 Shutting down Tech Discovery Recorder backend")


# Create FastAPI application
app = FastAPI(
    title="Tech Discovery Recorder API",
    description="""
    Backend API for compliant audio recording and AI transcription.

    ## Features
    - 🎙️ Audio recording with legal consent tracking
    - 🤖 Claude AI transcription and structured note extraction
    - 📋 Audit logging for compliance
    - 🔒 Encryption at rest and in transit
    - ⏰ Automatic retention policy enforcement
    - 🏥 Health monitoring and admin tools

    ## Compliance
    - California Penal Code § 632 compliance (all-party consent)
    - CCPA/CPRA privacy protections
    - Immutable audit trails
    - No unauthorized third-party data sharing

    ## Architecture
    Built with FastAPI, PostgreSQL, Claude API, and optional AWS S3 storage.
    """,
    version=settings.app_version,
    lifespan=lifespan,
    debug=settings.debug,
    docs_url="/docs" if settings.debug else None,  # Disable docs in production
    redoc_url="/redoc" if settings.debug else None
)


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_allowed_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"]
)


# Add security middleware
if settings.is_production:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*.render.com", "*.amazonaws.com", "localhost", "127.0.0.1"]
    )


# Include routers
app.include_router(recordings.router)
app.include_router(admin.router)


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled exception on {request.method} {request.url}: {str(exc)}", exc_info=True)

    if settings.debug:
        # In debug mode, show detailed error information
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal Server Error",
                "message": str(exc),
                "detail": exc.__class__.__name__,
                "path": str(request.url.path)
            }
        )
    else:
        # In production, show generic error message
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal Server Error",
                "message": "An unexpected error occurred",
                "request_id": request.headers.get("X-Request-ID")
            }
        )


# HTTP exception handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Custom HTTP exception handler."""
    logger.warning(f"HTTP {exc.status_code} on {request.method} {request.url}: {exc.detail}")

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": f"HTTP {exc.status_code}",
            "message": exc.detail,
            "path": str(request.url.path)
        }
    )


# Root endpoint
@app.get(
    "/",
    summary="API Root",
    description="Get API information and health status"
)
async def root() -> Dict[str, Any]:
    """API root endpoint with basic information."""
    return {
        "service": "Tech Discovery Recorder API",
        "version": settings.app_version,
        "status": "operational",
        "description": "Backend API for compliant audio recording and AI transcription",
        "documentation": "/docs" if settings.debug else "Contact administrator",
        "health_check": "/api/v1/health",
        "compliance": {
            "california_penal_code_632": "All-party consent enforced",
            "ccpa_cpra": "Privacy protections implemented",
            "audit_logging": "Immutable audit trails maintained",
            "data_isolation": "No unauthorized third-party sharing"
        },
        "features": [
            "Audio recording with consent tracking",
            "Claude AI transcription and structured notes",
            "Automatic retention policy enforcement",
            "Encrypted storage (local or S3)",
            "Comprehensive audit logging",
            "Health monitoring and admin tools"
        ]
    }


# Add middleware to add request ID header
@app.middleware("http")
async def add_request_id_header(request: Request, call_next):
    """Add unique request ID to all requests for tracing."""
    import uuid
    request_id = str(uuid.uuid4())[:8]

    # Add to request state for logging
    request.state.request_id = request_id

    # Process request
    response = await call_next(request)

    # Add to response headers
    response.headers["X-Request-ID"] = request_id

    return response


# Add request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests for monitoring and debugging."""
    import time

    start_time = time.time()

    # Process request
    response = await call_next(request)

    # Calculate duration
    duration = time.time() - start_time

    # Log request
    request_id = getattr(request.state, 'request_id', 'unknown')
    logger.info(
        f"{request.method} {request.url.path} - {response.status_code} - "
        f"{duration:.3f}s - ID:{request_id}"
    )

    return response


if __name__ == "__main__":
    # Run the application directly (for development)
    logger.info("Starting development server...")
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
        access_log=True
    )