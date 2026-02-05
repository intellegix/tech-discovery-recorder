"""
Main FastAPI application for Tech Discovery Recorder.
Implements the backend architecture from the plan.
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Dict, Any
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from config.settings import settings
from services.database_service import database_service
from routes import recordings, admin
from utils.result import Result


# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    logger.info("Starting Tech Discovery Recorder backend")

    # Initialize database tables with retry logic
    logger.info("Initializing database...")
    max_retries = 5
    retry_delay = 2

    for attempt in range(max_retries):
        try:
            db_init_result = await database_service.create_tables()
            if db_init_result.is_ok():
                logger.info("Database initialized successfully")
                break
            else:
                error = db_init_result.unwrap_err()
                logger.warning(f"Database initialization attempt {attempt + 1}/{max_retries} failed: {error}")
                if attempt == max_retries - 1:
                    logger.error(f"Failed to initialize database after {max_retries} attempts: {error}")
                    raise Exception(f"Database initialization failed after {max_retries} attempts: {error}")

                # Wait before retry
                import asyncio
                await asyncio.sleep(retry_delay)
                retry_delay *= 1.5  # Exponential backoff
        except Exception as e:
            logger.warning(f"Database initialization attempt {attempt + 1}/{max_retries} failed with exception: {e}")
            if attempt == max_retries - 1:
                logger.error(f"Failed to initialize database after {max_retries} attempts: {e}")
                raise Exception(f"Database initialization failed after {max_retries} attempts: {e}")

            # Wait before retry
            import asyncio
            await asyncio.sleep(retry_delay)
            retry_delay *= 1.5  # Exponential backoff

    logger.info(f"Server ready on port 8000 (debug={settings.debug})")

    yield  # Server is running

    logger.info("Shutting down Tech Discovery Recorder backend")


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


# Mount static files
static_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
if os.path.exists(static_path):
    app.mount("/static", StaticFiles(directory=static_path), name="static")
    logger.info(f"Static files mounted from: {static_path}")
else:
    logger.warning(f"Static directory not found: {static_path}")


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


# Root endpoint - serve HTML interface for browsers, JSON for API clients
@app.get(
    "/",
    summary="Tech Discovery Recorder Interface",
    description="Serve the web interface for browsers or API information for API clients"
)
async def root(request: Request):
    """Root endpoint - serves HTML interface or API information."""
    # Check if this is a browser request (Accept header contains text/html)
    accept_header = request.headers.get("accept", "")

    if "text/html" in accept_header:
        # Browser request - serve HTML interface
        static_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static", "index.html")
        if os.path.exists(static_path):
            return FileResponse(static_path, media_type="text/html")
        else:
            logger.warning(f"HTML file not found: {static_path}")
            # Fallback to inline HTML
            return FileResponse(
                path=os.path.join(os.path.dirname(os.path.dirname(__file__)), "Tech Discovery Recorder.html"),
                media_type="text/html"
            )

    # API request - return JSON information
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