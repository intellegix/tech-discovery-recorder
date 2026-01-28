#!/usr/bin/env python3
"""
Startup script for Tech Discovery Recorder backend.
Handles environment setup and graceful startup.
"""

import os
import sys
import logging
import asyncio
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))


async def check_dependencies():
    """Check that all required dependencies and services are available."""
    print("[CHECK] Checking dependencies...")

    # Check environment variables
    required_env_vars = ["DATABASE_URL", "JWT_SECRET", "CLAUDE_API_KEY"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]

    if missing_vars:
        print(f"[ERROR] Missing required environment variables: {missing_vars}")
        print("[INFO] Create a .env file with required variables. See .env.example")
        return False

    # Test database connection
    try:
        from src.services.database_service import database_service
        db_health = await database_service.health_check()
        if db_health.is_err():
            print(f"[ERROR] Database connection failed: {db_health.unwrap_or('Unknown error')}")
            return False
        print("[OK] Database connection verified")
    except Exception as e:
        print(f"[ERROR] Database check failed: {str(e)}")
        return False

    # Test Claude API (optional for demo mode)
    try:
        try:
            from src.services.claude_service import claude_service
        except ImportError:
            from src.services.claude_service_minimal import claude_service

        claude_health = await claude_service.health_check()
        if claude_health.is_err():
            print(f"[ERROR] Claude API connection failed: {claude_health.unwrap_or('Unknown error')}")
            return False
        print("[OK] Claude API connection verified (demo mode)")
    except Exception as e:
        print(f"[WARNING] Claude API check failed (using demo mode): {str(e)}")
        print("[OK] Continuing with demo mode")

    # Check storage
    try:
        from src.services.storage_service import storage_service
        storage_stats = await storage_service.get_storage_stats()
        if storage_stats.is_err():
            print(f"[WARNING] Storage check failed: {storage_stats.unwrap_or('Unknown error')}")
            print("[OK] Continuing with basic storage")
        else:
            print("[OK] Storage system verified")
    except Exception as e:
        print(f"[WARNING] Storage check failed: {str(e)}")
        print("[OK] Continuing with basic storage")

    print("[OK] All dependency checks passed")
    return True


async def setup_database():
    """Initialize database tables."""
    print("[DB] Setting up database...")

    try:
        from src.services.database_service import database_service
        result = await database_service.create_tables()
        if result.is_err():
            print(f"[ERROR] Database setup failed: {result.unwrap_or('Unknown error')}")
            return False
        print("[OK] Database tables created/verified")
        return True
    except Exception as e:
        print(f"[ERROR] Database setup failed: {str(e)}")
        return False


def main():
    """Main startup function."""
    print("[STARTUP] Starting Tech Discovery Recorder Backend")
    print("=" * 50)

    # Load environment variables
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        print("[INFO] python-dotenv not installed, relying on system environment variables")

    # Check dependencies and setup
    async def startup_checks():
        if not await check_dependencies():
            print("\n[ERROR] Startup checks failed. Please fix the issues above.")
            sys.exit(1)

        if not await setup_database():
            print("\n[ERROR] Database setup failed. Please check your database configuration.")
            sys.exit(1)

        print("\n[READY] All systems ready! Starting server...")

    # Run startup checks
    asyncio.run(startup_checks())

    # Start the server
    try:
        import uvicorn
        from src.config.settings import settings

        print(f"[SERVER] Server starting on http://0.0.0.0:8000")
        print(f"[DOCS] API docs: http://localhost:8000/docs")
        print(f"[HEALTH] Health check: http://localhost:8000/api/v1/health")
        print(f"[DEBUG] Debug mode: {settings.debug}")
        print("\n" + "=" * 50)

        uvicorn.run(
            "src.main:app",
            host="0.0.0.0",
            port=8000,
            reload=settings.debug,
            log_level=settings.log_level.lower(),
            access_log=True
        )

    except KeyboardInterrupt:
        print("\n\n[SHUTDOWN] Graceful shutdown requested")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] Server startup failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()