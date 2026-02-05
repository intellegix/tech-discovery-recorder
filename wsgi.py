"""
WSGI entry point for Render deployment.
Adds src/ to Python path and imports the FastAPI app.
"""
import sys
import os

# Add src directory to Python path so absolute imports work
src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.insert(0, src_dir)

from main import app  # noqa: E402
