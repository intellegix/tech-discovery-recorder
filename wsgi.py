"""
WSGI entry point for Render deployment.
Adds src/ to Python path and imports the FastAPI app.
"""
import sys
import os
import traceback

# Add src directory to Python path so absolute imports work
src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.insert(0, src_dir)

# Debug: print environment and path info
print(f"[WSGI] Python version: {sys.version}")
print(f"[WSGI] Working directory: {os.getcwd()}")
print(f"[WSGI] src_dir added to path: {src_dir}")
print(f"[WSGI] src_dir exists: {os.path.exists(src_dir)}")
print(f"[WSGI] sys.path: {sys.path[:5]}")

# Check critical env vars
for key in ["DATABASE_URL", "JWT_SECRET", "CLAUDE_API_KEY", "OPENAI_API_KEY"]:
    val = os.environ.get(key, "NOT SET")
    if val != "NOT SET":
        print(f"[WSGI] {key}: SET ({len(val)} chars)")
    else:
        print(f"[WSGI] {key}: *** NOT SET ***")

# List files in src_dir
if os.path.exists(src_dir):
    print(f"[WSGI] Files in src_dir: {os.listdir(src_dir)}")

try:
    from main import app  # noqa: E402
    print("[WSGI] Successfully imported app from main")
except Exception as e:
    print(f"[WSGI] FATAL: Failed to import app: {e}")
    traceback.print_exc()
    raise
