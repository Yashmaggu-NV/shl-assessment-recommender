"""
Vercel serverless entry point.

Wraps the FastAPI app for Vercel's Python runtime.
"""
import sys
from pathlib import Path

# Add project root to path so imports work
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import the FastAPI app
from app import app

# Vercel expects the ASGI app to be named 'app' or 'handler'
# FastAPI is already ASGI-compatible
