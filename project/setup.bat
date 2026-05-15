@echo off
echo ================================================
echo SHL Assessment Recommender - Setup Script
echo ================================================
echo.

echo [1/4] Installing Python dependencies...
pip install openai fastapi "uvicorn[standard]" pydantic sentence-transformers faiss-cpu numpy httpx python-dotenv pytest
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: pip install failed. Make sure Python and pip are in your PATH.
    pause
    exit /b 1
)
echo Dependencies installed successfully!
echo.

echo [2/4] Creating .env file from template...
if not exist .env (
    copy .env.example .env
    echo .env file created. Please edit it to add your XAI_API_KEY.
) else (
    echo .env file already exists - skipping.
)
echo.

echo [3/4] Building FAISS index...
python embeddings/build_index.py
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: FAISS index build failed. The server will use keyword-only search.
) else (
    echo FAISS index built successfully!
)
echo.

echo [4/4] Running quick tests (no LLM needed)...
python -m pytest tests/test_guards.py tests/test_state.py -v --tb=short 2>nul
echo.

echo ================================================
echo Setup complete!
echo.
echo To start the server, run:
echo   uvicorn app:app --host 0.0.0.0 --port 8000 --reload
echo.
echo Don't forget to set your XAI_API_KEY in .env
echo ================================================
pause
