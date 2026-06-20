#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "🚀 Starting EmbedShield services..."

# Start FastAPI Gateway in the background on port 8000
echo "📡 Launching FastAPI API Gateway on port 8000..."
uvicorn app:app --host 0.0.0.0 --port 8000 &
FASTAPI_PID=$!

# Wait briefly for FastAPI to initialize and cache the HF model
echo "⏳ Waiting for API Gateway to boot and load Hugging Face model..."
sleep 5

# Start Streamlit Dashboard in the background on port 8501
echo "📊 Launching Streamlit Dashboard on port 8501..."
streamlit run dashboard.py --server.port 8501 --server.address 0.0.0.0 &
STREAMLIT_PID=$!

# Define cleanup function to gracefully terminate both background tasks
cleanup() {
    echo "🛑 Shutting down EmbedShield processes gracefully..."
    kill $FASTAPI_PID 2>/dev/null || true
    kill $STREAMLIT_PID 2>/dev/null || true
}

# Bind cleanup to termination signals (SIGINT, SIGTERM, EXIT)
trap cleanup SIGINT SIGTERM EXIT

# Keep script running and monitor processes
echo "🛡️ EmbedShield is fully operational! Press Ctrl+C to terminate."
wait $FASTAPI_PID $STREAMLIT_PID
