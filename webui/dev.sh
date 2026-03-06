#!/usr/bin/env bash
# Start backend + frontend dev servers in parallel
set -e
cd "$(dirname "$0")/.."

echo "Starting backend on http://localhost:8001 ..."
uv run uvicorn webui.backend.main:app --reload --port 8001 &
BACKEND_PID=$!

echo "Starting frontend on http://localhost:5173 ..."
cd webui/frontend && npm run dev &
FRONTEND_PID=$!

trap "echo 'Stopping...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null" EXIT INT TERM
echo "Both servers started. Press Ctrl+C to stop."
wait
