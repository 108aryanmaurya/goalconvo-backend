#!/bin/bash
# Quick start script for GoalConvo backend server

cd "$(dirname "$0")"

echo "Starting GoalConvo Backend Server..."
echo ""

# Check if virtual environment exists, create if not
if [ ! -d "venv" ]; then
    echo "⚠️  Virtual environment not found. Creating one..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
# python -m pip install --upgrade pip --quiet

# Check if Flask is installed
# if ! python -c "import flask" 2>/dev/null; then
#     echo "Installing Flask dependencies..."
#     pip install flask flask-cors
# fi

# Check if other requirements are installed
# if ! python -c "import goalconvo" 2>/dev/null; then
#     echo "Installing GoalConvo dependencies..."
#     pip install -r requirements.txt
# fi

# Set default port if not set
export PORT=${PORT:-5000}
export HOST=${HOST:-0.0.0.0}

echo "Starting server on $HOST:$PORT"
echo "API endpoints will be available at:"
echo "  POST http://localhost:$PORT/api/run-pipeline"
echo "  POST http://localhost:$PORT/api/run-evaluation"
echo "  GET  http://localhost:$PORT/api/versions"
echo ""
echo "Health check: http://localhost:$PORT/health"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python backend_server.py

