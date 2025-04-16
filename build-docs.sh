#!/bin/bash
set -e

# Default port for the documentation server
PORT=${1:-8001}

# Set locale to C.UTF-8 which is commonly available
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

# Function to handle Ctrl+C gracefully
function cleanup {
    echo -e "\nShutting down documentation server..."
    exit 0
}

# Set up trap for Ctrl+C
trap cleanup SIGINT

# Activate the virtual environment if it exists
if [ -d "/workspace/kan_venv" ]; then
    source /workspace/kan_venv/bin/activate
fi

# Clean the build directory first to ensure a fresh build
echo "Cleaning previous build..."
rm -rf /workspace/docs/build

# Direct sphinx-build approach instead of using tox
# This bypasses the packaging backend issues
cd /workspace
echo "Building documentation..."
sphinx-build -b html docs/source docs/build/html

echo "Documentation built successfully!"
echo "Starting documentation server on http://localhost:$PORT"
echo "Press Ctrl+C to stop the server"

# Serve the documentation
cd /workspace/docs/build/html
python3 -m http.server $PORT
