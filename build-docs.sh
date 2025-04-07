#!/bin/bash
set -e

# Set locale to en_US.UTF-8
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

# Activate the virtual environment if it exists
if [ -d "/workspace/kan_venv" ]; then
    source /workspace/kan_venv/bin/activate
fi

# Direct sphinx-build approach instead of using tox
# This bypasses the packaging backend issues
cd /workspace
sphinx-build -b html docs/source docs/build/html

echo "Documentation built successfully! Open with:"
echo "\$BROWSER /workspace/docs/build/html/index.html"
