#!/bin/bash
# Start the Cryo-EM 3D Explorer

echo "Starting Cryo-EM 3D Explorer..."
echo ""
echo "Once the server starts, open your browser to:"
echo "  http://localhost:8000"
echo ""
echo "Press Ctrl+C to stop the server."
echo ""

cd "$(dirname "$0")"
python3 server.py
