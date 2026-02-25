#!/bin/bash
# Build F1 Winner Predictor Mac .app
# Run from project root. Requires: pip install -r requirements.txt pyinstaller

set -e
cd "$(dirname "$0")"

# Use venv if it exists and works; otherwise system python
if [ -d ".venv" ] && [ -x ".venv/bin/python" ]; then
  echo "Using .venv..."
  PYTHON=".venv/bin/python"
else
  echo "Using system python3..."
  PYTHON="python3"
fi

"$PYTHON" -m pip install -q pyinstaller 2>/dev/null || "$PYTHON" -m pip install pyinstaller

echo "Building Mac application..."
"$PYTHON" -m PyInstaller --clean --noconfirm F1WinnerPredictor.spec

# Rename to .app for double-click launching
if [ -d "dist/F1 Winner Predictor" ] && [ ! -d "dist/F1 Winner Predictor.app" ]; then
  mv "dist/F1 Winner Predictor" "dist/F1 Winner Predictor.app"
fi

echo ""
echo "Done! App is at: dist/F1 Winner Predictor.app"
