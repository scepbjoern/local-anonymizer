#!/bin/bash
# Move to the script's directory
cd "$(dirname "$0")"

echo "======================================================="
echo "  Starting Privacy-First Local Anonymizer on macOS..."
echo "======================================================="

# Check uv
if ! command -v uv &> /dev/null; then
    echo "[FEHLER] 'uv' wurde nicht gefunden."
    echo "Bitte installieren Sie uv mit: curl -LsSf https://astral.sh/uv/install.sh | sh"
    read -p "Drücken Sie Enter zum Beenden..."
    exit 1
fi

# Launch splash screen in background if python is available
if [ -f ".venv/bin/python" ]; then
    .venv/bin/python splash.py &
elif command -v python3 &> /dev/null; then
    python3 splash.py &
fi

# Run main application
uv run --extra gui python app.py
