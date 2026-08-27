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

uv run --extra gui python app.py
