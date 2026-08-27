#!/bin/bash
# Move to the script's directory
cd "$(dirname "$0")"

echo "======================================================="
echo "  Privacy-First Local Anonymizer - macOS Setup"
echo "======================================================="
echo ""

# Make start script executable
chmod +x start_mac.command

# Check uv
if ! command -v uv &> /dev/null; then
    echo "[INFO] 'uv' nicht gefunden. Installiere uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "1. Installiere Python-Abhängigkeiten via uv (inkl. Cocoa Backend für Mac)..."
uv sync --extra gui --extra dev

echo ""
echo "2. Erstelle Desktop-Verknüpfung..."
DESKTOP_DIR="$HOME/Desktop"
if [ -d "$DESKTOP_DIR" ]; then
    ln -sf "$(pwd)/start_mac.command" "$DESKTOP_DIR/Local Anonymizer.command"
    echo "[ERFOLGREICH] Verknüpfung 'Local Anonymizer.command' wurde auf Ihrem Desktop erstellt!"
fi

echo ""
echo "Installation abgeschlossen. Sie können die App durch Doppelklick auf 'start_mac.command' starten."
read -p "Drücken Sie Enter zum Beenden..."
