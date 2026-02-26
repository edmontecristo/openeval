#!/bin/bash
# Record OpenEval terminal demo GIF
# Requirements: terminalizer (npm install -g terminalizer) or asciinema

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  OpenEval Terminal Demo Recorder                               ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Install terminalizer: npm install -g terminalizer"
echo "Then run: terminalizer record openeval-demo"
echo ""

# Option 1: Using terminalizer
if command -v terminalizer &> /dev/null; then
    echo "Recording with terminalizer..."
    terminalizer record docs/demo --config docs/terminalizer-config.yml
    terminalizer render docs/demo -o docs/demo.gif
    echo "✅ Saved to docs/demo.gif"
# Option 2: Using asciinema
elif command -v asciinema &> /dev/null; then
    echo "Recording with asciinema..."
    asciinema rec docs/demo.cast --command "PYTHONPATH=. python3 -m openeval run examples/quickstart_ollama.py"
    echo "✅ Saved to docs/demo.cast"
    echo "Convert to GIF: https://asciinema.org/en/latest/cli/usage.html#gif"
else
    echo "❌ Neither terminalizer nor asciinema found."
    echo ""
    echo "Install one of:"
    echo "  npm install -g terminalizer"
    echo "  brew install asciinema"
fi
