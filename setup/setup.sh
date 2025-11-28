#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/setup.sh          # install uv (if needed) and sync deps
#   ./scripts/setup.sh --dev    # also install dev extras

PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJ_ROOT"

INSTALL_DEV=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --dev) INSTALL_DEV=true; shift ;;
    --help|-h) echo "Usage: $0 [--dev]"; exit 0 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

# Install uv if not present
if ! command -v uv &>/dev/null; then
  echo "uv not found. Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # Add uv to PATH for this session
  export PATH="$HOME/.cargo/bin:$PATH"
fi

echo "uv version: $(uv --version)"

# Initialize project if not already (creates uv.lock if missing)
if [[ ! -f "uv.lock" ]]; then
  echo "Initializing uv project..."
  uv init --no-readme || true
fi

# Sync dependencies
echo "Syncing dependencies..."
if [[ "$INSTALL_DEV" == "true" ]]; then
  uv sync --dev
else
  uv sync
fi

echo ""
echo "Setup complete!"
echo "To activate the environment:  source .venv/bin/activate"
echo "Or run commands via:          uv run <command>"