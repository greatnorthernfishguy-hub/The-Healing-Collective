#!/usr/bin/env bash
# ============================================================================
# The Healing Collective — One-Click Installer
#
# Installs The Healing Collective as an E-T Systems module with:
#   - Python dependency installation
#   - NG-Lite peer bridge setup (shared learning directory)
#   - ET Module Manager registration
#
# Usage:
#   ./install.sh              # Full installation
#   ./install.sh --deps-only  # Install dependencies only
#   ./install.sh --uninstall  # Remove (preserves learning data)
#   ./install.sh --status     # Check installation status
#
# Environment variable overrides:
#   HC_INSTALL_DIR   — Installation path (default: ~/The-Healing-Collective-)
#
# Follows the same patterns as TrollGuard's install.sh for consistency
# across the E-T Systems module ecosystem.
#
# Changelog:
# [2026-02-26] Claude (Opus 4.6) — Initial creation.
#   Modeled after TrollGuard's install.sh.
#   Healing Collective has no API server (no service_name/port).
# ============================================================================

set -euo pipefail

# --- Configuration (overridable via environment) ---
INSTALL_DIR="${HC_INSTALL_DIR:-$HOME/The-Healing-Collective-}"
SERVICE_NAME="healing_collective"
ET_MODULES_DIR="${ET_MODULES_DIR:-$HOME/.et_modules}"
SHARED_LEARNING_DIR="$ET_MODULES_DIR/shared_learning"
MODULE_DIR="$ET_MODULES_DIR/healing_collective"

# --- Colors ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }

# --- Detect environment ---
detect_environment() {
    info "Detecting environment..."

    # Python version
    if command -v python3 &>/dev/null; then
        PYTHON="python3"
        PY_VERSION=$($PYTHON --version 2>&1 | awk '{print $2}')
        info "Python: $PY_VERSION"
    else
        error "Python 3 not found. The Healing Collective requires Python 3.10+"
        exit 1
    fi

    # Check Python 3.10+
    PY_MAJOR=$($PYTHON -c "import sys; print(sys.version_info.major)")
    PY_MINOR=$($PYTHON -c "import sys; print(sys.version_info.minor)")
    if [ "$PY_MAJOR" -lt 3 ] || ([ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]); then
        error "Python 3.10+ required (found $PY_VERSION)"
        exit 1
    fi

    # Check for existing peer modules
    if [ -d "$HOME/TrollGuard" ] || [ -d "$ET_MODULES_DIR/modules/trollguard" ]; then
        info "TrollGuard detected — Tier 2 peer healing available"
        HAS_TROLLGUARD=true
    else
        HAS_TROLLGUARD=false
    fi

    if [ -d "$HOME/NeuroGraph" ] || [ -d "$ET_MODULES_DIR/modules/neurograph" ]; then
        info "NeuroGraph detected — Tier 3 predictive healing available"
        HAS_NEUROGRAPH=true
    else
        HAS_NEUROGRAPH=false
    fi

    if [ -d "$HOME/The-Inference-Difference" ]; then
        info "The-Inference-Difference detected"
        HAS_TID=true
    else
        HAS_TID=false
    fi
}

# --- Install dependencies ---
install_deps() {
    info "Installing Python dependencies..."

    $PYTHON -m pip install --upgrade pip 2>/dev/null || true

    # Core dependencies
    $PYTHON -m pip install numpy pyyaml msgpack 2>/dev/null

    # Sentence transformers (largest dependency)
    info "Installing sentence-transformers (this may take a minute)..."
    $PYTHON -m pip install sentence-transformers 2>/dev/null

    info "Dependencies installed."
}

# --- Deploy files ---
deploy_files() {
    info "Deploying The Healing Collective to $INSTALL_DIR..."

    mkdir -p "$INSTALL_DIR"

    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

    # Copy core module files
    cp "$SCRIPT_DIR/healing_collective_hook.py" "$INSTALL_DIR/"
    cp "$SCRIPT_DIR/ng_lite.py" "$INSTALL_DIR/"
    cp "$SCRIPT_DIR/ng_peer_bridge.py" "$INSTALL_DIR/"
    cp "$SCRIPT_DIR/ng_ecosystem.py" "$INSTALL_DIR/"
    cp "$SCRIPT_DIR/openclaw_adapter.py" "$INSTALL_DIR/"
    cp "$SCRIPT_DIR/et_module.json" "$INSTALL_DIR/"
    cp "$SCRIPT_DIR/requirements.txt" "$INSTALL_DIR/"

    # Copy core package
    mkdir -p "$INSTALL_DIR/core"
    cp "$SCRIPT_DIR/core/"*.py "$INSTALL_DIR/core/"

    # Update manifest with actual install path
    $PYTHON -c "
import json
with open('$INSTALL_DIR/et_module.json', 'r') as f:
    m = json.load(f)
m['install_path'] = '$INSTALL_DIR'
with open('$INSTALL_DIR/et_module.json', 'w') as f:
    json.dump(m, f, indent=2)
"

    info "Files deployed to $INSTALL_DIR"
}

# --- Setup shared learning (NGPeerBridge) ---
setup_shared_learning() {
    info "Setting up shared learning directory..."

    mkdir -p "$SHARED_LEARNING_DIR"
    mkdir -p "$MODULE_DIR"
    mkdir -p "$MODULE_DIR/checkpoints"
    mkdir -p "$ET_MODULES_DIR"

    info "Shared learning directory: $SHARED_LEARNING_DIR"
    info "Module data directory: $MODULE_DIR"

    # Register with ET Module Manager
    info "Registering with ET Module Manager..."

    $PYTHON -c "
import json, os, time
registry_path = '$ET_MODULES_DIR/registry.json'

try:
    with open(registry_path, 'r') as f:
        registry = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    registry = {'modules': {}}

registry['modules']['healing_collective'] = {
    'module_id': 'healing_collective',
    'display_name': 'The Healing Collective',
    'version': '0.4.0',
    'description': 'Adaptive self-healing intelligence for the E-T Systems ecosystem',
    'install_path': '$INSTALL_DIR',
    'git_remote': 'https://github.com/greatnorthernfishguy-hub/The-Healing-Collective-.git',
    'git_branch': 'main',
    'entry_point': 'healing_collective_hook.py',
    'ng_lite_version': '1.0.0',
    'dependencies': [],
    'service_name': 'healing_collective',
    'api_port': 0,
    'registered_at': time.time(),
}
registry['last_updated'] = time.time()

with open(registry_path, 'w') as f:
    json.dump(registry, f, indent=2)

print('Registered The Healing Collective in ET Module Manager')
" 2>/dev/null || warn "ET Module Manager registration failed (non-critical)"
}

# --- Status check ---
check_status() {
    info "The Healing Collective Status"
    echo "========================"

    if [ -d "$INSTALL_DIR" ]; then
        echo -e "Installed: ${GREEN}Yes${NC} ($INSTALL_DIR)"
    else
        echo -e "Installed: ${RED}No${NC}"
    fi

    if [ -d "$SHARED_LEARNING_DIR" ]; then
        PEER_FILES=$(ls "$SHARED_LEARNING_DIR"/*.jsonl 2>/dev/null | wc -l)
        echo -e "Peer Bridge: ${GREEN}Active${NC} ($PEER_FILES module event files)"
    else
        echo -e "Peer Bridge: ${YELLOW}Not configured${NC}"
    fi

    if [ -f "$MODULE_DIR/dvs.msgpack" ]; then
        echo -e "DVS: ${GREEN}Active${NC}"
    else
        echo -e "DVS: ${YELLOW}Empty (fresh install)${NC}"
    fi

    if [ -f "$ET_MODULES_DIR/registry.json" ]; then
        MODULE_COUNT=$($PYTHON -c "
import json
with open('$ET_MODULES_DIR/registry.json') as f:
    r = json.load(f)
print(len(r.get('modules', {})))
" 2>/dev/null || echo "?")
        echo -e "ET Modules:  ${GREEN}$MODULE_COUNT registered${NC}"
    fi
}

# --- Uninstall ---
uninstall() {
    warn "Uninstalling The Healing Collective..."

    if [ -d "$INSTALL_DIR" ]; then
        warn "Preserving learning data in $MODULE_DIR and $SHARED_LEARNING_DIR"
        rm -rf "$INSTALL_DIR"
        info "Removed $INSTALL_DIR"
    fi

    info "The Healing Collective uninstalled. Learning data preserved."
}

# --- Main ---
main() {
    echo "============================================"
    echo "  The Healing Collective Installer v0.4.0"
    echo "  Adaptive Self-Healing Intelligence"
    echo "============================================"
    echo ""

    case "${1:-}" in
        --deps-only)
            detect_environment
            install_deps
            ;;
        --uninstall)
            uninstall
            ;;
        --status)
            check_status
            ;;
        *)
            detect_environment
            install_deps
            deploy_files
            setup_shared_learning
            info ""
            info "============================================"
            info "  The Healing Collective installed!"
            info "============================================"
            info ""
            info "  Install dir:  $INSTALL_DIR"
            info "  Module data:  $MODULE_DIR"
            info ""
            info "  Peer modules detected:"
            [ "$HAS_TROLLGUARD" = true ] && info "    - TrollGuard"
            [ "$HAS_NEUROGRAPH" = true ] && info "    - NeuroGraph"
            [ "$HAS_TID" = true ]        && info "    - The-Inference-Difference"
            info ""
            ;;
    esac
}

main "$@"
