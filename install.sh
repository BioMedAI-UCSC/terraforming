#!/usr/bin/env bash
# install.sh — install the tform CLI
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/BioMedAI-UCSC/terraforming/main/install.sh | bash
#   bash install.sh [--prefix ~/.local]   # custom install prefix
#
# What it does:
#   1. Checks for Python >=3.12 and pip (or uv).
#   2. Creates an isolated virtual-env at $PREFIX/lib/tform-venv.
#   3. Installs the terraforming package + cli from PyPI (or the GitHub
#      release tarball if a VERSION is set in the environment).
#   4. Drops a thin launcher script at $PREFIX/bin/tform.
#
# Override the install prefix:
#   PREFIX=/usr/local bash install.sh
#
# Install a specific release:
#   VERSION=0.2.0 bash install.sh

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────

REPO="BioMedAI-UCSC/terraforming"
PREFIX="${PREFIX:-$HOME/.local}"
VERSION="${VERSION:-}"          # empty = latest release from PyPI
VENV_DIR="$PREFIX/lib/tform-venv"
BIN_DIR="$PREFIX/bin"
LAUNCHER="$BIN_DIR/tform"

RED='\033[0;31m'; YELLOW='\033[1;33m'; GREEN='\033[0;32m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

# ── Helpers ───────────────────────────────────────────────────────────────────

info()    { echo -e "${CYAN}  →${RESET}  $*"; }
success() { echo -e "${GREEN}  ✓${RESET}  $*"; }
warn()    { echo -e "${YELLOW}  ⚠${RESET}  $*"; }
die()     { echo -e "${RED}  ✖  $*${RESET}" >&2; exit 1; }

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --prefix) PREFIX="$2"; shift 2 ;;
            --prefix=*) PREFIX="${1#--prefix=}"; shift ;;
            --version) VERSION="$2"; shift 2 ;;
            --version=*) VERSION="${1#--version=}"; shift ;;
            -h|--help)
                echo "Usage: bash install.sh [--prefix PATH] [--version X.Y.Z]"
                exit 0 ;;
            *) die "Unknown argument: $1" ;;
        esac
    done
    VENV_DIR="$PREFIX/lib/tform-venv"
    BIN_DIR="$PREFIX/bin"
    LAUNCHER="$BIN_DIR/tform"
}

# ── Python check ──────────────────────────────────────────────────────────────

find_python() {
    for cmd in python3.13 python3.12 python3; do
        if command -v "$cmd" &>/dev/null; then
            local ver
            ver=$("$cmd" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
            local major minor
            major="${ver%%.*}"; minor="${ver##*.}"
            if [[ "$major" -ge 3 && "$minor" -ge 12 ]]; then
                echo "$cmd"; return 0
            fi
        fi
    done
    return 1
}

# ── Install target ─────────────────────────────────────────────────────────────

# Returns the pip install specifier: either a GitHub tarball URL or a PyPI spec.
install_specifier() {
    if [[ -n "$VERSION" ]]; then
        # GitHub release tarball (monorepo: cli sub-directory is the installable)
        echo "https://github.com/$REPO/releases/download/v${VERSION}/terraforming_cli-${VERSION}.tar.gz"
    else
        echo "terraforming-cli"   # PyPI (latest)
    fi
}

# ── Main ──────────────────────────────────────────────────────────────────────

main() {
    parse_args "$@"

    echo
    echo -e "${BOLD}  ★  tform installer${RESET}"
    echo -e "  Mars climate simulation CLI"
    echo -e "  https://github.com/$REPO"
    echo

    # 1. Python
    PYTHON=$(find_python) || die "Python >=3.12 not found. Install it from https://python.org"
    info "Using $PYTHON ($("$PYTHON" --version))"

    # 2. Create venv
    if [[ -d "$VENV_DIR" ]]; then
        warn "Existing installation found at $VENV_DIR — upgrading."
    fi
    info "Creating virtual environment at $VENV_DIR"
    "$PYTHON" -m venv "$VENV_DIR"

    VENV_PIP="$VENV_DIR/bin/pip"
    VENV_PYTHON="$VENV_DIR/bin/python"

    # 3. Install
    SPEC=$(install_specifier)
    info "Installing: $SPEC"
    "$VENV_PIP" install --quiet --upgrade pip
    "$VENV_PIP" install --quiet "$SPEC"

    # 4. Verify
    "$VENV_DIR/bin/tform" --version &>/dev/null \
        || die "Installation succeeded but 'tform --version' failed."

    # 5. Launcher wrapper (ensures the venv is used regardless of active env)
    mkdir -p "$BIN_DIR"
    cat > "$LAUNCHER" <<EOF
#!/usr/bin/env bash
exec "$VENV_DIR/bin/tform" "\$@"
EOF
    chmod +x "$LAUNCHER"
    success "Launcher written to $LAUNCHER"

    # 6. PATH hint
    echo
    if [[ ":$PATH:" != *":$BIN_DIR:"* ]]; then
        warn "$BIN_DIR is not in your PATH."
        echo -e "  Add this to your shell profile (~/.bashrc / ~/.zshrc):"
        echo -e "    ${CYAN}export PATH=\"\$HOME/.local/bin:\$PATH\"${RESET}"
    else
        success "tform is ready — run: ${BOLD}tform --help${RESET}"
    fi
    echo
}

main "$@"
