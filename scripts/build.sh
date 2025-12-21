#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   scripts/build.sh [entry_file] [app_name]
#
# Examples:
#   scripts/build.sh app.py ct-annotator
#   BUILD_DIR=build NUITKA_EXTRA_OPTS="--assume-yes-for-downloads" scripts/build.sh app.py ct-annotator

ENTRY_FILE="${1:-app.py}"
APP_NAME="${2:-ct-annotator}"
BUILD_DIR="${BUILD_DIR:-build}"

# Additional Nuitka CLI opts may be provided as a space-delimited string, e.g.
# NUITKA_EXTRA_OPTS="--nofollow-import-to=tkinter --warn-implicit-exceptions"
EXTRA_OPTS="${NUITKA_EXTRA_OPTS:-}"

echo "==> Entry: ${ENTRY_FILE}"
echo "==> App name: ${APP_NAME}"
echo "==> Build dir: ${BUILD_DIR}"

# Platform-specific options
NUITKA_PLATFORM_OPTS=()
UNAME_S="$(uname -s 2>/dev/null || echo "")"
case "${UNAME_S}" in
  MINGW*|MSYS*|CYGWIN*|Windows_NT)
    # Avoid opening a console window for GUI apps on Windows
    NUITKA_PLATFORM_OPTS+=(--windows-console-mode=disable)
    ;;
  *)
    ;;
esac

# Ensure dependencies are importable in the current Python environment
python - <<'PY'
import sys
for mod in ("nuitka", "pydicom", "nibabel"):
    __import__(mod)
print("==> Python:", sys.version)
print("==> Imports OK: nuitka, pydicom, nibabel")
PY

# Resolve pydicom data dir inside this environment and verify urls.json exists
PYDICOM_DATA_DIR="$(
  python -c "import pydicom, pathlib; print(pathlib.Path(pydicom.__file__).parent / 'data')"
)"

python - <<'PY'
import pathlib, pydicom
p = pathlib.Path(pydicom.__file__).parent / "data" / "urls.json"
if not p.exists():
    raise SystemExit(
        f"[FATAL] Missing pydicom data file: {p}\n"
        "Fix: reinstall pydicom in this environment:\n"
        "  pip install -U --force-reinstall pydicom\n"
    )
print("==> pydicom data OK:", p)
PY

echo "==> pydicom data dir: ${PYDICOM_DATA_DIR}"

# Build arguments as an array (robust quoting)
NUITKA_ARGS=(
  "${ENTRY_FILE}"
  --standalone
  --onefile
  "--output-dir=${BUILD_DIR}"
  "--output-filename=${APP_NAME}"

  # Qt / PySide6 support
  --enable-plugin=pyside6
  --include-qt-plugins=sensible,styles

  # Reasonable default: follow imports from entry graph
  --follow-imports

  # Ensure these packages are included
  --include-package=pydicom
  --include-package=nibabel

  # pydicom 3.x dynamic pixels plugins (prevent ModuleNotFoundError)
  --include-package=pydicom.pixels.decoders
  --include-package=pydicom.pixels.encoders
  --include-module=pydicom.pixels.decoders.gdcm

  # pydicom package data (fix urls.json missing in onefile)
  "--include-data-dir=${PYDICOM_DATA_DIR}=pydicom/data"
)

# Append platform opts
NUITKA_ARGS+=("${NUITKA_PLATFORM_OPTS[@]}")

# Append extra opts (space-delimited)
# shellcheck disable=SC2206
if [[ -n "${EXTRA_OPTS}" ]]; then
  EXTRA_ARR=(${EXTRA_OPTS})
  NUITKA_ARGS+=("${EXTRA_ARR[@]}")
fi

echo "==> Running Nuitka..."
python -m nuitka "${NUITKA_ARGS[@]}"

echo "==> Done. Output in ${BUILD_DIR}"
