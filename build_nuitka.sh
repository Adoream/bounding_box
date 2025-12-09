#!/usr/bin/env bash
set -euo pipefail

ENTRY_FILE="${1:-app.py}"
APP_NAME="${2:-ct-annotator}"
BUILD_DIR="${BUILD_DIR:-build}"
EXTRA_OPTS="${NUITKA_EXTRA_OPTS:-}"

NUITKA_PLATFORM_OPTS=""
UNAME_S="$(uname -s || echo "")"
case "${UNAME_S}" in
    MINGW*|MSYS*|CYGWIN*|Windows_NT)
        NUITKA_PLATFORM_OPTS="--windows-console-mode=disable"
        ;;
    *)
        NUITKA_PLATFORM_OPTS=""
        ;;
esac

echo "==> Entry: ${ENTRY_FILE}"
echo "==> App name: ${APP_NAME}"
echo "==> Build dir: ${BUILD_DIR}"

python -m nuitka \
  "${ENTRY_FILE}" \
  --standalone \
  --onefile \
  --output-dir="${BUILD_DIR}" \
  --output-filename="${APP_NAME}" \
  --enable-plugin=pyside6 \
  --include-qt-plugins=sensible,styles \
  --follow-imports \
  --include-package=pydicom,nibabel\
  ${NUITKA_PLATFORM_OPTS} \
  ${EXTRA_OPTS}

echo "==> Done. Output in ${BUILD_DIR}"
