#!/bin/sh
# Runs your program: ./your_program.sh <command> [options]. Set PYTHON to choose the interpreter.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}" exec "${PYTHON:-python3}" -m app.main "$@"
