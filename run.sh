#!/usr/bin/env bash
# Scope: Run host binary

set -e

BIN="./target/debug/mmsb_host"

if [ ! -f "$BIN" ]; then
  echo "[run] binary not found: $BIN"
  echo "[run] did you run ./build.sh ?"
  exit 1
fi

exec "$BIN" "$@"

# Usage
# ./run.sh
# ./run.sh server
# ./run.sh arg1 arg2
