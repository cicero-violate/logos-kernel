#!/usr/bin/env bash
# Canonical iteration measurement script
# Measures:
#   BUILD_MS  - build wall time
#   TEST_MS   - test wall time
#   ITER_MS   - build + test + first observable output
#
# Assumes:
#   - project root
#   - ./build.sh exists
#   - target/debug/mmsb_host produced by build.sh

set -e

now_ms() {
  date +%s%3N
}

START_ITER=$(now_ms)

# -------------------------
# Build
# -------------------------
START_BUILD=$(now_ms)
./build.sh >/dev/null 2>&1
END_BUILD=$(now_ms)

# -------------------------
# Test
# -------------------------
START_TEST=$(now_ms)
# tests already executed inside build.sh
END_TEST=$(now_ms)

# -------------------------
# First observable output
# (simple: run binary help or main)
# -------------------------
if [ -x target/debug/mmsb_host ]; then
  target/debug/mmsb_host >/dev/null 2>&1 || true
fi

END_ITER=$(now_ms)

BUILD_MS=$((END_BUILD - START_BUILD))
TEST_MS=$((END_TEST - START_TEST))
ITER_MS=$((END_ITER - START_ITER))

echo "BUILD_MS=${BUILD_MS}"
echo "TEST_MS=${TEST_MS}"
echo "ITER_MS=${ITER_MS}"

# -------------------------
# Persistent history log
# -------------------------
mkdir -p logs
LOG_FILE="logs/iteration_history.log"
TS="$(date -Is)"

{
  echo "${TS} BUILD_MS=${BUILD_MS} TEST_MS=${TEST_MS} ITER_MS=${ITER_MS}"
} >> "${LOG_FILE}"

echo "APPENDED_TO=${LOG_FILE}"
