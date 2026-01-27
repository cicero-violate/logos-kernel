#!/usr/bin/env bash
# Scope: Cargo-shaped build using raw rustc

set -e

ROOT="$(pwd)"
TARGET="$ROOT/target/debug"

DEPS="$TARGET/deps"
INCR="$TARGET/incremental"
MIR="$TARGET/mir"
LLVM="$TARGET/llvm-ir"
ASM="$TARGET/asm"
OBJ="$TARGET/obj"

mkdir -p "$TARGET" "$DEPS" "$INCR" "$MIR" "$LLVM" "$ASM" "$OBJ"

echo "[clean]"
rm -f "$TARGET/libmmsb.rlib"
rm -f "$TARGET/libmmsb.rmeta"
rm -f "$TARGET/mmsb_host"

echo "[build] mmsb (lib)"

rustc \
  -Z unstable-options \
  -C incremental="$INCR" \
  --crate-type=lib \
  --crate-name mmsb \
  src/lib.rs \
  --emit=link,metadata,dep-info,mir,llvm-ir,asm,obj \
  --out-dir "$TARGET" \
  -C save-temps \
  -C debuginfo=2

mv "$TARGET"/*.mir "$MIR" 2>/dev/null || true
mv "$TARGET"/*.ll  "$LLVM" 2>/dev/null || true
mv "$TARGET"/*.s   "$ASM" 2>/dev/null || true
mv "$TARGET"/*.o   "$OBJ" 2>/dev/null || true
mv "$TARGET"/*.rmeta "$TARGET/libmmsb.rmeta" 2>/dev/null || true

echo "[build] mmsb_tests (harness)"

rustc \
  -Z unstable-options \
  -C incremental="$INCR" \
  --test \
  src/lib.rs \
  --emit=link,metadata,dep-info,mir,llvm-ir,asm,obj \
  -o "$TARGET/mmsb_tests" \
  -C save-temps \
  -C debuginfo=2

mv "$TARGET"/*.mir "$MIR" 2>/dev/null || true
mv "$TARGET"/*.ll  "$LLVM" 2>/dev/null || true
mv "$TARGET"/*.s   "$ASM" 2>/dev/null || true
mv "$TARGET"/*.o   "$OBJ" 2>/dev/null || true

echo "[build] mmsb_host (bin)"

rustc \
  -Z unstable-options \
  -C incremental="$INCR" \
  src/driver/main.rs \
  --extern mmsb="$TARGET/libmmsb.rlib" \
  --emit=link,metadata,dep-info,mir,llvm-ir,asm,obj \
  -o "$TARGET/mmsb_host" \
  -C save-temps \
  -C debuginfo=2

mv "$TARGET"/*.mir "$MIR" 2>/dev/null || true
mv "$TARGET"/*.ll  "$LLVM" 2>/dev/null || true
mv "$TARGET"/*.s   "$ASM" 2>/dev/null || true
mv "$TARGET"/*.o   "$OBJ" 2>/dev/null || true

echo "[run] tests"
"$TARGET/mmsb_tests"

echo "[done]"

# -------------------------
# Build invocation logging
# -------------------------
mkdir -p logs
echo "$(date -Is) | build.sh invoked" >> logs/build.log
