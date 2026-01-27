# Cargo-Based Equivalent (Reference Only)
#
# This document describes the Cargo-based equivalent of the raw-rustc
# build pipeline used in this repository.
#
# It exists strictly for comparison and understanding.
# The canonical build path remains: build.sh
#
# --------------------------------------------------------------------
# Minimal Cargo.toml
# --------------------------------------------------------------------
#
# [package]
# name = "mmsb"
# version = "0.1.0"
# edition = "2021"
#
# [lib]
# name = "mmsb"
# path = "src/lib.rs"
#
# [[bin]]
# name = "mmsb_host"
# path = "src/driver/main.rs"
#
# --------------------------------------------------------------------
# Equivalent Commands
# --------------------------------------------------------------------
#
# Build library:
#   cargo build
#
# Build tests:
#   cargo test --no-run
#
# Run tests:
#   cargo test
#
# Run host:
#   cargo run
#
# --------------------------------------------------------------------
# Mapping to build.sh
# --------------------------------------------------------------------
#
# cargo build
#   ≈ rustc --crate-type=lib src/lib.rs ...
#
# cargo test
#   ≈ rustc --test src/lib.rs ...
#
# cargo run
#   ≈ rustc src/driver/main.rs --extern mmsb=...
#
# --------------------------------------------------------------------
# Rationale for NOT using Cargo as primary
# --------------------------------------------------------------------
#
# - Explicit artifact paths
# - Full control over rustc flags
# - Direct access to MIR / LLVM / ASM / OBJ
# - Deterministic build surface
#
# Cargo remains a convenience wrapper, not a semantic authority.

