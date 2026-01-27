# Policy Contract (Invariant)
#
# Status: FROZEN
#
# This document defines the canonical contract for the Policy layer.
# The Policy layer governs admission and constraint of execution.
# It does NOT execute computation.
# It does NOT inspect runtime values.
# It does NOT perform scheduling.
#
# Any change to this document MUST be:
#   1) Intentional
#   2) Reviewed
#   3) Versioned
#
# ---------------------------------------------------------------------------
# PLACEMENT IN CANONICAL CHAIN
# ---------------------------------------------------------------------------
#
# Intent
#   ↓
# Policy
#   ↓
# Judge
#   ↓
# Kernel (DagBuilder → DAG invariants → Execution)
#
# Policy runs exactly once per graph, before Judge.
#
# ---------------------------------------------------------------------------
# PURPOSE
# ---------------------------------------------------------------------------
#
# Policy is a pure, deterministic gate that decides:
#   - Whether a requested graph may execute
#   - What constraints apply to that execution
#
# Policy does NOT decide how computation is performed.
# That responsibility belongs to Judge.
#
# ---------------------------------------------------------------------------
# INPUTS (WHAT POLICY MAY SEE)
# ---------------------------------------------------------------------------
#
# Policy may ONLY observe declarative metadata.
#
# 1) IntentMeta
# 2) GraphMeta
# 3) TypeSig (requested capabilities)
#
# Policy must NOT observe:
#   - Input values
#   - Intermediate values
#   - Deltas
#   - Proofs
#   - Timing
#   - Runtime measurements
#   - Filesystem
#   - Network
#
# ---------------------------------------------------------------------------
# INTENT META
# ---------------------------------------------------------------------------
#
# IntentMeta is a small, declarative description of why execution is requested.
#
# Canonical fields:
#
#   intent_kind        : enum
#       - UserRequest
#       - BatchJob
#       - Replay
#       - Test
#
#   declared_priority  : u8
#
#   caller_identity    : Option<Hash>
#
# No additional fields are permitted without amending this document.
#
# ---------------------------------------------------------------------------
# GRAPH META
# ---------------------------------------------------------------------------
#
# GraphMeta is derived from the DAG structure.
#
# Canonical fields:
#
#   node_count            : u64
#   graph_depth           : u64
#   structural_graph_hash : u64
#   stage_type_names      : Vec<&'static str>
#   root_stage_name       : &'static str
#
# GraphMeta must be computable without executing stages.
#
# ---------------------------------------------------------------------------
# TYPESIG
# ---------------------------------------------------------------------------
#
# TypeSig expresses requested compute capabilities.
#
# Current fields:
#
#   allow_logic        : bool
#   allow_differential: bool
#
# Policy may only REDUCE these permissions, never expand them.
#
# ---------------------------------------------------------------------------
# POLICY DECISION
# ---------------------------------------------------------------------------
#
# Policy returns a PolicyDecision.
#
# Canonical shape:
#
#   PolicyDecision {
#       admit        : bool
#       max_nodes    : Option<u64>
#       max_depth    : Option<u64>
#       allowed_modes: ModeMask
#   }
#
#   ModeMask {
#       allow_logic        : bool
#       allow_differential: bool
#   }
#
# ---------------------------------------------------------------------------
# SEMANTICS
# ---------------------------------------------------------------------------
#
# If admit == false:
#   Execution is rejected before entering the kernel.
#
# If admit == true:
#   - GraphMeta must satisfy max_nodes if present
#   - GraphMeta must satisfy max_depth if present
#   - TypeSig is clamped by allowed_modes
#
# After clamping, Judge is invoked.
#
# ---------------------------------------------------------------------------
# DETERMINISM REQUIREMENT
# ---------------------------------------------------------------------------
#
# Policy must satisfy:
#
#   Same (IntentMeta, GraphMeta, TypeSig)
#       → Same PolicyDecision
#
# Policy must not consult external state.
#
# ---------------------------------------------------------------------------
# NON-GOALS (NOT POLICY RESPONSIBILITY)
# ---------------------------------------------------------------------------
#
# - Scheduling
# - Parallelism
# - Caching
# - Memoization
# - Execution ordering
# - Stage semantics
# - Proof verification
#
# ---------------------------------------------------------------------------
# STABILITY GUARANTEE
# ---------------------------------------------------------------------------
#
# Kernel code must not depend on Policy existence.
# Policy operates entirely outside the kernel.
#
# This document is the sole authority on Policy behavior.
#
