// Scope: Canonical DAG Graph Contract (documentation-only)
//
// This file defines the *structural constitution* of graphs accepted by the
// kernel DAG executor. Any graph reaching execution is assumed to satisfy
// ALL invariants listed below.
//
// These are enforced mechanically inside dag.rs.
//
// ---------------------------------------------------------------------------
// GRAPH SHAPE
// ---------------------------------------------------------------------------
//
// 1. Graph is non-empty.
//
// 2. Exactly one root node exists.
//    - Root = node with zero inputs.
//
// 3. Root node has zero inputs.
//
// 4. Every non-root node has exactly one input.
//
// 5. All input NodeIds must refer to existing nodes.
//
// 6. No node may list itself as an input.
//
// 7. Graph contains no unreachable nodes.
//    - Every node is reachable from the root.
//
// 8. Declared output node exists in graph.
//
// 9. Declared output node is reachable from root.
//
// 10. Topological sort contains exactly N nodes.
//
// ---------------------------------------------------------------------------
// EXECUTION MODEL CONSEQUENCES
// ---------------------------------------------------------------------------
//
// - Single external input enters at root.
// - Data flows along a single-parent chain per node.
// - Fan-out is allowed.
// - Fan-in is forbidden.
//
// This produces a tree-shaped forward-flow graph with a single entry point.
//
// ---------------------------------------------------------------------------
// NON-GOALS (NOT KERNEL RESPONSIBILITY)
// ---------------------------------------------------------------------------
//
// - Resource limits
// - Node count limits
// - Depth limits
// - Scheduling policies
// - Stage authorization
// - Security policy
// - Execution admission and budgeting (Policy layer)
//
// These belong to Policy / Judgment layers.
//
// ---------------------------------------------------------------------------
// STABILITY GUARANTEE
// ---------------------------------------------------------------------------
//
// Any change to these invariants MUST be:
//   1) Intentional
//   2) Reviewed
//   3) Reflected in dag.rs enforcement
//
// This file is the human-readable mirror of kernel truth.
//
