// Scope: Deterministic DAG construction (structure only)
//
// DagBuilder is responsible for converting higher-level Specs into
// HashMap<NodeId, DagNode<T>>.
//
// This module performs NO execution.
// This module performs NO scheduling.
// This module performs NO policy or judgment.
//
// It exists to centralize graph construction semantics inside the kernel.

use std::collections::HashMap;

use crate::kernel::dag::{DagNode, NodeId};
use crate::kernel::pipeline::Stage;
use std::sync::Arc;

// Symbolic node specification
pub struct NodeSpec<T> {
    pub name: &'static str,
    pub stage: Arc<dyn Stage<T = T>>,
    pub inputs: Vec<&'static str>,
}

pub struct DagSpec<T> {
    pub nodes: Vec<NodeSpec<T>>,
}

pub struct DagBuilder;

impl DagBuilder {
    pub fn build<T>(spec: &DagSpec<T>) -> HashMap<NodeId, DagNode<T>> {
        let mut by_name: HashMap<&'static str, DagNode<T>> = HashMap::new();

        // pass 1: create nodes without inputs
        for n in &spec.nodes {
            // Kernel invariant: nodes may have at most one input
            if n.inputs.len() > 1 {
                panic!(
                    "DagBuilder invariant violated: node '{}' has {} inputs (max 1 allowed)",
                    n.name,
                    n.inputs.len()
                );
            }

            by_name.insert(
                n.name,
                DagNode {
                    name: n.name,
                    stage: n.stage.clone(),
                    inputs: Vec::new(),
                },
            );
        }

        // pass 2: resolve input names -> NodeId
        for n in &spec.nodes {
            let mut ids = Vec::new();
            for inp in &n.inputs {
                let id = by_name[inp].id();
                ids.push(id);
            }
            by_name.get_mut(n.name).unwrap().inputs = ids;
        }

        // finalize map keyed by NodeId
        by_name
            .into_values()
            .map(|n| (n.id(), n))
            .collect()
    }
}
