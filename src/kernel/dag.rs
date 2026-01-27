use std::collections::{HashMap, HashSet};

use crate::kernel::pipeline::Stage;
use crate::kernel::types::{TypeSig, NodeResult};
use crate::kernel::error::PipelineError;
use crate::kernel::judge::DefaultJudge;
use crate::kernel::types::ComputeMode;
use crate::kernel::replay;
use crate::kernel::codec::Encode;
use crate::kernel::hash::Fnv64;
use std::any::type_name_of_val;

pub(crate) fn structural_graph_hash<T>(nodes: &HashMap<NodeId, DagNode<T>>) -> u64 {
    let mut ids: Vec<NodeId> = nodes.keys().cloned().collect();
    ids.sort();

    let mut h = Fnv64::new();

    for id in ids {
        h.write_u64(id);
        let node = &nodes[&id];
        h.write_str(node.name);
        // include stage identity (type name) for structural determinism
        let stage_name = type_name_of_val(&*node.stage);
        h.write_str(stage_name);
        // Deterministic input ordering
        let mut inputs = node.inputs.clone();
        inputs.sort();
        for inp in &inputs {
            h.write_u64(*inp);
        }
    }

    h.finish()
}

pub type NodeId = u64;

fn hash_node(name: &str, stage_name: &str, inputs: &[NodeId]) -> NodeId {
    let mut h = Fnv64::new();
    h.write_str(name);
    h.write_str(stage_name);
    // Deterministic input ordering
    let mut ins: Vec<NodeId> = inputs.to_vec();
    ins.sort();
    for i in ins {
        h.write_u64(i);
    }
    h.finish()
}

use std::sync::Arc;

 #[derive(Clone)]
 pub struct DagNode<T> {
    pub name: &'static str,
    pub stage: Arc<dyn Stage<T = T>>,
    pub inputs: Vec<NodeId>,
}

impl<T> DagNode<T> {
    pub fn id(&self) -> NodeId {
        let stage_name = type_name_of_val(&*self.stage);
        hash_node(self.name, stage_name, &self.inputs)
    }
}

pub trait DagPipeline {
    type T: Clone + Send + Sync + Encode + 'static;

    fn nodes(&self) -> HashMap<NodeId, DagNode<Self::T>>;
    fn output_node(&self) -> NodeId;

    fn run(&self, input: Self::T, sig: &TypeSig)
        -> Result<Self::T, PipelineError>
    {
        let raw = self.nodes();

        let mut nodes: HashMap<NodeId, DagNode<Self::T>> = HashMap::new();
        for (_, n) in raw {
            nodes.insert(n.id(), n);
        }

        // Kernel invariant: graph must contain at least one node
        if nodes.is_empty() {
            return Err(PipelineError::StageError(
                "graph contains zero nodes",
            ));
        }

        // Kernel invariant: all input NodeIds must exist
        for (id, node) in &nodes {
            for inp in &node.inputs {
                if !nodes.contains_key(inp) {
                    return Err(PipelineError::StageError(
                        "node input references missing NodeId",
                    ));
                }

                // Kernel invariant: no self-loop
                if inp == id {
                    return Err(PipelineError::StageError(
                        "node has self-loop input",
                    ));
                }
            }
            // silence unused warning potential
            let _ = id;
        }

        // Kernel invariant: exactly one root node (zero inputs)
        let root_count = nodes.values().filter(|n| n.inputs.is_empty()).count();
        if root_count != 1 {
            return Err(PipelineError::StageError(
                "graph must contain exactly one root node",
            ));
        }

        // Kernel invariant: all non-root nodes must have exactly one input
        for node in nodes.values() {
            if !node.inputs.is_empty() && node.inputs.len() != 1 {
                return Err(PipelineError::StageError(
                    "non-root node must have exactly one input",
                ));
            }
        }

        // Identify root
        let root_id = nodes
            .iter()
            .find(|(_, n)| n.inputs.is_empty())
            .map(|(id, _)| *id)
            .unwrap();

        // Kernel invariant: declared output node must exist
        let out_id = self.output_node();
        if !nodes.contains_key(&out_id) {
            return Err(PipelineError::StageError("output_node not present in graph"));
        }

        // Kernel invariant: output must be reachable from root
        {
            use std::collections::VecDeque;

            let mut q = VecDeque::new();
            let mut seen = HashSet::new();
            q.push_back(root_id);
            seen.insert(root_id);

            while let Some(cur) = q.pop_front() {
                for (id, node) in &nodes {
                    if node.inputs.contains(&cur) && !seen.contains(id) {
                        seen.insert(*id);
                        q.push_back(*id);
                    }
                }
            }

            if !seen.contains(&out_id) {
                return Err(PipelineError::StageError(
                    "output node is not reachable from root",
                ));
            }

            // Kernel invariant: no unreachable nodes (no dead subgraphs)
            if seen.len() != nodes.len() {
                return Err(PipelineError::StageError(
                    "graph contains unreachable nodes",
                ));
            }
        }

        let graph_hash = structural_graph_hash(&nodes);

        let order = topo_sort(&nodes)?;

        // Kernel invariant: topo order must include all nodes exactly once
        if order.len() != nodes.len() {
            return Err(PipelineError::StageError(
                "topo_sort did not return all nodes",
            ));
        }

        // compute execution levels (parallelizable layers)
        let mut level_map: HashMap<NodeId, usize> = HashMap::new();
        for id in &order {
            let node = &nodes[id];
            let mut max_parent = 0usize;
            for p in &node.inputs {
                if let Some(l) = level_map.get(p) {
                    if *l > max_parent {
                        max_parent = *l;
                    }
                }
            }
            // children must execute AFTER parents
            level_map.insert(*id, max_parent + 1);
        }

        let mut layers: HashMap<usize, Vec<NodeId>> = HashMap::new();
        for (id, lvl) in &level_map {
            layers.entry(*lvl).or_default().push(*id);
        }

        let mut layer_keys: Vec<usize> = layers.keys().cloned().collect();
        layer_keys.sort();

        // GLOBAL JUDGMENT (once per graph)
        let mode: ComputeMode = DefaultJudge::choose(sig);

        let mut values: HashMap<NodeId, NodeResult<Self::T>> = HashMap::new();

        let input_hash = replay::hash_of_bytes(&input.encode());

        for lvl in layer_keys {
            let mut ids = layers.get(&lvl).unwrap().clone();
            ids.sort();

            let mut handles = Vec::new();

            for id in ids {
                let node = nodes.get(&id).unwrap();

                let in_val = if node.inputs.is_empty() {
                    input.clone()
                } else {
                    match values.get(&node.inputs[0]).expect("missing input") {
                        NodeResult::Value { delta, .. } => delta.after.clone(),
                        NodeResult::Error(e) => {
                            values.insert(id, NodeResult::Error(e.clone()));
                            continue;
                        }
                    }
                };

                let stage = node.stage.clone();
                let sig = *sig;
                let mode = mode;

                handles.push(std::thread::spawn(move || {
                    match stage.run_with_mode(in_val, &sig, mode) {
                        Ok(out) => (
                            id,
                            NodeResult::Value {
                                delta: out.delta,
                                proof: out.proof,
                            },
                        ),
                        Err(e) => (id, NodeResult::Error(e)),
                    }
                }));
            }

            for h in handles {
                let (id, res) = h.join().unwrap();
                values.insert(id, res);
            }
        }

        match values.get(&out_id).expect("missing output") {
            NodeResult::Value { delta, .. } => {
                let out = delta.after.clone();
                 let out_hash = replay::hash_of_bytes(&out.encode());
                replay::record(graph_hash, input_hash, out_hash);
                Ok(out)
            }
            NodeResult::Error(e) => Err(e.clone()),
        }
    }
}

pub(crate) fn topo_sort<T>(
    nodes: &HashMap<NodeId, DagNode<T>>
) -> Result<Vec<NodeId>, PipelineError> {

    fn dfs<T>(
        n: NodeId,
        nodes: &HashMap<NodeId, DagNode<T>>,
        temp: &mut HashSet<NodeId>,
        perm: &mut HashSet<NodeId>,
        out: &mut Vec<NodeId>,
    ) -> Result<(), PipelineError> {
        if perm.contains(&n) { return Ok(()); }
        if temp.contains(&n) {
            return Err(PipelineError::StageError("cycle detected"));
        }

        temp.insert(n);
        for i in &nodes[&n].inputs {
            dfs(*i, nodes, temp, perm, out)?;
        }
        temp.remove(&n);
        perm.insert(n);
        out.push(n);
        Ok(())
    }

    let mut temp = HashSet::new();
    let mut perm = HashSet::new();
    let mut out = Vec::new();

    let mut keys: Vec<NodeId> = nodes.keys().cloned().collect();
    keys.sort();

    for k in keys {
        dfs(k, nodes, &mut temp, &mut perm, &mut out)?;
    }

    Ok(out)
}
