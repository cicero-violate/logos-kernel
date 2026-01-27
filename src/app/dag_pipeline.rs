use std::collections::HashMap;

use crate::kernel::dag::*;
use crate::kernel::dag_builder::*;
use crate::stages::add_one::AddOne;
use crate::stages::mul_two::MulTwo;
use crate::stages::sub_one::SubOne;

// Example pipeline used for kernel determinism + replay validation.
// Not a production pipeline.

pub struct MyDagPipeline;

impl DagPipeline for MyDagPipeline {
    type T = i64;

    fn nodes(&self) -> HashMap<NodeId, DagNode<i64>> {
        let spec = DagSpec {
            nodes: vec![
                NodeSpec {
                    name: "add_one_a",
                    stage: std::sync::Arc::new(AddOne),
                    inputs: vec![],
                },
                NodeSpec {
                    name: "mul_two",
                    stage: std::sync::Arc::new(MulTwo),
                    inputs: vec!["add_one_a"],
                },
                NodeSpec {
                    name: "sub_one",
                    stage: std::sync::Arc::new(SubOne),
                    inputs: vec!["mul_two"],
                },
            ],
        };

        DagBuilder::build(&spec)
    }

    fn output_node(&self) -> NodeId {
        // must match final node: sub_one
        let tmp = DagNode {
            name: "add_one_a",
            stage: std::sync::Arc::new(AddOne),
            inputs: vec![],
        };

        let id0 = tmp.id();

        let mid = DagNode {
            name: "mul_two",
            stage: std::sync::Arc::new(MulTwo),
            inputs: vec![id0],
        };

        let out = DagNode {
            name: "sub_one",
            stage: std::sync::Arc::new(SubOne),
            inputs: vec![mid.id()],
        };

        out.id()
    }
}
