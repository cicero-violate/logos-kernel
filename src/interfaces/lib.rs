// Scope: Library interface

use crate::app::dag_pipeline::MyDagPipeline;
use crate::kernel::types::DEFAULT_SIG;
use crate::kernel::dag::DagPipeline;

pub fn run_pipeline(input: i64) {
    let p = MyDagPipeline;

    match p.run(input, &DEFAULT_SIG) {
        Ok(out) => println!("RESULT => {}", out),
        Err(e) => println!("ERROR => {:?}", e),
    }
}
