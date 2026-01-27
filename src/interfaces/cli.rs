// Scope: Command-line interface

use crate::app::dag_pipeline::MyDagPipeline;
use crate::kernel::types::DEFAULT_SIG;
use crate::kernel::dag::DagPipeline;

pub fn run() {
    let p = MyDagPipeline;

    match p.run(0, &DEFAULT_SIG) {
        Ok(out) => println!("RESULT => {}", out),
        Err(e) => println!("ERROR => {:?}", e),
    }
}
