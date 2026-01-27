// Scope: Host entrypoint (binary)

extern crate mmsb;

use mmsb::interfaces;
use std::env;

fn main() {
    let mode = env::args().nth(1);

    match mode.as_deref() {
        Some("server") => {
            interfaces::server::run_server();
        }
        _ => {
            interfaces::cli::run();
        }
    }
}
