// Scope: Server interface (placeholder)
//
// WIRE INVARIANT:
// ----------------
// Network input MUST be canonical binary encoding of T.
//
// Current protocol:
//   - Client sends exactly 8 bytes (little-endian i64)
//   - Server performs Decode -> T
//   - Kernel executes
//   - Server returns Encode(T) as raw bytes
//
// No text, no JSON, no framing, no schema.
// Transport is byte-only. Semantics live in kernel.

use crate::app::dag_pipeline::MyDagPipeline;
use crate::kernel::types::DEFAULT_SIG;
use crate::kernel::dag::DagPipeline;
use std::net::{TcpListener, TcpStream};
use std::io::{Read, Write};
use crate::kernel::codec::Decode;

pub fn run_server() {
    let listener = TcpListener::bind("127.0.0.1:9000")
        .expect("failed to bind tcp port 9000");

    println!("Server listening on 127.0.0.1:9000");

    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                handle_client(stream);
            }
            Err(e) => {
                eprintln!("accept error: {}", e);
            }
        }
    }
}

fn handle_client(mut stream: TcpStream) {
    // Expect exactly 8 bytes = little-endian i64
    let mut buf = [0u8; 8];

    if stream.read_exact(&mut buf).is_err() {
        let _ = stream.write_all(b"read error\n");
        return;
    }

    let input: i64 = match i64::decode(&buf) {
        Some(v) => v,
        None => {
            let _ = stream.write_all(b"decode error\n");
            return;
        }
    };

    let p = MyDagPipeline;

    // Run pipeline
    match p.run(input, &DEFAULT_SIG) {
        Ok(out) => {
            // Return binary i64
            let bytes = out.encode();
            let _ = stream.write_all(&bytes);
        }
        Err(_e) => {
            let _ = stream.write_all(b"exec error\n");
        }
    }
}
use crate::kernel::codec::Encode;
