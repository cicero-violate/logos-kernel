use crate::kernel::hash::hash_bytes;
use std::sync::{Mutex, OnceLock};
use std::fs::OpenOptions;
use std::io::Write;

// (graph_hash, input_hash, output_hash)
static LOG: OnceLock<Mutex<Vec<(u64, u64, u64)>>> = OnceLock::new();

fn log_store() -> &'static Mutex<Vec<(u64, u64, u64)>> {
    LOG.get_or_init(|| Mutex::new(Vec::new()))
}

pub fn hash_of_bytes(bytes: &[u8]) -> u64 {
    hash_bytes(bytes)
}

pub fn record(graph_hash: u64, input_hash: u64, output_hash: u64) {
    let entry = (graph_hash, input_hash, output_hash);

    // In-memory
    log_store().lock().unwrap().push(entry);

    // On-disk (append-only)
    if let Ok(mut f) = OpenOptions::new()
        .create(true)
        .append(true)
        .open("replay.log")
    {
        let _ = writeln!(f, "{} {} {}", graph_hash, input_hash, output_hash);
    }
}

pub fn entries() -> Vec<(u64, u64, u64)> {
    log_store().lock().unwrap().clone()
}

// Clear in-memory replay log (tests / controlled resets only)
pub fn clear() {
    log_store().lock().unwrap().clear();
}

// Load replay entries from on-disk log
pub fn load_from_disk(path: &str) -> Vec<(u64, u64, u64)> {
    let data = match std::fs::read_to_string(path) {
        Ok(d) => d,
        Err(_) => return Vec::new(),
    };

    let mut out = Vec::new();

    for line in data.lines() {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() != 3 {
            continue;
        }

        if let (Ok(g), Ok(i), Ok(o)) = (
            parts[0].parse::<u64>(),
            parts[1].parse::<u64>(),
            parts[2].parse::<u64>(),
        ) {
            out.push((g, i, o));
        }
    }

    out
}

// Verify replay log consistency:
// same (graph_hash, input_hash) => same output_hash
pub fn verify(entries: &[(u64, u64, u64)]) -> bool {
    use std::collections::HashMap;

    let mut seen: HashMap<(u64, u64), u64> = HashMap::new();

    for (g, i, o) in entries {
        match seen.get(&(*g, *i)) {
            Some(prev) if *prev != *o => return false,
            None => {
                seen.insert((*g, *i), *o);
            }
            _ => {}
        }
    }

    true
}
