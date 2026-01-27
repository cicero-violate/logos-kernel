// Scope: Content-addressed memory store (hash -> delta)

use std::collections::HashMap;
use crate::kernel::codec::Encode;
use crate::kernel::hash::Fnv64;
use std::sync::{Mutex, OnceLock};

static STORE: OnceLock<Mutex<HashMap<u64, Vec<u8>>>> = OnceLock::new();

fn store() -> &'static Mutex<HashMap<u64, Vec<u8>>> {
    STORE.get_or_init(|| Mutex::new(HashMap::new()))
}

pub fn hash_of<T: Encode>(v: &T) -> u64 {
    let bytes = v.encode();
    let mut h = Fnv64::new();
    h.write(&bytes);
    h.finish()
}

pub fn get(h: u64) -> Option<Vec<u8>> {
    store().lock().unwrap().get(&h).cloned()
}

pub fn put(h: u64, v: Vec<u8>) {
    store().lock().unwrap().insert(h, v);
}
