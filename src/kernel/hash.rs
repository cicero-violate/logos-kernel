// Scope: Stable hashing primitives (explicit, deterministic)
// Algorithm: FNV-1a 64-bit

pub const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
pub const FNV_PRIME: u64 = 0x100000001b3;

#[derive(Clone)]
pub struct Fnv64 {
    state: u64,
}

impl Fnv64 {
    pub fn new() -> Self {
        Self { state: FNV_OFFSET_BASIS }
    }

    pub fn write(&mut self, bytes: &[u8]) {
        for b in bytes {
            self.state ^= *b as u64;
            self.state = self.state.wrapping_mul(FNV_PRIME);
        }
    }

    pub fn write_u64(&mut self, v: u64) {
        self.write(&v.to_le_bytes());
    }

    pub fn write_str(&mut self, s: &str) {
        self.write(s.as_bytes());
    }

    pub fn finish(&self) -> u64 {
        self.state
    }
}

pub fn hash_bytes(bytes: &[u8]) -> u64 {
    let mut h = Fnv64::new();
    h.write(bytes);
    h.finish()
}

pub fn hash_u64(v: u64) -> u64 {
    let mut h = Fnv64::new();
    h.write_u64(v);
    h.finish()
}

pub fn hash_str(s: &str) -> u64 {
    let mut h = Fnv64::new();
    h.write_str(s);
    h.finish()
}

