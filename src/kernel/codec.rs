// Canonical binary serialization (minimal, stable)

use crate::kernel::types::{Delta, NodeResult};
use crate::kernel::error::PipelineError;

pub trait Encode {
    fn encode(&self) -> Vec<u8>;
}

pub trait Decode: Sized {
    fn decode(bytes: &[u8]) -> Option<Self>;
}

impl Encode for u64 {
    fn encode(&self) -> Vec<u8> {
        self.to_le_bytes().to_vec()
    }
}

impl Decode for u64 {
    fn decode(bytes: &[u8]) -> Option<Self> {
        if bytes.len() != 8 { return None; }
        let mut arr = [0u8; 8];
        arr.copy_from_slice(bytes);
        Some(u64::from_le_bytes(arr))
    }
}

impl Encode for i64 {
    fn encode(&self) -> Vec<u8> {
        self.to_le_bytes().to_vec()
    }
}

impl Decode for i64 {
    fn decode(bytes: &[u8]) -> Option<Self> {
        if bytes.len() != 8 { return None; }
        let mut arr = [0u8; 8];
        arr.copy_from_slice(bytes);
        Some(i64::from_le_bytes(arr))
    }
}

impl Encode for bool {
    fn encode(&self) -> Vec<u8> {
        if *self { vec![1] } else { vec![0] }
    }
}

impl Decode for bool {
    fn decode(bytes: &[u8]) -> Option<Self> {
        if bytes.len() != 1 { return None; }
        match bytes[0] {
            0 => Some(false),
            1 => Some(true),
            _ => None,
        }
    }
}

impl<T: Encode> Encode for Vec<T> {
    fn encode(&self) -> Vec<u8> {
        let mut v = Vec::new();
        let len = self.len() as u64;
        v.extend(len.to_le_bytes());
        for item in self {
            let b = item.encode();
            let blen = b.len() as u64;
            v.extend(blen.to_le_bytes());
            v.extend(b);
        }
        v
    }
}

impl<T: Decode> Decode for Vec<T> {
    fn decode(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 8 { return None; }
        let mut idx = 0;

        let mut len_arr = [0u8; 8];
        len_arr.copy_from_slice(&bytes[idx..idx+8]);
        let count = u64::from_le_bytes(len_arr) as usize;
        idx += 8;

        let mut out = Vec::with_capacity(count);

        for _ in 0..count {
            if idx + 8 > bytes.len() { return None; }
            let mut larr = [0u8; 8];
            larr.copy_from_slice(&bytes[idx..idx+8]);
            let blen = u64::from_le_bytes(larr) as usize;
            idx += 8;

            if idx + blen > bytes.len() { return None; }
            let item = T::decode(&bytes[idx..idx+blen])?;
            idx += blen;
            out.push(item);
        }

        Some(out)
    }
}

impl<T: Encode> Encode for Delta<T> {
    fn encode(&self) -> Vec<u8> {
        let mut v = Vec::new();
        let b1 = self.before.encode();
        let b2 = self.after.encode();
        v.extend((b1.len() as u64).to_le_bytes());
        v.extend(b1);
        v.extend(b2);
        v
    }
}

impl<T: Decode> Decode for Delta<T> {
    fn decode(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 8 { return None; }

        let mut larr = [0u8; 8];
        larr.copy_from_slice(&bytes[..8]);
        let blen = u64::from_le_bytes(larr) as usize;

        if 8 + blen > bytes.len() { return None; }

        Some(Delta {
            before: T::decode(&bytes[8..8+blen])?,
            after: T::decode(&bytes[8+blen..])?,
        })
    }
}

impl Encode for PipelineError {
    fn encode(&self) -> Vec<u8> {
        match self {
            PipelineError::StageError(_) => vec![1],
        }
    }
}

impl Encode for NodeResult<u64> {
    fn encode(&self) -> Vec<u8> {
        match self {
            NodeResult::Value { delta, proof } => {
                let mut v = vec![0];
                v.extend(delta.encode());
                match proof {
                    Some(p) => {
                        v.push(1);
                        v.extend(p);
                    }
                    None => v.push(0),
                }
                v
            }
            NodeResult::Error(e) => {
                let mut v = vec![1];
                v.extend(e.encode());
                v
            }
        }
    }
}
