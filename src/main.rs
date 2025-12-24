use sha2::{Digest, Sha256};
use std::{
    ops::BitXor,
    sync::{
        Arc,
        atomic::{AtomicU32, Ordering},
        mpsc,
    },
    thread,
    time::Instant,
};

type Input = u32;

#[derive(Clone, Copy)]
struct Uint256 {
    hi: u128,
    lo: u128,
}

impl Uint256 {
    fn sha256(input: Input) -> Self {
        let hash = Sha256::digest(input.to_be_bytes());
        let (hi, lo) = hash.split_at(16);
        Self {
            hi: u128::from_be_bytes(hi.try_into().unwrap()),
            lo: u128::from_be_bytes(lo.try_into().unwrap()),
        }
    }

    fn count_zeros(self) -> u32 {
        self.hi.count_zeros() + self.lo.count_zeros()
    }
}

impl BitXor for Uint256 {
    type Output = Self;

    fn bitxor(self, rhs: Self) -> Self::Output {
        Self {
            hi: self.hi ^ rhs.hi,
            lo: self.lo ^ rhs.lo,
        }
    }
}

const PREFIX: &str = "
from hashlib import sha256
def check(c, s, t):
    a = sha256(bytes.fromhex(s)).digest()
    b = sha256(bytes.fromhex(t)).digest()
    assert c == sum((x ^ y ^ 0xff).bit_count() for (x, y) in zip(a, b))
";

const PAD: usize = (Input::BITS / 4) as usize;

fn l1_data_cache_size() -> usize {
    #[cfg(target_arch = "x86_64")]
    {
        if let Some(size) = cache_size::l1_cache_size() {
            return size;
        }
    }
    #[cfg(target_os = "macos")]
    {
        use sysctl::Sysctl;
        if let Ok(ctl) = sysctl::Ctl::new("hw.l1dcachesize")
            && let Ok(value) = ctl.value()
            && let Ok(size) = value.into_s64()
        {
            return size as usize;
        }
    }
    panic!("unable to determine L1 data cache size");
}

fn main() {
    println!("{}", PREFIX.trim());
    let threads = thread::available_parallelism().unwrap().get();
    let elements = (l1_data_cache_size() / 2) / size_of::<Uint256>();
    let best = Arc::new(AtomicU32::new(0));
    let (tx, rx) = mpsc::channel::<(u32, Input, Input)>();
    let began = Instant::now();
    let mut handles = Vec::from_iter((0..threads).map(|thread| {
        let best = Arc::clone(&best);
        let tx = tx.clone();
        thread::spawn(move || {
            let mut start = thread * elements;
            loop {
                let Ok(mut i) = Input::try_from(start + elements) else {
                    break;
                };
                let hashes = Vec::from_iter(((start as Input)..i).map(Uint256::sha256));
                loop {
                    let hash = Uint256::sha256(i);
                    let max_agree = best.load(Ordering::Relaxed);
                    for (j, &other) in hashes.iter().enumerate() {
                        let agree = (other ^ hash).count_zeros();
                        if agree > max_agree {
                            tx.send((agree, i, (start + j) as Input)).unwrap();
                        }
                    }
                    match i.checked_add(1) {
                        Some(next) => i = next,
                        None => break,
                    }
                }
                start += threads * elements;
            }
        })
    }));
    handles.push(thread::spawn({
        let best = Arc::clone(&best);
        move || {
            let mut max_agree = 0;
            while let Ok((agree, i, j)) = rx.recv() {
                if agree > max_agree {
                    max_agree = agree;
                    best.store(max_agree, Ordering::SeqCst);
                    let t = began.elapsed();
                    println!("check({agree}, \"{j:0PAD$x}\", \"{i:0PAD$x}\")  # found after {t:?}");
                }
            }
        }
    }));
    for handle in handles {
        let _ = handle.join();
    }
}
