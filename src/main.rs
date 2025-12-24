use rayon::prelude::*;
use sha2::{Digest, Sha256};
use std::{ops::BitXor, time::Instant};

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

const WIDTH: usize = (Input::BITS / 4) as usize;

fn main() {
    println!("{}", PREFIX.trim());
    let mut max_agree = 0;
    let mut hashes = Vec::<Uint256>::new();
    let start = Instant::now();
    for i in 0..=Input::MAX {
        let hash = Uint256::sha256(i);
        if let Some((agree, j)) = hashes
            .par_iter()
            .with_min_len(1024)
            .enumerate()
            .map(|(j, &other)| ((other ^ hash).count_zeros(), j as Input))
            .max_by_key(|&(agree, _)| agree)
            && agree > max_agree
        {
            let t = start.elapsed();
            println!("check({agree}, \"{j:0WIDTH$x}\", \"{i:0WIDTH$x}\")  # found after {t:?}");
            max_agree = agree;
        }
        hashes.push(hash);
    }
}
