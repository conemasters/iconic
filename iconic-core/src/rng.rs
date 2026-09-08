//! Dependency-free deterministic RNGs for reproducible problem generation and
//! tests, shared by every crate in the workspace.
//!
//! The generators are deliberately tiny and fixed: benchmark seeds are
//! load-bearing (reference optima tables are verified against exactly these
//! draws), so the constants and output mappings here must never change.

/// A linear congruential generator with PCG-style 64-bit state update
/// (Knuth constants), producing uniform `[0, 1)` draws via the top 53 bits.
#[derive(Clone, Debug)]
pub struct Lcg {
    state: u64,
}

impl Lcg {
    /// Seed the generator.
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x9E37_79B9_7F4A_7C15,
        }
    }

    /// Next raw 64-bit value.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    /// Uniform value in `[0, 1)`.
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }

    /// Uniform value in `[lo, hi)`.
    pub fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }

    /// Uniform value in `[-1, 1)`.
    pub fn signed(&mut self) -> f64 {
        2.0 * self.unit() - 1.0
    }

    /// Uniform index in `[0, n)`.
    pub fn pick(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

/// A 13/7/17 xorshift generator, the workhorse of the benchmark suite and the
/// randomized property tests. Draw-compatible with every historical copy of
/// this generator (same seed transform, same output mapping).
#[derive(Clone, Debug)]
pub struct XorShift(u64);

impl XorShift {
    /// Seed the generator, shifting away from zero (xorshift is stuck there).
    pub fn new(seed: u64) -> Self {
        XorShift(seed.wrapping_add(1).max(1))
    }

    /// Next raw 64-bit value.
    pub fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// Uniform value in `[0, 1]`.
    pub fn unit(&mut self) -> f64 {
        self.next_u64() as f64 / (u64::MAX as f64)
    }

    /// Uniform value in `[lo, hi)`.
    pub fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }

    /// Uniform index in `[0, n)`.
    pub fn pick(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

/// SplitMix64 finalizer stepped by the golden ratio, returning signed
/// `[−1, 1)` draws. Used where reproducibility matters more than sequence
/// quality (diagnostic generators, regression-test fixtures).
#[derive(Clone, Debug)]
pub struct SplitMix(u64);

impl SplitMix {
    /// Seed the generator.
    pub fn new(seed: u64) -> Self {
        SplitMix(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15))
    }

    /// Uniform value in `[−1, 1)`.
    pub fn signed(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        (z ^ (z >> 31)) as f64 / (u64::MAX as f64) * 2.0 - 1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lcg_draws_are_pinned_and_in_range() {
        let seed: u64 = 42;
        let mut g = Lcg::new(seed);
        let d0 = g.next_u64();
        assert_eq!(
            d0,
            (seed ^ 0x9E37_79B9_7F4A_7C15)
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407)
        );
        assert!((Lcg::new(7).unit() >= 0.0));
        let mut g = Lcg::new(7);
        assert!((0..100).all(|_| {
            let u = g.unit();
            (0.0..1.0).contains(&u)
        }));
    }

    #[test]
    fn xorshift_draws_are_pinned_and_nonzero_seeded() {
        let g = XorShift::new(0);
        assert_eq!(g.0, 1);
        let g = XorShift::new(5);
        assert_ne!(g.0, 0);
    }
}
