use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::marker::PhantomData;

// A toy implementation of HyperLogLog.
//
// This implementation is BYO Hash Function - callers should add data to the
// HLL using the [`iter`] method. Data should already be a 64-bit value
// drawn from a uniform distribution (read: hashed well).
pub struct HLL {
    pub m: usize,
    pub register_width: usize,
    b: usize, // log2m
    registers: Registers,
}

impl HLL {
    // Create a new HLL with the given register width and log2m set to the
    // given value. log2m must not be zero.
    pub fn new(log2m: usize, register_width: usize) -> HLL {
        assert!(log2m > 0, "log2m must not be zero");

        let b = log2m;
        let m = 1 << b;

        HLL { m, b, register_width, registers: Registers::new(register_width, m) }
    }

    // Add a raw value to the multiset. The value MUST have been hashed or
    // drawn from a uniform distribution.
    pub fn add_raw(&mut self, value: u64) {
        let j = (value as usize) & (self.m - 1);
        let w = value >> self.b;

        // NOTE: the paper defines p(0^k) == k + 1 but 0.trailing_zeros() == 0
        // so we have to correct here
        let p_w = 1 + {
            if value == 0 {
                (32 - self.b) + 1
            } else {
                w.trailing_zeros() as usize
            }
        };

        self.registers.set_max(j, p_w as u8);
    }

    // Union another HLL value into this one. This is equivalent to setting
    // every register in this HLL to the max of it's current value and the
    // corresponding register in the other HLL.
    //
    // Does not validate that the two HLLs are compatible, and will panic if
    // other has a higher log2m value than self.
    pub fn union(&mut self, other: &Self) {
        for (i, v) in other.registers.iter().enumerate() {
            self.registers.set_max(i, v);
        }
    }

    // Returns an estimate of the cardinality of the multiset.
    pub fn cardinality_estimate(&self) -> f64 {
        match self.estimator_and_zeros() {
            (e, z) if z > 0 && e <= HLL::small_estimator_cutoff(self.m) => {
                HLL::small_estimator(self.m, z)
            }
            (e, _) if e > HLL::large_estimator_cutoff(self.register_width, self.b) => {
                HLL::large_estimator(self.register_width, self.b, e)
            }
            (e, _) => e,
        }
    }

    #[inline]
    fn estimator_and_zeros(&self) -> (f64, usize) {
        let mut sum: f64 = 0.0;
        let mut zeros: usize = 0;

        for v in self.registers.iter() {
            sum += 1.0 / ((1 << v) as f64);
            if v == 0 {
                zeros += 1;
            }
        }

        (HLL::alpha_m_squared(self.m) / sum, zeros)
    }

    #[inline]
    fn alpha_m_squared(m: usize) -> f64 {
        let mf = m as f64;
        let alpha = match m {
            16 => 0.673,
            32 => 0.697,
            64 => 0.709,
            _ => 0.7213 / (1.0 + 1.079 / mf),
        };
        alpha * mf * mf
    }

    #[inline]
    fn small_estimator_cutoff(m: usize) -> f64 {
        let m = m as f64;
        (5.0 / 2.0) * m
    }

    #[inline]
    fn small_estimator(m: usize, zeros: usize) -> f64 {
        let m = m as f64;
        let zeros = zeros as f64;
        m * (m / zeros).ln()
    }

    #[inline]
    fn large_estimator_cutoff(rw: usize, log2m: usize) -> f64 {
        HLL::two_to_l(rw, log2m) / 30.0
    }

    #[inline]
    fn large_estimator(rw: usize, log2m: usize, est: f64) -> f64 {
        let ttl = HLL::two_to_l(rw, log2m);
        -1.0 * ttl * (1.0 - est / ttl).ln()
    }

    #[inline]
    fn two_to_l(register_width: usize, log2m: usize) -> f64 {
        // this needs to be -2 instead of -1 to account for the fact that
        // p_w(0) = 1 and not 0
        let max_register_val = (1 << register_width) - 1 - 1;
        (2.0_f64).powi((max_register_val + log2m) as i32)
    }
}

impl std::fmt::Debug for HLL {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (estimator, zeros) = self.estimator_and_zeros();

        writeln!(f, "m={} b={} width={}", self.m, self.b, self.register_width)?;
        writeln!(f, "zeros={}, estimator={}", zeros, estimator)
    }
}

// A fixed-size array of N-bit wide registers. Used as storage for an HLL.
struct Registers {
    ptr: *mut u8,
    width: usize,
    len: usize,
}

impl Registers {
    const MIN_WIDTH: usize = 1;
    const MAX_WIDTH: usize = 8;

    // Allocate a new set of len registers of the given width. The length
    // of the registers will never change.
    //
    // Registers must be between MIN_WIDTH and MAX_WIDTH bits wide.
    fn new(width: usize, len: usize) -> Registers {
        assert!(len > 0, "invalid register len");
        assert!(1 <= Registers::MIN_WIDTH && Registers::MAX_WIDTH <= 8, "invalid regsiter width");

        let ptr = unsafe { alloc_zeroed(Registers::layout(width, len)) };

        Registers { ptr, len, width }
    }

    // set the value of the ith register to v iff v is greater than the existing
    // value. this is equivalent to the following snippet but uses fewer
    // instructions.
    //
    //      if rs.get(i) > v {
    //          rs.set(i, v)
    //      }
    //
    fn set_max(&mut self, i: usize, v: u8) {
        assert!(i < self.len);

        unsafe {
            let mask = Registers::mask_for(self.width);
            let ptr = self.ptr.add(i * self.width);

            let prev = *ptr & mask;
            if prev < v {
                *ptr &= !mask;
                *ptr |= v & mask;
            }
        }
    }

    // Returns an iterator over the current values of every register.
    fn iter(&self) -> RegisterIterator {
        RegisterIterator {
            _pd: PhantomData,
            ptr: self.ptr,
            idx: 0,
            len: self.len,
            width: self.width,
        }
    }

    // return the value of the ith register. unused except for testing.
    #[allow(dead_code)]
    fn get(&self, i: usize) -> u8 {
        assert!(i < self.len);

        unsafe {
            let mask = Registers::mask_for(self.width);
            let ptr = self.ptr.add(i * self.width);

            *ptr & mask
        }
    }

    // set the value of the ith register to v. unused except for testing.
    #[allow(dead_code)]
    fn set(&mut self, i: usize, v: u8) {
        assert!(i < self.len);

        unsafe {
            let mask = Registers::mask_for(self.width);
            let ptr = self.ptr.add(i * self.width);

            *ptr &= !mask;
            *ptr |= v & mask;
        }
    }

    #[inline]
    fn layout(width: usize, len: usize) -> Layout {
        Layout::from_size_align(width * len, 8).expect("invalid register layout")
    }

    #[inline]
    fn mask_for(width: usize) -> u8 {
        (((1 as u64) << width) - 1) as u8
    }
}

impl Drop for Registers {
    fn drop(&mut self) {
        let layout = Registers::layout(self.width, self.len);
        unsafe {
            dealloc(self.ptr, layout);
        }
    }
}

// An iterator that yields the current value of dense HLL registers. This
// `struct` is created by the [`iter`] method on [`Registers`]. See its
// documentation for more details.
struct RegisterIterator<'a> {
    _pd: PhantomData<&'a Registers>,
    ptr: *mut u8,
    width: usize,
    idx: usize,
    len: usize,
}

impl<'a> Iterator for RegisterIterator<'a> {
    type Item = u8;

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx >= self.len {
            None
        } else {
            let val = unsafe {
                let mask = Registers::mask_for(self.width);
                let ptr = self.ptr.add(self.idx * self.width);

                *ptr & mask
            };

            self.idx += 1;
            Some(val)
        }
    }
}

#[cfg(test)]
mod test_registers {
    use super::*;
    use quickcheck::{self, empty_shrinker, single_shrinker, Arbitrary, Gen};
    use quickcheck_macros::quickcheck;
    use rand::Rng;

    #[derive(Debug, Clone)]
    struct TestCase {
        width: usize,
        len: usize,
    }

    impl Arbitrary for TestCase {
        fn arbitrary<G: Gen>(g: &mut G) -> TestCase {
            TestCase {
                width: g.gen_range(Registers::MIN_WIDTH, Registers::MAX_WIDTH),
                len: g.size(),
            }
        }

        fn shrink(&self) -> Box<dyn Iterator<Item = Self>> {
            if self.len <= 2 {
                empty_shrinker()
            } else {
                single_shrinker(TestCase { width: self.width, len: self.len / 2 })
            }
        }
    }

    #[quickcheck]
    fn test_init_zeroed(tc: TestCase) -> bool {
        let rs = Registers::new(tc.width, tc.len);

        (0..tc.len).map(|i| rs.get(i)).all(|v| v == 0)
    }

    #[quickcheck]
    fn test_iter_length(tc: TestCase) -> bool {
        Registers::new(tc.width, tc.len).iter().count() == tc.len
    }

    #[quickcheck]
    fn test_set_odd(tc: TestCase) -> bool {
        let mut rs = Registers::new(tc.width, tc.len);
        let val = 0b10101010 & Registers::mask_for(tc.width);

        for i in (0..tc.len).filter(|i| i % 2 == 0) {
            rs.set(i, val);
        }

        let all_odds_zero = rs.iter().enumerate().filter(|&(i, _)| i % 2 == 1).all(|(_, v)| v == 0);
        let all_evens_v = rs.iter().enumerate().filter(|&(i, _)| i % 2 == 0).all(|(_, v)| v == val);

        all_odds_zero && all_evens_v
    }

    #[quickcheck]
    fn test_set_max(tc: TestCase) -> bool {
        let mut rs = Registers::new(tc.width, tc.len);
        let low_v = 0b001 & Registers::mask_for(tc.width);
        let high_v = 0b111 & Registers::mask_for(tc.width);

        for i in 0..tc.len {
            rs.set_max(i, low_v);
            rs.set_max(i, high_v);
            rs.set_max(i, low_v);
        }

        rs.iter().all(|v| v == high_v)
    }
}

#[cfg(test)]
mod test_hll {
    use super::*;

    #[test]
    fn test_union_sets_max() {
        let mut h1 = HLL::new(2, 5);
        h1.registers.set_max(0, 3);
        h1.registers.set_max(1, 1);
        h1.registers.set_max(2, 1);
        h1.registers.set_max(3, 3);

        let mut h2 = HLL::new(2, 5);
        h2.registers.set_max(0, 2);
        h2.registers.set_max(1, 2);
        h2.registers.set_max(2, 2);
        h2.registers.set_max(3, 2);

        h1.union(&h2);
        assert_eq!(
            h1.registers.iter().collect::<Vec<u8>>(),
            vec![3, 2, 2, 3],
            "expected unioned registers to be the pairwise max of both HLLs"
        );
    }
}
