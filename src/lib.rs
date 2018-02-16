#![feature(pointer_methods, allocator_api)]

pub struct HLL {
    pub m: usize,
    pub b: usize,
    pub register_width: usize,
    registers: Registers,
}

impl HLL {
    pub fn new(log2m: usize, register_width: usize) -> HLL {
        let b = log2m;
        let m = 1 << b;

        HLL {
            m: m,
            b: b,
            register_width: register_width,
            registers: Registers::new(register_width, m),
        }
    }

    pub fn add_raw(&mut self, value: u32) {
        let j = (value as usize) & (self.m - 1);
        let w = value >> self.b;
        // NOTE(benl): the paper defines p(0^k) == k + 1 but lowest_one_bit(0) == 0
        // so we have to correct here
        let p_w = 1 + if value == 0 {
            (32 - self.b) + 1
        } else {
            lowest_one_bit(w)
        };
        self.registers.set_max(j, p_w as u8);
    }

    pub fn cardinality_estimate(&self) -> f64 {
        let (estimator, zeros) = self.estimator();
        match (estimator, zeros) {
            (e, z) if z > 0 && e <= HLL::small_estimator_cutoff(self.m) => {
                HLL::small_estimator(self.m, zeros)
            }
            (e, _) if e > HLL::large_estimator_cutoff(self.m) => HLL::large_estimator(e),
            _ => estimator,
        }
    }

    #[inline]
    fn estimator(&self) -> (f64, usize) {
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

    // NOTE(benl): this ignores its input on purpose. it's here as a placeholder
    // in case this implementation ever grows to include the improvements from
    // "HyperLogLog in Practice"
    #[inline]
    fn large_estimator_cutoff(_m: usize) -> f64 {
        (1.0 / 30.0) * ((1 as u64) << 32) as f64
    }

    #[inline]
    fn large_estimator(est: f64) -> f64 {
        let two_to_32 = ((1 as u64) << 32) as f64;
        -1.0 * two_to_32 * (1.0 - est / two_to_32).ln()
    }
}

impl std::fmt::Debug for HLL {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let (estimator, zeros) = self.estimator();

        writeln!(f, "m={} b={} width={}", self.m, self.b, self.register_width)?;
        writeln!(f, "zeros={}, estimator={}", zeros, estimator)
    }
}

use std::heap::{Alloc, Heap, Layout};

// A fixed-size array of N-bit wide registers.
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
        assert!(
            1 <= Registers::MIN_WIDTH && Registers::MAX_WIDTH <= 8,
            "invalid regsiter width"
        );

        let ptr = unsafe {
            // TODO(benl): is 8 byte alignment ok?
            let layout = Registers::layout(width, len);
            Heap::default()
                .alloc_zeroed(layout)
                .expect("hll: allocation failure")
        };

        Registers {
            ptr: ptr,
            len: len,
            width: width,
        }
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
        unsafe {
            let layout = Registers::layout(self.width, self.len);
            Heap::default().dealloc(self.ptr, layout);
        }
    }
}

// An iterator that yields the current value of dense HLL registers. This
// `struct` is created by the [`iter`] method on [`Registers`]. See its
// documentation for more details.
struct RegisterIterator {
    ptr: *mut u8,
    width: usize,
    idx: usize,
    len: usize,
}

impl Iterator for RegisterIterator {
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
#[macro_use]
extern crate quickcheck;

#[cfg(test)]
mod test_registers {
    use super::*;
    use quickcheck::*;

    #[derive(Debug, Clone)]
    struct TestCase {
        width: usize,
        len: usize,
    }

    impl Arbitrary for TestCase {
        fn arbitrary<G: Gen>(g: &mut G) -> TestCase {
            let reg_width = g.gen_range(Registers::MIN_WIDTH, Registers::MAX_WIDTH + 1);
            let len = g.size();

            TestCase {
                width: reg_width,
                len: len,
            }
        }

        fn shrink(&self) -> Box<Iterator<Item = Self>> {
            if self.len <= 2 {
                empty_shrinker()
            } else {
                single_shrinker(TestCase {
                    width: self.width,
                    len: self.len / 2,
                })
            }
        }
    }

    quickcheck!{
        fn qc_init_zero(tc: TestCase) -> bool {
            let rs = Registers::new(tc.width, tc.len);

            (0..tc.len).map(|i| rs.get(i)).all(|v| v == 0)
        }

        fn qc_set_odd(tc: TestCase) -> bool {
            let mut rs = Registers::new(tc.width, tc.len);
            let val = 0b10101010 & Registers::mask_for(tc.width);

            for i in (0..tc.len).filter(|i| i % 2 == 0) {
                rs.set(i, val);
            }

            let all_odds_zero = rs.iter().enumerate().filter(|&(i, _)| i % 2 == 1).all(|(_, v)| v == 0);
            let all_evens_v = rs.iter().enumerate().filter(|&(i, _)| i % 2 == 0).all(|(_, v)| v == val);

            all_odds_zero && all_evens_v
        }

        fn qc_set_max(tc: TestCase) -> bool {
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

        fn qc_iter_length(tc: TestCase) -> bool {
            Registers::new(tc.width, tc.len).iter().count() == tc.len
        }
    }
}

const _MULTIPLY_DEBRUJIN_POSITION: [usize; 32] = [
    0, 1, 28, 2, 29, 14, 24, 3, 30, 22, 20, 15, 25, 17, 4, 8, 31, 27, 13, 23, 21, 19, 16, 7, 26,
    12, 18, 6, 11, 5, 10, 9,
];

const _DEBRUJIN_SEQ: u32 = 0x077CB531;

// Find the position of the lowest set bit in a u32
//
// http://www.graphics.stanford.edu/~seander/bithacks.html#ZerosOnRightMultLookup
#[inline]
fn lowest_one_bit(x: u32) -> usize {
    let minus_x = (0 as u32).wrapping_sub(x);
    let idx = (x & minus_x).wrapping_mul(_DEBRUJIN_SEQ) >> 27;
    _MULTIPLY_DEBRUJIN_POSITION[idx as usize]
}

#[cfg(test)]
mod test_bits {
    use super::lowest_one_bit;

    #[test]
    fn test_lowest_one_bit() {
        assert_eq!(0, lowest_one_bit(0b0));
        assert_eq!(0, lowest_one_bit(0b1));

        assert_eq!(1, lowest_one_bit(0b010));
        assert_eq!(1, lowest_one_bit(0b110));

        assert_eq!(2, lowest_one_bit(0b0100));
        assert_eq!(2, lowest_one_bit(0b1100));

        assert_eq!(3, lowest_one_bit(0b01000));
        assert_eq!(3, lowest_one_bit(0b11000));

        assert_eq!(4, lowest_one_bit(0b010000));
        assert_eq!(4, lowest_one_bit(0b110000));
    }
}
