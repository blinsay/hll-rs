use std::alloc::{alloc_zeroed, dealloc, Layout};

// TODO: impl Clone for Registers and HLL
// TODO: docs
// TODO: serialization

// It would be awesome to provide no_std support. Math isn't stable in core
// and the intrinsics required for ln and powi are nightly only.
//
// https://github.com/rust-lang/rfcs/issues/2505

#[derive(Debug)]
pub enum Error {
    AllocError,
    InvalidRegisterWidth,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::AllocError => write!(f, "allocation failed"),
            Error::InvalidRegisterWidth => {
                write!(f, "invalid register width: registers must be between 1 and 8 bits wide")
            }
        }
    }
}

// A toy implementation of HyperLogLog.
//
// This implementation is BYO Hash Function - callers should add data to the
// HLL using the [`iter`] method. Data should already be a 64-bit value
// drawn from a uniform distribution (read: hashed well).
pub struct HLL {
    m: usize,
    b: usize,
    registers: Registers,
}

impl HLL {
    // Create a new HLL with the given register width and log2m set to the
    // given value. log2m must not be zero.
    pub fn new(log2m: usize, register_width: usize) -> Result<HLL, Error> {
        let b = log2m;
        let m = 1 << b;
        let registers = Registers::alloc(register_width, m)?;

        Ok(HLL { m, b, registers })
    }

    // The number of registers used in this HLL.
    #[inline]
    pub fn m(&self) -> usize {
        self.m
    }

    // The width of the registers in this HLL.
    #[inline]
    pub fn register_width(&self) -> usize {
        self.registers.width
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

    // Returns an estimate of the cardinality of the multiset.
    pub fn cardinality(&self) -> f64 {
        match self.estimator_and_zeros() {
            (e, z) if z > 0 && e <= HLL::small_estimator_cutoff(self.m) => {
                HLL::small_estimator(self.m, z)
            }
            (e, _) if e > HLL::large_estimator_cutoff(self.registers.width, self.b) => {
                HLL::large_estimator(self.registers.width, self.b, e)
            }
            (e, _) => e,
        }
    }

    // Union another HLL value into this one. This is equivalent to setting
    // every register in this HLL to the max of it's current value and the
    // corresponding register in the other HLL.
    //
    // Does not validate that the two HLLs are compatible, and will panic if
    // other has a higher log2m value than self.
    pub fn union(&mut self, other: &Self) -> Result<(), ()> {
        if self.b != other.b || self.registers.width != other.registers.width {
            return Err(());
        }

        for (i, v) in other.registers.iter().enumerate() {
            self.registers.set_max(i, v);
        }

        Ok(())
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

        writeln!(
            f,
            "HLL {{ m={}, b={}, reg_width={}, zeros={}, estimator={} }}",
            self.m, self.b, self.registers.width, zeros, estimator
        )
    }
}

// TODO: prop tests for unions
//
// TODO: prop tests for cardinality estimates. these are hard because they can
//       flake, and that's part of the error guarantee. the error bounds are
//       probabilistic!

#[cfg(test)]
mod test_hll {
    use super::*;

    #[test]
    fn test_union_set_max() {
        let mut h1 = HLL { b: 2, m: 4, registers: Registers::from_iter(4, 4, vec![3, 1, 1, 3]) };
        let h2 = HLL { b: 2, m: 4, registers: Registers::from_iter(4, 4, vec![2, 2, 2, 2]) };

        h1.union(&h2).expect("union should be ok");
        assert_eq!(
            h1.registers.iter().collect::<Vec<u8>>(),
            vec![3, 2, 2, 3],
            "unioned registers should be the pairwise max of registers in both HLLs"
        );
    }

    #[test]
    fn test_union_incompatible() {
        let mut h1 = HLL::new(4, 5).unwrap();
        let mut h2 = HLL::new(6, 7).unwrap();

        assert_eq!(Err(()), h1.union(&h2),);
        assert_eq!(Err(()), h2.union(&h1),);
    }
}

// A fixed-size array of n-bit-wide registers
//
// Registers and RegisterIters are unsafe and reference raw
struct Registers {
    mem: *mut u8,
    width: usize,
    len: usize,
    mask: u8,
}

impl Registers {
    const MIN_WIDTH: usize = 1;
    const MAX_WIDTH: usize = 8;

    fn alloc(width: usize, len: usize) -> Result<Registers, Error> {
        if !(Self::MIN_WIDTH..=Self::MAX_WIDTH).contains(&width) {
            return Err(Error::InvalidRegisterWidth);
        }
        let mask = Self::mask(width);

        let mem = unsafe { alloc_zeroed(Registers::layout(width, len)) };
        if mem.is_null() {
            return Err(Error::AllocError);
        }

        Ok(Registers { mem, width, len, mask })
    }

    // TODO: bench this vs `if get(i) < v { set(i, v) }`
    fn set_max(&mut self, i: usize, v: u8) {
        let v = v & self.mask;
        let bits = self.width * i;
        let low_idx = bits / 8;
        let high_idx = (bits + self.width - 1) / 8;
        let remainder = bits % 8;

        let low_byte = unsafe { *self.mem.add(low_idx) };
        let high_byte = unsafe { *self.mem.add(high_idx) };

        let current_v = ((low_byte >> remainder)
            | high_byte.checked_shl(8 - remainder as u32).unwrap_or(0))
            & self.mask;

        if current_v < v {
            unsafe {
                *self.mem.add(low_idx) &= !(self.mask << remainder);
                *self.mem.add(low_idx) |= v << remainder;

                *self.mem.add(high_idx) &=
                    !(self.mask.checked_shr(8 - remainder as u32).unwrap_or(0));
                *self.mem.add(high_idx) |= v.checked_shr(8 - remainder as u32).unwrap_or(0);
            }
        }
    }

    fn get(&self, i: usize) -> u8 {
        let bits = self.width * i;
        let low_idx = bits / 8;
        let high_idx = (bits + self.width - 1) / 8;
        let remainder = bits % 8;

        let low_byte = unsafe { *self.mem.add(low_idx) };
        let high_byte = unsafe { *self.mem.add(high_idx) };

        ((low_byte >> remainder) | high_byte.checked_shl(8 - remainder as u32).unwrap_or(0))
            & self.mask
    }

    fn iter(&self) -> RegisterIterator {
        RegisterIterator { registers: self, idx: 0 }
    }

    #[inline]
    fn layout(width: usize, len: usize) -> Layout {
        let total_bits = width * len;
        let bytes = (total_bits / 8) + if (total_bits % 8) == 0 { 0 } else { 1 };
        Layout::from_size_align(bytes, 1).expect("invalid register layout")
    }

    #[inline]
    fn mask(width: usize) -> u8 {
        (((1 as u64) << width) - 1) as u8
    }
}

#[cfg(test)]
impl Registers {
    fn from_iter<I>(width: usize, len: usize, items: I) -> Registers
    where
        I: IntoIterator,
        <I as IntoIterator>::Item: Into<u8>,
    {
        let mut registers = Registers::alloc(width, len).unwrap();

        for (i, val) in items.into_iter().enumerate() {
            registers.set(i, val.into());
        }

        registers
    }

    fn set(&mut self, i: usize, v: u8) {
        let v = v & self.mask;
        let bits = self.width * i;
        let low_idx = bits / 8;
        let high_idx = (bits + self.width - 1) / 8;
        let remainder = bits % 8;

        unsafe {
            *self.mem.add(low_idx) &= !(self.mask << remainder);
            *self.mem.add(low_idx) |= v << remainder;

            *self.mem.add(high_idx) &= !(self.mask.checked_shr(8 - remainder as u32).unwrap_or(0));
            *self.mem.add(high_idx) |= v.checked_shr(8 - remainder as u32).unwrap_or(0);
        }
    }
}

impl Drop for Registers {
    fn drop(&mut self) {
        let layout = Registers::layout(self.width, self.len);
        unsafe {
            dealloc(self.mem, layout);
        }
    }
}

// An iterator that yields the current value of dense HLL registers. This
// `struct` is created by the [`iter`] method on [`Registers`]. See its
// documentation for more details.
struct RegisterIterator<'a> {
    registers: &'a Registers,
    idx: usize,
}

impl<'a> Iterator for RegisterIterator<'a> {
    type Item = u8;

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx >= self.registers.len {
            None
        } else {
            let v = self.registers.get(self.idx);
            self.idx += 1;
            Some(v)
        }
    }
}

#[cfg(test)]
mod test_registers {
    use super::*;
    use quickcheck::{self, empty_shrinker, single_shrinker, Arbitrary, Gen};
    use quickcheck_macros::quickcheck;
    use rand::Rng;

    #[derive(Debug, Clone, Copy)]
    struct RegisterSize {
        width: usize,
        len: usize,
    }

    impl RegisterSize {
        fn registers(self) -> Result<Registers, Error> {
            Registers::alloc(self.width, self.len)
        }
    }

    impl Arbitrary for RegisterSize {
        fn arbitrary<G: Gen>(g: &mut G) -> RegisterSize {
            RegisterSize {
                width: g.gen_range(Registers::MIN_WIDTH, Registers::MAX_WIDTH),
                len: g.size(),
            }
        }

        fn shrink(&self) -> Box<dyn Iterator<Item = Self>> {
            if self.len <= 2 {
                empty_shrinker()
            } else {
                single_shrinker(RegisterSize { width: self.width, len: self.len / 2 })
            }
        }
    }

    #[test]
    fn test_set_get() {
        let tcs = vec![
            // inside one byte
            (0, 4, 0b0000000000000100, vec![4, 0, 0, 0]),
            (0, 31, 0b0000000000011111, vec![31, 0, 0, 0]),
            // spans two bytes
            (1, 4, 0b0000000010000000, vec![0, 4, 0, 0]),
            (1, 31, 0b0000001111100000, vec![0, 31, 0, 0]),
        ];

        for (idx, val, expected_bits, expected_registers) in tcs {
            let mut rs = RegisterSize { len: 4, width: 5 }.registers().unwrap();
            rs.set(idx, val);

            let bits = {
                let b1 = unsafe { *rs.mem } as u16;
                let b2 = unsafe { *rs.mem.add(1) } as u16;
                (b2 << 8) | b1
            };

            assert_eq!(
                expected_bits, bits,
                "i={}, v={}: expected bits: {:016b}, actual bits: {:016b}",
                idx, val, expected_bits, bits
            );

            let registers = rs.iter().collect::<Vec<u8>>();
            assert_eq!(
                expected_registers, registers,
                "i={}, v={}: expected registers don't match",
                idx, val
            );
        }
    }

    #[test]
    fn test_set_max() {
        let mut rs = RegisterSize { len: 4, width: 5 }.registers().unwrap();

        println!("{:?}", rs.iter().collect::<Vec<u8>>());
        for (idx, val) in vec![(1, 1), (1, 3), (1, 2)] {
            rs.set(idx, val);
            println!("{:?}", rs.iter().collect::<Vec<u8>>());
        }
        assert_eq!(rs.iter().collect::<Vec<u8>>(), vec![0, 2, 0, 0]);

        // for (idx, val) in vec![(0, 3), (0, 1)] {
        //     rs.set_max(idx, val);
        // }
        // assert_eq!(rs.iter().collect::<Vec<u8>>(), vec![3, 3, 0, 0]);
    }

    #[quickcheck]
    fn check_layout(rs: RegisterSize) -> bool {
        let layout = Registers::layout(rs.width, rs.len);

        let total_bits = rs.width * rs.len;
        let whole_bytes = total_bits / 8;
        let partial_bytes = if total_bits - (whole_bytes * 8) > 0 { 1 } else { 0 };
        let bytes = whole_bytes + partial_bytes;

        layout.size() == bytes
    }

    #[quickcheck]
    fn check_init_zeroed(rs: RegisterSize) -> bool {
        let rs = rs.registers().unwrap();

        (0..rs.len).map(|i| rs.get(i)).all(|v| v == 0)
    }

    #[quickcheck]
    fn check_iter_length(rs: RegisterSize) -> bool {
        rs.registers().unwrap().iter().count() == rs.len
    }

    #[quickcheck]
    fn check_set_odd_even(rs: RegisterSize) -> bool {
        let mut rs = rs.registers().unwrap();
        let val = 0b10101010 & Registers::mask(rs.width);

        for i in (0..rs.len).filter(|i| i % 2 == 0) {
            rs.set(i, val);
        }

        let all_odds_zero = rs.iter().enumerate().filter(|&(i, _)| i % 2 == 1).all(|(_, v)| v == 0);
        let all_evens_v = rs.iter().enumerate().filter(|&(i, _)| i % 2 == 0).all(|(_, v)| v == val);

        all_odds_zero && all_evens_v
    }

    #[quickcheck]
    fn check_set_max(rs: RegisterSize) -> bool {
        let mut rs = rs.registers().unwrap();
        let low_v = 1;
        let high_v = (1 << rs.width) - 1;

        for i in 0..rs.len {
            rs.set_max(i, low_v);
            rs.set_max(i, high_v);
            rs.set_max(i, low_v);
        }

        rs.iter().all(|v| v == high_v)
    }
}
