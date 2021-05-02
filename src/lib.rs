use std::alloc::{alloc, alloc_zeroed, dealloc, handle_alloc_error, Layout};

// TODO: docs
// TODO: serialization

// It would be awesome to provide no_std support. Math isn't stable in core
// and the intrinsics required for ln and powi are nightly only.
//
// https://github.com/rust-lang/rfcs/issues/2505

// A toy implementation of HyperLogLog.
//
// This implementation is BYO Hash Function - callers should add data to the
// HLL using the [`iter`] method. Data should already be a 64-bit value
// drawn from a uniform distribution (read: hashed well).
#[allow(clippy::upper_case_acronyms)]
pub struct HLL<const W: usize, const B: usize> {
    registers: Registers<W>,
}

impl<const W: usize, const B: usize> HLL<W, B> {
    // Create a new HLL with the given register width and log2m set to the
    // given value. log2m must not be zero.
    #[allow(clippy::new_without_default)]
    pub fn new() -> HLL<W, B> {
        let registers = Registers::alloc(Self::m());
        HLL { registers }
    }

    // The number of registers used in this HLL.
    #[inline]
    pub const fn m() -> usize {
        1 << B
    }

    // The width of the registers in this HLL.
    #[inline]
    pub const fn register_width(&self) -> usize {
        W
    }

    #[inline]
    pub const fn register_count(&self) -> usize {
        Self::m()
    }

    // Add a raw value to the multiset. The value MUST have been hashed or
    // drawn from a uniform distribution.
    pub fn add_raw(&mut self, value: u64) {
        let j = (value as usize) & (Self::m() - 1);
        let w = value >> B;

        // NOTE: the paper defines p(0^k) == k + 1 but 0.trailing_zeros() == 0
        // so we have to correct here
        let p_w = 1 + {
            if value == 0 {
                (32 - B) + 1
            } else {
                w.trailing_zeros() as usize
            }
        };

        self.registers.set_max(j, p_w as u8);
    }

    // Returns an estimate of the cardinality of the multiset.
    pub fn cardinality(&self) -> f64 {
        match self.estimator_and_zeros() {
            (e, z) if z > 0 && e <= Self::small_estimator_cutoff() => Self::small_estimator(z),
            (e, _) if e > Self::large_estimator_cutoff() => Self::large_estimator(e),
            (e, _) => e,
        }
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

    // NOTE: the large/small estimators and cutoffs can be consts once
    //       floating point const arithmetic stabilizes.
    //
    //       see: https://github.com/rust-lang/rust/issues/57241

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

        (Self::alpha_m_squared() / sum, zeros)
    }

    #[inline]
    fn alpha_m_squared() -> f64 {
        let mf = Self::m() as f64;
        let alpha = match Self::m() {
            16 => 0.673,
            32 => 0.697,
            64 => 0.709,
            _ => 0.7213 / (1.0 + 1.079 / mf),
        };
        alpha * mf * mf
    }

    #[inline]
    fn small_estimator_cutoff() -> f64 {
        (5.0 / 2.0) * (Self::m() as f64)
    }

    #[inline]
    fn small_estimator(zeros: usize) -> f64 {
        let m = Self::m() as f64;
        let zeros = zeros as f64;
        m * (m / zeros).ln()
    }

    #[inline]
    fn large_estimator_cutoff() -> f64 {
        Self::two_to_l() / 30.0
    }

    #[inline]
    fn large_estimator(est: f64) -> f64 {
        let ttl = Self::two_to_l();
        -1.0 * ttl * (1.0 - est / ttl).ln()
    }

    #[inline]
    fn two_to_l() -> f64 {
        // this needs to be -2 instead of -1 to account for the fact that
        // p_w(0) = 1 and not 0
        let max_register_val = (1 << W) - 1 - 1;
        (2.0_f64).powi((max_register_val + B) as i32)
    }
}

impl<const W: usize, const B: usize> std::fmt::Debug for HLL<W, B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (estimator, zeros) = self.estimator_and_zeros();

        writeln!(
            f,
            "HLL {{ m={}, b={}, reg_width={}, zeros={}, estimator={} }}",
            Self::m(),
            B,
            W,
            zeros,
            estimator
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
        let mut h1 = HLL::<4, 2> { registers: Registers::<4>::from_iter(4, vec![3, 1, 1, 3]) };
        let h2 = HLL::<4, 2> { registers: Registers::<4>::from_iter(4, vec![2, 2, 2, 2]) };

        h1.union(&h2);
        assert_eq!(
            h1.registers.iter().collect::<Vec<u8>>(),
            vec![3, 2, 2, 3],
            "unioned registers should be the pairwise max of registers in both HLLs"
        );
    }
}

// A fixed-size array of n-bit-wide registers used to back an HLL.
//
// Registers and RegisterIters allocate reference raw
struct Registers<const N: usize> {
    mem: *mut u8,
    len: usize,
}

impl<const N: usize> Clone for Registers<N> {
    fn clone(&self) -> Self {
        let layout = Self::layout(self.len);

        let mem = unsafe {
            let dst = alloc(layout);
            if dst.is_null() {
                handle_alloc_error(layout);
            }

            self.mem.copy_to_nonoverlapping(dst, layout.size());

            dst
        };

        Registers { mem, ..*self }
    }
}

impl<const N: usize> Drop for Registers<N> {
    fn drop(&mut self) {
        let layout = Self::layout(self.len);
        unsafe {
            dealloc(self.mem, layout);
        }
    }
}

impl<const N: usize> Registers<N> {
    const MIN_WIDTH: usize = 1;
    const MAX_WIDTH: usize = 8;

    fn alloc(len: usize) -> Registers<N> {
        assert!(
            Self::MIN_WIDTH <= N && N <= Self::MAX_WIDTH,
            "registers: invalid register width: {}",
            N
        );

        let layout = Self::layout(len);
        let mem = unsafe { alloc_zeroed(layout) };
        if mem.is_null() {
            handle_alloc_error(layout)
        }

        Registers { mem, len }
    }

    const MASK: u8 = ((1u64 << N) - 1) as u8;

    // TODO: bench this vs `if get(i) < v { set(i, v) }`
    fn set_max(&mut self, i: usize, v: u8) {
        let v = v & (Self::MASK as u8);
        let bits = N * i;
        let low_idx = bits / 8;
        let high_idx = (bits + N - 1) / 8;
        let remainder = bits % 8;

        let low_byte = unsafe { *self.mem.add(low_idx) };
        let high_byte = unsafe { *self.mem.add(high_idx) };

        let current_v = ((low_byte >> remainder)
            | high_byte.checked_shl(8 - remainder as u32).unwrap_or(0))
            & Self::MASK;

        if current_v < v {
            unsafe {
                *self.mem.add(low_idx) &= !(Self::MASK << remainder);
                *self.mem.add(low_idx) |= v << remainder;

                *self.mem.add(high_idx) &=
                    !(Self::MASK.checked_shr(8 - remainder as u32).unwrap_or(0));
                *self.mem.add(high_idx) |= v.checked_shr(8 - remainder as u32).unwrap_or(0);
            }
        }
    }

    fn get(&self, i: usize) -> u8 {
        let bits = N * i;
        let low_idx = bits / 8;
        let high_idx = (bits + N - 1) / 8;
        let remainder = bits % 8;

        let low_byte = unsafe { *self.mem.add(low_idx) };
        let high_byte = unsafe { *self.mem.add(high_idx) };

        ((low_byte >> remainder) | high_byte.checked_shl(8 - remainder as u32).unwrap_or(0))
            & Self::MASK
    }

    fn iter(&self) -> RegisterIterator<N> {
        RegisterIterator { registers: self, idx: 0 }
    }

    #[inline]
    fn layout(len: usize) -> Layout {
        let total_bits = N * len;
        let bytes = (total_bits / 8) + if (total_bits % 8) == 0 { 0 } else { 1 };
        Layout::from_size_align(bytes, 1).expect("invalid register layout")
    }
}

#[cfg(test)]
impl<const N: usize> Registers<N> {
    fn from_iter<I>(len: usize, items: I) -> Registers<N>
    where
        I: IntoIterator,
        <I as IntoIterator>::Item: Into<u8>,
    {
        let mut registers = Registers::alloc(len);

        for (i, val) in items.into_iter().enumerate() {
            registers.set(i, val.into());
        }

        registers
    }

    fn set(&mut self, i: usize, v: u8) {
        let v = v & Self::MASK;
        let bits = N * i;
        let low_idx = bits / 8;
        let high_idx = (bits + N - 1) / 8;
        let remainder = bits % 8;

        unsafe {
            *self.mem.add(low_idx) &= !(Self::MASK << remainder);
            *self.mem.add(low_idx) |= v << remainder;

            *self.mem.add(high_idx) &= !(Self::MASK.checked_shr(8 - remainder as u32).unwrap_or(0));
            *self.mem.add(high_idx) |= v.checked_shr(8 - remainder as u32).unwrap_or(0);
        }
    }
}

// An iterator that yields the current value of dense HLL registers. This
// `struct` is created by the [`iter`] method on [`Registers`]. See its
// documentation for more details.
struct RegisterIterator<'a, const N: usize> {
    registers: &'a Registers<N>,
    idx: usize,
}

impl<'a, const N: usize> Iterator for RegisterIterator<'a, N> {
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
    use quickcheck_macros::quickcheck;

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
            let mut rs = Registers::<5>::alloc(4);
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
        let mut rs = Registers::<5>::alloc(4);

        // set should reflect the most recent value
        rs.set(1, 4);
        rs.set(1, 3);
        rs.set(1, 2);
        assert_eq!(rs.iter().collect::<Vec<u8>>(), vec![0, 2, 0, 0]);

        // set max should reflect the max value
        rs.set_max(0, 2);
        rs.set_max(0, 3);
        rs.set_max(0, 1);
        rs.set_max(0, 0);
        assert_eq!(rs.iter().collect::<Vec<u8>>(), vec![3, 2, 0, 0]);
    }

    #[quickcheck]
    fn check_layout(len: u16) -> bool {
        let len = len as usize;
        let layout = Registers::<3>::layout(len);

        let total_bits = 3 * len;
        let whole_bytes = total_bits / 8;
        let partial_bytes = if total_bits - (whole_bytes * 8) > 0 { 1 } else { 0 };
        let bytes = whole_bytes + partial_bytes;

        layout.size() == bytes
    }

    macro_rules! check_zeroed {
        ($n:tt, $len:expr) => {{
            let rs = Registers::<$n>::alloc($len);
            (0..rs.len).map(|i| rs.get(i)).all(|v| v == 0)
        }};
    }

    #[quickcheck]
    fn check_init_zeroed(len: u16) -> bool {
        let len = len as usize;
        [
            check_zeroed!(1, len),
            check_zeroed!(2, len),
            check_zeroed!(3, len),
            check_zeroed!(4, len),
            check_zeroed!(5, len),
            check_zeroed!(6, len),
            check_zeroed!(7, len),
            check_zeroed!(8, len),
        ]
        .iter()
        .all(|v| *v)
    }

    macro_rules! check_iter_len {
        ($n:tt, $len:expr) => {{
            let rs = Registers::<$n>::alloc($len);
            rs.iter().count() == rs.len
        }};
    }

    #[quickcheck]
    fn check_iter_length(len: u16) -> bool {
        let len = len as usize;
        [
            check_iter_len!(1, len),
            check_iter_len!(2, len),
            check_iter_len!(3, len),
            check_iter_len!(4, len),
            check_iter_len!(5, len),
            check_iter_len!(6, len),
            check_iter_len!(7, len),
            check_iter_len!(8, len),
        ]
        .iter()
        .all(|v| *v)
    }

    macro_rules! check_set_max {
        ($n:tt, $len:expr) => {{
            let mut rs = Registers::<$n>::alloc($len);
            let low_v = 1;
            let high_v = (1 << $n) - 1;

            for i in 0..rs.len {
                rs.set_max(i, low_v);
                rs.set_max(i, high_v);
                rs.set_max(i, low_v);
            }

            rs.iter().all(|v| v == high_v)
        }};
    }

    #[quickcheck]
    fn check_set_max(len: u16) -> bool {
        let len = len as usize;
        [
            check_set_max!(1, len),
            check_set_max!(2, len),
            check_set_max!(3, len),
            check_set_max!(4, len),
            check_set_max!(5, len),
        ]
        .iter()
        .all(|&v| v)
    }
}
