//! Manual complex number arithmetic — zero allocations, fully inlinable.
//!
//! We avoid `num_complex::Complex64` which boxes and allocates unnecessarily
//! for our tight inner loops. Instead we work directly with (f64, f64) pairs
//! and write the arithmetic manually. The compiler can then auto-vectorize
//! these operations using SIMD registers.

#![allow(dead_code)]

/// A plain-old-data complex number stored as two f64 values.
/// This is the zero-cost abstraction we use in the hot path.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct C64(pub f64, pub f64);

impl C64 {
    /// Create a new complex number `re + i·im`.
    #[inline(always)]
    pub const fn new(re: f64, im: f64) -> Self {
        Self(re, im)
    }

    /// Real part
    #[inline(always)]
    pub const fn re(&self) -> f64 {
        self.0
    }

    /// Imaginary part
    #[inline(always)]
    pub const fn im(&self) -> f64 {
        self.1
    }

    /// Squared magnitude: |z|² = re² + im²
    #[inline(always)]
    pub fn norm_sqr(self) -> f64 {
        self.0 * self.0 + self.1 * self.1
    }

    /// Multiply by a real scalar
    #[inline(always)]
    pub fn scale(self, s: f64) -> Self {
        Self(self.0 * s, self.1 * s)
    }

    /// Complex multiplication: (a + ib)(c + id) = (ac - bd) + i(ad + bc)
    #[inline(always)]
    pub fn mul(self, other: Self) -> Self {
        Self(
            self.0 * other.0 - self.1 * other.1,
            self.0 * other.1 + self.1 * other.0,
        )
    }

    /// Complex addition
    #[inline(always)]
    pub fn add(self, other: Self) -> Self {
        Self(self.0 + other.0, self.1 + other.1)
    }

    /// Complex subtraction
    #[inline(always)]
    pub fn sub(self, other: Self) -> Self {
        Self(self.0 - other.0, self.1 - other.1)
    }

    /// Multiplicative inverse: 1 / (a + ib) = (a - ib) / (a² + b²)
    #[inline(always)]
    pub fn inv(self) -> Self {
        let n = self.norm_sqr();
        if n > 0.0 {
            Self(self.0 / n, -self.1 / n)
        } else {
            Self(0.0, 0.0)
        }
    }
}

impl std::ops::Add for C64 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        self.add(rhs)
    }
}

impl std::ops::Sub for C64 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        self.sub(rhs)
    }
}

impl std::ops::Mul for C64 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        self.mul(rhs)
    }
}

/// Compute 1 / ((λᵢ - λⱼ) - iη) using manual complex arithmetic.
///
/// This is the core term of the Stieltjes transform.
/// Returns a `C64` value.
#[inline(always)]
pub fn stieltjes_term_c64(lambda_i: f64, lambda_j: f64, eta: f64) -> C64 {
    let diff = lambda_i - lambda_j;
    let denom = diff * diff + eta * eta;
    let inv_denom = 1.0 / denom;
    C64(diff * inv_denom, eta * inv_denom)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_c64_arithmetic() {
        let a = C64::new(3.0, 4.0);
        let b = C64::new(1.0, 2.0);

        // addition
        let sum = a + b;
        assert!((sum.re() - 4.0).abs() < 1e-15);
        assert!((sum.im() - 6.0).abs() < 1e-15);

        // subtraction
        let diff = a - b;
        assert!((diff.re() - 2.0).abs() < 1e-15);
        assert!((diff.im() - 2.0).abs() < 1e-15);

        // multiplication: (3+4i)(1+2i) = -5 + 10i
        let prod = a * b;
        assert!((prod.re() - (-5.0)).abs() < 1e-14);
        assert!((prod.im() - 10.0).abs() < 1e-14);

        // norm_sqr: |3+4i|² = 25
        assert!((a.norm_sqr() - 25.0).abs() < 1e-14);

        // inv: 1/(3+4i) = 0.12 - 0.16i
        let inv = a.inv();
        assert!((inv.re() - 0.12).abs() < 1e-15);
        assert!((inv.im() - (-0.16)).abs() < 1e-15);
    }

    #[test]
    fn test_stieltjes_term() {
        // 1 / ((2.0 - 1.0) - i·0.1) = 1 / (1 - 0.1i)
        let t = stieltjes_term_c64(2.0, 1.0, 0.1);
        // 1 / ((2-1) - i·0.1) = 1 / (1 - 0.1i) = (1 + 0.1i) / 1.01 ≈ (0.990099, 0.0990099)
        let expected_re = 1.0 / 1.01;
        let expected_im = 0.1 / 1.01;
        assert!((t.re() - expected_re).abs() < 1e-15);
        assert!((t.im() - expected_im).abs() < 1e-15);
    }
}
