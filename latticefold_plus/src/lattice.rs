use std::ops::{Add, Mul, Sub};
use num_bigint::BigInt;
use num_traits::{One, Zero};
use rand::{CryptoRng, RngCore};
use zeroize::Zeroize;

#[derive(Clone, Debug, Zeroize)]
pub struct LatticePoint {
    coordinates: Vec<i64>,
    dimension: usize,
}

#[derive(Clone, Debug)]
pub struct LatticeParams {
    pub q: i64,           // Modulus
    pub n: usize,         // Dimension
    pub sigma: f64,       // Gaussian parameter
    pub beta: f64,        // Smoothing parameter
}

impl LatticePoint {
    pub fn new(coords: Vec<i64>) -> Self {
        let dimension = coords.len();
        Self {
            coordinates: coords,
            dimension,
        }
    }

    pub fn random<R: RngCore + CryptoRng>(params: &LatticeParams, rng: &mut R) -> Self {
        let coords: Vec<i64> = (0..params.n)
            .map(|_| {
                let val = rng.next_u64() as i64;
                val % params.q
            })
            .collect();
        Self::new(coords)
    }

    pub fn add_mod(&self, other: &Self, q: i64) -> Self {
        assert_eq!(self.dimension, other.dimension);
        let coords: Vec<i64> = self.coordinates
            .iter()
            .zip(other.coordinates.iter())
            .map(|(&x, &y)| {
                let sum = x + y;
                ((sum % q) + q) % q
            })
            .collect();
        Self::new(coords)
    }

    pub fn sub_mod(&self, other: &Self, q: i64) -> Self {
        assert_eq!(self.dimension, other.dimension);
        let coords: Vec<i64> = self.coordinates
            .iter()
            .zip(other.coordinates.iter())
            .map(|(&x, &y)| {
                let diff = x - y;
                ((diff % q) + q) % q
            })
            .collect();
        Self::new(coords)
    }

    pub fn scalar_mul_mod(&self, scalar: i64, q: i64) -> Self {
        let coords: Vec<i64> = self.coordinates
            .iter()
            .map(|&x| {
                let prod = x * scalar;
                ((prod % q) + q) % q
            })
            .collect();
        Self::new(coords)
    }

    pub fn inner_product_mod(&self, other: &Self, q: i64) -> i64 {
        assert_eq!(self.dimension, other.dimension);
        self.coordinates
            .iter()
            .zip(other.coordinates.iter())
            .map(|(&x, &y)| (x * y) % q)
            .sum::<i64>()
            .rem_euclid(q)
    }

    pub fn norm_squared(&self) -> i64 {
        self.coordinates
            .iter()
            .map(|&x| x * x)
            .sum()
    }

    pub fn is_short(&self, bound: f64) -> bool {
        (self.norm_squared() as f64).sqrt() <= bound
    }
}

impl Add for LatticePoint {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        assert_eq!(self.dimension, other.dimension);
        let coords: Vec<i64> = self.coordinates
            .iter()
            .zip(other.coordinates.iter())
            .map(|(&x, &y)| x + y)
            .collect();
        Self::new(coords)
    }
}

impl Sub for LatticePoint {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        assert_eq!(self.dimension, other.dimension);
        let coords: Vec<i64> = self.coordinates
            .iter()
            .zip(other.coordinates.iter())
            .map(|(&x, &y)| x - y)
            .collect();
        Self::new(coords)
    }
}

impl Mul<i64> for LatticePoint {
    type Output = Self;

    fn mul(self, scalar: i64) -> Self {
        let coords: Vec<i64> = self.coordinates
            .iter()
            .map(|&x| x * scalar)
            .collect();
        Self::new(coords)
    }
}

#[derive(Clone, Debug)]
pub struct LatticeMatrix {
    data: Vec<Vec<i64>>,
    rows: usize,
    cols: usize,
}

impl LatticeMatrix {
    pub fn new(data: Vec<Vec<i64>>) -> Self {
        let rows = data.len();
        let cols = if rows > 0 { data[0].len() } else { 0 };
        Self { data, rows, cols }
    }

    pub fn random<R: RngCore + CryptoRng>(
        rows: usize,
        cols: usize,
        params: &LatticeParams,
        rng: &mut R,
    ) -> Self {
        let data: Vec<Vec<i64>> = (0..rows)
            .map(|_| {
                (0..cols)
                    .map(|_| {
                        let val = rng.next_u64() as i64;
                        val % params.q
                    })
                    .collect()
            })
            .collect();
        Self::new(data)
    }

    pub fn mul_vector_mod(&self, vec: &LatticePoint, q: i64) -> LatticePoint {
        assert_eq!(self.cols, vec.dimension);
        let result: Vec<i64> = self.data
            .iter()
            .map(|row| {
                row.iter()
                    .zip(vec.coordinates.iter())
                    .map(|(&a, &b)| (a * b) % q)
                    .sum::<i64>()
                    .rem_euclid(q)
            })
            .collect();
        LatticePoint::new(result)
    }

    pub fn gram_schmidt(&self) -> Self {
        let mut orthogonal = self.data.clone();
        let mut coefficients = vec![vec![0.0; self.rows]; self.rows];

        for i in 0..self.rows {
            for j in 0..i {
                let mut numerator = 0.0;
                let mut denominator = 0.0;

                for k in 0..self.cols {
                    numerator += orthogonal[i][k] as f64 * orthogonal[j][k] as f64;
                    denominator += orthogonal[j][k] as f64 * orthogonal[j][k] as f64;
                }

                coefficients[i][j] = numerator / denominator;

                for k in 0..self.cols {
                    orthogonal[i][k] = (orthogonal[i][k] as f64 
                        - coefficients[i][j] * orthogonal[j][k] as f64) as i64;
                }
            }
        }

        Self::new(orthogonal)
    }
}
