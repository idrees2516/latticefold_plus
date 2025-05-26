use ark_ff::{Field, PrimeField};
use ark_std::{rand::Rng, UniformRand};
use std::ops::{Add, Mul, Sub};

#[derive(Clone, Debug)]
pub struct LatticeBasis<F: Field> {
    pub basis: Vec<Vec<F>>,
    pub dimension: usize,
}

impl<F: Field> LatticeBasis<F> {
    pub fn new(dimension: usize) -> Self {
        let mut basis = Vec::with_capacity(dimension);
        for i in 0..dimension {
            let mut row = vec![F::zero(); dimension];
            row[i] = F::one();
            basis.push(row);
        }
        Self { basis, dimension }
    }

    pub fn gram_schmidt(&self) -> Vec<Vec<F>> {
        let mut orthogonal = self.basis.clone();
        let mut coefficients = vec![vec![F::zero(); self.dimension]; self.dimension];

        for i in 0..self.dimension {
            for j in 0..i {
                let dot = self.dot_product(&orthogonal[i], &orthogonal[j]);
                let norm = self.dot_product(&orthogonal[j], &orthogonal[j]);
                coefficients[i][j] = dot / norm;
                
                for k in 0..self.dimension {
                    orthogonal[i][k] = orthogonal[i][k] - coefficients[i][j] * orthogonal[j][k];
                }
            }
        }

        orthogonal
    }

    fn dot_product(&self, a: &[F], b: &[F]) -> F {
        a.iter().zip(b.iter()).map(|(x, y)| *x * *y).sum()
    }
}

#[derive(Clone, Debug)]
pub struct LatticePoint<F: Field> {
    pub coordinates: Vec<F>,
    pub basis: LatticeBasis<F>,
}

impl<F: Field> LatticePoint<F> {
    pub fn new(coordinates: Vec<F>, basis: LatticeBasis<F>) -> Self {
        assert_eq!(coordinates.len(), basis.dimension);
        Self { coordinates, basis }
    }

    pub fn random<R: Rng>(basis: &LatticeBasis<F>, rng: &mut R) -> Self {
        let coordinates = (0..basis.dimension)
            .map(|_| F::rand(rng))
            .collect();
        Self::new(coordinates, basis.clone())
    }

    pub fn closest_vector(&self) -> Self {
        let orthogonal = self.basis.gram_schmidt();
        let mut closest = vec![F::zero(); self.basis.dimension];

        for i in 0..self.basis.dimension {
            let mut projection = F::zero();
            for j in 0..self.basis.dimension {
                projection += self.coordinates[j] * orthogonal[i][j];
            }
            closest[i] = projection;
        }

        Self::new(closest, self.basis.clone())
    }

    pub fn distance(&self, other: &Self) -> F {
        self.coordinates
            .iter()
            .zip(other.coordinates.iter())
            .map(|(a, b)| (*a - *b) * (*a - *b))
            .sum()
    }

    pub fn scalar_mul_mod(&self, scalar: i64, q: i64) -> Self {
        let coords: Vec<i64> = self.coordinates
            .iter()
            .map(|&x| {
                let product = x * scalar;
                ((product % q) + q) % q
            })
            .collect();
        Self::new(coords.iter().map(|&x| F::from(x)).collect(), self.basis.clone())
    }

    pub fn dot_product(&self, other: &Self) -> i64 {
        assert_eq!(self.basis.dimension, other.basis.dimension);
        self.coordinates
            .iter()
            .zip(other.coordinates.iter())
            .map(|(&x, &y)| x * y)
            .sum()
    }

    pub fn norm_squared(&self) -> i64 {
        self.dot_product(self)
    }

    pub fn gram_schmidt(&self, basis: &[Self]) -> Self {
        let mut orthogonal = self.clone();
        
        for b in basis {
            let dot = self.dot_product(b);
            let norm = b.norm_squared();
            if norm != 0 {
                let coef = dot / norm;
                orthogonal = orthogonal.sub_mod(&b.scalar_mul_mod(coef, self.basis.dimension as i64), self.basis.dimension as i64);
            }
        }
        
        orthogonal
    }

    pub fn closest_vector(&self, basis: &[Self]) -> Self {
        let mut closest = self.clone();
        let mut min_distance = self.norm_squared();
        
        for b in basis {
            let projection = self.dot_product(b) / b.norm_squared();
            let rounded = projection.round() as i64;
            let candidate = b.scalar_mul_mod(rounded, self.basis.dimension as i64);
            let distance = self.sub_mod(&candidate, self.basis.dimension as i64).norm_squared();
            
            if distance < min_distance {
                min_distance = distance;
                closest = candidate;
            }
        }
        
        closest
    }

    pub fn is_in_lattice(&self, basis: &[Self], q: i64) -> bool {
        let mut coefficients = vec![0; self.basis.dimension];
        let mut current = self.clone();
        
        for (i, b) in basis.iter().enumerate() {
            let dot = current.dot_product(b);
            let norm = b.norm_squared();
            if norm != 0 {
                coefficients[i] = dot / norm;
                current = current.sub_mod(&b.scalar_mul_mod(coefficients[i], q), q);
            }
        }
        
        current.norm_squared() == 0
    }
}

impl<F: Field> Add for LatticePoint<F> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        assert_eq!(self.basis.dimension, other.basis.dimension);
        let coordinates = self
            .coordinates
            .iter()
            .zip(other.coordinates.iter())
            .map(|(a, b)| *a + *b)
            .collect();
        Self::new(coordinates, self.basis)
    }
}

impl<F: Field> Sub for LatticePoint<F> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        assert_eq!(self.basis.dimension, other.basis.dimension);
        let coordinates = self
            .coordinates
            .iter()
            .zip(other.coordinates.iter())
            .map(|(a, b)| *a - *b)
            .collect();
        Self::new(coordinates, self.basis)
    }
}

impl<F: Field> Mul<F> for LatticePoint<F> {
    type Output = Self;

    fn mul(self, scalar: F) -> Self {
        let coordinates = self.coordinates.iter().map(|x| *x * scalar).collect();
        Self::new(coordinates, self.basis)
    }
}

#[derive(Clone, Debug)]
pub struct LatticeRelation<F: Field> {
    pub basis: LatticeBasis<F>,
    pub target: LatticePoint<F>,
    pub bound: F,
}

impl<F: Field> LatticeRelation<F> {
    pub fn new(basis: LatticeBasis<F>, target: LatticePoint<F>, bound: F) -> Self {
        Self {
            basis,
            target,
            bound,
        }
    }

    pub fn verify(&self, witness: &LatticePoint<F>) -> bool {
        let distance = witness.distance(&self.target);
        distance <= self.bound
    }
}

#[derive(Clone, Debug)]
pub struct LatticeMatrix {
    pub data: Vec<Vec<i64>>,
    pub rows: usize,
    pub cols: usize,
}

impl LatticeMatrix {
    pub fn new(data: Vec<Vec<i64>>) -> Result<Self> {
        if data.is_empty() {
            return Err(LatticeFoldError::InvalidParameters(
                "Matrix cannot be empty".to_string(),
            ));
        }

        let rows = data.len();
        let cols = data[0].len();

        for row in &data {
            if row.len() != cols {
                return Err(LatticeFoldError::InvalidParameters(
                    "Matrix rows must have equal length".to_string(),
                ));
            }
        }

        Ok(Self { data, rows, cols })
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

        Self { data, rows, cols }
    }

    pub fn mul(&self, other: &Self) -> Result<Self> {
        if self.cols != other.rows {
            return Err(LatticeFoldError::InvalidDimension {
                expected: self.cols,
                got: other.rows,
            });
        }

        let mut result = vec![vec![0; other.cols]; self.rows];

        for i in 0..self.rows {
            for j in 0..other.cols {
                for k in 0..self.cols {
                    result[i][j] += self.data[i][k] * other.data[k][j];
                }
            }
        }

        Ok(Self {
            data: result,
            rows: self.rows,
            cols: other.cols,
        })
    }

    pub fn transpose(&self) -> Self {
        let mut result = vec![vec![0; self.rows]; self.cols];

        for i in 0..self.rows {
            for j in 0..self.cols {
                result[j][i] = self.data[i][j];
            }
        }

        Self {
            data: result,
            rows: self.cols,
            cols: self.rows,
        }
    }

    pub fn gram_schmidt(&self) -> Self {
        let mut orthogonal = self.data.clone();

        for i in 0..self.rows {
            for j in 0..i {
                let dot: i64 = (0..self.cols)
                    .map(|k| orthogonal[i][k] * orthogonal[j][k])
                    .sum();
                let norm: i64 = (0..self.cols)
                    .map(|k| orthogonal[j][k] * orthogonal[j][k])
                    .sum();

                if norm != 0 {
                    let coef = dot / norm;
                    for k in 0..self.cols {
                        orthogonal[i][k] -= coef * orthogonal[j][k];
                    }
                }
            }
        }

        Self {
            data: orthogonal,
            rows: self.rows,
            cols: self.cols,
        }
    }

    pub fn lll_reduce(&mut self, delta: f64) -> Result<()> {
        if delta <= 0.25 || delta >= 1.0 {
            return Err(LatticeFoldError::InvalidParameters(
                "Delta must be in (0.25, 1.0)".to_string(),
            ));
        }

        let mut k = 1;
        while k < self.rows {
            // Size reduction
            for j in (0..k).rev() {
                let mu = self.compute_mu(k, j);
                if mu.abs() > 0.5 {
                    let r = mu.round() as i64;
                    for i in 0..self.cols {
                        self.data[k][i] -= r * self.data[j][i];
                    }
                }
            }

            // Lovász condition
            let prev_norm = self.compute_norm(k - 1);
            let curr_norm = self.compute_norm(k);
            let mu = self.compute_mu(k, k - 1);

            if curr_norm >= (delta - mu * mu) * prev_norm {
                k += 1;
            } else {
                self.swap_rows(k, k - 1);
                k = k.max(1) - 1;
            }
        }

        Ok(())
    }

    fn compute_mu(&self, i: usize, j: usize) -> f64 {
        let dot: i64 = (0..self.cols)
            .map(|k| self.data[i][k] * self.data[j][k])
            .sum();
        let norm: i64 = (0..self.cols)
            .map(|k| self.data[j][k] * self.data[j][k])
            .sum();

        if norm == 0 {
            0.0
        } else {
            dot as f64 / norm as f64
        }
    }

    fn compute_norm(&self, i: usize) -> f64 {
        let sum: i64 = (0..self.cols)
            .map(|j| self.data[i][j] * self.data[i][j])
            .sum();
        sum as f64
    }

    fn swap_rows(&mut self, i: usize, j: usize) {
        self.data.swap(i, j);
    }
} 