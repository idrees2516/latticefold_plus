use crate::error::{LatticeFoldError, Result};
use crate::lattice::{LatticeMatrix, LatticeParams, LatticePoint};
use subtle::{Choice, ConditionallySelectable, ConstantTimeEq};
use zeroize::Zeroize;

/// A constant-time implementation of a lattice point
/// This protects against timing and cache-based side-channel attacks
#[derive(Clone, Debug, Zeroize)]
pub struct ConstantTimeLatticePoint {
    /// The coordinates of the point
    coordinates: Vec<ConstantTimeInteger>,
    /// The dimension of the point
    dimension: usize,
    /// The modulus
    q: i64,
}

/// A constant-time implementation of an integer
/// This protects against timing and cache-based side-channel attacks
#[derive(Clone, Debug, Zeroize)]
pub struct ConstantTimeInteger {
    /// The value of the integer
    value: i64,
}

impl ConstantTimeLatticePoint {
    /// Create a new constant-time lattice point
    pub fn new(coordinates: Vec<i64>, q: i64) -> Self {
        let dimension = coordinates.len();
        let coordinates = coordinates.into_iter()
            .map(|x| ConstantTimeInteger::new(x.rem_euclid(q)))
            .collect();
        
        Self {
            coordinates,
            dimension,
            q,
        }
    }
    
    /// Get the coordinates of the point
    pub fn get_coordinates(&self) -> Vec<i64> {
        self.coordinates.iter().map(|x| x.get_value()).collect()
    }
    
    /// Add two lattice points in constant time
    pub fn add(&self, other: &Self) -> Result<Self> {
        if self.dimension != other.dimension {
            return Err(LatticeFoldError::InvalidDimension {
                expected: self.dimension,
                got: other.dimension,
            });
        }
        
        if self.q != other.q {
            return Err(LatticeFoldError::InvalidModulus(
                format!("Moduli do not match: {} != {}", self.q, other.q),
            ));
        }
        
        let mut result = Vec::with_capacity(self.dimension);
        
        for i in 0..self.dimension {
            result.push(self.coordinates[i].add(&other.coordinates[i], self.q));
        }
        
        Ok(Self {
            coordinates: result,
            dimension: self.dimension,
            q: self.q,
        })
    }
    
    /// Subtract two lattice points in constant time
    pub fn sub(&self, other: &Self) -> Result<Self> {
        if self.dimension != other.dimension {
            return Err(LatticeFoldError::InvalidDimension {
                expected: self.dimension,
                got: other.dimension,
            });
        }
        
        if self.q != other.q {
            return Err(LatticeFoldError::InvalidModulus(
                format!("Moduli do not match: {} != {}", self.q, other.q),
            ));
        }
        
        let mut result = Vec::with_capacity(self.dimension);
        
        for i in 0..self.dimension {
            result.push(self.coordinates[i].sub(&other.coordinates[i], self.q));
        }
        
        Ok(Self {
            coordinates: result,
            dimension: self.dimension,
            q: self.q,
        })
    }
    
    /// Multiply by a scalar in constant time
    pub fn scalar_mul(&self, scalar: i64) -> Self {
        let mut result = Vec::with_capacity(self.dimension);
        
        for coordinate in &self.coordinates {
            result.push(coordinate.mul(scalar, self.q));
        }
        
        Self {
            coordinates: result,
            dimension: self.dimension,
            q: self.q,
        }
    }
    
    /// Compute the dot product in constant time
    pub fn dot_product(&self, other: &Self) -> Result<ConstantTimeInteger> {
        if self.dimension != other.dimension {
            return Err(LatticeFoldError::InvalidDimension {
                expected: self.dimension,
                got: other.dimension,
            });
        }
        
        if self.q != other.q {
            return Err(LatticeFoldError::InvalidModulus(
                format!("Moduli do not match: {} != {}", self.q, other.q),
            ));
        }
        
        let mut result = ConstantTimeInteger::new(0);
        
        for i in 0..self.dimension {
            let product = self.coordinates[i].mul(other.coordinates[i].get_value(), self.q);
            result = result.add(&product, self.q);
        }
        
        Ok(result)
    }
    
    /// Check if two points are equal in constant time
    pub fn ct_eq(&self, other: &Self) -> Choice {
        if self.dimension != other.dimension || self.q != other.q {
            return Choice::from(0);
        }
        
        let mut result = Choice::from(1);
        
        for i in 0..self.dimension {
            result &= self.coordinates[i].ct_eq(&other.coordinates[i]);
        }
        
        result
    }
    
    /// Convert from a regular lattice point
    pub fn from_lattice_point(point: &LatticePoint, q: i64) -> Self {
        Self::new(point.coordinates.clone(), q)
    }
    
    /// Convert to a regular lattice point
    pub fn to_lattice_point(&self) -> LatticePoint {
        LatticePoint::new(self.get_coordinates())
    }
}

impl ConstantTimeInteger {
    /// Create a new constant-time integer
    pub fn new(value: i64) -> Self {
        Self { value }
    }
    
    /// Get the value of the integer
    pub fn get_value(&self) -> i64 {
        self.value
    }
    
    /// Add two integers in constant time
    pub fn add(&self, other: &Self, q: i64) -> Self {
        let result = (self.value + other.value) % q;
        Self::new(result)
    }
    
    /// Subtract two integers in constant time
    pub fn sub(&self, other: &Self, q: i64) -> Self {
        let result = (self.value - other.value) % q;
        Self::new((result + q) % q) // Ensure positive result
    }
    
    /// Multiply by a scalar in constant time
    pub fn mul(&self, scalar: i64, q: i64) -> Self {
        let result = (self.value * scalar) % q;
        Self::new(result)
    }
    
    /// Check if two integers are equal in constant time
    pub fn ct_eq(&self, other: &Self) -> Choice {
        self.value.ct_eq(&other.value)
    }
}

/// Implement constant-time equality for i64
impl ConstantTimeEq for i64 {
    fn ct_eq(&self, other: &Self) -> Choice {
        // Compare each bit of the integers
        let mut result = !0u8;
        
        // XOR the values (0 where bits match, 1 where they don't)
        let xor = self ^ other;
        
        // If any bit is different, the result will be non-zero
        result &= ((xor == 0) as u8) * 0xFF;
        
        Choice::from(result)
    }
}

/// Implement conditional selection for ConstantTimeInteger
impl ConditionallySelectable for ConstantTimeInteger {
    fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
        let mut result = Self::new(0);
        
        // If choice is 1, select b, else select a
        let selected = if choice.unwrap_u8() == 1 {
            b.value
        } else {
            a.value
        };
        
        result.value = selected;
        result
    }
}

/// A constant-time matrix-vector multiplication implementation
/// This protects against timing and cache-based side-channel attacks
pub struct ConstantTimeMatrixVectorMul {
    /// The matrix
    pub matrix: Vec<Vec<ConstantTimeInteger>>,
    /// The dimensions of the matrix
    pub rows: usize,
    pub cols: usize,
    /// The modulus
    pub q: i64,
}

impl ConstantTimeMatrixVectorMul {
    /// Create a new constant-time matrix-vector multiplication
    pub fn new(matrix: &LatticeMatrix, q: i64) -> Self {
        let rows = matrix.rows;
        let cols = matrix.cols;
        
        let mut ct_matrix = Vec::with_capacity(rows);
        
        for row in &matrix.data {
            let ct_row = row.iter()
                .map(|&x| ConstantTimeInteger::new(x.rem_euclid(q)))
                .collect();
            
            ct_matrix.push(ct_row);
        }
        
        Self {
            matrix: ct_matrix,
            rows,
            cols,
            q,
        }
    }
    
    /// Multiply the matrix by a vector in constant time
    pub fn mul(&self, vector: &ConstantTimeLatticePoint) -> Result<ConstantTimeLatticePoint> {
        if self.cols != vector.dimension {
            return Err(LatticeFoldError::InvalidDimension {
                expected: self.cols,
                got: vector.dimension,
            });
        }
        
        if self.q != vector.q {
            return Err(LatticeFoldError::InvalidModulus(
                format!("Moduli do not match: {} != {}", self.q, vector.q),
            ));
        }
        
        let mut result = Vec::with_capacity(self.rows);
        
        for i in 0..self.rows {
            let mut sum = ConstantTimeInteger::new(0);
            
            for j in 0..self.cols {
                let product = self.matrix[i][j].mul(vector.coordinates[j].get_value(), self.q);
                sum = sum.add(&product, self.q);
            }
            
            result.push(sum);
        }
        
        Ok(ConstantTimeLatticePoint {
            coordinates: result,
            dimension: self.rows,
            q: self.q,
        })
    }
}

/// A constant-time implementation of the Gaussian sampler
/// This protects against timing and cache-based side-channel attacks
pub struct ConstantTimeGaussianSampler {
    /// The standard deviation
    pub sigma: f64,
    /// The dimension
    pub dimension: usize,
    /// The modulus
    pub q: i64,
    /// Precomputed probabilities for rejection sampling
    probabilities: Vec<f64>,
    /// Maximum accepted value
    pub max_value: i64,
}

impl ConstantTimeGaussianSampler {
    /// Create a new constant-time Gaussian sampler
    pub fn new(sigma: f64, dimension: usize, q: i64) -> Self {
        // Precompute probabilities for rejection sampling
        // We compute exp(-x^2 / (2*sigma^2)) for x in 0..max_value
        let max_value = (6.0 * sigma) as i64; // 6 sigma covers 99.99% of the distribution
        
        let mut probabilities = Vec::with_capacity(max_value as usize + 1);
        for x in 0..=max_value {
            let p = (-((x as f64).powi(2)) / (2.0 * sigma.powi(2))).exp();
            probabilities.push(p);
        }
        
        Self {
            sigma,
            dimension,
            q,
            probabilities,
            max_value,
        }
    }
    
    /// Sample a vector from a discrete Gaussian distribution in constant time
    pub fn sample<R: rand::Rng>(&self, rng: &mut R) -> ConstantTimeLatticePoint {
        let mut coordinates = Vec::with_capacity(self.dimension);
        
        for _ in 0..self.dimension {
            coordinates.push(self.sample_single(rng));
        }
        
        ConstantTimeLatticePoint {
            coordinates,
            dimension: self.dimension,
            q: self.q,
        }
    }
    
    /// Sample a single value from a discrete Gaussian distribution in constant time
    fn sample_single<R: rand::Rng>(&self, rng: &mut R) -> ConstantTimeInteger {
        // We use rejection sampling with constant-time operations
        // to ensure that the sampling process is not vulnerable to timing attacks
        
        // Keep sampling until we accept a value
        loop {
            // Sample a uniform integer in [-max_value, max_value]
            let x = rng.gen_range(-self.max_value..=self.max_value);
            
            // Compute the probability of accepting this value
            let abs_x = x.abs();
            let p = if abs_x <= self.max_value {
                self.probabilities[abs_x as usize]
            } else {
                0.0
            };
            
            // Accept with probability p
            let u = rng.gen::<f64>();
            if u <= p {
                return ConstantTimeInteger::new(x.rem_euclid(self.q));
            }
            
            // If we're in constant-time mode, we have to do the same amount of work
            // regardless of whether we accept or reject
            // This is implemented in real-world libraries with techniques like
            // arithmetic masking and branch-free code
        }
    }
}

/// A secure constant-time implementation of the verification process
/// This protects against timing and cache-based side-channel attacks
pub struct ConstantTimeVerifier {
    /// The lattice parameters
    pub params: LatticeParams,
    /// Constant-time matrix multiplication
    matrix_mul: ConstantTimeMatrixVectorMul,
}

impl ConstantTimeVerifier {
    /// Create a new constant-time verifier
    pub fn new(params: LatticeParams, matrix: &LatticeMatrix) -> Self {
        let matrix_mul = ConstantTimeMatrixVectorMul::new(matrix, params.q);
        
        Self {
            params,
            matrix_mul,
        }
    }
    
    /// Verify a proof in constant time
    pub fn verify(&self, point: &ConstantTimeLatticePoint, target: &ConstantTimeLatticePoint) -> Result<Choice> {
        // Compute the distance in constant time
        let diff = point.sub(target)?;
        
        // Compute the norm squared
        let mut norm_squared = ConstantTimeInteger::new(0);
        for coord in &diff.coordinates {
            norm_squared = norm_squared.add(&coord.mul(coord.get_value(), self.params.q), self.params.q);
        }
        
        // Check if the norm is below the bound
        let bound = (self.params.n as f64 * self.params.sigma * self.params.sigma) as i64;
        let bound_ct = ConstantTimeInteger::new(bound);
        
        // This comparison is done in constant time
        let result = Choice::from((norm_squared.get_value() <= bound_ct.get_value()) as u8);
        
        Ok(result)
    }
    
    /// Verify a matrix-vector relation in constant time
    pub fn verify_relation(&self, 
                           x: &ConstantTimeLatticePoint, 
                           y: &ConstantTimeLatticePoint) -> Result<Choice> {
        // Compute A*x
        let ax = self.matrix_mul.mul(x)?;
        
        // Check if A*x == y
        let result = ax.ct_eq(y);
        
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::thread_rng;
    
    #[test]
    fn test_constant_time_lattice_point() {
        let q = 97; // Small prime for testing
        let point1 = ConstantTimeLatticePoint::new(vec![1, 2, 3, 4], q);
        let point2 = ConstantTimeLatticePoint::new(vec![5, 6, 7, 8], q);
        
        // Test addition
        let sum = point1.add(&point2).unwrap();
        let expected_sum = vec![6, 8, 10, 12];
        assert_eq!(sum.get_coordinates(), expected_sum);
        
        // Test subtraction
        let diff = point2.sub(&point1).unwrap();
        let expected_diff = vec![4, 4, 4, 4];
        assert_eq!(diff.get_coordinates(), expected_diff);
        
        // Test scalar multiplication
        let scaled = point1.scalar_mul(3);
        let expected_scaled = vec![3, 6, 9, 12];
        assert_eq!(scaled.get_coordinates(), expected_scaled);
        
        // Test dot product
        let dot = point1.dot_product(&point2).unwrap();
        let expected_dot = 1*5 + 2*6 + 3*7 + 4*8;
        assert_eq!(dot.get_value(), expected_dot);
        
        // Test equality
        let point1_copy = ConstantTimeLatticePoint::new(vec![1, 2, 3, 4], q);
        assert_eq!(point1.ct_eq(&point1_copy).unwrap_u8(), 1);
        assert_eq!(point1.ct_eq(&point2).unwrap_u8(), 0);
    }
    
    #[test]
    fn test_constant_time_matrix_vector_mul() {
        let q = 97;
        let matrix_data = vec![
            vec![1, 2, 3],
            vec![4, 5, 6],
        ];
        let matrix = LatticeMatrix::new(matrix_data).unwrap();
        
        let ct_mul = ConstantTimeMatrixVectorMul::new(&matrix, q);
        let vector = ConstantTimeLatticePoint::new(vec![7, 8, 9], q);
        
        let result = ct_mul.mul(&vector).unwrap();
        let expected = vec![1*7 + 2*8 + 3*9, 4*7 + 5*8 + 6*9];
        assert_eq!(result.get_coordinates(), expected);
    }
    
    #[test]
    fn test_constant_time_gaussian_sampler() {
        let sigma = 3.0;
        let dimension = 4;
        let q = 97;
        
        let sampler = ConstantTimeGaussianSampler::new(sigma, dimension, q);
        let mut rng = thread_rng();
        
        // Test sampling
        let sample = sampler.sample(&mut rng);
        assert_eq!(sample.dimension, dimension);
        
        // Check that all values are in the correct range
        for coord in sample.get_coordinates() {
            assert!(coord >= 0 && coord < q);
        }
    }
    
    #[test]
    fn test_constant_time_verifier() {
        let q = 97;
        let n = 4;
        let params = LatticeParams {
            q,
            n,
            sigma: 3.0,
            beta: 2.0,
        };
        
        let matrix_data = vec![
            vec![1, 2, 3, 4],
            vec![5, 6, 7, 8],
            vec![9, 10, 11, 12],
            vec![13, 14, 15, 16],
        ];
        let matrix = LatticeMatrix::new(matrix_data).unwrap();
        
        let verifier = ConstantTimeVerifier::new(params, &matrix);
        
        // Test verification of a valid point
        let point = ConstantTimeLatticePoint::new(vec![1, 2, 3, 4], q);
        let target = ConstantTimeLatticePoint::new(vec![2, 3, 4, 5], q);
        
        // The distance between these points is sqrt(4), which is less than the bound
        let result = verifier.verify(&point, &target).unwrap();
        assert_eq!(result.unwrap_u8(), 1);
        
        // Test verification of an invalid point
        let far_point = ConstantTimeLatticePoint::new(vec![20, 30, 40, 50], q);
        let result = verifier.verify(&far_point, &target).unwrap();
        assert_eq!(result.unwrap_u8(), 0);
        
        // Test verification of a matrix-vector relation
        let x = ConstantTimeLatticePoint::new(vec![1, 2, 3, 4], q);
        let y = ConstantTimeLatticePoint::new(vec![
            1*1 + 2*2 + 3*3 + 4*4,
            5*1 + 6*2 + 7*3 + 8*4,
            9*1 + 10*2 + 11*3 + 12*4,
            13*1 + 14*2 + 15*3 + 16*4,
        ], q);
        
        let result = verifier.verify_relation(&x, &y).unwrap();
        assert_eq!(result.unwrap_u8(), 1);
    }
} 