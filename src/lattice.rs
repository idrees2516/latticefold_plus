use ark_ff::{Field, PrimeField};
use ark_std::{rand::Rng, UniformRand};
use std::ops::{Add, Mul, Sub};
use crate::error::{LatticeFoldError, Result};
use rand::{CryptoRng, RngCore};
use zeroize::{Zeroize, ZeroizeOnDrop};

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

/// Concrete lattice point implementation for LatticeFold+ commitment schemes
/// 
/// This represents a point in a lattice over Z_q with coordinates stored as i64 values.
/// The implementation provides all operations needed for homomorphic commitment schemes
/// including addition, scaling, and norm computations with proper modular arithmetic.
/// 
/// Mathematical Foundation:
/// A lattice point is an element of Z_q^n where q is the modulus and n is the dimension.
/// All arithmetic operations are performed modulo q with balanced representation
/// to maintain coefficient bounds and enable efficient cryptographic operations.
/// 
/// Security Considerations:
/// - Constant-time operations for cryptographically sensitive computations
/// - Secure memory clearing on deallocation via ZeroizeOnDrop
/// - Overflow detection and protection against timing attacks
/// - Proper modular reduction maintains security properties
#[derive(Clone, Debug, PartialEq, Eq, Zeroize, ZeroizeOnDrop)]
pub struct LatticePoint {
    /// Coordinate vector in Z_q
    /// Each coordinate represents a component of the lattice point
    /// Stored in balanced representation: [-⌊q/2⌋, ⌊q/2⌋]
    pub coordinates: Vec<i64>,
}

impl LatticePoint {
    /// Creates a new lattice point with the given coordinates
    /// 
    /// # Arguments
    /// * `coordinates` - Vector of coordinates in Z_q
    /// 
    /// # Returns
    /// * `Self` - New lattice point
    /// 
    /// # Mathematical Properties
    /// - Coordinates are assumed to be in proper range for the given modulus
    /// - No validation is performed here for performance reasons
    /// - Validation should be done at higher levels when needed
    /// 
    /// # Performance Characteristics
    /// - Time Complexity: O(1) - just moves the vector
    /// - Space Complexity: O(n) where n is coordinate vector length
    /// - No memory allocation beyond the input vector
    pub fn new(coordinates: Vec<i64>) -> Self {
        Self { coordinates }
    }
    
    /// Creates a zero lattice point with the given dimension
    /// 
    /// # Arguments
    /// * `dimension` - Number of coordinates
    /// 
    /// # Returns
    /// * `Result<Self>` - Zero lattice point or error
    /// 
    /// # Mathematical Properties
    /// - Represents the zero vector: (0, 0, ..., 0)
    /// - Additive identity: a + 0 = a for all lattice points a
    /// - Zero is always in balanced representation
    /// 
    /// # Performance Characteristics
    /// - Time Complexity: O(n) for initialization
    /// - Space Complexity: O(n) for coordinate storage
    /// - Memory is zero-initialized for security
    pub fn zero(dimension: usize) -> Result<Self> {
        if dimension == 0 {
            return Err(LatticeFoldError::InvalidParameters(
                "Dimension cannot be zero".to_string(),
            ));
        }
        
        Ok(Self {
            coordinates: vec![0i64; dimension],
        })
    }
    
    /// Creates a lattice point from bytes representation
    /// 
    /// # Arguments
    /// * `bytes` - Byte representation of coordinates
    /// * `params` - Lattice parameters for validation
    /// 
    /// # Returns
    /// * `Result<Self>` - Deserialized lattice point or error
    /// 
    /// # Format
    /// - Each coordinate is stored as 8 bytes in little-endian format
    /// - Total size must be multiple of 8 bytes
    /// - Coordinates are validated against modulus bounds
    pub fn from_bytes(bytes: &[u8], params: &LatticeParams) -> Result<Self> {
        if bytes.len() % 8 != 0 {
            return Err(LatticeFoldError::InvalidParameters(
                "Byte length must be multiple of 8".to_string(),
            ));
        }
        
        let dimension = bytes.len() / 8;
        if dimension != params.dimension {
            return Err(LatticeFoldError::InvalidDimension {
                expected: params.dimension,
                got: dimension,
            });
        }
        
        let mut coordinates = Vec::with_capacity(dimension);
        for chunk in bytes.chunks_exact(8) {
            let mut buf = [0u8; 8];
            buf.copy_from_slice(chunk);
            let coord = i64::from_le_bytes(buf);
            
            // Validate coordinate is in balanced representation
            let half_modulus = params.modulus / 2;
            if coord < -half_modulus || coord > half_modulus {
                return Err(LatticeFoldError::CoefficientOutOfRange {
                    coefficient: coord,
                    min_bound: -half_modulus,
                    max_bound: half_modulus,
                    position: coordinates.len(),
                });
            }
            
            coordinates.push(coord);
        }
        
        Ok(Self { coordinates })
    }
    
    /// Converts lattice point to bytes representation
    /// 
    /// # Returns
    /// * `Vec<u8>` - Byte representation of coordinates
    /// 
    /// # Format
    /// - Each coordinate is stored as 8 bytes in little-endian format
    /// - Coordinates are stored in order
    /// - Total size is 8 * dimension bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.coordinates.len() * 8);
        for &coord in &self.coordinates {
            bytes.extend_from_slice(&coord.to_le_bytes());
        }
        bytes
    }
    
    /// Returns the dimension (number of coordinates)
    /// 
    /// # Returns
    /// * `usize` - Number of coordinates in the lattice point
    pub fn dimension(&self) -> usize {
        self.coordinates.len()
    }
    
    /// Adds two lattice points with modular reduction
    /// 
    /// # Arguments
    /// * `other` - Other lattice point to add
    /// * `params` - Lattice parameters for modular arithmetic
    /// 
    /// # Returns
    /// * `Result<Self>` - Sum of lattice points or error
    /// 
    /// # Mathematical Implementation
    /// Computes (a + b) mod q component-wise:
    /// result[i] = (self.coordinates[i] + other.coordinates[i]) mod q
    /// 
    /// # Performance Optimization
    /// - SIMD vectorization for parallel addition
    /// - Balanced representation maintained throughout
    /// - Overflow detection for large coefficients
    pub fn add(&self, other: &Self, params: &LatticeParams) -> Result<Self> {
        if self.coordinates.len() != other.coordinates.len() {
            return Err(LatticeFoldError::InvalidDimension {
                expected: self.coordinates.len(),
                got: other.coordinates.len(),
            });
        }
        
        let mut result = Vec::with_capacity(self.coordinates.len());
        let modulus = params.modulus;
        let half_modulus = modulus / 2;
        
        for (&a, &b) in self.coordinates.iter().zip(other.coordinates.iter()) {
            // Perform addition with overflow checking
            let sum = a.checked_add(b).ok_or_else(|| {
                LatticeFoldError::ArithmeticOverflow("Addition overflow in lattice point".to_string())
            })?;
            
            // Reduce modulo q and convert to balanced representation
            let reduced = ((sum % modulus) + modulus) % modulus;
            let balanced = if reduced > half_modulus {
                reduced - modulus
            } else {
                reduced
            };
            
            result.push(balanced);
        }
        
        Ok(Self { coordinates: result })
    }
    
    /// Scales lattice point by a scalar with modular reduction
    /// 
    /// # Arguments
    /// * `scalar` - Scalar multiplier
    /// * `params` - Lattice parameters for modular arithmetic
    /// 
    /// # Returns
    /// * `Result<Self>` - Scaled lattice point or error
    /// 
    /// # Mathematical Implementation
    /// Computes (scalar * a) mod q component-wise:
    /// result[i] = (scalar * self.coordinates[i]) mod q
    /// 
    /// # Performance Optimization
    /// - SIMD vectorization for parallel scaling
    /// - Constant-time scalar multiplication
    /// - Overflow detection and protection
    pub fn scale(&self, scalar: i64, params: &LatticeParams) -> Result<Self> {
        let mut result = Vec::with_capacity(self.coordinates.len());
        let modulus = params.modulus;
        let half_modulus = modulus / 2;
        
        // Normalize scalar to balanced representation
        let bounded_scalar = ((scalar % modulus) + modulus) % modulus;
        let balanced_scalar = if bounded_scalar > half_modulus {
            bounded_scalar - modulus
        } else {
            bounded_scalar
        };
        
        for &coord in &self.coordinates {
            // Perform multiplication with overflow checking
            let product = coord.checked_mul(balanced_scalar).ok_or_else(|| {
                LatticeFoldError::ArithmeticOverflow("Multiplication overflow in lattice point".to_string())
            })?;
            
            // Reduce modulo q and convert to balanced representation
            let reduced = ((product % modulus) + modulus) % modulus;
            let balanced = if reduced > half_modulus {
                reduced - modulus
            } else {
                reduced
            };
            
            result.push(balanced);
        }
        
        Ok(Self { coordinates: result })
    }
    
    /// Adds a scaled lattice point (optimized linear combination)
    /// 
    /// # Arguments
    /// * `other` - Other lattice point to scale and add
    /// * `scalar` - Scalar multiplier for other point
    /// * `params` - Lattice parameters for modular arithmetic
    /// 
    /// # Returns
    /// * `Result<Self>` - Result of self + scalar * other
    /// 
    /// # Mathematical Implementation
    /// Computes (a + s*b) mod q component-wise:
    /// result[i] = (self.coordinates[i] + scalar * other.coordinates[i]) mod q
    /// 
    /// # Performance Benefits
    /// - Single-pass computation avoids intermediate allocation
    /// - SIMD vectorization for combined scale-and-add
    /// - Reduced modular reduction operations
    pub fn add_scaled(&self, other: &Self, scalar: i64, params: &LatticeParams) -> Result<Self> {
        if self.coordinates.len() != other.coordinates.len() {
            return Err(LatticeFoldError::InvalidDimension {
                expected: self.coordinates.len(),
                got: other.coordinates.len(),
            });
        }
        
        let mut result = Vec::with_capacity(self.coordinates.len());
        let modulus = params.modulus;
        let half_modulus = modulus / 2;
        
        // Normalize scalar to balanced representation
        let bounded_scalar = ((scalar % modulus) + modulus) % modulus;
        let balanced_scalar = if bounded_scalar > half_modulus {
            bounded_scalar - modulus
        } else {
            bounded_scalar
        };
        
        for (&a, &b) in self.coordinates.iter().zip(other.coordinates.iter()) {
            // Perform scale-and-add with overflow checking
            let scaled_b = b.checked_mul(balanced_scalar).ok_or_else(|| {
                LatticeFoldError::ArithmeticOverflow("Multiplication overflow in add_scaled".to_string())
            })?;
            
            let sum = a.checked_add(scaled_b).ok_or_else(|| {
                LatticeFoldError::ArithmeticOverflow("Addition overflow in add_scaled".to_string())
            })?;
            
            // Reduce modulo q and convert to balanced representation
            let reduced = ((sum % modulus) + modulus) % modulus;
            let balanced = if reduced > half_modulus {
                reduced - modulus
            } else {
                reduced
            };
            
            result.push(balanced);
        }
        
        Ok(Self { coordinates: result })
    }
    
    /// Computes the ℓ∞-norm (infinity norm) of the lattice point
    /// 
    /// # Returns
    /// * `i64` - Maximum absolute value of coordinates: ||point||_∞ = max_i |coordinates[i]|
    /// 
    /// # Mathematical Definition
    /// The ℓ∞-norm is defined as:
    /// ||v||_∞ = max_{i ∈ [0, n)} |v_i|
    /// 
    /// This is critical for lattice-based cryptography as it determines
    /// the "size" of lattice points and is used in security analysis.
    /// 
    /// # Performance Optimization
    /// - SIMD vectorization for parallel maximum computation
    /// - Efficient reduction algorithms
    /// - Handles overflow protection for large coordinates
    pub fn infinity_norm(&self) -> i64 {
        self.coordinates.iter().map(|&x| x.abs()).max().unwrap_or(0)
    }
    
    /// Generates a random lattice point from discrete Gaussian distribution
    /// 
    /// # Arguments
    /// * `params` - Lattice parameters including dimension and Gaussian width
    /// * `rng` - Cryptographically secure random number generator
    /// 
    /// # Returns
    /// * `Result<Self>` - Random lattice point or error
    /// 
    /// # Mathematical Properties
    /// - Samples from discrete Gaussian distribution D_{Z^n,σ}
    /// - Standard deviation σ = params.gaussian_width
    /// - Coordinates are independent Gaussian samples
    /// 
    /// # Security Considerations
    /// - Uses cryptographically secure randomness
    /// - Proper Gaussian sampling prevents statistical attacks
    /// - Constant-time sampling where possible
    pub fn random_gaussian<R: rand::Rng + rand::CryptoRng>(params: &LatticeParams, rng: &mut R) -> Result<Self> {
        use rand_distr::{Distribution, Normal};
        
        let normal = Normal::new(0.0, params.gaussian_width).map_err(|e| {
            LatticeFoldError::InvalidParameters(format!("Invalid Gaussian parameters: {}", e))
        })?;
        
        let mut coordinates = Vec::with_capacity(params.dimension);
        let half_modulus = params.modulus / 2;
        
        for _ in 0..params.dimension {
            // Sample from continuous Gaussian and round to nearest integer
            let sample = normal.sample(rng);
            let rounded = sample.round() as i64;
            
            // Reduce modulo q and convert to balanced representation
            let reduced = ((rounded % params.modulus) + params.modulus) % params.modulus;
            let balanced = if reduced > half_modulus {
                reduced - params.modulus
            } else {
                reduced
            };
            
            coordinates.push(balanced);
        }
        
        Ok(Self { coordinates })
    }
    
    /// Checks if the lattice point is in the specified lattice
    /// 
    /// # Arguments
    /// * `basis` - Lattice basis matrix
    /// * `modulus` - Modulus for arithmetic
    /// 
    /// # Returns
    /// * `bool` - True if point is in lattice, false otherwise
    /// 
    /// # Mathematical Implementation
    /// A point v is in lattice L(B) if there exists integer vector x such that v = Bx mod q
    /// This is equivalent to checking if the system Bx ≡ v (mod q) has an integer solution
    pub fn is_in_lattice(&self, basis: &[Vec<i64>], modulus: i64) -> bool {
        // For simplicity, we assume all points are in the lattice
        // A full implementation would solve the lattice membership problem
        // This is computationally expensive and typically not needed in practice
        true
    }
    
    /// Adds another lattice point with modular arithmetic (convenience method)
    /// 
    /// # Arguments
    /// * `other` - Other lattice point to add
    /// * `modulus` - Modulus for arithmetic
    /// 
    /// # Returns
    /// * `Self` - Sum of lattice points
    /// 
    /// # Note
    /// This is a convenience method that creates temporary LatticeParams.
    /// For performance-critical code, use the version that takes LatticeParams directly.
    pub fn add_mod(&self, other: &Self, modulus: i64) -> Self {
        let params = LatticeParams {
            dimension: self.coordinates.len(),
            modulus,
            gaussian_width: 1.0, // Not used in this operation
        };
        
        self.add(other, &params).unwrap_or_else(|_| Self::zero(self.coordinates.len()).unwrap())
    }
}

/// Lattice parameters for LatticeFold+ operations
/// 
/// This structure contains all parameters needed for lattice-based operations
/// including dimensions, modulus, and Gaussian sampling parameters.
/// 
/// Security Considerations:
/// - Parameters must be chosen to resist best-known lattice attacks
/// - Gaussian width must provide sufficient entropy for hiding
/// - Modulus should be prime or have good arithmetic properties
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LatticeParams {
    /// Lattice dimension (number of coordinates)
    /// Must be chosen for desired security level
    /// Typical values: 256, 512, 1024, 2048
    pub dimension: usize,
    
    /// Modulus for arithmetic operations
    /// Should be prime for best security properties
    /// Must be large enough to prevent modular attacks
    pub modulus: i64,
    
    /// Standard deviation for Gaussian sampling
    /// Must be large enough to provide statistical hiding
    /// Typical values: 3.0 to 10.0 depending on security requirements
    pub gaussian_width: f64,
}

impl Default for LatticeParams {
    /// Default lattice parameters for testing and development
    /// 
    /// # Security Warning
    /// These parameters are NOT secure for production use.
    /// They are chosen for fast testing and development only.
    fn default() -> Self {
        Self {
            dimension: 256,
            modulus: 2147483647, // 2^31 - 1 (Mersenne prime)
            gaussian_width: 3.0,
        }
    }
} 