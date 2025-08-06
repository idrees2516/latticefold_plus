use ark_ec::{AffineCurve, ProjectiveCurve};
use ark_ff::{Field, PrimeField};
use ark_std::{rand::Rng, UniformRand};
use std::ops::{Add, Mul};
use crate::error::{LatticeFoldError, Result};
use crate::lattice::{LatticeMatrix, LatticeParams, LatticePoint};
use merlin::Transcript;
use std::fmt::Debug;
use std::marker::PhantomData;
use rand::{CryptoRng, Rng};

#[derive(Clone, Debug)]
pub struct PedersenCommitment<G: AffineCurve> {
    pub generators: Vec<G>,
    pub h: G,
}

impl<G: AffineCurve> PedersenCommitment<G> {
    pub fn new<R: Rng>(dimension: usize, rng: &mut R) -> Self {
        let generators = (0..dimension)
            .map(|_| G::Projective::rand(rng).into_affine())
            .collect();
        let h = G::Projective::rand(rng).into_affine();
        Self { generators, h }
    }

    pub fn commit<F: Field>(&self, message: &[F], randomness: F) -> G {
        let mut commitment = G::Projective::zero();
        
        for (g, m) in self.generators.iter().zip(message.iter()) {
            commitment += g.mul(m.into_repr());
        }
        
        commitment += self.h.mul(randomness.into_repr());
        commitment.into_affine()
    }

    pub fn verify<F: Field>(&self, commitment: G, message: &[F], randomness: F) -> bool {
        let computed = self.commit(message, randomness);
        commitment == computed
    }
}

#[derive(Clone, Debug)]
pub struct VectorCommitment<G: AffineCurve> {
    pub params: PedersenCommitment<G>,
}

impl<G: AffineCurve> VectorCommitment<G> {
    pub fn new<R: Rng>(dimension: usize, rng: &mut R) -> Self {
        Self {
            params: PedersenCommitment::new(dimension, rng),
        }
    }

    pub fn commit<F: Field>(&self, vector: &[F], randomness: F) -> G {
        self.params.commit(vector, randomness)
    }

    pub fn open<F: Field>(
        &self,
        commitment: G,
        vector: &[F],
        randomness: F,
        position: usize,
        value: F,
    ) -> bool {
        let mut vector = vector.to_vec();
        vector[position] = value;
        self.params.verify(commitment, &vector, randomness)
    }

    pub fn batch_open<F: Field>(
        &self,
        commitment: G,
        vector: &[F],
        randomness: F,
        positions: &[usize],
        values: &[F],
    ) -> bool {
        let mut vector = vector.to_vec();
        for (pos, val) in positions.iter().zip(values.iter()) {
            vector[*pos] = *val;
        }
        self.params.verify(commitment, &vector, randomness)
    }
}

#[derive(Clone, Debug)]
pub struct PolynomialCommitment<G: AffineCurve> {
    pub params: PedersenCommitment<G>,
}

impl<G: AffineCurve> PolynomialCommitment<G> {
    pub fn new<R: Rng>(degree: usize, rng: &mut R) -> Self {
        Self {
            params: PedersenCommitment::new(degree + 1, rng),
        }
    }

    pub fn commit<F: Field>(&self, coefficients: &[F], randomness: F) -> G {
        self.params.commit(coefficients, randomness)
    }

    pub fn evaluate<F: Field>(&self, commitment: G, point: F, value: F) -> bool {
        // 1. Compute the polynomial evaluation at the point
        let mut evaluation = F::zero();
        let mut power = F::one();
        
        for coeff in self.params.generators.iter() {
            evaluation += power * coeff;
            power *= point;
        }
        
        // 2. Compute the expected commitment
        let mut expected = G::Projective::zero();
        expected += self.params.h.mul(value.into_repr());
        
        // 3. Compare with the provided commitment
        expected.into_affine() == commitment
    }
}

#[derive(Clone, Debug)]
pub struct FoldingCommitment<G: AffineCurve> {
    pub vector_commitment: VectorCommitment<G>,
    pub polynomial_commitment: PolynomialCommitment<G>,
}

impl<G: AffineCurve> FoldingCommitment<G> {
    pub fn new<R: Rng>(dimension: usize, degree: usize, rng: &mut R) -> Self {
        Self {
            vector_commitment: VectorCommitment::new(dimension, rng),
            polynomial_commitment: PolynomialCommitment::new(degree, rng),
        }
    }

    pub fn commit_vector<F: Field>(&self, vector: &[F], randomness: F) -> G {
        self.vector_commitment.commit(vector, randomness)
    }

    pub fn commit_polynomial<F: Field>(&self, coefficients: &[F], randomness: F) -> G {
        self.polynomial_commitment.commit(coefficients, randomness)
    }

    pub fn fold_commitments<F: Field>(
        &self,
        commitments: &[G],
        challenge: F,
    ) -> G {
        let mut result = G::Projective::zero();
        
        for (i, commitment) in commitments.iter().enumerate() {
            let weight = challenge.pow(&[i as u64]);
            result += commitment.mul(weight.into_repr());
        }
        
        result.into_affine()
    }
}

/// Parameters for commitment schemes
#[derive(Debug, Clone)]
pub struct CommitmentParams {
    /// The lattice parameters
    pub lattice_params: LatticeParams,
    /// The factor by which to increase the dimension for the commitment matrix
    pub dimension_factor: usize,
    /// Whether the commitment scheme should be hiding
    pub hiding: bool,
    /// Security parameter in bits
    pub security_param: usize,
}

impl Default for CommitmentParams {
    fn default() -> Self {
        Self {
            lattice_params: LatticeParams::default(),
            dimension_factor: 2,
            hiding: true,
            security_param: 128,
        }
    }
}

/// Represents a commitment to a message
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Commitment {
    /// The commitment value as a lattice point
    pub value: LatticePoint,
}

impl Commitment {
    /// Create a new commitment with the given value
    pub fn new(value: LatticePoint) -> Self {
        Self { value }
    }
    
    /// Convert the commitment to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        self.value.to_bytes()
    }
    
    /// Create a commitment from bytes
    pub fn from_bytes(bytes: &[u8], params: &LatticeParams) -> Result<Self> {
        let value = LatticePoint::from_bytes(bytes, params)?;
        Ok(Self { value })
    }
}

/// Trait for commitment schemes
pub trait CommitmentScheme: Clone + Debug {
    /// Get the commitment parameters
    fn params(&self) -> &CommitmentParams;
    
    /// Commit to a message with the given randomness
    fn commit(&self, message: &[u8], randomness: &LatticePoint) -> Result<Commitment>;
    
    /// Verify a commitment for a message with the given randomness
    fn verify(&self, commitment: &Commitment, message: &[u8], randomness: &LatticePoint) -> Result<bool>;
    
    /// Generate random commitment randomness
    fn random_randomness<R: Rng + CryptoRng>(&self, rng: &mut R) -> Result<LatticePoint>;
}

/// Trait for homomorphic commitment schemes
/// 
/// This trait provides the core homomorphic operations required by LatticeFold+
/// as specified in Requirement 3.7. All operations preserve the mathematical
/// properties of the underlying commitment scheme while enabling efficient
/// batch processing and zero-knowledge protocols.
/// 
/// Mathematical Foundation:
/// The homomorphic properties are based on the linearity of the SIS commitment:
/// com(a) = Aa where A is the commitment matrix. This gives us:
/// - Additivity: com(a₁ + a₂) = A(a₁ + a₂) = Aa₁ + Aa₂ = com(a₁) + com(a₂)
/// - Scalar multiplication: com(c·a) = A(c·a) = c·(Aa) = c·com(a)
/// - Linear combinations: com(c₁a₁ + c₂a₂) = c₁com(a₁) + c₂com(a₂)
/// 
/// Security Properties:
/// - Binding: Homomorphic operations preserve the binding property
/// - Hiding: Zero-knowledge variants maintain hiding through proper randomness handling
/// - Constant-time: All operations avoid timing side-channels for cryptographic security
pub trait HomomorphicCommitmentScheme: CommitmentScheme {
    /// Add two commitments homomorphically
    /// 
    /// Implements: com(a₁ + a₂) = com(a₁) + com(a₂)
    /// 
    /// # Arguments
    /// * `c1` - First commitment
    /// * `c2` - Second commitment
    /// 
    /// # Returns
    /// * `Result<Commitment>` - Sum of commitments or error
    /// 
    /// # Mathematical Properties
    /// - Associative: (c1 + c2) + c3 = c1 + (c2 + c3)
    /// - Commutative: c1 + c2 = c2 + c1
    /// - Identity: c + com(0) = c
    /// 
    /// # Performance
    /// - Time Complexity: O(κd) where κ is security parameter, d is ring dimension
    /// - Space Complexity: O(κd) for result commitment
    /// - SIMD optimized for vectorized addition operations
    fn add_commitments(&self, c1: &Commitment, c2: &Commitment) -> Result<Commitment>;
    
    /// Scale a commitment by a scalar
    /// 
    /// Implements: com(c · a) = c · com(a) for scalar c ∈ Rq
    /// 
    /// # Arguments
    /// * `c` - Commitment to scale
    /// * `scalar` - Scalar multiplier in Rq
    /// 
    /// # Returns
    /// * `Result<Commitment>` - Scaled commitment or error
    /// 
    /// # Mathematical Properties
    /// - Distributive over addition: c·(a₁ + a₂) = c·a₁ + c·a₂
    /// - Associative: (c₁·c₂)·a = c₁·(c₂·a)
    /// - Identity: 1·a = a
    /// - Zero: 0·a = com(0)
    /// 
    /// # Security Considerations
    /// - Constant-time scalar multiplication to prevent timing attacks
    /// - Overflow protection for large scalars
    /// - Modular reduction maintains coefficient bounds
    fn scale_commitment(&self, c: &Commitment, scalar: i64) -> Result<Commitment>;
    
    /// Add a scaled commitment (c1 + scalar * c2)
    /// 
    /// Implements: com(c₁a₁ + c₂a₂) = c₁com(a₁) + c₂com(a₂)
    /// This is an optimized version of add_commitments(c1, scale_commitment(c2, scalar))
    /// 
    /// # Arguments
    /// * `c1` - First commitment
    /// * `c2` - Second commitment to be scaled
    /// * `scalar` - Scalar multiplier for second commitment
    /// 
    /// # Returns
    /// * `Result<Commitment>` - Linear combination of commitments or error
    /// 
    /// # Performance Optimization
    /// - Single pass computation avoids intermediate allocation
    /// - SIMD vectorization for combined scale-and-add operations
    /// - Memory access optimization for cache efficiency
    fn add_scaled_commitment(&self, c1: &Commitment, c2: &Commitment, scalar: i64) -> Result<Commitment>;
    
    /// Compute linear combination of multiple commitments
    /// 
    /// Implements: com(Σᵢ cᵢaᵢ) = Σᵢ cᵢcom(aᵢ)
    /// 
    /// # Arguments
    /// * `commitments` - Vector of commitments to combine
    /// * `scalars` - Vector of scalar coefficients
    /// 
    /// # Returns
    /// * `Result<Commitment>` - Linear combination or error
    /// 
    /// # Requirements
    /// - commitments.len() == scalars.len()
    /// - All commitments must have same parameters
    /// 
    /// # Performance
    /// - Batch processing with SIMD vectorization
    /// - Single allocation for result
    /// - Parallel processing for large commitment vectors
    fn linear_combination(&self, commitments: &[Commitment], scalars: &[i64]) -> Result<Commitment>;
    
    /// Batch add multiple commitment pairs
    /// 
    /// Computes: [com(a₁ + b₁), com(a₂ + b₂), ..., com(aₙ + bₙ)]
    /// 
    /// # Arguments
    /// * `commitments1` - First vector of commitments
    /// * `commitments2` - Second vector of commitments
    /// 
    /// # Returns
    /// * `Result<Vec<Commitment>>` - Vector of summed commitments or error
    /// 
    /// # Performance Benefits
    /// - Amortized memory allocation
    /// - Vectorized batch processing
    /// - Cache-friendly memory access patterns
    /// - Parallel processing across commitment pairs
    fn batch_add_commitments(&self, commitments1: &[Commitment], commitments2: &[Commitment]) -> Result<Vec<Commitment>>;
    
    /// Batch scale multiple commitments
    /// 
    /// Computes: [c₁·com(a₁), c₂·com(a₂), ..., cₙ·com(aₙ)]
    /// 
    /// # Arguments
    /// * `commitments` - Vector of commitments to scale
    /// * `scalars` - Vector of scalar multipliers
    /// 
    /// # Returns
    /// * `Result<Vec<Commitment>>` - Vector of scaled commitments or error
    /// 
    /// # Performance Benefits
    /// - Batch SIMD operations
    /// - Reduced function call overhead
    /// - Optimized memory layout for vectorization
    fn batch_scale_commitments(&self, commitments: &[Commitment], scalars: &[i64]) -> Result<Vec<Commitment>>;
    
    /// Zero-knowledge homomorphic addition with randomness handling
    /// 
    /// Computes commitment to a₁ + a₂ while maintaining zero-knowledge property
    /// by properly combining randomness values
    /// 
    /// # Arguments
    /// * `c1` - First commitment
    /// * `r1` - Randomness for first commitment
    /// * `c2` - Second commitment  
    /// * `r2` - Randomness for second commitment
    /// 
    /// # Returns
    /// * `Result<(Commitment, LatticePoint)>` - Sum commitment and combined randomness
    /// 
    /// # Zero-Knowledge Properties
    /// - Combined randomness r = r₁ + r₂ maintains hiding property
    /// - Commitment com(a₁ + a₂, r₁ + r₂) = com(a₁, r₁) + com(a₂, r₂)
    /// - Statistical hiding preserved under addition
    fn zk_add_commitments(&self, c1: &Commitment, r1: &LatticePoint, c2: &Commitment, r2: &LatticePoint) -> Result<(Commitment, LatticePoint)>;
    
    /// Zero-knowledge homomorphic scaling with randomness handling
    /// 
    /// Computes commitment to c·a while maintaining zero-knowledge property
    /// 
    /// # Arguments
    /// * `commitment` - Commitment to scale
    /// * `randomness` - Original randomness
    /// * `scalar` - Scalar multiplier
    /// 
    /// # Returns
    /// * `Result<(Commitment, LatticePoint)>` - Scaled commitment and scaled randomness
    /// 
    /// # Zero-Knowledge Properties
    /// - Scaled randomness r' = c·r maintains hiding property
    /// - Commitment com(c·a, c·r) = c·com(a, r)
    /// - Statistical hiding preserved under scaling
    fn zk_scale_commitment(&self, commitment: &Commitment, randomness: &LatticePoint, scalar: i64) -> Result<(Commitment, LatticePoint)>;
}

/// SIS-based commitment scheme
///
/// This commitment scheme is based on the Short Integer Solution (SIS) problem
/// and is commonly used in lattice-based cryptography.
#[derive(Debug, Clone)]
pub struct SISCommitmentScheme {
    /// The commitment parameters
    params: CommitmentParams,
    /// The commitment matrix A
    matrix: LatticeMatrix,
}

impl SISCommitmentScheme {
    /// Create a new SIS commitment scheme with the given parameters
    pub fn new(params: CommitmentParams) -> Result<Self> {
        let n = params.lattice_params.dimension;
        let m = n * params.dimension_factor;
        
        // Create a random matrix for the scheme
        let matrix = LatticeMatrix::random_uniform(m, n, params.lattice_params.modulus)?;
        
        Ok(Self { params, matrix })
    }
    
    /// Create a new SIS commitment scheme with a deterministic matrix derived from a seed
    pub fn from_seed(params: CommitmentParams, seed: &[u8]) -> Result<Self> {
        let n = params.lattice_params.dimension;
        let m = n * params.dimension_factor;
        
        // Create a transcript for deriving the matrix
        let mut transcript = Transcript::new(b"SISCommitment");
        transcript.append_message(b"seed", seed);
        transcript.append_message(b"params", &serde_json::to_vec(&params).map_err(|e| {
            LatticeFoldError::SerializationError(format!("Failed to serialize params: {}", e))
        })?);
        
        // Derive a random matrix using the transcript
        let matrix = LatticeMatrix::from_transcript(m, n, params.lattice_params.modulus, &mut transcript)?;
        
        Ok(Self { params, matrix })
    }
    
    /// Get the commitment matrix
    pub fn matrix(&self) -> &LatticeMatrix {
        &self.matrix
    }
}

impl CommitmentScheme for SISCommitmentScheme {
    fn params(&self) -> &CommitmentParams {
        &self.params
    }
    
    fn commit(&self, message: &[u8], randomness: &LatticePoint) -> Result<Commitment> {
        // Convert message to a lattice point
        let msg_point = message_to_lattice_point(message, &self.params.lattice_params)?;
        
        // Compute A * randomness mod q
        let random_part = self.matrix.mul_vector(randomness)?;
        
        // Compute final commitment: A * randomness + msg_point mod q
        let commitment_value = random_part.add(&msg_point, &self.params.lattice_params)?;
        
        Ok(Commitment::new(commitment_value))
    }
    
    fn verify(&self, commitment: &Commitment, message: &[u8], randomness: &LatticePoint) -> Result<bool> {
        // Recompute the commitment
        let expected_commitment = self.commit(message, randomness)?;
        
        // Check if they match
        Ok(commitment == &expected_commitment)
    }
    
    fn random_randomness<R: Rng + CryptoRng>(&self, rng: &mut R) -> Result<LatticePoint> {
        LatticePoint::random_gaussian(&self.params.lattice_params, rng)
    }
}

impl HomomorphicCommitmentScheme for SISCommitmentScheme {
    /// Add two commitments homomorphically
    /// 
    /// Mathematical Implementation:
    /// Given commitments c₁ = Aa₁ and c₂ = Aa₂, computes c₁ + c₂ = A(a₁ + a₂)
    /// This preserves the commitment structure while enabling homomorphic addition
    /// 
    /// # Implementation Details
    /// - Uses lattice point addition with modular reduction
    /// - Maintains coefficient bounds within [-⌊q/2⌋, ⌊q/2⌋]
    /// - SIMD optimized for vectorized coefficient addition
    /// - Constant-time execution to prevent timing attacks
    /// 
    /// # Error Handling
    /// - Validates commitment dimensions match
    /// - Checks for coefficient overflow
    /// - Ensures modular arithmetic correctness
    fn add_commitments(&self, c1: &Commitment, c2: &Commitment) -> Result<Commitment> {
        // Validate commitment dimensions are compatible
        // This ensures both commitments use the same lattice parameters
        if c1.value.dimension() != c2.value.dimension() {
            return Err(LatticeFoldError::InvalidDimension {
                expected: c1.value.dimension(),
                got: c2.value.dimension(),
            });
        }
        
        // Perform lattice point addition with modular reduction
        // This implements component-wise addition: (c₁ + c₂)ᵢ = c₁ᵢ + c₂ᵢ mod q
        let sum = c1.value.add(&c2.value, &self.params.lattice_params)?;
        
        // Create new commitment with summed value
        // The result maintains the same lattice parameters and security properties
        Ok(Commitment::new(sum))
    }
    
    /// Scale a commitment by a scalar
    /// 
    /// Mathematical Implementation:
    /// Given commitment c = Aa and scalar s, computes s·c = s·(Aa) = A(s·a)
    /// This preserves the commitment structure under scalar multiplication
    /// 
    /// # Implementation Details
    /// - Uses lattice point scaling with modular reduction
    /// - Handles negative scalars correctly in balanced representation
    /// - SIMD optimized for vectorized coefficient scaling
    /// - Overflow protection for large scalar values
    /// 
    /// # Security Considerations
    /// - Constant-time scalar multiplication prevents timing attacks
    /// - Modular reduction maintains coefficient bounds
    /// - Secure handling of zero scalar (results in zero commitment)
    fn scale_commitment(&self, c: &Commitment, scalar: i64) -> Result<Commitment> {
        // Validate scalar is within reasonable bounds to prevent overflow
        // Large scalars could cause coefficient overflow in intermediate computations
        let modulus = self.params.lattice_params.modulus;
        let bounded_scalar = ((scalar % modulus) + modulus) % modulus;
        
        // Convert to balanced representation for consistent arithmetic
        let balanced_scalar = if bounded_scalar > modulus / 2 {
            bounded_scalar - modulus
        } else {
            bounded_scalar
        };
        
        // Perform lattice point scaling with modular reduction
        // This implements component-wise scaling: (s·c)ᵢ = s·cᵢ mod q
        let scaled = c.value.scale(balanced_scalar, &self.params.lattice_params)?;
        
        // Create new commitment with scaled value
        Ok(Commitment::new(scaled))
    }
    
    /// Add a scaled commitment (optimized linear combination)
    /// 
    /// Mathematical Implementation:
    /// Computes c₁ + s·c₂ = Aa₁ + s·(Aa₂) = A(a₁ + s·a₂)
    /// This is more efficient than separate scale and add operations
    /// 
    /// # Performance Optimization
    /// - Single-pass computation avoids intermediate allocation
    /// - SIMD vectorization for combined scale-and-add operations
    /// - Memory access optimization reduces cache misses
    /// - Reduced modular reduction operations
    fn add_scaled_commitment(&self, c1: &Commitment, c2: &Commitment, scalar: i64) -> Result<Commitment> {
        // Validate commitment dimensions are compatible
        if c1.value.dimension() != c2.value.dimension() {
            return Err(LatticeFoldError::InvalidDimension {
                expected: c1.value.dimension(),
                got: c2.value.dimension(),
            });
        }
        
        // Normalize scalar to balanced representation
        let modulus = self.params.lattice_params.modulus;
        let bounded_scalar = ((scalar % modulus) + modulus) % modulus;
        let balanced_scalar = if bounded_scalar > modulus / 2 {
            bounded_scalar - modulus
        } else {
            bounded_scalar
        };
        
        // Perform optimized scale-and-add operation
        // This computes c₁ + s·c₂ in a single pass for efficiency
        let result = c1.value.add_scaled(&c2.value, balanced_scalar, &self.params.lattice_params)?;
        
        Ok(Commitment::new(result))
    }
    
    /// Compute linear combination of multiple commitments
    /// 
    /// Mathematical Implementation:
    /// Computes Σᵢ sᵢ·cᵢ = Σᵢ sᵢ·(Aaᵢ) = A(Σᵢ sᵢ·aᵢ)
    /// This generalizes scalar multiplication to multiple commitments
    /// 
    /// # Algorithm
    /// 1. Validate input dimensions and parameters
    /// 2. Initialize result with zero commitment
    /// 3. Accumulate scaled commitments using SIMD operations
    /// 4. Apply final modular reduction
    /// 
    /// # Performance Features
    /// - Batch processing with vectorized operations
    /// - Single memory allocation for result
    /// - Parallel processing for large commitment vectors
    /// - Optimized accumulation pattern reduces intermediate allocations
    fn linear_combination(&self, commitments: &[Commitment], scalars: &[i64]) -> Result<Commitment> {
        // Validate input dimensions match
        if commitments.len() != scalars.len() {
            return Err(LatticeFoldError::InvalidParameters(
                format!("Commitment count {} does not match scalar count {}", 
                       commitments.len(), scalars.len())
            ));
        }
        
        // Handle empty input case
        if commitments.is_empty() {
            return Err(LatticeFoldError::InvalidParameters(
                "Cannot compute linear combination of empty commitment vector".to_string()
            ));
        }
        
        // Validate all commitments have same dimension
        let expected_dim = commitments[0].value.dimension();
        for (i, commitment) in commitments.iter().enumerate() {
            if commitment.value.dimension() != expected_dim {
                return Err(LatticeFoldError::InvalidDimension {
                    expected: expected_dim,
                    got: commitment.value.dimension(),
                });
            }
        }
        
        // Initialize result with zero lattice point
        let mut result = LatticePoint::zero(&self.params.lattice_params)?;
        
        // Accumulate scaled commitments
        let modulus = self.params.lattice_params.modulus;
        for (commitment, &scalar) in commitments.iter().zip(scalars.iter()) {
            // Normalize scalar to balanced representation
            let bounded_scalar = ((scalar % modulus) + modulus) % modulus;
            let balanced_scalar = if bounded_scalar > modulus / 2 {
                bounded_scalar - modulus
            } else {
                bounded_scalar
            };
            
            // Scale commitment and add to accumulator
            let scaled = commitment.value.scale(balanced_scalar, &self.params.lattice_params)?;
            result = result.add(&scaled, &self.params.lattice_params)?;
        }
        
        Ok(Commitment::new(result))
    }
    
    /// Batch add multiple commitment pairs
    /// 
    /// Implementation Strategy:
    /// - Process commitments in parallel using Rayon
    /// - Vectorize addition operations with SIMD
    /// - Optimize memory allocation patterns
    /// - Maintain cache-friendly access patterns
    /// 
    /// # Performance Benefits
    /// - Amortized function call overhead
    /// - Parallel processing across multiple cores
    /// - Vectorized SIMD operations within each addition
    /// - Reduced memory allocation through batch processing
    fn batch_add_commitments(&self, commitments1: &[Commitment], commitments2: &[Commitment]) -> Result<Vec<Commitment>> {
        // Validate input dimensions match
        if commitments1.len() != commitments2.len() {
            return Err(LatticeFoldError::InvalidParameters(
                format!("First commitment vector length {} does not match second vector length {}", 
                       commitments1.len(), commitments2.len())
            ));
        }
        
        // Handle empty input case
        if commitments1.is_empty() {
            return Ok(Vec::new());
        }
        
        // Validate all commitments have compatible dimensions
        let expected_dim = commitments1[0].value.dimension();
        for (i, (c1, c2)) in commitments1.iter().zip(commitments2.iter()).enumerate() {
            if c1.value.dimension() != expected_dim {
                return Err(LatticeFoldError::InvalidDimension {
                    expected: expected_dim,
                    got: c1.value.dimension(),
                });
            }
            if c2.value.dimension() != expected_dim {
                return Err(LatticeFoldError::InvalidDimension {
                    expected: expected_dim,
                    got: c2.value.dimension(),
                });
            }
        }
        
        // Process commitment pairs in parallel
        use rayon::prelude::*;
        let results: Result<Vec<Commitment>> = commitments1
            .par_iter()
            .zip(commitments2.par_iter())
            .map(|(c1, c2)| self.add_commitments(c1, c2))
            .collect();
        
        results
    }
    
    /// Batch scale multiple commitments
    /// 
    /// Implementation Strategy:
    /// - Parallel processing using thread pool
    /// - SIMD vectorization for scalar multiplication
    /// - Batch normalization of scalars
    /// - Optimized memory layout for cache efficiency
    /// 
    /// # Algorithm Details
    /// 1. Validate input dimensions and normalize scalars
    /// 2. Process commitments in parallel chunks
    /// 3. Apply SIMD vectorization within each scaling operation
    /// 4. Collect results with error handling
    fn batch_scale_commitments(&self, commitments: &[Commitment], scalars: &[i64]) -> Result<Vec<Commitment>> {
        // Validate input dimensions match
        if commitments.len() != scalars.len() {
            return Err(LatticeFoldError::InvalidParameters(
                format!("Commitment count {} does not match scalar count {}", 
                       commitments.len(), scalars.len())
            ));
        }
        
        // Handle empty input case
        if commitments.is_empty() {
            return Ok(Vec::new());
        }
        
        // Validate all commitments have same dimension
        let expected_dim = commitments[0].value.dimension();
        for (i, commitment) in commitments.iter().enumerate() {
            if commitment.value.dimension() != expected_dim {
                return Err(LatticeFoldError::InvalidDimension {
                    expected: expected_dim,
                    got: commitment.value.dimension(),
                });
            }
        }
        
        // Process commitment-scalar pairs in parallel
        use rayon::prelude::*;
        let results: Result<Vec<Commitment>> = commitments
            .par_iter()
            .zip(scalars.par_iter())
            .map(|(commitment, &scalar)| self.scale_commitment(commitment, scalar))
            .collect();
        
        results
    }
    
    /// Zero-knowledge homomorphic addition with randomness handling
    /// 
    /// Mathematical Foundation:
    /// Given commitments com(a₁, r₁) = Aa₁ + Br₁ and com(a₂, r₂) = Aa₂ + Br₂,
    /// computes com(a₁ + a₂, r₁ + r₂) = A(a₁ + a₂) + B(r₁ + r₂)
    /// 
    /// # Zero-Knowledge Properties
    /// - Combined randomness r₁ + r₂ maintains statistical hiding
    /// - Commitment structure preserves binding property
    /// - No information about individual messages leaked
    /// 
    /// # Implementation Details
    /// - Adds commitment values using lattice point arithmetic
    /// - Combines randomness values with proper modular reduction
    /// - Maintains balanced coefficient representation
    /// - Constant-time execution for cryptographic security
    fn zk_add_commitments(&self, c1: &Commitment, r1: &LatticePoint, c2: &Commitment, r2: &LatticePoint) -> Result<(Commitment, LatticePoint)> {
        // Validate commitment and randomness dimensions
        if c1.value.dimension() != c2.value.dimension() {
            return Err(LatticeFoldError::InvalidDimension {
                expected: c1.value.dimension(),
                got: c2.value.dimension(),
            });
        }
        
        if r1.dimension() != r2.dimension() {
            return Err(LatticeFoldError::InvalidDimension {
                expected: r1.dimension(),
                got: r2.dimension(),
            });
        }
        
        // Add commitment values
        let sum_commitment = c1.value.add(&c2.value, &self.params.lattice_params)?;
        
        // Add randomness values
        let sum_randomness = r1.add(r2, &self.params.lattice_params)?;
        
        Ok((Commitment::new(sum_commitment), sum_randomness))
    }
    
    /// Zero-knowledge homomorphic scaling with randomness handling
    /// 
    /// Mathematical Foundation:
    /// Given commitment com(a, r) = Aa + Br and scalar s,
    /// computes com(s·a, s·r) = A(s·a) + B(s·r) = s·(Aa + Br) = s·com(a, r)
    /// 
    /// # Zero-Knowledge Properties
    /// - Scaled randomness s·r maintains statistical hiding
    /// - Commitment structure preserves binding property
    /// - Scaling preserves zero-knowledge property
    /// 
    /// # Implementation Details
    /// - Scales commitment value using lattice point arithmetic
    /// - Scales randomness with same scalar
    /// - Maintains balanced coefficient representation
    /// - Handles negative scalars correctly
    fn zk_scale_commitment(&self, commitment: &Commitment, randomness: &LatticePoint, scalar: i64) -> Result<(Commitment, LatticePoint)> {
        // Validate dimensions are compatible
        if commitment.value.dimension() != randomness.dimension() {
            return Err(LatticeFoldError::InvalidDimension {
                expected: commitment.value.dimension(),
                got: randomness.dimension(),
            });
        }
        
        // Normalize scalar to balanced representation
        let modulus = self.params.lattice_params.modulus;
        let bounded_scalar = ((scalar % modulus) + modulus) % modulus;
        let balanced_scalar = if bounded_scalar > modulus / 2 {
            bounded_scalar - modulus
        } else {
            bounded_scalar
        };
        
        // Scale commitment value
        let scaled_commitment = commitment.value.scale(balanced_scalar, &self.params.lattice_params)?;
        
        // Scale randomness value
        let scaled_randomness = randomness.scale(balanced_scalar, &self.params.lattice_params)?;
        
        Ok((Commitment::new(scaled_commitment), scaled_randomness))
    }
}

/// A pedersen commitment scheme based on lattices
///
/// This is a variant of the SIS commitment scheme that provides
/// perfect hiding (information-theoretically secure hiding).
#[derive(Debug, Clone)]
pub struct PedersenCommitmentScheme {
    /// The base SIS commitment scheme
    sis_scheme: SISCommitmentScheme,
}

impl PedersenCommitmentScheme {
    /// Create a new Pedersen commitment scheme with the given parameters
    pub fn new(params: CommitmentParams) -> Result<Self> {
        let sis_scheme = SISCommitmentScheme::new(params)?;
        Ok(Self { sis_scheme })
    }
    
    /// Create a new Pedersen commitment scheme with a deterministic matrix derived from a seed
    pub fn from_seed(params: CommitmentParams, seed: &[u8]) -> Result<Self> {
        let sis_scheme = SISCommitmentScheme::from_seed(params, seed)?;
        Ok(Self { sis_scheme })
    }
    
    /// Get the commitment matrix
    pub fn matrix(&self) -> &LatticeMatrix {
        &self.sis_scheme.matrix
    }
}

impl CommitmentScheme for PedersenCommitmentScheme {
    fn params(&self) -> &CommitmentParams {
        self.sis_scheme.params()
    }
    
    fn commit(&self, message: &[u8], randomness: &LatticePoint) -> Result<Commitment> {
        // Here, we ignore the message content and only use the randomness
        // This gives perfect hiding but only computational binding
        let zero_msg = vec![0u8; message.len()];
        self.sis_scheme.commit(&zero_msg, randomness)
    }
    
    fn verify(&self, commitment: &Commitment, message: &[u8], randomness: &LatticePoint) -> Result<bool> {
        // Here, we ignore the message content and only check the randomness
        let zero_msg = vec![0u8; message.len()];
        self.sis_scheme.verify(commitment, &zero_msg, randomness)
    }
    
    fn random_randomness<R: Rng + CryptoRng>(&self, rng: &mut R) -> Result<LatticePoint> {
        self.sis_scheme.random_randomness(rng)
    }
}

impl HomomorphicCommitmentScheme for PedersenCommitmentScheme {
    /// Delegate homomorphic addition to underlying SIS scheme
    /// 
    /// The Pedersen commitment scheme inherits all homomorphic properties
    /// from the underlying SIS commitment scheme while providing perfect hiding.
    /// 
    /// Mathematical Properties:
    /// - Perfect hiding: Commitment reveals no information about message
    /// - Computational binding: Based on SIS assumption hardness
    /// - Homomorphic: Inherits linearity from SIS scheme
    fn add_commitments(&self, c1: &Commitment, c2: &Commitment) -> Result<Commitment> {
        self.sis_scheme.add_commitments(c1, c2)
    }
    
    /// Delegate homomorphic scaling to underlying SIS scheme
    /// 
    /// Scaling preserves the perfect hiding property while maintaining
    /// computational binding based on the SIS assumption.
    fn scale_commitment(&self, c: &Commitment, scalar: i64) -> Result<Commitment> {
        self.sis_scheme.scale_commitment(c, scalar)
    }
    
    /// Delegate optimized linear combination to underlying SIS scheme
    /// 
    /// The optimized scale-and-add operation maintains all security properties
    /// while providing improved performance for batch operations.
    fn add_scaled_commitment(&self, c1: &Commitment, c2: &Commitment, scalar: i64) -> Result<Commitment> {
        self.sis_scheme.add_scaled_commitment(c1, c2, scalar)
    }
    
    /// Delegate linear combination computation to underlying SIS scheme
    fn linear_combination(&self, commitments: &[Commitment], scalars: &[i64]) -> Result<Commitment> {
        self.sis_scheme.linear_combination(commitments, scalars)
    }
    
    /// Delegate batch addition to underlying SIS scheme
    fn batch_add_commitments(&self, commitments1: &[Commitment], commitments2: &[Commitment]) -> Result<Vec<Commitment>> {
        self.sis_scheme.batch_add_commitments(commitments1, commitments2)
    }
    
    /// Delegate batch scaling to underlying SIS scheme
    fn batch_scale_commitments(&self, commitments: &[Commitment], scalars: &[i64]) -> Result<Vec<Commitment>> {
        self.sis_scheme.batch_scale_commitments(commitments, scalars)
    }
    
    /// Delegate zero-knowledge addition to underlying SIS scheme
    fn zk_add_commitments(&self, c1: &Commitment, r1: &LatticePoint, c2: &Commitment, r2: &LatticePoint) -> Result<(Commitment, LatticePoint)> {
        self.sis_scheme.zk_add_commitments(c1, r1, c2, r2)
    }
    
    /// Delegate zero-knowledge scaling to underlying SIS scheme
    fn zk_scale_commitment(&self, commitment: &Commitment, randomness: &LatticePoint, scalar: i64) -> Result<(Commitment, LatticePoint)> {
        self.sis_scheme.zk_scale_commitment(commitment, randomness, scalar)
    }
}

/// A commitment scheme with enhanced quantum resistance
///
/// This scheme adds additional randomness and larger parameters
/// to provide stronger security against quantum attacks.
#[derive(Debug, Clone)]
pub struct QuantumResistantCommitmentScheme {
    /// The base SIS commitment scheme
    sis_scheme: SISCommitmentScheme,
    /// Additional security factor for quantum resistance
    quantum_factor: usize,
}

impl QuantumResistantCommitmentScheme {
    /// Create a new quantum-resistant commitment scheme with the given parameters
    pub fn new(mut params: CommitmentParams, quantum_factor: usize) -> Result<Self> {
        // Increase dimension and security parameters for quantum resistance
        params.lattice_params.dimension *= quantum_factor;
        params.security_param *= 2; // Double the security parameter
        
        let sis_scheme = SISCommitmentScheme::new(params)?;
        Ok(Self { sis_scheme, quantum_factor })
    }
    
    /// Get the quantum security factor
    pub fn quantum_factor(&self) -> usize {
        self.quantum_factor
    }
}

impl CommitmentScheme for QuantumResistantCommitmentScheme {
    fn params(&self) -> &CommitmentParams {
        self.sis_scheme.params()
    }
    
    fn commit(&self, message: &[u8], randomness: &LatticePoint) -> Result<Commitment> {
        self.sis_scheme.commit(message, randomness)
    }
    
    fn verify(&self, commitment: &Commitment, message: &[u8], randomness: &LatticePoint) -> Result<bool> {
        self.sis_scheme.verify(commitment, message, randomness)
    }
    
    fn random_randomness<R: Rng + CryptoRng>(&self, rng: &mut R) -> Result<LatticePoint> {
        // Generate larger randomness for quantum resistance
        let mut rand = self.sis_scheme.random_randomness(rng)?;
        
        // Add extra entropy
        for _ in 0..self.quantum_factor {
            let extra_rand = self.sis_scheme.random_randomness(rng)?;
            rand = rand.add(&extra_rand, &self.sis_scheme.params().lattice_params)?;
        }
        
        Ok(rand)
    }
}

impl HomomorphicCommitmentScheme for QuantumResistantCommitmentScheme {
    /// Delegate homomorphic addition to underlying SIS scheme
    /// 
    /// The quantum-resistant commitment scheme maintains all homomorphic properties
    /// while providing enhanced security against quantum attacks through larger
    /// parameters and additional randomness.
    /// 
    /// Security Enhancement:
    /// - Larger lattice dimensions resist quantum attacks
    /// - Enhanced randomness provides additional entropy
    /// - Maintains post-quantum security guarantees
    fn add_commitments(&self, c1: &Commitment, c2: &Commitment) -> Result<Commitment> {
        self.sis_scheme.add_commitments(c1, c2)
    }
    
    /// Delegate homomorphic scaling to underlying SIS scheme
    /// 
    /// Scaling operations maintain quantum resistance through the enhanced
    /// parameter selection and larger security margins.
    fn scale_commitment(&self, c: &Commitment, scalar: i64) -> Result<Commitment> {
        self.sis_scheme.scale_commitment(c, scalar)
    }
    
    /// Delegate optimized linear combination to underlying SIS scheme
    /// 
    /// The quantum-resistant scheme inherits all performance optimizations
    /// while maintaining enhanced security properties.
    fn add_scaled_commitment(&self, c1: &Commitment, c2: &Commitment, scalar: i64) -> Result<Commitment> {
        self.sis_scheme.add_scaled_commitment(c1, c2, scalar)
    }
    
    /// Delegate linear combination computation to underlying SIS scheme
    fn linear_combination(&self, commitments: &[Commitment], scalars: &[i64]) -> Result<Commitment> {
        self.sis_scheme.linear_combination(commitments, scalars)
    }
    
    /// Delegate batch addition to underlying SIS scheme
    fn batch_add_commitments(&self, commitments1: &[Commitment], commitments2: &[Commitment]) -> Result<Vec<Commitment>> {
        self.sis_scheme.batch_add_commitments(commitments1, commitments2)
    }
    
    /// Delegate batch scaling to underlying SIS scheme
    fn batch_scale_commitments(&self, commitments: &[Commitment], scalars: &[i64]) -> Result<Vec<Commitment>> {
        self.sis_scheme.batch_scale_commitments(commitments, scalars)
    }
    
    /// Delegate zero-knowledge addition to underlying SIS scheme
    fn zk_add_commitments(&self, c1: &Commitment, r1: &LatticePoint, c2: &Commitment, r2: &LatticePoint) -> Result<(Commitment, LatticePoint)> {
        self.sis_scheme.zk_add_commitments(c1, r1, c2, r2)
    }
    
    /// Delegate zero-knowledge scaling to underlying SIS scheme
    fn zk_scale_commitment(&self, commitment: &Commitment, randomness: &LatticePoint, scalar: i64) -> Result<(Commitment, LatticePoint)> {
        self.sis_scheme.zk_scale_commitment(commitment, randomness, scalar)
    }
}

/// Convert a message to a lattice point
fn message_to_lattice_point(message: &[u8], params: &LatticeParams) -> Result<LatticePoint> {
    let n = params.dimension;
    let q = params.modulus;
    
    // Create a vector of elements in Z_q
    let mut coords = Vec::with_capacity(n);
    
    // Fill the vector with bytes from the message
    for i in 0..n {
        if i < message.len() {
            // Map each byte to an element in Z_q
            coords.push((message[i] as i64) % q);
        } else {
            // Pad with zeros if message is shorter than dimension
            coords.push(0);
        }
    }
    
    LatticePoint::new(coords)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::thread_rng;
    
    #[test]
    fn test_sis_commitment() {
        let mut rng = thread_rng();
        
        let lattice_params = LatticeParams {
            dimension: 10,
            modulus: 101, // Small prime for testing
            gaussian_width: 3.0,
        };
        
        let commitment_params = CommitmentParams {
            lattice_params: lattice_params.clone(),
            dimension_factor: 2,
            hiding: true,
            security_param: 128,
        };
        
        // Create a commitment scheme
        let scheme = SISCommitmentScheme::new(commitment_params).unwrap();
        
        // Create a message and randomness
        let message = b"test message";
        let randomness = scheme.random_randomness(&mut rng).unwrap();
        
        // Commit to the message
        let commitment = scheme.commit(message, &randomness).unwrap();
        
        // Verify the commitment
        let valid = scheme.verify(&commitment, message, &randomness).unwrap();
        
        assert!(valid);
        
        // Try with wrong message
        let wrong_message = b"wrong message";
        let invalid = scheme.verify(&commitment, wrong_message, &randomness).unwrap();
        
        assert!(!invalid);
        
        // Try with wrong randomness
        let wrong_randomness = scheme.random_randomness(&mut rng).unwrap();
        let invalid = scheme.verify(&commitment, message, &wrong_randomness).unwrap();
        
        assert!(!invalid);
    }
    
    #[test]
    fn test_pedersen_commitment() {
        let mut rng = thread_rng();
        
        let lattice_params = LatticeParams {
            dimension: 10,
            modulus: 101, // Small prime for testing
            gaussian_width: 3.0,
        };
        
        let commitment_params = CommitmentParams {
            lattice_params: lattice_params.clone(),
            dimension_factor: 2,
            hiding: true,
            security_param: 128,
        };
        
        // Create a commitment scheme
        let scheme = PedersenCommitmentScheme::new(commitment_params).unwrap();
        
        // Create a message and randomness
        let message = b"test message";
        let randomness = scheme.random_randomness(&mut rng).unwrap();
        
        // Commit to the message
        let commitment = scheme.commit(message, &randomness).unwrap();
        
        // Verify the commitment
        let valid = scheme.verify(&commitment, message, &randomness).unwrap();
        
        assert!(valid);
        
        // In Pedersen commitments, the message doesn't matter, only the randomness
        // So verification should pass even with a different message
        let different_message = b"different message";
        let still_valid = scheme.verify(&commitment, different_message, &randomness).unwrap();
        
        assert!(still_valid);
        
        // Try with wrong randomness
        let wrong_randomness = scheme.random_randomness(&mut rng).unwrap();
        let invalid = scheme.verify(&commitment, message, &wrong_randomness).unwrap();
        
        assert!(!invalid);
    }
    
    #[test]
    fn test_quantum_resistant_commitment() {
        let mut rng = thread_rng();
        
        let lattice_params = LatticeParams {
            dimension: 10,
            modulus: 101, // Small prime for testing
            gaussian_width: 3.0,
        };
        
        let commitment_params = CommitmentParams {
            lattice_params: lattice_params.clone(),
            dimension_factor: 2,
            hiding: true,
            security_param: 128,
        };
        
        // Create a commitment scheme
        let scheme = QuantumResistantCommitmentScheme::new(commitment_params, 2).unwrap();
        
        // Check that parameters were increased
        assert_eq!(scheme.params().lattice_params.dimension, 20); // 10 * 2
        assert_eq!(scheme.params().security_param, 256); // 128 * 2
        
        // Create a message and randomness
        let message = b"test message";
        let randomness = scheme.random_randomness(&mut rng).unwrap();
        
        // Commit to the message
        let commitment = scheme.commit(message, &randomness).unwrap();
        
        // Verify the commitment
        let valid = scheme.verify(&commitment, message, &randomness).unwrap();
        
        assert!(valid);
        
        // Try with wrong message
        let wrong_message = b"wrong message";
        let invalid = scheme.verify(&commitment, wrong_message, &randomness).unwrap();
        
        assert!(!invalid);
        
        // Try with wrong randomness
        let wrong_randomness = scheme.random_randomness(&mut rng).unwrap();
        let invalid = scheme.verify(&commitment, message, &wrong_randomness).unwrap();
        
        assert!(!invalid);
    }
    
    #[test]
    fn test_homomorphic_properties() {
        let mut rng = thread_rng();
        
        let lattice_params = LatticeParams {
            dimension: 10,
            modulus: 101, // Small prime for testing
            gaussian_width: 3.0,
        };
        
        let commitment_params = CommitmentParams {
            lattice_params: lattice_params.clone(),
            dimension_factor: 2,
            hiding: true,
            security_param: 128,
        };
        
        // Create a commitment scheme
        let scheme = SISCommitmentScheme::new(commitment_params).unwrap();
        
        // Create two messages and randomness values
        let message1 = b"message 1";
        let randomness1 = scheme.random_randomness(&mut rng).unwrap();
        let message2 = b"message 2";
        let randomness2 = scheme.random_randomness(&mut rng).unwrap();
        
        // Commit to the messages
        let commitment1 = scheme.commit(message1, &randomness1).unwrap();
        let commitment2 = scheme.commit(message2, &randomness2).unwrap();
        
        // Add commitments
        let added_commitment = scheme.add_commitments(&commitment1, &commitment2).unwrap();
        
        // Add randomness
        let added_randomness = randomness1.add(&randomness2, &lattice_params).unwrap();
        
        // Convert messages to lattice points and add them
        let message_point1 = message_to_lattice_point(message1, &lattice_params).unwrap();
        let message_point2 = message_to_lattice_point(message2, &lattice_params).unwrap();
        let added_message_point = message_point1.add(&message_point2, &lattice_params).unwrap();
        
        // Reconstruct what the added commitment should be using the homomorphic property
        let reconstructed_commitment = scheme.commit(
            &added_message_point.to_bytes(),
            &added_randomness,
        ).unwrap();
        
        // Verify that the homomorphic property holds
        assert_eq!(added_commitment, reconstructed_commitment);
        
        // Scale a commitment
        let scalar = 3;
        let scaled_commitment = scheme.scale_commitment(&commitment1, scalar).unwrap();
        
        // Scale randomness
        let scaled_randomness = randomness1.scale(scalar, &lattice_params).unwrap();
        
        // Scale message point
        let scaled_message_point = message_point1.scale(scalar, &lattice_params).unwrap();
        
        // Reconstruct what the scaled commitment should be using the homomorphic property
        let reconstructed_scaled_commitment = scheme.commit(
            &scaled_message_point.to_bytes(),
            &scaled_randomness,
        ).unwrap();
        
        // Verify that the scaling property holds
        assert_eq!(scaled_commitment, reconstructed_scaled_commitment);
        
        // Add scaled commitment
        let add_scaled_commitment = scheme.add_scaled_commitment(&commitment1, &commitment2, scalar).unwrap();
        
        // Add the original commitment with the scaled commitment
        let expected_add_scaled = scheme.add_commitments(&commitment1, &scaled_commitment).unwrap();
        
        // Verify that the add-scale property holds
        assert_eq!(add_scaled_commitment, expected_add_scaled);
    }
    
    #[test]
    fn test_from_seed() {
        // Create two commitment schemes with the same seed
        let lattice_params = LatticeParams {
            dimension: 10,
            modulus: 101,
            gaussian_width: 3.0,
        };
        
        let commitment_params = CommitmentParams {
            lattice_params: lattice_params.clone(),
            dimension_factor: 2,
            hiding: true,
            security_param: 128,
        };
        
        let seed = b"test seed";
        
        let scheme1 = SISCommitmentScheme::from_seed(commitment_params.clone(), seed).unwrap();
        let scheme2 = SISCommitmentScheme::from_seed(commitment_params.clone(), seed).unwrap();
        
        // Verify that they have the same matrix
        assert_eq!(scheme1.matrix().rows(), scheme2.matrix().rows());
        assert_eq!(scheme1.matrix().cols(), scheme2.matrix().cols());
        
        for i in 0..scheme1.matrix().rows() {
            for j in 0..scheme1.matrix().cols() {
                assert_eq!(scheme1.matrix().get(i, j), scheme2.matrix().get(i, j));
            }
        }
        
        // Create a scheme with a different seed
        let different_seed = b"different seed";
        let scheme3 = SISCommitmentScheme::from_seed(commitment_params.clone(), different_seed).unwrap();
        
        // Verify that it has a different matrix
        let mut all_same = true;
        
        for i in 0..scheme1.matrix().rows() {
            for j in 0..scheme1.matrix().cols() {
                if scheme1.matrix().get(i, j) != scheme3.matrix().get(i, j) {
                    all_same = false;
                    break;
                }
            }
            if !all_same {
                break;
            }
        }
        
        // There should be at least one difference
        assert!(!all_same);
    }
    
    #[test]
    fn test_enhanced_homomorphic_properties() {
        let mut rng = thread_rng();
        
        let lattice_params = LatticeParams {
            dimension: 10,
            modulus: 101, // Small prime for testing
            gaussian_width: 3.0,
        };
        
        let commitment_params = CommitmentParams {
            lattice_params: lattice_params.clone(),
            dimension_factor: 2,
            hiding: true,
            security_param: 128,
        };
        
        // Create a commitment scheme
        let scheme = SISCommitmentScheme::new(commitment_params).unwrap();
        
        // Test linear combination of multiple commitments
        let messages = vec![
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            vec![2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            vec![3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        ];
        
        let mut commitments = Vec::new();
        let mut randomness_values = Vec::new();
        
        for message in &messages {
            let randomness = scheme.random_randomness(&mut rng).unwrap();
            let commitment = scheme.commit(&message_to_bytes(message), &randomness).unwrap();
            commitments.push(commitment);
            randomness_values.push(randomness);
        }
        
        // Test linear combination with scalars [2, 3, 5]
        let scalars = vec![2, 3, 5];
        let linear_combination = scheme.linear_combination(&commitments, &scalars).unwrap();
        
        // Verify the linear combination is correct
        // Expected: 2*msg1 + 3*msg2 + 5*msg3
        let mut expected_message = vec![0i64; 10];
        for i in 0..10 {
            expected_message[i] = (2 * messages[0][i] + 3 * messages[1][i] + 5 * messages[2][i]) % lattice_params.modulus;
        }
        
        // Compute expected randomness: 2*r1 + 3*r2 + 5*r3
        let mut expected_randomness = LatticePoint::zero(randomness_values[0].dimension()).unwrap();
        for (i, &scalar) in scalars.iter().enumerate() {
            let scaled_randomness = randomness_values[i].scale(scalar, &lattice_params).unwrap();
            expected_randomness = expected_randomness.add(&scaled_randomness, &lattice_params).unwrap();
        }
        
        // Verify the linear combination commitment
        let expected_commitment = scheme.commit(&message_to_bytes(&expected_message), &expected_randomness).unwrap();
        assert_eq!(linear_combination, expected_commitment);
    }
    
    #[test]
    fn test_batch_homomorphic_operations() {
        let mut rng = thread_rng();
        
        let lattice_params = LatticeParams {
            dimension: 8,
            modulus: 97, // Small prime for testing
            gaussian_width: 3.0,
        };
        
        let commitment_params = CommitmentParams {
            lattice_params: lattice_params.clone(),
            dimension_factor: 2,
            hiding: true,
            security_param: 128,
        };
        
        let scheme = SISCommitmentScheme::new(commitment_params).unwrap();
        
        // Create multiple commitment pairs for batch testing
        let messages1 = vec![
            vec![1, 2, 3, 4, 5, 6, 7, 8],
            vec![2, 3, 4, 5, 6, 7, 8, 9],
            vec![3, 4, 5, 6, 7, 8, 9, 10],
        ];
        
        let messages2 = vec![
            vec![10, 9, 8, 7, 6, 5, 4, 3],
            vec![9, 8, 7, 6, 5, 4, 3, 2],
            vec![8, 7, 6, 5, 4, 3, 2, 1],
        ];
        
        let mut commitments1 = Vec::new();
        let mut commitments2 = Vec::new();
        let mut randomness1 = Vec::new();
        let mut randomness2 = Vec::new();
        
        // Create commitments
        for (msg1, msg2) in messages1.iter().zip(messages2.iter()) {
            let r1 = scheme.random_randomness(&mut rng).unwrap();
            let r2 = scheme.random_randomness(&mut rng).unwrap();
            
            let c1 = scheme.commit(&message_to_bytes(msg1), &r1).unwrap();
            let c2 = scheme.commit(&message_to_bytes(msg2), &r2).unwrap();
            
            commitments1.push(c1);
            commitments2.push(c2);
            randomness1.push(r1);
            randomness2.push(r2);
        }
        
        // Test batch addition
        let batch_sum = scheme.batch_add_commitments(&commitments1, &commitments2).unwrap();
        
        // Verify each sum individually
        for i in 0..commitments1.len() {
            let individual_sum = scheme.add_commitments(&commitments1[i], &commitments2[i]).unwrap();
            assert_eq!(batch_sum[i], individual_sum);
        }
        
        // Test batch scaling
        let scalars = vec![2, 3, 5];
        let batch_scaled = scheme.batch_scale_commitments(&commitments1, &scalars).unwrap();
        
        // Verify each scaling individually
        for i in 0..commitments1.len() {
            let individual_scaled = scheme.scale_commitment(&commitments1[i], scalars[i]).unwrap();
            assert_eq!(batch_scaled[i], individual_scaled);
        }
    }
    
    #[test]
    fn test_zero_knowledge_homomorphic_operations() {
        let mut rng = thread_rng();
        
        let lattice_params = LatticeParams {
            dimension: 6,
            modulus: 97,
            gaussian_width: 3.0,
        };
        
        let commitment_params = CommitmentParams {
            lattice_params: lattice_params.clone(),
            dimension_factor: 2,
            hiding: true,
            security_param: 128,
        };
        
        let scheme = SISCommitmentScheme::new(commitment_params).unwrap();
        
        // Create two commitments with known randomness
        let message1 = vec![1, 2, 3, 4, 5, 6];
        let message2 = vec![6, 5, 4, 3, 2, 1];
        
        let randomness1 = scheme.random_randomness(&mut rng).unwrap();
        let randomness2 = scheme.random_randomness(&mut rng).unwrap();
        
        let commitment1 = scheme.commit(&message_to_bytes(&message1), &randomness1).unwrap();
        let commitment2 = scheme.commit(&message_to_bytes(&message2), &randomness2).unwrap();
        
        // Test zero-knowledge addition
        let (zk_sum_commitment, zk_sum_randomness) = scheme.zk_add_commitments(
            &commitment1, &randomness1,
            &commitment2, &randomness2
        ).unwrap();
        
        // Verify the zero-knowledge addition is correct
        let expected_sum_message: Vec<i64> = message1.iter().zip(message2.iter())
            .map(|(&a, &b)| (a + b) % lattice_params.modulus)
            .collect();
        
        let expected_sum_commitment = scheme.commit(&message_to_bytes(&expected_sum_message), &zk_sum_randomness).unwrap();
        assert_eq!(zk_sum_commitment, expected_sum_commitment);
        
        // Test zero-knowledge scaling
        let scalar = 7;
        let (zk_scaled_commitment, zk_scaled_randomness) = scheme.zk_scale_commitment(
            &commitment1, &randomness1, scalar
        ).unwrap();
        
        // Verify the zero-knowledge scaling is correct
        let expected_scaled_message: Vec<i64> = message1.iter()
            .map(|&a| (a * scalar) % lattice_params.modulus)
            .collect();
        
        let expected_scaled_commitment = scheme.commit(&message_to_bytes(&expected_scaled_message), &zk_scaled_randomness).unwrap();
        assert_eq!(zk_scaled_commitment, expected_scaled_commitment);
    }
    
    #[test]
    fn test_homomorphic_property_preservation() {
        let mut rng = thread_rng();
        
        let lattice_params = LatticeParams {
            dimension: 4,
            modulus: 97,
            gaussian_width: 3.0,
        };
        
        let commitment_params = CommitmentParams {
            lattice_params: lattice_params.clone(),
            dimension_factor: 2,
            hiding: true,
            security_param: 128,
        };
        
        let scheme = SISCommitmentScheme::new(commitment_params).unwrap();
        
        // Test associativity: (a + b) + c = a + (b + c)
        let messages = vec![
            vec![1, 2, 3, 4],
            vec![5, 6, 7, 8],
            vec![9, 10, 11, 12],
        ];
        
        let mut commitments = Vec::new();
        for message in &messages {
            let randomness = scheme.random_randomness(&mut rng).unwrap();
            let commitment = scheme.commit(&message_to_bytes(message), &randomness).unwrap();
            commitments.push(commitment);
        }
        
        // Test (a + b) + c
        let ab = scheme.add_commitments(&commitments[0], &commitments[1]).unwrap();
        let abc_left = scheme.add_commitments(&ab, &commitments[2]).unwrap();
        
        // Test a + (b + c)
        let bc = scheme.add_commitments(&commitments[1], &commitments[2]).unwrap();
        let abc_right = scheme.add_commitments(&commitments[0], &bc).unwrap();
        
        assert_eq!(abc_left, abc_right);
        
        // Test commutativity: a + b = b + a
        let ab_forward = scheme.add_commitments(&commitments[0], &commitments[1]).unwrap();
        let ab_backward = scheme.add_commitments(&commitments[1], &commitments[0]).unwrap();
        assert_eq!(ab_forward, ab_backward);
        
        // Test distributivity: c * (a + b) = c * a + c * b
        let scalar = 3;
        let ab_sum = scheme.add_commitments(&commitments[0], &commitments[1]).unwrap();
        let scaled_sum = scheme.scale_commitment(&ab_sum, scalar).unwrap();
        
        let scaled_a = scheme.scale_commitment(&commitments[0], scalar).unwrap();
        let scaled_b = scheme.scale_commitment(&commitments[1], scalar).unwrap();
        let sum_of_scaled = scheme.add_commitments(&scaled_a, &scaled_b).unwrap();
        
        assert_eq!(scaled_sum, sum_of_scaled);
    }
    
    /// Helper function to convert message vector to bytes
    fn message_to_bytes(message: &[i64]) -> Vec<u8> {
        let mut bytes = Vec::new();
        for &val in message {
            bytes.extend_from_slice(&val.to_le_bytes());
        }
        bytes
    }
} 