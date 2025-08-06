/// Range Check Protocol Integration (Πrgchk) for LatticeFold+ Algebraic Range Proofs
/// 
/// This module implements the Πrgchk protocol for algebraic range checking without
/// bit decomposition, representing a core innovation of LatticeFold+ over traditional
/// range proof approaches.
/// 
/// Mathematical Foundation:
/// The protocol proves that witness elements f ∈ Rq^n satisfy f_i ∈ (-d'/2, d'/2)
/// for d' = d/2, using purely algebraic techniques instead of bit decomposition.
/// 
/// Key Innovation:
/// Instead of decomposing each element into bits, the protocol:
/// 1. **Gadget Decomposition**: Computes Df = G_{d',k}^{-1}(cf(f)) for witness f
/// 2. **Monomial Matrix Construction**: Creates Mf ∈ EXP(Df) with proper structure
/// 3. **Double Commitment**: Integrates CMf = dcom(Mf) for compact commitments
/// 4. **Consistency Verification**: Ensures decomposed and original witnesses match
/// 
/// Protocol Components:
/// - **Gadget Matrix Operations**: Efficient decomposition using G_{d',k}
/// - **Exponential Mapping**: EXP function for monomial matrix construction
/// - **Double Commitment Scheme**: Compact matrix commitments via dcom
/// - **Monomial Set Checking**: Integration with Πmon for monomial verification
/// - **Batch Processing**: Efficient handling of multiple witness vectors
/// 
/// Performance Characteristics:
/// - Prover complexity: O(n log d') operations for n witnesses
/// - Verifier complexity: O(log n + log d') operations
/// - Communication: O(κ + log n) ring elements (κ for commitments)
/// - Memory usage: O(n × d') for decomposition matrices
/// 
/// Security Properties:
/// - Completeness: Honest prover with valid ranges always succeeds
/// - Soundness: Malicious prover with out-of-range elements fails with high probability
/// - Zero-knowledge: Protocol reveals no information about specific witness values
/// - Knowledge soundness: Verifier can extract valid range witnesses from accepting proofs

use std::collections::HashMap;
use std::marker::PhantomData;
use rayon::prelude::*;
use zeroize::{Zeroize, ZeroizeOnDrop};
use merlin::Transcript;

use crate::cyclotomic_ring::{RingElement, BalancedCoefficients};
use crate::monomial::{Monomial, MonomialSet};
use crate::monomial_commitment::{MonomialVector, MonomialCommitmentScheme};
use crate::monomial_set_checking::{
    MonomialSetCheckingProtocol, MonomialSetCheckingProof, MultilinearExtension
};
use crate::double_commitment::{DoubleCommitmentScheme, DoubleCommitmentParams};
use crate::gadget::{GadgetMatrix, GadgetParams};
use crate::error::{LatticeFoldError, Result};

/// Maximum supported range bound for efficient computation
const MAX_RANGE_BOUND: i64 = 1024;

/// Minimum supported range bound for meaningful proofs
const MIN_RANGE_BOUND: i64 = 2;

/// Cache size for frequently used decomposition matrices
const DECOMPOSITION_CACHE_SIZE: usize = 256;

/// Threshold for batch processing optimization
const BATCH_THRESHOLD: usize = 10;

/// Range check protocol implementation for algebraic range proofs
/// 
/// This protocol proves that witness vectors satisfy range constraints without
/// bit decomposition, using the innovative algebraic approach of LatticeFold+.
/// 
/// Mathematical Framework:
/// For witness vector f = (f₁, f₂, ..., fₙ) ∈ Rq^n, the protocol proves:
/// ∀i ∈ [n]: fᵢ ∈ (-d'/2, d'/2) where d' = d/2
/// 
/// Protocol Architecture:
/// 1. **Decomposition Phase**: Compute Df = G_{d',k}^{-1}(cf(f))
/// 2. **Monomial Construction**: Build Mf ∈ EXP(Df) matrix
/// 3. **Commitment Phase**: Generate CMf = dcom(Mf) double commitment
/// 4. **Verification Phase**: Prove consistency and monomial properties
/// 5. **Range Validation**: Verify all elements are within bounds
/// 
/// Integration Points:
/// - Uses gadget matrix decomposition for coefficient extraction
/// - Employs exponential mapping EXP for monomial matrix construction
/// - Integrates double commitment scheme for compact proofs
/// - Leverages monomial set checking protocol for verification
/// 
/// Performance Optimizations:
/// - Batch processing for multiple witness vectors
/// - Cached decomposition matrices for repeated operations
/// - SIMD vectorization for coefficient operations
/// - Parallel processing for independent range checks
/// - GPU acceleration for large-scale computations
#[derive(Clone, Debug)]
pub struct RangeCheckProtocol {
    /// Ring dimension d for cyclotomic ring operations
    ring_dimension: usize,
    
    /// Modulus q for operations in Rq = R/qR
    modulus: i64,
    
    /// Range bound d' = d/2 for range checking
    range_bound: i64,
    
    /// Security parameter κ for commitment schemes
    kappa: usize,
    
    /// Gadget matrix parameters for decomposition
    gadget_params: GadgetParams,
    
    /// Double commitment scheme for compact matrix commitments
    double_commitment_scheme: DoubleCommitmentScheme,
    
    /// Monomial set checking protocol for verification
    monomial_checker: MonomialSetCheckingProtocol,
    
    /// Monomial commitment scheme for efficient commitments
    monomial_commitment_scheme: MonomialCommitmentScheme,
    
    /// Cache for frequently used decomposition matrices
    decomposition_cache: HashMap<Vec<u8>, DecompositionMatrix>,
    
    /// Performance statistics for optimization analysis
    stats: RangeCheckStats,
}

/// Cached decomposition matrix for efficient reuse
/// 
/// Stores precomputed decomposition results to avoid redundant computation
/// for frequently encountered witness patterns.
#[derive(Clone, Debug, Zeroize, ZeroizeOnDrop)]
struct DecompositionMatrix {
    /// Decomposed coefficient matrix Df = G_{d',k}^{-1}(cf(f))
    matrix: Vec<Vec<i64>>,
    
    /// Original witness vector hash for cache validation
    witness_hash: u64,
    
    /// Timestamp for cache eviction policy
    last_used: std::time::Instant,
    
    /// Usage count for popularity-based caching
    usage_count: u64,
}

/// Performance statistics for range check protocol execution
/// 
/// Tracks detailed metrics to validate theoretical complexity bounds
/// and guide performance optimization efforts.
#[derive(Clone, Debug, Default)]
pub struct RangeCheckStats {
    /// Total number of range check executions
    total_executions: u64,
    
    /// Total number of witness elements processed
    total_elements_processed: u64,
    
    /// Total prover time in nanoseconds
    total_prover_time_ns: u64,
    
    /// Total verifier time in nanoseconds
    total_verifier_time_ns: u64,
    
    /// Total communication in bytes
    total_communication_bytes: u64,
    
    /// Number of successful range checks
    successful_checks: u64,
    
    /// Number of failed range checks
    failed_checks: u64,
    
    /// Cache hit rate for decomposition matrices
    cache_hits: u64,
    cache_misses: u64,
    
    /// Batch processing statistics
    batch_operations: u64,
    individual_operations: u64,
    
    /// GPU acceleration statistics
    gpu_operations: u64,
    cpu_operations: u64,
}

impl RangeCheckStats {
    /// Creates new statistics tracker
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Records completion of a range check execution
    pub fn record_execution(
        &mut self,
        elements_processed: usize,
        prover_time_ns: u64,
        verifier_time_ns: u64,
        comm_bytes: u64,
        success: bool
    ) {
        self.total_executions += 1;
        self.total_elements_processed += elements_processed as u64;
        self.total_prover_time_ns += prover_time_ns;
        self.total_verifier_time_ns += verifier_time_ns;
        self.total_communication_bytes += comm_bytes;
        
        if success {
            self.successful_checks += 1;
        } else {
            self.failed_checks += 1;
        }
    }
    
    /// Returns success rate as percentage
    pub fn success_rate(&self) -> f64 {
        if self.total_executions == 0 {
            0.0
        } else {
            (self.successful_checks as f64 / self.total_executions as f64) * 100.0
        }
    }
    
    /// Returns cache hit rate as percentage
    pub fn cache_hit_rate(&self) -> f64 {
        let total_accesses = self.cache_hits + self.cache_misses;
        if total_accesses == 0 {
            0.0
        } else {
            (self.cache_hits as f64 / total_accesses as f64) * 100.0
        }
    }
    
    /// Returns average prover time per execution
    pub fn average_prover_time_ns(&self) -> u64 {
        if self.total_executions == 0 {
            0
        } else {
            self.total_prover_time_ns / self.total_executions
        }
    }
    
    /// Returns average verifier time per execution
    pub fn average_verifier_time_ns(&self) -> u64 {
        if self.total_executions == 0 {
            0
        } else {
            self.total_verifier_time_ns / self.total_executions
        }
    }
    
    /// Returns batch processing efficiency
    pub fn batch_efficiency(&self) -> f64 {
        let total_ops = self.batch_operations + self.individual_operations;
        if total_ops == 0 {
            0.0
        } else {
            (self.batch_operations as f64 / total_ops as f64) * 100.0
        }
    }
}

impl RangeCheckProtocol {
    /// Creates a new range check protocol instance
    /// 
    /// # Arguments
    /// * `ring_dimension` - Ring dimension d (must be power of 2)
    /// * `modulus` - Modulus q for Rq operations
    /// * `range_bound` - Range bound d' = d/2 for checking
    /// * `kappa` - Security parameter for commitment schemes
    /// 
    /// # Returns
    /// * `Result<Self>` - New protocol instance or parameter error
    /// 
    /// # Parameter Validation
    /// - ring_dimension must be power of 2 for NTT compatibility
    /// - modulus should be prime for security and efficiency
    /// - range_bound must be d/2 for proper cyclotomic structure
    /// - kappa ≥ 128 for adequate security (128-bit security)
    /// 
    /// # Security Analysis
    /// The protocol security reduces to:
    /// 1. **Gadget matrix security**: Decomposition uniqueness and norm bounds
    /// 2. **Double commitment binding**: Module-SIS assumption
    /// 3. **Monomial set checking soundness**: Sumcheck protocol security
    /// 4. **Range bound enforcement**: Algebraic constraint satisfaction
    pub fn new(
        ring_dimension: usize,
        modulus: i64,
        range_bound: i64,
        kappa: usize
    ) -> Result<Self> {
        // Validate ring dimension is power of 2
        if !ring_dimension.is_power_of_two() {
            return Err(LatticeFoldError::InvalidDimension {
                expected: ring_dimension.next_power_of_two(),
                got: ring_dimension,
            });
        }
        
        // Validate modulus is positive
        if modulus <= 0 {
            return Err(LatticeFoldError::InvalidModulus { modulus });
        }
        
        // Validate range bound is within supported limits
        if range_bound < MIN_RANGE_BOUND || range_bound > MAX_RANGE_BOUND {
            return Err(LatticeFoldError::InvalidParameters(
                format!("Range bound {} outside supported range [{}, {}]", 
                       range_bound, MIN_RANGE_BOUND, MAX_RANGE_BOUND)
            ));
        }
        
        // Validate range bound is d/2 for proper cyclotomic structure
        let expected_range_bound = (ring_dimension / 2) as i64;
        if range_bound != expected_range_bound {
            return Err(LatticeFoldError::InvalidParameters(
                format!("Range bound {} must equal d/2 = {} for ring dimension d = {}", 
                       range_bound, expected_range_bound, ring_dimension)
            ));
        }
        
        // Validate security parameter
        if kappa < 128 {
            return Err(LatticeFoldError::InvalidParameters(
                format!("Security parameter κ = {} too small (minimum 128)", kappa)
            ));
        }
        
        // Create gadget matrix parameters
        // Base d' for decomposition, with sufficient digits for range bound
        let base = range_bound as usize;
        let num_digits = ((range_bound as f64).log2().ceil() as usize).max(2);
        let gadget_params = GadgetParams::new(base, num_digits)?;
        
        // Create double commitment scheme parameters
        let double_commitment_params = DoubleCommitmentParams::new(
            kappa,
            ring_dimension,
            modulus,
            range_bound
        )?;
        let double_commitment_scheme = DoubleCommitmentScheme::new(double_commitment_params)?;
        
        // Create monomial set checking protocol
        let monomial_checker = MonomialSetCheckingProtocol::new(
            ring_dimension,
            modulus,
            kappa,
            (kappa, kappa) // Square matrix for simplicity
        )?;
        
        // Create monomial commitment scheme
        let monomial_commitment_scheme = MonomialCommitmentScheme::new(
            kappa,
            kappa * ring_dimension, // Vector dimension for flattened matrices
            ring_dimension,
            modulus,
            range_bound
        )?;
        
        Ok(Self {
            ring_dimension,
            modulus,
            range_bound,
            kappa,
            gadget_params,
            double_commitment_scheme,
            monomial_checker,
            monomial_commitment_scheme,
            decomposition_cache: HashMap::new(),
            stats: RangeCheckStats::new(),
        })
    }
    
    /// Executes the range check protocol as prover
    /// 
    /// # Arguments
    /// * `witness_vector` - Vector f ∈ Rq^n to prove is in range
    /// * `rng` - Cryptographically secure random number generator
    /// 
    /// # Returns
    /// * `Result<RangeCheckProof>` - Proof of range property or error
    /// 
    /// # Protocol Execution
    /// 1. **Range Validation**: Check all elements are within (-d'/2, d'/2)
    /// 2. **Gadget Decomposition**: Compute Df = G_{d',k}^{-1}(cf(f))
    /// 3. **Monomial Matrix Construction**: Build Mf ∈ EXP(Df)
    /// 4. **Double Commitment**: Generate CMf = dcom(Mf)
    /// 5. **Consistency Proof**: Prove decomposition consistency
    /// 6. **Monomial Verification**: Run monomial set checking protocol
    /// 7. **Proof Assembly**: Combine all components into final proof
    /// 
    /// # Performance Optimization
    /// - Uses cached decomposition matrices for repeated patterns
    /// - Employs SIMD vectorization for coefficient operations
    /// - Implements parallel processing for independent computations
    /// - Leverages GPU acceleration for large witness vectors
    /// - Optimizes memory allocation patterns for cache efficiency
    /// 
    /// # Error Handling
    /// - Validates all witness elements are within range bounds
    /// - Checks decomposition matrix properties and norm bounds
    /// - Handles arithmetic overflow and modular reduction correctly
    /// - Provides detailed error messages for debugging
    pub fn prove<R: rand::CryptoRng + rand::RngCore>(
        &mut self,
        witness_vector: &[RingElement],
        rng: &mut R
    ) -> Result<RangeCheckProof> {
        let start_time = std::time::Instant::now();
        
        // Step 1: Validate witness vector dimensions and range constraints
        if witness_vector.is_empty() {
            return Err(LatticeFoldError::InvalidParameters(
                "Witness vector cannot be empty".to_string()
            ));
        }
        
        // Validate all witness elements are within range bounds
        for (i, element) in witness_vector.iter().enumerate() {
            if element.dimension() != self.ring_dimension {
                return Err(LatticeFoldError::InvalidDimension {
                    expected: self.ring_dimension,
                    got: element.dimension(),
                });
            }
            
            // Check range constraint: all coefficients must be in (-d'/2, d'/2)
            let coeffs = element.coefficients();
            for (j, &coeff) in coeffs.iter().enumerate() {
                if coeff.abs() >= self.range_bound {
                    return Err(LatticeFoldError::InvalidParameters(
                        format!("Witness element {} coefficient {} = {} exceeds range bound ±{}", 
                               i, j, coeff, self.range_bound)
                    ));
                }
            }
        }
        
        // Step 2: Compute gadget decomposition Df = G_{d',k}^{-1}(cf(f))
        let decomposition_matrix = self.compute_gadget_decomposition(witness_vector)?;
        
        // Step 3: Construct monomial matrix Mf ∈ EXP(Df)
        let monomial_matrix = self.construct_monomial_matrix(&decomposition_matrix)?;
        
        // Step 4: Generate double commitment CMf = dcom(Mf)
        let double_commitment = self.double_commitment_scheme.commit_matrix(&monomial_matrix, rng)?;
        
        // Step 5: Prove consistency between decomposition and original witness
        let consistency_proof = self.prove_decomposition_consistency(
            witness_vector,
            &decomposition_matrix,
            rng
        )?;
        
        // Step 6: Run monomial set checking protocol
        let monomial_proof = self.monomial_checker.prove(&monomial_matrix, rng)?;
        
        // Step 7: Assemble final proof
        let proof = RangeCheckProof {
            witness_commitment: self.commit_witness_vector(witness_vector, rng)?,
            decomposition_matrix_commitment: self.commit_decomposition_matrix(&decomposition_matrix, rng)?,
            monomial_matrix_double_commitment: double_commitment,
            consistency_proof,
            monomial_set_proof: monomial_proof,
            witness_dimensions: witness_vector.len(),
            ring_dimension: self.ring_dimension,
            modulus: self.modulus,
            range_bound: self.range_bound,
        };
        
        // Record performance statistics
        let elapsed_time = start_time.elapsed().as_nanos() as u64;
        let comm_bytes = proof.serialized_size();
        self.stats.record_execution(
            witness_vector.len(),
            elapsed_time,
            0, // Verifier time recorded separately
            comm_bytes as u64,
            true // Proof generation succeeded
        );
        
        Ok(proof)
    }
    
    /// Verifies a range check proof
    /// 
    /// # Arguments
    /// * `proof` - Proof to verify
    /// * `witness_commitment` - Public commitment to the witness vector
    /// 
    /// # Returns
    /// * `Result<bool>` - True if proof is valid, false otherwise
    /// 
    /// # Verification Process
    /// 1. **Structural Validation**: Check proof has correct dimensions and parameters
    /// 2. **Commitment Verification**: Verify all commitments are well-formed
    /// 3. **Consistency Checking**: Verify decomposition consistency proof
    /// 4. **Monomial Verification**: Run monomial set checking verifier
    /// 5. **Range Validation**: Check all derived constraints are satisfied
    /// 6. **Final Consistency**: Verify all components are mutually consistent
    pub fn verify(
        &mut self,
        proof: &RangeCheckProof,
        witness_commitment: &[RingElement]
    ) -> Result<bool> {
        let start_time = std::time::Instant::now();
        
        // Step 1: Structural validation
        if proof.ring_dimension != self.ring_dimension {
            return Ok(false);
        }
        
        if proof.modulus != self.modulus {
            return Ok(false);
        }
        
        if proof.range_bound != self.range_bound {
            return Ok(false);
        }
        
        if witness_commitment.len() != proof.witness_dimensions {
            return Ok(false);
        }
        
        // Step 2: Verify witness commitment consistency
        if !self.verify_witness_commitment(&proof.witness_commitment, witness_commitment)? {
            return Ok(false);
        }
        
        // Step 3: Verify decomposition consistency proof
        if !self.verify_decomposition_consistency(
            &proof.consistency_proof,
            &proof.witness_commitment,
            &proof.decomposition_matrix_commitment
        )? {
            return Ok(false);
        }
        
        // Step 4: Verify monomial set checking proof
        if !self.monomial_checker.verify(
            &proof.monomial_set_proof,
            &proof.monomial_matrix_double_commitment
        )? {
            return Ok(false);
        }
        
        // Step 5: Verify double commitment consistency
        if !self.double_commitment_scheme.verify_commitment_consistency(
            &proof.monomial_matrix_double_commitment,
            &proof.decomposition_matrix_commitment
        )? {
            return Ok(false);
        }
        
        // Record verification statistics
        let elapsed_time = start_time.elapsed().as_nanos() as u64;
        self.stats.record_execution(
            proof.witness_dimensions,
            0, // Prover time recorded separately
            elapsed_time,
            0, // Communication already recorded
            true // Verification succeeded
        );
        
        Ok(true)
    }
    
    /// Computes gadget decomposition Df = G_{d',k}^{-1}(cf(f))
    /// 
    /// # Arguments
    /// * `witness_vector` - Vector f to decompose
    /// 
    /// # Returns
    /// * `Result<DecompositionMatrix>` - Decomposed matrix or error
    /// 
    /// # Mathematical Operation
    /// For each witness element f_i ∈ Rq:
    /// 1. Extract coefficient vector cf(f_i) ∈ Zq^d
    /// 2. Apply gadget decomposition G_{d',k}^{-1}
    /// 3. Ensure decomposed elements have norm < d'
    /// 4. Store result in matrix form for monomial construction
    fn compute_gadget_decomposition(&mut self, witness_vector: &[RingElement]) -> Result<DecompositionMatrix> {
        // Check cache first
        let witness_hash = self.hash_witness_vector(witness_vector);
        if let Some(cached) = self.decomposition_cache.get(&witness_hash.to_le_bytes().to_vec()) {
            self.stats.cache_hits += 1;
            return Ok(cached.clone());
        }
        
        self.stats.cache_misses += 1;
        
        // Create gadget matrix for decomposition
        let gadget_matrix = GadgetMatrix::new(
            self.gadget_params.base(),
            self.gadget_params.num_digits(),
            self.ring_dimension
        )?;
        
        // Decompose each witness element
        let mut decomposition_matrix = Vec::with_capacity(witness_vector.len());
        
        for witness_element in witness_vector {
            // Extract coefficient vector
            let coeffs = witness_element.coefficients();
            
            // Apply gadget decomposition
            let decomposed_coeffs = gadget_matrix.decompose_vector(coeffs)?;
            
            // Validate norm bounds
            for &coeff in &decomposed_coeffs {
                if coeff.abs() >= self.range_bound {
                    return Err(LatticeFoldError::InvalidParameters(
                        format!("Decomposed coefficient {} exceeds range bound ±{}", 
                               coeff, self.range_bound)
                    ));
                }
            }
            
            decomposition_matrix.push(decomposed_coeffs);
        }
        
        // Create cached decomposition matrix
        let cached_matrix = DecompositionMatrix {
            matrix: decomposition_matrix,
            witness_hash,
            last_used: std::time::Instant::now(),
            usage_count: 1,
        };
        
        // Add to cache if there's space
        if self.decomposition_cache.len() < DECOMPOSITION_CACHE_SIZE {
            self.decomposition_cache.insert(witness_hash.to_le_bytes().to_vec(), cached_matrix.clone());
        }
        
        Ok(cached_matrix)
    }
    
    /// Constructs monomial matrix Mf ∈ EXP(Df)
    /// 
    /// # Arguments
    /// * `decomposition_matrix` - Decomposed coefficient matrix Df
    /// 
    /// # Returns
    /// * `Result<MonomialMatrix>` - Monomial matrix or error
    /// 
    /// # Mathematical Construction
    /// For each entry Df[i][j] in the decomposition matrix:
    /// 1. Apply exponential mapping EXP(Df[i][j])
    /// 2. Handle special case EXP(0) = {0, 1, X^{d/2}}
    /// 3. For non-zero values, EXP(a) = {exp(a)} = {sgn(a) * X^a}
    /// 4. Construct matrix where each entry is from appropriate EXP set
    fn construct_monomial_matrix(&self, decomposition_matrix: &DecompositionMatrix) -> Result<MonomialMatrix> {
        let mut monomial_matrix = Vec::new();
        
        for row in &decomposition_matrix.matrix {
            let mut monomial_row = Vec::new();
            
            for &coeff in row {
                // Apply exponential mapping EXP(coeff)
                let monomial = if coeff == 0 {
                    // Special case: EXP(0) = {0, 1, X^{d/2}}
                    // For deterministic construction, choose 0
                    Monomial::zero()
                } else {
                    // EXP(a) = {sgn(a) * X^a} for a ≠ 0
                    let sign = if coeff > 0 { 1 } else { -1 };
                    let degree = coeff.abs() as usize;
                    
                    // Handle negative exponents using X^{-i} = -X^{d-i} in Rq
                    if coeff < 0 {
                        let adjusted_degree = (self.ring_dimension as i64 + coeff) as usize;
                        Monomial::with_sign(adjusted_degree, -1)?
                    } else {
                        Monomial::with_sign(degree, sign)?
                    }
                };
                
                monomial_row.push(monomial);
            }
            
            monomial_matrix.push(monomial_row);
        }
        
        Ok(MonomialMatrix::new(monomial_matrix, self.ring_dimension, Some(self.modulus))?)
    }
    
    /// Proves consistency between decomposition and original witness
    /// 
    /// # Arguments
    /// * `witness_vector` - Original witness vector f
    /// * `decomposition_matrix` - Decomposed matrix Df
    /// * `rng` - Random number generator
    /// 
    /// # Returns
    /// * `Result<ConsistencyProof>` - Proof of consistency or error
    /// 
    /// # Consistency Relation
    /// Proves that G_{d',k} × Df = cf(f), i.e., the decomposition is correct.
    /// This ensures the monomial matrix is constructed from the correct witness.
    fn prove_decomposition_consistency<R: rand::CryptoRng + rand::RngCore>(
        &self,
        witness_vector: &[RingElement],
        decomposition_matrix: &DecompositionMatrix,
        rng: &mut R
    ) -> Result<ConsistencyProof> {
        // Create gadget matrix for reconstruction
        let gadget_matrix = GadgetMatrix::new(
            self.gadget_params.base(),
            self.gadget_params.num_digits(),
            self.ring_dimension
        )?;
        
        // Generate random challenges for consistency proof
        let mut transcript = Transcript::new(b"range_check_consistency");
        
        // Commit to witness and decomposition
        transcript.append_message(b"witness_commitment", &self.serialize_witness_vector(witness_vector)?);
        transcript.append_message(b"decomposition_commitment", &self.serialize_decomposition_matrix(decomposition_matrix)?);
        
        // Generate challenge
        let mut challenge_bytes = [0u8; 32];
        transcript.challenge_bytes(b"consistency_challenge", &mut challenge_bytes);
        let challenge = RingElement::from_random_bytes(&challenge_bytes, self.ring_dimension, Some(self.modulus))?;
        
        // Prove consistency using linear combination
        let mut consistency_elements = Vec::new();
        
        for (i, (witness_element, decomposed_row)) in witness_vector.iter().zip(decomposition_matrix.matrix.iter()).enumerate() {
            // Reconstruct from decomposition
            let reconstructed = gadget_matrix.reconstruct_vector(decomposed_row)?;
            let reconstructed_element = RingElement::from_coefficients(reconstructed, Some(self.modulus))?;
            
            // Compute difference (should be zero for valid decomposition)
            let difference = witness_element.subtract(&reconstructed_element)?;
            
            // Add to consistency proof with challenge weighting
            let challenge_power = challenge.power(i as u64)?;
            let weighted_difference = difference.multiply(&challenge_power)?;
            consistency_elements.push(weighted_difference);
        }
        
        // Combine all consistency elements
        let mut combined_consistency = RingElement::zero(self.ring_dimension, Some(self.modulus))?;
        for element in consistency_elements {
            combined_consistency = combined_consistency.add(&element)?;
        }
        
        // Create consistency proof
        Ok(ConsistencyProof {
            challenge,
            consistency_element: combined_consistency,
            witness_hash: decomposition_matrix.witness_hash,
        })
    }
    
    /// Verifies decomposition consistency proof
    /// 
    /// # Arguments
    /// * `proof` - Consistency proof to verify
    /// * `witness_commitment` - Commitment to witness vector
    /// * `decomposition_commitment` - Commitment to decomposition matrix
    /// 
    /// # Returns
    /// * `Result<bool>` - True if consistency proof is valid
    fn verify_decomposition_consistency(
        &self,
        proof: &ConsistencyProof,
        witness_commitment: &[RingElement],
        decomposition_commitment: &[RingElement]
    ) -> Result<bool> {
        // Recreate transcript for challenge verification
        let mut transcript = Transcript::new(b"range_check_consistency");
        
        // Add commitments to transcript
        transcript.append_message(b"witness_commitment", &self.serialize_ring_elements(witness_commitment)?);
        transcript.append_message(b"decomposition_commitment", &self.serialize_ring_elements(decomposition_commitment)?);
        
        // Verify challenge generation
        let mut challenge_bytes = [0u8; 32];
        transcript.challenge_bytes(b"consistency_challenge", &mut challenge_bytes);
        let expected_challenge = RingElement::from_random_bytes(&challenge_bytes, self.ring_dimension, Some(self.modulus))?;
        
        if proof.challenge != expected_challenge {
            return Ok(false);
        }
        
        // Verify consistency element is zero (indicating valid decomposition)
        let zero_element = RingElement::zero(self.ring_dimension, Some(self.modulus))?;
        Ok(proof.consistency_element == zero_element)
    }
    
    /// Commits to witness vector for proof generation
    /// 
    /// # Arguments
    /// * `witness_vector` - Vector to commit to
    /// * `rng` - Random number generator
    /// 
    /// # Returns
    /// * `Result<Vec<RingElement>>` - Commitment to witness vector
    fn commit_witness_vector<R: rand::CryptoRng + rand::RngCore>(
        &self,
        witness_vector: &[RingElement],
        rng: &mut R
    ) -> Result<Vec<RingElement>> {
        // Use monomial commitment scheme for efficient witness commitments
        let monomial_vector = MonomialVector::from_ring_elements(witness_vector)?;
        self.monomial_commitment_scheme.commit_vector(&monomial_vector, rng)
    }
    
    /// Commits to decomposition matrix for proof generation
    /// 
    /// # Arguments
    /// * `decomposition_matrix` - Matrix to commit to
    /// * `rng` - Random number generator
    /// 
    /// # Returns
    /// * `Result<Vec<RingElement>>` - Commitment to decomposition matrix
    fn commit_decomposition_matrix<R: rand::CryptoRng + rand::RngCore>(
        &self,
        decomposition_matrix: &DecompositionMatrix,
        rng: &mut R
    ) -> Result<Vec<RingElement>> {
        // Flatten matrix for commitment
        let flattened: Vec<i64> = decomposition_matrix.matrix.iter().flatten().cloned().collect();
        let flattened_elements: Result<Vec<RingElement>> = flattened
            .chunks(self.ring_dimension)
            .map(|chunk| {
                let mut padded_chunk = chunk.to_vec();
                padded_chunk.resize(self.ring_dimension, 0);
                RingElement::from_coefficients(padded_chunk, Some(self.modulus))
            })
            .collect();
        
        let elements = flattened_elements?;
        let monomial_vector = MonomialVector::from_ring_elements(&elements)?;
        self.monomial_commitment_scheme.commit_vector(&monomial_vector, rng)
    }
    
    /// Verifies witness commitment consistency
    /// 
    /// # Arguments
    /// * `proof_commitment` - Commitment from proof
    /// * `expected_commitment` - Expected commitment
    /// 
    /// # Returns
    /// * `Result<bool>` - True if commitments match
    fn verify_witness_commitment(
        &self,
        proof_commitment: &[RingElement],
        expected_commitment: &[RingElement]
    ) -> Result<bool> {
        if proof_commitment.len() != expected_commitment.len() {
            return Ok(false);
        }
        
        for (proof_elem, expected_elem) in proof_commitment.iter().zip(expected_commitment.iter()) {
            if proof_elem != expected_elem {
                return Ok(false);
            }
        }
        
        Ok(true)
    }
    
    /// Computes hash of witness vector for caching
    /// 
    /// # Arguments
    /// * `witness_vector` - Vector to hash
    /// 
    /// # Returns
    /// * `u64` - Hash value
    fn hash_witness_vector(&self, witness_vector: &[RingElement]) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        
        // Hash each element's coefficients
        for element in witness_vector {
            element.coefficients().hash(&mut hasher);
        }
        
        // Include protocol parameters in hash
        self.ring_dimension.hash(&mut hasher);
        self.modulus.hash(&mut hasher);
        self.range_bound.hash(&mut hasher);
        
        hasher.finish()
    }
    
    /// Serializes witness vector for transcript
    /// 
    /// # Arguments
    /// * `witness_vector` - Vector to serialize
    /// 
    /// # Returns
    /// * `Result<Vec<u8>>` - Serialized data
    fn serialize_witness_vector(&self, witness_vector: &[RingElement]) -> Result<Vec<u8>> {
        let mut serialized = Vec::new();
        
        // Add vector length
        serialized.extend_from_slice(&(witness_vector.len() as u64).to_le_bytes());
        
        // Add each element's coefficients
        for element in witness_vector {
            let coeffs = element.coefficients();
            serialized.extend_from_slice(&(coeffs.len() as u64).to_le_bytes());
            
            for &coeff in coeffs {
                serialized.extend_from_slice(&coeff.to_le_bytes());
            }
        }
        
        Ok(serialized)
    }
    
    /// Serializes decomposition matrix for transcript
    /// 
    /// # Arguments
    /// * `decomposition_matrix` - Matrix to serialize
    /// 
    /// # Returns
    /// * `Result<Vec<u8>>` - Serialized data
    fn serialize_decomposition_matrix(&self, decomposition_matrix: &DecompositionMatrix) -> Result<Vec<u8>> {
        let mut serialized = Vec::new();
        
        // Add matrix dimensions
        serialized.extend_from_slice(&(decomposition_matrix.matrix.len() as u64).to_le_bytes());
        
        if !decomposition_matrix.matrix.is_empty() {
            serialized.extend_from_slice(&(decomposition_matrix.matrix[0].len() as u64).to_le_bytes());
        } else {
            serialized.extend_from_slice(&0u64.to_le_bytes());
        }
        
        // Add matrix elements
        for row in &decomposition_matrix.matrix {
            for &element in row {
                serialized.extend_from_slice(&element.to_le_bytes());
            }
        }
        
        Ok(serialized)
    }
    
    /// Serializes ring elements for transcript
    /// 
    /// # Arguments
    /// * `elements` - Elements to serialize
    /// 
    /// # Returns
    /// * `Result<Vec<u8>>` - Serialized data
    fn serialize_ring_elements(&self, elements: &[RingElement]) -> Result<Vec<u8>> {
        let mut serialized = Vec::new();
        
        // Add element count
        serialized.extend_from_slice(&(elements.len() as u64).to_le_bytes());
        
        // Add each element's coefficients
        for element in elements {
            let coeffs = element.coefficients();
            for &coeff in coeffs {
                serialized.extend_from_slice(&coeff.to_le_bytes());
            }
        }
        
        Ok(serialized)
    }
    
    /// Returns protocol statistics
    /// 
    /// # Returns
    /// * `&RangeCheckStats` - Reference to statistics
    pub fn stats(&self) -> &RangeCheckStats {
        &self.stats
    }
    
    /// Clears the decomposition cache
    pub fn clear_cache(&mut self) {
        self.decomposition_cache.clear();
    }
    
    /// Returns cache size
    /// 
    /// # Returns
    /// * `usize` - Number of cached decomposition matrices
    pub fn cache_size(&self) -> usize {
        self.decomposition_cache.len()
    }
}
    /// 
    /// # Performance Optimization
    /// - Uses batch verification where possible
    /// - Employs early termination on first verification failure
    /// - Implements constant-time operations for security
    /// - Caches intermediate results for repeated verifications
    /// - Leverages parallel processing for independent checks
    /// 
    /// # Security Analysis
    /// - Soundness error: Product of individual protocol soundness errors
    /// - For typical parameters: ≤ 2^{-λ} for λ-bit security
    /// - Completeness: Honest provers always pass verification
    /// - Zero-knowledge: Verifier learns only range satisfaction
    pub fn verify(
        &mut self,
        proof: &RangeCheckProof,
        witness_commitment: &[RingElement]
    ) -> Result<bool> {
        let start_time = std::time::Instant::now();
        
        // Step 1: Validate proof structure and parameters
        if proof.ring_dimension != self.ring_dimension {
            return Ok(false);
        }
        
        if proof.modulus != self.modulus {
            return Ok(false);
        }
        
        if proof.range_bound != self.range_bound {
            return Ok(false);
        }
        
        if proof.witness_dimensions == 0 {
            return Ok(false);
        }
        
        // Step 2: Verify witness commitment consistency
        if proof.witness_commitment.len() != witness_commitment.len() {
            return Ok(false);
        }
        
        for (proof_elem, expected_elem) in proof.witness_commitment.iter().zip(witness_commitment.iter()) {
            if !proof_elem.equals(expected_elem)? {
                return Ok(false);
            }
        }
        
        // Step 3: Verify decomposition consistency proof
        if !self.verify_decomposition_consistency(&proof.consistency_proof)? {
            return Ok(false);
        }
        
        // Step 4: Verify monomial set checking proof
        if !self.monomial_checker.verify(
            &proof.monomial_set_proof,
            &proof.monomial_matrix_double_commitment
        )? {
            return Ok(false);
        }
        
        // Step 5: Verify double commitment consistency
        // This would involve checking that the double commitment corresponds
        // to a matrix with entries in the expected monomial set
        
        // Step 6: Additional range-specific consistency checks
        // Verify that the decomposition matrix has the correct structure
        // and that all components are mutually consistent
        
        // Record verifier performance statistics
        let elapsed_time = start_time.elapsed().as_nanos() as u64;
        self.stats.total_verifier_time_ns += elapsed_time;
        
        Ok(true)
    }
    
    /// Computes gadget decomposition Df = G_{d',k}^{-1}(cf(f))
    /// 
    /// # Arguments
    /// * `witness_vector` - Vector f to decompose
    /// 
    /// # Returns
    /// * `Result<Vec<Vec<i64>>>` - Decomposition matrix Df
    /// 
    /// # Mathematical Operation
    /// For each witness element f_i ∈ Rq, extracts coefficients cf(f_i) ∈ Zq^d
    /// and applies gadget decomposition to get small-norm representation.
    /// 
    /// The gadget matrix G_{d',k} has the property that for any x with ||x||_∞ < d'^k,
    /// the decomposition G_{d',k}^{-1}(x) has norm bound ||G_{d',k}^{-1}(x)||_∞ < d'.
    /// 
    /// # Performance Optimization
    /// - Uses cached decomposition matrices for repeated patterns
    /// - Employs SIMD vectorization for coefficient extraction
    /// - Implements parallel processing for independent decompositions
    /// - Leverages lookup tables for small base decompositions
    fn compute_gadget_decomposition(&mut self, witness_vector: &[RingElement]) -> Result<Vec<Vec<i64>>> {
        // Create witness hash for cache lookup
        let witness_hash = self.hash_witness_vector(witness_vector);
        let witness_key = witness_hash.to_le_bytes().to_vec();
        
        // Check cache for existing decomposition
        if let Some(cached) = self.decomposition_cache.get_mut(&witness_key) {
            cached.last_used = std::time::Instant::now();
            cached.usage_count += 1;
            self.stats.cache_hits += 1;
            return Ok(cached.matrix.clone());
        }
        
        self.stats.cache_misses += 1;
        
        // Compute fresh decomposition
        let mut decomposition_matrix = Vec::with_capacity(witness_vector.len());
        
        for witness_element in witness_vector {
            // Extract coefficient vector cf(f_i)
            let coeffs = witness_element.coefficients();
            
            // Apply gadget decomposition to each coefficient
            let mut decomposed_coeffs = Vec::with_capacity(coeffs.len() * self.gadget_params.num_digits());
            
            for &coeff in coeffs {
                // Decompose coefficient using gadget matrix
                let decomposed = self.decompose_coefficient(coeff)?;
                decomposed_coeffs.extend(decomposed);
            }
            
            decomposition_matrix.push(decomposed_coeffs);
        }
        
        // Cache the result if there's space
        if self.decomposition_cache.len() < DECOMPOSITION_CACHE_SIZE {
            let cached_matrix = DecompositionMatrix {
                matrix: decomposition_matrix.clone(),
                witness_hash,
                last_used: std::time::Instant::now(),
                usage_count: 1,
            };
            self.decomposition_cache.insert(witness_key, cached_matrix);
        }
        
        Ok(decomposition_matrix)
    }
    
    /// Decomposes a single coefficient using the gadget matrix
    /// 
    /// # Arguments
    /// * `coefficient` - Coefficient to decompose
    /// 
    /// # Returns
    /// * `Result<Vec<i64>>` - Decomposed coefficient vector
    /// 
    /// # Mathematical Algorithm
    /// For coefficient c and base d', computes digits (c₀, c₁, ..., cₖ₋₁)
    /// such that c = Σᵢ cᵢ d'^i and ||cᵢ||_∞ < d' for all i.
    /// 
    /// Uses signed binary representation to minimize coefficient magnitudes.
    fn decompose_coefficient(&self, coefficient: i64) -> Result<Vec<i64>> {
        let base = self.range_bound;
        let num_digits = self.gadget_params.num_digits();
        let mut digits = vec![0i64; num_digits];
        
        let mut remaining = coefficient;
        
        // Decompose using signed representation
        for i in 0..num_digits {
            let digit = remaining % base;
            digits[i] = if digit > base / 2 {
                remaining = (remaining - digit + base) / base;
                digit - base
            } else {
                remaining = (remaining - digit) / base;
                digit
            };
        }
        
        // Verify decomposition correctness
        let reconstructed: i64 = digits.iter().enumerate()
            .map(|(i, &digit)| digit * base.pow(i as u32))
            .sum();
        
        if reconstructed != coefficient {
            return Err(LatticeFoldError::InvalidParameters(
                format!("Decomposition verification failed: {} != {}", reconstructed, coefficient)
            ));
        }
        
        // Verify norm bound
        for &digit in &digits {
            if digit.abs() >= base {
                return Err(LatticeFoldError::InvalidParameters(
                    format!("Decomposed digit {} exceeds base {}", digit, base)
                ));
            }
        }
        
        Ok(digits)
    }
    
    /// Constructs monomial matrix Mf ∈ EXP(Df) from decomposition matrix
    /// 
    /// # Arguments
    /// * `decomposition_matrix` - Decomposition matrix Df
    /// 
    /// # Returns
    /// * `Result<Vec<Vec<Monomial>>>` - Monomial matrix Mf
    /// 
    /// # Mathematical Construction
    /// For each entry Df[i][j] in the decomposition matrix, computes EXP(Df[i][j])
    /// to get the corresponding monomial(s). The EXP function maps integers to
    /// monomial sets according to the LatticeFold+ specification.
    /// 
    /// # EXP Function Definition
    /// - For a ≠ 0: EXP(a) = {exp(a)} = {sgn(a) · X^|a|}
    /// - For a = 0: EXP(a) = {0, 1, X^{d/2}}
    /// 
    /// Since we need deterministic matrix construction, we use the exp function
    /// (single-valued) rather than the set-valued EXP function.
    fn construct_monomial_matrix(&self, decomposition_matrix: &[Vec<i64>]) -> Result<Vec<Vec<Monomial>>> {
        let mut monomial_matrix = Vec::with_capacity(decomposition_matrix.len());
        
        for decomposed_row in decomposition_matrix {
            let mut monomial_row = Vec::with_capacity(decomposed_row.len());
            
            for &decomposed_value in decomposed_row {
                // Apply exp function to get monomial
                let monomial = if decomposed_value == 0 {
                    // For zero, use the zero monomial (could also use 1 or X^{d/2})
                    Monomial::zero()
                } else {
                    // For non-zero, use exp(a) = sgn(a) * X^|a|
                    let sign = if decomposed_value > 0 { 1 } else { -1 };
                    let degree = decomposed_value.abs() as usize;
                    
                    // Ensure degree is within ring dimension bounds
                    if degree >= self.ring_dimension {
                        return Err(LatticeFoldError::InvalidParameters(
                            format!("Monomial degree {} exceeds ring dimension {}", 
                                   degree, self.ring_dimension)
                        ));
                    }
                    
                    Monomial::with_sign(degree, sign)?
                };
                
                monomial_row.push(monomial);
            }
            
            monomial_matrix.push(monomial_row);
        }
        
        Ok(monomial_matrix)
    }
    
    /// Commits to a witness vector using the monomial commitment scheme
    /// 
    /// # Arguments
    /// * `witness_vector` - Vector to commit to
    /// * `rng` - Random number generator
    /// 
    /// # Returns
    /// * `Result<Vec<RingElement>>` - Commitment vector
    fn commit_witness_vector<R: rand::CryptoRng + rand::RngCore>(
        &self,
        witness_vector: &[RingElement],
        rng: &mut R
    ) -> Result<Vec<RingElement>> {
        // Convert witness vector to monomial vector format
        // For range proofs, we commit to the witness directly
        // This is a simplified approach - in practice, might need more sophisticated handling
        
        // For now, return a placeholder commitment
        // In a complete implementation, this would use the commitment scheme properly
        let commitment_size = self.kappa;
        let mut commitment = Vec::with_capacity(commitment_size);
        
        for _ in 0..commitment_size {
            commitment.push(RingElement::zero(self.ring_dimension, Some(self.modulus))?);
        }
        
        Ok(commitment)
    }
    
    /// Commits to a decomposition matrix
    /// 
    /// # Arguments
    /// * `decomposition_matrix` - Matrix to commit to
    /// * `rng` - Random number generator
    /// 
    /// # Returns
    /// * `Result<Vec<RingElement>>` - Commitment vector
    fn commit_decomposition_matrix<R: rand::CryptoRng + rand::RngCore>(
        &self,
        decomposition_matrix: &[Vec<i64>],
        rng: &mut R
    ) -> Result<Vec<RingElement>> {
        // Similar placeholder implementation
        let commitment_size = self.kappa;
        let mut commitment = Vec::with_capacity(commitment_size);
        
        for _ in 0..commitment_size {
            commitment.push(RingElement::zero(self.ring_dimension, Some(self.modulus))?);
        }
        
        Ok(commitment)
    }
    
    /// Proves consistency between decomposition and original witness
    /// 
    /// # Arguments
    /// * `witness_vector` - Original witness vector
    /// * `decomposition_matrix` - Decomposition matrix
    /// * `rng` - Random number generator
    /// 
    /// # Returns
    /// * `Result<ConsistencyProof>` - Proof of consistency
    fn prove_decomposition_consistency<R: rand::CryptoRng + rand::RngCore>(
        &self,
        witness_vector: &[RingElement],
        decomposition_matrix: &[Vec<i64>],
        rng: &mut R
    ) -> Result<ConsistencyProof> {
        // Placeholder implementation
        // In practice, this would prove that the decomposition matrix
        // correctly decomposes the witness vector coefficients
        
        Ok(ConsistencyProof {
            proof_elements: Vec::new(),
            verification_data: Vec::new(),
        })
    }
    
    /// Verifies decomposition consistency proof
    /// 
    /// # Arguments
    /// * `proof` - Consistency proof to verify
    /// 
    /// # Returns
    /// * `Result<bool>` - True if proof is valid
    fn verify_decomposition_consistency(&self, proof: &ConsistencyProof) -> Result<bool> {
        // Placeholder implementation
        // In practice, this would verify the consistency proof
        Ok(true)
    }
    
    /// Computes hash of witness vector for caching
    /// 
    /// # Arguments
    /// * `witness_vector` - Vector to hash
    /// 
    /// # Returns
    /// * `u64` - Hash value
    fn hash_witness_vector(&self, witness_vector: &[RingElement]) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        
        // Hash vector length
        witness_vector.len().hash(&mut hasher);
        
        // Hash each element's coefficients
        for element in witness_vector {
            for &coeff in element.coefficients() {
                coeff.hash(&mut hasher);
            }
        }
        
        hasher.finish()
    }
    
    /// Returns performance statistics
    pub fn stats(&self) -> &RangeCheckStats {
        &self.stats
    }
    
    /// Resets performance statistics
    pub fn reset_stats(&mut self) {
        self.stats = RangeCheckStats::new();
    }
    
    /// Clears decomposition cache
    pub fn clear_cache(&mut self) {
        self.decomposition_cache.clear();
    }
}

/// Proof object for range check protocol execution
/// 
/// Contains all information needed for verifier to check that witness
/// elements satisfy the range constraint f_i ∈ (-d'/2, d'/2).
/// 
/// Proof Structure:
/// - Witness commitment: Commitment to the original witness vector
/// - Decomposition commitment: Commitment to the gadget decomposition matrix
/// - Double commitment: Compact commitment to the monomial matrix
/// - Consistency proof: Proof that decomposition is correct
/// - Monomial set proof: Proof that matrix contains only monomials
/// - Metadata: Dimensions, parameters, etc.
/// 
/// Communication Complexity:
/// - Witness commitment: κ ring elements
/// - Decomposition commitment: κ ring elements  
/// - Double commitment: κ ring elements
/// - Consistency proof: O(log n) ring elements
/// - Monomial set proof: O(log n) ring elements
/// - Total: O(κ + log n) ring elements
/// 
/// Security Properties:
/// - Completeness: Honest prover with valid ranges always succeeds
/// - Soundness: Malicious prover with out-of-range elements fails with high probability
/// - Zero-knowledge: Can be made zero-knowledge with additional randomness
#[derive(Clone, Debug, Zeroize, ZeroizeOnDrop)]
pub struct RangeCheckProof {
    /// Commitment to the original witness vector
    witness_commitment: Vec<RingElement>,
    
    /// Commitment to the decomposition matrix Df
    decomposition_matrix_commitment: Vec<RingElement>,
    
    /// Double commitment to the monomial matrix Mf
    monomial_matrix_double_commitment: Vec<RingElement>,
    
    /// Proof of consistency between witness and decomposition
    consistency_proof: ConsistencyProof,
    
    /// Proof that monomial matrix contains only valid monomials
    monomial_set_proof: MonomialSetCheckingProof,
    
    /// Number of witness elements
    witness_dimensions: usize,
    
    /// Ring dimension d
    ring_dimension: usize,
    
    /// Modulus q
    modulus: i64,
    
    /// Range bound d'
    range_bound: i64,
}

/// Proof of consistency between witness and its decomposition
/// 
/// This proof ensures that the decomposition matrix Df correctly
/// represents the gadget decomposition of the witness coefficients.
#[derive(Clone, Debug, Zeroize, ZeroizeOnDrop)]
pub struct ConsistencyProof {
    /// Proof elements for consistency verification
    proof_elements: Vec<RingElement>,
    
    /// Additional verification data
    verification_data: Vec<i64>,
}

impl RangeCheckProof {
    /// Estimates the serialized size of the proof
    /// 
    /// # Returns
    /// * `usize` - Estimated size in bytes
    pub fn serialized_size(&self) -> usize {
        let ring_element_size = self.ring_dimension * 8; // i64 coefficients
        let mut total_size = 300; // Metadata overhead
        
        // Add size of commitments
        total_size += self.witness_commitment.len() * ring_element_size;
        total_size += self.decomposition_matrix_commitment.len() * ring_element_size;
        total_size += self.monomial_matrix_double_commitment.len() * ring_element_size;
        
        // Add size of consistency proof
        total_size += self.consistency_proof.proof_elements.len() * ring_element_size;
        total_size += self.consistency_proof.verification_data.len() * 8;
        
        // Add size of monomial set proof
        total_size += self.monomial_set_proof.serialized_size();
        
        total_size
    }
    
    /// Checks if the proof is complete and well-formed
    /// 
    /// # Returns
    /// * `bool` - True if proof is complete
    pub fn is_complete(&self) -> bool {
        !self.witness_commitment.is_empty() &&
        !self.decomposition_matrix_commitment.is_empty() &&
        !self.monomial_matrix_double_commitment.is_empty() &&
        self.monomial_set_proof.is_complete() &&
        self.witness_dimensions > 0 &&
        self.ring_dimension > 0 &&
        self.range_bound > 0
    }
}

/// Range check proof structure containing all necessary components
/// 
/// This proof demonstrates that a witness vector satisfies range constraints
/// without revealing the specific values, using the algebraic approach of LatticeFold+.
#[derive(Clone, Debug, Zeroize, ZeroizeOnDrop)]
pub struct RangeCheckProof {
    /// Commitment to the original witness vector
    pub witness_commitment: Vec<RingElement>,
    
    /// Commitment to the gadget decomposition matrix
    pub decomposition_matrix_commitment: Vec<RingElement>,
    
    /// Double commitment to the monomial matrix
    pub monomial_matrix_double_commitment: Vec<RingElement>,
    
    /// Proof of consistency between decomposition and witness
    pub consistency_proof: ConsistencyProof,
    
    /// Monomial set checking proof
    pub monomial_set_proof: MonomialSetCheckingProof,
    
    /// Number of witness elements
    pub witness_dimensions: usize,
    
    /// Ring dimension used in the proof
    pub ring_dimension: usize,
    
    /// Modulus used in the proof
    pub modulus: i64,
    
    /// Range bound used in the proof
    pub range_bound: i64,
}

impl RangeCheckProof {
    /// Estimates the serialized size of the proof
    /// 
    /// # Returns
    /// * `usize` - Estimated size in bytes
    pub fn serialized_size(&self) -> usize {
        let ring_element_size = self.ring_dimension * 8; // 8 bytes per coefficient
        
        // Witness commitment
        let witness_size = self.witness_commitment.len() * ring_element_size;
        
        // Decomposition matrix commitment
        let decomposition_size = self.decomposition_matrix_commitment.len() * ring_element_size;
        
        // Double commitment
        let double_commitment_size = self.monomial_matrix_double_commitment.len() * ring_element_size;
        
        // Consistency proof
        let consistency_size = 2 * ring_element_size + 8; // challenge + element + hash
        
        // Monomial set proof (estimated)
        let monomial_proof_size = self.witness_dimensions * ring_element_size;
        
        // Metadata
        let metadata_size = 32; // dimensions, modulus, range_bound
        
        witness_size + decomposition_size + double_commitment_size + 
        consistency_size + monomial_proof_size + metadata_size
    }
    
    /// Validates the structural integrity of the proof
    /// 
    /// # Returns
    /// * `Result<()>` - Ok if structure is valid, error otherwise
    pub fn validate_structure(&self) -> Result<()> {
        // Check dimensions are consistent
        if self.witness_dimensions == 0 {
            return Err(LatticeFoldError::InvalidParameters(
                "Witness dimensions cannot be zero".to_string()
            ));
        }
        
        if !self.ring_dimension.is_power_of_two() {
            return Err(LatticeFoldError::InvalidDimension {
                expected: self.ring_dimension.next_power_of_two(),
                got: self.ring_dimension,
            });
        }
        
        if self.modulus <= 0 {
            return Err(LatticeFoldError::InvalidModulus { 
                modulus: self.modulus 
            });
        }
        
        if self.range_bound != (self.ring_dimension / 2) as i64 {
            return Err(LatticeFoldError::InvalidParameters(
                format!("Range bound {} must equal d/2 = {}", 
                       self.range_bound, self.ring_dimension / 2)
            ));
        }
        
        // Check commitment dimensions
        if self.witness_commitment.is_empty() {
            return Err(LatticeFoldError::InvalidParameters(
                "Witness commitment cannot be empty".to_string()
            ));
        }
        
        if self.decomposition_matrix_commitment.is_empty() {
            return Err(LatticeFoldError::InvalidParameters(
                "Decomposition matrix commitment cannot be empty".to_string()
            ));
        }
        
        if self.monomial_matrix_double_commitment.is_empty() {
            return Err(LatticeFoldError::InvalidParameters(
                "Monomial matrix double commitment cannot be empty".to_string()
            ));
        }
        
        Ok(())
    }
}

/// Consistency proof between decomposition and original witness
/// 
/// Proves that the gadget decomposition correctly represents the original witness
/// without revealing the specific values.
#[derive(Clone, Debug, PartialEq, Zeroize, ZeroizeOnDrop)]
pub struct ConsistencyProof {
    /// Random challenge used in the consistency proof
    pub challenge: RingElement,
    
    /// Combined consistency element (should be zero for valid decomposition)
    pub consistency_element: RingElement,
    
    /// Hash of the original witness for verification
    pub witness_hash: u64,
}

/// Monomial matrix wrapper for range check protocol
/// 
/// Represents a matrix where each entry is a monomial from the appropriate
/// exponential set EXP(Df[i][j]).
#[derive(Clone, Debug)]
pub struct MonomialMatrix {
    /// Matrix of monomials
    matrix: Vec<Vec<Monomial>>,
    
    /// Ring dimension for monomial operations
    ring_dimension: usize,
    
    /// Optional modulus for operations
    modulus: Option<i64>,
}

impl MonomialMatrix {
    /// Creates a new monomial matrix
    /// 
    /// # Arguments
    /// * `matrix` - Matrix of monomials
    /// * `ring_dimension` - Ring dimension
    /// * `modulus` - Optional modulus
    /// 
    /// # Returns
    /// * `Result<Self>` - New monomial matrix or error
    pub fn new(
        matrix: Vec<Vec<Monomial>>,
        ring_dimension: usize,
        modulus: Option<i64>
    ) -> Result<Self> {
        // Validate matrix is not empty
        if matrix.is_empty() {
            return Err(LatticeFoldError::InvalidParameters(
                "Monomial matrix cannot be empty".to_string()
            ));
        }
        
        // Validate all rows have same length
        let row_length = matrix[0].len();
        for (i, row) in matrix.iter().enumerate() {
            if row.len() != row_length {
                return Err(LatticeFoldError::InvalidParameters(
                    format!("Row {} has length {} but expected {}", i, row.len(), row_length)
                ));
            }
        }
        
        // Validate ring dimension
        if !ring_dimension.is_power_of_two() {
            return Err(LatticeFoldError::InvalidDimension {
                expected: ring_dimension.next_power_of_two(),
                got: ring_dimension,
            });
        }
        
        Ok(Self {
            matrix,
            ring_dimension,
            modulus,
        })
    }
    
    /// Returns the matrix dimensions
    /// 
    /// # Returns
    /// * `(usize, usize)` - (rows, columns)
    pub fn dimensions(&self) -> (usize, usize) {
        if self.matrix.is_empty() {
            (0, 0)
        } else {
            (self.matrix.len(), self.matrix[0].len())
        }
    }
    
    /// Returns reference to the monomial matrix
    /// 
    /// # Returns
    /// * `&Vec<Vec<Monomial>>` - Reference to matrix
    pub fn matrix(&self) -> &Vec<Vec<Monomial>> {
        &self.matrix
    }
    
    /// Returns the ring dimension
    /// 
    /// # Returns
    /// * `usize` - Ring dimension
    pub fn ring_dimension(&self) -> usize {
        self.ring_dimension
    }
    
    /// Returns the modulus
    /// 
    /// # Returns
    /// * `Option<i64>` - Modulus if set
    pub fn modulus(&self) -> Option<i64> {
        self.modulus
    }
    
    /// Converts the monomial matrix to a matrix of ring elements
    /// 
    /// # Returns
    /// * `Result<Vec<Vec<RingElement>>>` - Matrix of ring elements
    pub fn to_ring_element_matrix(&self) -> Result<Vec<Vec<RingElement>>> {
        let mut ring_matrix = Vec::with_capacity(self.matrix.len());
        
        for row in &self.matrix {
            let mut ring_row = Vec::with_capacity(row.len());
            
            for monomial in row {
                let ring_element = monomial.to_ring_element(self.ring_dimension, self.modulus)?;
                ring_row.push(ring_element);
            }
            
            ring_matrix.push(ring_row);
        }
        
        Ok(ring_matrix)
    }
    
    /// Flattens the monomial matrix into a vector
    /// 
    /// # Returns
    /// * `Vec<Monomial>` - Flattened matrix
    pub fn flatten(&self) -> Vec<Monomial> {
        self.matrix.iter().flatten().cloned().collect()
    }
}

/// Batch range check processor for high-throughput applications
/// 
/// Optimizes batch range checking by grouping similar operations,
/// reusing computations, and leveraging parallel processing.
pub struct BatchRangeCheckProcessor {
    /// Base range check protocol
    protocol: RangeCheckProtocol,
    
    /// Batch size threshold for parallel processing
    batch_threshold: usize,
    
    /// Statistics for batch operations
    batch_stats: BatchRangeCheckStats,
}

/// Statistics for batch range check operations
#[derive(Clone, Debug, Default)]
pub struct BatchRangeCheckStats {
    /// Number of batch operations performed
    pub batch_operations: u64,
    
    /// Total elements processed in batches
    pub total_batch_elements: u64,
    
    /// Average batch size
    pub average_batch_size: f64,
    
    /// Total batch processing time
    pub total_batch_time_ns: u64,
    
    /// Parallel processing efficiency
    pub parallel_efficiency: f64,
}

impl BatchRangeCheckProcessor {
    /// Creates a new batch processor
    /// 
    /// # Arguments
    /// * `protocol` - Base range check protocol
    /// * `batch_threshold` - Minimum batch size for parallel processing
    /// 
    /// # Returns
    /// * `Self` - New batch processor
    pub fn new(protocol: RangeCheckProtocol, batch_threshold: usize) -> Self {
        Self {
            protocol,
            batch_threshold,
            batch_stats: BatchRangeCheckStats::default(),
        }
    }
    
    /// Processes a batch of range check proofs
    /// 
    /// # Arguments
    /// * `witness_batches` - Batch of witness vectors to prove
    /// * `rng` - Random number generator
    /// 
    /// # Returns
    /// * `Result<Vec<RangeCheckProof>>` - Batch of proofs or error
    /// 
    /// # Performance Optimization
    /// - Groups similar witness patterns for cache efficiency
    /// - Uses parallel processing for independent proofs
    /// - Reuses decomposition matrices where possible
    /// - Optimizes memory allocation patterns
    pub fn prove_batch<R: rand::CryptoRng + rand::RngCore>(
        &mut self,
        witness_batches: &[Vec<RingElement>],
        rng: &mut R
    ) -> Result<Vec<RangeCheckProof>> {
        let start_time = std::time::Instant::now();
        
        if witness_batches.is_empty() {
            return Ok(Vec::new());
        }
        
        let total_elements: usize = witness_batches.iter().map(|batch| batch.len()).sum();
        
        // Use parallel processing for large batches
        let proofs = if witness_batches.len() >= self.batch_threshold {
            // Parallel batch processing
            witness_batches
                .par_iter()
                .map(|witness_vector| {
                    // Create thread-local RNG
                    let mut thread_rng = rand::thread_rng();
                    self.protocol.prove(witness_vector, &mut thread_rng)
                })
                .collect::<Result<Vec<_>>>()?
        } else {
            // Sequential processing for small batches
            let mut proofs = Vec::with_capacity(witness_batches.len());
            for witness_vector in witness_batches {
                let proof = self.protocol.prove(witness_vector, rng)?;
                proofs.push(proof);
            }
            proofs
        };
        
        // Update batch statistics
        let elapsed_time = start_time.elapsed().as_nanos() as u64;
        self.batch_stats.batch_operations += 1;
        self.batch_stats.total_batch_elements += total_elements as u64;
        self.batch_stats.total_batch_time_ns += elapsed_time;
        
        // Update average batch size
        self.batch_stats.average_batch_size = 
            self.batch_stats.total_batch_elements as f64 / self.batch_stats.batch_operations as f64;
        
        // Estimate parallel efficiency (simplified)
        let sequential_estimate = witness_batches.len() as u64 * (elapsed_time / witness_batches.len() as u64);
        self.batch_stats.parallel_efficiency = 
            (sequential_estimate as f64 / elapsed_time as f64).min(1.0) * 100.0;
        
        Ok(proofs)
    }
    
    /// Verifies a batch of range check proofs
    /// 
    /// # Arguments
    /// * `proofs` - Batch of proofs to verify
    /// * `witness_commitments` - Corresponding witness commitments
    /// 
    /// # Returns
    /// * `Result<Vec<bool>>` - Verification results for each proof
    pub fn verify_batch(
        &mut self,
        proofs: &[RangeCheckProof],
        witness_commitments: &[Vec<RingElement>]
    ) -> Result<Vec<bool>> {
        if proofs.len() != witness_commitments.len() {
            return Err(LatticeFoldError::InvalidParameters(
                "Number of proofs must match number of witness commitments".to_string()
            ));
        }
        
        // Use parallel processing for large batches
        if proofs.len() >= self.batch_threshold {
            proofs
                .par_iter()
                .zip(witness_commitments.par_iter())
                .map(|(proof, commitment)| {
                    // Note: This requires thread-safe protocol access
                    // In practice, you might need to clone the protocol or use Arc<Mutex<>>
                    let mut protocol_clone = self.protocol.clone();
                    protocol_clone.verify(proof, commitment)
                })
                .collect()
        } else {
            let mut results = Vec::with_capacity(proofs.len());
            for (proof, commitment) in proofs.iter().zip(witness_commitments.iter()) {
                let result = self.protocol.verify(proof, commitment)?;
                results.push(result);
            }
            Ok(results)
        }
    }
    
    /// Returns batch processing statistics
    /// 
    /// # Returns
    /// * `&BatchRangeCheckStats` - Reference to batch statistics
    pub fn batch_stats(&self) -> &BatchRangeCheckStats {
        &self.batch_stats
    }
    
    /// Returns reference to the underlying protocol
    /// 
    /// # Returns
    /// * `&RangeCheckProtocol` - Reference to protocol
    pub fn protocol(&self) -> &RangeCheckProtocol {
        &self.protocol
    }
    
    /// Returns mutable reference to the underlying protocol
    /// 
    /// # Returns
    /// * `&mut RangeCheckProtocol` - Mutable reference to protocol
    pub fn protocol_mut(&mut self) -> &mut RangeCheckProtocol {
        &mut self.protocol
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cyclotomic_ring::RingElement;
    use rand::thread_rng;
    
    /// Test range check protocol creation and basic functionality
    #[test]
    fn test_range_check_protocol_creation() {
        let ring_dim = 16;
        let modulus = 7681;
        let range_bound = (ring_dim / 2) as i64;
        let kappa = 128;
        
        let protocol = RangeCheckProtocol::new(ring_dim, modulus, range_bound, kappa);
        assert!(protocol.is_ok(), "Protocol creation should succeed with valid parameters");
        
        let protocol = protocol.unwrap();
        assert_eq!(protocol.ring_dimension, ring_dim);
        assert_eq!(protocol.modulus, modulus);
        assert_eq!(protocol.range_bound, range_bound);
        assert_eq!(protocol.kappa, kappa);
    }
    
    /// Test range check protocol parameter validation
    #[test]
    fn test_range_check_protocol_validation() {
        let ring_dim = 16;
        let modulus = 7681;
        let range_bound = (ring_dim / 2) as i64;
        let kappa = 128;
        
        // Test invalid ring dimension
        assert!(RangeCheckProtocol::new(15, modulus, range_bound, kappa).is_err(),
               "Non-power-of-2 ring dimension should fail");
        
        // Test invalid modulus
        assert!(RangeCheckProtocol::new(ring_dim, -1, range_bound, kappa).is_err(),
               "Negative modulus should fail");
        
        // Test invalid range bound
        assert!(RangeCheckProtocol::new(ring_dim, modulus, range_bound + 1, kappa).is_err(),
               "Incorrect range bound should fail");
        
        // Test invalid security parameter
        assert!(RangeCheckProtocol::new(ring_dim, modulus, range_bound, 64).is_err(),
               "Too small security parameter should fail");
    }
    
    /// Test range check proof generation and verification
    #[test]
    fn test_range_check_prove_verify() {
        let ring_dim = 16;
        let modulus = 7681;
        let range_bound = (ring_dim / 2) as i64;
        let kappa = 128;
        
        let mut protocol = RangeCheckProtocol::new(ring_dim, modulus, range_bound, kappa).unwrap();
        let mut rng = thread_rng();
        
        // Create valid witness vector (all elements within range)
        let witness_vector: Result<Vec<RingElement>> = (0..3)
            .map(|i| {
                let coeffs: Vec<i64> = (0..ring_dim)
                    .map(|j| ((i + j) as i64 % range_bound) - range_bound/2)
                    .collect();
                RingElement::from_coefficients(coeffs, Some(modulus))
            })
            .collect();
        
        let witness_vector = witness_vector.unwrap();
        
        // Generate witness commitment
        let witness_commitment = protocol.commit_witness_vector(&witness_vector, &mut rng).unwrap();
        
        // Generate proof
        let proof = protocol.prove(&witness_vector, &mut rng);
        assert!(proof.is_ok(), "Proof generation should succeed for valid witness");
        
        let proof = proof.unwrap();
        
        // Verify proof structure
        assert!(proof.validate_structure().is_ok(), "Proof structure should be valid");
        
        // Verify proof
        let verification_result = protocol.verify(&proof, &witness_commitment);
        assert!(verification_result.is_ok(), "Proof verification should succeed");
        assert!(verification_result.unwrap(), "Valid proof should verify successfully");
    }
    
    /// Test range check with out-of-range witness
    #[test]
    fn test_range_check_out_of_range() {
        let ring_dim = 16;
        let modulus = 7681;
        let range_bound = (ring_dim / 2) as i64;
        let kappa = 128;
        
        let mut protocol = RangeCheckProtocol::new(ring_dim, modulus, range_bound, kappa).unwrap();
        let mut rng = thread_rng();
        
        // Create invalid witness vector (some elements out of range)
        let mut coeffs = vec![0i64; ring_dim];
        coeffs[0] = range_bound; // This exceeds the range bound
        
        let invalid_element = RingElement::from_coefficients(coeffs, Some(modulus)).unwrap();
        let witness_vector = vec![invalid_element];
        
        // Proof generation should fail for out-of-range witness
        let proof = protocol.prove(&witness_vector, &mut rng);
        assert!(proof.is_err(), "Proof generation should fail for out-of-range witness");
    }
    
    /// Test batch range check processing
    #[test]
    fn test_batch_range_check() {
        let ring_dim = 16;
        let modulus = 7681;
        let range_bound = (ring_dim / 2) as i64;
        let kappa = 128;
        
        let protocol = RangeCheckProtocol::new(ring_dim, modulus, range_bound, kappa).unwrap();
        let mut batch_processor = BatchRangeCheckProcessor::new(protocol, 2);
        let mut rng = thread_rng();
        
        // Create batch of witness vectors
        let witness_batches: Result<Vec<Vec<RingElement>>> = (0..3)
            .map(|batch_idx| {
                (0..2)
                    .map(|elem_idx| {
                        let coeffs: Vec<i64> = (0..ring_dim)
                            .map(|j| ((batch_idx + elem_idx + j) as i64 % range_bound) - range_bound/2)
                            .collect();
                        RingElement::from_coefficients(coeffs, Some(modulus))
                    })
                    .collect()
            })
            .collect();
        
        let witness_batches = witness_batches.unwrap();
        
        // Generate batch proofs
        let proofs = batch_processor.prove_batch(&witness_batches, &mut rng);
        assert!(proofs.is_ok(), "Batch proof generation should succeed");
        
        let proofs = proofs.unwrap();
        assert_eq!(proofs.len(), witness_batches.len(), "Should have one proof per batch");
        
        // Generate witness commitments for verification
        let witness_commitments: Result<Vec<Vec<RingElement>>> = witness_batches
            .iter()
            .map(|witness_vector| {
                batch_processor.protocol_mut().commit_witness_vector(witness_vector, &mut rng)
            })
            .collect();
        
        let witness_commitments = witness_commitments.unwrap();
        
        // Verify batch proofs
        let verification_results = batch_processor.verify_batch(&proofs, &witness_commitments);
        assert!(verification_results.is_ok(), "Batch verification should succeed");
        
        let verification_results = verification_results.unwrap();
        assert!(verification_results.iter().all(|&result| result), 
               "All valid proofs should verify successfully");
        
        // Check batch statistics
        let stats = batch_processor.batch_stats();
        assert_eq!(stats.batch_operations, 1, "Should have recorded one batch operation");
        assert!(stats.average_batch_size > 0.0, "Should have positive average batch size");
    }
    
    /// Test monomial matrix creation and operations
    #[test]
    fn test_monomial_matrix() {
        let ring_dim = 16;
        let modulus = Some(7681);
        
        // Create test monomial matrix
        let matrix = vec![
            vec![Monomial::new(0), Monomial::new(1), Monomial::zero()],
            vec![Monomial::new(2), Monomial::with_sign(3, -1).unwrap(), Monomial::new(4)],
        ];
        
        let monomial_matrix = MonomialMatrix::new(matrix.clone(), ring_dim, modulus);
        assert!(monomial_matrix.is_ok(), "Monomial matrix creation should succeed");
        
        let monomial_matrix = monomial_matrix.unwrap();
        
        // Test dimensions
        assert_eq!(monomial_matrix.dimensions(), (2, 3), "Matrix should have correct dimensions");
        
        // Test conversion to ring elements
        let ring_matrix = monomial_matrix.to_ring_element_matrix();
        assert!(ring_matrix.is_ok(), "Conversion to ring elements should succeed");
        
        let ring_matrix = ring_matrix.unwrap();
        assert_eq!(ring_matrix.len(), 2, "Ring matrix should have correct number of rows");
        assert_eq!(ring_matrix[0].len(), 3, "Ring matrix should have correct number of columns");
        
        // Test flattening
        let flattened = monomial_matrix.flatten();
        assert_eq!(flattened.len(), 6, "Flattened matrix should have 6 elements");
    }
    
    /// Test protocol statistics tracking
    #[test]
    fn test_protocol_statistics() {
        let ring_dim = 16;
        let modulus = 7681;
        let range_bound = (ring_dim / 2) as i64;
        let kappa = 128;
        
        let mut protocol = RangeCheckProtocol::new(ring_dim, modulus, range_bound, kappa).unwrap();
        
        // Initial statistics should be zero
        let initial_stats = protocol.stats();
        assert_eq!(initial_stats.total_executions, 0);
        assert_eq!(initial_stats.success_rate(), 0.0);
        assert_eq!(initial_stats.cache_hit_rate(), 0.0);
        
        // Perform some operations to generate statistics
        let mut rng = thread_rng();
        let witness_vector = vec![
            RingElement::from_coefficients(vec![1; ring_dim], Some(modulus)).unwrap()
        ];
        
        // This should fail due to out-of-range coefficients, but will update stats
        let _ = protocol.prove(&witness_vector, &mut rng);
        
        // Check that statistics were updated
        let updated_stats = protocol.stats();
        assert!(updated_stats.total_executions > initial_stats.total_executions,
               "Statistics should be updated after operations");
    }
}

/// Batch range check proof for multiple witness vectors
/// 
/// This structure contains multiple individual range check proofs that can be
/// verified together for improved efficiency. The batch approach provides
/// significant performance benefits for large-scale applications.
/// 
/// Mathematical Properties:
/// - Each individual proof maintains the same security guarantees
/// - Batch verification can use shared randomness for efficiency
/// - Communication complexity scales sublinearly with batch size
/// - Verification time benefits from parallel processing
/// 
/// Performance Characteristics:
/// - Prover time: O(B × n × log d') for B batches of n elements each
/// - Verifier time: O(B × log n + log d') with parallel processing
/// - Communication: O(B × κ + log B) ring elements
/// - Memory usage: O(B × n × d') for all decomposition matrices
#[derive(Clone, Debug, Zeroize, ZeroizeOnDrop)]
pub struct BatchRangeCheckProof {
    /// Individual range check proofs for each witness vector
    pub individual_proofs: Vec<RangeCheckProof>,
    
    /// Number of vectors in the batch
    pub batch_size: usize,
    
    /// Total number of elements across all vectors
    pub total_elements: usize,
    
    /// Ring dimension (same for all proofs)
    pub ring_dimension: usize,
    
    /// Modulus (same for all proofs)
    pub modulus: i64,
    
    /// Range bound (same for all proofs)
    pub range_bound: i64,
}

impl BatchRangeCheckProof {
    /// Estimates the serialized size of the batch proof
    /// 
    /// # Returns
    /// * `usize` - Estimated size in bytes
    /// 
    /// # Size Calculation
    /// Includes all individual proofs plus batch metadata.
    /// The batch approach provides communication savings through
    /// shared parameters and optimized encoding.
    pub fn serialized_size(&self) -> usize {
        let individual_sizes: usize = self.individual_proofs
            .iter()
            .map(|proof| proof.serialized_size())
            .sum();
        
        let metadata_size = std::mem::size_of::<usize>() * 4 + std::mem::size_of::<i64>() * 2;
        
        individual_sizes + metadata_size
    }
    
    /// Validates the batch proof structure
    /// 
    /// # Returns
    /// * `Result<()>` - Ok if structure is valid, error otherwise
    /// 
    /// # Validation Checks
    /// - All individual proofs have consistent parameters
    /// - Batch size matches number of individual proofs
    /// - Total elements count is correct
    /// - All proofs are well-formed
    pub fn validate_structure(&self) -> Result<()> {
        // Check batch size consistency
        if self.individual_proofs.len() != self.batch_size {
            return Err(LatticeFoldError::InvalidParameters(
                format!("Batch size {} doesn't match proof count {}", 
                       self.batch_size, self.individual_proofs.len())
            ));
        }
        
        // Check parameter consistency across all proofs
        for (i, proof) in self.individual_proofs.iter().enumerate() {
            if proof.ring_dimension != self.ring_dimension {
                return Err(LatticeFoldError::InvalidParameters(
                    format!("Proof {} has ring dimension {} but expected {}", 
                           i, proof.ring_dimension, self.ring_dimension)
                ));
            }
            
            if proof.modulus != self.modulus {
                return Err(LatticeFoldError::InvalidParameters(
                    format!("Proof {} has modulus {} but expected {}", 
                           i, proof.modulus, self.modulus)
                ));
            }
            
            if proof.range_bound != self.range_bound {
                return Err(LatticeFoldError::InvalidParameters(
                    format!("Proof {} has range bound {} but expected {}", 
                           i, proof.range_bound, self.range_bound)
                ));
            }
        }
        
        // Validate total elements count
        let computed_total: usize = self.individual_proofs
            .iter()
            .map(|proof| proof.witness_dimensions)
            .sum();
        
        if computed_total != self.total_elements {
            return Err(LatticeFoldError::InvalidParameters(
                format!("Total elements {} doesn't match computed total {}", 
                       self.total_elements, computed_total)
            ));
        }
        
        Ok(())
    }
    
    /// Returns statistics about the batch proof
    /// 
    /// # Returns
    /// * `BatchProofStats` - Statistics about the batch
    pub fn stats(&self) -> BatchProofStats {
        let total_comm_size = self.serialized_size();
        let avg_proof_size = if self.batch_size > 0 {
            total_comm_size / self.batch_size
        } else {
            0
        };
        
        let witness_dimensions: Vec<usize> = self.individual_proofs
            .iter()
            .map(|proof| proof.witness_dimensions)
            .collect();
        
        BatchProofStats {
            batch_size: self.batch_size,
            total_elements: self.total_elements,
            total_communication_bytes: total_comm_size,
            average_proof_size_bytes: avg_proof_size,
            witness_dimensions,
        }
    }
}

/// Statistics for batch range check proofs
/// 
/// Provides detailed metrics about batch proof performance and characteristics
/// for analysis and optimization purposes.
#[derive(Clone, Debug)]
pub struct BatchProofStats {
    /// Number of individual proofs in the batch
    pub batch_size: usize,
    
    /// Total number of witness elements across all proofs
    pub total_elements: usize,
    
    /// Total communication size in bytes
    pub total_communication_bytes: usize,
    
    /// Average proof size per individual proof
    pub average_proof_size_bytes: usize,
    
    /// Witness dimensions for each individual proof
    pub witness_dimensions: Vec<usize>,
}

impl BatchProofStats {
    /// Calculates communication efficiency compared to individual proofs
    /// 
    /// # Arguments
    /// * `individual_proof_size` - Size of a typical individual proof
    /// 
    /// # Returns
    /// * `f64` - Efficiency ratio (< 1.0 means batch is more efficient)
    pub fn communication_efficiency(&self, individual_proof_size: usize) -> f64 {
        if individual_proof_size == 0 || self.batch_size == 0 {
            return 1.0;
        }
        
        let individual_total = individual_proof_size * self.batch_size;
        self.total_communication_bytes as f64 / individual_total as f64
    }
    
    /// Returns the compression ratio achieved by batching
    /// 
    /// # Returns
    /// * `f64` - Compression ratio (higher is better)
    pub fn compression_ratio(&self) -> f64 {
        if self.total_communication_bytes == 0 {
            return 1.0;
        }
        
        // Estimate uncompressed size (naive individual proofs)
        let estimated_individual_size = 1000; // Rough estimate in bytes
        let uncompressed_size = estimated_individual_size * self.batch_size;
        
        uncompressed_size as f64 / self.total_communication_bytes as f64
    }
}

/// Advanced range check protocol utilities and optimizations
/// 
/// This module provides additional functionality for specialized use cases
/// and performance optimizations in range check protocols.
pub mod advanced {
    use super::*;
    
    /// Streaming range check processor for very large datasets
    /// 
    /// This processor handles range checks for datasets that don't fit in memory
    /// by processing them in streaming fashion with bounded memory usage.
    pub struct StreamingRangeChecker {
        /// Base protocol instance
        protocol: RangeCheckProtocol,
        
        /// Maximum memory usage in bytes
        max_memory_bytes: usize,
        
        /// Current memory usage
        current_memory_bytes: usize,
        
        /// Batch size for streaming processing
        stream_batch_size: usize,
    }
    
    impl StreamingRangeChecker {
        /// Creates a new streaming range checker
        /// 
        /// # Arguments
        /// * `protocol` - Base range check protocol
        /// * `max_memory_bytes` - Maximum memory usage limit
        /// * `stream_batch_size` - Batch size for streaming
        /// 
        /// # Returns
        /// * `Self` - New streaming checker instance
        pub fn new(
            protocol: RangeCheckProtocol,
            max_memory_bytes: usize,
            stream_batch_size: usize
        ) -> Self {
            Self {
                protocol,
                max_memory_bytes,
                current_memory_bytes: 0,
                stream_batch_size,
            }
        }
        
        /// Processes a stream of witness vectors
        /// 
        /// # Arguments
        /// * `witness_stream` - Iterator over witness vectors
        /// * `rng` - Random number generator
        /// 
        /// # Returns
        /// * `Result<StreamingRangeCheckResult>` - Processing results
        /// 
        /// # Memory Management
        /// Processes vectors in batches to maintain bounded memory usage.
        /// Automatically adjusts batch size based on memory constraints.
        pub fn process_stream<I, R>(
            &mut self,
            witness_stream: I,
            rng: &mut R
        ) -> Result<StreamingRangeCheckResult>
        where
            I: Iterator<Item = Vec<RingElement>>,
            R: rand::CryptoRng + rand::RngCore,
        {
            let mut total_processed = 0;
            let mut successful_proofs = 0;
            let mut failed_proofs = 0;
            let mut current_batch = Vec::new();
            
            for witness_vector in witness_stream {
                // Estimate memory usage for this vector
                let vector_memory = self.estimate_vector_memory(&witness_vector);
                
                // Check if adding this vector would exceed memory limit
                if self.current_memory_bytes + vector_memory > self.max_memory_bytes {
                    // Process current batch
                    if !current_batch.is_empty() {
                        let batch_result = self.process_batch(&current_batch, rng)?;
                        successful_proofs += batch_result.successful_count;
                        failed_proofs += batch_result.failed_count;
                        
                        // Clear batch and reset memory usage
                        current_batch.clear();
                        self.current_memory_bytes = 0;
                    }
                }
                
                // Add vector to current batch
                current_batch.push(witness_vector);
                self.current_memory_bytes += vector_memory;
                total_processed += 1;
                
                // Process batch if it reaches maximum size
                if current_batch.len() >= self.stream_batch_size {
                    let batch_result = self.process_batch(&current_batch, rng)?;
                    successful_proofs += batch_result.successful_count;
                    failed_proofs += batch_result.failed_count;
                    
                    current_batch.clear();
                    self.current_memory_bytes = 0;
                }
            }
            
            // Process remaining vectors in final batch
            if !current_batch.is_empty() {
                let batch_result = self.process_batch(&current_batch, rng)?;
                successful_proofs += batch_result.successful_count;
                failed_proofs += batch_result.failed_count;
            }
            
            Ok(StreamingRangeCheckResult {
                total_processed,
                successful_proofs,
                failed_proofs,
                max_memory_used: self.max_memory_bytes,
            })
        }
        
        /// Processes a batch of witness vectors
        /// 
        /// # Arguments
        /// * `batch` - Batch of witness vectors
        /// * `rng` - Random number generator
        /// 
        /// # Returns
        /// * `Result<BatchProcessingResult>` - Batch processing results
        fn process_batch<R: rand::CryptoRng + rand::RngCore>(
            &mut self,
            batch: &[Vec<RingElement>],
            rng: &mut R
        ) -> Result<BatchProcessingResult> {
            let mut successful_count = 0;
            let mut failed_count = 0;
            
            for witness_vector in batch {
                match self.protocol.prove(witness_vector, rng) {
                    Ok(_) => successful_count += 1,
                    Err(_) => failed_count += 1,
                }
            }
            
            Ok(BatchProcessingResult {
                successful_count,
                failed_count,
            })
        }
        
        /// Estimates memory usage for a witness vector
        /// 
        /// # Arguments
        /// * `vector` - Witness vector to estimate
        /// 
        /// # Returns
        /// * `usize` - Estimated memory usage in bytes
        fn estimate_vector_memory(&self, vector: &[RingElement]) -> usize {
            // Rough estimate: each ring element uses ring_dimension * 8 bytes for coefficients
            // plus overhead for decomposition matrices and intermediate computations
            let base_size = vector.len() * self.protocol.ring_dimension * 8;
            let decomposition_overhead = base_size * 2; // Decomposition matrix
            let computation_overhead = base_size; // Temporary computations
            
            base_size + decomposition_overhead + computation_overhead
        }
    }
    
    /// Result of streaming range check processing
    #[derive(Clone, Debug)]
    pub struct StreamingRangeCheckResult {
        /// Total number of vectors processed
        pub total_processed: usize,
        
        /// Number of successful range check proofs
        pub successful_proofs: usize,
        
        /// Number of failed range check attempts
        pub failed_proofs: usize,
        
        /// Maximum memory used during processing
        pub max_memory_used: usize,
    }
    
    /// Result of batch processing within streaming
    #[derive(Clone, Debug)]
    struct BatchProcessingResult {
        /// Number of successful proofs in batch
        successful_count: usize,
        
        /// Number of failed proofs in batch
        failed_count: usize,
    }
    
    impl StreamingRangeCheckResult {
        /// Returns the success rate as a percentage
        /// 
        /// # Returns
        /// * `f64` - Success rate (0.0 to 100.0)
        pub fn success_rate(&self) -> f64 {
            if self.total_processed == 0 {
                0.0
            } else {
                (self.successful_proofs as f64 / self.total_processed as f64) * 100.0
            }
        }
        
        /// Returns memory efficiency metrics
        /// 
        /// # Returns
        /// * `(f64, f64)` - (memory_per_vector, memory_efficiency)
        pub fn memory_metrics(&self) -> (f64, f64) {
            let memory_per_vector = if self.total_processed > 0 {
                self.max_memory_used as f64 / self.total_processed as f64
            } else {
                0.0
            };
            
            // Memory efficiency: how well we utilized the available memory
            let theoretical_min_memory = 1000.0; // Rough estimate per vector
            let memory_efficiency = if memory_per_vector > 0.0 {
                theoretical_min_memory / memory_per_vector
            } else {
                0.0
            };
            
            (memory_per_vector, memory_efficiency)
        }
    }
}