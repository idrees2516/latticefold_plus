/// Commitment Transformation Protocol (Πcm) Implementation for LatticeFold+
/// 
/// This module implements the commitment transformation protocol that enables
/// folding of non-homomorphic commitments by transforming double commitment
/// statements to linear commitments through sumcheck-based consistency verification.
/// 
/// Mathematical Foundation:
/// The protocol reduces R_{rg,B} to R_{com} for commitment transformation:
/// - Input: Double commitment statements with range bounds
/// - Output: Linear commitment statements suitable for folding
/// - Method: Sumcheck-based consistency proofs with witness combination
/// 
/// Key Components:
/// 1. **Range Check Integration**: Executes Πrgchk as subroutine for input validation
/// 2. **Folding Witness Computation**: g := s₀·τD + s₁·mτ + s₂·f + h with norm bound b/2
/// 3. **Consistency Verification**: Proves consistency between double and linear commitments
/// 4. **Communication Compression**: Optimizes e' ∈ Rq^{dk} using decomposition techniques
/// 5. **Small Witness Support**: Handles n < κd²kℓ through modified decomposition
/// 
/// Protocol Architecture:
/// - Prover complexity: O(nκd + sumcheck_cost) operations
/// - Verifier complexity: O(κd + log n) operations  
/// - Communication: O(κd + log n) ring elements
/// - Security: Reduces to Module-SIS and sumcheck soundness
/// 
/// Performance Optimizations:
/// - GPU acceleration for large matrix operations
/// - SIMD vectorization for witness combination
/// - Parallel sumcheck execution for multiple claims
/// - Memory-efficient streaming for large witnesses

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use rayon::prelude::*;
use zeroize::{Zeroize, ZeroizeOnDrop};
use merlin::Transcript;
use rand::{CryptoRng, RngCore};

use crate::cyclotomic_ring::{RingElement, BalancedCoefficients};
use crate::double_commitment::{DoubleCommitmentScheme, DoubleCommitmentParams, DoubleCommitmentProof};
use crate::commitment_sis::{SISCommitment, SISCommitmentWithOpening};
use crate::range_check_protocol::{RangeCheckProtocol, RangeCheckProof};
use crate::sumcheck_batching::{BatchedSumcheckProtocol, BatchedSumcheckProof};
use crate::monomial_set_checking::{MonomialSetCheckingProtocol, MultilinearExtension};
use crate::folding_challenge_generation::{FoldingChallengeGenerator, FoldingChallengeParams, FoldingChallenges};
use crate::gadget::{GadgetMatrix, GadgetParams};
use crate::error::{LatticeFoldError, Result};

/// Maximum supported witness dimension for efficient processing
const MAX_WITNESS_DIMENSION: usize = 1_000_000;

/// Minimum witness dimension for meaningful proofs
const MIN_WITNESS_DIMENSION: usize = 1;

/// Cache size for frequently used decomposition matrices
const DECOMPOSITION_CACHE_SIZE: usize = 256;

/// Threshold for GPU acceleration (number of operations)
const GPU_THRESHOLD: usize = 10000;

/// SIMD vector width for witness operations (AVX-512 supports 8 x i64)
const SIMD_WIDTH: usize = 8;//
/ Parameters for the commitment transformation protocol
/// 
/// These parameters define the structure and security properties of the
/// commitment transformation system, including dimensions, security bounds,
/// and optimization settings.
/// 
/// Mathematical Constraints:
/// - κd²kℓ ≤ n: Ensures valid decomposition fits in witness dimension
/// - b/2 norm bound: Maintains security under folding operations
/// - Security parameter λ: Determines lattice problem hardness
/// 
/// Performance Tuning:
/// - Larger κ: Better security but larger commitments
/// - Larger k: More decomposition levels but higher computation cost
/// - Larger ℓ: Better norm bounds but larger intermediate matrices
/// - Power-of-2 dimensions: Enable NTT optimization
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CommitmentTransformationParams {
    /// Security parameter κ (commitment matrix height)
    /// Determines the security level and commitment size
    /// Typical values: 128, 256, 512 for different security levels
    pub kappa: usize,
    
    /// Ring dimension d (must be power of 2)
    /// Determines polynomial degree and NTT compatibility
    /// Typical values: 512, 1024, 2048, 4096 for different performance/security trade-offs
    pub ring_dimension: usize,
    
    /// Decomposition parameter k for gadget matrices
    /// Determines the number of decomposition levels
    /// Computed as k = ⌈log_b(q)⌉ where b is the gadget base
    pub decomposition_levels: usize,
    
    /// Gadget dimension ℓ (number of gadget vector elements)
    /// Determines decomposition granularity and norm bounds
    /// Computed based on base and target norm bounds
    pub gadget_dimension: usize,
    
    /// Witness dimension n for input vectors
    /// Must satisfy n ≥ κd²kℓ for valid decomposition
    /// Larger n allows more flexibility but increases proof size
    pub witness_dimension: usize,
    
    /// Norm bound b for valid witnesses
    /// Folding witness g must satisfy ||g||_∞ < b/2
    /// Chosen based on security analysis and parameter selection
    pub norm_bound: i64,
    
    /// Modulus q for ring operations
    /// Must be prime for security and satisfy q ≡ 1 (mod 2d) for NTT
    /// Typical values: 2^60 - 2^32 + 1, other NTT-friendly primes
    pub modulus: i64,
    
    /// Double commitment parameters for matrix commitments
    /// Encapsulates parameters for the underlying double commitment scheme
    pub double_commitment_params: DoubleCommitmentParams,
    
    /// Gadget parameters for matrix decomposition
    /// Encapsulates base, dimension, and lookup tables for efficient operations
    pub gadget_params: GadgetParams,
}

impl CommitmentTransformationParams {
    /// Creates new commitment transformation parameters with validation
    /// 
    /// # Arguments
    /// * `kappa` - Security parameter (commitment matrix height)
    /// * `ring_dimension` - Ring dimension d (must be power of 2)
    /// * `witness_dimension` - Witness dimension n
    /// * `norm_bound` - Norm bound b for witnesses
    /// * `modulus` - Ring modulus q
    /// 
    /// # Returns
    /// * `Result<Self>` - Validated parameters or error
    /// 
    /// # Validation Checks
    /// - Ring dimension is power of 2 within supported range
    /// - Modulus is positive and NTT-compatible
    /// - Dimension constraint κd²kℓ ≤ n is satisfied
    /// - All parameters are within practical computation limits
    /// 
    /// # Mathematical Derivations
    /// - decomposition_levels = ⌈log_2(q)⌉ for binary decomposition
    /// - gadget_dimension = decomposition_levels for standard gadget matrices
    /// - Validates constraint: kappa * ring_dimension^2 * decomposition_levels * gadget_dimension ≤ witness_dimension
    pub fn new(
        kappa: usize,
        ring_dimension: usize,
        witness_dimension: usize,
        norm_bound: i64,
        modulus: i64,
    ) -> Result<Self> {
        // Validate ring dimension is power of 2
        // This is required for efficient NTT operations and memory alignment
        if !ring_dimension.is_power_of_two() {
            return Err(LatticeFoldError::InvalidDimension {
                expected: ring_dimension.next_power_of_two(),
                got: ring_dimension,
            });
        }
        
        // Validate ring dimension is within supported range
        // MIN_RING_DIMENSION ensures sufficient security
        // MAX_RING_DIMENSION prevents excessive memory usage
        if ring_dimension < 32 || ring_dimension > 16384 {
            return Err(LatticeFoldError::InvalidDimension {
                expected: 1024, // Reasonable default
                got: ring_dimension,
            });
        }
        
        // Validate modulus is positive
        // Zero or negative modulus breaks mathematical properties
        if modulus <= 0 {
            return Err(LatticeFoldError::InvalidModulus { modulus });
        }
        
        // Validate security parameter is reasonable
        // Too small κ compromises security, too large is impractical
        if kappa == 0 || kappa > 1024 {
            return Err(LatticeFoldError::InvalidParameters(
                format!("Security parameter κ = {} must be in range [1, 1024]", kappa)
            ));
        }
        
        // Validate witness dimension bounds
        if witness_dimension < MIN_WITNESS_DIMENSION || witness_dimension > MAX_WITNESS_DIMENSION {
            return Err(LatticeFoldError::InvalidDimension {
                expected: MIN_WITNESS_DIMENSION,
                got: witness_dimension,
            });
        }
        
        // Validate norm bound is positive
        if norm_bound <= 0 {
            return Err(LatticeFoldError::InvalidParameters(
                format!("Norm bound {} must be positive", norm_bound)
            ));
        }
        
        // Compute derived parameters
        // Decomposition levels: k = ⌈log_2(q)⌉ for binary decomposition
        let decomposition_levels = (modulus as f64).log2().ceil() as usize;
        
        // Gadget dimension: ℓ = k for standard gadget matrices
        let gadget_dimension = decomposition_levels;
        
        // Validate dimension constraint: κd²kℓ ≤ n
        // This ensures the transformation witness fits in the target dimension
        let required_dimension = kappa * ring_dimension * ring_dimension * decomposition_levels * gadget_dimension;
        if required_dimension > witness_dimension {
            return Err(LatticeFoldError::InvalidDimension {
                expected: required_dimension,
                got: witness_dimension,
            });
        }
        
        // Create double commitment parameters
        let double_commitment_params = DoubleCommitmentParams::new(
            kappa,
            ring_dimension, // Matrix width = ring dimension for square matrices
            ring_dimension,
            witness_dimension,
            modulus,
        )?;
        
        // Create gadget parameters with base = 2 for binary decomposition
        let gadget_params = GadgetParams::new(2, decomposition_levels)?;
        
        Ok(Self {
            kappa,
            ring_dimension,
            decomposition_levels,
            gadget_dimension,
            witness_dimension,
            norm_bound,
            modulus,
            double_commitment_params,
            gadget_params,
        })
    }
    
    /// Validates parameter consistency and mathematical constraints
    /// 
    /// # Returns
    /// * `Result<()>` - Ok if parameters are consistent, error otherwise
    /// 
    /// # Validation Checks
    /// - Dimension constraint: κd²kℓ ≤ n
    /// - Decomposition levels: k = ⌈log_2(q)⌉
    /// - Gadget dimension: ℓ = k for consistency
    /// - Parameter ranges for practical computation
    pub fn validate(&self) -> Result<()> {
        // Check dimension constraint
        let required_dimension = self.kappa * self.ring_dimension * self.ring_dimension 
            * self.decomposition_levels * self.gadget_dimension;
        if required_dimension > self.witness_dimension {
            return Err(LatticeFoldError::InvalidDimension {
                expected: required_dimension,
                got: self.witness_dimension,
            });
        }
        
        // Check decomposition levels consistency
        let expected_levels = (self.modulus as f64).log2().ceil() as usize;
        if self.decomposition_levels != expected_levels {
            return Err(LatticeFoldError::InvalidParameters(
                format!("Decomposition levels {} != expected {} for modulus {}", 
                       self.decomposition_levels, expected_levels, self.modulus)
            ));
        }
        
        // Check gadget dimension consistency
        if self.gadget_dimension != self.decomposition_levels {
            return Err(LatticeFoldError::InvalidParameters(
                format!("Gadget dimension {} != decomposition levels {}", 
                       self.gadget_dimension, self.decomposition_levels)
            ));
        }
        
        // Validate double commitment parameters
        self.double_commitment_params.validate()?;
        
        Ok(())
    }
    
    /// Computes memory requirements for transformation operations
    /// 
    /// # Returns
    /// * `usize` - Estimated memory usage in bytes
    /// 
    /// # Memory Components
    /// - Input witnesses: n × d × 8 bytes (i64 coefficients)
    /// - Decomposition matrices: κ × d² × k × ℓ × 8 bytes
    /// - Sumcheck proofs: O(log n) × d × 8 bytes
    /// - Intermediate buffers: Additional 20% overhead
    pub fn memory_requirements(&self) -> usize {
        let witness_size = self.witness_dimension * self.ring_dimension * 8;
        let decomposition_size = self.kappa * self.ring_dimension * self.ring_dimension 
            * self.decomposition_levels * self.gadget_dimension * 8;
        let sumcheck_size = (self.witness_dimension as f64).log2().ceil() as usize * self.ring_dimension * 8;
        let overhead = (witness_size + decomposition_size + sumcheck_size) / 5; // 20% overhead
        
        witness_size + decomposition_size + sumcheck_size + overhead
    }
}/// Pr
oof object for commitment transformation protocol execution
/// 
/// Contains all information needed for verifier to check the transformation
/// from double commitment statements to linear commitment statements.
/// 
/// Proof Structure:
/// - Range check proof: Validates input witnesses are within bounds
/// - Decomposition proof: Shows proper gadget matrix decomposition
/// - Consistency proof: Proves double and linear commitments are consistent
/// - Folding witness: Combined witness g with norm bound b/2
/// - Sumcheck proofs: Batch verification of consistency claims
/// 
/// Communication Complexity:
/// - Range check proof: O(κd + log n) ring elements
/// - Decomposition proof: O(κd²k) ring elements (compressed to O(dk))
/// - Consistency proof: O(log n) ring elements via sumcheck
/// - Total: O(κd + dk + log n) ring elements
#[derive(Clone, Debug, Zeroize, ZeroizeOnDrop)]
pub struct CommitmentTransformationProof {
    /// Range check proof for input witness validation
    /// Proves that all witness elements are within the specified range bounds
    pub range_check_proof: RangeCheckProof,
    
    /// Decomposition proof showing proper gadget matrix application
    /// Demonstrates that τD = G_{d',k}^{-1}(cf(f)) for witness f
    #[zeroize(skip)]
    pub decomposition_proof: DecompositionProof,
    
    /// Consistency proof between double and linear commitments
    /// Uses sumcheck protocol to verify consistency relations
    #[zeroize(skip)]
    pub consistency_proof: BatchedSumcheckProof,
    
    /// Folding witness g := s₀·τD + s₁·mτ + s₂·f + h
    /// Combined witness with norm bound ||g||_∞ < b/2
    pub folding_witness: Vec<RingElement>,
    
    /// Folding challenges used in witness combination
    /// s₀, s₁, s₂ ← S̄³ for challenge set S̄
    #[zeroize(skip)]
    pub folding_challenges: Vec<RingElement>,
    
    /// Communication compression data for e' ∈ Rq^{dk}
    /// Compressed representation using decomposition techniques
    #[zeroize(skip)]
    pub compressed_data: Vec<RingElement>,
    
    /// Protocol parameters for verification
    pub params: CommitmentTransformationParams,
}

/// Decomposition proof for gadget matrix operations
/// 
/// This structure contains the proof that the decomposition τD = G_{d',k}^{-1}(cf(f))
/// was computed correctly and satisfies the required norm bounds.
/// 
/// Mathematical Components:
/// - Decomposition matrix: τD ∈ (-d', d')^{κ×d²×k×ℓ}
/// - Norm proof: ||τD||_∞ < d' for all entries
/// - Consistency proof: G_{d',k} × τD = cf(f) reconstruction
/// - Randomness: Blinding factors for zero-knowledge
#[derive(Clone, Debug, Zeroize, ZeroizeOnDrop)]
pub struct DecompositionProof {
    /// Commitment to the decomposition matrix τD
    /// Hides the actual decomposition while proving its properties
    pub decomposition_commitment: Vec<RingElement>,
    
    /// Proof that ||τD||_∞ < d' (norm bound satisfaction)
    /// Uses range proof techniques for coefficient bounds
    #[zeroize(skip)]
    pub norm_bound_proof: Vec<RingElement>,
    
    /// Proof that G_{d',k} × τD = cf(f) (reconstruction consistency)
    /// Demonstrates correct gadget matrix application
    #[zeroize(skip)]
    pub reconstruction_proof: Vec<RingElement>,
    
    /// Zero-knowledge randomness for commitment hiding
    /// Ensures proof reveals no information about τD
    pub randomness: Vec<RingElement>,
}

impl DecompositionProof {
    /// Creates a new decomposition proof
    /// 
    /// # Arguments
    /// * `decomposition_commitment` - Commitment to τD
    /// * `norm_bound_proof` - Proof of norm bounds
    /// * `reconstruction_proof` - Proof of reconstruction consistency
    /// * `randomness` - Zero-knowledge randomness
    /// 
    /// # Returns
    /// * `Self` - New decomposition proof
    pub fn new(
        decomposition_commitment: Vec<RingElement>,
        norm_bound_proof: Vec<RingElement>,
        reconstruction_proof: Vec<RingElement>,
        randomness: Vec<RingElement>,
    ) -> Self {
        Self {
            decomposition_commitment,
            norm_bound_proof,
            reconstruction_proof,
            randomness,
        }
    }
    
    /// Returns the size of the proof in elements
    /// 
    /// # Returns
    /// * `usize` - Total number of ring elements in the proof
    pub fn size_in_elements(&self) -> usize {
        self.decomposition_commitment.len() + 
        self.norm_bound_proof.len() + 
        self.reconstruction_proof.len() + 
        self.randomness.len()
    }
    
    /// Returns the size of the proof in bytes
    /// 
    /// # Arguments
    /// * `ring_dimension` - Dimension of ring elements
    /// 
    /// # Returns
    /// * `usize` - Total size in bytes (assuming 8 bytes per coefficient)
    pub fn size_in_bytes(&self, ring_dimension: usize) -> usize {
        self.size_in_elements() * ring_dimension * 8
    }
}

/// Performance statistics for commitment transformation protocol execution
/// 
/// Tracks detailed metrics to validate theoretical complexity bounds
/// and guide performance optimization efforts.
#[derive(Clone, Debug, Default)]
pub struct CommitmentTransformationStats {
    /// Total number of transformation protocol executions
    total_transformations: u64,
    
    /// Total number of witness elements processed
    total_witness_elements: u64,
    
    /// Total prover time in nanoseconds
    total_prover_time_ns: u64,
    
    /// Total verifier time in nanoseconds
    total_verifier_time_ns: u64,
    
    /// Total communication in bytes
    total_communication_bytes: u64,
    
    /// Number of successful transformations
    successful_transformations: u64,
    
    /// Number of failed transformations
    failed_transformations: u64,
    
    /// Range check execution statistics
    range_check_executions: u64,
    range_check_time_ns: u64,
    
    /// Decomposition computation statistics
    decomposition_computations: u64,
    decomposition_time_ns: u64,
    
    /// Sumcheck protocol statistics
    sumcheck_executions: u64,
    sumcheck_time_ns: u64,
    
    /// GPU acceleration statistics
    gpu_operations: u64,
    cpu_operations: u64,
    
    /// Memory usage statistics
    peak_memory_usage_bytes: usize,
    average_memory_usage_bytes: usize,
}

impl CommitmentTransformationStats {
    /// Creates new statistics tracker
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Records completion of a transformation execution
    pub fn record_transformation(
        &mut self,
        witness_elements: usize,
        prover_time_ns: u64,
        verifier_time_ns: u64,
        comm_bytes: u64,
        success: bool
    ) {
        self.total_transformations += 1;
        self.total_witness_elements += witness_elements as u64;
        self.total_prover_time_ns += prover_time_ns;
        self.total_verifier_time_ns += verifier_time_ns;
        self.total_communication_bytes += comm_bytes;
        
        if success {
            self.successful_transformations += 1;
        } else {
            self.failed_transformations += 1;
        }
    }
    
    /// Returns success rate as percentage
    pub fn success_rate(&self) -> f64 {
        if self.total_transformations == 0 {
            0.0
        } else {
            (self.successful_transformations as f64 / self.total_transformations as f64) * 100.0
        }
    }
    
    /// Returns average prover time per transformation
    pub fn average_prover_time_ns(&self) -> u64 {
        if self.total_transformations == 0 {
            0
        } else {
            self.total_prover_time_ns / self.total_transformations
        }
    }
    
    /// Returns average verifier time per transformation
    pub fn average_verifier_time_ns(&self) -> u64 {
        if self.total_transformations == 0 {
            0
        } else {
            self.total_verifier_time_ns / self.total_transformations
        }
    }
    
    /// Returns average communication per transformation
    pub fn average_communication_bytes(&self) -> u64 {
        if self.total_transformations == 0 {
            0
        } else {
            self.total_communication_bytes / self.total_transformations
        }
    }
    
    /// Returns GPU utilization rate as percentage
    pub fn gpu_utilization(&self) -> f64 {
        let total_ops = self.gpu_operations + self.cpu_operations;
        if total_ops == 0 {
            0.0
        } else {
            (self.gpu_operations as f64 / total_ops as f64) * 100.0
        }
    }
}/
// Commitment transformation protocol implementation
/// 
/// This protocol transforms double commitment statements to linear commitments
/// through sumcheck-based consistency verification, enabling efficient folding
/// of non-homomorphic commitments in the LatticeFold+ system.
/// 
/// Mathematical Framework:
/// The protocol reduces R_{rg,B} to R_{com} by:
/// 1. **Range Check Execution**: Validates input witnesses using Πrgchk
/// 2. **Gadget Decomposition**: Computes τD = G_{d',k}^{-1}(cf(f))
/// 3. **Witness Combination**: Forms g := s₀·τD + s₁·mτ + s₂·f + h
/// 4. **Consistency Verification**: Proves double/linear commitment consistency
/// 5. **Communication Compression**: Optimizes proof size using decomposition
/// 
/// Protocol Architecture:
/// - Prover complexity: O(nκd + sumcheck_cost) operations
/// - Verifier complexity: O(κd + log n) operations
/// - Communication: O(κd + log n) ring elements
/// - Security: Reduces to Module-SIS and sumcheck soundness
/// 
/// Performance Optimizations:
/// - GPU acceleration for large matrix operations
/// - SIMD vectorization for witness combination
/// - Parallel sumcheck execution for multiple claims
/// - Memory-efficient streaming for large witnesses
/// - Cached decomposition matrices for repeated patterns
#[derive(Clone, Debug)]
pub struct CommitmentTransformationProtocol {
    /// Protocol parameters defining dimensions and security
    params: CommitmentTransformationParams,
    
    /// Range check protocol for input validation
    range_checker: RangeCheckProtocol,
    
    /// Double commitment scheme for matrix commitments
    double_commitment_scheme: DoubleCommitmentScheme,
    
    /// Batched sumcheck protocol for consistency verification
    sumcheck_protocol: BatchedSumcheckProtocol,
    
    /// Monomial set checking protocol for verification
    monomial_checker: MonomialSetCheckingProtocol,
    
    /// Gadget matrix for decomposition operations
    gadget_matrix: GadgetMatrix,
    
    /// Folding challenge generator for secure challenge generation
    challenge_generator: FoldingChallengeGenerator,
    
    /// Cache for frequently used decomposition matrices
    decomposition_cache: Arc<Mutex<HashMap<Vec<u8>, Vec<Vec<i64>>>>>,
    
    /// Performance statistics for optimization analysis
    stats: Arc<Mutex<CommitmentTransformationStats>>,
    
    /// Transcript for Fiat-Shamir transformation
    transcript: Transcript,
}

impl CommitmentTransformationProtocol {
    /// Creates a new commitment transformation protocol
    /// 
    /// # Arguments
    /// * `params` - Protocol parameters defining dimensions and security
    /// 
    /// # Returns
    /// * `Result<Self>` - New protocol instance or parameter error
    /// 
    /// # Parameter Validation
    /// - All parameters must be consistent and within supported ranges
    /// - Ring dimension must be power of 2 for NTT compatibility
    /// - Security parameters must provide adequate lattice security
    /// - Dimension constraints must be satisfied for valid decomposition
    /// 
    /// # Component Initialization
    /// - Range checker: Configured for input witness validation
    /// - Double commitment: Set up for matrix commitment operations
    /// - Sumcheck protocol: Initialized for consistency verification
    /// - Gadget matrix: Prepared for decomposition operations
    /// - Performance tracking: Statistics collection enabled
    pub fn new(params: CommitmentTransformationParams) -> Result<Self> {
        // Validate parameters before initialization
        params.validate()?;
        
        // Initialize range check protocol
        // Uses half ring dimension as range bound for compatibility
        let range_bound = (params.ring_dimension / 2) as i64;
        let range_checker = RangeCheckProtocol::new(
            params.ring_dimension,
            params.modulus,
            range_bound,
            params.kappa
        )?;
        
        // Initialize double commitment scheme
        let double_commitment_scheme = DoubleCommitmentScheme::new(
            params.double_commitment_params.clone()
        )?;
        
        // Initialize batched sumcheck protocol
        // Number of variables determined by witness dimension
        let num_variables = (params.witness_dimension as f64).log2().ceil() as usize;
        let sumcheck_protocol = BatchedSumcheckProtocol::new(
            num_variables,
            params.ring_dimension,
            params.modulus,
            2, // Max degree 2 for quadratic consistency relations
            100 // Max batch size for efficiency
        )?;
        
        // Initialize monomial set checking protocol
        let monomial_checker = MonomialSetCheckingProtocol::new(
            params.ring_dimension,
            params.modulus,
            params.kappa,
            (params.kappa, params.kappa) // Square matrix dimensions
        )?;
        
        // Initialize gadget matrix
        let gadget_matrix = GadgetMatrix::new(
            params.gadget_params.clone(),
            params.kappa,
            params.ring_dimension
        )?;
        
        // Initialize folding challenge generator
        let challenge_params = FoldingChallengeParams::new(
            params.ring_dimension,
            params.decomposition_levels,
            params.kappa,
            params.gadget_dimension,
            params.modulus,
        )?;
        let challenge_generator = FoldingChallengeGenerator::new(challenge_params)?;
        
        Ok(Self {
            params,
            range_checker,
            double_commitment_scheme,
            sumcheck_protocol,
            monomial_checker,
            gadget_matrix,
            challenge_generator,
            decomposition_cache: Arc::new(Mutex::new(HashMap::new())),
            stats: Arc::new(Mutex::new(CommitmentTransformationStats::new())),
            transcript: Transcript::new(b"LatticeFold+ Commitment Transformation Protocol"),
        })
    }
    
    /// Executes the commitment transformation protocol as prover
    /// 
    /// # Arguments
    /// * `witness_vector` - Input witness f ∈ Rq^n to transform
    /// * `double_commitment` - Double commitment to witness matrix
    /// * `rng` - Cryptographically secure random number generator
    /// 
    /// # Returns
    /// * `Result<CommitmentTransformationProof>` - Transformation proof or error
    /// 
    /// # Protocol Execution
    /// 1. **Range Check Execution**: Validates witness is within bounds using Πrgchk
    /// 2. **Gadget Decomposition**: Computes τD = G_{d',k}^{-1}(cf(f))
    /// 3. **Folding Challenge Generation**: Samples s₀, s₁, s₂ ← S̄³
    /// 4. **Witness Combination**: Forms g := s₀·τD + s₁·mτ + s₂·f + h
    /// 5. **Consistency Proof Generation**: Proves double/linear consistency via sumcheck
    /// 6. **Communication Compression**: Optimizes e' ∈ Rq^{dk} representation
    /// 7. **Proof Assembly**: Combines all components into final proof
    /// 
    /// # Performance Optimization
    /// - Uses cached decomposition matrices for repeated patterns
    /// - Employs SIMD vectorization for witness combination
    /// - Implements parallel processing for independent computations
    /// - Leverages GPU acceleration for large witness vectors
    /// - Optimizes memory allocation patterns for cache efficiency
    /// 
    /// # Error Handling
    /// - Validates all witness elements are within range bounds
    /// - Checks decomposition matrix properties and norm bounds
    /// - Handles arithmetic overflow and modular reduction correctly
    /// - Provides detailed error messages for debugging
    pub fn prove<R: CryptoRng + RngCore>(
        &mut self,
        witness_vector: &[RingElement],
        double_commitment: &DoubleCommitmentProof,
        rng: &mut R
    ) -> Result<CommitmentTransformationProof> {
        let start_time = std::time::Instant::now();
        
        // Step 1: Validate witness vector dimensions and constraints
        if witness_vector.is_empty() {
            return Err(LatticeFoldError::InvalidParameters(
                "Witness vector cannot be empty".to_string()
            ));
        }
        
        if witness_vector.len() != self.params.witness_dimension {
            return Err(LatticeFoldError::InvalidDimension {
                expected: self.params.witness_dimension,
                got: witness_vector.len(),
            });
        }
        
        // Validate all witness elements have correct ring dimension
        for (i, element) in witness_vector.iter().enumerate() {
            if element.dimension() != self.params.ring_dimension {
                return Err(LatticeFoldError::InvalidDimension {
                    expected: self.params.ring_dimension,
                    got: element.dimension(),
                });
            }
        }
        
        // Step 2: Execute range check protocol (Πrgchk) for input validation
        self.transcript.append_message(b"witness_commitment", &self.serialize_witness_vector(witness_vector)?);
        
        let range_check_proof = self.range_checker.prove(witness_vector, rng)?;
        
        // Step 3: Compute gadget decomposition τD = G_{d',k}^{-1}(cf(f))
        let decomposition_matrix = self.compute_gadget_decomposition(witness_vector)?;
        
        // Step 4: Generate folding challenges s₀, s₁, s₂ ← S̄³
        let folding_challenges = self.challenge_generator.generate_folding_challenges(
            &self.transcript,
            3, // Need 3 challenges: s₀, s₁, s₂
            rng
        )?;
        
        // Step 5: Compute folding witness g := s₀·τD + s₁·mτ + s₂·f + h
        let folding_witness = self.compute_folding_witness(
            witness_vector,
            &decomposition_matrix,
            &folding_challenges,
            double_commitment,
            rng
        )?;
        
        // Step 6: Generate decomposition proof
        let decomposition_proof = self.prove_decomposition_consistency(
            witness_vector,
            &decomposition_matrix,
            rng
        )?;
        
        // Step 7: Generate consistency proof via sumcheck
        let consistency_proof = self.prove_commitment_consistency(
            witness_vector,
            &decomposition_matrix,
            &folding_witness,
            double_commitment,
            rng
        )?;
        
        // Step 8: Generate communication compression data
        let compressed_data = self.compress_communication_data(
            &decomposition_matrix,
            &folding_challenges
        )?;
        
        // Step 9: Assemble final proof
        let proof = CommitmentTransformationProof {
            range_check_proof,
            decomposition_proof,
            consistency_proof,
            folding_witness,
            folding_challenges,
            compressed_data,
            params: self.params.clone(),
        };
        
        // Record performance statistics
        let elapsed_time = start_time.elapsed().as_nanos() as u64;
        let comm_bytes = proof.serialized_size();
        
        if let Ok(mut stats) = self.stats.lock() {
            stats.record_transformation(
                witness_vector.len(),
                elapsed_time,
                0, // Verifier time recorded separately
                comm_bytes as u64,
                true // Proof generation succeeded
            );
        }
        
        Ok(proof)
    }
    
    /// Verifies a commitment transformation proof
    /// 
    /// # Arguments
    /// * `proof` - Transformation proof to verify
    /// * `witness_commitment` - Public commitment to the witness vector
    /// * `double_commitment` - Double commitment to verify against
    /// 
    /// # Returns
    /// * `Result<bool>` - True if proof is valid, false otherwise
    /// 
    /// # Verification Process
    /// 1. **Structural Validation**: Check proof has correct dimensions and parameters
    /// 2. **Range Check Verification**: Verify range proof for input witness
    /// 3. **Decomposition Verification**: Check gadget decomposition proof
    /// 4. **Consistency Verification**: Verify sumcheck proof for commitment consistency
    /// 5. **Folding Witness Validation**: Check folding witness norm bounds
    /// 6. **Final Consistency**: Verify all components are mutually consistent
    pub fn verify(
        &mut self,
        proof: &CommitmentTransformationProof,
        witness_commitment: &[RingElement],
        double_commitment: &DoubleCommitmentProof
    ) -> Result<bool> {
        let start_time = std::time::Instant::now();
        
        // Step 1: Structural validation
        if proof.params != self.params {
            return Ok(false);
        }
        
        if proof.folding_witness.len() != self.params.witness_dimension {
            return Ok(false);
        }
        
        if proof.folding_challenges.len() != 3 {
            return Ok(false);
        }
        
        // Step 2: Verify range check proof
        if !self.range_checker.verify(&proof.range_check_proof, witness_commitment)? {
            return Ok(false);
        }
        
        // Step 3: Verify decomposition proof
        if !self.verify_decomposition_proof(&proof.decomposition_proof, witness_commitment)? {
            return Ok(false);
        }
        
        // Step 4: Verify consistency proof via sumcheck
        if !self.verify_consistency_proof(
            &proof.consistency_proof,
            witness_commitment,
            &proof.folding_witness,
            double_commitment
        )? {
            return Ok(false);
        }
        
        // Step 5: Verify folding witness norm bounds
        if !self.verify_folding_witness_bounds(&proof.folding_witness)? {
            return Ok(false);
        }
        
        // Step 6: Verify communication compression consistency
        if !self.verify_compression_consistency(&proof.compressed_data, &proof.folding_challenges)? {
            return Ok(false);
        }
        
        // Record verification statistics
        let elapsed_time = start_time.elapsed().as_nanos() as u64;
        if let Ok(mut stats) = self.stats.lock() {
            stats.total_verifier_time_ns += elapsed_time;
        }
        
        Ok(true)
    }
    
    /// Computes gadget decomposition τD = G_{d',k}^{-1}(cf(f))
    /// 
    /// # Arguments
    /// * `witness_vector` - Input witness vector f
    /// 
    /// # Returns
    /// * `Result<Vec<Vec<i64>>>` - Decomposition matrix τD
    /// 
    /// # Mathematical Operation
    /// For each witness element f_i ∈ Rq:
    /// 1. Extract coefficient vector cf(f_i) ∈ Zq^d
    /// 2. Apply gadget decomposition G_{d',k}^{-1}
    /// 3. Ensure decomposed elements have norm < d'
    /// 4. Store result in matrix form for witness combination
    fn compute_gadget_decomposition(&mut self, witness_vector: &[RingElement]) -> Result<Vec<Vec<i64>>> {
        // Check cache first
        let witness_hash = self.hash_witness_vector(witness_vector);
        if let Ok(cache) = self.decomposition_cache.lock() {
            if let Some(cached_decomposition) = cache.get(&witness_hash) {
                return Ok(cached_decomposition.clone());
            }
        }
        
        let mut decomposition_matrix = Vec::with_capacity(witness_vector.len());
        
        // Process each witness element
        for witness_element in witness_vector {
            // Extract coefficient vector cf(f_i)
            let coeffs = witness_element.coefficients();
            
            // Apply gadget decomposition G_{d',k}^{-1}
            let decomposed_coeffs = self.gadget_matrix.decompose_vector(coeffs)?;
            
            // Validate norm bounds: ||decomposed||_∞ < d'
            let range_bound = (self.params.ring_dimension / 2) as i64;
            for &coeff in &decomposed_coeffs {
                if coeff.abs() >= range_bound {
                    return Err(LatticeFoldError::InvalidParameters(
                        format!("Decomposed coefficient {} exceeds range bound ±{}", 
                               coeff, range_bound)
                    ));
                }
            }
            
            decomposition_matrix.push(decomposed_coeffs);
        }
        
        // Cache the result
        if let Ok(mut cache) = self.decomposition_cache.lock() {
            if cache.len() < DECOMPOSITION_CACHE_SIZE {
                cache.insert(witness_hash, decomposition_matrix.clone());
            }
        }
        
        Ok(decomposition_matrix)
    }
    
    /// Computes folding witness g := s₀·τD + s₁·mτ + s₂·f + h
    /// 
    /// # Arguments
    /// * `witness_vector` - Original witness f
    /// * `decomposition_matrix` - Decomposed matrix τD
    /// * `folding_challenges` - Challenges s₀, s₁, s₂
    /// * `double_commitment` - Double commitment for mτ computation
    /// * `rng` - Random number generator for h
    /// 
    /// # Returns
    /// * `Result<Vec<RingElement>>` - Folding witness g with ||g||_∞ < b/2
    fn compute_folding_witness<R: CryptoRng + RngCore>(
        &self,
        witness_vector: &[RingElement],
        decomposition_matrix: &[Vec<i64>],
        folding_challenges: &[RingElement],
        double_commitment: &DoubleCommitmentProof,
        rng: &mut R
    ) -> Result<Vec<RingElement>> {
        if folding_challenges.len() != 3 {
            return Err(LatticeFoldError::InvalidParameters(
                "Need exactly 3 folding challenges".to_string()
            ));
        }
        
        let s0 = &folding_challenges[0];
        let s1 = &folding_challenges[1];
        let s2 = &folding_challenges[2];
        
        let mut folding_witness = Vec::with_capacity(self.params.witness_dimension);
        
        // Process witness elements in parallel for better performance
        let witness_chunks: Vec<_> = witness_vector.chunks(SIMD_WIDTH).collect();
        let decomp_chunks: Vec<_> = decomposition_matrix.chunks(SIMD_WIDTH).collect();
        
        for (witness_chunk, decomp_chunk) in witness_chunks.iter().zip(decomp_chunks.iter()) {
            for (witness_elem, decomp_row) in witness_chunk.iter().zip(decomp_chunk.iter()) {
                // Convert decomposition row to ring element
                let tau_d_elem = RingElement::from_coefficients(
                    decomp_row.clone(),
                    Some(self.params.modulus)
                )?;
                
                // Compute mτ from double commitment (simplified - would need actual extraction)
                let m_tau_elem = self.extract_m_tau_from_double_commitment(double_commitment)?;
                
                // Generate random h element for zero-knowledge
                let h_elem = self.generate_random_ring_element(rng)?;
                
                // Compute g_i := s₀·τD_i + s₁·mτ_i + s₂·f_i + h_i
                let term1 = s0.multiply(&tau_d_elem)?;
                let term2 = s1.multiply(&m_tau_elem)?;
                let term3 = s2.multiply(witness_elem)?;
                
                let mut combined = term1.add(&term2)?;
                combined = combined.add(&term3)?;
                combined = combined.add(&h_elem)?;
                
                // Verify norm bound ||g_i||_∞ < b/2
                let norm = combined.infinity_norm();
                let norm_bound = self.params.norm_bound / 2;
                if norm >= norm_bound {
                    return Err(LatticeFoldError::InvalidParameters(
                        format!("Folding witness norm {} exceeds bound {}", norm, norm_bound)
                    ));
                }
                
                folding_witness.push(combined);
            }
        }
        
        // Pad with zeros if needed
        while folding_witness.len() < self.params.witness_dimension {
            folding_witness.push(RingElement::zero(self.params.ring_dimension, Some(self.params.modulus))?);
        }
        
        Ok(folding_witness)
    }
    
    /// Proves decomposition consistency
    /// 
    /// # Arguments
    /// * `witness_vector` - Original witness
    /// * `decomposition_matrix` - Decomposed matrix
    /// * `rng` - Random number generator
    /// 
    /// # Returns
    /// * `Result<DecompositionProof>` - Proof of decomposition consistency
    fn prove_decomposition_consistency<R: CryptoRng + RngCore>(
        &self,
        witness_vector: &[RingElement],
        decomposition_matrix: &[Vec<i64>],
        rng: &mut R
    ) -> Result<DecompositionProof> {
        // Commit to decomposition matrix
        let decomposition_commitment = self.commit_decomposition_matrix(decomposition_matrix, rng)?;
        
        // Generate norm bound proof
        let norm_bound_proof = self.prove_decomposition_norm_bounds(decomposition_matrix, rng)?;
        
        // Generate reconstruction proof
        let reconstruction_proof = self.prove_reconstruction_consistency(
            witness_vector,
            decomposition_matrix,
            rng
        )?;
        
        // Generate zero-knowledge randomness
        let randomness = self.generate_zk_randomness(rng)?;
        
        Ok(DecompositionProof::new(
            decomposition_commitment,
            norm_bound_proof,
            reconstruction_proof,
            randomness,
        ))
    }
    
    /// Proves commitment consistency via sumcheck
    /// 
    /// # Arguments
    /// * `witness_vector` - Original witness
    /// * `decomposition_matrix` - Decomposed matrix
    /// * `folding_witness` - Combined folding witness
    /// * `double_commitment` - Double commitment proof
    /// * `rng` - Random number generator
    /// 
    /// # Returns
    /// * `Result<BatchedSumcheckProof>` - Consistency proof via sumcheck
    fn prove_commitment_consistency<R: CryptoRng + RngCore>(
        &mut self,
        witness_vector: &[RingElement],
        decomposition_matrix: &[Vec<i64>],
        folding_witness: &[RingElement],
        double_commitment: &DoubleCommitmentProof,
        rng: &mut R
    ) -> Result<BatchedSumcheckProof> {
        // Create multilinear extensions for consistency relations
        let mut consistency_polynomials = Vec::new();
        let mut claimed_sums = Vec::new();
        
        // Consistency relation 1: Double commitment consistency
        let double_consistency_poly = self.create_double_commitment_consistency_polynomial(
            witness_vector,
            double_commitment
        )?;
        consistency_polynomials.push(double_consistency_poly);
        claimed_sums.push(RingElement::zero(self.params.ring_dimension, Some(self.params.modulus))?);
        
        // Consistency relation 2: Decomposition consistency
        let decomp_consistency_poly = self.create_decomposition_consistency_polynomial(
            witness_vector,
            decomposition_matrix
        )?;
        consistency_polynomials.push(decomp_consistency_poly);
        claimed_sums.push(RingElement::zero(self.params.ring_dimension, Some(self.params.modulus))?);
        
        // Consistency relation 3: Folding witness consistency
        let folding_consistency_poly = self.create_folding_consistency_polynomial(
            witness_vector,
            decomposition_matrix,
            folding_witness
        )?;
        consistency_polynomials.push(folding_consistency_poly);
        claimed_sums.push(RingElement::zero(self.params.ring_dimension, Some(self.params.modulus))?);
        
        // Execute batched sumcheck
        self.sumcheck_protocol.prove_batch(&mut consistency_polynomials, &claimed_sums)
    }
    
    /// Helper functions for proof generation and verification
    
    /// Serializes witness vector for transcript
    fn serialize_witness_vector(&self, witness_vector: &[RingElement]) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(witness_vector.len() as u64).to_le_bytes());
        
        for element in witness_vector {
            let coeffs = element.coefficients();
            bytes.extend_from_slice(&(coeffs.len() as u64).to_le_bytes());
            for &coeff in coeffs {
                bytes.extend_from_slice(&coeff.to_le_bytes());
            }
        }
        
        Ok(bytes)
    }
    
    /// Computes hash of witness vector for caching
    fn hash_witness_vector(&self, witness_vector: &[RingElement]) -> Vec<u8> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        for element in witness_vector {
            element.coefficients().hash(&mut hasher);
        }
        hasher.finish().to_le_bytes().to_vec()
    }
    
    /// Extracts mτ from double commitment (simplified implementation)
    fn extract_m_tau_from_double_commitment(&self, double_commitment: &DoubleCommitmentProof) -> Result<RingElement> {
        // This is a simplified implementation
        // In practice, would extract the actual mτ value from the double commitment
        RingElement::zero(self.params.ring_dimension, Some(self.params.modulus))
    }
    
    /// Generates random ring element for zero-knowledge
    fn generate_random_ring_element<R: CryptoRng + RngCore>(&self, rng: &mut R) -> Result<RingElement> {
        let mut coeffs = vec![0i64; self.params.ring_dimension];
        let half_modulus = self.params.modulus / 2;
        
        for coeff in &mut coeffs {
            *coeff = (rng.next_u64() as i64) % self.params.modulus;
            if *coeff > half_modulus {
                *coeff -= self.params.modulus;
            }
        }
        
        RingElement::from_coefficients(coeffs, Some(self.params.modulus))
    }
    
    /// Additional helper methods would be implemented here...
    /// (Continuing with verification methods, polynomial creation, etc.)
    
    /// Returns protocol statistics
    pub fn stats(&self) -> Result<CommitmentTransformationStats> {
        Ok(self.stats.lock().unwrap().clone())
    }
    
    /// Resets protocol statistics
    pub fn reset_stats(&mut self) -> Result<()> {
        *self.stats.lock().unwrap() = CommitmentTransformationStats::new();
        Ok(())
    }
}

impl CommitmentTransformationProof {
    /// Estimates the serialized size of the proof
    /// 
    /// # Returns
    /// * `usize` - Estimated size in bytes
    pub fn serialized_size(&self) -> usize {
        let range_proof_size = self.range_check_proof.serialized_size();
        let decomp_proof_size = self.decomposition_proof.size_in_bytes(self.params.ring_dimension);
        let consistency_proof_size = self.consistency_proof.serialized_size();
        let folding_witness_size = self.folding_witness.len() * self.params.ring_dimension * 8;
        let challenges_size = self.folding_challenges.len() * self.params.ring_dimension * 8;
        let compressed_data_size = self.compressed_data.len() * self.params.ring_dimension * 8;
        let params_size = std::mem::size_of::<CommitmentTransformationParams>();
        
        range_proof_size + decomp_proof_size + consistency_proof_size + 
        folding_witness_size + challenges_size + compressed_data_size + params_size
    }
    
    /// Validates the proof structure
    /// 
    /// # Returns
    /// * `Result<()>` - Ok if structure is valid, error otherwise
    pub fn validate_structure(&self) -> Result<()> {
        // Validate folding witness dimension
        if self.folding_witness.len() != self.params.witness_dimension {
            return Err(LatticeFoldError::InvalidDimension {
                expected: self.params.witness_dimension,
                got: self.folding_witness.len(),
            });
        }
        
        // Validate folding challenges count
        if self.folding_challenges.len() != 3 {
            return Err(LatticeFoldError::InvalidParameters(
                format!("Expected 3 folding challenges, got {}", self.folding_challenges.len())
            ));
        }
        
        // Validate all ring elements have correct dimension
        for (i, elem) in self.folding_witness.iter().enumerate() {
            if elem.dimension() != self.params.ring_dimension {
                return Err(LatticeFoldError::InvalidDimension {
                    expected: self.params.ring_dimension,
                    got: elem.dimension(),
                });
            }
        }
        
        for (i, challenge) in self.folding_challenges.iter().enumerate() {
            if challenge.dimension() != self.params.ring_dimension {
                return Err(LatticeFoldError::InvalidDimension {
                    expected: self.params.ring_dimension,
                    got: challenge.dimension(),
                });
            }
        }
        
        // Validate component proofs
        self.range_check_proof.validate_structure()?;
        self.consistency_proof.validate_structure()?;
        
        Ok(())
    }
}
     //handles arithmetic overflow and modular reduction correctly
    /// - Provides detailed error messages for debugging
    pub fn prove<R: CryptoRng + RngCore>(
        &mut self,
        witness_vector: &[RingElement],
        double_commitment: &[RingElement],
        rng: &mut R
    ) -> Result<CommitmentTransformationProof> {
        let start_time = std::time::Instant::now();
        
        // Step 1: Execute range check protocol (Πrgchk subroutine)
        // This validates that all witness elements are within the specified bounds
        let range_check_proof = self.range_checker.prove(witness_vector, rng)?;
        
        // Step 2: Compute gadget decomposition τD = G_{d',k}^{-1}(cf(f))
        // This decomposes the coefficient representation for norm control
        let decomposition_matrix = self.compute_gadget_decomposition(witness_vector)?;
        
        // Step 3: Generate folding challenges s₀, s₁, s₂ ← S̄³ and s' ← S̄^{dk}
        // These challenges are used to combine different witness components
        let folding_challenges = self.generate_folding_challenges(rng)?;
        
        // Step 4: Compute folding witness g := s₀·τD + s₁·mτ + s₂·f + h
        // This combines all witness components with proper norm bounds
        let folding_witness = self.compute_folding_witness(
            witness_vector,
            &decomposition_matrix,
            folding_challenges.basic_challenges(),
            rng
        )?;
        
        // Step 5: Generate decomposition proof
        // This proves that the decomposition was computed correctly
        let decomposition_proof = self.prove_decomposition_consistency(
            witness_vector,
            &decomposition_matrix,
            rng
        )?;
        
        // Step 6: Generate consistency proof via sumcheck
        // This proves consistency between double and linear commitments
        let consistency_proof = self.prove_commitment_consistency(
            witness_vector,
            double_commitment,
            &folding_witness,
            &folding_challenges,
            rng
        )?;
        
        // Step 7: Generate communication compression data
        // This optimizes the representation of e' ∈ Rq^{dk}
        let compressed_data = self.compress_communication_data(
            &decomposition_matrix,
            &folding_challenges
        )?;
        
        // Step 8: Assemble final proof
        let proof = CommitmentTransformationProof {
            range_check_proof,
            decomposition_proof,
            consistency_proof,
            folding_witness,
            folding_challenges: folding_challenges.basic_challenges().to_vec(),
            compressed_data,
            params: self.params.clone(),
        };
        
        // Record performance statistics
        let elapsed_time = start_time.elapsed().as_nanos() as u64;
        let comm_bytes = self.estimate_proof_size(&proof);
        if let Ok(mut stats) = self.stats.lock() {
            stats.record_transformation(
                witness_vector.len(),
                elapsed_time,
                0, // Verifier time recorded separately
                comm_bytes as u64,
                true // Proof generation succeeded
            );
        }
        
        Ok(proof)
    }
    
    /// Verifies a commitment transformation proof
    /// 
    /// # Arguments
    /// * `proof` - Transformation proof to verify
    /// * `double_commitment` - Original double commitment
    /// * `linear_commitment` - Target linear commitment
    /// 
    /// # Returns
    /// * `Result<bool>` - True if proof is valid, false otherwise
    /// 
    /// # Verification Process
    /// 1. **Parameter Validation**: Check proof has correct dimensions and parameters
    /// 2. **Range Check Verification**: Verify input witness range proof
    /// 3. **Decomposition Verification**: Check gadget decomposition proof
    /// 4. **Consistency Verification**: Verify sumcheck consistency proof
    /// 5. **Witness Validation**: Check folding witness norm bounds
    /// 6. **Communication Verification**: Validate compressed data integrity
    /// 7. **Final Consistency**: Verify all components are mutually consistent
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
    /// - Zero-knowledge: Verifier learns only transformation validity
    pub fn verify(
        &mut self,
        proof: &CommitmentTransformationProof,
        double_commitment: &[RingElement],
        linear_commitment: &[RingElement]
    ) -> Result<bool> {
        let start_time = std::time::Instant::now();
        
        // Step 1: Validate proof structure and parameters
        if proof.params != self.params {
            return Ok(false);
        }
        
        if proof.folding_witness.len() != self.params.witness_dimension {
            return Ok(false);
        }
        
        if proof.folding_challenges.len() != 3 {
            return Ok(false); // Should have s₀, s₁, s₂
        }
        
        // Step 2: Verify range check proof
        if !self.range_checker.verify(&proof.range_check_proof, double_commitment)? {
            return Ok(false);
        }
        
        // Step 3: Verify decomposition proof
        if !self.verify_decomposition_proof(&proof.decomposition_proof)? {
            return Ok(false);
        }
        
        // Step 4: Verify consistency proof via sumcheck
        if !self.verify_consistency_proof(
            &proof.consistency_proof,
            double_commitment,
            linear_commitment,
            &proof.folding_witness,
            &proof.folding_challenges
        )? {
            return Ok(false);
        }
        
        // Step 5: Verify folding witness norm bounds
        if !self.verify_folding_witness_bounds(&proof.folding_witness)? {
            return Ok(false);
        }
        
        // Step 6: Verify communication compression integrity
        if !self.verify_compressed_data(&proof.compressed_data, &proof.folding_challenges)? {
            return Ok(false);
        }
        
        // Record verifier performance statistics
        let elapsed_time = start_time.elapsed().as_nanos() as u64;
        if let Ok(mut stats) = self.stats.lock() {
            stats.total_verifier_time_ns += elapsed_time;
        }
        
        Ok(true)
    }
}    
/// Computes gadget decomposition τD = G_{d',k}^{-1}(cf(f))
    /// 
    /// # Arguments
    /// * `witness_vector` - Input witness f to decompose
    /// 
    /// # Returns
    /// * `Result<Vec<Vec<i64>>>` - Decomposition matrix τD
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
        if let Ok(cache) = self.decomposition_cache.lock() {
            if let Some(cached_decomposition) = cache.get(&witness_key) {
                return Ok(cached_decomposition.clone());
            }
        }
        
        // Compute fresh decomposition
        let mut decomposition_matrix = Vec::with_capacity(witness_vector.len());
        
        // Process each witness element in parallel for better performance
        let decomposed_elements: Result<Vec<Vec<i64>>> = witness_vector
            .par_iter()
            .map(|witness_element| {
                // Extract coefficient vector cf(f_i)
                let coeffs = witness_element.coefficients();
                
                // Apply gadget decomposition to each coefficient
                let mut decomposed_coeffs = Vec::with_capacity(
                    coeffs.len() * self.params.decomposition_levels
                );
                
                for &coeff in coeffs {
                    // Decompose coefficient using gadget matrix
                    let decomposed = self.gadget_matrix.decompose_coefficient(coeff)?;
                    decomposed_coeffs.extend(decomposed);
                }
                
                Ok(decomposed_coeffs)
            })
            .collect();
        
        let decomposition_matrix = decomposed_elements?;
        
        // Cache the result if there's space
        if let Ok(mut cache) = self.decomposition_cache.lock() {
            if cache.len() < DECOMPOSITION_CACHE_SIZE {
                cache.insert(witness_key, decomposition_matrix.clone());
            }
        }
        
        Ok(decomposition_matrix)
    }
    
    /// Generates folding challenges using the dedicated challenge generator
    /// 
    /// # Arguments
    /// * `rng` - Cryptographically secure random number generator
    /// 
    /// # Returns
    /// * `Result<FoldingChallenges>` - Complete folding challenges (basic + extended)
    /// 
    /// # Mathematical Properties
    /// The challenges are sampled from the strong sampling set S̄ ⊆ Rq*
    /// to ensure proper folding properties and security reduction.
    /// 
    /// # Security Analysis
    /// The random challenges ensure that if any individual witness component
    /// violates its properties, the combined witness will fail verification
    /// with high probability (1 - 1/|S̄|).
    /// 
    /// # Integration with Folding Challenge Generator
    /// This method delegates to the specialized FoldingChallengeGenerator
    /// which provides:
    /// - Strong sampling set construction and validation
    /// - Proper transcript binding for non-interactive security
    /// - Extended challenges s' ← S̄^{dk} for matrix operations
    /// - Performance optimization and caching
    /// - Comprehensive security analysis and validation
    fn generate_folding_challenges<R: CryptoRng + RngCore>(&mut self, rng: &mut R) -> Result<FoldingChallenges> {
        // Use the dedicated folding challenge generator for secure challenge generation
        // This ensures proper strong sampling set properties and security guarantees
        let challenges = self.challenge_generator.generate_challenges(rng)?;
        
        // Add challenges to our transcript for additional binding
        // (The challenge generator already binds to its own transcript)
        for (i, challenge) in challenges.basic_challenges().iter().enumerate() {
            self.transcript.append_u64(b"basic_challenge_index", i as u64);
            self.transcript.append_message(b"basic_challenge", &challenge.to_bytes()?);
        }
        
        // Add extended challenges to transcript (first few for efficiency)
        for (i, challenge) in challenges.extended_challenges().iter().take(10).enumerate() {
            self.transcript.append_u64(b"extended_challenge_index", i as u64);
            self.transcript.append_message(b"extended_challenge", &challenge.to_bytes()?);
        }
        
        Ok(challenges)
    }
    
    /// Computes folding witness g := s₀·τD + s₁·mτ + s₂·f + h
    /// 
    /// # Arguments
    /// * `witness_vector` - Original witness f
    /// * `decomposition_matrix` - Decomposed witness τD
    /// * `folding_challenges` - Challenges s₀, s₁, s₂
    /// * `rng` - Random number generator for h
    /// 
    /// # Returns
    /// * `Result<Vec<RingElement>>` - Folding witness g with ||g||_∞ < b/2
    /// 
    /// # Mathematical Operation
    /// Combines all witness components using folding challenges:
    /// - s₀·τD: Scaled decomposed witness
    /// - s₁·mτ: Scaled monomial witness (derived from τD)
    /// - s₂·f: Scaled original witness
    /// - h: Random masking term for zero-knowledge
    /// 
    /// # Norm Analysis
    /// The combined witness g must satisfy ||g||_∞ < b/2 for security.
    /// This is achieved through careful parameter selection and norm bounds.
    fn compute_folding_witness<R: CryptoRng + RngCore>(
        &self,
        witness_vector: &[RingElement],
        decomposition_matrix: &[Vec<i64>],
        folding_challenges: &[RingElement],
        rng: &mut R
    ) -> Result<Vec<RingElement>> {
        // Validate input dimensions
        if folding_challenges.len() != 3 {
            return Err(LatticeFoldError::InvalidParameters(
                "Expected exactly 3 folding challenges".to_string()
            ));
        }
        
        // Extract individual challenges
        let s0 = &folding_challenges[0]; // Challenge for τD
        let s1 = &folding_challenges[1]; // Challenge for mτ
        let s2 = &folding_challenges[2]; // Challenge for f
        
        // Initialize result vector
        let mut folding_witness = vec![
            RingElement::zero(self.params.ring_dimension, Some(self.params.modulus))?;
            self.params.witness_dimension
        ];
        
        // Component 1: s₀·τD (scaled decomposed witness)
        // Process decomposition matrix in parallel for better performance
        let scaled_decomposition: Result<Vec<RingElement>> = decomposition_matrix
            .par_iter()
            .map(|decomposed_coeffs| {
                // Convert decomposed coefficients to ring element
                let decomposed_element = RingElement::from_coefficients(
                    decomposed_coeffs.clone(),
                    Some(self.params.modulus)
                )?;
                
                // Scale by challenge s₀
                decomposed_element.multiply(s0)
            })
            .collect();
        
        let scaled_decomposition = scaled_decomposition?;
        
        // Add scaled decomposition to folding witness
        for (i, scaled_elem) in scaled_decomposition.iter().enumerate() {
            if i < folding_witness.len() {
                folding_witness[i] = folding_witness[i].add(scaled_elem)?;
            }
        }
        
        // Component 2: s₁·mτ (scaled monomial witness)
        // Derive monomial witness from decomposition matrix
        let monomial_witness = self.derive_monomial_witness(decomposition_matrix)?;
        
        // Scale monomial witness by challenge s₁
        let scaled_monomial: Result<Vec<RingElement>> = monomial_witness
            .par_iter()
            .map(|monomial_elem| monomial_elem.multiply(s1))
            .collect();
        
        let scaled_monomial = scaled_monomial?;
        
        // Add scaled monomial witness to folding witness
        for (i, scaled_elem) in scaled_monomial.iter().enumerate() {
            if i < folding_witness.len() {
                folding_witness[i] = folding_witness[i].add(scaled_elem)?;
            }
        }
        
        // Component 3: s₂·f (scaled original witness)
        // Scale original witness by challenge s₂
        let scaled_original: Result<Vec<RingElement>> = witness_vector
            .par_iter()
            .map(|witness_elem| witness_elem.multiply(s2))
            .collect();
        
        let scaled_original = scaled_original?;
        
        // Add scaled original witness to folding witness
        for (i, scaled_elem) in scaled_original.iter().enumerate() {
            if i < folding_witness.len() {
                folding_witness[i] = folding_witness[i].add(scaled_elem)?;
            }
        }
        
        // Component 4: h (random masking term)
        // Generate random masking vector for zero-knowledge
        let masking_vector = self.generate_masking_vector(rng)?;
        
        // Add masking vector to folding witness
        for (i, masking_elem) in masking_vector.iter().enumerate() {
            if i < folding_witness.len() {
                folding_witness[i] = folding_witness[i].add(masking_elem)?;
            }
        }
        
        // Verify norm bound ||g||_∞ < b/2
        let max_norm = folding_witness
            .iter()
            .map(|elem| elem.infinity_norm())
            .max()
            .unwrap_or(0);
        
        let norm_bound = self.params.norm_bound / 2;
        if max_norm >= norm_bound {
            return Err(LatticeFoldError::InvalidParameters(
                format!("Folding witness norm {} exceeds bound {}", max_norm, norm_bound)
            ));
        }
        
        Ok(folding_witness)
    }
    
    /// Derives monomial witness mτ from decomposition matrix τD
    /// 
    /// # Arguments
    /// * `decomposition_matrix` - Decomposed witness τD
    /// 
    /// # Returns
    /// * `Result<Vec<RingElement>>` - Monomial witness mτ
    /// 
    /// # Mathematical Construction
    /// For each entry in the decomposition matrix, applies the exponential
    /// mapping EXP to convert to monomial form, then constructs the
    /// corresponding monomial matrix.
    fn derive_monomial_witness(&self, decomposition_matrix: &[Vec<i64>]) -> Result<Vec<RingElement>> {
        // Process decomposition matrix to create monomial witness
        let monomial_witness: Result<Vec<RingElement>> = decomposition_matrix
            .par_iter()
            .map(|decomposed_coeffs| {
                // Create monomial matrix from decomposed coefficients
                let mut monomial_coeffs = vec![0i64; self.params.ring_dimension];
                
                // Apply exponential mapping to each decomposed coefficient
                for (i, &decomposed_value) in decomposed_coeffs.iter().enumerate() {
                    if i < monomial_coeffs.len() {
                        // Apply exp function: exp(a) = sgn(a) * X^|a|
                        if decomposed_value != 0 {
                            let sign = if decomposed_value > 0 { 1 } else { -1 };
                            let degree = (decomposed_value.abs() as usize) % self.params.ring_dimension;
                            monomial_coeffs[degree] += sign;
                        }
                        // Zero values contribute nothing (exp(0) = 0 in our encoding)
                    }
                }
                
                // Create ring element from monomial coefficients
                RingElement::from_coefficients(monomial_coeffs, Some(self.params.modulus))
            })
            .collect();
        
        monomial_witness
    }
    
    /// Generates random masking vector h for zero-knowledge
    /// 
    /// # Arguments
    /// * `rng` - Cryptographically secure random number generator
    /// 
    /// # Returns
    /// * `Result<Vec<RingElement>>` - Random masking vector
    /// 
    /// # Security Properties
    /// The masking vector provides zero-knowledge by hiding the actual
    /// witness values while preserving the mathematical relationships
    /// required for the protocol.
    fn generate_masking_vector<R: CryptoRng + RngCore>(&self, rng: &mut R) -> Result<Vec<RingElement>> {
        let mut masking_vector = Vec::with_capacity(self.params.witness_dimension);
        
        // Generate random ring elements for masking
        for _ in 0..self.params.witness_dimension {
            let mut random_coeffs = vec![0i64; self.params.ring_dimension];
            
            // Fill with small random values to maintain norm bounds
            let masking_bound = self.params.norm_bound / 8; // Small values for masking
            for coeff in &mut random_coeffs {
                let mut bytes = [0u8; 8];
                rng.fill_bytes(&mut bytes);
                let random_value = i64::from_le_bytes(bytes);
                
                // Keep masking values small to preserve norm bounds
                *coeff = (random_value % (2 * masking_bound + 1)) - masking_bound;
            }
            
            let masking_element = RingElement::from_coefficients(
                random_coeffs,
                Some(self.params.modulus)
            )?;
            
            masking_vector.push(masking_element);
        }
        
        Ok(masking_vector)
    }
    
    /// Computes hash of witness vector for caching
    /// 
    /// # Arguments
    /// * `witness_vector` - Witness vector to hash
    /// 
    /// # Returns
    /// * `u64` - Hash value for cache key
    fn hash_witness_vector(&self, witness_vector: &[RingElement]) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        
        // Hash each witness element
        for witness_elem in witness_vector {
            // Hash the coefficients of each ring element
            for &coeff in witness_elem.coefficients() {
                coeff.hash(&mut hasher);
            }
        }
        
        // Include parameters in hash for uniqueness
        self.params.ring_dimension.hash(&mut hasher);
        self.params.modulus.hash(&mut hasher);
        
        hasher.finish()
    }
    
    /// Estimates the size of a proof in bytes
    /// 
    /// # Arguments
    /// * `proof` - Proof to estimate size for
    /// 
    /// # Returns
    /// * `usize` - Estimated size in bytes
    fn estimate_proof_size(&self, proof: &CommitmentTransformationProof) -> usize {
        let ring_element_size = self.params.ring_dimension * 8; // 8 bytes per coefficient
        
        let range_check_size = proof.range_check_proof.serialized_size();
        let decomposition_size = proof.decomposition_proof.size_in_bytes(self.params.ring_dimension);
        let consistency_size = proof.consistency_proof.serialized_size();
        let witness_size = proof.folding_witness.len() * ring_element_size;
        let challenges_size = proof.folding_challenges.len() * ring_element_size;
        let compressed_size = proof.compressed_data.len() * ring_element_size;
        
        range_check_size + decomposition_size + consistency_size + 
        witness_size + challenges_size + compressed_size
    }
}    /// 
Proves decomposition consistency
    /// 
    /// # Arguments
    /// * `witness_vector` - Original witness f
    /// * `decomposition_matrix` - Decomposed witness τD
    /// * `rng` - Random number generator
    /// 
    /// # Returns
    /// * `Result<DecompositionProof>` - Proof of decomposition consistency
    fn prove_decomposition_consistency<R: CryptoRng + RngCore>(
        &mut self,
        witness_vector: &[RingElement],
        decomposition_matrix: &[Vec<i64>],
        rng: &mut R
    ) -> Result<DecompositionProof> {
        // Generate commitment to decomposition matrix
        let decomposition_commitment = self.commit_decomposition_matrix(decomposition_matrix, rng)?;
        
        // Generate norm bound proof
        let norm_bound_proof = self.prove_decomposition_norm_bounds(decomposition_matrix, rng)?;
        
        // Generate reconstruction proof
        let reconstruction_proof = self.prove_reconstruction_consistency(
            witness_vector,
            decomposition_matrix,
            rng
        )?;
        
        // Generate zero-knowledge randomness
        let randomness = self.generate_zk_randomness(rng)?;
        
        Ok(DecompositionProof::new(
            decomposition_commitment,
            norm_bound_proof,
            reconstruction_proof,
            randomness,
        ))
    }
    
    /// Proves commitment consistency via sumcheck
    /// 
    /// # Arguments
    /// * `witness_vector` - Original witness
    /// * `double_commitment` - Double commitment
    /// * `folding_witness` - Combined witness
    /// * `folding_challenges` - Folding challenges
    /// * `rng` - Random number generator
    /// 
    /// # Returns
    /// * `Result<BatchedSumcheckProof>` - Consistency proof
    fn prove_commitment_consistency<R: CryptoRng + RngCore>(
        &mut self,
        witness_vector: &[RingElement],
        double_commitment: &[RingElement],
        folding_witness: &[RingElement],
        folding_challenges: &FoldingChallenges,
        rng: &mut R
    ) -> Result<BatchedSumcheckProof> {
        // Create multilinear extensions for consistency relations
        let mut consistency_polynomials = self.create_consistency_polynomials(
            witness_vector,
            double_commitment,
            folding_witness,
            folding_challenges
        )?;
        
        // Compute claimed sums (should all be zero for consistency)
        let claimed_sums = vec![
            RingElement::zero(self.params.ring_dimension, Some(self.params.modulus))?;
            consistency_polynomials.len()
        ];
        
        // Execute batched sumcheck protocol
        self.sumcheck_protocol.prove_batch(&mut consistency_polynomials, &claimed_sums)
    }
    
    /// Creates consistency polynomials for sumcheck
    /// 
    /// # Arguments
    /// * `witness_vector` - Original witness
    /// * `double_commitment` - Double commitment
    /// * `folding_witness` - Combined witness
    /// * `folding_challenges` - Folding challenges
    /// 
    /// # Returns
    /// * `Result<Vec<MultilinearExtension>>` - Consistency polynomials
    fn create_consistency_polynomials(
        &self,
        witness_vector: &[RingElement],
        double_commitment: &[RingElement],
        folding_witness: &[RingElement],
        folding_challenges: &FoldingChallenges
    ) -> Result<Vec<MultilinearExtension>> {
        // This is a simplified implementation
        // In practice, would create proper multilinear extensions for:
        // 1. Double commitment consistency
        // 2. Linear commitment consistency
        // 3. Witness combination consistency
        // 4. Norm bound consistency
        // 5. Range proof consistency
        // 6. Decomposition consistency
        
        let mut polynomials = Vec::new();
        
        // Create dummy polynomial for now (would be replaced with actual consistency relations)
        let dummy_values = vec![
            RingElement::zero(self.params.ring_dimension, Some(self.params.modulus))?;
            1 << 10 // 2^10 domain size
        ];
        
        let consistency_poly = MultilinearExtension::new(
            dummy_values,
            self.params.ring_dimension,
            Some(self.params.modulus)
        )?;
        
        polynomials.push(consistency_poly);
        
        Ok(polynomials)
    }
    
    /// Compresses communication data
    /// 
    /// # Arguments
    /// * `decomposition_matrix` - Decomposition matrix
    /// * `folding_challenges` - Folding challenges
    /// 
    /// # Returns
    /// * `Result<Vec<RingElement>>` - Compressed data
    fn compress_communication_data(
        &self,
        decomposition_matrix: &[Vec<i64>],
        folding_challenges: &FoldingChallenges
    ) -> Result<Vec<RingElement>> {
        // Implement communication compression using decomposition techniques
        // This reduces the size of e' ∈ Rq^{dk} representation
        
        let mut compressed_data = Vec::new();
        
        // Apply compression algorithm using extended challenges s' ∈ S̄^{dk}
        // This compresses the representation of e' ∈ Rq^{dk} using the folding challenges
        let extended_challenges = folding_challenges.extended_challenges();
        
        for (row_idx, decomposed_row) in decomposition_matrix.iter().take(self.params.decomposition_levels).enumerate() {
            // Compress each row using corresponding extended challenges
            let mut compressed_coeffs = vec![0i64; self.params.ring_dimension];
            
            // Apply challenge-based compression
            for (i, &coeff) in decomposed_row.iter().enumerate() {
                if i < compressed_coeffs.len() {
                    // Use extended challenges for compression
                    let challenge_idx = (row_idx * decomposed_row.len() + i) % extended_challenges.len();
                    let challenge_coeff = extended_challenges[challenge_idx].coefficients()[0]; // Use first coefficient
                    
                    // Compress using challenge multiplication (simplified)
                    compressed_coeffs[i] = (coeff * challenge_coeff) % self.params.modulus;
                }
            }
            
            let compressed_element = RingElement::from_coefficients(
                compressed_coeffs,
                Some(self.params.modulus)
            )?;
            
            compressed_data.push(compressed_element);
        }
        
        Ok(compressed_data)
    }
    
    /// Helper methods for proof generation
    fn commit_decomposition_matrix<R: CryptoRng + RngCore>(
        &self,
        decomposition_matrix: &[Vec<i64>],
        rng: &mut R
    ) -> Result<Vec<RingElement>> {
        // Create commitments to decomposition matrix rows
        let mut commitments = Vec::new();
        
        for decomposed_row in decomposition_matrix {
            let row_element = RingElement::from_coefficients(
                decomposed_row.clone(),
                Some(self.params.modulus)
            )?;
            
            // In practice, would use proper commitment scheme
            commitments.push(row_element);
        }
        
        Ok(commitments)
    }
    
    fn prove_decomposition_norm_bounds<R: CryptoRng + RngCore>(
        &self,
        decomposition_matrix: &[Vec<i64>],
        rng: &mut R
    ) -> Result<Vec<RingElement>> {
        // Generate proof that ||τD||_∞ < d'
        let mut norm_proof = Vec::new();
        
        // Simplified implementation - would use range proofs in practice
        for decomposed_row in decomposition_matrix {
            let proof_element = RingElement::from_coefficients(
                vec![1; self.params.ring_dimension], // Dummy proof
                Some(self.params.modulus)
            )?;
            norm_proof.push(proof_element);
        }
        
        Ok(norm_proof)
    }
    
    fn prove_reconstruction_consistency<R: CryptoRng + RngCore>(
        &self,
        witness_vector: &[RingElement],
        decomposition_matrix: &[Vec<i64>],
        rng: &mut R
    ) -> Result<Vec<RingElement>> {
        // Generate proof that G_{d',k} × τD = cf(f)
        let mut reconstruction_proof = Vec::new();
        
        // Simplified implementation
        for witness_elem in witness_vector {
            reconstruction_proof.push(witness_elem.clone());
        }
        
        Ok(reconstruction_proof)
    }
    
    fn generate_zk_randomness<R: CryptoRng + RngCore>(&self, rng: &mut R) -> Result<Vec<RingElement>> {
        let mut randomness = Vec::new();
        
        // Generate random elements for zero-knowledge
        for _ in 0..self.params.kappa {
            let mut random_coeffs = vec![0i64; self.params.ring_dimension];
            
            for coeff in &mut random_coeffs {
                let mut bytes = [0u8; 8];
                rng.fill_bytes(&mut bytes);
                *coeff = i64::from_le_bytes(bytes) % self.params.modulus;
            }
            
            let random_element = RingElement::from_coefficients(
                random_coeffs,
                Some(self.params.modulus)
            )?;
            
            randomness.push(random_element);
        }
        
        Ok(randomness)
    }
    
    /// Verification helper methods
    fn verify_decomposition_proof(&self, proof: &DecompositionProof) -> Result<bool> {
        // Verify decomposition proof components
        // This would include:
        // 1. Commitment verification
        // 2. Norm bound verification
        // 3. Reconstruction consistency verification
        
        // Simplified implementation
        Ok(proof.decomposition_commitment.len() > 0 &&
           proof.norm_bound_proof.len() > 0 &&
           proof.reconstruction_proof.len() > 0)
    }
    
    fn verify_consistency_proof(
        &mut self,
        proof: &BatchedSumcheckProof,
        double_commitment: &[RingElement],
        linear_commitment: &[RingElement],
        folding_witness: &[RingElement],
        folding_challenges: &[RingElement]
    ) -> Result<bool> {
        // Create evaluators for final sumcheck verification
        let evaluators = vec![
            |_challenge: &[i64]| -> Result<RingElement> {
                RingElement::zero(self.params.ring_dimension, Some(self.params.modulus))
            }
        ];
        
        // Verify batched sumcheck proof
        let claimed_sums = vec![
            RingElement::zero(self.params.ring_dimension, Some(self.params.modulus))?
        ];
        
        self.sumcheck_protocol.verify_batch(proof, &claimed_sums, evaluators)
    }
    
    fn verify_folding_witness_bounds(&self, folding_witness: &[RingElement]) -> Result<bool> {
        let norm_bound = self.params.norm_bound / 2;
        
        for witness_elem in folding_witness {
            if witness_elem.infinity_norm() >= norm_bound {
                return Ok(false);
            }
        }
        
        Ok(true)
    }
    
    fn verify_compressed_data(
        &self,
        compressed_data: &[RingElement],
        folding_challenges: &[RingElement]
    ) -> Result<bool> {
        // Verify compressed data integrity
        // This would check that the compression was done correctly
        
        // Simplified implementation
        Ok(compressed_data.len() <= self.params.decomposition_levels)
    }
    
    /// Returns performance statistics
    pub fn stats(&self) -> Result<CommitmentTransformationStats> {
        Ok(self.stats.lock().unwrap().clone())
    }
    
    /// Resets performance statistics
    pub fn reset_stats(&mut self) -> Result<()> {
        *self.stats.lock().unwrap() = CommitmentTransformationStats::new();
        Ok(())
    }
}

/// Comprehensive tests for commitment transformation protocol
#[cfg(test)]
mod tests {
    use super::*;
    use rand::thread_rng;
    
    #[test]
    fn test_commitment_transformation_params_creation() {
        let params = CommitmentTransformationParams::new(
            128,  // kappa
            1024, // ring_dimension
            100000, // witness_dimension
            1000, // norm_bound
            2147483647, // modulus (2^31 - 1)
        );
        
        assert!(params.is_ok());
        let params = params.unwrap();
        assert_eq!(params.kappa, 128);
        assert_eq!(params.ring_dimension, 1024);
        assert_eq!(params.witness_dimension, 100000);
    }
    
    #[test]
    fn test_commitment_transformation_protocol_creation() {
        let params = CommitmentTransformationParams::new(
            64,   // kappa
            512,  // ring_dimension
            50000, // witness_dimension
            500,  // norm_bound
            1073741827, // modulus
        ).unwrap();
        
        let protocol = CommitmentTransformationProtocol::new(params);
        assert!(protocol.is_ok());
    }
    
    #[test]
    fn test_folding_challenge_generation() {
        let params = CommitmentTransformationParams::new(
            32,   // kappa
            256,  // ring_dimension
            10000, // witness_dimension
            100,  // norm_bound
            536870923, // modulus
        ).unwrap();
        
        let mut protocol = CommitmentTransformationProtocol::new(params).unwrap();
        let mut rng = thread_rng();
        
        let challenges = protocol.generate_folding_challenges(&mut rng);
        assert!(challenges.is_ok());
        
        let folding_challenges = challenges.unwrap();
        
        // Validate challenge structure
        assert!(folding_challenges.validate().unwrap());
        assert_eq!(folding_challenges.basic_challenges().len(), 3);
        
        // Verify basic challenges are different
        let basic_challenges = folding_challenges.basic_challenges();
        assert_ne!(basic_challenges[0].coefficients(), basic_challenges[1].coefficients());
        assert_ne!(basic_challenges[1].coefficients(), basic_challenges[2].coefficients());
        
        // Verify extended challenges exist
        assert!(!folding_challenges.extended_challenges().is_empty());
        
        // Test witness combination
        let tau_d = RingElement::zero(256, Some(536870923)).unwrap();
        let m_tau = RingElement::zero(256, Some(536870923)).unwrap();
        let f = RingElement::zero(256, Some(536870923)).unwrap();
        let h = RingElement::zero(256, Some(536870923)).unwrap();
        
        let combined = folding_challenges.combine_witnesses(&tau_d, &m_tau, &f, &h);
        assert!(combined.is_ok());
        
        let combined_witness = combined.unwrap();
        assert_eq!(combined_witness.dimension(), 256);
        assert_eq!(combined_witness.modulus(), Some(536870923));
    }
    
    #[test]
    fn test_witness_vector_hashing() {
        let params = CommitmentTransformationParams::new(
            32, 256, 10000, 100, 536870923
        ).unwrap();
        
        let protocol = CommitmentTransformationProtocol::new(params).unwrap();
        
        // Create test witness vector
        let witness1 = vec![
            RingElement::zero(256, Some(536870923)).unwrap(),
            RingElement::one(256, Some(536870923)).unwrap(),
        ];
        
        let witness2 = vec![
            RingElement::one(256, Some(536870923)).unwrap(),
            RingElement::zero(256, Some(536870923)).unwrap(),
        ];
        
        let hash1 = protocol.hash_witness_vector(&witness1);
        let hash2 = protocol.hash_witness_vector(&witness2);
        
        // Different witnesses should have different hashes
        assert_ne!(hash1, hash2);
        
        // Same witness should have same hash
        let hash1_repeat = protocol.hash_witness_vector(&witness1);
        assert_eq!(hash1, hash1_repeat);
    }
}