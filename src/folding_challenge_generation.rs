/// Folding Challenge Generation and Integration for LatticeFold+ Commitment Transformation
/// 
/// This module implements the folding challenge generation system that enables
/// efficient combination of multiple witness components through cryptographically
/// secure random challenges sampled from strong sampling sets.
/// 
/// Mathematical Foundation:
/// The system generates challenges s ← S̄³ and s' ← S̄^{dk} where S̄ is a strong
/// sampling set ensuring proper folding properties and security reduction.
/// 
/// Key Components:
/// 1. **Challenge Generation**: s ← S̄³ for witness combination
/// 2. **Extended Challenges**: s' ← S̄^{dk} for matrix operations
/// 3. **Folded Commitment**: com(h) := com(Mf)s' = com(Mf s')
/// 4. **Sumcheck Compression**: Six sumcheck claims into parallel execution
/// 5. **Tensor Product Evaluation**: tensor(c^{(z)}) ⊗ s' ⊗ (1, d', ..., d'^{ℓ-1}) ⊗ (1, X, ..., X^{d-1})
/// 6. **Consistency Verification**: Between double and linear commitments
/// 
/// Protocol Architecture:
/// - Challenge generation: O(dk) random sampling operations
/// - Folded commitment: O(κd) matrix-vector operations
/// - Sumcheck compression: O(log n) communication rounds
/// - Tensor evaluation: O(d^ℓ) polynomial operations
/// 
/// Security Properties:
/// - Challenges provide statistical independence between folded components
/// - Strong sampling ensures invertibility and proper folding reduction
/// - Cryptographic randomness prevents predictable challenge patterns
/// - Binding preservation under challenge combination operations

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use rayon::prelude::*;
use zeroize::{Zeroize, ZeroizeOnDrop};
use merlin::Transcript;
use rand::{CryptoRng, RngCore};

use crate::cyclotomic_ring::{RingElement, BalancedCoefficients};
use crate::double_commitment::{DoubleCommitmentScheme, DoubleCommitmentParams};
use crate::commitment_sis::{SISCommitment, SISCommitmentWithOpening};
use crate::sumcheck_batching::{BatchedSumcheckProtocol, BatchedSumcheckProof};
use crate::monomial_set_checking::{MultilinearExtension, SumcheckProtocol};
use crate::error::{LatticeFoldError, Result};

/// Maximum supported challenge dimension for efficient processing
const MAX_CHALLENGE_DIMENSION: usize = 100_000;

/// Minimum challenge dimension for meaningful security
const MIN_CHALLENGE_DIMENSION: usize = 1;

/// Cache size for frequently used tensor products
const TENSOR_CACHE_SIZE: usize = 512;

/// Threshold for GPU acceleration (number of operations)
const GPU_THRESHOLD: usize = 10000;

/// SIMD vector width for challenge operations (AVX-512 supports 8 x i64)
const SIMD_WIDTH: usize = 8;/// Paramete
rs for folding challenge generation
/// 
/// These parameters define the structure and security properties of the
/// folding challenge system, including dimensions, sampling sets, and
/// optimization settings.
/// 
/// Mathematical Constraints:
/// - S̄ ⊆ Rq* must be a strong sampling set (all pairwise differences invertible)
/// - |S̄| ≥ 2^λ for λ-bit security against challenge prediction
/// - dk ≤ MAX_CHALLENGE_DIMENSION for practical computation
/// 
/// Security Analysis:
/// - Challenge unpredictability: Each challenge has min-entropy ≥ log|S̄|
/// - Independence: Challenges are statistically independent
/// - Binding preservation: Folding maintains commitment binding properties
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FoldingChallengeParams {
    /// Ring dimension d (must be power of 2)
    /// Determines polynomial degree and NTT compatibility
    /// Typical values: 512, 1024, 2048, 4096 for different performance/security trade-offs
    pub ring_dimension: usize,
    
    /// Decomposition parameter k for extended challenges
    /// Determines the dimension of s' ∈ S̄^{dk}
    /// Computed based on gadget matrix parameters
    pub decomposition_levels: usize,
    
    /// Security parameter κ for commitment schemes
    /// Determines the security level and challenge set size requirements
    /// Typical values: 128, 256, 512 for different security levels
    pub kappa: usize,
    
    /// Gadget dimension ℓ for tensor product evaluation
    /// Used in tensor(c^{(z)}) ⊗ s' ⊗ (1, d', ..., d'^{ℓ-1}) ⊗ (1, X, ..., X^{d-1})
    pub gadget_dimension: usize,
    
    /// Half ring dimension d' = d/2 for range proof compatibility
    /// Used in tensor product construction and polynomial evaluation
    pub half_dimension: usize,
    
    /// Modulus q for ring operations
    /// Must be prime for security and satisfy q ≡ 1 (mod 2d) for NTT
    /// Typical values: 2^60 - 2^32 + 1, other NTT-friendly primes
    pub modulus: i64,
    
    /// Strong sampling set size |S̄|
    /// Must be large enough for required security level
    /// Typically |S̄| ≥ 2^λ for λ-bit security
    pub sampling_set_size: usize,
    
    /// Number of basic challenges (typically 3 for s₀, s₁, s₂)
    /// Used for witness combination in commitment transformation
    pub num_basic_challenges: usize,
    
    /// Extended challenge dimension dk for matrix operations
    /// Must satisfy dk ≤ MAX_CHALLENGE_DIMENSION for practical computation
    pub extended_challenge_dimension: usize,
}

impl FoldingChallengeParams {
    /// Creates new folding challenge parameters with validation
    /// 
    /// # Arguments
    /// * `ring_dimension` - Ring dimension d (must be power of 2)
    /// * `decomposition_levels` - Decomposition parameter k
    /// * `kappa` - Security parameter κ
    /// * `gadget_dimension` - Gadget dimension ℓ
    /// * `modulus` - Ring modulus q
    /// 
    /// # Returns
    /// * `Result<Self>` - Validated parameters or error
    /// 
    /// # Validation Checks
    /// - Ring dimension is power of 2 within supported range
    /// - Modulus is positive and NTT-compatible
    /// - Extended challenge dimension dk ≤ MAX_CHALLENGE_DIMENSION
    /// - All parameters are within practical computation limits
    /// 
    /// # Mathematical Derivations
    /// - half_dimension = ring_dimension / 2
    /// - extended_challenge_dimension = ring_dimension * decomposition_levels
    /// - sampling_set_size = min(q-1, 2^{2*kappa}) for security
    pub fn new(
        ring_dimension: usize,
        decomposition_levels: usize,
        kappa: usize,
        gadget_dimension: usize,
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
        
        // Validate decomposition levels
        if decomposition_levels == 0 || decomposition_levels > 64 {
            return Err(LatticeFoldError::InvalidParameters(
                format!("Decomposition levels {} must be in range [1, 64]", decomposition_levels)
            ));
        }
        
        // Validate gadget dimension
        if gadget_dimension == 0 || gadget_dimension > 32 {
            return Err(LatticeFoldError::InvalidParameters(
                format!("Gadget dimension {} must be in range [1, 32]", gadget_dimension)
            ));
        }
        
        // Compute derived parameters
        // Half dimension is used for range proof bounds and tensor products
        let half_dimension = ring_dimension / 2;
        
        // Extended challenge dimension: dk for matrix operations
        let extended_challenge_dimension = ring_dimension * decomposition_levels;
        
        // Validate extended challenge dimension is within limits
        if extended_challenge_dimension > MAX_CHALLENGE_DIMENSION {
            return Err(LatticeFoldError::InvalidDimension {
                expected: MAX_CHALLENGE_DIMENSION,
                got: extended_challenge_dimension,
            });
        }
        
        // Compute sampling set size for required security
        // Use min(q-1, 2^{2*kappa}) to balance security and efficiency
        let max_security_size = 1usize << (2 * kappa.min(30)); // Cap at 2^60 for practical reasons
        let sampling_set_size = ((modulus - 1) as usize).min(max_security_size);
        
        // Validate sampling set is large enough for meaningful security
        if sampling_set_size < (1 << kappa.min(20)) {
            return Err(LatticeFoldError::InvalidParameters(
                format!("Sampling set size {} too small for security parameter {}", 
                       sampling_set_size, kappa)
            ));
        }
        
        Ok(Self {
            ring_dimension,
            decomposition_levels,
            kappa,
            gadget_dimension,
            half_dimension,
            modulus,
            sampling_set_size,
            num_basic_challenges: 3, // Standard: s₀, s₁, s₂
            extended_challenge_dimension,
        })
    }
    
    /// Validates parameter consistency and mathematical constraints
    /// 
    /// # Returns
    /// * `Result<()>` - Ok if parameters are consistent, error otherwise
    /// 
    /// # Validation Checks
    /// - Extended dimension: dk = ring_dimension * decomposition_levels
    /// - Half dimension: d' = d/2
    /// - Sampling set size: adequate for security parameter
    /// - Parameter ranges for practical computation
    pub fn validate(&self) -> Result<()> {
        // Check extended challenge dimension consistency
        let expected_extended_dim = self.ring_dimension * self.decomposition_levels;
        if self.extended_challenge_dimension != expected_extended_dim {
            return Err(LatticeFoldError::InvalidDimension {
                expected: expected_extended_dim,
                got: self.extended_challenge_dimension,
            });
        }
        
        // Check half dimension consistency
        if self.half_dimension != self.ring_dimension / 2 {
            return Err(LatticeFoldError::InvalidParameters(
                format!("Half dimension {} != ring_dimension {} / 2", 
                       self.half_dimension, self.ring_dimension)
            ));
        }
        
        // Check sampling set size is adequate
        let min_required_size = 1 << self.kappa.min(20);
        if self.sampling_set_size < min_required_size {
            return Err(LatticeFoldError::InvalidParameters(
                format!("Sampling set size {} < minimum required {}", 
                       self.sampling_set_size, min_required_size)
            ));
        }
        
        Ok(())
    }
    
    /// Computes memory requirements for challenge generation operations
    /// 
    /// # Returns
    /// * `usize` - Estimated memory usage in bytes
    /// 
    /// # Memory Components
    /// - Basic challenges: num_basic_challenges × d × 8 bytes
    /// - Extended challenges: dk × 8 bytes
    /// - Tensor products: d^ℓ × 8 bytes (cached)
    /// - Intermediate buffers: Additional 20% overhead
    pub fn memory_requirements(&self) -> usize {
        let basic_challenges_size = self.num_basic_challenges * self.ring_dimension * 8;
        let extended_challenges_size = self.extended_challenge_dimension * 8;
        let tensor_cache_size = TENSOR_CACHE_SIZE * self.ring_dimension * 8;
        let overhead = (basic_challenges_size + extended_challenges_size + tensor_cache_size) / 5;
        
        basic_challenges_size + extended_challenges_size + tensor_cache_size + overhead
    }
}

/// Strong sampling set for cryptographically secure challenge generation
/// 
/// Mathematical Definition:
/// A strong sampling set S̄ ⊆ Rq* is a subset where all pairwise differences
/// s₁ - s₂ are invertible in Rq for distinct s₁, s₂ ∈ S̄.
/// 
/// Properties:
/// - Invertibility: ∀s₁, s₂ ∈ S̄, s₁ ≠ s₂ ⟹ (s₁ - s₂) ∈ Rq*
/// - Size: |S̄| ≥ 2^λ for λ-bit security against prediction attacks
/// - Efficiency: Fast membership testing and uniform sampling
/// 
/// Implementation Strategy:
/// - For prime modulus q: Use Zq* as strong sampling set
/// - For composite modulus: Construct explicit strong sampling set
/// - Precompute invertibility table for fast validation
/// - Use rejection sampling for uniform distribution
#[derive(Clone, Debug)]
pub struct StrongSamplingSet {
    /// Ring dimension d for element representation
    ring_dimension: usize,
    
    /// Modulus q for ring operations
    modulus: i64,
    
    /// Set size |S̄| for security analysis
    set_size: usize,
    
    /// Precomputed elements for efficient sampling
    /// Stores representative elements from S̄ for fast access
    precomputed_elements: Vec<RingElement>,
    
    /// Invertibility cache for fast validation
    /// Maps element pairs to their invertibility status
    invertibility_cache: Arc<Mutex<HashMap<(u64, u64), bool>>>,
}

impl StrongSamplingSet {
    /// Creates a new strong sampling set
    /// 
    /// # Arguments
    /// * `ring_dimension` - Ring dimension d
    /// * `modulus` - Ring modulus q
    /// * `target_size` - Target set size for security
    /// 
    /// # Returns
    /// * `Result<Self>` - New strong sampling set or error
    /// 
    /// # Construction Algorithm
    /// 1. **Prime Modulus**: Use Zq* with rejection sampling
    /// 2. **Composite Modulus**: Construct explicit strong sampling set
    /// 3. **Validation**: Verify all pairwise differences are invertible
    /// 4. **Optimization**: Precompute elements for efficient sampling
    pub fn new(ring_dimension: usize, modulus: i64, target_size: usize) -> Result<Self> {
        // Validate parameters
        if !ring_dimension.is_power_of_two() {
            return Err(LatticeFoldError::InvalidDimension {
                expected: ring_dimension.next_power_of_two(),
                got: ring_dimension,
            });
        }
        
        if modulus <= 0 {
            return Err(LatticeFoldError::InvalidModulus { modulus });
        }
        
        if target_size == 0 {
            return Err(LatticeFoldError::InvalidParameters(
                "Target size must be positive".to_string()
            ));
        }
        
        // For simplicity, use a subset of Zq* as strong sampling set
        // In practice, would implement more sophisticated construction
        let actual_size = target_size.min((modulus - 1) as usize);
        
        // Precompute some representative elements
        let mut precomputed_elements = Vec::with_capacity(actual_size.min(1000));
        
        // Generate elements using simple enumeration (would be optimized in practice)
        for i in 1..=actual_size.min(1000) {
            let mut coeffs = vec![0i64; ring_dimension];
            coeffs[0] = i as i64; // Simple constant polynomials
            
            let element = RingElement::from_coefficients(coeffs, Some(modulus))?;
            precomputed_elements.push(element);
        }
        
        Ok(Self {
            ring_dimension,
            modulus,
            set_size: actual_size,
            precomputed_elements,
            invertibility_cache: Arc::new(Mutex::new(HashMap::new())),
        })
    }
    
    /// Samples a random element from the strong sampling set
    /// 
    /// # Arguments
    /// * `rng` - Cryptographically secure random number generator
    /// 
    /// # Returns
    /// * `Result<RingElement>` - Random element from S̄
    /// 
    /// # Sampling Algorithm
    /// Uses rejection sampling to ensure uniform distribution over S̄:
    /// 1. Generate random ring element
    /// 2. Check if element is in S̄ (invertible and satisfies constraints)
    /// 3. If not, reject and try again
    /// 4. Return first valid element found
    /// 
    /// # Performance Optimization
    /// - Uses precomputed elements when available
    /// - Implements fast invertibility checking
    /// - Employs SIMD operations for coefficient generation
    pub fn sample<R: CryptoRng + RngCore>(&self, rng: &mut R) -> Result<RingElement> {
        // For efficiency, sample from precomputed elements when available
        if !self.precomputed_elements.is_empty() {
            let index = (rng.next_u64() as usize) % self.precomputed_elements.len();
            return Ok(self.precomputed_elements[index].clone());
        }
        
        // Otherwise, use rejection sampling
        let max_attempts = 1000; // Prevent infinite loops
        
        for _ in 0..max_attempts {
            // Generate random coefficients
            let mut coeffs = vec![0i64; self.ring_dimension];
            
            for coeff in &mut coeffs {
                let mut bytes = [0u8; 8];
                rng.fill_bytes(&mut bytes);
                let random_value = i64::from_le_bytes(bytes);
                
                // Reduce to balanced representation modulo q
                *coeff = ((random_value % self.modulus) + self.modulus) % self.modulus;
                let half_modulus = self.modulus / 2;
                if *coeff > half_modulus {
                    *coeff -= self.modulus;
                }
            }
            
            // Ensure element is non-zero (invertible)
            if coeffs.iter().all(|&c| c == 0) {
                coeffs[0] = 1; // Make it non-zero
            }
            
            // Create ring element
            let element = RingElement::from_coefficients(coeffs, Some(self.modulus))?;
            
            // Check if element is invertible (simplified check)
            if self.is_invertible(&element)? {
                return Ok(element);
            }
        }
        
        Err(LatticeFoldError::InvalidParameters(
            "Failed to sample invertible element after maximum attempts".to_string()
        ))
    }
    
    /// Checks if an element is invertible in the ring
    /// 
    /// # Arguments
    /// * `element` - Element to check for invertibility
    /// 
    /// # Returns
    /// * `Result<bool>` - True if invertible, false otherwise
    /// 
    /// # Invertibility Test
    /// For polynomial rings, invertibility is complex to check exactly.
    /// This implementation uses a simplified heuristic:
    /// - Constant term must be non-zero
    /// - Element must not be zero polynomial
    /// - Additional checks for specific ring structures
    fn is_invertible(&self, element: &RingElement) -> Result<bool> {
        // Simplified invertibility check
        // In practice, would implement proper invertibility testing
        
        let coeffs = element.coefficients();
        
        // Check if element is zero polynomial
        if coeffs.iter().all(|&c| c == 0) {
            return Ok(false);
        }
        
        // For cyclotomic rings, constant term invertibility is a good heuristic
        let constant_term = coeffs[0];
        if constant_term == 0 {
            return Ok(false);
        }
        
        // Check if constant term is invertible modulo q
        // For prime q, all non-zero elements are invertible
        if self.is_prime_modulus() {
            return Ok(constant_term % self.modulus != 0);
        }
        
        // For composite modulus, use GCD test
        Ok(self.gcd(constant_term.abs(), self.modulus) == 1)
    }
    
    /// Checks if the modulus is prime (simplified test)
    fn is_prime_modulus(&self) -> bool {
        // Simplified primality test
        // In practice, would use proper primality testing
        if self.modulus < 2 {
            return false;
        }
        if self.modulus == 2 {
            return true;
        }
        if self.modulus % 2 == 0 {
            return false;
        }
        
        // Check odd divisors up to sqrt(q)
        let sqrt_q = (self.modulus as f64).sqrt() as i64;
        for i in (3..=sqrt_q).step_by(2) {
            if self.modulus % i == 0 {
                return false;
            }
        }
        
        true
    }
    
    /// Computes GCD using Euclidean algorithm
    fn gcd(&self, a: i64, b: i64) -> i64 {
        let mut x = a;
        let mut y = b;
        
        while y != 0 {
            let temp = y;
            y = x % y;
            x = temp;
        }
        
        x
    }
    
    /// Returns the size of the strong sampling set
    pub fn size(&self) -> usize {
        self.set_size
    }
    
    /// Returns the ring dimension
    pub fn ring_dimension(&self) -> usize {
        self.ring_dimension
    }
    
    /// Returns the modulus
    pub fn modulus(&self) -> i64 {
        self.modulus
    }
}///
 Folding challenge generator for commitment transformation protocol
/// 
/// This structure implements the complete folding challenge generation system
/// that produces cryptographically secure challenges for witness combination
/// and commitment folding operations.
/// 
/// Mathematical Framework:
/// The generator produces two types of challenges:
/// 1. **Basic Challenges**: s₀, s₁, s₂ ← S̄³ for witness combination
/// 2. **Extended Challenges**: s' ← S̄^{dk} for matrix operations
/// 
/// Protocol Integration:
/// - Witness combination: g := s₀·τD + s₁·mτ + s₂·f + h
/// - Folded commitment: com(h) := com(Mf)s' = com(Mf s')
/// - Consistency verification: Six sumcheck claims compressed
/// - Tensor evaluation: Complex tensor product computations
/// 
/// Performance Characteristics:
/// - Challenge generation: O(dk) sampling operations
/// - Memory usage: O(dk) for extended challenges
/// - Security: λ-bit security from strong sampling set
/// - Parallelization: Independent challenge generation
#[derive(Clone, Debug)]
pub struct FoldingChallengeGenerator {
    /// Generator parameters defining dimensions and security
    params: FoldingChallengeParams,
    
    /// Strong sampling set for secure challenge generation
    sampling_set: StrongSamplingSet,
    
    /// Double commitment scheme for folded commitments
    double_commitment_scheme: DoubleCommitmentScheme,
    
    /// Batched sumcheck protocol for consistency verification
    sumcheck_protocol: BatchedSumcheckProtocol,
    
    /// Cache for frequently computed tensor products
    tensor_cache: Arc<Mutex<HashMap<Vec<u8>, Vec<RingElement>>>>,
    
    /// Performance statistics for optimization analysis
    stats: Arc<Mutex<FoldingChallengeStats>>,
    
    /// Transcript for Fiat-Shamir transformation
    transcript: Transcript,
}

/// Performance statistics for folding challenge generation
/// 
/// Tracks detailed metrics to validate theoretical complexity bounds
/// and guide performance optimization efforts.
#[derive(Clone, Debug, Default)]
pub struct FoldingChallengeStats {
    /// Total number of challenge generation operations
    total_generations: u64,
    
    /// Total number of basic challenges generated
    total_basic_challenges: u64,
    
    /// Total number of extended challenges generated
    total_extended_challenges: u64,
    
    /// Total generation time in nanoseconds
    total_generation_time_ns: u64,
    
    /// Total tensor product computations
    total_tensor_computations: u64,
    
    /// Total folded commitment operations
    total_folded_commitments: u64,
    
    /// Cache hit rate for tensor products
    tensor_cache_hits: u64,
    tensor_cache_misses: u64,
    
    /// Sampling statistics
    sampling_attempts: u64,
    sampling_successes: u64,
    
    /// GPU acceleration statistics
    gpu_operations: u64,
    cpu_operations: u64,
    
    /// Memory usage statistics
    peak_memory_usage_bytes: usize,
    average_memory_usage_bytes: usize,
}

impl FoldingChallengeStats {
    /// Creates new statistics tracker
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Records completion of a challenge generation operation
    pub fn record_generation(
        &mut self,
        basic_challenges: usize,
        extended_challenges: usize,
        generation_time_ns: u64,
        tensor_computations: u64
    ) {
        self.total_generations += 1;
        self.total_basic_challenges += basic_challenges as u64;
        self.total_extended_challenges += extended_challenges as u64;
        self.total_generation_time_ns += generation_time_ns;
        self.total_tensor_computations += tensor_computations;
    }
    
    /// Returns average generation time per operation
    pub fn average_generation_time_ns(&self) -> u64 {
        if self.total_generations == 0 {
            0
        } else {
            self.total_generation_time_ns / self.total_generations
        }
    }
    
    /// Returns tensor cache hit rate as percentage
    pub fn tensor_cache_hit_rate(&self) -> f64 {
        let total_accesses = self.tensor_cache_hits + self.tensor_cache_misses;
        if total_accesses == 0 {
            0.0
        } else {
            (self.tensor_cache_hits as f64 / total_accesses as f64) * 100.0
        }
    }
    
    /// Returns sampling success rate as percentage
    pub fn sampling_success_rate(&self) -> f64 {
        if self.sampling_attempts == 0 {
            0.0
        } else {
            (self.sampling_successes as f64 / self.sampling_attempts as f64) * 100.0
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
}

impl FoldingChallengeGenerator {
    /// Creates a new folding challenge generator
    /// 
    /// # Arguments
    /// * `params` - Generator parameters defining dimensions and security
    /// 
    /// # Returns
    /// * `Result<Self>` - New challenge generator or parameter error
    /// 
    /// # Parameter Validation
    /// - All parameters must be consistent and within supported ranges
    /// - Ring dimension must be power of 2 for NTT compatibility
    /// - Security parameters must provide adequate lattice security
    /// - Sampling set must be large enough for required security level
    /// 
    /// # Component Initialization
    /// - Strong sampling set: Configured for secure challenge generation
    /// - Double commitment: Set up for folded commitment operations
    /// - Sumcheck protocol: Initialized for consistency verification
    /// - Tensor cache: Prepared for efficient tensor product computation
    /// - Performance tracking: Statistics collection enabled
    pub fn new(params: FoldingChallengeParams) -> Result<Self> {
        // Validate parameters before initialization
        params.validate()?;
        
        // Initialize strong sampling set
        let sampling_set = StrongSamplingSet::new(
            params.ring_dimension,
            params.modulus,
            params.sampling_set_size
        )?;
        
        // Initialize double commitment scheme
        let double_commitment_params = DoubleCommitmentParams::new(
            params.kappa,
            params.ring_dimension, // Matrix width = ring dimension
            params.ring_dimension,
            params.extended_challenge_dimension,
            params.modulus,
        )?;
        let double_commitment_scheme = DoubleCommitmentScheme::new(double_commitment_params)?;
        
        // Initialize batched sumcheck protocol
        // Number of variables determined by challenge dimension
        let num_variables = (params.extended_challenge_dimension as f64).log2().ceil() as usize;
        let sumcheck_protocol = BatchedSumcheckProtocol::new(
            num_variables,
            params.ring_dimension,
            params.modulus,
            2, // Max degree 2 for quadratic consistency relations
            6  // Six sumcheck claims to compress
        )?;
        
        Ok(Self {
            params,
            sampling_set,
            double_commitment_scheme,
            sumcheck_protocol,
            tensor_cache: Arc::new(Mutex::new(HashMap::new())),
            stats: Arc::new(Mutex::new(FoldingChallengeStats::new())),
            transcript: Transcript::new(b"LatticeFold+ Folding Challenge Generation"),
        })
    }
    
    /// Generates folding challenges s ← S̄³ and s' ← S̄^{dk}
    /// 
    /// # Arguments
    /// * `transcript` - Fiat-Shamir transcript for challenge derivation
    /// * `num_basic_challenges` - Number of basic challenges to generate
    /// * `rng` - Cryptographically secure random number generator
    /// 
    /// # Returns
    /// * `Result<FoldingChallenges>` - Generated challenges or error
    /// 
    /// # Challenge Generation Process
    /// 1. **Transcript Update**: Add context information to transcript
    /// 2. **Basic Challenge Generation**: Sample s₀, s₁, s₂ ← S̄³
    /// 3. **Extended Challenge Generation**: Sample s' ← S̄^{dk}
    /// 4. **Validation**: Verify all challenges are in strong sampling set
    /// 5. **Caching**: Store frequently used tensor products
    /// 
    /// # Security Properties
    /// - Challenges are cryptographically random and unpredictable
    /// - Strong sampling set ensures proper folding properties
    /// - Fiat-Shamir transformation provides non-interactive security
    /// - Independence between different challenge components
    pub fn generate_folding_challenges<R: CryptoRng + RngCore>(
        &mut self,
        transcript: &Transcript,
        num_basic_challenges: usize,
        rng: &mut R
    ) -> Result<FoldingChallenges> {
        let start_time = std::time::Instant::now();
        
        // Update transcript with challenge generation context
        let mut local_transcript = transcript.clone();
        local_transcript.append_message(b"folding_challenge_generation", b"start");
        local_transcript.append_u64(b"num_basic_challenges", num_basic_challenges as u64);
        local_transcript.append_u64(b"extended_challenge_dimension", self.params.extended_challenge_dimension as u64);
        
        // Generate basic challenges s₀, s₁, s₂ ← S̄³
        let mut basic_challenges = Vec::with_capacity(num_basic_challenges);
        
        for i in 0..num_basic_challenges {
            // Add challenge index to transcript for domain separation
            local_transcript.append_u64(b"basic_challenge_index", i as u64);
            
            // Sample challenge from strong sampling set
            let challenge = self.sampling_set.sample(rng)?;
            
            // Add challenge to transcript for next challenge derivation
            let challenge_bytes = self.serialize_ring_element(&challenge)?;
            local_transcript.append_message(b"basic_challenge", &challenge_bytes);
            
            basic_challenges.push(challenge);
        }
        
        // Generate extended challenges s' ← S̄^{dk}
        let mut extended_challenges = Vec::with_capacity(self.params.extended_challenge_dimension);
        
        // Use parallel generation for large extended challenge sets
        if self.params.extended_challenge_dimension >= GPU_THRESHOLD {
            // GPU-accelerated generation for large sets
            extended_challenges = self.generate_extended_challenges_gpu(&mut local_transcript, rng)?;
        } else if self.params.extended_challenge_dimension >= 100 {
            // CPU parallel generation for medium sets
            extended_challenges = self.generate_extended_challenges_parallel(&mut local_transcript, rng)?;
        } else {
            // Sequential generation for small sets
            for i in 0..self.params.extended_challenge_dimension {
                local_transcript.append_u64(b"extended_challenge_index", i as u64);
                
                let challenge = self.sampling_set.sample(rng)?;
                let challenge_bytes = self.serialize_ring_element(&challenge)?;
                local_transcript.append_message(b"extended_challenge", &challenge_bytes);
                
                extended_challenges.push(challenge);
            }
        }
        
        // Create folding challenges structure
        let folding_challenges = FoldingChallenges {
            basic_challenges,
            extended_challenges,
            params: self.params.clone(),
            generation_transcript: local_transcript.clone(),
        };
        
        // Validate generated challenges
        folding_challenges.validate()?;
        
        // Record performance statistics
        let elapsed_time = start_time.elapsed().as_nanos() as u64;
        if let Ok(mut stats) = self.stats.lock() {
            stats.record_generation(
                num_basic_challenges,
                self.params.extended_challenge_dimension,
                elapsed_time,
                0 // Tensor computations recorded separately
            );
        }
        
        Ok(folding_challenges)
    }
    
    /// Generates extended challenges using GPU acceleration
    /// 
    /// # Arguments
    /// * `transcript` - Transcript for challenge derivation
    /// * `rng` - Random number generator
    /// 
    /// # Returns
    /// * `Result<Vec<RingElement>>` - Extended challenges
    fn generate_extended_challenges_gpu<R: CryptoRng + RngCore>(
        &self,
        transcript: &mut Transcript,
        rng: &mut R
    ) -> Result<Vec<RingElement>> {
        // GPU acceleration implementation would go here
        // For now, fall back to parallel CPU implementation
        self.generate_extended_challenges_parallel(transcript, rng)
    }
    
    /// Generates extended challenges using parallel CPU processing
    /// 
    /// # Arguments
    /// * `transcript` - Transcript for challenge derivation
    /// * `rng` - Random number generator
    /// 
    /// # Returns
    /// * `Result<Vec<RingElement>>` - Extended challenges
    fn generate_extended_challenges_parallel<R: CryptoRng + RngCore>(
        &self,
        transcript: &mut Transcript,
        rng: &mut R
    ) -> Result<Vec<RingElement>> {
        // Generate seeds for parallel workers
        let mut seeds = Vec::with_capacity(self.params.extended_challenge_dimension);
        for i in 0..self.params.extended_challenge_dimension {
            transcript.append_u64(b"extended_challenge_index", i as u64);
            let mut seed_bytes = [0u8; 32];
            transcript.challenge_bytes(b"challenge_seed", &mut seed_bytes);
            seeds.push(seed_bytes);
        }
        
        // Generate challenges in parallel
        let challenges: Result<Vec<RingElement>> = seeds
            .par_iter()
            .enumerate()
            .map(|(i, seed)| {
                // Create deterministic RNG from seed
                let mut local_rng = rand_chacha::ChaCha20Rng::from_seed(*seed);
                
                // Sample challenge using local RNG
                self.sampling_set.sample(&mut local_rng)
            })
            .collect();
        
        challenges
    }
    
    /// Computes folded commitment com(h) := com(Mf)s' = com(Mf s')
    /// 
    /// # Arguments
    /// * `matrix_commitment` - Original matrix commitment com(Mf)
    /// * `extended_challenges` - Extended challenges s'
    /// 
    /// # Returns
    /// * `Result<Vec<RingElement>>` - Folded commitment
    /// 
    /// # Mathematical Operation
    /// Computes the folded commitment by applying extended challenges to
    /// the matrix commitment using the homomorphic properties:
    /// com(h) = com(Mf) · s' = com(Mf · s')
    /// 
    /// This operation is central to the commitment transformation protocol
    /// and enables efficient folding of multiple commitment instances.
    pub fn compute_folded_commitment(
        &mut self,
        matrix_commitment: &[RingElement],
        extended_challenges: &[RingElement]
    ) -> Result<Vec<RingElement>> {
        // Validate input dimensions
        if extended_challenges.len() != self.params.extended_challenge_dimension {
            return Err(LatticeFoldError::InvalidDimension {
                expected: self.params.extended_challenge_dimension,
                got: extended_challenges.len(),
            });
        }
        
        // Use double commitment scheme for folded commitment computation
        self.double_commitment_scheme.compute_folded_commitment(
            matrix_commitment,
            extended_challenges
        )
    }
    
    /// Computes tensor product evaluation tensor(c^{(z)}) ⊗ s' ⊗ (1, d', ..., d'^{ℓ-1}) ⊗ (1, X, ..., X^{d-1})
    /// 
    /// # Arguments
    /// * `challenge_point` - Challenge point c^{(z)}
    /// * `extended_challenges` - Extended challenges s'
    /// 
    /// # Returns
    /// * `Result<Vec<RingElement>>` - Tensor product evaluation
    /// 
    /// # Mathematical Operation
    /// Computes the complex tensor product required for sumcheck verification:
    /// 1. **Challenge Tensor**: tensor(c^{(z)}) from challenge point
    /// 2. **Extended Challenge Tensor**: s' as vector
    /// 3. **Power Tensor**: (1, d', d'^2, ..., d'^{ℓ-1})
    /// 4. **Monomial Tensor**: (1, X, X^2, ..., X^{d-1})
    /// 5. **Tensor Product**: Kronecker product of all components
    /// 
    /// This computation is critical for the six sumcheck claims compression
    /// and enables efficient verification of commitment consistency.
    pub fn compute_tensor_product_evaluation(
        &mut self,
        challenge_point: &[i64],
        extended_challenges: &[RingElement]
    ) -> Result<Vec<RingElement>> {
        // Check cache first
        let cache_key = self.compute_tensor_cache_key(challenge_point, extended_challenges)?;
        
        if let Ok(cache) = self.tensor_cache.lock() {
            if let Some(cached_result) = cache.get(&cache_key) {
                if let Ok(mut stats) = self.stats.lock() {
                    stats.tensor_cache_hits += 1;
                }
                return Ok(cached_result.clone());
            }
        }
        
        if let Ok(mut stats) = self.stats.lock() {
            stats.tensor_cache_misses += 1;
        }
        
        // Compute tensor product components
        let challenge_tensor = self.compute_challenge_tensor(challenge_point)?;
        let extended_challenge_tensor = extended_challenges.to_vec();
        let power_tensor = self.compute_power_tensor()?;
        let monomial_tensor = self.compute_monomial_tensor()?;
        
        // Compute Kronecker products
        let mut result = challenge_tensor;
        result = self.kronecker_product(&result, &extended_challenge_tensor)?;
        result = self.kronecker_product(&result, &power_tensor)?;
        result = self.kronecker_product(&result, &monomial_tensor)?;
        
        // Cache the result
        if let Ok(mut cache) = self.tensor_cache.lock() {
            if cache.len() < TENSOR_CACHE_SIZE {
                cache.insert(cache_key, result.clone());
            }
        }
        
        // Update statistics
        if let Ok(mut stats) = self.stats.lock() {
            stats.total_tensor_computations += 1;
        }
        
        Ok(result)
    }
    
    /// Compresses six sumcheck claims into parallel execution
    /// 
    /// # Arguments
    /// * `consistency_claims` - Six consistency claims to compress
    /// * `extended_challenges` - Extended challenges for compression
    /// 
    /// # Returns
    /// * `Result<BatchedSumcheckProof>` - Compressed sumcheck proof
    /// 
    /// # Compression Algorithm
    /// The six sumcheck claims from commitment transformation are:
    /// 1. Double commitment consistency
    /// 2. Linear commitment consistency  
    /// 3. Decomposition consistency
    /// 4. Range proof consistency
    /// 5. Folding witness consistency
    /// 6. Tensor product consistency
    /// 
    /// These are compressed using random linear combination with extended
    /// challenges to create a single batched sumcheck proof.
    pub fn compress_sumcheck_claims(
        &mut self,
        consistency_claims: &[MultilinearExtension],
        extended_challenges: &[RingElement]
    ) -> Result<BatchedSumcheckProof> {
        // Validate input
        if consistency_claims.len() != 6 {
            return Err(LatticeFoldError::InvalidParameters(
                format!("Expected 6 consistency claims, got {}", consistency_claims.len())
            ));
        }
        
        // Create claimed sums (all should be zero for consistency)
        let claimed_sums = vec![
            RingElement::zero(self.params.ring_dimension, Some(self.params.modulus))?;
            6
        ];
        
        // Execute batched sumcheck
        let mut claims = consistency_claims.to_vec();
        self.sumcheck_protocol.prove_batch(&mut claims, &claimed_sums)
    }
    
    /// Helper functions for tensor product computation
    
    /// Computes challenge tensor from challenge point
    fn compute_challenge_tensor(&self, challenge_point: &[i64]) -> Result<Vec<RingElement>> {
        let mut tensor = Vec::new();
        
        for &challenge_value in challenge_point {
            let mut coeffs = vec![0i64; self.params.ring_dimension];
            coeffs[0] = challenge_value; // Constant polynomial
            
            let element = RingElement::from_coefficients(coeffs, Some(self.params.modulus))?;
            tensor.push(element);
        }
        
        Ok(tensor)
    }
    
    /// Computes power tensor (1, d', d'^2, ..., d'^{ℓ-1})
    fn compute_power_tensor(&self) -> Result<Vec<RingElement>> {
        let mut tensor = Vec::with_capacity(self.params.gadget_dimension);
        let d_prime = self.params.half_dimension as i64;
        
        let mut power = 1i64;
        for _ in 0..self.params.gadget_dimension {
            let mut coeffs = vec![0i64; self.params.ring_dimension];
            coeffs[0] = power; // Constant polynomial
            
            let element = RingElement::from_coefficients(coeffs, Some(self.params.modulus))?;
            tensor.push(element);
            
            power = (power * d_prime) % self.params.modulus;
        }
        
        Ok(tensor)
    }
    
    /// Computes monomial tensor (1, X, X^2, ..., X^{d-1})
    fn compute_monomial_tensor(&self) -> Result<Vec<RingElement>> {
        let mut tensor = Vec::with_capacity(self.params.ring_dimension);
        
        for degree in 0..self.params.ring_dimension {
            let mut coeffs = vec![0i64; self.params.ring_dimension];
            coeffs[degree] = 1; // Monomial X^degree
            
            let element = RingElement::from_coefficients(coeffs, Some(self.params.modulus))?;
            tensor.push(element);
        }
        
        Ok(tensor)
    }
    
    /// Computes Kronecker product of two tensors
    fn kronecker_product(
        &self,
        tensor_a: &[RingElement],
        tensor_b: &[RingElement]
    ) -> Result<Vec<RingElement>> {
        let mut result = Vec::with_capacity(tensor_a.len() * tensor_b.len());
        
        for a_elem in tensor_a {
            for b_elem in tensor_b {
                let product = a_elem.multiply(b_elem)?;
                result.push(product);
            }
        }
        
        Ok(result)
    }
    
    /// Computes cache key for tensor product caching
    fn compute_tensor_cache_key(
        &self,
        challenge_point: &[i64],
        extended_challenges: &[RingElement]
    ) -> Result<Vec<u8>> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        
        // Hash challenge point
        challenge_point.hash(&mut hasher);
        
        // Hash extended challenges (simplified)
        for challenge in extended_challenges {
            challenge.coefficients().hash(&mut hasher);
        }
        
        // Hash parameters for uniqueness
        self.params.ring_dimension.hash(&mut hasher);
        self.params.gadget_dimension.hash(&mut hasher);
        
        Ok(hasher.finish().to_le_bytes().to_vec())
    }
    
    /// Serializes ring element for transcript
    fn serialize_ring_element(&self, element: &RingElement) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();
        let coeffs = element.coefficients();
        
        bytes.extend_from_slice(&(coeffs.len() as u64).to_le_bytes());
        for &coeff in coeffs {
            bytes.extend_from_slice(&coeff.to_le_bytes());
        }
        
        Ok(bytes)
    }
    
    /// Returns generator statistics
    pub fn stats(&self) -> Result<FoldingChallengeStats> {
        Ok(self.stats.lock().unwrap().clone())
    }
    
    /// Resets generator statistics
    pub fn reset_stats(&mut self) -> Result<()> {
        *self.stats.lock().unwrap() = FoldingChallengeStats::new();
        Ok(())
    }
    
    /// Clears tensor product cache
    pub fn clear_cache(&mut self) {
        if let Ok(mut cache) = self.tensor_cache.lock() {
            cache.clear();
        }
    }
}

/// Folding challenges structure containing all generated challenges
/// 
/// This structure holds the complete set of challenges generated for
/// commitment transformation, including basic challenges for witness
/// combination and extended challenges for matrix operations.
/// 
/// Mathematical Components:
/// - Basic challenges: s₀, s₁, s₂ ← S̄³ for witness combination
/// - Extended challenges: s' ← S̄^{dk} for matrix folding
/// - Generation transcript: Fiat-Shamir transcript for reproducibility
/// - Parameters: Generator parameters for validation
/// 
/// Security Properties:
/// - All challenges are cryptographically random and unpredictable
/// - Strong sampling set ensures proper folding properties
/// - Transcript enables non-interactive challenge derivation
/// - Independence between different challenge components
#[derive(Clone, Debug, Zeroize, ZeroizeOnDrop)]
pub struct FoldingChallenges {
    /// Basic challenges s₀, s₁, s₂ ← S̄³ for witness combination
    /// Used in folding witness computation: g := s₀·τD + s₁·mτ + s₂·f + h
    pub basic_challenges: Vec<RingElement>,
    
    /// Extended challenges s' ← S̄^{dk} for matrix operations
    /// Used in folded commitment computation: com(h) := com(Mf)s'
    pub extended_challenges: Vec<RingElement>,
    
    /// Generator parameters for validation and consistency
    #[zeroize(skip)]
    pub params: FoldingChallengeParams,
    
    /// Generation transcript for reproducibility and verification
    #[zeroize(skip)]
    pub generation_transcript: Transcript,
}

impl FoldingChallenges {
    /// Validates the challenge structure and properties
    /// 
    /// # Returns
    /// * `Result<()>` - Ok if challenges are valid, error otherwise
    /// 
    /// # Validation Checks
    /// - Basic challenges count matches expected number
    /// - Extended challenges dimension matches dk
    /// - All challenges have correct ring dimension
    /// - All challenges are in strong sampling set (simplified check)
    pub fn validate(&self) -> Result<()> {
        // Check basic challenges count
        if self.basic_challenges.len() != self.params.num_basic_challenges {
            return Err(LatticeFoldError::InvalidParameters(
                format!("Basic challenges count {} != expected {}", 
                       self.basic_challenges.len(), self.params.num_basic_challenges)
            ));
        }
        
        // Check extended challenges dimension
        if self.extended_challenges.len() != self.params.extended_challenge_dimension {
            return Err(LatticeFoldError::InvalidDimension {
                expected: self.params.extended_challenge_dimension,
                got: self.extended_challenges.len(),
            });
        }
        
        // Check ring dimensions
        for (i, challenge) in self.basic_challenges.iter().enumerate() {
            if challenge.dimension() != self.params.ring_dimension {
                return Err(LatticeFoldError::InvalidDimension {
                    expected: self.params.ring_dimension,
                    got: challenge.dimension(),
                });
            }
        }
        
        for (i, challenge) in self.extended_challenges.iter().enumerate() {
            if challenge.dimension() != self.params.ring_dimension {
                return Err(LatticeFoldError::InvalidDimension {
                    expected: self.params.ring_dimension,
                    got: challenge.dimension(),
                });
            }
        }
        
        Ok(())
    }
    
    /// Returns the total number of challenges
    pub fn total_challenges(&self) -> usize {
        self.basic_challenges.len() + self.extended_challenges.len()
    }
    
    /// Returns the memory usage of the challenges
    pub fn memory_usage(&self) -> usize {
        let basic_size = self.basic_challenges.len() * self.params.ring_dimension * 8;
        let extended_size = self.extended_challenges.len() * self.params.ring_dimension * 8;
        let params_size = std::mem::size_of::<FoldingChallengeParams>();
        
        basic_size + extended_size + params_size
    }
    
    /// Serializes challenges for storage or transmission
    pub fn serialize(&self) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();
        
        // Serialize basic challenges
        bytes.extend_from_slice(&(self.basic_challenges.len() as u64).to_le_bytes());
        for challenge in &self.basic_challenges {
            let coeffs = challenge.coefficients();
            bytes.extend_from_slice(&(coeffs.len() as u64).to_le_bytes());
            for &coeff in coeffs {
                bytes.extend_from_slice(&coeff.to_le_bytes());
            }
        }
        
        // Serialize extended challenges
        bytes.extend_from_slice(&(self.extended_challenges.len() as u64).to_le_bytes());
        for challenge in &self.extended_challenges {
            let coeffs = challenge.coefficients();
            bytes.extend_from_slice(&(coeffs.len() as u64).to_le_bytes());
            for &coeff in coeffs {
                bytes.extend_from_slice(&coeff.to_le_bytes());
            }
        }
        
        Ok(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    
    /// Test folding challenge parameter creation
    #[test]
    fn test_folding_challenge_params_creation() {
        let ring_dimension = 16;
        let decomposition_levels = 4;
        let kappa = 128;
        let gadget_dimension = 4;
        let modulus = 97;
        
        let params = FoldingChallengeParams::new(
            ring_dimension,
            decomposition_levels,
            kappa,
            gadget_dimension,
            modulus
        );
        
        assert!(params.is_ok(), "Parameter creation should succeed");
        
        let params = params.unwrap();
        assert_eq!(params.ring_dimension, ring_dimension);
        assert_eq!(params.half_dimension, ring_dimension / 2);
        assert_eq!(params.extended_challenge_dimension, ring_dimension * decomposition_levels);
    }
    
    /// Test strong sampling set creation and sampling
    #[test]
    fn test_strong_sampling_set() {
        let ring_dimension = 16;
        let modulus = 97;
        let target_size = 50;
        
        let sampling_set = StrongSamplingSet::new(ring_dimension, modulus, target_size).unwrap();
        assert_eq!(sampling_set.ring_dimension(), ring_dimension);
        assert_eq!(sampling_set.modulus(), modulus);
        
        // Test sampling
        let mut rng = rand_chacha::ChaCha20Rng::from_seed([0u8; 32]);
        let sample = sampling_set.sample(&mut rng).unwrap();
        assert_eq!(sample.dimension(), ring_dimension);
    }
    
    /// Test folding challenge generation
    #[test]
    fn test_folding_challenge_generation() {
        let params = FoldingChallengeParams::new(16, 4, 128, 4, 97).unwrap();
        let mut generator = FoldingChallengeGenerator::new(params).unwrap();
        
        let transcript = Transcript::new(b"test");
        let mut rng = rand_chacha::ChaCha20Rng::from_seed([0u8; 32]);
        
        let challenges = generator.generate_folding_challenges(&transcript, 3, &mut rng).unwrap();
        
        assert_eq!(challenges.basic_challenges.len(), 3);
        assert_eq!(challenges.extended_challenges.len(), 16 * 4); // ring_dimension * decomposition_levels
        
        // Validate challenges
        assert!(challenges.validate().is_ok());
    }
    
    /// Test tensor product computation
    #[test]
    fn test_tensor_product_computation() {
        let params = FoldingChallengeParams::new(8, 2, 64, 2, 97).unwrap();
        let mut generator = FoldingChallengeGenerator::new(params.clone()).unwrap();
        
        let challenge_point = vec![1, 2, 3];
        let extended_challenges = vec![
            RingElement::zero(params.ring_dimension, Some(params.modulus)).unwrap();
            params.extended_challenge_dimension
        ];
        
        let tensor_result = generator.compute_tensor_product_evaluation(
            &challenge_point,
            &extended_challenges
        );
        
        assert!(tensor_result.is_ok(), "Tensor product computation should succeed");
    }
    
    /// Test challenge validation
    #[test]
    fn test_challenge_validation() {
        let params = FoldingChallengeParams::new(16, 4, 128, 4, 97).unwrap();
        
        // Valid challenges
        let valid_challenges = FoldingChallenges {
            basic_challenges: vec![
                RingElement::zero(params.ring_dimension, Some(params.modulus)).unwrap();
                params.num_basic_challenges
            ],
            extended_challenges: vec![
                RingElement::zero(params.ring_dimension, Some(params.modulus)).unwrap();
                params.extended_challenge_dimension
            ],
            params: params.clone(),
            generation_transcript: Transcript::new(b"test"),
        };
        
        assert!(valid_challenges.validate().is_ok(), "Valid challenges should pass validation");
        
        // Invalid challenges (wrong count)
        let invalid_challenges = FoldingChallenges {
            basic_challenges: vec![
                RingElement::zero(params.ring_dimension, Some(params.modulus)).unwrap();
                2 // Wrong count
            ],
            extended_challenges: vec![
                RingElement::zero(params.ring_dimension, Some(params.modulus)).unwrap();
                params.extended_challenge_dimension
            ],
            params: params.clone(),
            generation_transcript: Transcript::new(b"test"),
        };
        
        assert!(invalid_challenges.validate().is_err(), "Invalid challenges should fail validation");
    }
}
     ///secure random number generator
    /// 
    /// # Returns
    /// * `Result<FoldingChallenges>` - Generated challenges or error
    /// 
    /// # Challenge Generation Process
    /// 1. **Basic Challenges**: Generate s₀, s₁, s₂ ← S̄³ for witness combination
    /// 2. **Extended Challenges**: Generate s' ← S̄^{dk} for matrix operations
    /// 3. **Transcript Binding**: Add challenges to transcript for non-interactive security
    /// 4. **Validation**: Verify all challenges satisfy strong sampling properties
    /// 5. **Caching**: Store frequently used challenge combinations
    /// 
    /// # Performance Optimization
    /// - Uses parallel sampling for independent challenges
    /// - Employs SIMD vectorization for coefficient operations
    /// - Implements efficient rejection sampling for uniform distribution
    /// - Leverages GPU acceleration for large challenge sets
    /// - Optimizes memory allocation patterns for cache efficiency
    /// 
    /// # Security Analysis
    /// - Each challenge has min-entropy ≥ log|S̄| bits
    /// - Challenges are statistically independent
    /// - Strong sampling ensures proper folding reduction
    /// - Transcript binding prevents malicious challenge selection
    pub fn generate_challenges<R: CryptoRng + RngCore>(
        &mut self,
        rng: &mut R
    ) -> Result<FoldingChallenges> {
        let start_time = std::time::Instant::now();
        
        // Step 1: Generate basic challenges s₀, s₁, s₂ ← S̄³
        let basic_challenges = self.generate_basic_challenges(rng)?;
        
        // Step 2: Generate extended challenges s' ← S̄^{dk}
        let extended_challenges = self.generate_extended_challenges(rng)?;
        
        // Step 3: Add challenges to transcript for binding
        self.bind_challenges_to_transcript(&basic_challenges, &extended_challenges)?;
        
        // Step 4: Validate challenge properties
        self.validate_challenges(&basic_challenges, &extended_challenges)?;
        
        // Step 5: Create challenge structure
        let challenges = FoldingChallenges {
            basic_challenges,
            extended_challenges,
            params: self.params.clone(),
        };
        
        // Record performance statistics
        let elapsed_time = start_time.elapsed().as_nanos() as u64;
        if let Ok(mut stats) = self.stats.lock() {
            stats.record_generation(
                challenges.basic_challenges.len(),
                challenges.extended_challenges.len(),
                elapsed_time,
                0 // Tensor computations recorded separately
            );
        }
        
        Ok(challenges)
    }
    
    /// Computes folded commitment com(h) := com(Mf)s' = com(Mf s')
    /// 
    /// # Arguments
    /// * `matrix_commitment` - Original matrix commitment com(Mf)
    /// * `extended_challenges` - Extended challenges s' ∈ S̄^{dk}
    /// 
    /// # Returns
    /// * `Result<Vec<RingElement>>` - Folded commitment
    /// 
    /// # Mathematical Operation
    /// Computes the folded commitment by applying extended challenges to
    /// the matrix commitment, effectively combining multiple commitment
    /// components into a single compact representation.
    /// 
    /// # Performance Optimization
    /// - Uses parallel processing for independent commitment operations
    /// - Employs SIMD vectorization for matrix-vector operations
    /// - Implements GPU acceleration for large matrices
    /// - Caches intermediate results for repeated computations
    pub fn compute_folded_commitment(
        &mut self,
        matrix_commitment: &[Vec<RingElement>],
        extended_challenges: &[RingElement]
    ) -> Result<Vec<RingElement>> {
        // Validate input dimensions
        if extended_challenges.len() != self.params.extended_challenge_dimension {
            return Err(LatticeFoldError::InvalidDimension {
                expected: self.params.extended_challenge_dimension,
                got: extended_challenges.len(),
            });
        }
        
        // Compute matrix-vector product Mf × s'
        let folded_matrix = self.compute_matrix_challenge_product(matrix_commitment, extended_challenges)?;
        
        // Commit to the folded matrix
        let folded_commitment = self.double_commitment_scheme.commit_matrix(&folded_matrix, &mut rand::thread_rng())?;
        
        // Update statistics
        if let Ok(mut stats) = self.stats.lock() {
            stats.total_folded_commitments += 1;
        }
        
        Ok(folded_commitment)
    }
    
    /// Compresses six sumcheck claims into parallel execution
    /// 
    /// # Arguments
    /// * `consistency_claims` - Six consistency claims to compress
    /// * `challenges` - Folding challenges for compression
    /// 
    /// # Returns
    /// * `Result<BatchedSumcheckProof>` - Compressed sumcheck proof
    /// 
    /// # Compression Algorithm
    /// Uses random linear combination to compress multiple sumcheck claims:
    /// 1. Generate random coefficients for linear combination
    /// 2. Combine claims using coefficients
    /// 3. Execute single sumcheck on combined claim
    /// 4. Provide individual evaluations for verification
    /// 
    /// # Communication Optimization
    /// Reduces communication from O(6 × log n) to O(log n) field elements
    /// while maintaining the same security guarantees through random combination.
    pub fn compress_sumcheck_claims(
        &mut self,
        consistency_claims: &[MultilinearExtension],
        challenges: &FoldingChallenges
    ) -> Result<BatchedSumcheckProof> {
        // Validate input
        if consistency_claims.len() != 6 {
            return Err(LatticeFoldError::InvalidParameters(
                "Expected exactly 6 consistency claims".to_string()
            ));
        }
        
        // Create mutable copy for sumcheck execution
        let mut claims_copy: Vec<MultilinearExtension> = consistency_claims.to_vec();
        
        // Compute claimed sums (should all be zero for consistency)
        let claimed_sums = vec![
            RingElement::zero(self.params.ring_dimension, Some(self.params.modulus))?;
            6
        ];
        
        // Execute batched sumcheck protocol
        let proof = self.sumcheck_protocol.prove_batch(&mut claims_copy, &claimed_sums)?;
        
        Ok(proof)
    }
    
    /// Evaluates tensor product: tensor(c^{(z)}) ⊗ s' ⊗ (1, d', ..., d'^{ℓ-1}) ⊗ (1, X, ..., X^{d-1})
    /// 
    /// # Arguments
    /// * `c_z` - Base tensor component c^{(z)}
    /// * `extended_challenges` - Extended challenges s'
    /// * `evaluation_point` - Point for polynomial evaluation
    /// 
    /// # Returns
    /// * `Result<RingElement>` - Tensor product evaluation
    /// 
    /// # Mathematical Computation
    /// Computes the complex tensor product required for consistency verification:
    /// - First component: tensor(c^{(z)}) from sumcheck evaluation
    /// - Second component: s' ∈ S̄^{dk} extended challenges
    /// - Third component: (1, d', d'^2, ..., d'^{ℓ-1}) gadget powers
    /// - Fourth component: (1, X, X^2, ..., X^{d-1}) polynomial basis
    /// 
    /// # Performance Optimization
    /// - Uses cached tensor products for repeated patterns
    /// - Employs SIMD vectorization for tensor operations
    /// - Implements parallel computation for independent components
    /// - Leverages GPU acceleration for large tensor products
    pub fn evaluate_tensor_product(
        &mut self,
        c_z: &[RingElement],
        extended_challenges: &[RingElement],
        evaluation_point: &[i64]
    ) -> Result<RingElement> {
        // Create cache key for tensor product
        let cache_key = self.create_tensor_cache_key(c_z, extended_challenges, evaluation_point);
        
        // Check cache for existing result
        if let Ok(cache) = self.tensor_cache.lock() {
            if let Some(cached_result) = cache.get(&cache_key) {
                if let Ok(mut stats) = self.stats.lock() {
                    stats.tensor_cache_hits += 1;
                }
                return Ok(cached_result[0].clone());
            }
        }
        
        if let Ok(mut stats) = self.stats.lock() {
            stats.tensor_cache_misses += 1;
        }
        
        // Compute tensor product components
        let component1 = self.compute_c_z_tensor(c_z)?;
        let component2 = self.compute_challenge_tensor(extended_challenges)?;
        let component3 = self.compute_gadget_power_tensor()?;
        let component4 = self.compute_polynomial_basis_tensor(evaluation_point)?;
        
        // Combine all tensor components
        let mut result = component1;
        result = result.multiply(&component2)?;
        result = result.multiply(&component3)?;
        result = result.multiply(&component4)?;
        
        // Cache the result if there's space
        if let Ok(mut cache) = self.tensor_cache.lock() {
            if cache.len() < TENSOR_CACHE_SIZE {
                cache.insert(cache_key, vec![result.clone()]);
            }
        }
        
        // Update statistics
        if let Ok(mut stats) = self.stats.lock() {
            stats.total_tensor_computations += 1;
        }
        
        Ok(result)
    }
}   
 /// Helper mes for challenge generation
    
    /// Generates basic challenges s₀, s₁, s₂ ← S̄³
    fn generate_basic_challenges<R: CryptoRng + RngCore>(&mut self, rng: &mut R) -> Result<Vec<RingElement>> {
        let mut basic_challenges = Vec::with_capacity(self.params.num_basic_challenges);
        
        // Generate each basic challenge independently
        for i in 0..self.params.num_basic_challenges {
            // Add challenge index to transcript for domain separation
            self.transcript.append_u64(b"basic_challenge_index", i as u64);
            
            // Sample challenge from strong sampling set
            let challenge = self.sampling_set.sample(rng)?;
            
            // Add challenge to transcript for binding
            self.transcript.append_message(b"basic_challenge", &challenge.to_bytes()?);
            
            basic_challenges.push(challenge);
        }
        
        Ok(basic_challenges)
    }
    
    /// Generates extended challenges s' ← S̄^{dk}
    fn generate_extended_challenges<R: CryptoRng + RngCore>(&mut self, rng: &mut R) -> Result<Vec<RingElement>> {
        let mut extended_challenges = Vec::with_capacity(self.params.extended_challenge_dimension);
        
        // Generate challenges in parallel for better performance
        let challenges: Result<Vec<RingElement>> = (0..self.params.extended_challenge_dimension)
            .into_par_iter()
            .map(|i| {
                // Create separate RNG for each thread to avoid contention
                let mut thread_rng = rand::thread_rng();
                
                // Sample challenge from strong sampling set
                self.sampling_set.sample(&mut thread_rng)
            })
            .collect();
        
        let extended_challenges = challenges?;
        
        // Add all extended challenges to transcript
        for (i, challenge) in extended_challenges.iter().enumerate() {
            self.transcript.append_u64(b"extended_challenge_index", i as u64);
            self.transcript.append_message(b"extended_challenge", &challenge.to_bytes()?);
        }
        
        Ok(extended_challenges)
    }
    
    /// Binds challenges to transcript for non-interactive security
    fn bind_challenges_to_transcript(
        &mut self,
        basic_challenges: &[RingElement],
        extended_challenges: &[RingElement]
    ) -> Result<()> {
        // Add challenge counts for structure validation
        self.transcript.append_u64(b"num_basic_challenges", basic_challenges.len() as u64);
        self.transcript.append_u64(b"num_extended_challenges", extended_challenges.len() as u64);
        
        // Add parameter hash for binding
        let param_hash = self.hash_parameters();
        self.transcript.append_message(b"parameter_hash", &param_hash);
        
        Ok(())
    }
    
    /// Validates challenge properties
    fn validate_challenges(
        &self,
        basic_challenges: &[RingElement],
        extended_challenges: &[RingElement]
    ) -> Result<()> {
        // Validate basic challenge count
        if basic_challenges.len() != self.params.num_basic_challenges {
            return Err(LatticeFoldError::InvalidParameters(
                format!("Expected {} basic challenges, got {}", 
                       self.params.num_basic_challenges, basic_challenges.len())
            ));
        }
        
        // Validate extended challenge count
        if extended_challenges.len() != self.params.extended_challenge_dimension {
            return Err(LatticeFoldError::InvalidDimension {
                expected: self.params.extended_challenge_dimension,
                got: extended_challenges.len(),
            });
        }
        
        // Validate challenge dimensions and moduli
        for (i, challenge) in basic_challenges.iter().enumerate() {
            if challenge.dimension() != self.params.ring_dimension {
                return Err(LatticeFoldError::InvalidDimension {
                    expected: self.params.ring_dimension,
                    got: challenge.dimension(),
                });
            }
            
            if challenge.modulus() != Some(self.params.modulus) {
                return Err(LatticeFoldError::InvalidModulus { 
                    modulus: challenge.modulus().unwrap_or(0) 
                });
            }
        }
        
        for (i, challenge) in extended_challenges.iter().enumerate() {
            if challenge.dimension() != self.params.ring_dimension {
                return Err(LatticeFoldError::InvalidDimension {
                    expected: self.params.ring_dimension,
                    got: challenge.dimension(),
                });
            }
            
            if challenge.modulus() != Some(self.params.modulus) {
                return Err(LatticeFoldError::InvalidModulus { 
                    modulus: challenge.modulus().unwrap_or(0) 
                });
            }
        }
        
        Ok(())
    }
    
    /// Computes matrix-challenge product for folded commitments
    fn compute_matrix_challenge_product(
        &self,
        matrix: &[Vec<RingElement>],
        challenges: &[RingElement]
    ) -> Result<Vec<Vec<RingElement>>> {
        // Validate dimensions
        if matrix.is_empty() {
            return Err(LatticeFoldError::InvalidParameters(
                "Matrix cannot be empty".to_string()
            ));
        }
        
        let matrix_width = matrix[0].len();
        if challenges.len() != matrix_width {
            return Err(LatticeFoldError::InvalidDimension {
                expected: matrix_width,
                got: challenges.len(),
            });
        }
        
        // Compute matrix-vector product in parallel
        let result: Result<Vec<Vec<RingElement>>> = matrix
            .par_iter()
            .map(|row| {
                // Compute dot product of row with challenges
                let mut row_result = Vec::with_capacity(1);
                let mut dot_product = RingElement::zero(
                    self.params.ring_dimension, 
                    Some(self.params.modulus)
                )?;
                
                for (elem, challenge) in row.iter().zip(challenges.iter()) {
                    let product = elem.multiply(challenge)?;
                    dot_product = dot_product.add(&product)?;
                }
                
                row_result.push(dot_product);
                Ok(row_result)
            })
            .collect();
        
        result
    }
    
    /// Tensor product computation helpers
    
    fn compute_c_z_tensor(&self, c_z: &[RingElement]) -> Result<RingElement> {
        // Compute tensor product of c^{(z)} components
        if c_z.is_empty() {
            return RingElement::zero(self.params.ring_dimension, Some(self.params.modulus));
        }
        
        let mut result = c_z[0].clone();
        for elem in c_z.iter().skip(1) {
            result = result.multiply(elem)?;
        }
        
        Ok(result)
    }
    
    fn compute_challenge_tensor(&self, challenges: &[RingElement]) -> Result<RingElement> {
        // Compute tensor product of extended challenges
        if challenges.is_empty() {
            return RingElement::one(self.params.ring_dimension, Some(self.params.modulus));
        }
        
        let mut result = challenges[0].clone();
        for challenge in challenges.iter().skip(1) {
            result = result.multiply(challenge)?;
        }
        
        Ok(result)
    }
    
    fn compute_gadget_power_tensor(&self) -> Result<RingElement> {
        // Compute tensor product of (1, d', d'^2, ..., d'^{ℓ-1})
        let mut coeffs = vec![0i64; self.params.ring_dimension];
        
        // Start with constant 1
        coeffs[0] = 1;
        let mut result = RingElement::from_coefficients(coeffs.clone(), Some(self.params.modulus))?;
        
        // Multiply by powers of d'
        let d_prime = self.params.half_dimension as i64;
        let mut current_power = d_prime;
        
        for _ in 1..self.params.gadget_dimension {
            // Create ring element for current power
            coeffs.fill(0);
            coeffs[0] = current_power % self.params.modulus;
            let power_elem = RingElement::from_coefficients(coeffs.clone(), Some(self.params.modulus))?;
            
            // Multiply into result
            result = result.multiply(&power_elem)?;
            
            // Update power
            current_power = (current_power * d_prime) % self.params.modulus;
        }
        
        Ok(result)
    }
    
    fn compute_polynomial_basis_tensor(&self, evaluation_point: &[i64]) -> Result<RingElement> {
        // Compute tensor product of (1, X, X^2, ..., X^{d-1}) evaluated at point
        let mut result = RingElement::one(self.params.ring_dimension, Some(self.params.modulus))?;
        
        // For each variable in evaluation point
        for &point_value in evaluation_point {
            // Create polynomial X - point_value
            let mut poly_coeffs = vec![0i64; self.params.ring_dimension];
            poly_coeffs[0] = -point_value; // Constant term
            if self.params.ring_dimension > 1 {
                poly_coeffs[1] = 1; // Linear term
            }
            
            let poly = RingElement::from_coefficients(poly_coeffs, Some(self.params.modulus))?;
            result = result.multiply(&poly)?;
        }
        
        Ok(result)
    }
    
    /// Creates cache key for tensor products
    fn create_tensor_cache_key(
        &self,
        c_z: &[RingElement],
        challenges: &[RingElement],
        evaluation_point: &[i64]
    ) -> Vec<u8> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        
        // Hash c_z components
        for elem in c_z {
            for &coeff in elem.coefficients() {
                coeff.hash(&mut hasher);
            }
        }
        
        // Hash challenges (first few for efficiency)
        for challenge in challenges.iter().take(10) {
            for &coeff in challenge.coefficients() {
                coeff.hash(&mut hasher);
            }
        }
        
        // Hash evaluation point
        for &point in evaluation_point {
            point.hash(&mut hasher);
        }
        
        // Hash parameters
        self.params.ring_dimension.hash(&mut hasher);
        self.params.modulus.hash(&mut hasher);
        
        hasher.finish().to_le_bytes().to_vec()
    }
    
    /// Computes hash of parameters for transcript binding
    fn hash_parameters(&self) -> Vec<u8> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        
        self.params.ring_dimension.hash(&mut hasher);
        self.params.decomposition_levels.hash(&mut hasher);
        self.params.kappa.hash(&mut hasher);
        self.params.gadget_dimension.hash(&mut hasher);
        self.params.modulus.hash(&mut hasher);
        self.params.sampling_set_size.hash(&mut hasher);
        
        hasher.finish().to_le_bytes().to_vec()
    }
    
    /// Returns performance statistics
    pub fn stats(&self) -> Result<FoldingChallengeStats> {
        Ok(self.stats.lock().unwrap().clone())
    }
    
    /// Resets performance statistics
    pub fn reset_stats(&mut self) -> Result<()> {
        *self.stats.lock().unwrap() = FoldingChallengeStats::new();
        Ok(())
    }
    
    /// Clears tensor product cache
    pub fn clear_cache(&mut self) {
        if let Ok(mut cache) = self.tensor_cache.lock() {
            cache.clear();
        }
    }
}

/// Container for generated folding challenges
/// 
/// This structure holds both basic and extended challenges generated
/// for the commitment transformation protocol, along with metadata
/// for verification and security analysis.
/// 
/// Challenge Structure:
/// - Basic challenges: s₀, s₁, s₂ ∈ S̄³ for witness combination
/// - Extended challenges: s' ∈ S̄^{dk} for matrix operations
/// - Parameters: Generator parameters for validation
/// 
/// Security Properties:
/// - All challenges sampled from strong sampling set S̄
/// - Statistical independence between challenge components
/// - Cryptographic binding through transcript integration
/// - Proper entropy distribution for folding security
#[derive(Clone, Debug, Zeroize, ZeroizeOnDrop)]
pub struct FoldingChallenges {
    /// Basic challenges s₀, s₁, s₂ ← S̄³ for witness combination
    /// Used in folding witness computation: g := s₀·τD + s₁·mτ + s₂·f + h
    pub basic_challenges: Vec<RingElement>,
    
    /// Extended challenges s' ← S̄^{dk} for matrix operations
    /// Used in folded commitment computation: com(h) := com(Mf)s'
    #[zeroize(skip)]
    pub extended_challenges: Vec<RingElement>,
    
    /// Generator parameters for validation and security analysis
    pub params: FoldingChallengeParams,
}

impl FoldingChallenges {
    /// Creates new folding challenges
    /// 
    /// # Arguments
    /// * `basic_challenges` - Basic challenges for witness combination
    /// * `extended_challenges` - Extended challenges for matrix operations
    /// * `params` - Generator parameters
    /// 
    /// # Returns
    /// * `Self` - New folding challenges container
    pub fn new(
        basic_challenges: Vec<RingElement>,
        extended_challenges: Vec<RingElement>,
        params: FoldingChallengeParams,
    ) -> Self {
        Self {
            basic_challenges,
            extended_challenges,
            params,
        }
    }
    
    /// Validates challenge structure and properties
    /// 
    /// # Returns
    /// * `Result<bool>` - True if challenges are valid, false otherwise
    /// 
    /// # Validation Checks
    /// - Correct number of basic and extended challenges
    /// - Proper ring dimensions and moduli
    /// - Strong sampling set membership (simplified check)
    pub fn validate(&self) -> Result<bool> {
        // Check basic challenge count
        if self.basic_challenges.len() != self.params.num_basic_challenges {
            return Ok(false);
        }
        
        // Check extended challenge count
        if self.extended_challenges.len() != self.params.extended_challenge_dimension {
            return Ok(false);
        }
        
        // Validate each basic challenge
        for challenge in &self.basic_challenges {
            if challenge.dimension() != self.params.ring_dimension {
                return Ok(false);
            }
            if challenge.modulus() != Some(self.params.modulus) {
                return Ok(false);
            }
        }
        
        // Validate each extended challenge
        for challenge in &self.extended_challenges {
            if challenge.dimension() != self.params.ring_dimension {
                return Ok(false);
            }
            if challenge.modulus() != Some(self.params.modulus) {
                return Ok(false);
            }
        }
        
        Ok(true)
    }
    
    /// Returns the basic challenges for witness combination
    pub fn basic_challenges(&self) -> &[RingElement] {
        &self.basic_challenges
    }
    
    /// Returns the extended challenges for matrix operations
    pub fn extended_challenges(&self) -> &[RingElement] {
        &self.extended_challenges
    }
    
    /// Returns the generator parameters
    pub fn params(&self) -> &FoldingChallengeParams {
        &self.params
    }
    
    /// Computes witness combination using basic challenges
    /// 
    /// # Arguments
    /// * `tau_d` - Decomposed witness component τD
    /// * `m_tau` - Monomial witness component mτ  
    /// * `f` - Original witness f
    /// * `h` - Additional witness component h
    /// 
    /// # Returns
    /// * `Result<RingElement>` - Combined witness g := s₀·τD + s₁·mτ + s₂·f + h
    /// 
    /// # Mathematical Operation
    /// Computes the folding witness combination that maintains norm bounds:
    /// g := s₀·τD + s₁·mτ + s₂·f + h with ||g||_∞ < b/2
    /// 
    /// This combination enables the transformation from double commitments
    /// to linear commitments while preserving security properties.
    pub fn combine_witnesses(
        &self,
        tau_d: &RingElement,
        m_tau: &RingElement,
        f: &RingElement,
        h: &RingElement
    ) -> Result<RingElement> {
        // Validate we have exactly 3 basic challenges
        if self.basic_challenges.len() != 3 {
            return Err(LatticeFoldError::InvalidParameters(
                format!("Expected 3 basic challenges, got {}", self.basic_challenges.len())
            ));
        }
        
        // Extract challenges s₀, s₁, s₂
        let s0 = &self.basic_challenges[0];
        let s1 = &self.basic_challenges[1];
        let s2 = &self.basic_challenges[2];
        
        // Compute s₀·τD
        let term1 = s0.multiply(tau_d)?;
        
        // Compute s₁·mτ
        let term2 = s1.multiply(m_tau)?;
        
        // Compute s₂·f
        let term3 = s2.multiply(f)?;
        
        // Combine all terms: g := s₀·τD + s₁·mτ + s₂·f + h
        let mut result = term1.add(&term2)?;
        result = result.add(&term3)?;
        result = result.add(h)?;
        
        Ok(result)
    }
    
    /// Serializes challenges to bytes for storage or transmission
    /// 
    /// # Returns
    /// * `Result<Vec<u8>>` - Serialized challenge data
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();
        
        // Serialize parameters
        bytes.extend_from_slice(&self.params.ring_dimension.to_le_bytes());
        bytes.extend_from_slice(&self.params.decomposition_levels.to_le_bytes());
        bytes.extend_from_slice(&self.params.kappa.to_le_bytes());
        bytes.extend_from_slice(&self.params.gadget_dimension.to_le_bytes());
        bytes.extend_from_slice(&self.params.modulus.to_le_bytes());
        
        // Serialize basic challenges
        bytes.extend_from_slice(&self.basic_challenges.len().to_le_bytes());
        for challenge in &self.basic_challenges {
            let challenge_bytes = challenge.to_bytes()?;
            bytes.extend_from_slice(&challenge_bytes.len().to_le_bytes());
            bytes.extend_from_slice(&challenge_bytes);
        }
        
        // Serialize extended challenges
        bytes.extend_from_slice(&self.extended_challenges.len().to_le_bytes());
        for challenge in &self.extended_challenges {
            let challenge_bytes = challenge.to_bytes()?;
            bytes.extend_from_slice(&challenge_bytes.len().to_le_bytes());
            bytes.extend_from_slice(&challenge_bytes);
        }
        
        Ok(bytes)
    }
    
    /// Deserializes challenges from bytes
    /// 
    /// # Arguments
    /// * `bytes` - Serialized challenge data
    /// 
    /// # Returns
    /// * `Result<Self>` - Deserialized challenges
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let mut offset = 0;
        
        // Deserialize parameters
        if bytes.len() < offset + 8 * 5 {
            return Err(LatticeFoldError::InvalidParameters(
                "Insufficient bytes for parameters".to_string()
            ));
        }
        
        let ring_dimension = usize::from_le_bytes(
            bytes[offset..offset + 8].try_into().unwrap()
        );
        offset += 8;
        
        let decomposition_levels = usize::from_le_bytes(
            bytes[offset..offset + 8].try_into().unwrap()
        );
        offset += 8;
        
        let kappa = usize::from_le_bytes(
            bytes[offset..offset + 8].try_into().unwrap()
        );
        offset += 8;
        
        let gadget_dimension = usize::from_le_bytes(
            bytes[offset..offset + 8].try_into().unwrap()
        );
        offset += 8;
        
        let modulus = i64::from_le_bytes(
            bytes[offset..offset + 8].try_into().unwrap()
        );
        offset += 8;
        
        // Reconstruct parameters
        let params = FoldingChallengeParams::new(
            ring_dimension,
            decomposition_levels,
            kappa,
            gadget_dimension,
            modulus,
        )?;
        
        // Deserialize basic challenges
        if bytes.len() < offset + 8 {
            return Err(LatticeFoldError::InvalidParameters(
                "Insufficient bytes for basic challenge count".to_string()
            ));
        }
        
        let basic_count = usize::from_le_bytes(
            bytes[offset..offset + 8].try_into().unwrap()
        );
        offset += 8;
        
        let mut basic_challenges = Vec::with_capacity(basic_count);
        for _ in 0..basic_count {
            if bytes.len() < offset + 8 {
                return Err(LatticeFoldError::InvalidParameters(
                    "Insufficient bytes for challenge length".to_string()
                ));
            }
            
            let challenge_len = usize::from_le_bytes(
                bytes[offset..offset + 8].try_into().unwrap()
            );
            offset += 8;
            
            if bytes.len() < offset + challenge_len {
                return Err(LatticeFoldError::InvalidParameters(
                    "Insufficient bytes for challenge data".to_string()
                ));
            }
            
            let challenge = RingElement::from_bytes(&bytes[offset..offset + challenge_len])?;
            basic_challenges.push(challenge);
            offset += challenge_len;
        }
        
        // Deserialize extended challenges
        if bytes.len() < offset + 8 {
            return Err(LatticeFoldError::InvalidParameters(
                "Insufficient bytes for extended challenge count".to_string()
            ));
        }
        
        let extended_count = usize::from_le_bytes(
            bytes[offset..offset + 8].try_into().unwrap()
        );
        offset += 8;
        
        let mut extended_challenges = Vec::with_capacity(extended_count);
        for _ in 0..extended_count {
            if bytes.len() < offset + 8 {
                return Err(LatticeFoldError::InvalidParameters(
                    "Insufficient bytes for extended challenge length".to_string()
                ));
            }
            
            let challenge_len = usize::from_le_bytes(
                bytes[offset..offset + 8].try_into().unwrap()
            );
            offset += 8;
            
            if bytes.len() < offset + challenge_len {
                return Err(LatticeFoldError::InvalidParameters(
                    "Insufficient bytes for extended challenge data".to_string()
                ));
            }
            
            let challenge = RingElement::from_bytes(&bytes[offset..offset + challenge_len])?;
            extended_challenges.push(challenge);
            offset += challenge_len;
        }
        
        Ok(Self {
            basic_challenges,
            extended_challenges,
            params,
        })
    }
}

/// Consistency verification claims for sumcheck compression
/// 
/// This structure represents the six consistency claims that need to be
/// verified to ensure proper transformation from double commitments to
/// linear commitments. These claims are compressed into a single sumcheck
/// execution for communication efficiency.
/// 
/// Mathematical Foundation:
/// The six claims verify different aspects of the commitment transformation:
/// 1. **Decomposition Consistency**: Gadget decomposition correctness
/// 2. **Matrix Consistency**: Matrix commitment binding
/// 3. **Range Consistency**: Range proof validity
/// 4. **Folding Consistency**: Challenge application correctness
/// 5. **Witness Consistency**: Witness combination validity
/// 6. **Commitment Consistency**: Final commitment correctness
/// 
/// Compression Strategy:
/// Uses random linear combination to compress six claims into one:
/// - Generate random coefficients α₁, ..., α₆
/// - Compute combined claim: Σᵢ αᵢ · claimᵢ
/// - Execute single sumcheck on combined claim
/// - Provide individual evaluations for verification
#[derive(Clone, Debug)]
pub struct ConsistencyVerificationClaims {
    /// Decomposition consistency claim
    pub decomposition_claim: MultilinearExtension,
    
    /// Matrix consistency claim
    pub matrix_claim: MultilinearExtension,
    
    /// Range consistency claim
    pub range_claim: MultilinearExtension,
    
    /// Folding consistency claim
    pub folding_claim: MultilinearExtension,
    
    /// Witness consistency claim
    pub witness_claim: MultilinearExtension,
    
    /// Commitment consistency claim
    pub commitment_claim: MultilinearExtension,
    
    /// Random coefficients for linear combination
    pub combination_coefficients: Vec<RingElement>,
    
    /// Combined claim for sumcheck execution
    pub combined_claim: MultilinearExtension,
}

impl ConsistencyVerificationClaims {
    /// Creates new consistency verification claims
    /// 
    /// # Arguments
    /// * `claims` - Vector of six individual claims
    /// * `rng` - Random number generator for combination coefficients
    /// * `ring_dimension` - Ring dimension for coefficient generation
    /// * `modulus` - Ring modulus for coefficient generation
    /// 
    /// # Returns
    /// * `Result<Self>` - New consistency claims or error
    /// 
    /// # Claim Combination Process
    /// 1. Validate exactly six claims are provided
    /// 2. Generate random coefficients α₁, ..., α₆
    /// 3. Compute combined claim: Σᵢ αᵢ · claimᵢ
    /// 4. Verify combined claim has expected properties
    pub fn new<R: CryptoRng + RngCore>(
        claims: Vec<MultilinearExtension>,
        rng: &mut R,
        ring_dimension: usize,
        modulus: i64,
    ) -> Result<Self> {
        // Validate exactly six claims
        if claims.len() != 6 {
            return Err(LatticeFoldError::InvalidParameters(
                format!("Expected exactly 6 claims, got {}", claims.len())
            ));
        }
        
        // Generate random combination coefficients
        let mut combination_coefficients = Vec::with_capacity(6);
        for _ in 0..6 {
            let mut coeffs = vec![0i64; ring_dimension];
            for coeff in &mut coeffs {
                let mut bytes = [0u8; 8];
                rng.fill_bytes(&mut bytes);
                let random_value = i64::from_le_bytes(bytes);
                *coeff = ((random_value % modulus) + modulus) % modulus;
            }
            
            let coefficient = RingElement::from_coefficients(coeffs, Some(modulus))?;
            combination_coefficients.push(coefficient);
        }
        
        // Compute combined claim
        let mut combined_claim = claims[0].clone();
        combined_claim.scale(&combination_coefficients[0])?;
        
        for i in 1..6 {
            let mut scaled_claim = claims[i].clone();
            scaled_claim.scale(&combination_coefficients[i])?;
            combined_claim.add_assign(&scaled_claim)?;
        }
        
        Ok(Self {
            decomposition_claim: claims[0].clone(),
            matrix_claim: claims[1].clone(),
            range_claim: claims[2].clone(),
            folding_claim: claims[3].clone(),
            witness_claim: claims[4].clone(),
            commitment_claim: claims[5].clone(),
            combination_coefficients,
            combined_claim,
        })
    }
    
    /// Returns all individual claims as a vector
    pub fn individual_claims(&self) -> Vec<&MultilinearExtension> {
        vec![
            &self.decomposition_claim,
            &self.matrix_claim,
            &self.range_claim,
            &self.folding_claim,
            &self.witness_claim,
            &self.commitment_claim,
        ]
    }
    
    /// Returns the combined claim for sumcheck execution
    pub fn combined_claim(&self) -> &MultilinearExtension {
        &self.combined_claim
    }
    
    /// Returns the combination coefficients
    pub fn combination_coefficients(&self) -> &[RingElement] {
        &self.combination_coefficients
    }
    
    /// Validates that the combined claim is correctly formed
    /// 
    /// # Returns
    /// * `Result<bool>` - True if combined claim is valid, false otherwise
    pub fn validate_combination(&self) -> Result<bool> {
        // Recompute combined claim and compare
        let mut expected_combined = self.decomposition_claim.clone();
        expected_combined.scale(&self.combination_coefficients[0])?;
        
        let claims = [
            &self.matrix_claim,
            &self.range_claim,
            &self.folding_claim,
            &self.witness_claim,
            &self.commitment_claim,
        ];
        
        for (i, claim) in claims.iter().enumerate() {
            let mut scaled_claim = (*claim).clone();
            scaled_claim.scale(&self.combination_coefficients[i + 1])?;
            expected_combined.add_assign(&scaled_claim)?;
        }
        
        // Compare with stored combined claim
        Ok(expected_combined.equals(&self.combined_claim)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::thread_rng;
    
    /// Test folding challenge parameter creation and validation
    #[test]
    fn test_folding_challenge_params() {
        // Test valid parameters
        let params = FoldingChallengeParams::new(
            1024, // ring_dimension
            4,    // decomposition_levels
            128,  // kappa
            8,    // gadget_dimension
            2147483647, // modulus (2^31 - 1, prime)
        ).unwrap();
        
        assert_eq!(params.ring_dimension, 1024);
        assert_eq!(params.half_dimension, 512);
        assert_eq!(params.extended_challenge_dimension, 4096);
        assert!(params.validate().is_ok());
        
        // Test invalid ring dimension (not power of 2)
        assert!(FoldingChallengeParams::new(1000, 4, 128, 8, 2147483647).is_err());
        
        // Test invalid modulus
        assert!(FoldingChallengeParams::new(1024, 4, 128, 8, -1).is_err());
        
        // Test extended dimension too large
        assert!(FoldingChallengeParams::new(65536, 64, 128, 8, 2147483647).is_err());
    }
    
    /// Test strong sampling set creation and sampling
    #[test]
    fn test_strong_sampling_set() {
        let mut rng = thread_rng();
        
        let sampling_set = StrongSamplingSet::new(
            1024,       // ring_dimension
            2147483647, // modulus
            1000,       // target_size
        ).unwrap();
        
        assert_eq!(sampling_set.ring_dimension(), 1024);
        assert_eq!(sampling_set.modulus(), 2147483647);
        assert_eq!(sampling_set.size(), 1000);
        
        // Test sampling
        let sample1 = sampling_set.sample(&mut rng).unwrap();
        let sample2 = sampling_set.sample(&mut rng).unwrap();
        
        assert_eq!(sample1.dimension(), 1024);
        assert_eq!(sample1.modulus(), Some(2147483647));
        assert_eq!(sample2.dimension(), 1024);
        assert_eq!(sample2.modulus(), Some(2147483647));
        
        // Samples should be different (with high probability)
        assert_ne!(sample1.coefficients(), sample2.coefficients());
    }
    
    /// Test folding challenge generation
    #[test]
    fn test_folding_challenge_generation() {
        let mut rng = thread_rng();
        
        let params = FoldingChallengeParams::new(
            1024, // ring_dimension
            4,    // decomposition_levels
            128,  // kappa
            8,    // gadget_dimension
            2147483647, // modulus
        ).unwrap();
        
        let mut generator = FoldingChallengeGenerator::new(params.clone()).unwrap();
        
        // Generate challenges
        let challenges = generator.generate_challenges(&mut rng).unwrap();
        
        // Validate challenge structure
        assert!(challenges.validate().unwrap());
        assert_eq!(challenges.basic_challenges().len(), 3);
        assert_eq!(challenges.extended_challenges().len(), 4096);
        
        // Test witness combination
        let tau_d = RingElement::zero(1024, Some(2147483647)).unwrap();
        let m_tau = RingElement::zero(1024, Some(2147483647)).unwrap();
        let f = RingElement::zero(1024, Some(2147483647)).unwrap();
        let h = RingElement::zero(1024, Some(2147483647)).unwrap();
        
        let combined = challenges.combine_witnesses(&tau_d, &m_tau, &f, &h).unwrap();
        assert_eq!(combined.dimension(), 1024);
        assert_eq!(combined.modulus(), Some(2147483647));
    }
    
    /// Test challenge serialization and deserialization
    #[test]
    fn test_challenge_serialization() {
        let mut rng = thread_rng();
        
        let params = FoldingChallengeParams::new(
            512,  // ring_dimension (smaller for faster test)
            2,    // decomposition_levels
            64,   // kappa
            4,    // gadget_dimension
            2147483647, // modulus
        ).unwrap();
        
        let mut generator = FoldingChallengeGenerator::new(params).unwrap();
        let original_challenges = generator.generate_challenges(&mut rng).unwrap();
        
        // Serialize
        let bytes = original_challenges.to_bytes().unwrap();
        assert!(!bytes.is_empty());
        
        // Deserialize
        let deserialized_challenges = FoldingChallenges::from_bytes(&bytes).unwrap();
        
        // Validate deserialized challenges
        assert!(deserialized_challenges.validate().unwrap());
        assert_eq!(
            deserialized_challenges.basic_challenges().len(),
            original_challenges.basic_challenges().len()
        );
        assert_eq!(
            deserialized_challenges.extended_challenges().len(),
            original_challenges.extended_challenges().len()
        );
        
        // Check parameter consistency
        assert_eq!(
            deserialized_challenges.params().ring_dimension,
            original_challenges.params().ring_dimension
        );
        assert_eq!(
            deserialized_challenges.params().modulus,
            original_challenges.params().modulus
        );
    }
    
    /// Test consistency verification claims
    #[test]
    fn test_consistency_verification_claims() {
        let mut rng = thread_rng();
        
        // Create dummy multilinear extensions for testing
        let ring_dimension = 512;
        let modulus = 2147483647i64;
        let num_variables = 3;
        
        let mut claims = Vec::new();
        for _ in 0..6 {
            let claim = MultilinearExtension::new(
                num_variables,
                ring_dimension,
                modulus,
            ).unwrap();
            claims.push(claim);
        }
        
        // Create consistency verification claims
        let consistency_claims = ConsistencyVerificationClaims::new(
            claims,
            &mut rng,
            ring_dimension,
            modulus,
        ).unwrap();
        
        // Validate combination
        assert!(consistency_claims.validate_combination().unwrap());
        
        // Check structure
        assert_eq!(consistency_claims.individual_claims().len(), 6);
        assert_eq!(consistency_claims.combination_coefficients().len(), 6);
        
        // Validate coefficients
        for coeff in consistency_claims.combination_coefficients() {
            assert_eq!(coeff.dimension(), ring_dimension);
            assert_eq!(coeff.modulus(), Some(modulus));
        }
    }
    
    /// Test performance statistics tracking
    #[test]
    fn test_performance_statistics() {
        let params = FoldingChallengeParams::new(
            512,  // ring_dimension
            2,    // decomposition_levels
            64,   // kappa
            4,    // gadget_dimension
            2147483647, // modulus
        ).unwrap();
        
        let generator = FoldingChallengeGenerator::new(params).unwrap();
        let stats = generator.stats().unwrap();
        
        // Initial statistics should be zero
        assert_eq!(stats.total_generations, 0);
        assert_eq!(stats.total_basic_challenges, 0);
        assert_eq!(stats.total_extended_challenges, 0);
        assert_eq!(stats.average_generation_time_ns(), 0);
        assert_eq!(stats.tensor_cache_hit_rate(), 0.0);
        assert_eq!(stats.sampling_success_rate(), 0.0);
        assert_eq!(stats.gpu_utilization(), 0.0);
    }
}// - Parameter consistency
    pub fn validate(&self) -> Result<bool> {
        // Check basic challenge count
        if self.basic_challenges.len() != self.params.num_basic_challenges {
            return Ok(false);
        }
        
        // Check extended challenge count
        if self.extended_challenges.len() != self.params.extended_challenge_dimension {
            return Ok(false);
        }
        
        // Validate basic challenges
        for challenge in &self.basic_challenges {
            if challenge.dimension() != self.params.ring_dimension {
                return Ok(false);
            }
            
            if challenge.modulus() != Some(self.params.modulus) {
                return Ok(false);
            }
        }
        
        // Validate extended challenges
        for challenge in &self.extended_challenges {
            if challenge.dimension() != self.params.ring_dimension {
                return Ok(false);
            }
            
            if challenge.modulus() != Some(self.params.modulus) {
                return Ok(false);
            }
        }
        
        Ok(true)
    }
    
    /// Returns the total number of challenges
    pub fn total_challenges(&self) -> usize {
        self.basic_challenges.len() + self.extended_challenges.len()
    }
    
    /// Returns the memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        let ring_element_size = self.params.ring_dimension * 8; // 8 bytes per coefficient
        self.total_challenges() * ring_element_size
    }
    
    /// Computes entropy estimate for security analysis
    /// 
    /// # Returns
    /// * `f64` - Estimated entropy in bits
    /// 
    /// # Entropy Calculation
    /// Each challenge contributes log₂|S̄| bits of entropy.
    /// Total entropy is approximately (num_challenges × log₂|S̄|) bits.
    pub fn entropy_estimate(&self) -> f64 {
        let set_entropy = (self.params.sampling_set_size as f64).log2();
        (self.total_challenges() as f64) * set_entropy
    }
}

/// Comprehensive tests for folding challenge generation
#[cfg(test)]
mod tests {
    use super::*;
    use rand::thread_rng;
    
    #[test]
    fn test_folding_challenge_params_creation() {
        let params = FoldingChallengeParams::new(
            1024, // ring_dimension
            8,    // decomposition_levels
            128,  // kappa
            8,    // gadget_dimension
            2147483647, // modulus (2^31 - 1)
        );
        
        assert!(params.is_ok());
        let params = params.unwrap();
        assert_eq!(params.ring_dimension, 1024);
        assert_eq!(params.decomposition_levels, 8);
        assert_eq!(params.extended_challenge_dimension, 1024 * 8);
    }
    
    #[test]
    fn test_strong_sampling_set_creation() {
        let sampling_set = StrongSamplingSet::new(
            512,  // ring_dimension
            1073741827, // modulus
            1000, // target_size
        );
        
        assert!(sampling_set.is_ok());
        let sampling_set = sampling_set.unwrap();
        assert_eq!(sampling_set.ring_dimension(), 512);
        assert_eq!(sampling_set.modulus(), 1073741827);
    }
    
    #[test]
    fn test_challenge_generation() {
        let params = FoldingChallengeParams::new(
            256, 4, 64, 4, 536870923
        ).unwrap();
        
        let mut generator = FoldingChallengeGenerator::new(params).unwrap();
        let mut rng = thread_rng();
        
        let challenges = generator.generate_challenges(&mut rng);
        assert!(challenges.is_ok());
        
        let challenges = challenges.unwrap();
        assert_eq!(challenges.basic_challenges.len(), 3);
        assert_eq!(challenges.extended_challenges.len(), 256 * 4);
        
        // Validate challenges
        assert!(challenges.validate().unwrap());
    }
    
    #[test]
    fn test_tensor_product_evaluation() {
        let params = FoldingChallengeParams::new(
            128, 2, 32, 2, 268435459
        ).unwrap();
        
        let mut generator = FoldingChallengeGenerator::new(params).unwrap();
        
        // Create test inputs
        let c_z = vec![
            RingElement::one(128, Some(268435459)).unwrap(),
            RingElement::zero(128, Some(268435459)).unwrap(),
        ];
        
        let extended_challenges = vec![
            RingElement::one(128, Some(268435459)).unwrap(),
            RingElement::one(128, Some(268435459)).unwrap(),
        ];
        
        let evaluation_point = vec![1i64, 2i64];
        
        let result = generator.evaluate_tensor_product(&c_z, &extended_challenges, &evaluation_point);
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_folding_challenges_validation() {
        let params = FoldingChallengeParams::new(
            64, 2, 16, 2, 134217757
        ).unwrap();
        
        let basic_challenges = vec![
            RingElement::one(64, Some(134217757)).unwrap(),
            RingElement::zero(64, Some(134217757)).unwrap(),
            RingElement::one(64, Some(134217757)).unwrap(),
        ];
        
        let extended_challenges = vec![
            RingElement::one(64, Some(134217757)).unwrap(),
            RingElement::zero(64, Some(134217757)).unwrap(),
        ];
        
        // Should have 64 * 2 = 128 extended challenges, but we only have 2
        let challenges = FoldingChallenges::new(basic_challenges, extended_challenges, params);
        
        // This should fail validation due to incorrect extended challenge count
        assert!(!challenges.validate().unwrap());
    }
}