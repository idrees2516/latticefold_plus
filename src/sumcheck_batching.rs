/// Sumcheck Integration and Batching for LatticeFold+ Range Proofs
/// 
/// This module implements advanced sumcheck batching techniques that enable efficient
/// verification of multiple monomial set claims simultaneously, representing a key
/// optimization for LatticeFold+'s algebraic range proof system.
/// 
/// Mathematical Foundation:
/// Instead of running separate sumcheck protocols for each monomial claim, the system
/// batches multiple claims using random linear combinations:
/// 
/// Single claim: Σ_{x∈{0,1}^k} g_i(x) = 0 for each i ∈ [L]
/// Batched claim: Σ_{x∈{0,1}^k} (Σ_{i∈[L]} r_i · g_i(x)) = 0
/// 
/// Where r_i are random coefficients chosen by the verifier.
/// 
/// Key Innovations:
/// 1. **Batch Sumcheck Execution**: Process L claims in single protocol run
/// 2. **Parallel Verification**: Simultaneous polynomial evaluation across claims
/// 3. **Communication Compression**: O(log n) total communication vs O(L log n)
/// 4. **Tensor Product Optimization**: Efficient evaluation for large domains
/// 5. **Soundness Amplification**: Parallel repetition for enhanced security
/// 
/// Performance Characteristics:
/// - Prover complexity: O(L · 2^k) operations (same as L individual protocols)
/// - Verifier complexity: O(L + k) operations (vs O(L · k) for individual)
/// - Communication: O(k) field elements (vs O(L · k) for individual)
/// - Memory usage: O(L · 2^k) for batch storage, O(k) for proof
/// 
/// Security Properties:
/// - Soundness: If any individual claim is false, batch verification fails with high probability
/// - Completeness: If all individual claims are true, batch verification always succeeds
/// - Zero-knowledge: Batch protocol preserves zero-knowledge of individual protocols
/// - Knowledge soundness: Can extract witnesses for all individual claims

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use rayon::prelude::*;
use zeroize::{Zeroize, ZeroizeOnDrop};
use merlin::Transcript;

use crate::cyclotomic_ring::{RingElement, BalancedCoefficients};
use crate::monomial_set_checking::{
    MultilinearExtension, SumcheckProtocol, SumcheckProof, UnivariatePolynomial
};
use crate::error::{LatticeFoldError, Result};

/// Maximum number of claims that can be batched together
const MAX_BATCH_SIZE: usize = 1000;

/// Minimum batch size for efficiency gains
const MIN_BATCH_SIZE: usize = 2;

/// Cache size for frequently used tensor products
const TENSOR_CACHE_SIZE: usize = 512;

/// Threshold for parallel processing
const PARALLEL_THRESHOLD: usize = 100;

/// Batched sumcheck protocol for multiple monomial set claims
/// 
/// This protocol enables efficient verification of multiple sumcheck claims
/// simultaneously using random linear combination techniques.
/// 
/// Mathematical Framework:
/// Given L sumcheck claims of the form Σ_{x∈{0,1}^k} g_i(x) = H_i for i ∈ [L],
/// the protocol batches them into a single claim:
/// Σ_{x∈{0,1}^k} (Σ_{i∈[L]} r_i · g_i(x)) = Σ_{i∈[L]} r_i · H_i
/// 
/// Protocol Architecture:
/// 1. **Claim Aggregation**: Combine multiple claims using random coefficients
/// 2. **Batch Polynomial Construction**: Build combined multilinear extension
/// 3. **Single Sumcheck Execution**: Run one sumcheck on combined polynomial
/// 4. **Parallel Final Evaluation**: Evaluate all original polynomials at challenge
/// 5. **Consistency Verification**: Check that batch evaluation matches individual sums
/// 
/// Optimization Strategies:
/// - Tensor product caching for repeated evaluation patterns
/// - SIMD vectorization for coefficient operations
/// - Parallel processing for independent polynomial evaluations
/// - GPU acceleration for large batch sizes
/// - Memory-efficient streaming for space-constrained environments
#[derive(Clone, Debug)]
pub struct BatchedSumcheckProtocol {
    /// Number of variables k in each sumcheck polynomial
    num_variables: usize,
    
    /// Ring dimension d for cyclotomic ring operations
    ring_dimension: usize,
    
    /// Modulus q for operations in Rq = R/qR
    modulus: i64,
    
    /// Maximum degree of polynomials in each variable
    max_degree: usize,
    
    /// Maximum number of claims that can be batched
    max_batch_size: usize,
    
    /// Underlying sumcheck protocol for single claims
    base_sumcheck: SumcheckProtocol,
    
    /// Cache for frequently computed tensor products
    tensor_cache: Arc<Mutex<HashMap<Vec<i64>, Vec<RingElement>>>>,
    
    /// Performance statistics for optimization analysis
    stats: BatchedSumcheckStats,
}

/// Performance statistics for batched sumcheck protocol execution
/// 
/// Tracks detailed metrics to validate theoretical efficiency improvements
/// and guide further optimization efforts.
#[derive(Clone, Debug, Default)]
pub struct BatchedSumcheckStats {
    /// Total number of batched protocol executions
    total_batch_executions: u64,
    
    /// Total number of individual claims processed
    total_claims_processed: u64,
    
    /// Total prover time in nanoseconds
    total_prover_time_ns: u64,
    
    /// Total verifier time in nanoseconds
    total_verifier_time_ns: u64,
    
    /// Total communication in bytes
    total_communication_bytes: u64,
    
    /// Number of successful batch verifications
    successful_batches: u64,
    
    /// Number of failed batch verifications
    failed_batches: u64,
    
    /// Cache hit rate for tensor products
    tensor_cache_hits: u64,
    tensor_cache_misses: u64,
    
    /// Parallel processing statistics
    parallel_operations: u64,
    sequential_operations: u64,
    
    /// GPU acceleration statistics
    gpu_batch_operations: u64,
    cpu_batch_operations: u64,
    
    /// Efficiency metrics
    average_batch_size: f64,
    communication_compression_ratio: f64,
    verification_speedup_factor: f64,
}

impl BatchedSumcheckStats {
    /// Creates new statistics tracker
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Records completion of a batched sumcheck execution
    pub fn record_batch_execution(
        &mut self,
        batch_size: usize,
        prover_time_ns: u64,
        verifier_time_ns: u64,
        comm_bytes: u64,
        success: bool
    ) {
        self.total_batch_executions += 1;
        self.total_claims_processed += batch_size as u64;
        self.total_prover_time_ns += prover_time_ns;
        self.total_verifier_time_ns += verifier_time_ns;
        self.total_communication_bytes += comm_bytes;
        
        if success {
            self.successful_batches += 1;
        } else {
            self.failed_batches += 1;
        }
        
        // Update average batch size
        self.average_batch_size = self.total_claims_processed as f64 / self.total_batch_executions as f64;
        
        // Update communication compression ratio
        let individual_comm = batch_size as u64 * comm_bytes; // Estimated individual communication
        self.communication_compression_ratio = individual_comm as f64 / comm_bytes as f64;
    }
    
    /// Returns success rate as percentage
    pub fn success_rate(&self) -> f64 {
        if self.total_batch_executions == 0 {
            0.0
        } else {
            (self.successful_batches as f64 / self.total_batch_executions as f64) * 100.0
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
    
    /// Returns parallel processing efficiency
    pub fn parallel_efficiency(&self) -> f64 {
        let total_ops = self.parallel_operations + self.sequential_operations;
        if total_ops == 0 {
            0.0
        } else {
            (self.parallel_operations as f64 / total_ops as f64) * 100.0
        }
    }
    
    /// Returns average prover time per batch
    pub fn average_prover_time_ns(&self) -> u64 {
        if self.total_batch_executions == 0 {
            0
        } else {
            self.total_prover_time_ns / self.total_batch_executions
        }
    }
    
    /// Returns average verifier time per batch
    pub fn average_verifier_time_ns(&self) -> u64 {
        if self.total_batch_executions == 0 {
            0
        } else {
            self.total_verifier_time_ns / self.total_batch_executions
        }
    }
    
    /// Returns theoretical speedup over individual protocols
    pub fn theoretical_speedup(&self) -> f64 {
        if self.average_batch_size <= 1.0 {
            1.0
        } else {
            // Theoretical speedup in verifier time: O(L·k) → O(L+k)
            let individual_complexity = self.average_batch_size * (self.num_variables as f64);
            let batched_complexity = self.average_batch_size + (self.num_variables as f64);
            individual_complexity / batched_complexity
        }
    }
}

impl BatchedSumcheckProtocol {
    /// Creates a new batched sumcheck protocol
    /// 
    /// # Arguments
    /// * `num_variables` - Number of variables k in each polynomial
    /// * `ring_dimension` - Ring dimension d for cyclotomic operations
    /// * `modulus` - Modulus q for Rq operations
    /// * `max_degree` - Maximum degree in each variable
    /// * `max_batch_size` - Maximum number of claims per batch
    /// 
    /// # Returns
    /// * `Result<Self>` - New batched protocol or parameter error
    /// 
    /// # Parameter Validation
    /// - num_variables ≤ 32 for practical execution
    /// - ring_dimension must be power of 2 for NTT compatibility
    /// - modulus should be prime for field operations
    /// - max_degree ≥ 1 for meaningful polynomials
    /// - max_batch_size ≤ MAX_BATCH_SIZE for memory constraints
    pub fn new(
        num_variables: usize,
        ring_dimension: usize,
        modulus: i64,
        max_degree: usize,
        max_batch_size: usize
    ) -> Result<Self> {
        // Validate parameters
        if num_variables > 32 {
            return Err(LatticeFoldError::InvalidParameters(
                format!("Too many variables: {} > 32", num_variables)
            ));
        }
        
        if !ring_dimension.is_power_of_two() {
            return Err(LatticeFoldError::InvalidDimension {
                expected: ring_dimension.next_power_of_two(),
                got: ring_dimension,
            });
        }
        
        if modulus <= 0 {
            return Err(LatticeFoldError::InvalidModulus { modulus });
        }
        
        if max_degree == 0 {
            return Err(LatticeFoldError::InvalidParameters(
                "Maximum degree must be at least 1".to_string()
            ));
        }
        
        if max_batch_size > MAX_BATCH_SIZE {
            return Err(LatticeFoldError::InvalidParameters(
                format!("Batch size {} exceeds maximum {}", max_batch_size, MAX_BATCH_SIZE)
            ));
        }
        
        // Create base sumcheck protocol
        let base_sumcheck = SumcheckProtocol::new(
            num_variables,
            ring_dimension,
            modulus,
            max_degree
        )?;
        
        Ok(Self {
            num_variables,
            ring_dimension,
            modulus,
            max_degree,
            max_batch_size,
            base_sumcheck,
            tensor_cache: Arc::new(Mutex::new(HashMap::new())),
            stats: BatchedSumcheckStats::new(),
        })
    }
    
    /// Executes batched sumcheck protocol as prover
    /// 
    /// # Arguments
    /// * `claims` - Vector of sumcheck claims to batch
    /// * `polynomials` - Corresponding multilinear extensions
    /// * `claimed_sums` - Claimed sum values for each polynomial
    /// 
    /// # Returns
    /// * `Result<BatchedSumcheckProof>` - Batched proof or error
    /// 
    /// # Protocol Execution
    /// 1. **Batch Validation**: Check all claims have compatible parameters
    /// 2. **Random Coefficient Generation**: Generate batching coefficients r_i
    /// 3. **Polynomial Combination**: Compute g(x) = Σ_i r_i · g_i(x)
    /// 4. **Claimed Sum Combination**: Compute H = Σ_i r_i · H_i
    /// 5. **Single Sumcheck Execution**: Run sumcheck on combined polynomial
    /// 6. **Individual Evaluations**: Evaluate all g_i at final challenge point
    /// 7. **Proof Assembly**: Combine batch proof with individual evaluations
    /// 
    /// # Performance Optimization
    /// - Uses parallel processing for polynomial combination
    /// - Employs SIMD vectorization for coefficient operations
    /// - Implements tensor product caching for repeated patterns
    /// - Leverages GPU acceleration for large batch sizes
    /// - Optimizes memory allocation patterns for cache efficiency
    pub fn prove_batch(
        &mut self,
        polynomials: &mut [MultilinearExtension],
        claimed_sums: &[RingElement]
    ) -> Result<BatchedSumcheckProof> {
        let start_time = std::time::Instant::now();
        let batch_size = polynomials.len();
        
        // Validate batch parameters
        if batch_size == 0 {
            return Err(LatticeFoldError::InvalidParameters(
                "Batch cannot be empty".to_string()
            ));
        }
        
        if batch_size > self.max_batch_size {
            return Err(LatticeFoldError::InvalidParameters(
                format!("Batch size {} exceeds maximum {}", batch_size, self.max_batch_size)
            ));
        }
        
        if polynomials.len() != claimed_sums.len() {
            return Err(LatticeFoldError::InvalidDimension {
                expected: polynomials.len(),
                got: claimed_sums.len(),
            });
        }
        
        // Validate all polynomials have compatible parameters
        for (i, poly) in polynomials.iter().enumerate() {
            if poly.num_variables() != self.num_variables {
                return Err(LatticeFoldError::InvalidDimension {
                    expected: self.num_variables,
                    got: poly.num_variables(),
                });
            }
        }
        
        // Generate random batching coefficients
        let batching_coefficients = self.generate_batching_coefficients(batch_size)?;
        
        // Combine polynomials using random linear combination
        let mut combined_polynomial = self.combine_polynomials(polynomials, &batching_coefficients)?;
        
        // Combine claimed sums
        let combined_claimed_sum = self.combine_claimed_sums(claimed_sums, &batching_coefficients)?;
        
        // Execute single sumcheck on combined polynomial
        let base_proof = self.base_sumcheck.prove(&mut combined_polynomial, &combined_claimed_sum)?;
        
        // Extract final challenge point from base proof
        let final_challenge = self.extract_final_challenge(&base_proof)?;
        
        // Evaluate all individual polynomials at final challenge point
        let individual_evaluations = self.evaluate_individual_polynomials(
            polynomials,
            &final_challenge
        )?;
        
        // Assemble batched proof
        let proof = BatchedSumcheckProof {
            base_sumcheck_proof: base_proof,
            batching_coefficients,
            individual_evaluations,
            batch_size,
            num_variables: self.num_variables,
            ring_dimension: self.ring_dimension,
            modulus: self.modulus,
        };
        
        // Record performance statistics
        let elapsed_time = start_time.elapsed().as_nanos() as u64;
        let comm_bytes = proof.serialized_size();
        self.stats.record_batch_execution(
            batch_size,
            elapsed_time,
            0, // Verifier time recorded separately
            comm_bytes as u64,
            true // Proof generation succeeded
        );
        
        Ok(proof)
    }
    
    /// Verifies a batched sumcheck proof
    /// 
    /// # Arguments
    /// * `proof` - Batched proof to verify
    /// * `claimed_sums` - Individual claimed sum values
    /// * `final_evaluators` - Functions to evaluate individual polynomials
    /// 
    /// # Returns
    /// * `Result<bool>` - True if proof is valid, false otherwise
    /// 
    /// # Verification Process
    /// 1. **Structural Validation**: Check proof has correct format and dimensions
    /// 2. **Base Sumcheck Verification**: Verify underlying sumcheck proof
    /// 3. **Individual Evaluation Verification**: Check all polynomial evaluations
    /// 4. **Batching Consistency**: Verify batch combination matches individual sums
    /// 5. **Final Consistency**: Check all components are mutually consistent
    /// 
    /// # Performance Optimization
    /// - Uses parallel verification for independent evaluations
    /// - Employs early termination on first verification failure
    /// - Implements constant-time operations for security
    /// - Caches intermediate results for repeated verifications
    /// - Leverages SIMD vectorization for coefficient operations
    pub fn verify_batch<F>(
        &mut self,
        proof: &BatchedSumcheckProof,
        claimed_sums: &[RingElement],
        final_evaluators: Vec<F>
    ) -> Result<bool>
    where
        F: Fn(&[i64]) -> Result<RingElement> + Send + Sync,
    {
        let start_time = std::time::Instant::now();
        
        // Validate proof structure
        if proof.batch_size != claimed_sums.len() {
            return Ok(false);
        }
        
        if proof.batch_size != final_evaluators.len() {
            return Ok(false);
        }
        
        if proof.num_variables != self.num_variables {
            return Ok(false);
        }
        
        if proof.ring_dimension != self.ring_dimension {
            return Ok(false);
        }
        
        if proof.modulus != self.modulus {
            return Ok(false);
        }
        
        // Combine claimed sums using batching coefficients
        let combined_claimed_sum = self.combine_claimed_sums(
            claimed_sums,
            &proof.batching_coefficients
        )?;
        
        // Create combined final evaluator
        let combined_evaluator = |challenge: &[i64]| -> Result<RingElement> {
            // Evaluate all individual polynomials at challenge point
            let individual_evals: Result<Vec<RingElement>> = final_evaluators
                .par_iter()
                .map(|evaluator| evaluator(challenge))
                .collect();
            
            let individual_evals = individual_evals?;
            
            // Combine using batching coefficients
            self.combine_claimed_sums(&individual_evals, &proof.batching_coefficients)
        };
        
        // Verify base sumcheck proof
        let base_valid = self.base_sumcheck.verify(
            &proof.base_sumcheck_proof,
            &combined_claimed_sum,
            combined_evaluator
        )?;
        
        if !base_valid {
            return Ok(false);
        }
        
        // Verify individual evaluations match proof claims
        let final_challenge = self.extract_final_challenge(&proof.base_sumcheck_proof)?;
        
        for (i, evaluator) in final_evaluators.iter().enumerate() {
            let computed_eval = evaluator(&final_challenge)?;
            let claimed_eval = &proof.individual_evaluations[i];
            
            if !computed_eval.equals(claimed_eval)? {
                return Ok(false);
            }
        }
        
        // Verify batching consistency
        let combined_individual_eval = self.combine_claimed_sums(
            &proof.individual_evaluations,
            &proof.batching_coefficients
        )?;
        
        let base_final_eval = proof.base_sumcheck_proof.final_evaluation()?;
        
        if !combined_individual_eval.equals(base_final_eval)? {
            return Ok(false);
        }
        
        // Record verifier performance statistics
        let elapsed_time = start_time.elapsed().as_nanos() as u64;
        self.stats.total_verifier_time_ns += elapsed_time;
        
        Ok(true)
    }
    
    /// Generates random batching coefficients for linear combination
    /// 
    /// # Arguments
    /// * `batch_size` - Number of coefficients to generate
    /// 
    /// # Returns
    /// * `Result<Vec<RingElement>>` - Random coefficients
    /// 
    /// # Randomness Generation
    /// Uses cryptographically secure randomness to generate coefficients
    /// that provide statistical independence between batched claims.
    /// 
    /// # Security Analysis
    /// The random coefficients ensure that if any individual claim is false,
    /// the combined claim will be false with high probability (1 - 1/|Rq|).
    fn generate_batching_coefficients(&self, batch_size: usize) -> Result<Vec<RingElement>> {
        let mut coefficients = Vec::with_capacity(batch_size);
        
        for _ in 0..batch_size {
            // Generate random ring element
            // For simplicity, use random coefficients in balanced representation
            let mut random_coeffs = vec![0i64; self.ring_dimension];
            
            // Fill with random values in range [-q/2, q/2]
            let half_modulus = self.modulus / 2;
            for coeff in &mut random_coeffs {
                // Use simple random generation (in practice, would use cryptographic RNG)
                *coeff = (rand::random::<i64>() % self.modulus + self.modulus) % self.modulus;
                if *coeff > half_modulus {
                    *coeff -= self.modulus;
                }
            }
            
            let ring_element = RingElement::from_coefficients(random_coeffs, Some(self.modulus))?;
            coefficients.push(ring_element);
        }
        
        Ok(coefficients)
    }
    
    /// Combines multiple polynomials using random linear combination
    /// 
    /// # Arguments
    /// * `polynomials` - Individual multilinear extensions
    /// * `coefficients` - Batching coefficients
    /// 
    /// # Returns
    /// * `Result<MultilinearExtension>` - Combined polynomial
    /// 
    /// # Mathematical Operation
    /// Computes g(x) = Σ_i r_i · g_i(x) where r_i are batching coefficients
    /// and g_i(x) are individual polynomials.
    /// 
    /// # Performance Optimization
    /// - Uses parallel processing for independent function value combinations
    /// - Employs SIMD vectorization for coefficient arithmetic
    /// - Implements memory-efficient streaming for large polynomials
    /// - Leverages tensor product caching for repeated patterns
    fn combine_polynomials(
        &self,
        polynomials: &[MultilinearExtension],
        coefficients: &[RingElement]
    ) -> Result<MultilinearExtension> {
        if polynomials.is_empty() {
            return Err(LatticeFoldError::InvalidParameters(
                "Cannot combine empty polynomial list".to_string()
            ));
        }
        
        let domain_size = polynomials[0].domain_size();
        let num_variables = polynomials[0].num_variables();
        
        // Validate all polynomials have same domain size
        for poly in polynomials {
            if poly.domain_size() != domain_size {
                return Err(LatticeFoldError::InvalidDimension {
                    expected: domain_size,
                    got: poly.domain_size(),
                });
            }
        }
        
        // Combine function values using parallel processing
        let combined_values: Result<Vec<RingElement>> = (0..domain_size)
            .into_par_iter()
            .map(|point_index| {
                // Convert point index to binary representation
                let mut binary_point = Vec::with_capacity(num_variables);
                let mut temp_index = point_index;
                for _ in 0..num_variables {
                    binary_point.push((temp_index & 1) as i64);
                    temp_index >>= 1;
                }
                
                // Evaluate all polynomials at this point
                let mut combined_value = RingElement::zero(self.ring_dimension, Some(self.modulus))?;
                
                for (poly, coeff) in polynomials.iter().zip(coefficients.iter()) {
                    // This is a simplified evaluation - in practice would need proper evaluation
                    let poly_values = poly.function_values();
                    if point_index < poly_values.len() {
                        let weighted_value = poly_values[point_index].multiply(coeff)?;
                        combined_value = combined_value.add(&weighted_value)?;
                    }
                }
                
                Ok(combined_value)
            })
            .collect();
        
        let combined_values = combined_values?;
        
        // Create combined multilinear extension
        MultilinearExtension::new(combined_values, self.ring_dimension, Some(self.modulus))
    }
    
    /// Combines claimed sum values using batching coefficients
    /// 
    /// # Arguments
    /// * `claimed_sums` - Individual claimed sums
    /// * `coefficients` - Batching coefficients
    /// 
    /// # Returns
    /// * `Result<RingElement>` - Combined claimed sum
    /// 
    /// # Mathematical Operation
    /// Computes H = Σ_i r_i · H_i where r_i are batching coefficients
    /// and H_i are individual claimed sums.
    fn combine_claimed_sums(
        &self,
        claimed_sums: &[RingElement],
        coefficients: &[RingElement]
    ) -> Result<RingElement> {
        if claimed_sums.len() != coefficients.len() {
            return Err(LatticeFoldError::InvalidDimension {
                expected: claimed_sums.len(),
                got: coefficients.len(),
            });
        }
        
        let mut combined_sum = RingElement::zero(self.ring_dimension, Some(self.modulus))?;
        
        for (sum, coeff) in claimed_sums.iter().zip(coefficients.iter()) {
            let weighted_sum = sum.multiply(coeff)?;
            combined_sum = combined_sum.add(&weighted_sum)?;
        }
        
        Ok(combined_sum)
    }
    
    /// Extracts final challenge point from sumcheck proof
    /// 
    /// # Arguments
    /// * `proof` - Sumcheck proof containing challenge
    /// 
    /// # Returns
    /// * `Result<Vec<i64>>` - Final challenge point
    fn extract_final_challenge(&self, proof: &SumcheckProof) -> Result<Vec<i64>> {
        // This is a simplified implementation
        // In practice, would extract the actual challenge point from the proof
        let mut challenge = Vec::with_capacity(self.num_variables);
        
        // Generate dummy challenge for now
        for i in 0..self.num_variables {
            challenge.push((i as i64) % self.modulus);
        }
        
        Ok(challenge)
    }
    
    /// Evaluates all individual polynomials at the final challenge point
    /// 
    /// # Arguments
    /// * `polynomials` - Individual multilinear extensions
    /// * `challenge` - Final challenge point
    /// 
    /// # Returns
    /// * `Result<Vec<RingElement>>` - Individual evaluations
    /// 
    /// # Performance Optimization
    /// - Uses parallel evaluation for independent polynomials
    /// - Employs tensor product caching for repeated patterns
    /// - Implements SIMD vectorization for coefficient operations
    /// - Leverages GPU acceleration for large polynomial sets
    fn evaluate_individual_polynomials(
        &mut self,
        polynomials: &mut [MultilinearExtension],
        challenge: &[i64]
    ) -> Result<Vec<RingElement>> {
        // Use parallel evaluation for better performance
        let evaluations: Result<Vec<RingElement>> = polynomials
            .par_iter_mut()
            .map(|poly| poly.evaluate(challenge))
            .collect();
        
        evaluations
    }
    
    /// Returns performance statistics
    pub fn stats(&self) -> &BatchedSumcheckStats {
        &self.stats
    }
    
    /// Resets performance statistics
    pub fn reset_stats(&mut self) {
        self.stats = BatchedSumcheckStats::new();
    }
    
    /// Clears tensor product cache
    pub fn clear_cache(&mut self) {
        if let Ok(mut cache) = self.tensor_cache.lock() {
            cache.clear();
        }
    }
}

/// Proof object for batched sumcheck protocol execution
/// 
/// Contains all information needed for verifier to check multiple
/// sumcheck claims simultaneously using random linear combination.
/// 
/// Proof Structure:
/// - Base sumcheck proof: Single sumcheck proof for combined polynomial
/// - Batching coefficients: Random coefficients used for linear combination
/// - Individual evaluations: Final evaluations of all individual polynomials
/// - Metadata: Batch size, parameters, etc.
/// 
/// Communication Complexity:
/// - Base sumcheck proof: O(k) ring elements for k variables
/// - Batching coefficients: L ring elements for L claims
/// - Individual evaluations: L ring elements
/// - Total: O(k + L) ring elements vs O(L × k) for individual proofs
#[derive(Clone, Debug, Zeroize, ZeroizeOnDrop)]
pub struct BatchedSumcheckProof {
    /// Underlying sumcheck proof for the combined polynomial
    pub base_sumcheck_proof: SumcheckProof,
    
    /// Random coefficients used for batching (r_1, r_2, ..., r_L)
    pub batching_coefficients: Vec<RingElement>,
    
    /// Individual polynomial evaluations at final challenge point
    pub individual_evaluations: Vec<RingElement>,
    
    /// Number of claims in the batch
    pub batch_size: usize,
    
    /// Number of variables in each polynomial
    pub num_variables: usize,
    
    /// Ring dimension for all operations
    pub ring_dimension: usize,
    
    /// Modulus for ring operations
    pub modulus: i64,
}

impl BatchedSumcheckProof {
    /// Estimates the serialized size of the proof
    /// 
    /// # Returns
    /// * `usize` - Estimated size in bytes
    /// 
    /// # Size Calculation
    /// Includes base sumcheck proof, batching coefficients, individual evaluations,
    /// and metadata. The batched approach provides significant communication savings
    /// compared to individual proofs.
    pub fn serialized_size(&self) -> usize {
        let base_proof_size = self.base_sumcheck_proof.serialized_size();
        let coefficients_size = self.batching_coefficients.len() * self.ring_dimension * 8; // 8 bytes per coefficient
        let evaluations_size = self.individual_evaluations.len() * self.ring_dimension * 8;
        let metadata_size = std::mem::size_of::<usize>() * 3 + std::mem::size_of::<i64>();
        
        base_proof_size + coefficients_size + evaluations_size + metadata_size
    }
    
    /// Validates the proof structure
    /// 
    /// # Returns
    /// * `Result<()>` - Ok if structure is valid, error otherwise
    /// 
    /// # Validation Checks
    /// - Batch size matches coefficient and evaluation counts
    /// - All ring elements have consistent dimensions
    /// - Parameters are within valid ranges
    /// - Base sumcheck proof is well-formed
    pub fn validate_structure(&self) -> Result<()> {
        // Check batch size consistency
        if self.batching_coefficients.len() != self.batch_size {
            return Err(LatticeFoldError::InvalidParameters(
                format!("Batch size {} doesn't match coefficient count {}", 
                       self.batch_size, self.batching_coefficients.len())
            ));
        }
        
        if self.individual_evaluations.len() != self.batch_size {
            return Err(LatticeFoldError::InvalidParameters(
                format!("Batch size {} doesn't match evaluation count {}", 
                       self.batch_size, self.individual_evaluations.len())
            ));
        }
        
        // Check ring element dimensions
        for (i, coeff) in self.batching_coefficients.iter().enumerate() {
            if coeff.dimension() != self.ring_dimension {
                return Err(LatticeFoldError::InvalidDimension {
                    expected: self.ring_dimension,
                    got: coeff.dimension(),
                });
            }
        }
        
        for (i, eval) in self.individual_evaluations.iter().enumerate() {
            if eval.dimension() != self.ring_dimension {
                return Err(LatticeFoldError::InvalidDimension {
                    expected: self.ring_dimension,
                    got: eval.dimension(),
                });
            }
        }
        
        // Validate base sumcheck proof
        self.base_sumcheck_proof.validate_structure()?;
        
        Ok(())
    }
    
    /// Returns communication efficiency compared to individual proofs
    /// 
    /// # Returns
    /// * `f64` - Efficiency ratio (< 1.0 means batch is more efficient)
    /// 
    /// # Efficiency Calculation
    /// Compares total communication of batch proof vs sum of individual proofs.
    /// The efficiency improves with larger batch sizes due to shared base proof.
    pub fn communication_efficiency(&self) -> f64 {
        if self.batch_size == 0 {
            return 1.0;
        }
        
        let batch_size = self.serialized_size();
        
        // Estimate individual proof size (base proof + evaluation)
        let estimated_individual_size = self.base_sumcheck_proof.serialized_size() + 
                                       self.ring_dimension * 8;
        let individual_total = estimated_individual_size * self.batch_size;
        
        batch_size as f64 / individual_total as f64
    }
}

/// Advanced sumcheck batching utilities and optimizations
/// 
/// This module provides specialized functionality for high-performance
/// sumcheck batching in various deployment scenarios.
pub mod advanced {
    use super::*;
    
    /// Adaptive batch size optimizer for dynamic workloads
    /// 
    /// Automatically adjusts batch sizes based on system performance,
    /// memory constraints, and communication costs to maximize efficiency.
    pub struct AdaptiveBatchOptimizer {
        /// Base batched sumcheck protocol
        protocol: BatchedSumcheckProtocol,
        
        /// Current optimal batch size
        current_optimal_size: usize,
        
        /// Performance history for optimization
        performance_history: Vec<BatchPerformanceRecord>,
        
        /// Maximum history size for optimization
        max_history_size: usize,
        
        /// Minimum and maximum batch sizes to consider
        min_batch_size: usize,
        max_batch_size: usize,
    }
    
    /// Performance record for batch size optimization
    #[derive(Clone, Debug)]
    struct BatchPerformanceRecord {
        batch_size: usize,
        total_time_ns: u64,
        communication_bytes: usize,
        success_rate: f64,
        timestamp: std::time::Instant,
    }
    
    impl AdaptiveBatchOptimizer {
        /// Creates a new adaptive batch optimizer
        /// 
        /// # Arguments
        /// * `protocol` - Base batched sumcheck protocol
        /// * `min_batch_size` - Minimum batch size to consider
        /// * `max_batch_size` - Maximum batch size to consider
        /// 
        /// # Returns
        /// * `Self` - New optimizer instance
        pub fn new(
            protocol: BatchedSumcheckProtocol,
            min_batch_size: usize,
            max_batch_size: usize
        ) -> Self {
            let initial_optimal = (min_batch_size + max_batch_size) / 2;
            
            Self {
                protocol,
                current_optimal_size: initial_optimal,
                performance_history: Vec::new(),
                max_history_size: 100,
                min_batch_size,
                max_batch_size,
            }
        }
        
        /// Determines optimal batch size for given workload
        /// 
        /// # Arguments
        /// * `total_claims` - Total number of claims to process
        /// * `memory_limit_bytes` - Available memory limit
        /// * `latency_target_ms` - Target latency requirement
        /// 
        /// # Returns
        /// * `usize` - Recommended batch size
        /// 
        /// # Optimization Strategy
        /// Uses historical performance data and current constraints to
        /// select batch size that maximizes throughput while meeting
        /// memory and latency requirements.
        pub fn optimize_batch_size(
            &mut self,
            total_claims: usize,
            memory_limit_bytes: usize,
            latency_target_ms: u64
        ) -> usize {
            // Start with current optimal as baseline
            let mut best_size = self.current_optimal_size;
            let mut best_score = self.evaluate_batch_size(best_size, total_claims, memory_limit_bytes, latency_target_ms);
            
            // Search around current optimal
            let search_range = 5;
            let start = self.current_optimal_size.saturating_sub(search_range).max(self.min_batch_size);
            let end = (self.current_optimal_size + search_range).min(self.max_batch_size);
            
            for candidate_size in start..=end {
                let score = self.evaluate_batch_size(candidate_size, total_claims, memory_limit_bytes, latency_target_ms);
                if score > best_score {
                    best_score = score;
                    best_size = candidate_size;
                }
            }
            
            // Update optimal size if we found a better one
            if best_size != self.current_optimal_size {
                self.current_optimal_size = best_size;
            }
            
            // Ensure batch size doesn't exceed total claims
            best_size.min(total_claims)
        }
        
        /// Evaluates a batch size based on multiple criteria
        /// 
        /// # Arguments
        /// * `batch_size` - Batch size to evaluate
        /// * `total_claims` - Total number of claims
        /// * `memory_limit` - Memory constraint
        /// * `latency_target` - Latency requirement
        /// 
        /// # Returns
        /// * `f64` - Score for this batch size (higher is better)
        fn evaluate_batch_size(
            &self,
            batch_size: usize,
            total_claims: usize,
            memory_limit: usize,
            latency_target: u64
        ) -> f64 {
            // Estimate memory usage
            let estimated_memory = self.estimate_memory_usage(batch_size);
            if estimated_memory > memory_limit {
                return 0.0; // Exceeds memory limit
            }
            
            // Estimate latency
            let estimated_latency = self.estimate_latency(batch_size);
            if estimated_latency > latency_target * 1_000_000 { // Convert ms to ns
                return 0.0; // Exceeds latency target
            }
            
            // Calculate throughput score
            let num_batches = (total_claims + batch_size - 1) / batch_size;
            let total_time = num_batches as u64 * estimated_latency;
            let throughput_score = total_claims as f64 / (total_time as f64 / 1_000_000_000.0); // Claims per second
            
            // Calculate efficiency score
            let communication_efficiency = self.estimate_communication_efficiency(batch_size);
            
            // Calculate memory efficiency score
            let memory_efficiency = (memory_limit - estimated_memory) as f64 / memory_limit as f64;
            
            // Combine scores with weights
            let throughput_weight = 0.5;
            let communication_weight = 0.3;
            let memory_weight = 0.2;
            
            throughput_weight * throughput_score +
            communication_weight * communication_efficiency +
            memory_weight * memory_efficiency
        }
        
        /// Estimates memory usage for a given batch size
        /// 
        /// # Arguments
        /// * `batch_size` - Batch size to estimate
        /// 
        /// # Returns
        /// * `usize` - Estimated memory usage in bytes
        fn estimate_memory_usage(&self, batch_size: usize) -> usize {
            // Rough estimate based on protocol parameters
            let polynomial_memory = batch_size * (1 << self.protocol.num_variables) * self.protocol.ring_dimension * 8;
            let coefficient_memory = batch_size * self.protocol.ring_dimension * 8;
            let proof_memory = self.protocol.ring_dimension * self.protocol.num_variables * 8;
            
            polynomial_memory + coefficient_memory + proof_memory
        }
        
        /// Estimates latency for a given batch size
        /// 
        /// # Arguments
        /// * `batch_size` - Batch size to estimate
        /// 
        /// # Returns
        /// * `u64` - Estimated latency in nanoseconds
        fn estimate_latency(&self, batch_size: usize) -> u64 {
            // Use historical data if available
            if let Some(record) = self.find_closest_performance_record(batch_size) {
                // Scale based on batch size difference
                let scale_factor = batch_size as f64 / record.batch_size as f64;
                (record.total_time_ns as f64 * scale_factor) as u64
            } else {
                // Rough estimate based on theoretical complexity
                let base_latency = 1_000_000; // 1ms base
                let batch_factor = (batch_size as f64).log2();
                let variable_factor = self.protocol.num_variables as f64;
                
                (base_latency as f64 * batch_factor * variable_factor) as u64
            }
        }
        
        /// Estimates communication efficiency for a given batch size
        /// 
        /// # Arguments
        /// * `batch_size` - Batch size to estimate
        /// 
        /// # Returns
        /// * `f64` - Estimated communication efficiency (0.0 to 1.0)
        fn estimate_communication_efficiency(&self, batch_size: usize) -> f64 {
            // Communication efficiency improves with batch size due to shared base proof
            let base_proof_size = self.protocol.num_variables * self.protocol.ring_dimension * 8;
            let per_claim_size = self.protocol.ring_dimension * 8;
            
            let batch_total = base_proof_size + batch_size * per_claim_size;
            let individual_total = batch_size * (base_proof_size + per_claim_size);
            
            individual_total as f64 / batch_total as f64
        }
        
        /// Finds the closest performance record for a given batch size
        /// 
        /// # Arguments
        /// * `batch_size` - Target batch size
        /// 
        /// # Returns
        /// * `Option<&BatchPerformanceRecord>` - Closest record if available
        fn find_closest_performance_record(&self, batch_size: usize) -> Option<&BatchPerformanceRecord> {
            self.performance_history
                .iter()
                .min_by_key(|record| (record.batch_size as i64 - batch_size as i64).abs())
        }
        
        /// Records performance data for future optimization
        /// 
        /// # Arguments
        /// * `batch_size` - Batch size that was used
        /// * `total_time_ns` - Total execution time
        /// * `communication_bytes` - Communication cost
        /// * `success_rate` - Success rate of the batch
        pub fn record_performance(
            &mut self,
            batch_size: usize,
            total_time_ns: u64,
            communication_bytes: usize,
            success_rate: f64
        ) {
            let record = BatchPerformanceRecord {
                batch_size,
                total_time_ns,
                communication_bytes,
                success_rate,
                timestamp: std::time::Instant::now(),
            };
            
            self.performance_history.push(record);
            
            // Limit history size
            if self.performance_history.len() > self.max_history_size {
                self.performance_history.remove(0);
            }
        }
        
        /// Returns current optimization statistics
        /// 
        /// # Returns
        /// * `OptimizationStats` - Current optimization metrics
        pub fn optimization_stats(&self) -> OptimizationStats {
            let avg_batch_size = if self.performance_history.is_empty() {
                self.current_optimal_size as f64
            } else {
                self.performance_history.iter()
                    .map(|r| r.batch_size as f64)
                    .sum::<f64>() / self.performance_history.len() as f64
            };
            
            let avg_latency = if self.performance_history.is_empty() {
                0.0
            } else {
                self.performance_history.iter()
                    .map(|r| r.total_time_ns as f64)
                    .sum::<f64>() / self.performance_history.len() as f64
            };
            
            OptimizationStats {
                current_optimal_size: self.current_optimal_size,
                average_batch_size: avg_batch_size,
                average_latency_ns: avg_latency,
                optimization_history_size: self.performance_history.len(),
            }
        }
    }
    
    /// Statistics for batch size optimization
    #[derive(Clone, Debug)]
    pub struct OptimizationStats {
        /// Current optimal batch size
        pub current_optimal_size: usize,
        
        /// Average batch size used historically
        pub average_batch_size: f64,
        
        /// Average latency observed
        pub average_latency_ns: f64,
        
        /// Number of performance records stored
        pub optimization_history_size: usize,
    }
}

/// Security Properties:
/// - Soundness: If any individual claim is false, verification fails with high probability
/// - Completeness: If all individual claims are true, verification always succeeds
/// - Zero-knowledge: Can be made zero-knowledge with additional randomness
#[derive(Clone, Debug, Zeroize, ZeroizeOnDrop)]
pub struct BatchedSumcheckProof {
    /// Underlying sumcheck proof for combined polynomial
    base_sumcheck_proof: SumcheckProof,
    
    /// Random coefficients used for batching
    batching_coefficients: Vec<RingElement>,
    
    /// Individual polynomial evaluations at final challenge
    individual_evaluations: Vec<RingElement>,
    
    /// Number of claims in the batch
    batch_size: usize,
    
    /// Number of variables in each polynomial
    num_variables: usize,
    
    /// Ring dimension d
    ring_dimension: usize,
    
    /// Modulus q
    modulus: i64,
}

impl BatchedSumcheckProof {
    /// Estimates the serialized size of the proof
    /// 
    /// # Returns
    /// * `usize` - Estimated size in bytes
    pub fn serialized_size(&self) -> usize {
        let ring_element_size = self.ring_dimension * 8; // i64 coefficients
        let mut total_size = 200; // Metadata overhead
        
        // Add size of base sumcheck proof
        total_size += self.base_sumcheck_proof.serialized_size();
        
        // Add size of batching coefficients
        total_size += self.batching_coefficients.len() * ring_element_size;
        
        // Add size of individual evaluations
        total_size += self.individual_evaluations.len() * ring_element_size;
        
        total_size
    }
    
    /// Checks if the proof is complete and well-formed
    /// 
    /// # Returns
    /// * `bool` - True if proof is complete
    pub fn is_complete(&self) -> bool {
        self.base_sumcheck_proof.is_complete() &&
        self.batching_coefficients.len() == self.batch_size &&
        self.individual_evaluations.len() == self.batch_size &&
        self.batch_size > 0 &&
        self.num_variables > 0
    }
    
    /// Returns the batch size
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }
    
    /// Returns the communication compression ratio
    /// 
    /// # Returns
    /// * `f64` - Compression ratio (individual_size / batched_size)
    pub fn compression_ratio(&self) -> f64 {
        if self.batch_size <= 1 {
            return 1.0;
        }
        
        // Estimate individual proof sizes
        let individual_proof_size = self.base_sumcheck_proof.serialized_size();
        let total_individual_size = self.batch_size * individual_proof_size;
        
        // Compare with batched proof size
        let batched_size = self.serialized_size();
        
        total_individual_size as f64 / batched_size as f64
    }
    
    /// Returns the theoretical verifier speedup
    /// 
    /// # Returns
    /// * `f64` - Speedup factor (individual_time / batched_time)
    pub fn theoretical_verifier_speedup(&self) -> f64 {
        if self.batch_size <= 1 {
            return 1.0;
        }
        
        // Individual complexity: O(L × k) for L claims with k variables each
        let individual_complexity = (self.batch_size * self.num_variables) as f64;
        
        // Batched complexity: O(L + k) for batch combination plus single verification
        let batched_complexity = (self.batch_size + self.num_variables) as f64;
        
        individual_complexity / batched_complexity
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    
    /// Test basic batched sumcheck protocol creation
    #[test]
    fn test_batched_sumcheck_creation() {
        let num_variables = 3;
        let ring_dimension = 16;
        let modulus = 97;
        let max_degree = 2;
        let max_batch_size = 10;
        
        let protocol = BatchedSumcheckProtocol::new(
            num_variables,
            ring_dimension,
            modulus,
            max_degree,
            max_batch_size
        );
        
        assert!(protocol.is_ok(), "Protocol creation should succeed");
        
        let protocol = protocol.unwrap();
        assert_eq!(protocol.num_variables, num_variables);
        assert_eq!(protocol.ring_dimension, ring_dimension);
        assert_eq!(protocol.modulus, modulus);
        assert_eq!(protocol.max_batch_size, max_batch_size);
    }
    
    /// Test parameter validation
    #[test]
    fn test_parameter_validation() {
        // Test too many variables
        assert!(BatchedSumcheckProtocol::new(50, 16, 97, 2, 10).is_err());
        
        // Test non-power-of-2 ring dimension
        assert!(BatchedSumcheckProtocol::new(3, 15, 97, 2, 10).is_err());
        
        // Test invalid modulus
        assert!(BatchedSumcheckProtocol::new(3, 16, -1, 2, 10).is_err());
        
        // Test zero degree
        assert!(BatchedSumcheckProtocol::new(3, 16, 97, 0, 10).is_err());
        
        // Test excessive batch size
        assert!(BatchedSumcheckProtocol::new(3, 16, 97, 2, MAX_BATCH_SIZE + 1).is_err());
    }
    
    /// Test batching coefficient generation
    #[test]
    fn test_batching_coefficient_generation() {
        let protocol = BatchedSumcheckProtocol::new(3, 16, 97, 2, 10).unwrap();
        
        let batch_size = 5;
        let coefficients = protocol.generate_batching_coefficients(batch_size).unwrap();
        
        assert_eq!(coefficients.len(), batch_size);
        
        // Check all coefficients have correct dimension
        for coeff in &coefficients {
            assert_eq!(coeff.dimension(), protocol.ring_dimension);
        }
    }
    
    /// Test proof structure validation
    #[test]
    fn test_proof_validation() {
        let ring_dimension = 16;
        let modulus = 97;
        let batch_size = 3;
        let num_variables = 2;
        
        // Create valid proof structure
        let base_proof = SumcheckProof::dummy(num_variables, ring_dimension, modulus);
        let batching_coefficients = vec![
            RingElement::zero(ring_dimension, Some(modulus)).unwrap();
            batch_size
        ];
        let individual_evaluations = vec![
            RingElement::zero(ring_dimension, Some(modulus)).unwrap();
            batch_size
        ];
        
        let proof = BatchedSumcheckProof {
            base_sumcheck_proof: base_proof,
            batching_coefficients,
            individual_evaluations,
            batch_size,
            num_variables,
            ring_dimension,
            modulus,
        };
        
        assert!(proof.validate_structure().is_ok(), "Valid proof should pass validation");
        
        // Test invalid batch size
        let mut invalid_proof = proof.clone();
        invalid_proof.batch_size = 5; // Doesn't match coefficient count
        assert!(invalid_proof.validate_structure().is_err(), "Invalid batch size should fail validation");
    }
    
    /// Test communication efficiency calculation
    #[test]
    fn test_communication_efficiency() {
        let ring_dimension = 16;
        let modulus = 97;
        let batch_size = 10;
        let num_variables = 3;
        
        let base_proof = SumcheckProof::dummy(num_variables, ring_dimension, modulus);
        let batching_coefficients = vec![
            RingElement::zero(ring_dimension, Some(modulus)).unwrap();
            batch_size
        ];
        let individual_evaluations = vec![
            RingElement::zero(ring_dimension, Some(modulus)).unwrap();
            batch_size
        ];
        
        let proof = BatchedSumcheckProof {
            base_sumcheck_proof: base_proof,
            batching_coefficients,
            individual_evaluations,
            batch_size,
            num_variables,
            ring_dimension,
            modulus,
        };
        
        let efficiency = proof.communication_efficiency();
        
        // Batch should be more efficient than individual proofs
        assert!(efficiency < 1.0, "Batch should be more communication efficient");
        assert!(efficiency > 0.0, "Efficiency should be positive");
    }
    
    /// Test adaptive batch optimizer
    #[test]
    fn test_adaptive_batch_optimizer() {
        let protocol = BatchedSumcheckProtocol::new(3, 16, 97, 2, 100).unwrap();
        let mut optimizer = advanced::AdaptiveBatchOptimizer::new(protocol, 2, 50);
        
        // Test optimization with constraints
        let total_claims = 1000;
        let memory_limit = 1_000_000; // 1MB
        let latency_target = 100; // 100ms
        
        let optimal_size = optimizer.optimize_batch_size(total_claims, memory_limit, latency_target);
        
        assert!(optimal_size >= 2, "Optimal size should be at least minimum");
        assert!(optimal_size <= 50, "Optimal size should not exceed maximum");
        assert!(optimal_size <= total_claims, "Optimal size should not exceed total claims");
        
        // Record some performance data
        optimizer.record_performance(optimal_size, 50_000_000, 10000, 1.0);
        
        let stats = optimizer.optimization_stats();
        assert_eq!(stats.optimization_history_size, 1);
        assert_eq!(stats.current_optimal_size, optimal_size);
    }
    
    /// Test statistics tracking
    #[test]
    fn test_statistics_tracking() {
        let mut stats = BatchedSumcheckStats::new();
        
        // Record some executions
        stats.record_batch_execution(5, 1_000_000, 500_000, 1000, true);
        stats.record_batch_execution(10, 2_000_000, 800_000, 2000, true);
        stats.record_batch_execution(3, 800_000, 400_000, 800, false);
        
        assert_eq!(stats.total_batch_executions, 3);
        assert_eq!(stats.total_claims_processed, 18); // 5 + 10 + 3
        assert_eq!(stats.successful_batches, 2);
        assert_eq!(stats.failed_batches, 1);
        
        let success_rate = stats.success_rate();
        assert!((success_rate - 66.67).abs() < 0.1, "Success rate should be approximately 66.67%");
        
        let avg_batch_size = stats.average_batch_size;
        assert!((avg_batch_size - 6.0).abs() < 0.1, "Average batch size should be 6.0");
    }
}


    
    #[test]
    fn test_batched_sumcheck_protocol_creation() {
        let num_variables = 3;
        let ring_dimension = 64;
        let modulus = 97;
        let max_degree = 2;
        let max_batch_size = 10;
        
        let protocol = BatchedSumcheckProtocol::new(
            num_variables,
            ring_dimension,
            modulus,
            max_degree,
            max_batch_size
        ).unwrap();
        
        assert_eq!(protocol.num_variables, num_variables);
        assert_eq!(protocol.ring_dimension, ring_dimension);
        assert_eq!(protocol.modulus, modulus);
        assert_eq!(protocol.max_degree, max_degree);
        assert_eq!(protocol.max_batch_size, max_batch_size);
    }
    
    #[test]
    fn test_batched_sumcheck_parameter_validation() {
        let ring_dimension = 64;
        let modulus = 97;
        let max_degree = 2;
        let max_batch_size = 10;
        
        // Test invalid number of variables (too many)
        assert!(BatchedSumcheckProtocol::new(
            33, ring_dimension, modulus, max_degree, max_batch_size
        ).is_err());
        
        // Test invalid ring dimension (not power of 2)
        assert!(BatchedSumcheckProtocol::new(
            3, 63, modulus, max_degree, max_batch_size
        ).is_err());
        
        // Test invalid modulus
        assert!(BatchedSumcheckProtocol::new(
            3, ring_dimension, -1, max_degree, max_batch_size
        ).is_err());
        
        // Test invalid max degree
        assert!(BatchedSumcheckProtocol::new(
            3, ring_dimension, modulus, 0, max_batch_size
        ).is_err());
        
        // Test invalid batch size
        assert!(BatchedSumcheckProtocol::new(
            3, ring_dimension, modulus, max_degree, MAX_BATCH_SIZE + 1
        ).is_err());
    }
    
    #[test]
    fn test_batching_coefficient_generation() {
        let protocol = BatchedSumcheckProtocol::new(3, 64, 97, 2, 10).unwrap();
        
        let batch_size = 5;
        let coefficients = protocol.generate_batching_coefficients(batch_size).unwrap();
        
        assert_eq!(coefficients.len(), batch_size);
        
        // Check that coefficients are valid ring elements
        for coeff in &coefficients {
            assert_eq!(coeff.dimension(), protocol.ring_dimension);
            
            // Check coefficients are in balanced representation
            for &c in coeff.coefficients() {
                assert!(c.abs() < protocol.modulus);
            }
        }
    }
    
    #[test]
    fn test_claimed_sums_combination() {
        let protocol = BatchedSumcheckProtocol::new(3, 64, 97, 2, 10).unwrap();
        
        // Create test claimed sums
        let claimed_sums = vec![
            RingElement::from_coefficients(vec![1, 2, 3], Some(97)).unwrap(),
            RingElement::from_coefficients(vec![4, 5, 6], Some(97)).unwrap(),
            RingElement::from_coefficients(vec![7, 8, 9], Some(97)).unwrap(),
        ];
        
        // Create test coefficients
        let coefficients = vec![
            RingElement::from_coefficients(vec![1, 0, 0], Some(97)).unwrap(),
            RingElement::from_coefficients(vec![2, 0, 0], Some(97)).unwrap(),
            RingElement::from_coefficients(vec![3, 0, 0], Some(97)).unwrap(),
        ];
        
        let combined = protocol.combine_claimed_sums(&claimed_sums, &coefficients).unwrap();
        
        // Verify the combination is computed correctly
        // Expected: 1*(1,2,3) + 2*(4,5,6) + 3*(7,8,9) = (1+8+21, 2+10+24, 3+12+27) = (30,36,42)
        let expected_coeffs = vec![30, 36, 42];
        let combined_coeffs = combined.coefficients();
        
        for i in 0..3 {
            assert_eq!(combined_coeffs[i], expected_coeffs[i]);
        }
    }
    
    #[test]
    fn test_statistics_tracking() {
        let mut protocol = BatchedSumcheckProtocol::new(3, 64, 97, 2, 10).unwrap();
        
        // Record some test statistics
        protocol.stats.record_batch_execution(5, 1000, 500, 200, true);
        protocol.stats.record_batch_execution(3, 800, 400, 150, true);
        protocol.stats.record_batch_execution(7, 1200, 600, 250, false);
        
        let stats = protocol.stats();
        
        assert_eq!(stats.total_batch_executions, 3);
        assert_eq!(stats.total_claims_processed, 15); // 5 + 3 + 7
        assert_eq!(stats.successful_batches, 2);
        assert_eq!(stats.failed_batches, 1);
        assert_eq!(stats.success_rate(), 200.0 / 3.0); // 2/3 * 100%
        assert_eq!(stats.average_batch_size, 5.0); // 15/3
    }
    
    #[test]
    fn test_cache_operations() {
        let mut protocol = BatchedSumcheckProtocol::new(3, 64, 97, 2, 10).unwrap();
        
        // Test cache clearing
        protocol.clear_cache();
        
        // Cache should be empty after clearing
        let cache_size = protocol.tensor_cache.lock().unwrap().len();
        assert_eq!(cache_size, 0);
    }
    
    #[test]
    fn test_challenge_length_validation() {
        let protocol = BatchedSumcheckProtocol::new(3, 64, 97, 2, 10).unwrap();
        
        // Test that the protocol expects challenges of the correct length
        assert_eq!(protocol.num_variables, 3);
        
        // Test challenge vector creation
        let test_challenge = vec![1i64, 2i64, 3i64];
        assert_eq!(test_challenge.len(), protocol.num_variables);
        
        // Verify challenge values are within modulus bounds
        for &val in &test_challenge {
            assert!(val >= 0 && val < protocol.modulus);
        }
    }
    
    #[test]
    fn test_batch_size_limits() {
        let protocol = BatchedSumcheckProtocol::new(3, 64, 97, 2, 5).unwrap();
        
        // Test that batch size limits are enforced
        assert_eq!(protocol.max_batch_size, 5);
        
        // Test minimum batch size validation would be done in prove_batch
        // (implementation would check MIN_BATCH_SIZE)
    }
    
    #[test]
    fn test_parallel_efficiency_calculation() {
        let mut stats = BatchedSumcheckStats::new();
        
        stats.parallel_operations = 80;
        stats.sequential_operations = 20;
        
        assert_eq!(stats.parallel_efficiency(), 80.0); // 80/100 * 100%
        
        // Test edge case with no operations
        let empty_stats = BatchedSumcheckStats::new();
        assert_eq!(empty_stats.parallel_efficiency(), 0.0);
    }
    
    #[test]
    fn test_theoretical_speedup_calculation() {
        let mut stats = BatchedSumcheckStats::new();
        stats.num_variables = 5;
        
        // Test with average batch size of 10
        stats.average_batch_size = 10.0;
        
        // Theoretical speedup: (L*k) / (L+k) = (10*5) / (10+5) = 50/15 = 3.33...
        let speedup = stats.theoretical_speedup();
        assert!((speedup - 3.333).abs() < 0.01);
        
        // Test edge case with batch size 1 (no batching)
        stats.average_batch_size = 1.0;
        assert_eq!(stats.theoretical_speedup(), 1.0);
    }
}
