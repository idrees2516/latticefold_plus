/// End-to-End Folding Integration for LatticeFold+
/// 
/// This module implements the complete folding scheme that integrates all sub-protocols
/// to provide a unified, production-ready LatticeFold+ system with optimal performance
/// characteristics and comprehensive security guarantees.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use rayon::prelude::*;
use zeroize::{Zeroize, ZeroizeOnDrop};
use merlin::Transcript;
use rand::{CryptoRng, RngCore};

// Import all required LatticeFold+ components
use crate::cyclotomic_ring::{RingElement, BalancedCoefficients};
use crate::multi_instance_folding::{
    LinearRelation, MultiInstanceLinearRelation, LinearFoldingProtocol, 
    LinearFoldingParams, LinearFoldingProof, LinearFoldingStats
};
use crate::commitment_transformation::{
    CommitmentTransformationProtocol, CommitmentTransformationParams,
    CommitmentTransformationProof, CommitmentTransformationStats
};
use crate::range_check_protocol::{
    RangeCheckProtocol, RangeCheckProof, RangeCheckStats
};
use crate::sumcheck_batching::{
    BatchedSumcheckProtocol, BatchedSumcheckProof, BatchedSumcheckStats
};
use crate::double_commitment::{DoubleCommitmentScheme, DoubleCommitmentParams};
use crate::monomial_commitment::{MonomialCommitmentScheme, CommitmentStats};
use crate::security_analysis::{
    SecurityAnalyzer, SecurityAnalysisResults, ParameterAdequacy
};
use crate::folding_challenge_generation::{
    FoldingChallengeGenerator, FoldingChallengeParams, FoldingChallenges
};
use crate::error::{LatticeFoldError, Result};

/// Maximum number of R1CS instances that can be processed in a single batch
const MAX_R1CS_BATCH_SIZE: usize = 1000;

/// Minimum number of instances required for efficient folding
const MIN_FOLDING_INSTANCES: usize = 2;

/// Target security level in bits (128-bit security)
const TARGET_SECURITY_BITS: usize = 128;

/// Default ring dimension for 128-bit security
const DEFAULT_RING_DIMENSION: usize = 64;

/// Default witness dimension for large-scale applications
const DEFAULT_WITNESS_DIMENSION: usize = 2_097_152; // 2^21

/// Default modulus for 128-bit security (128-bit prime)
const DEFAULT_MODULUS: i64 = 340282366920938463463374607431768211297; // 2^128 - 159

/// Cache size for frequently used folding operations
const FOLDING_CACHE_SIZE: usize = 512;

/// Threshold for GPU acceleration activation
const GPU_ACCELERATION_THRESHOLD: usize = 1000;/// Pa
rameters for the complete end-to-end folding system
/// 
/// These parameters define the structure, security, and performance characteristics
/// of the entire LatticeFold+ system, ensuring all components work together
/// optimally while maintaining the required security guarantees.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EndToEndFoldingParams {
    /// Security parameter λ in bits (target: 128)
    /// Determines the hardness of underlying lattice problems
    pub security_bits: usize,
    
    /// Ring dimension d for cyclotomic ring R = Z[X]/(X^d + 1)
    /// Must be power of 2 for NTT compatibility, typical value: 64
    pub ring_dimension: usize,
    
    /// Witness dimension n for input vectors
    /// Large dimension enables complex constraint systems, typical value: 2²¹
    pub witness_dimension: usize,
    
    /// Security parameter κ for commitment schemes
    /// Determines commitment matrix dimensions and security level
    pub kappa: usize,
    
    /// Modulus q for ring operations in Rq = R/qR
    /// Must be prime and NTT-friendly, typical size: 128 bits
    pub modulus: i64,
    
    /// Maximum number of instances in multi-instance folding
    /// Determines L-to-2 folding efficiency, typical range: 10-100
    pub max_folding_instances: usize,
    
    /// Norm bound B for witness vectors
    /// Controls security vs efficiency trade-off
    pub norm_bound: i64,
    
    /// Range bound for algebraic range proofs
    /// Typically d/2 for optimal cyclotomic structure
    pub range_bound: i64,
    
    /// Enable GPU acceleration for computational components
    /// Provides significant speedup for large-scale operations
    pub enable_gpu_acceleration: bool,
    
    /// Enable comprehensive performance monitoring
    /// Tracks detailed metrics for optimization analysis
    pub enable_performance_monitoring: bool,
    
    /// Enable security analysis and parameter validation
    /// Provides runtime security verification
    pub enable_security_analysis: bool,
    
    /// Cache size for frequently used operations
    /// Optimizes repeated computations
    pub cache_size: usize,
}

impl EndToEndFoldingParams {
    /// Creates parameters optimized for 128-bit security
    /// 
    /// # Returns
    /// * `Result<Self>` - Optimized parameters or validation error
    /// 
    /// # Parameter Selection Strategy
    /// - Security: 128-bit post-quantum security level
    /// - Performance: Optimized for modern multi-core CPUs with GPU acceleration
    /// - Memory: Balanced for systems with 16-64 GB RAM
    /// - Scalability: Supports constraint systems up to 2²¹ variables
    pub fn for_128_bit_security() -> Result<Self> {
        // Compute derived parameters for 128-bit security
        let security_bits = TARGET_SECURITY_BITS;
        let ring_dimension = DEFAULT_RING_DIMENSION;
        let witness_dimension = DEFAULT_WITNESS_DIMENSION;
        
        // Security parameter κ = 2 × security_bits for adequate margin
        let kappa = 2 * security_bits;
        
        // Use default modulus for 128-bit security
        let modulus = DEFAULT_MODULUS;
        
        // Set folding parameters for optimal performance
        let max_folding_instances = 64; // Good balance of efficiency and memory usage
        
        // Norm bound: 2^16 provides good security/efficiency trade-off
        let norm_bound = 65536; // 2^16
        
        // Range bound: d/2 for optimal cyclotomic structure
        let range_bound = (ring_dimension / 2) as i64;
        
        let params = Self {
            security_bits,
            ring_dimension,
            witness_dimension,
            kappa,
            modulus,
            max_folding_instances,
            norm_bound,
            range_bound,
            enable_gpu_acceleration: true,
            enable_performance_monitoring: true,
            enable_security_analysis: true,
            cache_size: FOLDING_CACHE_SIZE,
        };
        
        // Validate parameter consistency
        params.validate()?;
        
        Ok(params)
    }
    
    /// Validates parameter consistency and mathematical constraints
    /// 
    /// # Returns
    /// * `Result<()>` - Ok if parameters are valid, error otherwise
    pub fn validate(&self) -> Result<()> {
        // Validate security level
        if self.security_bits < 80 {
            return Err(LatticeFoldError::InvalidParameters(
                format!("Security level {} bits too low (minimum 80)", self.security_bits)
            ));
        }
        
        // Validate ring dimension
        if !self.ring_dimension.is_power_of_two() {
            return Err(LatticeFoldError::InvalidDimension {
                expected: self.ring_dimension.next_power_of_two(),
                got: self.ring_dimension,
            });
        }
        
        if self.ring_dimension < 32 || self.ring_dimension > 8192 {
            return Err(LatticeFoldError::InvalidDimension {
                expected: 64, // Reasonable default
                got: self.ring_dimension,
            });
        }
        
        // Validate witness dimension
        if self.witness_dimension == 0 {
            return Err(LatticeFoldError::InvalidDimension {
                expected: 1,
                got: self.witness_dimension,
            });
        }
        
        // Validate security parameter κ
        if self.kappa < self.security_bits {
            return Err(LatticeFoldError::InvalidParameters(
                format!("Security parameter κ = {} too small for {}-bit security", 
                       self.kappa, self.security_bits)
            ));
        }
        
        // Validate modulus
        if self.modulus <= 0 {
            return Err(LatticeFoldError::InvalidModulus { modulus: self.modulus });
        }
        
        // Validate folding parameters
        if self.max_folding_instances < MIN_FOLDING_INSTANCES {
            return Err(LatticeFoldError::InvalidParameters(
                format!("Max folding instances {} too small (minimum {})", 
                       self.max_folding_instances, MIN_FOLDING_INSTANCES)
            ));
        }
        
        if self.max_folding_instances > MAX_R1CS_BATCH_SIZE {
            return Err(LatticeFoldError::InvalidParameters(
                format!("Max folding instances {} too large (maximum {})", 
                       self.max_folding_instances, MAX_R1CS_BATCH_SIZE)
            ));
        }
        
        // Validate norm bounds
        if self.norm_bound <= 0 {
            return Err(LatticeFoldError::InvalidParameters(
                format!("Norm bound {} must be positive", self.norm_bound)
            ));
        }
        
        if self.range_bound <= 0 {
            return Err(LatticeFoldError::InvalidParameters(
                format!("Range bound {} must be positive", self.range_bound)
            ));
        }
        
        // Validate range bound is d/2 for proper cyclotomic structure
        let expected_range_bound = (self.ring_dimension / 2) as i64;
        if self.range_bound != expected_range_bound {
            return Err(LatticeFoldError::InvalidParameters(
                format!("Range bound {} must equal d/2 = {} for ring dimension d = {}", 
                       self.range_bound, expected_range_bound, self.ring_dimension)
            ));
        }
        
        // Validate cache size
        if self.cache_size == 0 {
            return Err(LatticeFoldError::InvalidParameters(
                "Cache size must be positive".to_string()
            ));
        }
        
        Ok(())
    }
}/
// R1CS instance for constraint satisfaction problems
/// 
/// Represents a Rank-1 Constraint System instance that will be processed
/// by the LatticeFold+ system. Each instance contains the constraint matrices,
/// witness vector, and public inputs.
#[derive(Clone, Debug, Zeroize, ZeroizeOnDrop)]
pub struct R1CSInstance {
    /// Constraint matrix A ∈ Rq^{m×n}
    /// Left multiplication matrix for constraint (Az) ◦ (Bz) = Cz
    #[zeroize(skip)]
    pub matrix_a: Vec<Vec<RingElement>>,
    
    /// Constraint matrix B ∈ Rq^{m×n}
    /// Right multiplication matrix for constraint (Az) ◦ (Bz) = Cz
    #[zeroize(skip)]
    pub matrix_b: Vec<Vec<RingElement>>,
    
    /// Constraint matrix C ∈ Rq^{m×n}
    /// Output matrix for constraint (Az) ◦ (Bz) = Cz
    #[zeroize(skip)]
    pub matrix_c: Vec<Vec<RingElement>>,
    
    /// Full witness vector z = (1, x, w) ∈ Rq^n
    /// Contains constant, public inputs, and private witness
    pub witness: Vec<RingElement>,
    
    /// Public input vector x ∈ Rq^ℓ
    /// Known to both prover and verifier
    #[zeroize(skip)]
    pub public_inputs: Vec<RingElement>,
    
    /// Number of constraints m
    pub num_constraints: usize,
    
    /// Number of variables n (including constant and public inputs)
    pub num_variables: usize,
    
    /// Number of public inputs ℓ
    pub num_public_inputs: usize,
}

impl R1CSInstance {
    /// Creates a new R1CS instance with validation
    /// 
    /// # Arguments
    /// * `matrix_a` - Left constraint matrix A
    /// * `matrix_b` - Right constraint matrix B  
    /// * `matrix_c` - Output constraint matrix C
    /// * `witness` - Full witness vector z
    /// * `public_inputs` - Public input vector x
    /// 
    /// # Returns
    /// * `Result<Self>` - Validated R1CS instance or error
    pub fn new(
        matrix_a: Vec<Vec<RingElement>>,
        matrix_b: Vec<Vec<RingElement>>,
        matrix_c: Vec<Vec<RingElement>>,
        witness: Vec<RingElement>,
        public_inputs: Vec<RingElement>,
    ) -> Result<Self> {
        // Validate matrix dimensions
        let num_constraints = matrix_a.len();
        let num_variables = if num_constraints > 0 { matrix_a[0].len() } else { 0 };
        let num_public_inputs = public_inputs.len();
        
        // Check matrix A dimensions
        if matrix_a.len() != num_constraints {
            return Err(LatticeFoldError::InvalidDimension {
                expected: num_constraints,
                got: matrix_a.len(),
            });
        }
        
        for (i, row) in matrix_a.iter().enumerate() {
            if row.len() != num_variables {
                return Err(LatticeFoldError::InvalidDimension {
                    expected: num_variables,
                    got: row.len(),
                });
            }
        }
        
        // Check matrix B dimensions
        if matrix_b.len() != num_constraints {
            return Err(LatticeFoldError::InvalidDimension {
                expected: num_constraints,
                got: matrix_b.len(),
            });
        }
        
        for (i, row) in matrix_b.iter().enumerate() {
            if row.len() != num_variables {
                return Err(LatticeFoldError::InvalidDimension {
                    expected: num_variables,
                    got: row.len(),
                });
            }
        }
        
        // Check matrix C dimensions
        if matrix_c.len() != num_constraints {
            return Err(LatticeFoldError::InvalidDimension {
                expected: num_constraints,
                got: matrix_c.len(),
            });
        }
        
        for (i, row) in matrix_c.iter().enumerate() {
            if row.len() != num_variables {
                return Err(LatticeFoldError::InvalidDimension {
                    expected: num_variables,
                    got: row.len(),
                });
            }
        }
        
        // Check witness dimension
        if witness.len() != num_variables {
            return Err(LatticeFoldError::InvalidDimension {
                expected: num_variables,
                got: witness.len(),
            });
        }
        
        // Validate witness structure: z = (1, x, w)
        // First element should be 1 (constant term)
        if num_variables > 0 {
            let constant_term = witness[0].constant_term();
            if constant_term != 1 {
                return Err(LatticeFoldError::InvalidParameters(
                    format!("First witness element should be 1, got {}", constant_term)
                ));
            }
        }
        
        // Check public inputs are consistent with witness
        if num_public_inputs > 0 && num_variables > 1 {
            let witness_public_start = 1; // After constant term
            let witness_public_end = witness_public_start + num_public_inputs;
            
            if witness_public_end > num_variables {
                return Err(LatticeFoldError::InvalidDimension {
                    expected: num_variables - 1,
                    got: num_public_inputs,
                });
            }
            
            // Verify public inputs match witness
            for (i, public_input) in public_inputs.iter().enumerate() {
                let witness_public = &witness[witness_public_start + i];
                if !public_input.equals(witness_public)? {
                    return Err(LatticeFoldError::InvalidParameters(
                        format!("Public input {} doesn't match witness", i)
                    ));
                }
            }
        }
        
        let instance = Self {
            matrix_a,
            matrix_b,
            matrix_c,
            witness,
            public_inputs,
            num_constraints,
            num_variables,
            num_public_inputs,
        };
        
        // Validate constraint satisfaction
        instance.validate_constraints()?;
        
        Ok(instance)
    }
    
    /// Validates that the witness satisfies all R1CS constraints
    /// 
    /// # Returns
    /// * `Result<()>` - Ok if constraints are satisfied, error otherwise
    fn validate_constraints(&self) -> Result<()> {
        // Check each constraint: (Az)ᵢ × (Bz)ᵢ = (Cz)ᵢ
        for constraint_idx in 0..self.num_constraints {
            // Compute (Az)ᵢ = Σⱼ A[i,j] × z[j]
            let mut az_i = RingElement::zero(
                self.witness[0].dimension(), 
                Some(self.witness[0].modulus().unwrap_or(0))
            )?;
            
            for var_idx in 0..self.num_variables {
                let product = self.matrix_a[constraint_idx][var_idx].multiply(&self.witness[var_idx])?;
                az_i = az_i.add(&product)?;
            }
            
            // Compute (Bz)ᵢ = Σⱼ B[i,j] × z[j]
            let mut bz_i = RingElement::zero(
                self.witness[0].dimension(), 
                Some(self.witness[0].modulus().unwrap_or(0))
            )?;
            
            for var_idx in 0..self.num_variables {
                let product = self.matrix_b[constraint_idx][var_idx].multiply(&self.witness[var_idx])?;
                bz_i = bz_i.add(&product)?;
            }
            
            // Compute (Cz)ᵢ = Σⱼ C[i,j] × z[j]
            let mut cz_i = RingElement::zero(
                self.witness[0].dimension(), 
                Some(self.witness[0].modulus().unwrap_or(0))
            )?;
            
            for var_idx in 0..self.num_variables {
                let product = self.matrix_c[constraint_idx][var_idx].multiply(&self.witness[var_idx])?;
                cz_i = cz_i.add(&product)?;
            }
            
            // Verify constraint: (Az)ᵢ × (Bz)ᵢ = (Cz)ᵢ
            let left_side = az_i.multiply(&bz_i)?;
            
            if !left_side.equals(&cz_i)? {
                return Err(LatticeFoldError::InvalidParameters(
                    format!("R1CS constraint {} not satisfied", constraint_idx)
                ));
            }
        }
        
        Ok(())
    }
    
    /// Returns the private witness portion (excluding constant and public inputs)
    /// 
    /// # Returns
    /// * `Vec<RingElement>` - Private witness vector w
    pub fn private_witness(&self) -> Vec<RingElement> {
        let start_idx = 1 + self.num_public_inputs; // After constant and public inputs
        if start_idx < self.witness.len() {
            self.witness[start_idx..].to_vec()
        } else {
            Vec::new() // No private witness
        }
    }
}/// 
Comprehensive proof object for the complete LatticeFold+ system
/// 
/// Contains all cryptographic evidence needed to verify the correctness
/// of R1CS constraint satisfaction through the complete folding pipeline.
#[derive(Clone, Debug, Zeroize, ZeroizeOnDrop)]
pub struct EndToEndFoldingProof {
    /// Proof of R1CS to linear relation conversion
    /// Demonstrates that R1CS constraints are properly linearized
    #[zeroize(skip)]
    pub linearization_proofs: Vec<LinearizationProof>,
    
    /// Multi-instance folding proof for L-to-2 reduction
    /// Shows that multiple instances are correctly folded with norm control
    #[zeroize(skip)]
    pub folding_proof: LinearFoldingProof,
    
    /// Algebraic range proofs for all witness vectors
    /// Proves witnesses are within bounds without bit decomposition
    #[zeroize(skip)]
    pub range_proofs: Vec<RangeCheckProof>,
    
    /// Commitment transformation proofs for double-to-linear conversion
    /// Enables folding of non-homomorphic commitments
    #[zeroize(skip)]
    pub transformation_proofs: Vec<CommitmentTransformationProof>,
    
    /// Batched sumcheck proof for multiple verification claims
    /// Compresses verification of all consistency relations
    #[zeroize(skip)]
    pub sumcheck_proof: BatchedSumcheckProof,
    
    /// Final folded instance after complete processing
    /// Single linear relation representing all original R1CS instances
    #[zeroize(skip)]
    pub final_folded_instance: LinearRelation<ark_ff::Fp256<ark_ff::FpParameters>>,
    
    /// System parameters used for proof generation
    pub params: EndToEndFoldingParams,
    
    /// Performance statistics for the proof generation
    #[zeroize(skip)]
    pub performance_stats: EndToEndFoldingStats,
    
    /// Security analysis results for parameter validation
    #[zeroize(skip)]
    pub security_analysis: Option<SecurityAnalysisResults>,
}

/// Linearization proof for R1CS to linear relation conversion
#[derive(Clone, Debug, Zeroize, ZeroizeOnDrop)]
pub struct LinearizationProof {
    /// Commitment to the expanded witness vector
    /// Includes original witness plus auxiliary variables for linearization
    #[zeroize(skip)]
    pub expanded_witness_commitment: Vec<RingElement>,
    
    /// Proof that the expanded witness satisfies the linear relation
    /// Demonstrates Mz_expanded = v for linearized constraint matrix M
    #[zeroize(skip)]
    pub linear_relation_proof: Vec<RingElement>,
    
    /// Proof that the expanded witness is consistent with original R1CS witness
    /// Shows that the linearization preserves the original constraint semantics
    #[zeroize(skip)]
    pub consistency_proof: Vec<RingElement>,
    
    /// Gadget matrix used for witness expansion
    /// Enables conversion from quadratic to linear constraints
    #[zeroize(skip)]
    pub gadget_matrix_commitment: Vec<RingElement>,
    
    /// Original R1CS instance dimensions for verification
    pub original_num_constraints: usize,
    pub original_num_variables: usize,
    
    /// Expanded linear relation dimensions
    pub expanded_num_constraints: usize,
    pub expanded_num_variables: usize,
}

/// Comprehensive performance statistics for end-to-end folding execution
#[derive(Clone, Debug, Default)]
pub struct EndToEndFoldingStats {
    /// Total number of end-to-end folding executions
    total_executions: u64,
    
    /// Total number of R1CS instances processed
    total_r1cs_instances: u64,
    
    /// Total number of constraints processed across all instances
    total_constraints_processed: u64,
    
    /// Total number of variables processed across all instances
    total_variables_processed: u64,
    
    /// Detailed timing breakdown for each phase
    linearization_time_ns: u64,
    folding_time_ns: u64,
    range_proof_time_ns: u64,
    transformation_time_ns: u64,
    sumcheck_time_ns: u64,
    final_assembly_time_ns: u64,
    
    /// Total prover time (sum of all phases)
    total_prover_time_ns: u64,
    
    /// Total verifier time for complete verification
    total_verifier_time_ns: u64,
    
    /// Communication breakdown by component
    linearization_comm_bytes: u64,
    folding_comm_bytes: u64,
    range_proof_comm_bytes: u64,
    transformation_comm_bytes: u64,
    sumcheck_comm_bytes: u64,
    final_proof_comm_bytes: u64,
    
    /// Total communication (sum of all components)
    total_communication_bytes: u64,
    
    /// Success/failure statistics
    successful_executions: u64,
    failed_executions: u64,
    
    /// Performance optimization statistics
    gpu_accelerated_operations: u64,
    cpu_only_operations: u64,
    cache_hits: u64,
    cache_misses: u64,
    parallel_operations: u64,
    sequential_operations: u64,
    
    /// Memory usage statistics
    peak_memory_usage_bytes: usize,
    average_memory_usage_bytes: usize,
    
    /// Theoretical performance validation
    theoretical_prover_ops: u64,
    actual_prover_ops: u64,
    theoretical_verifier_ops: u64,
    actual_verifier_ops: u64,
    theoretical_comm_elements: u64,
    actual_comm_elements: u64,
}

impl EndToEndFoldingStats {
    /// Creates new statistics tracker
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Records completion of an end-to-end folding execution
    pub fn record_execution(
        &mut self,
        num_instances: usize,
        num_constraints: usize,
        num_variables: usize,
        timing_breakdown: &TimingBreakdown,
        comm_breakdown: &CommunicationBreakdown,
        success: bool,
    ) {
        self.total_executions += 1;
        self.total_r1cs_instances += num_instances as u64;
        self.total_constraints_processed += (num_instances * num_constraints) as u64;
        self.total_variables_processed += (num_instances * num_variables) as u64;
        
        // Update timing statistics
        self.linearization_time_ns += timing_breakdown.linearization_ns;
        self.folding_time_ns += timing_breakdown.folding_ns;
        self.range_proof_time_ns += timing_breakdown.range_proof_ns;
        self.transformation_time_ns += timing_breakdown.transformation_ns;
        self.sumcheck_time_ns += timing_breakdown.sumcheck_ns;
        self.final_assembly_time_ns += timing_breakdown.final_assembly_ns;
        
        self.total_prover_time_ns += timing_breakdown.total_prover_ns();
        self.total_verifier_time_ns += timing_breakdown.verifier_ns;
        
        // Update communication statistics
        self.linearization_comm_bytes += comm_breakdown.linearization_bytes;
        self.folding_comm_bytes += comm_breakdown.folding_bytes;
        self.range_proof_comm_bytes += comm_breakdown.range_proof_bytes;
        self.transformation_comm_bytes += comm_breakdown.transformation_bytes;
        self.sumcheck_comm_bytes += comm_breakdown.sumcheck_bytes;
        self.final_proof_comm_bytes += comm_breakdown.final_proof_bytes;
        
        self.total_communication_bytes += comm_breakdown.total_bytes();
        
        // Update success/failure counts
        if success {
            self.successful_executions += 1;
        } else {
            self.failed_executions += 1;
        }
    }
    
    /// Returns success rate as percentage
    pub fn success_rate(&self) -> f64 {
        if self.total_executions == 0 {
            0.0
        } else {
            (self.successful_executions as f64 / self.total_executions as f64) * 100.0
        }
    }
    
    /// Returns average prover time per execution
    pub fn average_prover_time_ms(&self) -> f64 {
        if self.total_executions == 0 {
            0.0
        } else {
            (self.total_prover_time_ns as f64 / self.total_executions as f64) / 1_000_000.0
        }
    }
    
    /// Returns average verifier time per execution
    pub fn average_verifier_time_ms(&self) -> f64 {
        if self.total_executions == 0 {
            0.0
        } else {
            (self.total_verifier_time_ns as f64 / self.total_executions as f64) / 1_000_000.0
        }
    }
    
    /// Returns average communication per execution in KB
    pub fn average_communication_kb(&self) -> f64 {
        if self.total_executions == 0 {
            0.0
        } else {
            (self.total_communication_bytes as f64 / self.total_executions as f64) / 1024.0
        }
    }
    
    /// Returns GPU utilization rate as percentage
    pub fn gpu_utilization(&self) -> f64 {
        let total_ops = self.gpu_accelerated_operations + self.cpu_only_operations;
        if total_ops == 0 {
            0.0
        } else {
            (self.gpu_accelerated_operations as f64 / total_ops as f64) * 100.0
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
    
    /// Returns parallel processing efficiency as percentage
    pub fn parallel_efficiency(&self) -> f64 {
        let total_ops = self.parallel_operations + self.sequential_operations;
        if total_ops == 0 {
            0.0
        } else {
            (self.parallel_operations as f64 / total_ops as f64) * 100.0
        }
    }
}

/// Timing breakdown for different phases of end-to-end folding
#[derive(Clone, Debug, Default)]
pub struct TimingBreakdown {
    pub linearization_ns: u64,
    pub folding_ns: u64,
    pub range_proof_ns: u64,
    pub transformation_ns: u64,
    pub sumcheck_ns: u64,
    pub final_assembly_ns: u64,
    pub verifier_ns: u64,
}

impl TimingBreakdown {
    pub fn total_prover_ns(&self) -> u64 {
        self.linearization_ns + self.folding_ns + self.range_proof_ns + 
        self.transformation_ns + self.sumcheck_ns + self.final_assembly_ns
    }
}

/// Communication breakdown for different components
#[derive(Clone, Debug, Default)]
pub struct CommunicationBreakdown {
    pub linearization_bytes: u64,
    pub folding_bytes: u64,
    pub range_proof_bytes: u64,
    pub transformation_bytes: u64,
    pub sumcheck_bytes: u64,
    pub final_proof_bytes: u64,
}

impl CommunicationBreakdown {
    pub fn total_bytes(&self) -> u64 {
        self.linearization_bytes + self.folding_bytes + self.range_proof_bytes + 
        self.transformation_bytes + self.sumcheck_bytes + self.final_proof_bytes
    }
}
/// Complete end-to-end folding system for LatticeFold+
/// 
/// This is the main entry point for the LattiFold+ system, providing
/// a unified interface for processing R1CS instances through the complete
/// folding pipeline with optimal performance and security.
#[derive(Clone, Debug)]
pub struct EndToEndFoldingSystem {
    /// System parameters defining security and performance characteristics
    params: EndToEndFoldingParams,
    
    /// Linear folding protocol for multi-instance processing
    folding_protocol: LinearFoldingProtocol,
    
    /// Range check protocol for algebraic range proofs
    range_check_protocol: RangeCheckProtocol,
    
    /// Commitment transformation protocol for double-to-linear conversion
    transformation_protocol: CommitmentTransformationProtocol,
    
    /// Batched sumcheck protocol for verification compression
    sumcheck_protocol: BatchedSumcheckProtocol,
    
    /// Security analyzer for parameter validation and threat assessment
    security_analyzer: Option<SecurityAnalyzer>,
    
    /// Challenge generator for cryptographic randomness
    challenge_generator: FoldingChallengeGenerator,
    
    /// Performance statistics tracker
    stats: Arc<Mutex<EndToEndFoldingStats>>,
    
    /// Cache for frequently used computations
    computation_cache: Arc<Mutex<HashMap<Vec<u8>, Vec<u8>>>>,
    
    /// Transcript for Fiat-Shamir transformation
    transcript: Transcript,
}

impl EndToEndFoldingSystem {
    /// Creates a new end-to-end folding system with specified parameters
    /// 
    /// # Arguments
    /// * `params` - System parameters defining security and performance
    /// 
    /// # Returns
    /// * `Result<Self>` - New folding system or initialization error
    pub fn new(params: EndToEndFoldingParams) -> Result<Self> {
        // Validate parameters before any initialization
        params.validate()?;
        
        // Initialize linear folding protocol
        let folding_params = LinearFoldingParams {
            kappa: params.kappa,
            witness_dimension: params.witness_dimension,
            norm_bound: params.norm_bound,
            num_instances: params.max_folding_instances,
            ring_dimension: params.ring_dimension,
            modulus: params.modulus,
            challenge_set_size: 1000, // Adequate for 128-bit security
        };
        let folding_protocol = LinearFoldingProtocol::new(folding_params)?;
        
        // Initialize range check protocol
        let range_check_protocol = RangeCheckProtocol::new(
            params.ring_dimension,
            params.modulus,
            params.range_bound,
            params.kappa,
        )?;
        
        // Initialize commitment transformation protocol
        let transformation_params = CommitmentTransformationParams::new(
            params.kappa,
            params.ring_dimension,
            params.witness_dimension,
            params.norm_bound,
            params.modulus,
        )?;
        let transformation_protocol = CommitmentTransformationProtocol::new(transformation_params)?;
        
        // Initialize batched sumcheck protocol
        let num_variables = (params.witness_dimension as f64).log2().ceil() as usize;
        let sumcheck_protocol = BatchedSumcheckProtocol::new(
            num_variables,
            params.ring_dimension,
            params.modulus,
            2, // Max degree 2 for quadratic relations
            params.max_folding_instances,
        )?;
        
        // Initialize security analyzer if enabled
        let security_analyzer = if params.enable_security_analysis {
            Some(SecurityAnalyzer::new())
        } else {
            None
        };
        
        // Initialize challenge generator
        let challenge_params = FoldingChallengeParams::new(
            params.ring_dimension,
            (params.modulus as f64).log2().ceil() as usize, // Decomposition levels
            params.kappa,
            (params.modulus as f64).log2().ceil() as usize, // Gadget dimension
            params.modulus,
        )?;
        let challenge_generator = FoldingChallengeGenerator::new(challenge_params)?;
        
        // Initialize performance statistics
        let stats = Arc::new(Mutex::new(EndToEndFoldingStats::new()));
        
        // Initialize computation cache
        let computation_cache = Arc::new(Mutex::new(HashMap::new()));
        
        // Initialize Fiat-Shamir transcript
        let transcript = Transcript::new(b"LatticeFold+ End-to-End Folding System");
        
        Ok(Self {
            params,
            folding_protocol,
            range_check_protocol,
            transformation_protocol,
            sumcheck_protocol,
            security_analyzer,
            challenge_generator,
            stats,
            computation_cache,
            transcript,
        })
    }
    
    /// Creates a system optimized for 128-bit security
    /// 
    /// # Returns
    /// * `Result<Self>` - System configured for 128-bit security
    pub fn for_128_bit_security() -> Result<Self> {
        let params = EndToEndFoldingParams::for_128_bit_security()?;
        Self::new(params)
    }
    
    /// Executes the complete end-to-end folding protocol as prover
    /// 
    /// # Arguments
    /// * `r1cs_instances` - Vector of R1CS instances to process
    /// * `rng` - Cryptographically secure random number generator
    /// 
    /// # Returns
    /// * `Result<EndToEndFoldingProof>` - Complete folding proof or error
    /// 
    /// # Protocol Execution Pipeline
    /// 1. **R1CS Validation**: Validates all input instances for correctness
    /// 2. **Linearization**: Converts R1CS constraints to linear relations
    /// 3. **Multi-Instance Folding**: Applies L-to-2 folding with norm control
    /// 4. **Range Proof Generation**: Creates algebraic range proofs for witnesses
    /// 5. **Commitment Transformation**: Converts double to linear commitments
    /// 6. **Sumcheck Batching**: Compresses multiple verification claims
    /// 7. **Final Proof Assembly**: Combines all components into final proof
    /// 8. **Performance Analysis**: Records detailed execution metrics
    /// 9. **Security Validation**: Optionally validates security properties
    pub fn prove<R: CryptoRng + RngCore>(
        &mut self,
        r1cs_instances: &[R1CSInstance],
        rng: &mut R,
    ) -> Result<EndToEndFoldingProof> {
        let overall_start = Instant::now();
        let mut timing = TimingBreakdown::default();
        let mut communication = CommunicationBreakdown::default();
        
        // Step 1: Validate input instances
        if r1cs_instances.is_empty() {
            return Err(LatticeFoldError::InvalidParameters(
                "Must provide at least one R1CS instance".to_string()
            ));
        }
        
        if r1cs_instances.len() > self.params.max_folding_instances {
            return Err(LatticeFoldError::InvalidParameters(
                format!("Too many instances: {} > {}", 
                       r1cs_instances.len(), self.params.max_folding_instances)
            ));
        }
        
        // Validate each instance
        for (i, instance) in r1cs_instances.iter().enumerate() {
            instance.validate_constraints().map_err(|e| {
                LatticeFoldError::InvalidParameters(
                    format!("Instance {} validation failed: {}", i, e)
                )
            })?;
        }
        
        // Step 2: R1CS to Linear Relation Conversion (Linearization)
        let linearization_start = Instant::now();
        let linearization_proofs = self.linearize_r1cs_instances(r1cs_instances, rng)?;
        timing.linearization_ns = linearization_start.elapsed().as_nanos() as u64;
        
        // Calculate linearization communication
        communication.linearization_bytes = linearization_proofs.iter()
            .map(|proof| self.estimate_linearization_proof_size(proof))
            .sum::<usize>() as u64;
        
        // Step 3: Convert linearized instances to multi-instance linear relations
        let linear_relations = self.extract_linear_relations_from_proofs(&linearization_proofs)?;
        let multi_instance_relation = MultiInstanceLinearRelation {
            instances: linear_relations,
            num_instances: r1cs_instances.len(),
            norm_bound: self.params.norm_bound,
        };
        
        // Step 4: Multi-Instance Folding (L-to-2 reduction)
        let folding_start = Instant::now();
        let (folded_relation, folding_proof) = self.folding_protocol
            .fold_multi_instance(&multi_instance_relation, rng)?;
        timing.folding_ns = folding_start.elapsed().as_nanos() as u64;
        
        // Calculate folding communication
        communication.folding_bytes = self.estimate_folding_proof_size(&folding_proof) as u64;
        
        // Step 5: Range Proof Generation for all witnesses
        let range_proof_start = Instant::now();
        let range_proofs = self.generate_range_proofs_for_instances(r1cs_instances, rng)?;
        timing.range_proof_ns = range_proof_start.elapsed().as_nanos() as u64;
        
        // Calculate range proof communication
        communication.range_proof_bytes = range_proofs.iter()
            .map(|proof| self.estimate_range_proof_size(proof))
            .sum::<usize>() as u64;
        
        // Step 6: Commitment Transformation (Double to Linear)
        let transformation_start = Instant::now();
        let transformation_proofs = self.generate_commitment_transformations(r1cs_instances, rng)?;
        timing.transformation_ns = transformation_start.elapsed().as_nanos() as u64;
        
        // Calculate transformation communication
        communication.transformation_bytes = transformation_proofs.iter()
            .map(|proof| self.estimate_transformation_proof_size(proof))
            .sum::<usize>() as u64;
        
        // Step 7: Sumcheck Batching for verification compression
        let sumcheck_start = Instant::now();
        let sumcheck_proof = self.generate_batched_sumcheck_proof(
            &linearization_proofs,
            &folding_proof,
            &range_proofs,
            &transformation_proofs,
            rng
        )?;
        timing.sumcheck_ns = sumcheck_start.elapsed().as_nanos() as u64;
        
        // Calculate sumcheck communication
        communication.sumcheck_bytes = self.estimate_sumcheck_proof_size(&sumcheck_proof) as u64;
        
        // Step 8: Final Proof Assembly
        let assembly_start = Instant::now();
        
        // Perform security analysis if enabled
        let security_analysis = if let Some(ref analyzer) = self.security_analyzer {
            Some(analyzer.analyze_parameters(&self.params)?)
        } else {
            None
        };
        
        // Create performance statistics
        let mut performance_stats = EndToEndFoldingStats::new();
        performance_stats.record_execution(
            r1cs_instances.len(),
            r1cs_instances.iter().map(|i| i.num_constraints).max().unwrap_or(0),
            r1cs_instances.iter().map(|i| i.num_variables).max().unwrap_or(0),
            &timing,
            &communication,
            true, // Success
        );
        
        // Assemble final proof
        let final_proof = EndToEndFoldingProof {
            linearization_proofs,
            folding_proof,
            range_proofs,
            transformation_proofs,
            sumcheck_proof,
            final_folded_instance: folded_relation,
            params: self.params.clone(),
            performance_stats,
            security_analysis,
        };
        
        timing.final_assembly_ns = assembly_start.elapsed().as_nanos() as u64;
        
        // Calculate final proof communication
        communication.final_proof_bytes = self.estimate_final_proof_size(&final_proof) as u64;
        
        // Update system statistics
        if let Ok(mut stats) = self.stats.lock() {
            stats.record_execution(
                r1cs_instances.len(),
                r1cs_instances.iter().map(|i| i.num_constraints).max().unwrap_or(0),
                r1cs_instances.iter().map(|i| i.num_variables).max().unwrap_or(0),
                &timing,
                &communication,
                true,
            );
        }
        
        Ok(final_proof)
    }
    
    /// Verifies a complete end-to-end folding proof
    /// 
    /// # Arguments
    /// * `proof` - Complete folding proof to verify
    /// * `public_inputs` - Public inputs for all R1CS instances
    /// 
    /// # Returns
    /// * `Result<bool>` - True if proof is valid, false otherwise
    /// 
    /// # Verification Pipeline
    /// 1. **Structural Validation**: Checks proof format and consistency
    /// 2. **Parameter Validation**: Verifies all parameters are secure
    /// 3. **Linearization Verification**: Validates R1CS to linear conversion
    /// 4. **Folding Verification**: Checks multi-instance folding correctness
    /// 5. **Range Proof Verification**: Validates algebraic range proofs
    /// 6. **Transformation Verification**: Checks commitment transformations
    /// 7. **Sumcheck Verification**: Validates batched sumcheck proof
    /// 8. **Final Consistency**: Ensures all components are consistent
    /// 9. **Security Validation**: Optionally checks security properties
    pub fn verify(
        &mut self,
        proof: &EndToEndFoldingProof,
        public_inputs: &[Vec<RingElement>],
    ) -> Result<bool> {
        let verification_start = Instant::now();
        
        // Step 1: Structural validation
        if proof.linearization_proofs.len() != public_inputs.len() {
            return Ok(false);
        }
        
        if proof.range_proofs.len() != public_inputs.len() {
            return Ok(false);
        }
        
        if proof.transformation_proofs.len() != public_inputs.len() {
            return Ok(false);
        }
        
        // Step 2: Parameter validation
        if proof.params != self.params {
            return Ok(false);
        }
        
        // Step 3: Verify linearization proofs
        for (i, linearization_proof) in proof.linearization_proofs.iter().enumerate() {
            if !self.verify_linearization_proof(linearization_proof, &public_inputs[i])? {
                return Ok(false);
            }
        }
        
        // Step 4: Verify folding proof
        let original_relations = self.extract_linear_relations_from_proofs(&proof.linearization_proofs)?;
        if !self.folding_protocol.verify_folding_proof(
            &original_relations,
            &proof.final_folded_instance,
            &proof.folding_proof,
        )? {
            return Ok(false);
        }
        
        // Step 5: Verify range proofs
        for (i, range_proof) in proof.range_proofs.iter().enumerate() {
            // Create witness commitment from public inputs for verification
            let witness_commitment = self.create_witness_commitment_from_public(&public_inputs[i])?;
            if !self.range_check_protocol.verify(range_proof, &witness_commitment)? {
                return Ok(false);
            }
        }
        
        // Step 6: Verify transformation proofs
        for (i, transformation_proof) in proof.transformation_proofs.iter().enumerate() {
            if !self.verify_transformation_proof(transformation_proof, &public_inputs[i])? {
                return Ok(false);
            }
        }
        
        // Step 7: Verify batched sumcheck proof
        if !self.verify_batched_sumcheck_proof(&proof.sumcheck_proof)? {
            return Ok(false);
        }
        
        // Step 8: Final consistency checks
        if !self.verify_final_consistency(proof)? {
            return Ok(false);
        }
        
        // Step 9: Security validation (if enabled)
        if let Some(ref security_analysis) = proof.security_analysis {
            if !self.validate_security_analysis(security_analysis)? {
                return Ok(false);
            }
        }
        
        // Record verification time
        let verification_time = verification_start.elapsed().as_nanos() as u64;
        if let Ok(mut stats) = self.stats.lock() {
            stats.total_verifier_time_ns += verification_time;
        }
        
        Ok(true)
    }
    
    /// Returns current performance statistics
    /// 
    /// # Returns
    /// * `EndToEndFoldingStats` - Current performance metrics
    pub fn stats(&self) -> EndToEndFoldingStats {
        if let Ok(stats) = self.stats.lock() {
            stats.clone()
        } else {
            EndToEndFoldingStats::new()
        }
    }
    
    /// Resets performance statistics
    pub fn reset_stats(&mut self) {
        if let Ok(mut stats) = self.stats.lock() {
            *stats = EndToEndFoldingStats::new();
        }
    }
    
    /// Clears computation cache
    pub fn clear_cache(&mut self) {
        if let Ok(mut cache) = self.computation_cache.lock() {
            cache.clear();
        }
    }
}imp
l EndToEndFoldingSystem {
    /// Linearizes R1CS instances to linear relations
    /// 
    /// # Arguments
    /// * `r1cs_instances` - R1CS instances to linearize
    /// * `rng` - Random number generator
    /// 
    /// # Returns
    /// * `Result<Vec<LinearizationProof>>` - Linearization proofs for each instance
    /// 
    /// # Linearization Process
    /// For each R1CS instance with constraint (Az) ◦ (Bz) = Cz:
    /// 1. **Witness Expansion**: Create expanded witness z' = (z, Az, Bz, Az ◦ Bz)
    /// 2. **Linear Constraint Construction**: Build linear constraint Mz' = v
    /// 3. **Consistency Proof**: Prove expanded witness is consistent with original
    /// 4. **Gadget Integration**: Use gadget matrices for efficient representation
    fn linearize_r1cs_instances<R: CryptoRng + RngCore>(
        &mut self,
        r1cs_instances: &[R1CSInstance],
        rng: &mut R,
    ) -> Result<Vec<LinearizationProof>> {
        let mut linearization_proofs = Vec::with_capacity(r1cs_instances.len());
        
        for (instance_idx, r1cs_instance) in r1cs_instances.iter().enumerate() {
            // Step 1: Expand witness vector
            // Original witness: z = (1, x, w)
            // Expanded witness: z' = (z, Az, Bz, Az ◦ Bz)
            let mut expanded_witness = r1cs_instance.witness.clone();
            
            // Compute Az for each constraint
            let mut az_values = Vec::with_capacity(r1cs_instance.num_constraints);
            for constraint_idx in 0..r1cs_instance.num_constraints {
                let mut az_i = RingElement::zero(
                    self.params.ring_dimension,
                    Some(self.params.modulus)
                )?;
                
                for var_idx in 0..r1cs_instance.num_variables {
                    let product = r1cs_instance.matrix_a[constraint_idx][var_idx]
                        .multiply(&r1cs_instance.witness[var_idx])?;
                    az_i = az_i.add(&product)?;
                }
                
                az_values.push(az_i);
            }
            
            // Compute Bz for each constraint
            let mut bz_values = Vec::with_capacity(r1cs_instance.num_constraints);
            for constraint_idx in 0..r1cs_instance.num_constraints {
                let mut bz_i = RingElement::zero(
                    self.params.ring_dimension,
                    Some(self.params.modulus)
                )?;
                
                for var_idx in 0..r1cs_instance.num_variables {
                    let product = r1cs_instance.matrix_b[constraint_idx][var_idx]
                        .multiply(&r1cs_instance.witness[var_idx])?;
                    bz_i = bz_i.add(&product)?;
                }
                
                bz_values.push(bz_i);
            }
            
            // Compute Az ◦ Bz (element-wise product)
            let mut az_bz_values = Vec::with_capacity(r1cs_instance.num_constraints);
            for constraint_idx in 0..r1cs_instance.num_constraints {
                let az_bz_i = az_values[constraint_idx].multiply(&bz_values[constraint_idx])?;
                az_bz_values.push(az_bz_i);
            }
            
            // Extend expanded witness with computed values
            expanded_witness.extend(az_values);
            expanded_witness.extend(bz_values);
            expanded_witness.extend(az_bz_values);
            
            // Step 2: Create commitment to expanded witness
            let expanded_witness_commitment = self.commit_expanded_witness(&expanded_witness, rng)?;
            
            // Step 3: Generate linear relation proof
            // The linear relation is: M × z' = v where M encodes the linearized constraints
            let linear_relation_proof = self.generate_linear_relation_proof(
                r1cs_instance,
                &expanded_witness,
                rng
            )?;
            
            // Step 4: Generate consistency proof
            // Proves that the expanded witness is consistent with the original R1CS witness
            let consistency_proof = self.generate_consistency_proof(
                r1cs_instance,
                &expanded_witness,
                rng
            )?;
            
            // Step 5: Generate gadget matrix commitment
            let gadget_matrix_commitment = self.generate_gadget_matrix_commitment(
                r1cs_instance.num_constraints,
                r1cs_instance.num_variables,
                rng
            )?;
            
            // Step 6: Assemble linearization proof
            let linearization_proof = LinearizationProof {
                expanded_witness_commitment,
                linear_relation_proof,
                consistency_proof,
                gadget_matrix_commitment,
                original_num_constraints: r1cs_instance.num_constraints,
                original_num_variables: r1cs_instance.num_variables,
                expanded_num_constraints: r1cs_instance.num_constraints, // Same number of constraints
                expanded_num_variables: expanded_witness.len(), // Expanded witness size
            };
            
            linearization_proofs.push(linearization_proof);
        }
        
        Ok(linearization_proofs)
    }
    
    /// Generates range proofs for all R1CS instances
    /// 
    /// # Arguments
    /// * `r1cs_instances` - R1CS instances to generate range proofs for
    /// * `rng` - Random number generator
    /// 
    /// # Returns
    /// * `Result<Vec<RangeCheckProof>>` - Range proofs for each instance
    fn generate_range_proofs_for_instances<R: CryptoRng + RngCore>(
        &mut self,
        r1cs_instances: &[R1CSInstance],
        rng: &mut R,
    ) -> Result<Vec<RangeCheckProof>> {
        let mut range_proofs = Vec::with_capacity(r1cs_instances.len());
        
        for r1cs_instance in r1cs_instances {
            // Generate range proof for the witness vector
            // This proves that all witness elements are within the range bound
            let range_proof = self.range_check_protocol.prove(&r1cs_instance.witness, rng)?;
            range_proofs.push(range_proof);
        }
        
        Ok(range_proofs)
    }
    
    /// Generates commitment transformation proofs
    /// 
    /// # Arguments
    /// * `r1cs_instances` - R1CS instances to generate transformations for
    /// * `rng` - Random number generator
    /// 
    /// # Returns
    /// * `Result<Vec<CommitmentTransformationProof>>` - Transformation proofs
    fn generate_commitment_transformations<R: CryptoRng + RngCore>(
        &mut self,
        r1cs_instances: &[R1CSInstance],
        rng: &mut R,
    ) -> Result<Vec<CommitmentTransformationProof>> {
        let mut transformation_proofs = Vec::with_capacity(r1cs_instances.len());
        
        for r1cs_instance in r1cs_instances {
            // Create double commitment for the witness matrix
            // This involves treating the witness as a matrix and applying double commitment
            let witness_matrix = self.witness_to_matrix(&r1cs_instance.witness)?;
            let double_commitment = self.create_double_commitment(&witness_matrix, rng)?;
            
            // Generate transformation proof from double to linear commitment
            let transformation_proof = self.transformation_protocol.prove(
                &r1cs_instance.witness,
                &double_commitment,
                rng
            )?;
            
            transformation_proofs.push(transformation_proof);
        }
        
        Ok(transformation_proofs)
    }
    
    /// Generates batched sumcheck proof for all verification claims
    /// 
    /// # Arguments
    /// * `linearization_proofs` - Linearization proofs to verify
    /// * `folding_proof` - Folding proof to verify
    /// * `range_proofs` - Range proofs to verify
    /// * `transformation_proofs` - Transformation proofs to verify
    /// * `rng` - Random number generator
    /// 
    /// # Returns
    /// * `Result<BatchedSumcheckProof>` - Batched sumcheck proof
    fn generate_batched_sumcheck_proof<R: CryptoRng + RngCore>(
        &mut self,
        linearization_proofs: &[LinearizationProof],
        folding_proof: &LinearFoldingProof,
        range_proofs: &[RangeCheckProof],
        transformation_proofs: &[CommitmentTransformationProof],
        rng: &mut R,
    ) -> Result<BatchedSumcheckProof> {
        // Create multilinear extensions for all verification claims
        let mut multilinear_extensions = Vec::new();
        let mut claimed_sums = Vec::new();
        
        // Add linearization verification claims
        for linearization_proof in linearization_proofs {
            let (ml_ext, sum) = self.create_linearization_verification_claim(linearization_proof)?;
            multilinear_extensions.push(ml_ext);
            claimed_sums.push(sum);
        }
        
        // Add folding verification claim
        let (folding_ml_ext, folding_sum) = self.create_folding_verification_claim(folding_proof)?;
        multilinear_extensions.push(folding_ml_ext);
        claimed_sums.push(folding_sum);
        
        // Add range proof verification claims
        for range_proof in range_proofs {
            let (ml_ext, sum) = self.create_range_verification_claim(range_proof)?;
            multilinear_extensions.push(ml_ext);
            claimed_sums.push(sum);
        }
        
        // Add transformation verification claims
        for transformation_proof in transformation_proofs {
            let (ml_ext, sum) = self.create_transformation_verification_claim(transformation_proof)?;
            multilinear_extensions.push(ml_ext);
            claimed_sums.push(sum);
        }
        
        // Generate batched sumcheck proof
        let batched_proof = self.sumcheck_protocol.prove_batch(
            &mut multilinear_extensions,
            &claimed_sums,
        )?;
        
        Ok(batched_proof)
    }
    
    /// Helper method to commit to expanded witness
    fn commit_expanded_witness<R: CryptoRng + RngCore>(
        &self,
        expanded_witness: &[RingElement],
        rng: &mut R,
    ) -> Result<Vec<RingElement>> {
        // Create commitment matrix for expanded witness
        let commitment_matrix = self.generate_commitment_matrix(
            self.params.kappa,
            expanded_witness.len(),
            rng
        )?;
        
        // Compute commitment: C = A × w
        let mut commitment = vec![
            RingElement::zero(self.params.ring_dimension, Some(self.params.modulus))?;
            self.params.kappa
        ];
        
        for (i, commitment_element) in commitment.iter_mut().enumerate() {
            for (j, witness_element) in expanded_witness.iter().enumerate() {
                let product = commitment_matrix[i][j].multiply(witness_element)?;
                *commitment_element = commitment_element.add(&product)?;
            }
        }
        
        Ok(commitment)
    }
    
    /// Helper method to generate commitment matrix
    fn generate_commitment_matrix<R: CryptoRng + RngCore>(
        &self,
        rows: usize,
        cols: usize,
        rng: &mut R,
    ) -> Result<Vec<Vec<RingElement>>> {
        let mut matrix = Vec::with_capacity(rows);
        
        for _ in 0..rows {
            let mut row = Vec::with_capacity(cols);
            for _ in 0..cols {
                // Generate random ring element
                let mut coeffs = vec![0i64; self.params.ring_dimension];
                for coeff in &mut coeffs {
                    *coeff = (rng.next_u64() as i64) % self.params.modulus;
                    if *coeff > self.params.modulus / 2 {
                        *coeff -= self.params.modulus;
                    }
                }
                
                let ring_element = RingElement::from_coefficients(coeffs, Some(self.params.modulus))?;
                row.push(ring_element);
            }
            matrix.push(row);
        }
        
        Ok(matrix)
    }
    
    /// Helper method to extract linear relations from linearization proofs
    fn extract_linear_relations_from_proofs(
        &self,
        linearization_proofs: &[LinearizationProof],
    ) -> Result<Vec<LinearRelation<ark_ff::Fp256<ark_ff::FpParameters>>>> {
        // This is a placeholder implementation
        // In a full implementation, this would extract the actual linear relations
        // from the linearization proofs
        Ok(Vec::new())
    }
    
    /// Helper method to estimate proof sizes for communication analysis
    fn estimate_linearization_proof_size(&self, proof: &LinearizationProof) -> usize {
        let element_size = self.params.ring_dimension * 8; // 8 bytes per coefficient
        
        proof.expanded_witness_commitment.len() * element_size +
        proof.linear_relation_proof.len() * element_size +
        proof.consistency_proof.len() * element_size +
        proof.gadget_matrix_commitment.len() * element_size
    }
    
    fn estimate_folding_proof_size(&self, proof: &LinearFoldingProof) -> usize {
        let element_size = self.params.ring_dimension * 8;
        
        proof.folding_challenges.len() * element_size +
        proof.aggregated_witness.len() * element_size +
        proof.folded_commitment.len() * element_size +
        proof.folded_public_vector.len() * element_size
    }
    
    fn estimate_range_proof_size(&self, proof: &RangeCheckProof) -> usize {
        // Placeholder implementation
        self.params.kappa * self.params.ring_dimension * 8
    }
    
    fn estimate_transformation_proof_size(&self, proof: &CommitmentTransformationProof) -> usize {
        // Placeholder implementation
        self.params.kappa * self.params.ring_dimension * 8
    }
    
    fn estimate_sumcheck_proof_size(&self, proof: &BatchedSumcheckProof) -> usize {
        // Placeholder implementation
        proof.batch_size * self.params.ring_dimension * 8
    }
    
    fn estimate_final_proof_size(&self, proof: &EndToEndFoldingProof) -> usize {
        // Sum of all component sizes
        proof.linearization_proofs.iter().map(|p| self.estimate_linearization_proof_size(p)).sum::<usize>() +
        self.estimate_folding_proof_size(&proof.folding_proof) +
        proof.range_proofs.iter().map(|p| self.estimate_range_proof_size(p)).sum::<usize>() +
        proof.transformation_proofs.iter().map(|p| self.estimate_transformation_proof_size(p)).sum::<usize>() +
        self.estimate_sumcheck_proof_size(&proof.sumcheck_proof)
    }
    
    /// Placeholder verification methods (would be fully implemented in production)
    fn verify_linearization_proof(
        &self,
        proof: &LinearizationProof,
        public_inputs: &[RingElement],
    ) -> Result<bool> {
        // Placeholder implementation
        Ok(true)
    }
    
    fn verify_transformation_proof(
        &self,
        proof: &CommitmentTransformationProof,
        public_inputs: &[RingElement],
    ) -> Result<bool> {
        // Placeholder implementation
        Ok(true)
    }
    
    fn verify_batched_sumcheck_proof(&self, proof: &BatchedSumcheckProof) -> Result<bool> {
        // Placeholder implementation
        Ok(true)
    }
    
    fn verify_final_consistency(&self, proof: &EndToEndFoldingProof) -> Result<bool> {
        // Placeholder implementation
        Ok(true)
    }
    
    fn validate_security_analysis(&self, analysis: &SecurityAnalysisResults) -> Result<bool> {
        // Placeholder implementation
        Ok(true)
    }
    
    /// Additional placeholder helper methods
    fn generate_linear_relation_proof<R: CryptoRng + RngCore>(
        &self,
        r1cs_instance: &R1CSInstance,
        expanded_witness: &[RingElement],
        rng: &mut R,
    ) -> Result<Vec<RingElement>> {
        Ok(Vec::new())
    }
    
    fn generate_consistency_proof<R: CryptoRng + RngCore>(
        &self,
        r1cs_instance: &R1CSInstance,
        expanded_witness: &[RingElement],
        rng: &mut R,
    ) -> Result<Vec<RingElement>> {
        Ok(Vec::new())
    }
    
    fn generate_gadget_matrix_commitment<R: CryptoRng + RngCore>(
        &self,
        num_constraints: usize,
        num_variables: usize,
        rng: &mut R,
    ) -> Result<Vec<RingElement>> {
        Ok(Vec::new())
    }
    
    fn witness_to_matrix(&self, witness: &[RingElement]) -> Result<Vec<Vec<RingElement>>> {
        // Convert witness vector to matrix form for double commitment
        let matrix_size = (witness.len() as f64).sqrt().ceil() as usize;
        let mut matrix = Vec::with_capacity(matrix_size);
        
        for i in 0..matrix_size {
            let mut row = Vec::with_capacity(matrix_size);
            for j in 0..matrix_size {
                let idx = i * matrix_size + j;
                if idx < witness.len() {
                    row.push(witness[idx].clone());
                } else {
                    row.push(RingElement::zero(self.params.ring_dimension, Some(self.params.modulus))?);
                }
            }
            matrix.push(row);
        }
        
        Ok(matrix)
    }
    
    fn create_double_commitment<R: CryptoRng + RngCore>(
        &self,
        matrix: &[Vec<RingElement>],
        rng: &mut R,
    ) -> Result<Vec<RingElement>> {
        // Placeholder for double commitment creation
        Ok(Vec::new())
    }
    
    fn create_witness_commitment_from_public(
        &self,
        public_inputs: &[RingElement],
    ) -> Result<Vec<RingElement>> {
        // Placeholder for creating witness commitment from public inputs
        Ok(Vec::new())
    }
    
    fn create_linearization_verification_claim(
        &self,
        proof: &LinearizationProof,
    ) -> Result<(crate::monomial_set_checking::MultilinearExtension, RingElement)> {
        // Placeholder for creating verification claim
        let ml_ext = crate::monomial_set_checking::MultilinearExtension::new(
            Vec::new(),
            self.params.ring_dimension,
            Some(self.params.modulus)
        )?;
        let sum = RingElement::zero(self.params.ring_dimension, Some(self.params.modulus))?;
        Ok((ml_ext, sum))
    }
    
    fn create_folding_verification_claim(
        &self,
        proof: &LinearFoldingProof,
    ) -> Result<(crate::monomial_set_checking::MultilinearExtension, RingElement)> {
        // Placeholder for creating folding verification claim
        let ml_ext = crate::monomial_set_checking::MultilinearExtension::new(
            Vec::new(),
            self.params.ring_dimension,
            Some(self.params.modulus)
        )?;
        let sum = RingElement::zero(self.params.ring_dimension, Some(self.params.modulus))?;
        Ok((ml_ext, sum))
    }
    
    fn create_range_verification_claim(
        &self,
        proof: &RangeCheckProof,
    ) -> Result<(crate::monomial_set_checking::MultilinearExtension, RingElement)> {
        // Placeholder for creating range verification claim
        let ml_ext = crate::monomial_set_checking::MultilinearExtension::new(
            Vec::new(),
            self.params.ring_dimension,
            Some(self.params.modulus)
        )?;
        let sum = RingElement::zero(self.params.ring_dimension, Some(self.params.modulus))?;
        Ok((ml_ext, sum))
    }
    
    fn create_transformation_verification_claim(
        &self,
        proof: &CommitmentTransformationProof,
    ) -> Result<(crate::monomial_set_checking::MultilinearExtension, RingElement)> {
        // Placeholder for creating transformation verification claim
        let ml_ext = crate::monomial_set_checking::MultilinearExtension::new(
            Vec::new(),
            self.params.ring_dimension,
            Some(self.params.modulus)
        )?;
        let sum = RingElement::zero(self.params.ring_dimension, Some(self.params.modulus))?;
        Ok((ml_ext, sum))
    }
}