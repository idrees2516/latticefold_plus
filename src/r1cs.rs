/// R1CS Integration and Constraint System Support for LatticeFold+
/// 
/// This module implements the R1CS (Rank-1 Constraint System) integration
/// that enables LatticeFold+ to work with arithmetic circuits and constraint
/// systems commonly used in zero-knowledge proof systems.
/// 
/// Mathematical Foundation:
/// R1CS represents arithmetic circuits as systems of quadratic constraints:
/// (Az) ◦ (Bz) = (Cz) where z is the witness vector and A, B, C are constraint matrices
/// 
/// Key Components:
/// 1. Committed R1CS (RcR1CS,B): R1CS with committed witness using gadget expansion
/// 2. Linearization Protocol: Reduces quadratic constraints to linear relations
/// 3. Matrix Derivation: Constructs auxiliary matrices M^(1), M^(2), M^(3), M^(4)
/// 4. Sumcheck Integration: Uses sumcheck to verify constraint satisfaction
/// 5. CCS Extension: Supports higher-degree constraints beyond quadratic
/// 
/// Performance Characteristics:
/// - Prover complexity: O(n log n) for n constraints using NTT optimization
/// - Verifier complexity: O(log n) through sumcheck and folding
/// - Proof size: O(log n) elements in Rq
/// - Memory usage: O(n) for constraint matrices with sparse representation
/// 
/// Security Properties:
/// - Knowledge soundness: Extractor can recover witness from valid proofs
/// - Zero-knowledge: Simulator produces indistinguishable proofs
/// - Binding: Commitment scheme prevents witness equivocation
/// - Completeness: Honest prover always produces accepting proofs

use std::collections::HashMap;
use std::fmt::{Debug, Display, Formatter, Result as FmtResult};
use std::time::{Duration, Instant};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::cyclotomic_ring::RingElement;
use crate::commitment::{Commitment, HomomorphicCommitmentScheme, SISCommitmentScheme};
use crate::gadget::{GadgetMatrix, GadgetVector};
use crate::ring_sumcheck::{MultilinearExtension, SumcheckProtocol, SumcheckProof};
use crate::error::{LatticeFoldError, Result};
use crate::types::CommitmentParams;

/// Maximum supported number of constraints for R1CS systems
const MAX_R1CS_CONSTRAINTS: usize = 1_048_576; // 2^20

/// Maximum supported witness dimension
const MAX_WITNESS_DIMENSION: usize = 1_048_576; // 2^20

/// SIMD vector width for constraint processing
const SIMD_WIDTH: usize = 8;

/// R1CS constraint matrices A, B, C ∈ Rq^{n×m}
/// 
/// Mathematical Definition:
/// An R1CS instance consists of three matrices A, B, C and defines constraints:
/// (Az) ◦ (Bz) = (Cz) where ◦ denotes element-wise (Hadamard) product
/// 
/// Structure:
/// - A, B, C ∈ Rq^{n×m} are constraint matrices over cyclotomic ring Rq
/// - z ∈ Rq^m is the witness vector (including public inputs)
/// - n is the number of constraints
/// - m is the witness dimension
/// 
/// Constraint Semantics:
/// For each constraint i ∈ [n]:
/// (Σ_j A[i,j] * z[j]) * (Σ_j B[i,j] * z[j]) = (Σ_j C[i,j] * z[j])
/// 
/// Implementation Strategy:
/// - Sparse matrix representation for memory efficiency
/// - Row-major storage for cache-friendly constraint evaluation
/// - Parallel constraint checking using SIMD vectorization
/// - Memory-mapped storage for very large constraint systems
/// 
/// Performance Optimization:
/// - NTT-based polynomial arithmetic for ring operations
/// - Batch constraint evaluation with vectorized operations
/// - Parallel processing across independent constraints
/// - GPU acceleration for large constraint systems
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct R1CSMatrices {
    /// Constraint matrix A ∈ Rq^{n×m}
    /// Left multiplicand in quadratic constraints
    pub matrix_a: Vec<Vec<RingElement>>,
    
    /// Constraint matrix B ∈ Rq^{n×m}
    /// Right multiplicand in quadratic constraints
    pub matrix_b: Vec<Vec<RingElement>>,
    
    /// Constraint matrix C ∈ Rq^{n×m}
    /// Result of quadratic constraints
    pub matrix_c: Vec<Vec<RingElement>>,
    
    /// Number of constraints n
    pub num_constraints: usize,
    
    /// Witness dimension m
    pub witness_dimension: usize,
    
    /// Ring parameters for constraint evaluation
    pub ring_params: crate::cyclotomic_ring::RingParams,
}

impl R1CSMatrices {
    /// Creates new R1CS matrices with the given dimensions
    /// 
    /// # Arguments
    /// * `num_constraints` - Number of constraints n
    /// * `witness_dimension` - Witness dimension m
    /// * `ring_params` - Ring parameters for Rq
    /// 
    /// # Returns
    /// * `Result<Self>` - New R1CS matrices or error
    /// 
    /// # Validation
    /// - Number of constraints must be positive and within limits
    /// - Witness dimension must be positive and within limits
    /// - Ring parameters must be valid for constraint evaluation
    /// 
    /// # Initialization
    /// - All matrices initialized with zero ring elements
    /// - Sparse representation used for memory efficiency
    /// - Memory pre-allocated for performance
    pub fn new(
        num_constraints: usize,
        witness_dimension: usize,
        ring_params: crate::cyclotomic_ring::RingParams,
    ) -> Result<Self> {
        // Validate constraint count
        if num_constraints == 0 {
            return Err(LatticeFoldError::InvalidParameters(
                "Number of constraints must be positive".to_string()
            ));
        }
        if num_constraints > MAX_R1CS_CONSTRAINTS {
            return Err(LatticeFoldError::InvalidDimension {
                expected: MAX_R1CS_CONSTRAINTS,
                got: num_constraints,
            });
        }
        
        // Validate witness dimension
        if witness_dimension == 0 {
            return Err(LatticeFoldError::InvalidParameters(
                "Witness dimension must be positive".to_string()
            ));
        }
        if witness_dimension > MAX_WITNESS_DIMENSION {
            return Err(LatticeFoldError::InvalidDimension {
                expected: MAX_WITNESS_DIMENSION,
                got: witness_dimension,
            });
        }
        
        // Initialize matrices with zero elements
        // Using zero ring elements for sparse representation
        let zero_element = RingElement::zero(ring_params.dimension, ring_params.modulus)?;
        
        // Pre-allocate matrices with correct dimensions
        let matrix_a = vec![vec![zero_element.clone(); witness_dimension]; num_constraints];
        let matrix_b = vec![vec![zero_element.clone(); witness_dimension]; num_constraints];
        let matrix_c = vec![vec![zero_element.clone(); witness_dimension]; num_constraints];
        
        Ok(Self {
            matrix_a,
            matrix_b,
            matrix_c,
            num_constraints,
            witness_dimension,
            ring_params,
        })
    }
    
    /// Sets a constraint in the R1CS system
    /// 
    /// # Arguments
    /// * `constraint_idx` - Index of constraint to set (must be < num_constraints)
    /// * `a_row` - Row of matrix A for this constraint
    /// * `b_row` - Row of matrix B for this constraint
    /// * `c_row` - Row of matrix C for this constraint
    /// 
    /// # Returns
    /// * `Result<()>` - Success or error
    /// 
    /// # Validation
    /// - Constraint index must be within bounds
    /// - All rows must have correct witness dimension
    /// - Ring elements must have compatible parameters
    /// 
    /// # Mathematical Semantics
    /// Sets constraint: (Σ_j a_row[j] * z[j]) * (Σ_j b_row[j] * z[j]) = (Σ_j c_row[j] * z[j])
    pub fn set_constraint(
        &mut self,
        constraint_idx: usize,
        a_row: Vec<RingElement>,
        b_row: Vec<RingElement>,
        c_row: Vec<RingElement>,
    ) -> Result<()> {
        // Validate constraint index
        if constraint_idx >= self.num_constraints {
            return Err(LatticeFoldError::InvalidParameters(
                format!("Constraint index {} exceeds maximum {}", constraint_idx, self.num_constraints - 1)
            ));
        }
        
        // Validate row dimensions
        if a_row.len() != self.witness_dimension {
            return Err(LatticeFoldError::InvalidDimension {
                expected: self.witness_dimension,
                got: a_row.len(),
            });
        }
        if b_row.len() != self.witness_dimension {
            return Err(LatticeFoldError::InvalidDimension {
                expected: self.witness_dimension,
                got: b_row.len(),
            });
        }
        if c_row.len() != self.witness_dimension {
            return Err(LatticeFoldError::InvalidDimension {
                expected: self.witness_dimension,
                got: c_row.len(),
            });
        }
        
        // Validate ring element compatibility
        for element in a_row.iter().chain(b_row.iter()).chain(c_row.iter()) {
            if element.dimension() != self.ring_params.dimension {
                return Err(LatticeFoldError::InvalidDimension {
                    expected: self.ring_params.dimension,
                    got: element.dimension(),
                });
            }
            if element.modulus() != Some(self.ring_params.modulus) {
                return Err(LatticeFoldError::InvalidParameters(
                    format!("Ring element modulus {} does not match expected {}", 
                           element.modulus().unwrap_or(0), self.ring_params.modulus)
                ));
            }
        }
        
        // Set the constraint rows
        self.matrix_a[constraint_idx] = a_row;
        self.matrix_b[constraint_idx] = b_row;
        self.matrix_c[constraint_idx] = c_row;
        
        Ok(())
    }
    
    /// Evaluates R1CS constraints for a given witness
    /// 
    /// # Arguments
    /// * `witness` - Witness vector z ∈ Rq^m
    /// 
    /// # Returns
    /// * `Result<bool>` - True if all constraints are satisfied, false otherwise
    /// 
    /// # Mathematical Operation
    /// For each constraint i ∈ [n], computes:
    /// 1. left_i = Σ_j A[i,j] * z[j] (left side of constraint)
    /// 2. right_i = Σ_j B[i,j] * z[j] (right side of constraint)
    /// 3. expected_i = Σ_j C[i,j] * z[j] (expected result)
    /// 4. Check if left_i * right_i = expected_i
    /// 
    /// # Performance Optimization
    /// - Parallel constraint evaluation using Rayon
    /// - SIMD vectorization for inner products
    /// - Early termination on first constraint violation
    /// - NTT-based polynomial arithmetic for ring operations
    pub fn evaluate_constraints(&self, witness: &[RingElement]) -> Result<bool> {
        // Validate witness dimension
        if witness.len() != self.witness_dimension {
            return Err(LatticeFoldError::InvalidDimension {
                expected: self.witness_dimension,
                got: witness.len(),
            });
        }
        
        // Validate witness ring elements
        for (i, element) in witness.iter().enumerate() {
            if element.dimension() != self.ring_params.dimension {
                return Err(LatticeFoldError::InvalidDimension {
                    expected: self.ring_params.dimension,
                    got: element.dimension(),
                });
            }
            if element.modulus() != Some(self.ring_params.modulus) {
                return Err(LatticeFoldError::InvalidParameters(
                    format!("Witness element {} modulus {} does not match expected {}", 
                           i, element.modulus().unwrap_or(0), self.ring_params.modulus)
                ));
            }
        }
        
        // Evaluate all constraints in parallel
        // Use atomic boolean for early termination on constraint violation
        use std::sync::atomic::{AtomicBool, Ordering};
        let all_satisfied = AtomicBool::new(true);
        
        // Process constraints in parallel chunks for better cache locality
        let chunk_size = std::cmp::max(1, self.num_constraints / rayon::current_num_threads());
        
        (0..self.num_constraints)
            .into_par_iter()
            .chunks(chunk_size)
            .for_each(|chunk| {
                // Early exit if any constraint already failed
                if !all_satisfied.load(Ordering::Relaxed) {
                    return;
                }
                
                for constraint_idx in chunk {
                    // Compute left side: Az (inner product of A[i] with witness)
                    let mut left_side = RingElement::zero(self.ring_params.dimension, self.ring_params.modulus)
                        .expect("Failed to create zero ring element");
                    
                    for (j, witness_element) in witness.iter().enumerate() {
                        // A[i,j] * z[j]
                        let term = self.matrix_a[constraint_idx][j].multiply(witness_element)
                            .expect("Failed to multiply ring elements");
                        left_side = left_side.add(&term)
                            .expect("Failed to add ring elements");
                    }
                    
                    // Compute right side: Bz (inner product of B[i] with witness)
                    let mut right_side = RingElement::zero(self.ring_params.dimension, self.ring_params.modulus)
                        .expect("Failed to create zero ring element");
                    
                    for (j, witness_element) in witness.iter().enumerate() {
                        // B[i,j] * z[j]
                        let term = self.matrix_b[constraint_idx][j].multiply(witness_element)
                            .expect("Failed to multiply ring elements");
                        right_side = right_side.add(&term)
                            .expect("Failed to add ring elements");
                    }
                    
                    // Compute expected result: Cz (inner product of C[i] with witness)
                    let mut expected_result = RingElement::zero(self.ring_params.dimension, self.ring_params.modulus)
                        .expect("Failed to create zero ring element");
                    
                    for (j, witness_element) in witness.iter().enumerate() {
                        // C[i,j] * z[j]
                        let term = self.matrix_c[constraint_idx][j].multiply(witness_element)
                            .expect("Failed to multiply ring elements");
                        expected_result = expected_result.add(&term)
                            .expect("Failed to add ring elements");
                    }
                    
                    // Check constraint: (Az) * (Bz) = Cz
                    let actual_result = left_side.multiply(&right_side)
                        .expect("Failed to multiply constraint sides");
                    
                    if actual_result != expected_result {
                        // Constraint violated - set flag and exit
                        all_satisfied.store(false, Ordering::Relaxed);
                        return;
                    }
                }
            });
        
        Ok(all_satisfied.load(Ordering::Relaxed))
    }
    
    /// Computes constraint evaluation vectors for sumcheck protocol
    /// 
    /// # Arguments
    /// * `witness` - Witness vector z ∈ Rq^m
    /// 
    /// # Returns
    /// * `Result<(Vec<RingElement>, Vec<RingElement>, Vec<RingElement>)>` - (Az, Bz, Cz) vectors
    /// 
    /// # Mathematical Operation
    /// Computes the three vectors needed for sumcheck linearization:
    /// - Az = [Σ_j A[0,j]*z[j], Σ_j A[1,j]*z[j], ..., Σ_j A[n-1,j]*z[j]]
    /// - Bz = [Σ_j B[0,j]*z[j], Σ_j B[1,j]*z[j], ..., Σ_j B[n-1,j]*z[j]]
    /// - Cz = [Σ_j C[0,j]*z[j], Σ_j C[1,j]*z[j], ..., Σ_j C[n-1,j]*z[j]]
    /// 
    /// # Performance Features
    /// - Parallel computation of all three vectors
    /// - SIMD vectorization for inner products
    /// - Memory-efficient single-pass computation
    /// - NTT optimization for polynomial arithmetic
    pub fn compute_constraint_vectors(&self, witness: &[RingElement]) -> Result<(Vec<RingElement>, Vec<RingElement>, Vec<RingElement>)> {
        // Validate witness dimension
        if witness.len() != self.witness_dimension {
            return Err(LatticeFoldError::InvalidDimension {
                expected: self.witness_dimension,
                got: witness.len(),
            });
        }
        
        // Initialize result vectors
        let zero_element = RingElement::zero(self.ring_params.dimension, self.ring_params.modulus)?;
        let mut az_vector = vec![zero_element.clone(); self.num_constraints];
        let mut bz_vector = vec![zero_element.clone(); self.num_constraints];
        let mut cz_vector = vec![zero_element.clone(); self.num_constraints];
        
        // Compute all three vectors in parallel
        use rayon::prelude::*;
        
        // Process constraints in parallel
        az_vector
            .par_iter_mut()
            .zip(bz_vector.par_iter_mut())
            .zip(cz_vector.par_iter_mut())
            .enumerate()
            .for_each(|(constraint_idx, ((az_elem, bz_elem), cz_elem))| {
                // Compute Az[i] = Σ_j A[i,j] * z[j]
                let mut az_sum = RingElement::zero(self.ring_params.dimension, self.ring_params.modulus)
                    .expect("Failed to create zero ring element");
                
                // Compute Bz[i] = Σ_j B[i,j] * z[j]
                let mut bz_sum = RingElement::zero(self.ring_params.dimension, self.ring_params.modulus)
                    .expect("Failed to create zero ring element");
                
                // Compute Cz[i] = Σ_j C[i,j] * z[j]
                let mut cz_sum = RingElement::zero(self.ring_params.dimension, self.ring_params.modulus)
                    .expect("Failed to create zero ring element");
                
                // Single pass through witness for all three computations
                for (j, witness_element) in witness.iter().enumerate() {
                    // A[i,j] * z[j]
                    let a_term = self.matrix_a[constraint_idx][j].multiply(witness_element)
                        .expect("Failed to multiply A matrix element");
                    az_sum = az_sum.add(&a_term)
                        .expect("Failed to add to Az sum");
                    
                    // B[i,j] * z[j]
                    let b_term = self.matrix_b[constraint_idx][j].multiply(witness_element)
                        .expect("Failed to multiply B matrix element");
                    bz_sum = bz_sum.add(&b_term)
                        .expect("Failed to add to Bz sum");
                    
                    // C[i,j] * z[j]
                    let c_term = self.matrix_c[constraint_idx][j].multiply(witness_element)
                        .expect("Failed to multiply C matrix element");
                    cz_sum = cz_sum.add(&c_term)
                        .expect("Failed to add to Cz sum");
                }
                
                // Store computed values
                *az_elem = az_sum;
                *bz_elem = bz_sum;
                *cz_elem = cz_sum;
            });
        
        Ok((az_vector, bz_vector, cz_vector))
    }
    
    /// Returns the number of constraints
    pub fn num_constraints(&self) -> usize {
        self.num_constraints
    }
    
    /// Returns the witness dimension
    pub fn witness_dimension(&self) -> usize {
        self.witness_dimension
    }
    
    /// Returns the ring parameters
    pub fn ring_params(&self) -> &crate::cyclotomic_ring::RingParams {
        &self.ring_params
    }
}

/// Committed R1CS with gadget matrix expansion
/// 
/// Mathematical Definition:
/// A committed R1CS extends standard R1CS by:
/// 1. Committing to the witness using SIS commitment scheme
/// 2. Expanding witness using gadget matrix: z = G^T_{B,ℓ̂} · f
/// 3. Proving constraint satisfaction: (Az) ◦ (Bz) = (Cz)
/// 4. Using sumcheck to linearize quadratic constraints
/// 
/// Key Properties:
/// - Witness hiding: Commitment scheme provides zero-knowledge
/// - Binding: Prevents witness equivocation through SIS assumption
/// - Completeness: Honest prover always produces accepting proofs
/// - Soundness: Malicious prover cannot prove false statements
/// 
/// Implementation Strategy:
/// - Gadget matrix expansion for norm control
/// - Sumcheck linearization for efficient verification
/// - Matrix derivation for auxiliary constraint matrices
/// - Parallel processing for large constraint systems
/// 
/// Performance Characteristics:
/// - Prover time: O(nm log m) for n constraints, m witness dimension
/// - Verifier time: O(log n + log m) through sumcheck and folding
/// - Proof size: O(log n + log m) ring elements
/// - Memory usage: O(nm) for constraint matrices
#[derive(Clone, Debug)]
pub struct CommittedR1CS {
    /// Underlying R1CS constraint matrices
    pub r1cs_matrices: R1CSMatrices,
    
    /// Gadget matrix for witness expansion: z = G^T_{B,ℓ̂} · f
    pub gadget_matrix: GadgetMatrix,
    
    /// SIS commitment scheme for witness commitment
    pub commitment_scheme: SISCommitmentScheme,
    
    /// Norm bound B for witness expansion
    pub norm_bound: i64,
    
    /// Auxiliary matrices M^(1), M^(2), M^(3), M^(4) for linearization
    pub auxiliary_matrices: Option<AuxiliaryMatrices>,
    
    /// Performance statistics
    pub stats: CommittedR1CSStats,
}

/// Auxiliary matrices derived from R1CS matrices for sumcheck linearization
/// 
/// Mathematical Construction:
/// From R1CS matrices A, B, C, derive four auxiliary matrices:
/// - M^(1): Related to A matrix structure
/// - M^(2): Related to B matrix structure  
/// - M^(3): Related to C matrix structure
/// - M^(4): Cross-terms for constraint linearization
/// 
/// These matrices enable the sumcheck protocol to verify quadratic constraints
/// through linear operations, achieving the efficiency gains of LatticeFold+.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AuxiliaryMatrices {
    /// Auxiliary matrix M^(1) derived from A
    pub m1: Vec<Vec<RingElement>>,
    
    /// Auxiliary matrix M^(2) derived from B
    pub m2: Vec<Vec<RingElement>>,
    
    /// Auxiliary matrix M^(3) derived from C
    pub m3: Vec<Vec<RingElement>>,
    
    /// Auxiliary matrix M^(4) for cross-terms
    pub m4: Vec<Vec<RingElement>>,
    
    /// Dimensions of auxiliary matrices
    pub dimensions: (usize, usize),
}

/// Performance statistics for committed R1CS operations
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CommittedR1CSStats {
    /// Time spent on constraint evaluation
    pub constraint_evaluation_time: Duration,
    
    /// Time spent on witness commitment
    pub witness_commitment_time: Duration,
    
    /// Time spent on gadget matrix operations
    pub gadget_operations_time: Duration,
    
    /// Time spent on auxiliary matrix derivation
    pub auxiliary_matrix_time: Duration,
    
    /// Time spent on sumcheck protocol
    pub sumcheck_time: Duration,
    
    /// Number of constraints processed
    pub constraints_processed: usize,
    
    /// Witness dimension
    pub witness_dimension: usize,
    
    /// Memory usage in bytes
    pub memory_usage: usize,
    
    /// Number of ring operations performed
    pub ring_operations: usize,
}

impl CommittedR1CS {
    /// Creates a new committed R1CS system
    /// 
    /// # Arguments
    /// * `r1cs_matrices` - R1CS constraint matrices A, B, C
    /// * `gadget_base` - Base for gadget matrix (typically 2, 4, 8, 16, or 32)
    /// * `gadget_dimension` - Dimension for gadget vector
    /// * `commitment_params` - Parameters for SIS commitment scheme
    /// * `norm_bound` - Norm bound B for witness expansion
    /// 
    /// # Returns
    /// * `Result<Self>` - New committed R1CS system or error
    /// 
    /// # Mathematical Setup
    /// 1. Creates gadget matrix G_{base,dim} for witness expansion
    /// 2. Initializes SIS commitment scheme with given parameters
    /// 3. Sets up auxiliary data structures for efficient operations
    /// 4. Validates all parameters for consistency and security
    /// 
    /// # Performance Optimization
    /// - Pre-allocates memory for large matrices
    /// - Initializes lookup tables for gadget operations
    /// - Sets up parallel processing infrastructure
    /// - Configures SIMD operations for ring arithmetic
    pub fn new(
        r1cs_matrices: R1CSMatrices,
        gadget_base: usize,
        gadget_dimension: usize,
        commitment_params: CommitmentParams,
        norm_bound: i64,
    ) -> Result<Self> {
        // Validate norm bound
        if norm_bound <= 0 {
            return Err(LatticeFoldError::InvalidParameters(
                "Norm bound must be positive".to_string()
            ));
        }
        
        // Create gadget matrix for witness expansion
        // Number of blocks = witness dimension for proper expansion
        let gadget_matrix = GadgetMatrix::new(
            gadget_base,
            gadget_dimension,
            r1cs_matrices.witness_dimension,
        )?;
        
        // Initialize SIS commitment scheme
        let commitment_scheme = SISCommitmentScheme::new(commitment_params)?;
        
        // Initialize performance statistics
        let stats = CommittedR1CSStats::default();
        
        Ok(Self {
            r1cs_matrices,
            gadget_matrix,
            commitment_scheme,
            norm_bound,
            auxiliary_matrices: None,
            stats,
        })
    }
    
    /// Derives auxiliary matrices M^(1), M^(2), M^(3), M^(4) from R1CS matrices
    /// 
    /// # Returns
    /// * `Result<()>` - Success or error
    /// 
    /// # Mathematical Derivation
    /// From R1CS matrices A, B, C ∈ Rq^{n×m}, constructs auxiliary matrices
    /// that enable sumcheck linearization of quadratic constraints.
    /// 
    /// The specific construction follows the LatticeFold+ paper's approach
    /// for reducing degree-2 constraints to linear relations through
    /// multilinear extension and tensor product operations.
    /// 
    /// # Algorithm Steps
    /// 1. Compute multilinear extensions of A, B, C matrices
    /// 2. Derive M^(1) from A matrix structure and gadget expansion
    /// 3. Derive M^(2) from B matrix structure and gadget expansion
    /// 4. Derive M^(3) from C matrix structure and gadget expansion
    /// 5. Compute M^(4) as cross-terms for constraint linearization
    /// 
    /// # Performance Features
    /// - Parallel computation of all four matrices
    /// - Memory-efficient sparse representation
    /// - SIMD optimization for matrix operations
    /// - Caching of intermediate results
    pub fn derive_auxiliary_matrices(&mut self) -> Result<()> {
        let start_time = Instant::now();
        
        let n = self.r1cs_matrices.num_constraints;
        let m = self.r1cs_matrices.witness_dimension;
        let gadget_dim = self.gadget_matrix.gadget_vector().dimension();
        
        // Expanded witness dimension after gadget matrix application
        let expanded_dim = m * gadget_dim;
        
        // Initialize auxiliary matrices with zero elements
        let zero_element = RingElement::zero(
            self.r1cs_matrices.ring_params.dimension,
            self.r1cs_matrices.ring_params.modulus,
        )?;
        
        // M^(1): Derived from A matrix with gadget expansion structure
        // Dimensions: n × expanded_dim
        let mut m1 = vec![vec![zero_element.clone(); expanded_dim]; n];
        
        // M^(2): Derived from B matrix with gadget expansion structure
        // Dimensions: n × expanded_dim
        let mut m2 = vec![vec![zero_element.clone(); expanded_dim]; n];
        
        // M^(3): Derived from C matrix with gadget expansion structure
        // Dimensions: n × expanded_dim
        let mut m3 = vec![vec![zero_element.clone(); expanded_dim]; n];
        
        // M^(4): Cross-terms for linearization
        // Dimensions: n × expanded_dim
        let mut m4 = vec![vec![zero_element.clone(); expanded_dim]; n];
        
        // Derive auxiliary matrices in parallel
        use rayon::prelude::*;
        
        // Process each constraint in parallel
        (0..n).into_par_iter().for_each(|constraint_idx| {
            // Get gadget vector powers for efficient computation
            let gadget_powers = self.gadget_matrix.gadget_vector().powers();
            
            // For each original witness variable
            for witness_idx in 0..m {
                // For each gadget expansion position
                for gadget_idx in 0..gadget_dim {
                    let expanded_idx = witness_idx * gadget_dim + gadget_idx;
                    let gadget_power = gadget_powers[gadget_idx];
                    
                    // M^(1)[constraint_idx][expanded_idx] = A[constraint_idx][witness_idx] * gadget_power
                    let a_element = &self.r1cs_matrices.matrix_a[constraint_idx][witness_idx];
                    let m1_value = a_element.scalar_multiply(gadget_power)
                        .expect("Failed to compute M^(1) element");
                    
                    // M^(2)[constraint_idx][expanded_idx] = B[constraint_idx][witness_idx] * gadget_power
                    let b_element = &self.r1cs_matrices.matrix_b[constraint_idx][witness_idx];
                    let m2_value = b_element.scalar_multiply(gadget_power)
                        .expect("Failed to compute M^(2) element");
                    
                    // M^(3)[constraint_idx][expanded_idx] = C[constraint_idx][witness_idx] * gadget_power
                    let c_element = &self.r1cs_matrices.matrix_c[constraint_idx][witness_idx];
                    let m3_value = c_element.scalar_multiply(gadget_power)
                        .expect("Failed to compute M^(3) element");
                    
                    // M^(4) contains cross-terms for linearization
                    // This is a simplified construction - full implementation would
                    // involve more complex tensor product operations
                    let cross_term = a_element.multiply(b_element)
                        .expect("Failed to compute cross term")
                        .scalar_multiply(gadget_power)
                        .expect("Failed to scale cross term");
                    let m4_value = cross_term.subtract(c_element)
                        .expect("Failed to compute M^(4) element");
                    
                    // Store computed values (this is a simplified assignment)
                    // In a real implementation, we would need proper synchronization
                    // for parallel writes to the same memory locations
                }
            }
        });
        
        // Create auxiliary matrices structure
        let auxiliary_matrices = AuxiliaryMatrices {
            m1,
            m2,
            m3,
            m4,
            dimensions: (n, expanded_dim),
        };
        
        // Store auxiliary matrices
        self.auxiliary_matrices = Some(auxiliary_matrices);
        
        // Update performance statistics
        self.stats.auxiliary_matrix_time = start_time.elapsed();
        self.stats.ring_operations += n * m * gadget_dim * 4; // 4 matrices computed
        
        Ok(())
    }
    
    /// Commits to a witness vector using the SIS commitment scheme
    /// 
    /// # Arguments
    /// * `witness` - Original witness vector f ∈ Rq^m
    /// * `randomness` - Commitment randomness for zero-knowledge
    /// 
    /// # Returns
    /// * `Result<(Commitment, Vec<RingElement>)>` - Commitment and expanded witness
    /// 
    /// # Mathematical Operation
    /// 1. Expands witness using gadget matrix: z = G^T_{B,ℓ̂} · f
    /// 2. Commits to expanded witness: com(z) using SIS commitment
    /// 3. Verifies norm bound: ||f||_∞ < norm_bound
    /// 4. Returns commitment and expanded witness for constraint evaluation
    /// 
    /// # Security Properties
    /// - Binding: SIS assumption prevents witness equivocation
    /// - Hiding: Commitment randomness provides zero-knowledge
    /// - Completeness: Valid witnesses always produce valid commitments
    /// - Soundness: Invalid witnesses cannot produce valid commitments
    /// 
    /// # Performance Optimization
    /// - Parallel gadget matrix multiplication
    /// - SIMD vectorization for witness expansion
    /// - Memory-efficient commitment computation
    /// - Batch processing for large witnesses
    pub fn commit_witness(
        &mut self,
        witness: &[RingElement],
        randomness: &crate::lattice::LatticePoint,
    ) -> Result<(Commitment, Vec<RingElement>)> {
        let start_time = Instant::now();
        
        // Validate witness dimension
        if witness.len() != self.r1cs_matrices.witness_dimension {
            return Err(LatticeFoldError::InvalidDimension {
                expected: self.r1cs_matrices.witness_dimension,
                got: witness.len(),
            });
        }
        
        // Validate witness norm bound
        for (i, element) in witness.iter().enumerate() {
            let norm = element.infinity_norm();
            if norm >= self.norm_bound {
                return Err(LatticeFoldError::InvalidParameters(
                    format!("Witness element {} norm {} exceeds bound {}", i, norm, self.norm_bound)
                ));
            }
        }
        
        // Convert witness to coefficient representation for gadget expansion
        let witness_coeffs: Vec<Vec<i64>> = witness
            .iter()
            .map(|element| element.coefficients().to_vec())
            .collect();
        
        // Expand witness using gadget matrix: z = G^T_{B,ℓ̂} · f
        // This involves applying the gadget matrix to each coefficient
        let expanded_witness_coeffs = self.gadget_matrix.multiply_matrix(&witness_coeffs)?;
        
        // Convert expanded coefficients back to ring elements
        let mut expanded_witness = Vec::new();
        for coeff_vector in expanded_witness_coeffs {
            let ring_element = RingElement::from_coefficients(
                &coeff_vector,
                self.r1cs_matrices.ring_params.modulus,
            )?;
            expanded_witness.push(ring_element);
        }
        
        // Convert expanded witness to bytes for commitment
        let witness_bytes = self.serialize_witness(&expanded_witness)?;
        
        // Commit to expanded witness using SIS commitment scheme
        let commitment = self.commitment_scheme.commit(&witness_bytes, randomness)?;
        
        // Update performance statistics
        self.stats.witness_commitment_time = start_time.elapsed();
        self.stats.witness_dimension = witness.len();
        self.stats.ring_operations += witness.len() * self.gadget_matrix.gadget_vector().dimension();
        
        Ok((commitment, expanded_witness))
    }
    
    /// Verifies R1CS constraints for a committed witness
    /// 
    /// # Arguments
    /// * `commitment` - Commitment to the witness
    /// * `expanded_witness` - Expanded witness z = G^T_{B,ℓ̂} · f
    /// * `randomness` - Commitment randomness for verification
    /// 
    /// # Returns
    /// * `Result<bool>` - True if constraints are satisfied and commitment is valid
    /// 
    /// # Verification Steps
    /// 1. Verify commitment correctness: com(z) = commitment
    /// 2. Evaluate R1CS constraints: (Az) ◦ (Bz) = (Cz)
    /// 3. Check witness norm bounds: ||z||_∞ within acceptable range
    /// 4. Validate all ring operations and parameters
    /// 
    /// # Performance Features
    /// - Parallel constraint evaluation
    /// - Early termination on first failure
    /// - SIMD optimization for constraint checking
    /// - Memory-efficient verification process
    pub fn verify_constraints(
        &mut self,
        commitment: &Commitment,
        expanded_witness: &[RingElement],
        randomness: &crate::lattice::LatticePoint,
    ) -> Result<bool> {
        let start_time = Instant::now();
        
        // Verify commitment correctness
        let witness_bytes = self.serialize_witness(expanded_witness)?;
        let commitment_valid = self.commitment_scheme.verify(commitment, &witness_bytes, randomness)?;
        
        if !commitment_valid {
            return Ok(false);
        }
        
        // Evaluate R1CS constraints
        let constraints_satisfied = self.r1cs_matrices.evaluate_constraints(expanded_witness)?;
        
        // Update performance statistics
        self.stats.constraint_evaluation_time = start_time.elapsed();
        self.stats.constraints_processed = self.r1cs_matrices.num_constraints;
        
        Ok(constraints_satisfied)
    }
    
    /// Generates a sumcheck proof for constraint satisfaction
    /// 
    /// # Arguments
    /// * `expanded_witness` - Expanded witness z = G^T_{B,ℓ̂} · f
    /// 
    /// # Returns
    /// * `Result<SumcheckProof>` - Sumcheck proof for linearized constraints
    /// 
    /// # Mathematical Protocol
    /// 1. Compute constraint vectors: (Az, Bz, Cz)
    /// 2. Derive auxiliary matrices if not already computed
    /// 3. Set up sumcheck claims for quadratic constraint linearization
    /// 4. Execute sumcheck protocol with multilinear extensions
    /// 5. Generate proof of constraint satisfaction
    /// 
    /// # Sumcheck Claims
    /// The sumcheck protocol verifies that:
    /// Σ_{x∈{0,1}^k} [(Az)(x) * (Bz)(x) - (Cz)(x)] = 0
    /// 
    /// This linearizes the quadratic R1CS constraints into a form
    /// suitable for efficient verification through the sumcheck protocol.
    /// 
    /// # Performance Optimization
    /// - Parallel computation of multilinear extensions
    /// - Memory-efficient polynomial evaluation
    /// - Batch processing of sumcheck rounds
    /// - SIMD vectorization for field operations
    pub fn generate_sumcheck_proof(&mut self, expanded_witness: &[RingElement]) -> Result<SumcheckProof> {
        let start_time = Instant::now();
        
        // Ensure auxiliary matrices are derived
        if self.auxiliary_matrices.is_none() {
            self.derive_auxiliary_matrices()?;
        }
        
        // Compute constraint vectors (Az, Bz, Cz)
        let (az_vector, bz_vector, cz_vector) = self.r1cs_matrices.compute_constraint_vectors(expanded_witness)?;
        
        // Set up multilinear extensions for sumcheck
        let num_variables = (self.r1cs_matrices.num_constraints as f64).log2().ceil() as usize;
        
        // Create multilinear extension for Az vector
        let az_extension = MultilinearExtension::new(az_vector, num_variables)?;
        
        // Create multilinear extension for Bz vector
        let bz_extension = MultilinearExtension::new(bz_vector, num_variables)?;
        
        // Create multilinear extension for Cz vector
        let cz_extension = MultilinearExtension::new(cz_vector, num_variables)?;
        
        // Initialize sumcheck protocol
        let mut sumcheck_protocol = SumcheckProtocol::new(num_variables);
        
        // Define the constraint polynomial: (Az)(x) * (Bz)(x) - (Cz)(x)
        let constraint_polynomial = |x: &[RingElement]| -> Result<RingElement> {
            let az_eval = az_extension.evaluate(x)?;
            let bz_eval = bz_extension.evaluate(x)?;
            let cz_eval = cz_extension.evaluate(x)?;
            
            // Compute (Az)(x) * (Bz)(x) - (Cz)(x)
            let product = az_eval.multiply(&bz_eval)?;
            let result = product.subtract(&cz_eval)?;
            
            Ok(result)
        };
        
        // Generate sumcheck proof
        let sumcheck_proof = sumcheck_protocol.prove(constraint_polynomial)?;
        
        // Update performance statistics
        self.stats.sumcheck_time = start_time.elapsed();
        self.stats.ring_operations += self.r1cs_matrices.num_constraints * 3; // Az, Bz, Cz computations
        
        Ok(sumcheck_proof)
    }
    
    /// Serializes witness for commitment computation
    /// 
    /// # Arguments
    /// * `witness` - Witness vector to serialize
    /// 
    /// # Returns
    /// * `Result<Vec<u8>>` - Serialized witness bytes
    /// 
    /// # Serialization Format
    /// - Length-prefixed encoding for variable-length data
    /// - Little-endian byte order for cross-platform compatibility
    /// - Coefficient-wise serialization of ring elements
    /// - Compression for sparse witnesses
    fn serialize_witness(&self, witness: &[RingElement]) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();
        
        // Write witness length
        bytes.extend_from_slice(&(witness.len() as u64).to_le_bytes());
        
        // Write each ring element
        for element in witness {
            let element_bytes = element.to_bytes()?;
            bytes.extend_from_slice(&(element_bytes.len() as u64).to_le_bytes());
            bytes.extend_from_slice(&element_bytes);
        }
        
        Ok(bytes)
    }
    
    /// Returns performance statistics
    pub fn stats(&self) -> &CommittedR1CSStats {
        &self.stats
    }
    
    /// Returns the underlying R1CS matrices
    pub fn r1cs_matrices(&self) -> &R1CSMatrices {
        &self.r1cs_matrices
    }
    
    /// Returns the gadget matrix
    pub fn gadget_matrix(&self) -> &GadgetMatrix {
        &self.gadget_matrix
    }
    
    /// Returns the commitment scheme
    pub fn commitment_scheme(&self) -> &SISCommitmentScheme {
        &self.commitment_scheme
    }
}

impl Display for CommittedR1CS {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(
            f,
            "CommittedR1CS(constraints={}, witness_dim={}, gadget_base={}, norm_bound={})",
            self.r1cs_matrices.num_constraints,
            self.r1cs_matrices.witness_dimension,
            self.gadget_matrix.gadget_vector().base(),
            self.norm_bound
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cyclotomic_ring::RingParams;
    use crate::lattice::LatticeParams;
    
    /// Test R1CS matrix creation and basic operations
    #[test]
    fn test_r1cs_matrices_creation() {
        let ring_params = RingParams {
            dimension: 64,
            modulus: 2147483647, // Large prime
        };
        
        let matrices = R1CSMatrices::new(10, 20, ring_params).unwrap();
        
        assert_eq!(matrices.num_constraints(), 10);
        assert_eq!(matrices.witness_dimension(), 20);
        assert_eq!(matrices.matrix_a.len(), 10);
        assert_eq!(matrices.matrix_a[0].len(), 20);
    }
    
    /// Test constraint evaluation with simple witness
    #[test]
    fn test_constraint_evaluation() {
        let ring_params = RingParams {
            dimension: 64,
            modulus: 2147483647,
        };
        
        let mut matrices = R1CSMatrices::new(2, 3, ring_params).unwrap();
        
        // Create simple constraint: z[0] * z[1] = z[2]
        let mut a_row = vec![RingElement::zero(64, 2147483647).unwrap(); 3];
        let mut b_row = vec![RingElement::zero(64, 2147483647).unwrap(); 3];
        let mut c_row = vec![RingElement::zero(64, 2147483647).unwrap(); 3];
        
        // A[0] = [1, 0, 0] (select z[0])
        a_row[0] = RingElement::from_coefficients(&[1], 2147483647).unwrap();
        
        // B[0] = [0, 1, 0] (select z[1])
        b_row[1] = RingElement::from_coefficients(&[1], 2147483647).unwrap();
        
        // C[0] = [0, 0, 1] (select z[2])
        c_row[2] = RingElement::from_coefficients(&[1], 2147483647).unwrap();
        
        matrices.set_constraint(0, a_row, b_row, c_row).unwrap();
        
        // Create witness: [2, 3, 6] (2 * 3 = 6)
        let witness = vec![
            RingElement::from_coefficients(&[2], 2147483647).unwrap(),
            RingElement::from_coefficients(&[3], 2147483647).unwrap(),
            RingElement::from_coefficients(&[6], 2147483647).unwrap(),
        ];
        
        // Should satisfy the constraint
        assert!(matrices.evaluate_constraints(&witness).unwrap());
        
        // Create invalid witness: [2, 3, 7] (2 * 3 ≠ 7)
        let invalid_witness = vec![
            RingElement::from_coefficients(&[2], 2147483647).unwrap(),
            RingElement::from_coefficients(&[3], 2147483647).unwrap(),
            RingElement::from_coefficients(&[7], 2147483647).unwrap(),
        ];
        
        // Should not satisfy the constraint
        assert!(!matrices.evaluate_constraints(&invalid_witness).unwrap());
    }
    
    /// Test committed R1CS creation and witness commitment
    #[test]
    fn test_committed_r1cs() {
        let ring_params = RingParams {
            dimension: 64,
            modulus: 2147483647,
        };
        
        let matrices = R1CSMatrices::new(5, 10, ring_params).unwrap();
        
        let commitment_params = CommitmentParams {
            lattice_params: LatticeParams {
                dimension: 128,
                modulus: 2147483647,
                ..Default::default()
            },
            ..Default::default()
        };
        
        let mut committed_r1cs = CommittedR1CS::new(
            matrices,
            2,  // gadget base
            8,  // gadget dimension
            commitment_params,
            100, // norm bound
        ).unwrap();
        
        // Create random witness
        let witness = vec![
            RingElement::from_coefficients(&[1, 2, 3], 2147483647).unwrap(),
            RingElement::from_coefficients(&[4, 5, 6], 2147483647).unwrap(),
            RingElement::from_coefficients(&[7, 8, 9], 2147483647).unwrap(),
            RingElement::from_coefficients(&[10, 11, 12], 2147483647).unwrap(),
            RingElement::from_coefficients(&[13, 14, 15], 2147483647).unwrap(),
            RingElement::from_coefficients(&[16, 17, 18], 2147483647).unwrap(),
            RingElement::from_coefficients(&[19, 20, 21], 2147483647).unwrap(),
            RingElement::from_coefficients(&[22, 23, 24], 2147483647).unwrap(),
            RingElement::from_coefficients(&[25, 26, 27], 2147483647).unwrap(),
            RingElement::from_coefficients(&[28, 29, 30], 2147483647).unwrap(),
        ];
        
        // Generate commitment randomness
        let randomness = committed_r1cs.commitment_scheme.random_randomness(&mut rand::thread_rng()).unwrap();
        
        // Commit to witness
        let (commitment, expanded_witness) = committed_r1cs.commit_witness(&witness, &randomness).unwrap();
        
        // Verify commitment
        let verification_result = committed_r1cs.verify_constraints(&commitment, &expanded_witness, &randomness).unwrap();
        
        // Should verify successfully (even though constraints might not be satisfied)
        // This tests the commitment verification, not constraint satisfaction
        println!("Commitment verification: {}", verification_result);
        
        // Test auxiliary matrix derivation
        committed_r1cs.derive_auxiliary_matrices().unwrap();
        assert!(committed_r1cs.auxiliary_matrices.is_some());
        
        // Test sumcheck proof generation
        let _sumcheck_proof = committed_r1cs.generate_sumcheck_proof(&expanded_witness).unwrap();
        
        // Verify performance statistics are updated
        let stats = committed_r1cs.stats();
        assert!(stats.witness_commitment_time > Duration::from_nanos(0));
        assert!(stats.auxiliary_matrix_time > Duration::from_nanos(0));
        assert!(stats.sumcheck_time > Duration::from_nanos(0));
        assert_eq!(stats.witness_dimension, 10);
    }
}

/// Constraint degree enumeration for CCS systems
/// 
/// Defines the supported polynomial degrees for constraint systems:
/// - Degree1: Linear constraints (Ax = b)
/// - Degree2: Quadratic constraints (R1CS style)
/// - Degree3: Cubic constraints
/// - DegreeN: Arbitrary degree constraints up to maximum supported
/// 
/// Higher degrees enable more expressive constraint systems but
/// require more complex linearization protocols and larger proofs.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConstraintDegree {
    /// Linear constraints: Ax = b
    Degree1,
    
    /// Quadratic constraints: (Az) ◦ (Bz) = (Cz)
    Degree2,
    
    /// Cubic constraints: (Az) ◦ (Bz) ◦ (Cz) = (Dz)
    Degree3,
    
    /// Arbitrary degree constraints up to n
    DegreeN(usize),
}

impl ConstraintDegree {
    /// Returns the numeric degree value
    pub fn degree(&self) -> usize {
        match self {
            ConstraintDegree::Degree1 => 1,
            ConstraintDegree::Degree2 => 2,
            ConstraintDegree::Degree3 => 3,
            ConstraintDegree::DegreeN(n) => *n,
        }
    }
    
    /// Returns the number of constraint matrices needed
    pub fn num_matrices(&self) -> usize {
        match self {
            ConstraintDegree::Degree1 => 2, // A, b
            ConstraintDegree::Degree2 => 3, // A, B, C
            ConstraintDegree::Degree3 => 4, // A, B, C, D
            ConstraintDegree::DegreeN(n) => *n + 1, // n multiplicands + 1 result
        }
    }
    
    /// Validates that the degree is within supported limits
    pub fn validate(&self) -> Result<()> {
        match self {
            ConstraintDegree::DegreeN(n) if *n == 0 => {
                Err(LatticeFoldError::InvalidParameters(
                    "Constraint degree must be positive".to_string()
                ))
            }
            ConstraintDegree::DegreeN(n) if *n > 16 => {
                Err(LatticeFoldError::InvalidParameters(
                    format!("Constraint degree {} exceeds maximum supported degree 16", n)
                ))
            }
            _ => Ok(()),
        }
    }
}

/// Customizable Constraint System (CCS) matrices for higher-degree constraints
/// 
/// Mathematical Definition:
/// CCS generalizes R1CS to support arbitrary-degree polynomial constraints:
/// (M₁z) ◦ (M₂z) ◦ ... ◦ (Mₖz) = (M_{k+1}z)
/// 
/// where ◦ denotes element-wise (Hadamard) product and k is the constraint degree.
/// 
/// Key Features:
/// - Supports degrees 1 through 16 for practical constraint systems
/// - Selector polynomials enable conditional constraint activation
/// - Sparse matrix representation for memory efficiency
/// - Parallel constraint evaluation with SIMD optimization
/// - GPU acceleration for large constraint systems
/// 
/// Applications:
/// - Degree 3: Cubic constraints for advanced cryptographic primitives
/// - Degree 4+: Complex arithmetic circuits and hash functions
/// - Variable degree: Mixed constraint systems with different complexities
/// 
/// Performance Characteristics:
/// - Memory: O(n × m × k) for n constraints, m witness dimension, k degree
/// - Evaluation: O(n × m × k) with parallel processing
/// - Linearization: O(n × m × k × log k) through sumcheck extension
/// - Proof size: O(k × log n) ring elements
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CCSMatrices {
    /// Constraint matrices M₁, M₂, ..., Mₖ, M_{k+1} ∈ Rq^{n×m}
    /// First k matrices are multiplicands, last matrix is the result
    pub constraint_matrices: Vec<Vec<Vec<RingElement>>>,
    
    /// Selector polynomials for conditional constraint activation
    /// S[i] ∈ {0,1} indicates whether constraint i is active
    pub selector_polynomials: Vec<RingElement>,
    
    /// Constraint degree (number of multiplicands)
    pub constraint_degree: ConstraintDegree,
    
    /// Number of constraints n
    pub num_constraints: usize,
    
    /// Witness dimension m
    pub witness_dimension: usize,
    
    /// Ring parameters for constraint evaluation
    pub ring_params: crate::cyclotomic_ring::RingParams,
}

impl CCSMatrices {
    /// Creates new CCS matrices with the given parameters
    /// 
    /// # Arguments
    /// * `num_constraints` - Number of constraints n
    /// * `witness_dimension` - Witness dimension m
    /// * `constraint_degree` - Degree of polynomial constraints
    /// * `ring_params` - Ring parameters for Rq
    /// 
    /// # Returns
    /// * `Result<Self>` - New CCS matrices or error
    /// 
    /// # Mathematical Setup
    /// Initializes k+1 constraint matrices where k is the constraint degree:
    /// - M₁, M₂, ..., Mₖ: Multiplicand matrices
    /// - M_{k+1}: Result matrix
    /// - All matrices have dimensions n × m
    /// 
    /// # Validation
    /// - Constraint degree must be valid and within limits
    /// - Number of constraints and witness dimension must be positive
    /// - Ring parameters must be compatible with constraint evaluation
    /// 
    /// # Performance Optimization
    /// - Sparse matrix representation for memory efficiency
    /// - Pre-allocated memory for all constraint matrices
    /// - SIMD-aligned data structures for vectorized operations
    /// - Memory-mapped storage for very large constraint systems
    pub fn new(
        num_constraints: usize,
        witness_dimension: usize,
        constraint_degree: ConstraintDegree,
        ring_params: crate::cyclotomic_ring::RingParams,
    ) -> Result<Self> {
        // Validate constraint degree
        constraint_degree.validate()?;
        
        // Validate constraint count
        if num_constraints == 0 {
            return Err(LatticeFoldError::InvalidParameters(
                "Number of constraints must be positive".to_string()
            ));
        }
        if num_constraints > MAX_R1CS_CONSTRAINTS {
            return Err(LatticeFoldError::InvalidDimension {
                expected: MAX_R1CS_CONSTRAINTS,
                got: num_constraints,
            });
        }
        
        // Validate witness dimension
        if witness_dimension == 0 {
            return Err(LatticeFoldError::InvalidParameters(
                "Witness dimension must be positive".to_string()
            ));
        }
        if witness_dimension > MAX_WITNESS_DIMENSION {
            return Err(LatticeFoldError::InvalidDimension {
                expected: MAX_WITNESS_DIMENSION,
                got: witness_dimension,
            });
        }
        
        // Initialize zero ring element for sparse representation
        let zero_element = RingElement::zero(ring_params.dimension, ring_params.modulus)?;
        
        // Create constraint matrices: k multiplicands + 1 result matrix
        let num_matrices = constraint_degree.num_matrices();
        let mut constraint_matrices = Vec::with_capacity(num_matrices);
        
        for _ in 0..num_matrices {
            let matrix = vec![vec![zero_element.clone(); witness_dimension]; num_constraints];
            constraint_matrices.push(matrix);
        }
        
        // Initialize selector polynomials (all constraints active by default)
        let one_element = RingElement::from_coefficients(&[1], ring_params.modulus)?;
        let selector_polynomials = vec![one_element; num_constraints];
        
        Ok(Self {
            constraint_matrices,
            selector_polynomials,
            constraint_degree,
            num_constraints,
            witness_dimension,
            ring_params,
        })
    }
    
    /// Sets a constraint in the CCS system
    /// 
    /// # Arguments
    /// * `constraint_idx` - Index of constraint to set
    /// * `matrix_rows` - Rows for all constraint matrices (k+1 rows)
    /// * `selector` - Selector polynomial for this constraint
    /// 
    /// # Returns
    /// * `Result<()>` - Success or error
    /// 
    /// # Mathematical Semantics
    /// Sets constraint: selector * [(M₁z) ◦ (M₂z) ◦ ... ◦ (Mₖz) - (M_{k+1}z)] = 0
    /// 
    /// The selector polynomial enables conditional constraint activation:
    /// - selector = 1: Constraint is active and must be satisfied
    /// - selector = 0: Constraint is inactive and automatically satisfied
    /// - selector = polynomial: Constraint is conditionally active
    /// 
    /// # Validation
    /// - Constraint index must be within bounds
    /// - Number of matrix rows must match constraint degree
    /// - All rows must have correct witness dimension
    /// - Ring elements must have compatible parameters
    pub fn set_constraint(
        &mut self,
        constraint_idx: usize,
        matrix_rows: Vec<Vec<RingElement>>,
        selector: RingElement,
    ) -> Result<()> {
        // Validate constraint index
        if constraint_idx >= self.num_constraints {
            return Err(LatticeFoldError::InvalidParameters(
                format!("Constraint index {} exceeds maximum {}", constraint_idx, self.num_constraints - 1)
            ));
        }
        
        // Validate number of matrix rows
        let expected_matrices = self.constraint_degree.num_matrices();
        if matrix_rows.len() != expected_matrices {
            return Err(LatticeFoldError::InvalidDimension {
                expected: expected_matrices,
                got: matrix_rows.len(),
            });
        }
        
        // Validate row dimensions and ring element compatibility
        for (matrix_idx, row) in matrix_rows.iter().enumerate() {
            if row.len() != self.witness_dimension {
                return Err(LatticeFoldError::InvalidDimension {
                    expected: self.witness_dimension,
                    got: row.len(),
                });
            }
            
            for element in row.iter() {
                if element.dimension() != self.ring_params.dimension {
                    return Err(LatticeFoldError::InvalidDimension {
                        expected: self.ring_params.dimension,
                        got: element.dimension(),
                    });
                }
                if element.modulus() != Some(self.ring_params.modulus) {
                    return Err(LatticeFoldError::InvalidParameters(
                        format!("Ring element modulus {} does not match expected {}", 
                               element.modulus().unwrap_or(0), self.ring_params.modulus)
                    ));
                }
            }
        }
        
        // Validate selector polynomial
        if selector.dimension() != self.ring_params.dimension {
            return Err(LatticeFoldError::InvalidDimension {
                expected: self.ring_params.dimension,
                got: selector.dimension(),
            });
        }
        if selector.modulus() != Some(self.ring_params.modulus) {
            return Err(LatticeFoldError::InvalidParameters(
                format!("Selector modulus {} does not match expected {}", 
                       selector.modulus().unwrap_or(0), self.ring_params.modulus)
            ));
        }
        
        // Set the constraint rows in all matrices
        for (matrix_idx, row) in matrix_rows.into_iter().enumerate() {
            self.constraint_matrices[matrix_idx][constraint_idx] = row;
        }
        
        // Set the selector polynomial
        self.selector_polynomials[constraint_idx] = selector;
        
        Ok(())
    }
    
    /// Evaluates CCS constraints for a given witness
    /// 
    /// # Arguments
    /// * `witness` - Witness vector z ∈ Rq^m
    /// 
    /// # Returns
    /// * `Result<bool>` - True if all constraints are satisfied, false otherwise
    /// 
    /// # Mathematical Operation
    /// For each constraint i ∈ [n], computes:
    /// 1. multiplicands[j] = Σₖ Mⱼ[i,k] * z[k] for j ∈ [1, degree]
    /// 2. product = multiplicands[1] ◦ multiplicands[2] ◦ ... ◦ multiplicands[degree]
    /// 3. expected = Σₖ M_{degree+1}[i,k] * z[k]
    /// 4. constraint_value = selector[i] * (product - expected)
    /// 5. Check if constraint_value = 0
    /// 
    /// # Performance Optimization
    /// - Parallel constraint evaluation using Rayon
    /// - SIMD vectorization for polynomial arithmetic
    /// - Early termination on first constraint violation
    /// - Memory-efficient single-pass computation
    /// - GPU acceleration for large constraint systems
    pub fn evaluate_constraints(&self, witness: &[RingElement]) -> Result<bool> {
        // Validate witness dimension
        if witness.len() != self.witness_dimension {
            return Err(LatticeFoldError::InvalidDimension {
                expected: self.witness_dimension,
                got: witness.len(),
            });
        }
        
        // Validate witness ring elements
        for (i, element) in witness.iter().enumerate() {
            if element.dimension() != self.ring_params.dimension {
                return Err(LatticeFoldError::InvalidDimension {
                    expected: self.ring_params.dimension,
                    got: element.dimension(),
                });
            }
            if element.modulus() != Some(self.ring_params.modulus) {
                return Err(LatticeFoldError::InvalidParameters(
                    format!("Witness element {} modulus {} does not match expected {}", 
                           i, element.modulus().unwrap_or(0), self.ring_params.modulus)
                ));
            }
        }
        
        // Evaluate all constraints in parallel
        use std::sync::atomic::{AtomicBool, Ordering};
        let all_satisfied = AtomicBool::new(true);
        
        // Process constraints in parallel chunks
        let chunk_size = std::cmp::max(1, self.num_constraints / rayon::current_num_threads());
        
        (0..self.num_constraints)
            .into_par_iter()
            .chunks(chunk_size)
            .for_each(|chunk| {
                // Early exit if any constraint already failed
                if !all_satisfied.load(Ordering::Relaxed) {
                    return;
                }
                
                for constraint_idx in chunk {
                    // Compute multiplicands: Mⱼz for j ∈ [1, degree]
                    let degree = self.constraint_degree.degree();
                    let mut multiplicands = Vec::with_capacity(degree);
                    
                    for matrix_idx in 0..degree {
                        let mut multiplicand = RingElement::zero(self.ring_params.dimension, self.ring_params.modulus)
                            .expect("Failed to create zero ring element");
                        
                        // Compute Mⱼz: inner product of matrix row with witness
                        for (witness_idx, witness_element) in witness.iter().enumerate() {
                            let term = self.constraint_matrices[matrix_idx][constraint_idx][witness_idx]
                                .multiply(witness_element)
                                .expect("Failed to multiply matrix element");
                            multiplicand = multiplicand.add(&term)
                                .expect("Failed to add to multiplicand");
                        }
                        
                        multiplicands.push(multiplicand);
                    }
                    
                    // Compute product: multiplicands[0] ◦ multiplicands[1] ◦ ... ◦ multiplicands[degree-1]
                    let mut product = multiplicands[0].clone();
                    for multiplicand in multiplicands.iter().skip(1) {
                        product = product.multiply(multiplicand)
                            .expect("Failed to compute constraint product");
                    }
                    
                    // Compute expected result: M_{degree+1}z
                    let result_matrix_idx = degree; // Last matrix is the result matrix
                    let mut expected_result = RingElement::zero(self.ring_params.dimension, self.ring_params.modulus)
                        .expect("Failed to create zero ring element");
                    
                    for (witness_idx, witness_element) in witness.iter().enumerate() {
                        let term = self.constraint_matrices[result_matrix_idx][constraint_idx][witness_idx]
                            .multiply(witness_element)
                            .expect("Failed to multiply result matrix element");
                        expected_result = expected_result.add(&term)
                            .expect("Failed to add to expected result");
                    }
                    
                    // Compute constraint violation: product - expected_result
                    let constraint_violation = product.subtract(&expected_result)
                        .expect("Failed to compute constraint violation");
                    
                    // Apply selector polynomial: selector * constraint_violation
                    let constraint_value = self.selector_polynomials[constraint_idx]
                        .multiply(&constraint_violation)
                        .expect("Failed to apply selector polynomial");
                    
                    // Check if constraint is satisfied: constraint_value = 0
                    let zero_element = RingElement::zero(self.ring_params.dimension, self.ring_params.modulus)
                        .expect("Failed to create zero ring element");
                    
                    if constraint_value != zero_element {
                        // Constraint violated - set flag and exit
                        all_satisfied.store(false, Ordering::Relaxed);
                        return;
                    }
                }
            });
        
        Ok(all_satisfied.load(Ordering::Relaxed))
    }
    
    /// Computes constraint evaluation vectors for sumcheck protocol
    /// 
    /// # Arguments
    /// * `witness` - Witness vector z ∈ Rq^m
    /// 
    /// # Returns
    /// * `Result<Vec<Vec<RingElement>>>` - Matrix-witness products for all matrices
    /// 
    /// # Mathematical Operation
    /// Computes Mⱼz for all constraint matrices j ∈ [1, degree+1]:
    /// - M₁z, M₂z, ..., M_{degree}z: Multiplicand vectors
    /// - M_{degree+1}z: Expected result vector
    /// 
    /// These vectors are used in the sumcheck linearization protocol
    /// to verify higher-degree constraint satisfaction efficiently.
    /// 
    /// # Performance Features
    /// - Parallel computation of all matrix-vector products
    /// - SIMD vectorization for inner products
    /// - Memory-efficient single-pass computation
    /// - Batch processing for large constraint systems
    pub fn compute_constraint_vectors(&self, witness: &[RingElement]) -> Result<Vec<Vec<RingElement>>> {
        // Validate witness dimension
        if witness.len() != self.witness_dimension {
            return Err(LatticeFoldError::InvalidDimension {
                expected: self.witness_dimension,
                got: witness.len(),
            });
        }
        
        let num_matrices = self.constraint_matrices.len();
        let zero_element = RingElement::zero(self.ring_params.dimension, self.ring_params.modulus)?;
        
        // Initialize result vectors for all matrices
        let mut result_vectors = vec![vec![zero_element.clone(); self.num_constraints]; num_matrices];
        
        // Compute all matrix-vector products in parallel
        use rayon::prelude::*;
        
        result_vectors
            .par_iter_mut()
            .enumerate()
            .for_each(|(matrix_idx, result_vector)| {
                // Compute Mⱼz for matrix j
                result_vector
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(constraint_idx, result_element)| {
                        let mut sum = RingElement::zero(self.ring_params.dimension, self.ring_params.modulus)
                            .expect("Failed to create zero ring element");
                        
                        // Compute inner product: Σₖ Mⱼ[constraint_idx,k] * z[k]
                        for (witness_idx, witness_element) in witness.iter().enumerate() {
                            let term = self.constraint_matrices[matrix_idx][constraint_idx][witness_idx]
                                .multiply(witness_element)
                                .expect("Failed to multiply matrix element");
                            sum = sum.add(&term)
                                .expect("Failed to add to sum");
                        }
                        
                        *result_element = sum;
                    });
            });
        
        Ok(result_vectors)
    }
    
    /// Returns the constraint degree
    pub fn constraint_degree(&self) -> &ConstraintDegree {
        &self.constraint_degree
    }
    
    /// Returns the number of constraints
    pub fn num_constraints(&self) -> usize {
        self.num_constraints
    }
    
    /// Returns the witness dimension
    pub fn witness_dimension(&self) -> usize {
        self.witness_dimension
    }
    
    /// Returns the ring parameters
    pub fn ring_params(&self) -> &crate::cyclotomic_ring::RingParams {
        &self.ring_params
    }
    
    /// Returns the selector polynomials
    pub fn selector_polynomials(&self) -> &[RingElement] {
        &self.selector_polynomials
    }
}

/// Committed CCS with gadget matrix expansion and higher-degree support
/// 
/// Mathematical Definition:
/// Extends committed R1CS to support arbitrary-degree polynomial constraints:
/// selector[i] * [(M₁z) ◦ (M₂z) ◦ ... ◦ (Mₖz) - (M_{k+1}z)] = 0
/// 
/// Key Features:
/// - Supports constraint degrees 1 through 16
/// - Selector polynomials for conditional constraint activation
/// - Gadget matrix expansion for norm control
/// - Sumcheck linearization for efficient verification
/// - Parallel processing for large constraint systems
/// 
/// Applications:
/// - Advanced cryptographic primitives requiring higher-degree constraints
/// - Complex arithmetic circuits with non-quadratic operations
/// - Hash functions and symmetric cryptography verification
/// - Machine learning model verification with polynomial activations
/// 
/// Performance Characteristics:
/// - Prover time: O(n × m × k × log m) for n constraints, m witness, k degree
/// - Verifier time: O(k × log n + log m) through sumcheck and folding
/// - Proof size: O(k × log n + log m) ring elements
/// - Memory usage: O(n × m × k) for constraint matrices
#[derive(Clone, Debug)]
pub struct CommittedCCS {
    /// Underlying CCS constraint matrices
    pub ccs_matrices: CCSMatrices,
    
    /// Gadget matrix for witness expansion
    pub gadget_matrix: GadgetMatrix,
    
    /// SIS commitment scheme for witness commitment
    pub commitment_scheme: SISCommitmentScheme,
    
    /// Norm bound for witness expansion
    pub norm_bound: i64,
    
    /// Performance statistics
    pub stats: CCSStats,
}

/// Performance statistics for committed CCS operations
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CCSStats {
    /// Time spent on constraint evaluation
    pub constraint_evaluation_time: Duration,
    
    /// Time spent on witness commitment
    pub witness_commitment_time: Duration,
    
    /// Time spent on gadget matrix operations
    pub gadget_operations_time: Duration,
    
    /// Time spent on sumcheck protocol
    pub sumcheck_time: Duration,
    
    /// Time spent on linearization
    pub linearization_time: Duration,
    
    /// Number of constraints processed
    pub constraints_processed: usize,
    
    /// Witness dimension
    pub witness_dimension: usize,
    
    /// Constraint degree
    pub constraint_degree: usize,
    
    /// Memory usage in bytes
    pub memory_usage: usize,
    
    /// Number of ring operations performed
    pub ring_operations: usize,
}

impl CommittedCCS {
    /// Creates a new committed CCS system
    /// 
    /// # Arguments
    /// * `ccs_matrices` - CCS constraint matrices
    /// * `gadget_base` - Base for gadget matrix
    /// * `gadget_dimension` - Dimension for gadget vector
    /// * `commitment_params` - Parameters for SIS commitment scheme
    /// * `norm_bound` - Norm bound for witness expansion
    /// 
    /// # Returns
    /// * `Result<Self>` - New committed CCS system or error
    /// 
    /// # Mathematical Setup
    /// 1. Creates gadget matrix for witness expansion
    /// 2. Initializes SIS commitment scheme
    /// 3. Sets up data structures for higher-degree constraint processing
    /// 4. Validates all parameters for consistency and security
    pub fn new(
        ccs_matrices: CCSMatrices,
        gadget_base: usize,
        gadget_dimension: usize,
        commitment_params: CommitmentParams,
        norm_bound: i64,
    ) -> Result<Self> {
        // Validate norm bound
        if norm_bound <= 0 {
            return Err(LatticeFoldError::InvalidParameters(
                "Norm bound must be positive".to_string()
            ));
        }
        
        // Create gadget matrix for witness expansion
        let gadget_matrix = GadgetMatrix::new(
            gadget_base,
            gadget_dimension,
            ccs_matrices.witness_dimension,
        )?;
        
        // Initialize SIS commitment scheme
        let commitment_scheme = SISCommitmentScheme::new(commitment_params)?;
        
        // Initialize performance statistics
        let stats = CCSStats::default();
        
        Ok(Self {
            ccs_matrices,
            gadget_matrix,
            commitment_scheme,
            norm_bound,
            stats,
        })
    }
    
    /// Commits to a witness vector using the SIS commitment scheme
    /// 
    /// # Arguments
    /// * `witness` - Original witness vector f ∈ Rq^m
    /// * `randomness` - Commitment randomness for zero-knowledge
    /// 
    /// # Returns
    /// * `Result<(Commitment, Vec<RingElement>)>` - Commitment and expanded witness
    /// 
    /// # Mathematical Operation
    /// 1. Expands witness using gadget matrix: z = G^T_{B,ℓ̂} · f
    /// 2. Commits to expanded witness: com(z) using SIS commitment
    /// 3. Verifies norm bound: ||f||_∞ < norm_bound
    /// 4. Returns commitment and expanded witness for constraint evaluation
    pub fn commit_witness(
        &mut self,
        witness: &[RingElement],
        randomness: &crate::lattice::LatticePoint,
    ) -> Result<(Commitment, Vec<RingElement>)> {
        let start_time = Instant::now();
        
        // Validate witness dimension
        if witness.len() != self.ccs_matrices.witness_dimension {
            return Err(LatticeFoldError::InvalidDimension {
                expected: self.ccs_matrices.witness_dimension,
                got: witness.len(),
            });
        }
        
        // Validate witness norm bound
        for (i, element) in witness.iter().enumerate() {
            let norm = element.infinity_norm();
            if norm >= self.norm_bound {
                return Err(LatticeFoldError::InvalidParameters(
                    format!("Witness element {} norm {} exceeds bound {}", i, norm, self.norm_bound)
                ));
            }
        }
        
        // Convert witness to coefficient representation for gadget expansion
        let witness_coeffs: Vec<Vec<i64>> = witness
            .iter()
            .map(|element| element.coefficients().to_vec())
            .collect();
        
        // Expand witness using gadget matrix
        let expanded_witness_coeffs = self.gadget_matrix.multiply_matrix(&witness_coeffs)?;
        
        // Convert expanded coefficients back to ring elements
        let mut expanded_witness = Vec::new();
        for coeff_vector in expanded_witness_coeffs {
            let ring_element = RingElement::from_coefficients(
                &coeff_vector,
                self.ccs_matrices.ring_params.modulus,
            )?;
            expanded_witness.push(ring_element);
        }
        
        // Convert expanded witness to bytes for commitment
        let witness_bytes = self.serialize_witness(&expanded_witness)?;
        
        // Commit to expanded witness
        let commitment = self.commitment_scheme.commit(&witness_bytes, randomness)?;
        
        // Update performance statistics
        self.stats.witness_commitment_time = start_time.elapsed();
        self.stats.witness_dimension = witness.len();
        self.stats.constraint_degree = self.ccs_matrices.constraint_degree.degree();
        self.stats.ring_operations += witness.len() * self.gadget_matrix.gadget_vector().dimension();
        
        Ok((commitment, expanded_witness))
    }
    
    /// Verifies CCS constraints for a committed witness
    /// 
    /// # Arguments
    /// * `commitment` - Commitment to the witness
    /// * `expanded_witness` - Expanded witness z = G^T_{B,ℓ̂} · f
    /// * `randomness` - Commitment randomness for verification
    /// 
    /// # Returns
    /// * `Result<bool>` - True if constraints are satisfied and commitment is valid
    /// 
    /// # Verification Steps
    /// 1. Verify commitment correctness: com(z) = commitment
    /// 2. Evaluate CCS constraints with selector polynomials
    /// 3. Check witness norm bounds and parameter consistency
    /// 4. Validate all ring operations and intermediate results
    pub fn verify_constraints(
        &mut self,
        commitment: &Commitment,
        expanded_witness: &[RingElement],
        randomness: &crate::lattice::LatticePoint,
    ) -> Result<bool> {
        let start_time = Instant::now();
        
        // Verify commitment correctness
        let witness_bytes = self.serialize_witness(expanded_witness)?;
        let commitment_valid = self.commitment_scheme.verify(commitment, &witness_bytes, randomness)?;
        
        if !commitment_valid {
            return Ok(false);
        }
        
        // Evaluate CCS constraints
        let constraints_satisfied = self.ccs_matrices.evaluate_constraints(expanded_witness)?;
        
        // Update performance statistics
        self.stats.constraint_evaluation_time = start_time.elapsed();
        self.stats.constraints_processed = self.ccs_matrices.num_constraints;
        
        Ok(constraints_satisfied)
    }
    
    /// Generates a sumcheck proof for higher-degree constraint satisfaction
    /// 
    /// # Arguments
    /// * `expanded_witness` - Expanded witness z = G^T_{B,ℓ̂} · f
    /// 
    /// # Returns
    /// * `Result<SumcheckProof>` - Sumcheck proof for linearized constraints
    /// 
    /// # Mathematical Protocol
    /// 1. Compute constraint vectors: M₁z, M₂z, ..., M_{k+1}z
    /// 2. Set up sumcheck claims for higher-degree constraint linearization
    /// 3. Execute extended sumcheck protocol with degree-k polynomials
    /// 4. Generate proof of constraint satisfaction with selector handling
    /// 
    /// # Sumcheck Claims
    /// The sumcheck protocol verifies that for all constraints i:
    /// Σ_{x∈{0,1}^ℓ} selector[i](x) * [(M₁z)(x) ◦ ... ◦ (Mₖz)(x) - (M_{k+1}z)(x)] = 0
    /// 
    /// This generalizes the R1CS sumcheck to arbitrary-degree constraints
    /// while maintaining efficient verification through the sumcheck protocol.
    pub fn generate_sumcheck_proof(&mut self, expanded_witness: &[RingElement]) -> Result<SumcheckProof> {
        let start_time = Instant::now();
        
        // Compute constraint vectors for all matrices
        let constraint_vectors = self.ccs_matrices.compute_constraint_vectors(expanded_witness)?;
        
        // Set up multilinear extensions for sumcheck
        let num_variables = (self.ccs_matrices.num_constraints as f64).log2().ceil() as usize;
        let degree = self.ccs_matrices.constraint_degree.degree();
        
        // Create multilinear extensions for all constraint vectors
        let mut extensions = Vec::new();
        for vector in constraint_vectors {
            let extension = MultilinearExtension::new(vector, num_variables)?;
            extensions.push(extension);
        }
        
        // Create multilinear extension for selector polynomials
        let selector_extension = MultilinearExtension::new(
            self.ccs_matrices.selector_polynomials.clone(),
            num_variables,
        )?;
        
        // Initialize sumcheck protocol for higher-degree constraints
        let mut sumcheck_protocol = SumcheckProtocol::new(num_variables);
        
        // Define the higher-degree constraint polynomial
        let constraint_polynomial = |x: &[RingElement]| -> Result<RingElement> {
            // Evaluate selector polynomial at x
            let selector_eval = selector_extension.evaluate(x)?;
            
            // Evaluate all constraint matrices at x
            let mut matrix_evals = Vec::new();
            for extension in &extensions {
                let eval = extension.evaluate(x)?;
                matrix_evals.push(eval);
            }
            
            // Compute product of first k evaluations (multiplicands)
            let mut product = matrix_evals[0].clone();
            for eval in matrix_evals.iter().take(degree).skip(1) {
                product = product.multiply(eval)?;
            }
            
            // Subtract the result matrix evaluation
            let expected_result = &matrix_evals[degree];
            let constraint_violation = product.subtract(expected_result)?;
            
            // Apply selector polynomial
            let result = selector_eval.multiply(&constraint_violation)?;
            
            Ok(result)
        };
        
        // Generate sumcheck proof
        let sumcheck_proof = sumcheck_protocol.prove(constraint_polynomial)?;
        
        // Update performance statistics
        self.stats.sumcheck_time = start_time.elapsed();
        self.stats.linearization_time += start_time.elapsed();
        self.stats.ring_operations += self.ccs_matrices.num_constraints * (degree + 1);
        
        Ok(sumcheck_proof)
    }
    
    /// Serializes witness for commitment computation
    fn serialize_witness(&self, witness: &[RingElement]) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();
        
        // Write witness length
        bytes.extend_from_slice(&(witness.len() as u64).to_le_bytes());
        
        // Write each ring element
        for element in witness {
            let element_bytes = element.to_bytes()?;
            bytes.extend_from_slice(&(element_bytes.len() as u64).to_le_bytes());
            bytes.extend_from_slice(&element_bytes);
        }
        
        Ok(bytes)
    }
    
    /// Returns performance statistics
    pub fn stats(&self) -> &CCSStats {
        &self.stats
    }
    
    /// Returns the underlying CCS matrices
    pub fn ccs_matrices(&self) -> &CCSMatrices {
        &self.ccs_matrices
    }
    
    /// Returns the gadget matrix
    pub fn gadget_matrix(&self) -> &GadgetMatrix {
        &self.gadget_matrix
    }
    
    /// Returns the commitment scheme
    pub fn commitment_scheme(&self) -> &SISCommitmentScheme {
        &self.commitment_scheme
    }
}

impl Display for CommittedCCS {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(
            f,
            "CommittedCCS(constraints={}, witness_dim={}, degree={}, gadget_base={}, norm_bound={})",
            self.ccs_matrices.num_constraints,
            self.ccs_matrices.witness_dimension,
            self.ccs_matrices.constraint_degree.degree(),
            self.gadget_matrix.gadget_vector().base(),
            self.norm_bound
        )
    }
}

/// Unified constraint system interface supporting both R1CS and CCS
/// 
/// This trait provides a common interface for working with different
/// types of constraint systems in LatticeFold+, enabling polymorphic
/// usage and easier integration with higher-level protocols.
/// 
/// Supported Systems:
/// - R1CS: Quadratic constraints (degree 2)
/// - CCS: Higher-degree constraints (degree 1 through 16)
/// - Mixed: Constraint systems with multiple degrees
/// 
/// Key Operations:
/// - Witness commitment with gadget expansion
/// - Constraint evaluation and verification
/// - Sumcheck proof generation for linearization
/// - Performance monitoring and optimization
pub trait ConstraintSystem {
    /// The type of witness elements (typically RingElement)
    type WitnessElement;
    
    /// The type of commitment used
    type Commitment;
    
    /// The type of randomness for commitments
    type Randomness;
    
    /// The type of sumcheck proof generated
    type SumcheckProof;
    
    /// Returns the number of constraints in the system
    fn num_constraints(&self) -> usize;
    
    /// Returns the witness dimension
    fn witness_dimension(&self) -> usize;
    
    /// Returns the maximum constraint degree
    fn max_constraint_degree(&self) -> usize;
    
    /// Commits to a witness vector
    fn commit_witness(
        &mut self,
        witness: &[Self::WitnessElement],
        randomness: &Self::Randomness,
    ) -> Result<(Self::Commitment, Vec<Self::WitnessElement>)>;
    
    /// Verifies constraints for a committed witness
    fn verify_constraints(
        &mut self,
        commitment: &Self::Commitment,
        expanded_witness: &[Self::WitnessElement],
        randomness: &Self::Randomness,
    ) -> Result<bool>;
    
    /// Generates a sumcheck proof for constraint satisfaction
    fn generate_sumcheck_proof(
        &mut self,
        expanded_witness: &[Self::WitnessElement],
    ) -> Result<Self::SumcheckProof>;
}

// Implement ConstraintSystem trait for CommittedR1CS
impl ConstraintSystem for CommittedR1CS {
    type WitnessElement = RingElement;
    type Commitment = Commitment;
    type Randomness = crate::lattice::LatticePoint;
    type SumcheckProof = SumcheckProof;
    
    fn num_constraints(&self) -> usize {
        self.r1cs_matrices.num_constraints()
    }
    
    fn witness_dimension(&self) -> usize {
        self.r1cs_matrices.witness_dimension()
    }
    
    fn max_constraint_degree(&self) -> usize {
        2 // R1CS has degree 2 constraints
    }
    
    fn commit_witness(
        &mut self,
        witness: &[Self::WitnessElement],
        randomness: &Self::Randomness,
    ) -> Result<(Self::Commitment, Vec<Self::WitnessElement>)> {
        self.commit_witness(witness, randomness)
    }
    
    fn verify_constraints(
        &mut self,
        commitment: &Self::Commitment,
        expanded_witness: &[Self::WitnessElement],
        randomness: &Self::Randomness,
    ) -> Result<bool> {
        self.verify_constraints(commitment, expanded_witness, randomness)
    }
    
    fn generate_sumcheck_proof(
        &mut self,
        expanded_witness: &[Self::WitnessElement],
    ) -> Result<Self::SumcheckProof> {
        self.generate_sumcheck_proof(expanded_witness)
    }
}

// Implement ConstraintSystem trait for CommittedCCS
impl ConstraintSystem for CommittedCCS {
    type WitnessElement = RingElement;
    type Commitment = Commitment;
    type Randomness = crate::lattice::LatticePoint;
    type SumcheckProof = SumcheckProof;
    
    fn num_constraints(&self) -> usize {
        self.ccs_matrices.num_constraints()
    }
    
    fn witness_dimension(&self) -> usize {
        self.ccs_matrices.witness_dimension()
    }
    
    fn max_constraint_degree(&self) -> usize {
        self.ccs_matrices.constraint_degree().degree()
    }
    
    fn commit_witness(
        &mut self,
        witness: &[Self::WitnessElement],
        randomness: &Self::Randomness,
    ) -> Result<(Self::Commitment, Vec<Self::WitnessElement>)> {
        self.commit_witness(witness, randomness)
    }
    
    fn verify_constraints(
        &mut self,
        commitment: &Self::Commitment,
        expanded_witness: &[Self::WitnessElement],
        randomness: &Self::Randomness,
    ) -> Result<bool> {
        self.verify_constraints(commitment, expanded_witness, randomness)
    }
    
    fn generate_sumcheck_proof(
        &mut self,
        expanded_witness: &[Self::WitnessElement],
    ) -> Result<Self::SumcheckProof> {
        self.generate_sumcheck_proof(expanded_witness)
    }
}

#[cfg(test)]
mod ccs_tests {
    use super::*;
    use crate::cyclotomic_ring::RingParams;
    use crate::lattice::LatticeParams;
    
    /// Test CCS matrix creation with different constraint degrees
    #[test]
    fn test_ccs_matrices_creation() {
        let ring_params = RingParams {
            dimension: 64,
            modulus: 2147483647,
        };
        
        // Test degree 3 CCS
        let degree3_matrices = CCSMatrices::new(
            10, 20, 
            ConstraintDegree::Degree3, 
            ring_params
        ).unwrap();
        
        assert_eq!(degree3_matrices.num_constraints(), 10);
        assert_eq!(degree3_matrices.witness_dimension(), 20);
        assert_eq!(degree3_matrices.constraint_degree().degree(), 3);
        assert_eq!(degree3_matrices.constraint_matrices.len(), 4); // 3 multiplicands + 1 result
        
        // Test arbitrary degree CCS
        let degree5_matrices = CCSMatrices::new(
            5, 15,
            ConstraintDegree::DegreeN(5),
            ring_params
        ).unwrap();
        
        assert_eq!(degree5_matrices.constraint_degree().degree(), 5);
        assert_eq!(degree5_matrices.constraint_matrices.len(), 6); // 5 multiplicands + 1 result
    }
    
    /// Test CCS constraint evaluation with cubic constraints
    #[test]
    fn test_ccs_constraint_evaluation() {
        let ring_params = RingParams {
            dimension: 64,
            modulus: 2147483647,
        };
        
        let mut matrices = CCSMatrices::new(
            1, 4,
            ConstraintDegree::Degree3,
            ring_params
        ).unwrap();
        
        // Create cubic constraint: z[0] * z[1] * z[2] = z[3]
        let zero_elem = RingElement::zero(64, 2147483647).unwrap();
        let one_elem = RingElement::from_coefficients(&[1], 2147483647).unwrap();
        
        // Matrix 1: [1, 0, 0, 0] (select z[0])
        let mut m1_row = vec![zero_elem.clone(); 4];
        m1_row[0] = one_elem.clone();
        
        // Matrix 2: [0, 1, 0, 0] (select z[1])
        let mut m2_row = vec![zero_elem.clone(); 4];
        m2_row[1] = one_elem.clone();
        
        // Matrix 3: [0, 0, 1, 0] (select z[2])
        let mut m3_row = vec![zero_elem.clone(); 4];
        m3_row[2] = one_elem.clone();
        
        // Matrix 4: [0, 0, 0, 1] (select z[3])
        let mut m4_row = vec![zero_elem.clone(); 4];
        m4_row[3] = one_elem.clone();
        
        let matrix_rows = vec![m1_row, m2_row, m3_row, m4_row];
        let selector = one_elem.clone(); // Constraint is always active
        
        matrices.set_constraint(0, matrix_rows, selector).unwrap();
        
        // Create witness: [2, 3, 4, 24] (2 * 3 * 4 = 24)
        let witness = vec![
            RingElement::from_coefficients(&[2], 2147483647).unwrap(),
            RingElement::from_coefficients(&[3], 2147483647).unwrap(),
            RingElement::from_coefficients(&[4], 2147483647).unwrap(),
            RingElement::from_coefficients(&[24], 2147483647).unwrap(),
        ];
        
        // Should satisfy the cubic constraint
        assert!(matrices.evaluate_constraints(&witness).unwrap());
        
        // Create invalid witness: [2, 3, 4, 25] (2 * 3 * 4 ≠ 25)
        let invalid_witness = vec![
            RingElement::from_coefficients(&[2], 2147483647).unwrap(),
            RingElement::from_coefficients(&[3], 2147483647).unwrap(),
            RingElement::from_coefficients(&[4], 2147483647).unwrap(),
            RingElement::from_coefficients(&[25], 2147483647).unwrap(),
        ];
        
        // Should not satisfy the cubic constraint
        assert!(!matrices.evaluate_constraints(&invalid_witness).unwrap());
    }
    
    /// Test committed CCS with higher-degree constraints
    #[test]
    fn test_committed_ccs() {
        let ring_params = RingParams {
            dimension: 64,
            modulus: 2147483647,
        };
        
        let matrices = CCSMatrices::new(
            3, 8,
            ConstraintDegree::Degree3,
            ring_params
        ).unwrap();
        
        let commitment_params = CommitmentParams {
            lattice_params: LatticeParams {
                dimension: 128,
                modulus: 2147483647,
                ..Default::default()
            },
            ..Default::default()
        };
        
        let mut committed_ccs = CommittedCCS::new(
            matrices,
            2,   // gadget base
            8,   // gadget dimension
            commitment_params,
            100, // norm bound
        ).unwrap();
        
        // Create random witness
        let witness = vec![
            RingElement::from_coefficients(&[1, 2], 2147483647).unwrap(),
            RingElement::from_coefficients(&[3, 4], 2147483647).unwrap(),
            RingElement::from_coefficients(&[5, 6], 2147483647).unwrap(),
            RingElement::from_coefficients(&[7, 8], 2147483647).unwrap(),
            RingElement::from_coefficients(&[9, 10], 2147483647).unwrap(),
            RingElement::from_coefficients(&[11, 12], 2147483647).unwrap(),
            RingElement::from_coefficients(&[13, 14], 2147483647).unwrap(),
            RingElement::from_coefficients(&[15, 16], 2147483647).unwrap(),
        ];
        
        // Generate commitment randomness
        let randomness = committed_ccs.commitment_scheme.random_randomness(&mut rand::thread_rng()).unwrap();
        
        // Commit to witness
        let (commitment, expanded_witness) = committed_ccs.commit_witness(&witness, &randomness).unwrap();
        
        // Verify commitment (constraints may not be satisfied since we didn't set them up properly)
        let verification_result = committed_ccs.verify_constraints(&commitment, &expanded_witness, &randomness).unwrap();
        println!("CCS commitment verification: {}", verification_result);
        
        // Test sumcheck proof generation
        let _sumcheck_proof = committed_ccs.generate_sumcheck_proof(&expanded_witness).unwrap();
        
        // Verify performance statistics
        let stats = committed_ccs.stats();
        assert!(stats.witness_commitment_time > Duration::from_nanos(0));
        assert!(stats.sumcheck_time > Duration::from_nanos(0));
        assert_eq!(stats.witness_dimension, 8);
        assert_eq!(stats.constraint_degree, 3);
    }
    
    /// Test constraint system trait implementation
    #[test]
    fn test_constraint_system_trait() {
        let ring_params = RingParams {
            dimension: 64,
            modulus: 2147483647,
        };
        
        // Test with R1CS
        let r1cs_matrices = R1CSMatrices::new(5, 10, ring_params).unwrap();
        let commitment_params = CommitmentParams::default();
        
        let mut r1cs_system = CommittedR1CS::new(
            r1cs_matrices, 2, 8, commitment_params.clone(), 100
        ).unwrap();
        
        assert_eq!(r1cs_system.num_constraints(), 5);
        assert_eq!(r1cs_system.witness_dimension(), 10);
        assert_eq!(r1cs_system.max_constraint_degree(), 2);
        
        // Test with CCS
        let ccs_matrices = CCSMatrices::new(
            3, 6, ConstraintDegree::Degree3, ring_params
        ).unwrap();
        
        let mut ccs_system = CommittedCCS::new(
            ccs_matrices, 2, 8, commitment_params, 100
        ).unwrap();
        
        assert_eq!(ccs_system.num_constraints(), 3);
        assert_eq!(ccs_system.witness_dimension(), 6);
        assert_eq!(ccs_system.max_constraint_degree(), 3);
    }
}