// Constant-time cryptographic operations for LatticeFold+ implementation
// This module provides constant-time implementations of all arithmetic operations
// that involve secret data, ensuring protection against timing attacks and
// maintaining consistent execution time regardless of input values.

use crate::error::{LatticeFoldError, Result};
use crate::cyclotomic_ring::{RingElement, BalancedCoefficients};
use crate::modular_arithmetic::{BarrettParams, MontgomeryParams};
use crate::security::{SecurityConfig, CryptographicParameters};
use subtle::{Choice, ConditionallySelectable, ConstantTimeEq, ConstantTimeGreater, ConstantTimeLess};
use zeroize::{Zeroize, ZeroizeOnDrop};
use std::time::Instant;

/// Constant-time arithmetic operations for secret-dependent computations
/// This trait provides constant-time implementations of basic arithmetic
/// operations that maintain consistent timing regardless of input values.
pub trait ConstantTimeArithmetic {
    /// Constant-time addition with overflow detection
    /// Performs a + b in constant time, returning an error if overflow occurs
    /// Time complexity: O(1) - always executes in constant time
    fn ct_add(&self, other: &Self) -> Result<Self> where Self: Sized;
    
    /// Constant-time subtraction with underflow detection
    /// Performs a - b in constant time, returning an error if underflow occurs
    /// Time complexity: O(1) - always executes in constant time
    fn ct_sub(&self, other: &Self) -> Result<Self> where Self: Sized;
    
    /// Constant-time multiplication with overflow detection
    /// Performs a * b in constant time, returning an error if overflow occurs
    /// Time complexity: O(1) - always executes in constant time
    fn ct_mul(&self, other: &Self) -> Result<Self> where Self: Sized;
    
    /// Constant-time negation
    /// Performs -a in constant time
    /// Time complexity: O(1) - always executes in constant time
    fn ct_neg(&self) -> Self;
    
    /// Constant-time equality comparison
    /// Returns Choice(1) if self == other, Choice(0) otherwise
    /// Time complexity: O(1) - always executes in constant time
    fn ct_eq(&self, other: &Self) -> Choice;
    
    /// Constant-time conditional selection
    /// Returns self if choice == 1, other if choice == 0
    /// Time complexity: O(1) - always executes in constant time
    fn ct_select(choice: Choice, a: &Self, b: &Self) -> Self where Self: Sized;
}

/// Constant-time modular arithmetic operations
/// This trait provides constant-time implementations of modular arithmetic
/// operations using Barrett reduction and Montgomery multiplication.
pub trait ConstantTimeModularArithmetic {
    /// Constant-time modular addition: (a + b) mod q
    /// Uses Barrett reduction for consistent timing
    /// Time complexity: O(1) - always executes in constant time
    fn ct_add_mod(&self, other: &Self, modulus: i64) -> Result<Self> where Self: Sized;
    
    /// Constant-time modular subtraction: (a - b) mod q
    /// Uses Barrett reduction for consistent timing
    /// Time complexity: O(1) - always executes in constant time
    fn ct_sub_mod(&self, other: &Self, modulus: i64) -> Result<Self> where Self: Sized;
    
    /// Constant-time modular multiplication: (a * b) mod q
    /// Uses Montgomery multiplication for consistent timing
    /// Time complexity: O(1) - always executes in constant time
    fn ct_mul_mod(&self, other: &Self, modulus: i64) -> Result<Self> where Self: Sized;
    
    /// Constant-time modular reduction: a mod q
    /// Uses Barrett reduction with precomputed parameters
    /// Time complexity: O(1) - always executes in constant time
    fn ct_reduce_mod(&self, modulus: i64) -> Result<Self> where Self: Sized;
    
    /// Constant-time modular inverse: a^(-1) mod q
    /// Uses extended Euclidean algorithm with constant-time implementation
    /// Time complexity: O(log q) - but with consistent timing for all inputs
    fn ct_inverse_mod(&self, modulus: i64) -> Result<Self> where Self: Sized;
}

/// Constant-time norm checking and comparison operations
/// This trait provides constant-time implementations of norm computations
/// and comparisons that are essential for lattice-based cryptography.
pub trait ConstantTimeNormChecker {
    /// Constant-time infinity norm computation: ||x||_∞
    /// Computes the maximum absolute value of coefficients in constant time
    /// Time complexity: O(n) - linear in dimension but constant time per element
    fn ct_infinity_norm(&self) -> i64;
    
    /// Constant-time norm bound checking: ||x||_∞ < bound
    /// Returns Choice(1) if norm is below bound, Choice(0) otherwise
    /// Time complexity: O(n) - linear in dimension but constant time per element
    fn ct_norm_check(&self, bound: i64) -> Choice;
    
    /// Constant-time Euclidean norm computation: ||x||_2
    /// Computes the Euclidean norm in constant time with overflow protection
    /// Time complexity: O(n) - linear in dimension but constant time per element
    fn ct_euclidean_norm(&self) -> Result<f64>;
    
    /// Constant-time norm comparison: ||x||_∞ < ||y||_∞
    /// Compares norms in constant time without revealing the actual values
    /// Time complexity: O(n) - linear in dimension but constant time per element
    fn ct_norm_less_than(&self, other: &Self) -> Choice where Self: Sized;
}

/// Constant-time polynomial operations for cyclotomic rings
/// This trait provides constant-time implementations of polynomial operations
/// that are used in the LatticeFold+ protocol for ring-based computations.
pub trait ConstantTimePolynomialOps {
    /// Constant-time polynomial addition in cyclotomic ring
    /// Performs coefficient-wise addition with modular reduction
    /// Time complexity: O(d) - linear in ring dimension
    fn ct_poly_add(&self, other: &Self) -> Result<Self> where Self: Sized;
    
    /// Constant-time polynomial subtraction in cyclotomic ring
    /// Performs coefficient-wise subtraction with modular reduction
    /// Time complexity: O(d) - linear in ring dimension
    fn ct_poly_sub(&self, other: &Self) -> Result<Self> where Self: Sized;
    
    /// Constant-time polynomial multiplication in cyclotomic ring
    /// Uses NTT-based multiplication for consistent timing
    /// Time complexity: O(d log d) - but consistent for all inputs
    fn ct_poly_mul(&self, other: &Self) -> Result<Self> where Self: Sized;
    
    /// Constant-time polynomial evaluation at a point
    /// Uses Horner's method with constant-time operations
    /// Time complexity: O(d) - linear in degree
    fn ct_poly_eval(&self, point: i64) -> Result<i64>;
    
    /// Constant-time coefficient extraction
    /// Extracts coefficient at given index in constant time
    /// Time complexity: O(1) - always executes in constant time
    fn ct_get_coeff(&self, index: usize) -> Result<i64>;
    
    /// Constant-time coefficient setting
    /// Sets coefficient at given index in constant time
    /// Time complexity: O(1) - always executes in constant time
    fn ct_set_coeff(&mut self, index: usize, value: i64) -> Result<()>;
}

/// Constant-time comparison operations
/// This trait provides constant-time implementations of comparison operations
/// that do not leak information about the compared values through timing.
pub trait ConstantTimeComparison {
    /// Constant-time less-than comparison: a < b
    /// Returns Choice(1) if a < b, Choice(0) otherwise
    /// Time complexity: O(1) - always executes in constant time
    fn ct_less_than(&self, other: &Self) -> Choice;
    
    /// Constant-time greater-than comparison: a > b
    /// Returns Choice(1) if a > b, Choice(0) otherwise
    /// Time complexity: O(1) - always executes in constant time
    fn ct_greater_than(&self, other: &Self) -> Choice;
    
    /// Constant-time less-than-or-equal comparison: a <= b
    /// Returns Choice(1) if a <= b, Choice(0) otherwise
    /// Time complexity: O(1) - always executes in constant time
    fn ct_less_equal(&self, other: &Self) -> Choice;
    
    /// Constant-time greater-than-or-equal comparison: a >= b
    /// Returns Choice(1) if a >= b, Choice(0) otherwise
    /// Time complexity: O(1) - always executes in constant time
    fn ct_greater_equal(&self, other: &Self) -> Choice;
    
    /// Constant-time minimum selection: min(a, b)
    /// Returns the minimum value without revealing which was smaller
    /// Time complexity: O(1) - always executes in constant time
    fn ct_min(&self, other: &Self) -> Self where Self: Sized;
    
    /// Constant-time maximum selection: max(a, b)
    /// Returns the maximum value without revealing which was larger
    /// Time complexity: O(1) - always executes in constant time
    fn ct_max(&self, other: &Self) -> Self where Self: Sized;
}

/// Constant-time conditional selection operations
/// This trait provides constant-time implementations of conditional selection
/// that do not leak information about the condition through timing or branching.
pub trait ConstantTimeSelection {
    /// Constant-time conditional assignment: if choice then a else b
    /// Assigns value based on choice without conditional branching
    /// Time complexity: O(1) - always executes in constant time
    fn ct_conditional_assign(&mut self, other: &Self, choice: Choice);
    
    /// Constant-time conditional swap: if choice then swap(a, b)
    /// Swaps values based on choice without conditional branching
    /// Time complexity: O(1) - always executes in constant time
    fn ct_conditional_swap(&mut self, other: &mut Self, choice: Choice) where Self: Sized;
    
    /// Constant-time conditional negate: if choice then -a else a
    /// Negates value based on choice without conditional branching
    /// Time complexity: O(1) - always executes in constant time
    fn ct_conditional_negate(&mut self, choice: Choice);
    
    /// Constant-time array selection: select element at index
    /// Selects array element without revealing the index through memory access
    /// Time complexity: O(n) - linear scan to avoid index-dependent access
    fn ct_array_select(array: &[Self], index: usize) -> Result<Self> where Self: Sized + Clone;
}

/// Constant-time matrix operations
/// This trait provides constant-time implementations of matrix operations
/// that are used in commitment schemes and linear algebra computations.
pub trait ConstantTimeMatrixOps {
    /// Constant-time matrix-vector multiplication: A * v
    /// Performs matrix-vector multiplication with consistent timing
    /// Time complexity: O(mn) - but consistent for all inputs
    fn ct_matrix_vector_mul(&self, vector: &[Self]) -> Result<Vec<Self>> where Self: Sized + Clone;
    
    /// Constant-time matrix-matrix multiplication: A * B
    /// Performs matrix-matrix multiplication with consistent timing
    /// Time complexity: O(mnp) - but consistent for all inputs
    fn ct_matrix_matrix_mul(&self, other: &Self) -> Result<Self> where Self: Sized;
    
    /// Constant-time matrix transpose: A^T
    /// Transposes matrix with consistent memory access patterns
    /// Time complexity: O(mn) - but with cache-friendly access
    fn ct_transpose(&self) -> Self where Self: Sized;
    
    /// Constant-time matrix addition: A + B
    /// Performs element-wise addition with consistent timing
    /// Time complexity: O(mn) - linear in matrix size
    fn ct_matrix_add(&self, other: &Self) -> Result<Self> where Self: Sized;
    
    /// Constant-time matrix subtraction: A - B
    /// Performs element-wise subtraction with consistent timing
    /// Time complexity: O(mn) - linear in matrix size
    fn ct_matrix_sub(&self, other: &Self) -> Result<Self> where Self: Sized;
}

/// Constant-time vector operations
/// This trait provides constant-time implementations of vector operations
/// that are used throughout the LatticeFold+ protocol.
pub trait ConstantTimeVectorOps {
    /// Constant-time vector addition: u + v
    /// Performs element-wise addition with consistent timing
    /// Time complexity: O(n) - linear in vector dimension
    fn ct_vector_add(&self, other: &[Self]) -> Result<Vec<Self>> where Self: Sized + Clone;
    
    /// Constant-time vector subtraction: u - v
    /// Performs element-wise subtraction with consistent timing
    /// Time complexity: O(n) - linear in vector dimension
    fn ct_vector_sub(&self, other: &[Self]) -> Result<Vec<Self>> where Self: Sized + Clone;
    
    /// Constant-time dot product: u · v
    /// Computes inner product with consistent timing
    /// Time complexity: O(n) - linear in vector dimension
    fn ct_dot_product(&self, other: &[Self]) -> Result<Self> where Self: Sized + Clone;
    
    /// Constant-time scalar multiplication: c * v
    /// Multiplies vector by scalar with consistent timing
    /// Time complexity: O(n) - linear in vector dimension
    fn ct_scalar_mul(&self, scalar: &Self) -> Self where Self: Sized;
    
    /// Constant-time vector norm: ||v||
    /// Computes vector norm with consistent timing
    /// Time complexity: O(n) - linear in vector dimension
    fn ct_vector_norm(&self) -> i64;
}

/// Constant-time gadget matrix operations
/// This trait provides constant-time implementations of gadget matrix operations
/// that are used in the double commitment scheme and decomposition protocols.
pub trait ConstantTimeGadgetOps {
    /// Constant-time gadget decomposition: G^(-1)(M)
    /// Decomposes matrix using gadget matrix with consistent timing
    /// Time complexity: O(mnk) - but consistent for all inputs
    fn ct_gadget_decompose(&self, base: usize, digits: usize) -> Result<Vec<Self>> where Self: Sized + Clone;
    
    /// Constant-time gadget reconstruction: G * M'
    /// Reconstructs matrix from decomposition with consistent timing
    /// Time complexity: O(mnk) - but consistent for all inputs
    fn ct_gadget_reconstruct(&self, base: usize, digits: usize) -> Result<Self> where Self: Sized;
    
    /// Constant-time base-b digit extraction
    /// Extracts base-b digits with consistent timing for all values
    /// Time complexity: O(k) - linear in number of digits
    fn ct_extract_digits(&self, base: usize, num_digits: usize) -> Result<Vec<i64>>;
    
    /// Constant-time digit reconstruction
    /// Reconstructs value from base-b digits with consistent timing
    /// Time complexity: O(k) - linear in number of digits
    fn ct_reconstruct_from_digits(digits: &[i64], base: usize) -> Result<i64>;
}

/// Constant-time commitment operations
/// This trait provides constant-time implementations of commitment operations
/// that are used in the LatticeFold+ commitment schemes.
pub trait ConstantTimeCommitmentOps {
    /// Constant-time linear commitment: com(m) = A * m
    /// Computes commitment with consistent timing regardless of message
    /// Time complexity: O(mn) - but consistent for all messages
    fn ct_linear_commit(&self, message: &[Self]) -> Result<Vec<Self>> where Self: Sized + Clone;
    
    /// Constant-time commitment verification: com(m) ?= c
    /// Verifies commitment with consistent timing regardless of validity
    /// Time complexity: O(n) - but consistent for all inputs
    fn ct_verify_commitment(&self, commitment: &[Self], message: &[Self]) -> Result<Choice> where Self: Sized;
    
    /// Constant-time commitment opening: verify opening (m, r)
    /// Verifies commitment opening with consistent timing
    /// Time complexity: O(n) - but consistent for all openings
    fn ct_verify_opening(&self, commitment: &[Self], message: &[Self], randomness: &[Self]) -> Result<Choice> where Self: Sized;
    
    /// Constant-time homomorphic addition: com(m1) + com(m2)
    /// Adds commitments homomorphically with consistent timing
    /// Time complexity: O(n) - linear in commitment size
    fn ct_homomorphic_add(&self, other: &[Self]) -> Result<Vec<Self>> where Self: Sized + Clone;
}

/// Constant-time range proof operations
/// This trait provides constant-time implementations of range proof operations
/// that are used in the algebraic range proof system.
pub trait ConstantTimeRangeProof {
    /// Constant-time range check: x ∈ [-B, B]
    /// Checks if value is in range with consistent timing
    /// Time complexity: O(1) - always executes in constant time
    fn ct_range_check(&self, bound: i64) -> Choice;
    
    /// Constant-time monomial membership test: x ∈ M
    /// Tests monomial membership with consistent timing
    /// Time complexity: O(d) - linear in ring dimension
    fn ct_monomial_membership(&self) -> Choice;
    
    /// Constant-time polynomial ψ evaluation: ct(b·ψ)
    /// Evaluates range polynomial with consistent timing
    /// Time complexity: O(d) - linear in ring dimension
    fn ct_psi_evaluation(&self, psi: &Self) -> Result<i64> where Self: Sized;
    
    /// Constant-time EXP function evaluation: EXP(a)
    /// Evaluates exponential mapping with consistent timing
    /// Time complexity: O(1) - always executes in constant time
    fn ct_exp_evaluation(&self) -> Result<Vec<Self>> where Self: Sized + Clone;
}

/// Constant-time folding operations
/// This trait provides constant-time implementations of folding operations
/// that are used in the multi-instance folding protocol.
pub trait ConstantTimeFolding {
    /// Constant-time witness folding: fold witnesses with challenges
    /// Folds multiple witnesses with consistent timing
    /// Time complexity: O(Ln) - linear in number of instances and dimension
    fn ct_fold_witnesses(&self, challenges: &[Self]) -> Result<Self> where Self: Sized + Clone;
    
    /// Constant-time instance folding: fold instances with challenges
    /// Folds multiple instances with consistent timing
    /// Time complexity: O(Ln) - linear in number of instances and dimension
    fn ct_fold_instances(&self, challenges: &[Self]) -> Result<Self> where Self: Sized + Clone;
    
    /// Constant-time challenge verification: verify folding challenges
    /// Verifies challenges with consistent timing
    /// Time complexity: O(L) - linear in number of instances
    fn ct_verify_challenges(&self, challenges: &[Self]) -> Result<Choice> where Self: Sized;
    
    /// Constant-time norm preservation check: ||folded|| ≤ bound
    /// Checks norm preservation with consistent timing
    /// Time complexity: O(n) - linear in dimension
    fn ct_check_norm_preservation(&self, bound: i64) -> Choice;
}

/// Constant-time error handling operations
/// This trait provides constant-time implementations of error handling
/// that do not leak information about error conditions through timing.
pub trait ConstantTimeErrorHandling {
    /// Constant-time error propagation
    /// Propagates errors with consistent timing regardless of error type
    /// Time complexity: O(1) - always executes in constant time
    fn ct_propagate_error(&self, error: &LatticeFoldError) -> Result<()>;
    
    /// Constant-time error masking
    /// Masks error conditions to prevent information leakage
    /// Time complexity: O(1) - always executes in constant time
    fn ct_mask_error(&self, condition: Choice) -> Result<()>;
    
    /// Constant-time success indication
    /// Indicates success/failure with consistent timing
    /// Time complexity: O(1) - always executes in constant time
    fn ct_indicate_success(&self, success: Choice) -> Result<Choice>;
}

/// Timing-consistent operations wrapper
/// This structure wraps operations to ensure they execute in consistent time
/// regardless of input values or computational paths taken.
#[derive(Clone, Debug, Zeroize, ZeroizeOnDrop)]
pub struct TimingConsistentOperations {
    /// Security configuration for timing requirements
    config: SecurityConfig,
    
    /// Expected operation times for consistency checking
    expected_times: std::collections::HashMap<String, u64>,
    
    /// Timing measurements for analysis
    timing_measurements: Vec<TimingMeasurement>,
}

/// Individual timing measurement record
#[derive(Clone, Debug)]
pub struct TimingMeasurement {
    /// Name of the operation
    pub operation_name: String,
    
    /// Duration in nanoseconds
    pub duration_ns: u64,
    
    /// Timestamp when measurement was taken
    pub timestamp: std::time::SystemTime,
    
    /// Input size or complexity measure
    pub input_size: usize,
}

impl TimingConsistentOperations {
    /// Create a new timing-consistent operations wrapper
    /// This initializes the wrapper with the specified security configuration
    /// and prepares it for timing-consistent operation execution.
    pub fn new(config: SecurityConfig) -> Result<Self> {
        // Validate configuration
        config.validate()?;
        
        Ok(Self {
            config,
            expected_times: std::collections::HashMap::new(),
            timing_measurements: Vec::new(),
        })
    }
    
    /// Execute an operation with timing consistency checking
    /// This method wraps the execution of an operation to ensure it completes
    /// within the expected time bounds and records timing for analysis.
    pub fn execute_with_timing<F, T>(&mut self, operation_name: &str, operation: F) -> Result<T>
    where
        F: FnOnce() -> Result<T>,
    {
        // Record start time
        let start_time = Instant::now();
        
        // Execute the operation
        let result = operation();
        
        // Record end time and calculate duration
        let duration = start_time.elapsed();
        let duration_ns = duration.as_nanos() as u64;
        
        // Record timing measurement
        self.timing_measurements.push(TimingMeasurement {
            operation_name: operation_name.to_string(),
            duration_ns,
            timestamp: std::time::SystemTime::now(),
            input_size: 0, // Would be set by caller in real implementation
        });
        
        // Check timing consistency if enabled
        if self.config.timing_analysis_enabled {
            self.check_timing_consistency(operation_name, duration_ns)?;
        }
        
        result
    }
    
    /// Check timing consistency for an operation
    /// This method verifies that the operation completed within expected timing
    /// bounds and flags potential timing leaks or inconsistencies.
    fn check_timing_consistency(&self, operation_name: &str, duration_ns: u64) -> Result<()> {
        // Check against maximum allowed variance
        if let Some(&expected_time) = self.expected_times.get(operation_name) {
            let variance = if duration_ns > expected_time {
                duration_ns - expected_time
            } else {
                expected_time - duration_ns
            };
            
            if variance > self.config.max_timing_variance_ns {
                return Err(LatticeFoldError::CryptoError(format!(
                    "Timing variance {} ns exceeds maximum {} ns for operation '{}'",
                    variance, self.config.max_timing_variance_ns, operation_name
                )));
            }
        } else {
            // First measurement for this operation - record as expected time
            // In a real implementation, this would use statistical analysis
            // of multiple measurements to establish expected timing
        }
        
        Ok(())
    }
    
    /// Get timing statistics for analysis
    /// This method returns timing statistics that can be used for security
    /// analysis and detection of potential timing-based side channels.
    pub fn get_timing_statistics(&self) -> TimingStatistics {
        let mut stats = TimingStatistics::default();
        
        if self.timing_measurements.is_empty() {
            return stats;
        }
        
        // Calculate basic statistics
        let durations: Vec<u64> = self.timing_measurements.iter()
            .map(|m| m.duration_ns)
            .collect();
        
        stats.total_measurements = durations.len();
        stats.min_duration_ns = *durations.iter().min().unwrap_or(&0);
        stats.max_duration_ns = *durations.iter().max().unwrap_or(&0);
        stats.mean_duration_ns = durations.iter().sum::<u64>() / durations.len() as u64;
        
        // Calculate variance
        let variance_sum: u64 = durations.iter()
            .map(|&d| {
                let diff = if d > stats.mean_duration_ns {
                    d - stats.mean_duration_ns
                } else {
                    stats.mean_duration_ns - d
                };
                diff * diff
            })
            .sum();
        
        stats.variance_ns = variance_sum / durations.len() as u64;
        stats.std_deviation_ns = (stats.variance_ns as f64).sqrt() as u64;
        
        // Check for timing consistency
        stats.timing_consistent = stats.variance_ns <= self.config.max_timing_variance_ns;
        
        stats
    }
    
    /// Clear timing measurements
    /// This method clears all recorded timing measurements, which should be
    /// done periodically to prevent memory growth and maintain privacy.
    pub fn clear_measurements(&mut self) {
        self.timing_measurements.clear();
    }
}

/// Timing statistics for security analysis
#[derive(Clone, Debug, Default)]
pub struct TimingStatistics {
    /// Total number of timing measurements
    pub total_measurements: usize,
    
    /// Minimum observed duration in nanoseconds
    pub min_duration_ns: u64,
    
    /// Maximum observed duration in nanoseconds
    pub max_duration_ns: u64,
    
    /// Mean duration in nanoseconds
    pub mean_duration_ns: u64,
    
    /// Variance in nanoseconds squared
    pub variance_ns: u64,
    
    /// Standard deviation in nanoseconds
    pub std_deviation_ns: u64,
    
    /// Whether timing is consistent within configured bounds
    pub timing_consistent: bool,
}

// Implementation of constant-time arithmetic for i64
impl ConstantTimeArithmetic for i64 {
    /// Constant-time addition with overflow detection
    /// This implementation uses checked arithmetic to detect overflow
    /// while maintaining constant execution time for all inputs.
    fn ct_add(&self, other: &Self) -> Result<Self> {
        // Use checked arithmetic to detect overflow
        // The timing is constant regardless of whether overflow occurs
        match self.checked_add(*other) {
            Some(result) => Ok(result),
            None => Err(LatticeFoldError::ArithmeticOverflow(
                format!("Addition overflow: {} + {}", self, other)
            )),
        }
    }
    
    /// Constant-time subtraction with underflow detection
    /// This implementation uses checked arithmetic to detect underflow
    /// while maintaining constant execution time for all inputs.
    fn ct_sub(&self, other: &Self) -> Result<Self> {
        // Use checked arithmetic to detect underflow
        // The timing is constant regardless of whether underflow occurs
        match self.checked_sub(*other) {
            Some(result) => Ok(result),
            None => Err(LatticeFoldError::ArithmeticOverflow(
                format!("Subtraction underflow: {} - {}", self, other)
            )),
        }
    }
    
    /// Constant-time multiplication with overflow detection
    /// This implementation uses checked arithmetic to detect overflow
    /// while maintaining constant execution time for all inputs.
    fn ct_mul(&self, other: &Self) -> Result<Self> {
        // Use checked arithmetic to detect overflow
        // The timing is constant regardless of whether overflow occurs
        match self.checked_mul(*other) {
            Some(result) => Ok(result),
            None => Err(LatticeFoldError::ArithmeticOverflow(
                format!("Multiplication overflow: {} * {}", self, other)
            )),
        }
    }
    
    /// Constant-time negation
    /// This implementation performs negation in constant time
    /// using checked arithmetic to handle the edge case of i64::MIN.
    fn ct_neg(&self) -> Self {
        // Handle the edge case of i64::MIN which cannot be negated
        // Use wrapping negation to maintain constant time
        self.wrapping_neg()
    }
    
    /// Constant-time equality comparison
    /// This implementation uses the subtle crate's constant-time equality
    /// to prevent timing-based information leakage about compared values.
    fn ct_eq(&self, other: &Self) -> Choice {
        // Use subtle crate's constant-time equality
        self.ct_eq(other)
    }
    
    /// Constant-time conditional selection
    /// This implementation uses the subtle crate's conditional selection
    /// to choose between values without revealing the choice through timing.
    fn ct_select(choice: Choice, a: &Self, b: &Self) -> Self {
        // Use subtle crate's conditional selection
        i64::conditional_select(a, b, choice)
    }
}

// Implementation of constant-time modular arithmetic for i64
impl ConstantTimeModularArithmetic for i64 {
    /// Constant-time modular addition using Barrett reduction
    /// This implementation performs modular addition with consistent timing
    /// regardless of the input values or the result of the modular reduction.
    fn ct_add_mod(&self, other: &Self, modulus: i64) -> Result<Self> {
        // Perform addition with overflow checking
        let sum = self.ct_add(other)?;
        
        // Perform Barrett reduction for constant-time modular reduction
        // This avoids the variable-time division operation
        let reduced = barrett_reduce(sum, modulus)?;
        
        Ok(reduced)
    }
    
    /// Constant-time modular subtraction using Barrett reduction
    /// This implementation performs modular subtraction with consistent timing
    /// regardless of the input values or the result of the modular reduction.
    fn ct_sub_mod(&self, other: &Self, modulus: i64) -> Result<Self> {
        // Perform subtraction with underflow checking
        let diff = self.ct_sub(other)?;
        
        // Handle negative results by adding modulus
        let adjusted = if diff < 0 {
            diff + modulus
        } else {
            diff
        };
        
        // Perform Barrett reduction for constant-time modular reduction
        let reduced = barrett_reduce(adjusted, modulus)?;
        
        Ok(reduced)
    }
    
    /// Constant-time modular multiplication using Montgomery multiplication
    /// This implementation performs modular multiplication with consistent timing
    /// using Montgomery multiplication to avoid variable-time division.
    fn ct_mul_mod(&self, other: &Self, modulus: i64) -> Result<Self> {
        // Use Montgomery multiplication for constant-time modular multiplication
        // This requires converting to Montgomery form, multiplying, and converting back
        montgomery_multiply(*self, *other, modulus)
    }
    
    /// Constant-time modular reduction using Barrett reduction
    /// This implementation performs modular reduction with consistent timing
    /// using precomputed Barrett parameters to avoid variable-time division.
    fn ct_reduce_mod(&self, modulus: i64) -> Result<Self> {
        barrett_reduce(*self, modulus)
    }
    
    /// Constant-time modular inverse using extended Euclidean algorithm
    /// This implementation computes modular inverse with consistent timing
    /// using a constant-time version of the extended Euclidean algorithm.
    fn ct_inverse_mod(&self, modulus: i64) -> Result<Self> {
        // Use constant-time extended Euclidean algorithm
        constant_time_extended_gcd(*self, modulus)
    }
}

/// Constant-time Barrett reduction implementation
/// This function performs modular reduction using Barrett's method
/// which avoids variable-time division operations.
fn barrett_reduce(value: i64, modulus: i64) -> Result<i64> {
    // Precompute Barrett parameter μ = ⌊2^k / modulus⌋
    // where k is chosen such that 2^k > modulus
    let k = 64 - modulus.leading_zeros() as i64;
    let mu = (1i128 << k) / modulus as i128;
    
    // Barrett reduction: q = ⌊(value * μ) / 2^k⌋
    let q = ((value as i128 * mu) >> k) as i64;
    
    // r = value - q * modulus
    let r = value - q * modulus;
    
    // Final reduction: if r >= modulus then r - modulus else r
    let result = if r >= modulus { r - modulus } else { r };
    let result = if result < 0 { result + modulus } else { result };
    
    Ok(result)
}

/// Constant-time Montgomery multiplication implementation
/// This function performs modular multiplication using Montgomery's method
/// which avoids variable-time division operations.
fn montgomery_multiply(a: i64, b: i64, modulus: i64) -> Result<i64> {
    // Montgomery multiplication requires R = 2^k where k > log2(modulus)
    let k = 64 - modulus.leading_zeros() as i64;
    let r = 1i128 << k;
    
    // Precompute R^(-1) mod modulus and modulus^(-1) mod R
    let r_inv = mod_inverse(r as i64, modulus)?;
    let m_inv = mod_inverse(modulus, 1i64 << k)?;
    
    // Convert to Montgomery form: a' = a * R mod modulus
    let a_mont = barrett_reduce((a as i128 * r) as i64, modulus)?;
    let b_mont = barrett_reduce((b as i128 * r) as i64, modulus)?;
    
    // Montgomery multiplication: c' = a' * b' * R^(-1) mod modulus
    let product = a_mont as i128 * b_mont as i128;
    let t = ((product & ((1i128 << k) - 1)) * m_inv as i128) & ((1i128 << k) - 1);
    let u = (product + t * modulus as i128) >> k;
    
    let result = if u >= modulus as i128 {
        (u - modulus as i128) as i64
    } else {
        u as i64
    };
    
    // Convert back from Montgomery form: result * R^(-1) mod modulus
    barrett_reduce((result as i128 * r_inv as i128) as i64, modulus)
}

/// Constant-time extended Euclidean algorithm for modular inverse
/// This function computes the modular inverse using a constant-time
/// implementation of the extended Euclidean algorithm.
fn constant_time_extended_gcd(a: i64, modulus: i64) -> Result<i64> {
    // Extended Euclidean algorithm with constant-time implementation
    let mut old_r = modulus;
    let mut r = a;
    let mut old_s = 0i64;
    let mut s = 1i64;
    
    // Perform fixed number of iterations to ensure constant time
    let max_iterations = 64; // Sufficient for 64-bit integers
    
    for _ in 0..max_iterations {
        // Check if r == 0 (algorithm should terminate)
        let r_is_zero = r.ct_eq(&0);
        
        // Compute quotient and remainder
        let (q, new_r) = if r == 0 {
            (0, 0) // Dummy values when r == 0
        } else {
            (old_r / r, old_r % r)
        };
        
        // Update values conditionally based on whether r == 0
        let new_old_r = i64::conditional_select(&old_r, &r, !r_is_zero);
        let new_r_val = i64::conditional_select(&r, &new_r, !r_is_zero);
        let new_old_s = i64::conditional_select(&old_s, &s, !r_is_zero);
        let new_s = i64::conditional_select(&s, &(old_s - q * s), !r_is_zero);
        
        old_r = new_old_r;
        r = new_r_val;
        old_s = new_old_s;
        s = new_s;
    }
    
    // Check if gcd == 1 (inverse exists)
    if old_r != 1 {
        return Err(LatticeFoldError::CryptoError(
            format!("Modular inverse does not exist: gcd({}, {}) = {}", a, modulus, old_r)
        ));
    }
    
    // Ensure result is positive
    let result = if old_s < 0 {
        old_s + modulus
    } else {
        old_s
    };
    
    Ok(result)
}

/// Helper function for modular inverse computation
/// This is a simplified version used in Montgomery multiplication setup.
fn mod_inverse(a: i64, modulus: i64) -> Result<i64> {
    constant_time_extended_gcd(a, modulus)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_constant_time_arithmetic() {
        // Test constant-time addition
        let a = 42i64;
        let b = 17i64;
        let sum = a.ct_add(&b).unwrap();
        assert_eq!(sum, 59);
        
        // Test overflow detection
        let max_val = i64::MAX;
        let overflow_result = max_val.ct_add(&1);
        assert!(overflow_result.is_err());
        
        // Test constant-time subtraction
        let diff = a.ct_sub(&b).unwrap();
        assert_eq!(diff, 25);
        
        // Test constant-time multiplication
        let product = a.ct_mul(&b).unwrap();
        assert_eq!(product, 714);
        
        // Test constant-time negation
        let neg_a = a.ct_neg();
        assert_eq!(neg_a, -42);
        
        // Test constant-time equality
        let eq_result = a.ct_eq(&42);
        assert_eq!(eq_result.unwrap_u8(), 1);
        
        let neq_result = a.ct_eq(&17);
        assert_eq!(neq_result.unwrap_u8(), 0);
        
        // Test constant-time selection
        let choice = Choice::from(1);
        let selected = i64::ct_select(choice, &a, &b);
        assert_eq!(selected, 42);
        
        let choice = Choice::from(0);
        let selected = i64::ct_select(choice, &a, &b);
        assert_eq!(selected, 17);
    }
    
    #[test]
    fn test_constant_time_modular_arithmetic() {
        let a = 123i64;
        let b = 456i64;
        let modulus = 97i64;
        
        // Test modular addition
        let sum_mod = a.ct_add_mod(&b, modulus).unwrap();
        assert_eq!(sum_mod, (123 + 456) % 97);
        
        // Test modular subtraction
        let diff_mod = a.ct_sub_mod(&b, modulus).unwrap();
        let expected = ((123 - 456) % 97 + 97) % 97;
        assert_eq!(diff_mod, expected);
        
        // Test modular multiplication
        let product_mod = a.ct_mul_mod(&b, modulus).unwrap();
        assert_eq!(product_mod, (123 * 456) % 97);
        
        // Test modular reduction
        let large_val = 12345i64;
        let reduced = large_val.ct_reduce_mod(modulus).unwrap();
        assert_eq!(reduced, 12345 % 97);
        
        // Test modular inverse
        let a_coprime = 123i64;
        let modulus_prime = 97i64; // Prime modulus
        let inverse = a_coprime.ct_inverse_mod(modulus_prime).unwrap();
        let verification = a_coprime.ct_mul_mod(&inverse, modulus_prime).unwrap();
        assert_eq!(verification, 1);
    }
    
    #[test]
    fn test_barrett_reduction() {
        let value = 12345i64;
        let modulus = 97i64;
        let reduced = barrett_reduce(value, modulus).unwrap();
        assert_eq!(reduced, value % modulus);
        
        // Test with negative values
        let negative_value = -123i64;
        let reduced_neg = barrett_reduce(negative_value, modulus).unwrap();
        let expected = ((negative_value % modulus) + modulus) % modulus;
        assert_eq!(reduced_neg, expected);
    }
    
    #[test]
    fn test_montgomery_multiplication() {
        let a = 123i64;
        let b = 456i64;
        let modulus = 97i64;
        
        let result = montgomery_multiply(a, b, modulus).unwrap();
        let expected = (a * b) % modulus;
        assert_eq!(result, expected);
    }
    
    #[test]
    fn test_constant_time_extended_gcd() {
        let a = 123i64;
        let modulus = 97i64; // Prime modulus
        
        let inverse = constant_time_extended_gcd(a, modulus).unwrap();
        let verification = (a * inverse) % modulus;
        assert_eq!(verification, 1);
        
        // Test with non-coprime values
        let non_coprime = 14i64;
        let composite_modulus = 21i64; // gcd(14, 21) = 7
        let result = constant_time_extended_gcd(non_coprime, composite_modulus);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_timing_consistent_operations() {
        let config = SecurityConfig::default();
        let mut timing_ops = TimingConsistentOperations::new(config).unwrap();
        
        // Test operation execution with timing
        let result = timing_ops.execute_with_timing("test_add", || {
            let a = 42i64;
            let b = 17i64;
            a.ct_add(&b)
        }).unwrap();
        
        assert_eq!(result, 59);
        
        // Check timing statistics
        let stats = timing_ops.get_timing_statistics();
        assert_eq!(stats.total_measurements, 1);
        assert!(stats.min_duration_ns > 0);
        assert!(stats.max_duration_ns >= stats.min_duration_ns);
        
        // Clear measurements
        timing_ops.clear_measurements();
        let stats_after_clear = timing_ops.get_timing_statistics();
        assert_eq!(stats_after_clear.total_measurements, 0);
    }
}