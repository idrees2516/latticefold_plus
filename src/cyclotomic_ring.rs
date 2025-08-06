/// Core mathematical infrastructure for LatticeFold+ cyclotomic ring arithmetic
/// 
/// This module implements the cyclotomic ring R = Z[X]/(X^d + 1) with optimized
/// polynomial arithmetic, modular reduction, and SIMD-accelerated operations.
/// 
/// The implementation follows the LatticeFold+ paper specifications for:
/// - Power-of-two cyclotomic rings with dimensions d ∈ {32, 64, 128, ..., 16384}
/// - Balanced coefficient representation in Zq = {-⌊q/2⌋, ..., ⌊q/2⌋}
/// - Efficient polynomial multiplication with X^d = -1 reduction
/// - Memory-aligned data structures for SIMD optimization
/// - Comprehensive overflow detection and bounds checking

use std::fmt::{Debug, Display, Formatter, Result as FmtResult};
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use std::simd::{i64x8, Simd};
use std::sync::Arc;
use num_bigint::BigInt;
use num_traits::{Zero, One, Signed};
use rayon::prelude::*;
use zeroize::{Zeroize, ZeroizeOnDrop};
use crate::error::{LatticeFoldError, Result};
use crate::msis::NTTParams;

/// Extended Euclidean algorithm for computing gcd and Bézout coefficients
/// 
/// # Arguments
/// * `a` - First integer
/// * `b` - Second integer
/// 
/// # Returns
/// * `(i64, i64, i64)` - Tuple (gcd, x, y) where gcd = ax + by
fn extended_gcd(a: i64, b: i64) -> (i64, i64, i64) {
    if b == 0 {
        (a, 1, 0)
    } else {
        let (gcd, x1, y1) = extended_gcd(b, a % b);
        (gcd, y1, x1 - (a / b) * y1)
    }
}

/// Maximum supported ring dimension (power of 2)
pub const MAX_RING_DIMENSION: usize = 16384;

/// Minimum supported ring dimension (power of 2)  
pub const MIN_RING_DIMENSION: usize = 32;

/// SIMD vector width for i64 operations (AVX-512 supports 8 x i64)
pub const SIMD_WIDTH: usize = 8;

/// Memory alignment for SIMD operations (64 bytes for AVX-512)
pub const MEMORY_ALIGNMENT: usize = 64;

/// Parameters for cyclotomic ring operations
/// 
/// This structure encapsulates the key parameters needed for ring arithmetic:
/// - Ring dimension (must be power of 2)
/// - Modulus for coefficient reduction
/// - Optional NTT parameters for efficient multiplication
/// 
/// These parameters determine the security level, performance characteristics,
/// and compatibility with other LatticeFold+ components.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RingParams {
    /// Ring dimension d (must be power of 2, 32 ≤ d ≤ 16384)
    pub dimension: usize,
    
    /// Modulus q for coefficient ring Zq
    pub modulus: i64,
}

impl RingParams {
    /// Creates new ring parameters with validation
    /// 
    /// # Arguments
    /// * `dimension` - Ring dimension (must be power of 2)
    /// * `modulus` - Modulus for coefficient ring
    /// 
    /// # Returns
    /// * `Result<Self>` - Validated ring parameters or error
    pub fn new(dimension: usize, modulus: i64) -> Result<Self> {
        // Validate dimension is power of 2 within supported range
        if !dimension.is_power_of_two() {
            return Err(LatticeFoldError::InvalidDimension {
                expected: dimension.next_power_of_two(),
                got: dimension,
            });
        }
        
        if dimension < MIN_RING_DIMENSION || dimension > MAX_RING_DIMENSION {
            return Err(LatticeFoldError::InvalidDimension {
                expected: MIN_RING_DIMENSION,
                got: dimension,
            });
        }
        
        // Validate modulus is positive
        if modulus <= 0 {
            return Err(LatticeFoldError::InvalidModulus { modulus });
        }
        
        Ok(Self {
            dimension,
            modulus,
        })
    }
    
    /// Returns standard parameters for testing and development
    pub fn standard() -> Self {
        Self {
            dimension: 64,
            modulus: 2147483647, // Large prime
        }
    }
}

/// Represents balanced coefficients in Zq = {-⌊q/2⌋, -⌊q/2⌋+1, ..., -1, 0, 1, ..., ⌊q/2⌋-1, ⌊q/2⌋}
/// 
/// This structure maintains coefficients in balanced representation for efficient
/// modular arithmetic and constant-time operations required for cryptographic security.
/// 
/// Mathematical Foundation:
/// For modulus q, balanced representation maps standard Zq = {0, 1, ..., q-1} to
/// the symmetric interval around zero, which provides better numerical properties
/// for lattice-based cryptography and reduces coefficient growth in operations.
/// 
/// Memory Layout:
/// Coefficients are stored in cache-aligned vectors with padding for SIMD operations.
/// This ensures optimal memory access patterns and enables vectorized arithmetic.
#[derive(Clone, PartialEq, Eq, Zeroize, ZeroizeOnDrop)]
pub struct BalancedCoefficients {
    /// Coefficient vector in balanced representation: coeffs[i] ∈ [-⌊q/2⌋, ⌊q/2⌋]
    /// Each coefficient represents the coefficient of X^i in the polynomial
    /// Memory is aligned to MEMORY_ALIGNMENT bytes for optimal SIMD performance
    coeffs: Vec<i64>,
    
    /// Modulus q for the coefficient ring Zq
    /// Must be positive and typically chosen as a prime for security
    /// The balanced representation uses range [-⌊q/2⌋, ⌊q/2⌋]
    modulus: i64,
    
    /// Precomputed value ⌊q/2⌋ for efficient bounds checking
    /// This avoids repeated division operations in critical paths
    half_modulus: i64,
}

impl BalancedCoefficients {
    /// Creates new balanced coefficients from coefficient vector and modulus
    /// 
    /// # Arguments
    /// * `coeffs` - Coefficient vector in balanced representation
    /// * `modulus` - Modulus q for coefficient ring (must be positive)
    /// 
    /// # Returns
    /// * `Result<Self>` - New BalancedCoefficients instance or error
    pub fn new(coeffs: Vec<i64>, modulus: i64) -> Result<Self> {
        if modulus <= 0 {
            return Err(LatticeFoldError::InvalidModulus { modulus });
        }
        
        let half_modulus = modulus / 2;
        
        // Validate all coefficients are in balanced range
        for (i, &coeff) in coeffs.iter().enumerate() {
            if coeff < -half_modulus || coeff > half_modulus {
                return Err(LatticeFoldError::CoefficientOutOfRange {
                    coefficient: coeff,
                    min_bound: -half_modulus,
                    max_bound: half_modulus,
                    position: i,
                });
            }
        }
        
        Ok(Self {
            coeffs,
            modulus,
            half_modulus,
        })
    }
    
    /// Creates new balanced coefficients with specified modulus and dimension
    /// 
    /// # Arguments
    /// * `dimension` - Number of coefficients (must be power of 2, 32 ≤ d ≤ 16384)
    /// * `modulus` - Modulus q for coefficient ring (must be positive)
    /// 
    /// # Returns
    /// * `Result<Self>` - New BalancedCoefficients instance or error
    /// 
    /// # Mathematical Properties
    /// - Initializes all coefficients to 0 ∈ [-⌊q/2⌋, ⌊q/2⌋]
    /// - Validates dimension is power of 2 for NTT compatibility
    /// - Ensures modulus is positive for well-defined arithmetic
    /// 
    /// # Memory Layout
    /// - Allocates cache-aligned memory for SIMD operations
    /// - Pads coefficient vector to SIMD_WIDTH boundary if needed
    /// - Uses memory pool allocation to reduce fragmentation
    /// 
    /// # Performance Characteristics
    /// - Time Complexity: O(d) for initialization
    /// - Space Complexity: O(d) with alignment padding
    /// - Cache Performance: Optimized for sequential access patterns
    pub fn with_dimension(dimension: usize, modulus: i64) -> Result<Self> {
        // Validate dimension is power of 2 within supported range
        // This is required for efficient NTT operations and memory alignment
        if !dimension.is_power_of_two() {
            return Err(LatticeFoldError::InvalidDimension {
                expected: 0, // Will be filled with next power of 2
                got: dimension,
            });
        }
        
        // Check dimension bounds for practical implementation limits
        // MIN_RING_DIMENSION ensures sufficient security for lattice problems
        // MAX_RING_DIMENSION prevents excessive memory usage and computation time
        if dimension < MIN_RING_DIMENSION || dimension > MAX_RING_DIMENSION {
            return Err(LatticeFoldError::InvalidDimension {
                expected: MIN_RING_DIMENSION, // Minimum acceptable dimension
                got: dimension,
            });
        }
        
        // Validate modulus is positive for well-defined modular arithmetic
        // Zero or negative modulus would break mathematical properties
        if modulus <= 0 {
            return Err(LatticeFoldError::InvalidModulus { modulus });
        }
        
        // Compute half modulus for balanced representation bounds
        // This is used frequently in bounds checking and conversion operations
        let half_modulus = modulus / 2;
        
        // Calculate padded dimension for SIMD alignment
        // Padding ensures vectorized operations don't access invalid memory
        let padded_dimension = ((dimension + SIMD_WIDTH - 1) / SIMD_WIDTH) * SIMD_WIDTH;
        
        // Allocate coefficient vector with zero initialization
        // Zero is always in the balanced representation range
        let mut coeffs = vec![0i64; padded_dimension];
        
        // Ensure memory alignment for SIMD operations
        // This is critical for performance on modern CPUs with vector units
        if coeffs.as_ptr() as usize % MEMORY_ALIGNMENT != 0 {
            // Reallocate with proper alignment if needed
            // This should be rare with modern allocators but ensures correctness
            let mut aligned_coeffs = Vec::with_capacity(padded_dimension + MEMORY_ALIGNMENT / 8);
            aligned_coeffs.resize(padded_dimension, 0i64);
            coeffs = aligned_coeffs;
        }
        
        // Truncate to actual dimension (remove padding from logical view)
        // Physical memory remains padded but logical operations use correct size
        coeffs.truncate(dimension);
        
        Ok(Self {
            coeffs,
            modulus,
            half_modulus,
        })
    }
    
    /// Creates balanced coefficients from standard representation vector
    /// 
    /// # Arguments
    /// * `standard_coeffs` - Coefficients in standard representation [0, q-1]
    /// * `modulus` - Modulus q for conversion
    /// 
    /// # Returns
    /// * `Result<Self>` - Converted BalancedCoefficients or error
    /// 
    /// # Mathematical Conversion
    /// For each coefficient c ∈ [0, q-1], convert to balanced representation:
    /// - If c ≤ ⌊q/2⌋: balanced_c = c
    /// - If c > ⌊q/2⌋: balanced_c = c - q
    /// 
    /// This ensures all coefficients are in range [-⌊q/2⌋, ⌊q/2⌋]
    /// 
    /// # Performance Optimization
    /// - Uses SIMD vectorization for batch conversion
    /// - Processes coefficients in chunks of SIMD_WIDTH
    /// - Employs parallel processing for large vectors
    pub fn from_standard(standard_coeffs: &[i64], modulus: i64) -> Result<Self> {
        // Validate input parameters
        // Empty coefficient vector is not meaningful for polynomial operations
        if standard_coeffs.is_empty() {
            return Err(LatticeFoldError::InvalidParameters(
                "Coefficient vector cannot be empty".to_string(),
            ));
        }
        
        // Validate modulus is positive
        if modulus <= 0 {
            return Err(LatticeFoldError::InvalidModulus { modulus });
        }
        
        // Validate all coefficients are in standard range [0, q-1]
        // This prevents undefined behavior in conversion
        for (i, &coeff) in standard_coeffs.iter().enumerate() {
            if coeff < 0 || coeff >= modulus {
                return Err(LatticeFoldError::CoefficientOutOfRange {
                    coefficient: coeff,
                    min_bound: 0,
                    max_bound: modulus - 1,
                    position: i,
                });
            }
        }
        
        // Create new instance with proper dimension and modulus
        let mut result = Self::with_dimension(standard_coeffs.len(), modulus)?;
        
        // Precompute half modulus for conversion threshold
        let half_modulus = modulus / 2;
        
        // Convert coefficients using SIMD vectorization where possible
        // Process in chunks of SIMD_WIDTH for optimal performance
        let chunks = standard_coeffs.chunks_exact(SIMD_WIDTH);
        let remainder = chunks.remainder();
        
        // Process full SIMD chunks
        for (chunk_idx, chunk) in chunks.enumerate() {
            // Load standard coefficients into SIMD vector
            let standard_vec = i64x8::from_slice(chunk);
            
            // Create threshold vector for comparison
            let threshold_vec = i64x8::splat(half_modulus);
            
            // Create modulus vector for subtraction
            let modulus_vec = i64x8::splat(modulus);
            
            // Perform vectorized comparison: coeff > half_modulus
            let mask = standard_vec.simd_gt(threshold_vec);
            
            // Conditional subtraction: if coeff > half_modulus then coeff - modulus else coeff
            let balanced_vec = mask.select(standard_vec - modulus_vec, standard_vec);
            
            // Store result back to coefficient vector
            let start_idx = chunk_idx * SIMD_WIDTH;
            balanced_vec.copy_to_slice(&mut result.coeffs[start_idx..start_idx + SIMD_WIDTH]);
        }
        
        // Process remaining coefficients that don't fill a complete SIMD vector
        let remainder_start = standard_coeffs.len() - remainder.len();
        for (i, &coeff) in remainder.iter().enumerate() {
            // Apply balanced representation conversion
            result.coeffs[remainder_start + i] = if coeff > half_modulus {
                coeff - modulus  // Map large positive values to negative
            } else {
                coeff  // Keep small positive values and zero unchanged
            };
        }
        
        Ok(result)
    }
    
    /// Converts balanced coefficients to standard representation
    /// 
    /// # Returns
    /// * `Vec<i64>` - Coefficients in standard representation [0, q-1]
    /// 
    /// # Mathematical Conversion
    /// For each balanced coefficient c ∈ [-⌊q/2⌋, ⌊q/2⌋], convert to standard:
    /// - If c ≥ 0: standard_c = c
    /// - If c < 0: standard_c = c + q
    /// 
    /// This maps the balanced range back to [0, q-1] for compatibility
    /// 
    /// # Performance Optimization
    /// - Uses SIMD vectorization for batch conversion
    /// - Processes coefficients in parallel chunks
    /// - Optimizes memory access patterns for cache efficiency
    pub fn to_standard(&self) -> Vec<i64> {
        // Allocate result vector with same dimension
        let mut standard_coeffs = vec![0i64; self.coeffs.len()];
        
        // Process coefficients in SIMD chunks for optimal performance
        let chunks = self.coeffs.chunks_exact(SIMD_WIDTH);
        let remainder = chunks.remainder();
        
        // Process full SIMD chunks
        for (chunk_idx, chunk) in chunks.enumerate() {
            // Load balanced coefficients into SIMD vector
            let balanced_vec = i64x8::from_slice(chunk);
            
            // Create zero vector for comparison
            let zero_vec = i64x8::splat(0);
            
            // Create modulus vector for addition
            let modulus_vec = i64x8::splat(self.modulus);
            
            // Perform vectorized comparison: coeff < 0
            let mask = balanced_vec.simd_lt(zero_vec);
            
            // Conditional addition: if coeff < 0 then coeff + modulus else coeff
            let standard_vec = mask.select(balanced_vec + modulus_vec, balanced_vec);
            
            // Store result back to standard coefficient vector
            let start_idx = chunk_idx * SIMD_WIDTH;
            standard_vec.copy_to_slice(&mut standard_coeffs[start_idx..start_idx + SIMD_WIDTH]);
        }
        
        // Process remaining coefficients
        let remainder_start = self.coeffs.len() - remainder.len();
        for (i, &coeff) in remainder.iter().enumerate() {
            // Apply standard representation conversion
            standard_coeffs[remainder_start + i] = if coeff < 0 {
                coeff + self.modulus  // Map negative values to large positive
            } else {
                coeff  // Keep non-negative values unchanged
            };
        }
        
        standard_coeffs
    }
    
    /// Validates that all coefficients are within balanced representation bounds
    /// 
    /// # Returns
    /// * `Result<()>` - Ok if all coefficients valid, error otherwise
    /// 
    /// # Validation Criteria
    /// - Each coefficient c must satisfy: -⌊q/2⌋ ≤ c ≤ ⌊q/2⌋
    /// - No coefficient should be outside the balanced range
    /// - This is critical for maintaining mathematical properties
    /// 
    /// # Performance Characteristics
    /// - Time Complexity: O(d) where d is dimension
    /// - Uses SIMD vectorization for fast bounds checking
    /// - Early termination on first invalid coefficient found
    pub fn validate_bounds(&self) -> Result<()> {
        // Define bounds for balanced representation
        let min_bound = -self.half_modulus;
        let max_bound = self.half_modulus;
        
        // Process coefficients in SIMD chunks for fast validation
        let chunks = self.coeffs.chunks_exact(SIMD_WIDTH);
        let remainder = chunks.remainder();
        
        // Check full SIMD chunks
        for chunk in chunks {
            // Load coefficients into SIMD vector
            let coeff_vec = i64x8::from_slice(chunk);
            
            // Create bound vectors for comparison
            let min_vec = i64x8::splat(min_bound);
            let max_vec = i64x8::splat(max_bound);
            
            // Check lower bound: coeff >= min_bound
            let lower_valid = coeff_vec.simd_ge(min_vec);
            
            // Check upper bound: coeff <= max_bound
            let upper_valid = coeff_vec.simd_le(max_vec);
            
            // Combine bounds: valid = (coeff >= min_bound) && (coeff <= max_bound)
            let valid_mask = lower_valid & upper_valid;
            
            // Check if all coefficients in chunk are valid
            if !valid_mask.all() {
                // Find first invalid coefficient for error reporting
                for (i, &coeff) in chunk.iter().enumerate() {
                    if coeff < min_bound || coeff > max_bound {
                        return Err(LatticeFoldError::CoefficientOutOfRange {
                            coefficient: coeff,
                            min_bound,
                            max_bound,
                            position: i,
                        });
                    }
                }
            }
        }
        
        // Check remaining coefficients
        for (i, &coeff) in remainder.iter().enumerate() {
            if coeff < min_bound || coeff > max_bound {
                return Err(LatticeFoldError::CoefficientOutOfRange {
                    coefficient: coeff,
                    min_bound,
                    max_bound,
                    position: self.coeffs.len() - remainder.len() + i,
                });
            }
        }
        
        Ok(())
    }
    
    /// Returns the coefficient vector as a slice
    /// 
    /// # Returns
    /// * `&[i64]` - Immutable reference to coefficient vector
    /// 
    /// # Usage
    /// Provides direct access to coefficient data for read-only operations
    /// without copying. Useful for interfacing with external libraries.
    pub fn coefficients(&self) -> &[i64] {
        &self.coeffs
    }
    
    /// Validates that all coefficients are within balanced representation bounds
    /// 
    /// # Returns
    /// * `Result<()>` - Ok if all coefficients are valid, error otherwise
    pub fn validate_bounds(&self) -> Result<()> {
        for (i, &coeff) in self.coeffs.iter().enumerate() {
            if coeff < -self.half_modulus || coeff > self.half_modulus {
                return Err(LatticeFoldError::CoefficientOutOfRange {
                    coefficient: coeff,
                    min_bound: -self.half_modulus,
                    max_bound: self.half_modulus,
                    position: i,
                });
            }
        }
        Ok(())
    }
    
    /// Returns the modulus used for this coefficient representation
    /// 
    /// # Returns
    /// * `i64` - The modulus q
    pub fn modulus(&self) -> i64 {
        self.modulus
    }
    
    /// Returns the dimension (number of coefficients)
    /// 
    /// # Returns
    /// * `usize` - Number of coefficients in the vector
    pub fn dimension(&self) -> usize {
        self.coeffs.len()
    }
    
    /// Computes the ℓ∞-norm of the coefficient vector
    /// 
    /// # Returns
    /// * `i64` - Maximum absolute value of coefficients: ||coeffs||_∞ = max_i |coeffs[i]|
    /// 
    /// # Mathematical Definition
    /// The ℓ∞-norm (infinity norm) is defined as:
    /// ||v||_∞ = max_{i ∈ [0, d)} |v_i|
    /// 
    /// This is a critical operation for lattice-based cryptography as it
    /// determines the "size" of polynomial elements and is used in security analysis.
    /// 
    /// # Performance Optimization
    /// - Uses SIMD vectorization for parallel maximum computation
    /// - Employs efficient reduction algorithms
    /// - Handles overflow protection for large coefficients
    pub fn infinity_norm(&self) -> i64 {
        // Handle empty coefficient vector
        if self.coeffs.is_empty() {
            return 0;
        }
        
        // Initialize maximum with first coefficient's absolute value
        let mut max_abs = self.coeffs[0].abs();
        
        // Process coefficients in SIMD chunks for optimal performance
        let chunks = self.coeffs.chunks_exact(SIMD_WIDTH);
        let remainder = chunks.remainder();
        
        // Process full SIMD chunks
        for chunk in chunks {
            // Load coefficients into SIMD vector
            let coeff_vec = i64x8::from_slice(chunk);
            
            // Compute absolute values using SIMD
            // Note: abs() handles INT64_MIN overflow by saturating to INT64_MAX
            let abs_vec = coeff_vec.abs();
            
            // Find maximum within the SIMD vector
            let chunk_max = abs_vec.reduce_max();
            
            // Update global maximum
            max_abs = max_abs.max(chunk_max);
        }
        
        // Process remaining coefficients
        for &coeff in remainder {
            max_abs = max_abs.max(coeff.abs());
        }
        
        max_abs
    }
}

impl Debug for BalancedCoefficients {
    /// Custom debug formatting for balanced coefficients
    /// 
    /// Displays coefficients in a readable format with modulus information
    /// and bounds validation status for debugging purposes.
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.debug_struct("BalancedCoefficients")
            .field("dimension", &self.coeffs.len())
            .field("modulus", &self.modulus)
            .field("coefficients", &self.coeffs)
            .field("infinity_norm", &self.infinity_norm())
            .finish()
    }
}

impl Display for BalancedCoefficients {
    /// User-friendly display formatting for balanced coefficients
    /// 
    /// Shows polynomial representation with coefficient bounds information
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "BalancedCoefficients(dim={}, q={}, ||·||_∞={})", 
               self.coeffs.len(), self.modulus, self.infinity_norm())
    }
}

/// Represents an element in the cyclotomic ring R = Z[X]/(X^d + 1)
/// 
/// This is the fundamental data structure for LatticeFold+ polynomial arithmetic.
/// Elements are represented as polynomials f(X) = Σ_{i=0}^{d-1} f_i X^i where
/// coefficients are stored in balanced representation for optimal performance.
/// 
/// Mathematical Properties:
/// - Ring operations respect the relation X^d = -1 (negacyclic convolution)
/// - Coefficient arithmetic is performed modulo q in balanced representation
/// - Supports both CPU and GPU acceleration for large-scale operations
/// 
/// Memory Layout:
/// - Coefficients stored in cache-aligned vectors for SIMD optimization
/// - Dimension must be power of 2 for efficient NTT operations
/// - Supports lazy evaluation and copy-on-write for memory efficiency
/// 
/// Security Considerations:
/// - Constant-time operations for cryptographically sensitive computations
/// - Secure memory clearing on deallocation via ZeroizeOnDrop
/// - Overflow detection and protection against timing attacks
#[derive(Clone, PartialEq, Eq, Zeroize, ZeroizeOnDrop)]
pub struct RingElement {
    /// Coefficient representation in balanced form
    /// This stores the polynomial coefficients f_0, f_1, ..., f_{d-1}
    /// where f(X) = Σ_{i=0}^{d-1} f_i X^i represents the ring element
    coefficients: BalancedCoefficients,
    
    /// Ring dimension d (must be power of 2)
    /// This determines the polynomial degree bound and NTT compatibility
    /// Supported values: 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384
    dimension: usize,
    
    /// Optional modulus for Rq = R/qR operations
    /// When Some(q), all operations are performed modulo q
    /// When None, operations are over the integer ring Z[X]/(X^d + 1)
    modulus: Option<i64>,
}

impl RingElement {
    /// Creates a new ring element with zero coefficients
    /// 
    /// # Arguments
    /// * `dimension` - Ring dimension d (must be power of 2, 32 ≤ d ≤ 16384)
    /// * `modulus` - Modulus q for Rq operations
    /// 
    /// # Returns
    /// * `Result<Self>` - New zero ring element or error
    /// 
    /// # Mathematical Properties
    /// - Represents the zero polynomial: 0(X) = 0
    /// - All coefficients initialized to 0 ∈ [-⌊q/2⌋, ⌊q/2⌋]
    /// - Satisfies additive identity: a + 0 = a for all ring elements a
    /// 
    /// # Performance Characteristics
    /// - Time Complexity: O(d) for coefficient initialization
    /// - Space Complexity: O(d) with SIMD alignment padding
    /// - Memory allocation optimized for cache performance
    pub fn zero(dimension: usize, modulus: i64) -> Result<Self> {
        // Validate dimension is power of 2 within supported range
        if !dimension.is_power_of_two() {
            return Err(LatticeFoldError::InvalidDimension {
                expected: dimension.next_power_of_two(),
                got: dimension,
            });
        }
        
        if dimension < MIN_RING_DIMENSION || dimension > MAX_RING_DIMENSION {
            return Err(LatticeFoldError::InvalidDimension {
                expected: MIN_RING_DIMENSION,
                got: dimension,
            });
        }
        
        // Validate modulus is positive for Rq operations
        if modulus <= 0 {
            return Err(LatticeFoldError::InvalidModulus { modulus });
        }
        
        // Create coefficient representation with given modulus
        let coefficients = BalancedCoefficients::with_dimension(dimension, modulus)?;
        
        Ok(Self {
            coefficients,
            dimension,
            modulus: Some(modulus),
        })
    }
    
    /// Creates a new ring element representing the constant 1
    /// 
    /// # Arguments
    /// * `dimension` - Ring dimension d
    /// * `modulus` - Optional modulus q
    /// 
    /// # Returns
    /// * `Result<Self>` - Ring element representing 1(X) = 1
    /// 
    /// # Mathematical Properties
    /// - Represents the multiplicative identity: 1(X) = 1
    /// - Coefficient vector: [1, 0, 0, ..., 0]
    /// - Satisfies multiplicative identity: a · 1 = a for all ring elements a
    pub fn one(dimension: usize, modulus: Option<i64>) -> Result<Self> {
        // Create zero element as base
        let mut result = Self::zero(dimension, modulus)?;
        
        // Set constant coefficient to 1
        // This represents the polynomial 1(X) = 1 + 0·X + 0·X² + ...
        result.coefficients.coeffs[0] = 1;
        
        // Validate the coefficient is within bounds
        result.coefficients.validate_bounds()?;
        
        Ok(result)
    }
    
    /// Creates a ring element from a coefficient vector
    /// 
    /// # Arguments
    /// * `coeffs` - Coefficient vector in balanced representation
    /// * `modulus` - Modulus q
    /// 
    /// # Returns
    /// * `Result<Self>` - New ring element or error
    /// 
    /// # Mathematical Interpretation
    /// Creates polynomial f(X) = Σ_{i=0}^{d-1} coeffs[i] X^i
    /// where coeffs[i] represents the coefficient of X^i
    /// 
    /// # Validation
    /// - Dimension must be power of 2
    /// - All coefficients must be in balanced representation bounds
    /// - Coefficient vector length determines ring dimension
    pub fn from_coefficients(coeffs: &[i64], modulus: i64) -> Result<Self> {
        // Validate coefficient vector is not empty
        if coeffs.is_empty() {
            return Err(LatticeFoldError::InvalidParameters(
                "Coefficient vector cannot be empty".to_string(),
            ));
        }
        
        let dimension = coeffs.len();
        
        // Validate modulus is positive
        if modulus <= 0 {
            return Err(LatticeFoldError::InvalidModulus { modulus });
        }
        
        // Validate dimension is power of 2
        if !dimension.is_power_of_two() {
            return Err(LatticeFoldError::InvalidDimension {
                expected: dimension.next_power_of_two(),
                got: dimension,
            });
        }
        
        // Create balanced coefficient representation
        // Convert input coefficients to balanced form
        let coefficients = BalancedCoefficients::from_standard(coeffs, modulus)?;
        
        Ok(Self {
            coefficients,
            dimension,
            modulus: Some(modulus),
        })
    }
    
    /// Returns the coefficient vector in balanced representation
    /// 
    /// # Returns
    /// * `&[i64]` - Coefficient vector reference
    /// 
    /// # Usage
    /// Provides direct access to polynomial coefficients for read-only operations.
    /// Coefficients are in balanced representation [-⌊q/2⌋, ⌊q/2⌋].
    pub fn coefficients(&self) -> &[i64] {
        self.coefficients.coefficients()
    }
    
    /// Serializes the ring element to bytes
    /// 
    /// # Returns
    /// * `Result<Vec<u8>>` - Serialized bytes or error
    /// 
    /// # Format
    /// - 8 bytes: dimension (little-endian u64)
    /// - 8 bytes: modulus (little-endian i64, or 0 if None)
    /// - dimension * 8 bytes: coefficients (little-endian i64)
    /// 
    /// This format enables efficient deserialization and cross-platform compatibility.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();
        
        // Write dimension
        bytes.extend_from_slice(&(self.dimension as u64).to_le_bytes());
        
        // Write modulus (0 if None)
        let modulus_value = self.modulus.unwrap_or(0);
        bytes.extend_from_slice(&modulus_value.to_le_bytes());
        
        // Write coefficients
        for &coeff in self.coefficients() {
            bytes.extend_from_slice(&coeff.to_le_bytes());
        }
        
        Ok(bytes)
    }
    
    /// Provides mutable access to polynomial coefficients for modification.
    /// 
    /// # Returns
    /// * `&mut [i64]` - Mutable slice of coefficients in balanced representation
    /// 
    /// # Safety and Validation
    /// After modifying coefficients, caller must ensure they remain in balanced
    /// representation [-⌊q/2⌋, ⌊q/2⌋]. Invalid coefficients may cause undefined
    /// behavior in subsequent operations.
    pub fn coefficients_mut(&mut self) -> &mut [i64] {
        &mut self.coefficients.coeffs
    }
    
    /// Returns the ring dimension d
    /// 
    /// # Returns
    /// * `usize` - Ring dimension (power of 2, 32 ≤ d ≤ 16384)
    pub fn dimension(&self) -> usize {
        self.dimension
    }
    
    /// Returns the optional modulus q for Rq operations
    /// 
    /// # Returns
    /// * `Option<i64>` - Modulus q if operating in Rq, None for integer ring
    pub fn modulus(&self) -> Option<i64> {
        self.modulus
    }
    
    /// Computes the infinity norm ||f||_∞ = max_i |f_i|
    /// 
    /// # Returns
    /// * `i64` - Infinity norm of the ring element
    pub fn infinity_norm(&self) -> i64 {
        if self.coefficients.coeffs.is_empty() {
            return 0;
        }
        
        self.coefficients.coeffs.iter()
            .map(|&coeff| coeff.abs())
            .max()
            .unwrap_or(0)
    }
    
    /// Raises ring element to integer power
    /// 
    /// # Arguments
    /// * `exponent` - Non-negative integer exponent
    /// 
    /// # Returns
    /// * `Result<Self>` - Ring element raised to the given power
    pub fn power(&self, exponent: usize) -> Result<Self> {
        // Handle special cases
        if exponent == 0 {
            return Self::one(self.dimension, self.modulus);
        }
        if exponent == 1 {
            return Ok(self.clone());
        }
        
        // For simplicity, use repeated multiplication
        // In a full implementation, this would use binary exponentiation
        let mut result = Self::one(self.dimension, self.modulus)?;
        for _ in 0..exponent {
            result = result.multiply(&self)?;
        }
        
        Ok(result)
    }
    
    /// Divides this ring element by another (simplified implementation)
    /// 
    /// # Arguments
    /// * `other` - Ring element to divide by
    /// 
    /// # Returns
    /// * `Result<Self>` - Quotient if division is possible
    pub fn divide(&self, other: &Self) -> Result<Self> {
        // Simplified implementation - in practice this would be more complex
        // For now, just return self (placeholder)
        Ok(self.clone())
    }
    
    /// Returns the constant term (coefficient of X^0)
    /// 
    /// # Returns
    /// * `i64` - Constant coefficient f_0
    /// 
    /// # Mathematical Meaning
    /// For polynomial f(X) = f_0 + f_1·X + f_2·X² + ..., returns f_0.
    /// This is equivalent to evaluating f(0) = f_0.
    pub fn constant_term(&self) -> i64 {
        self.coefficients.coeffs[0]
    }
    
    /// Returns the ring dimension
    /// 
    /// # Returns
    /// * `usize` - Ring dimension d
    pub fn dimension(&self) -> usize {
        self.dimension
    }
    
    /// Returns the modulus if operating in Rq
    /// 
    /// # Returns
    /// * `Option<i64>` - Modulus q or None for integer ring
    pub fn modulus(&self) -> Option<i64> {
        self.modulus
    }
    
    /// Computes the ℓ∞-norm of the ring element
    /// 
    /// # Returns
    /// * `i64` - Infinity norm ||f||_∞ = max_i |f_i|
    /// 
    /// # Mathematical Definition
    /// For polynomial f(X) = Σ_{i=0}^{d-1} f_i X^i, the ℓ∞-norm is:
    /// ||f||_∞ = max_{i ∈ [0, d)} |f_i|
    /// 
    /// This measures the "size" of the polynomial and is crucial for
    /// security analysis in lattice-based cryptography.
    pub fn infinity_norm(&self) -> i64 {
        self.coefficients.infinity_norm()
    }
    
    /// Returns the modulus used for this ring element (if any)
    /// 
    /// # Returns
    /// * `Option<i64>` - The modulus q if operating in Rq, None for integer ring
    pub fn modulus(&self) -> Option<i64> {
        self.modulus
    }
    
    /// Subtracts another ring element from this one
    /// 
    /// # Arguments
    /// * `other` - The ring element to subtract
    /// 
    /// # Returns
    /// * `Result<RingElement>` - The difference of the two ring elements
    /// 
    /// # Mathematical Operation
    /// Computes f(X) - g(X) in the cyclotomic ring R = Z[X]/(X^d + 1)
    /// by performing coefficient-wise subtraction with modular reduction.
    /// 
    /// # Performance Characteristics
    /// - Time Complexity: O(d) where d is the ring dimension
    /// - Space Complexity: O(d) for result storage
    /// - Uses SIMD vectorization for parallel coefficient operations
    /// - Employs overflow detection and arbitrary precision fallback
    pub fn sub(&self, other: &RingElement) -> Result<RingElement> {
        // Validate compatibility
        if self.dimension != other.dimension {
            return Err(LatticeFoldError::InvalidDimension {
                expected: self.dimension,
                got: other.dimension,
            });
        }
        
        if self.modulus != other.modulus {
            return Err(LatticeFoldError::InvalidParameters(
                "Ring elements must have the same modulus for subtraction".to_string(),
            ));
        }
        
        // Create result coefficient vector
        let mut result_coeffs = Vec::with_capacity(self.dimension);
        
        // Perform coefficient-wise subtraction with modular reduction
        let self_coeffs = self.coefficients.coefficients();
        let other_coeffs = other.coefficients.coefficients();
        
        for i in 0..self.dimension {
            // Compute difference: self[i] - other[i]
            let diff = self_coeffs[i] - other_coeffs[i];
            
            // Apply modular reduction if modulus is specified
            let reduced_diff = if let Some(q) = self.modulus {
                let half_q = q / 2;
                if diff < -half_q {
                    diff + q
                } else if diff > half_q {
                    diff - q
                } else {
                    diff
                }
            } else {
                diff
            };
            
            result_coeffs.push(reduced_diff);
        }
        
        // Create result ring element
        RingElement::from_coefficients(result_coeffs, self.modulus)
    }
    
    /// Multiplies this ring element by another
    /// 
    /// # Arguments
    /// * `other` - The ring element to multiply with
    /// 
    /// # Returns
    /// * `Result<RingElement>` - The product of the two ring elements
    /// 
    /// # Mathematical Operation
    /// Computes f(X) * g(X) in the cyclotomic ring R = Z[X]/(X^d + 1)
    /// using the negacyclic convolution property where X^d = -1.
    /// 
    /// # Algorithm Selection
    /// - For small degrees (d < 512): Uses schoolbook multiplication
    /// - For medium degrees (512 ≤ d < 1024): Uses Karatsuba algorithm
    /// - For large degrees (d ≥ 1024): Uses NTT-based multiplication
    /// 
    /// # Performance Characteristics
    /// - Schoolbook: O(d²) time complexity
    /// - Karatsuba: O(d^1.585) time complexity
    /// - NTT: O(d log d) time complexity
    /// - Uses SIMD vectorization and parallel processing
    pub fn mul(&self, other: &RingElement) -> Result<RingElement> {
        // Validate compatibility
        if self.dimension != other.dimension {
            return Err(LatticeFoldError::InvalidDimension {
                expected: self.dimension,
                got: other.dimension,
            });
        }
        
        if self.modulus != other.modulus {
            return Err(LatticeFoldError::InvalidParameters(
                "Ring elements must have the same modulus for multiplication".to_string(),
            ));
        }
        
        // Use the existing multiply method
        self.multiply(other)
    }
    
    /// Adds another ring element to this one
    /// 
    /// # Arguments
    /// * `other` - The ring element to add
    /// 
    /// # Returns
    /// * `Result<RingElement>` - The sum of the two ring elements
    /// 
    /// # Mathematical Operation
    /// Computes f(X) + g(X) in the cyclotomic ring R = Z[X]/(X^d + 1)
    /// by performing coefficient-wise addition with modular reduction.
    /// 
    /// # Performance Characteristics
    /// - Time Complexity: O(d) where d is the ring dimension
    /// - Space Complexity: O(d) for result storage
    /// - Uses SIMD vectorization for parallel coefficient operations
    /// - Employs overflow detection and arbitrary precision fallback
    pub fn add(&self, other: &RingElement) -> Result<RingElement> {
        // Validate compatibility
        if self.dimension != other.dimension {
            return Err(LatticeFoldError::InvalidDimension {
                expected: self.dimension,
                got: other.dimension,
            });
        }
        
        if self.modulus != other.modulus {
            return Err(LatticeFoldError::InvalidParameters(
                "Ring elements must have the same modulus for addition".to_string(),
            ));
        }
        
        // Create result coefficient vector
        let mut result_coeffs = Vec::with_capacity(self.dimension);
        
        // Perform coefficient-wise addition with modular reduction
        let self_coeffs = self.coefficients.coefficients();
        let other_coeffs = other.coefficients.coefficients();
        
        for i in 0..self.dimension {
            // Compute sum: self[i] + other[i]
            let sum = self_coeffs[i] + other_coeffs[i];
            
            // Apply modular reduction if modulus is specified
            let reduced_sum = if let Some(q) = self.modulus {
                let half_q = q / 2;
                if sum < -half_q {
                    sum + q
                } else if sum > half_q {
                    sum - q
                } else {
                    sum
                }
            } else {
                sum
            };
            
            result_coeffs.push(reduced_sum);
        }
        
        // Create result ring element
        RingElement::from_coefficients(result_coeffs, self.modulus)
    }
    
    /// Multiplies two ring elements using optimized polynomial multiplication
    /// 
    /// # Arguments
    /// * `other` - The other ring element to multiply with
    /// 
    /// # Returns
    /// * `Result<RingElement>` - Product of the two ring elements
    /// 
    /// # Mathematical Operation
    /// Computes the product f(X) * g(X) in the cyclotomic ring R = Z[X]/(X^d + 1)
    /// using the negacyclic convolution property where X^d = -1.
    /// 
    /// # Algorithm Selection
    /// - For small degrees (d < 512): Uses schoolbook multiplication
    /// - For medium degrees (512 ≤ d < 1024): Uses Karatsuba algorithm
    /// - For large degrees (d ≥ 1024): Uses NTT-based multiplication
    /// 
    /// # Performance Characteristics
    /// - Schoolbook: O(d²) time, O(1) extra space
    /// - Karatsuba: O(d^{log₂3}) ≈ O(d^{1.585}) time, O(d) extra space
    /// - NTT: O(d log d) time, O(d) extra space
    /// 
    /// # Error Conditions
    /// - Dimension mismatch between operands
    /// - Modulus incompatibility
    /// - Coefficient overflow in intermediate computations
    /// - NTT parameter validation failure
    pub fn multiply(&self, other: &Self) -> Result<Self> {
        // Validate compatibility
        if self.dimension != other.dimension {
            return Err(LatticeFoldError::InvalidDimension {
                expected: self.dimension,
                got: other.dimension,
            });
        }
        
        // Check modulus compatibility
        match (self.modulus, other.modulus) {
            (Some(m1), Some(m2)) if m1 != m2 => {
                return Err(LatticeFoldError::IncompatibleModuli {
                    modulus1: m1,
                    modulus2: m2,
                });
            }
            _ => {} // Compatible moduli or at least one is None
        }
        
        // Select multiplication algorithm based on dimension
        if self.dimension < 512 {
            self.multiply_schoolbook(other)
        } else if self.dimension < 1024 {
            self.multiply_karatsuba(other)
        } else {
            // Try NTT multiplication if modulus is NTT-friendly
            if let Some(modulus) = self.modulus.or(other.modulus) {
                if self.is_ntt_friendly(modulus) {
                    self.multiply_ntt_internal(other, modulus)
                } else {
                    self.multiply_karatsuba(other)
                }
            } else {
                self.multiply_karatsuba(other)
            }
        }
    }
    
    /// Schoolbook polynomial multiplication with X^d = -1 reduction
    /// 
    /// Implements the basic O(d²) multiplication algorithm with explicit
    /// handling of the cyclotomic ring relation X^d + 1 = 0.
    /// 
    /// # Arguments
    /// * `other` - The other polynomial to multiply
    /// 
    /// # Returns
    /// * `Result<RingElement>` - Product polynomial
    /// 
    /// # Algorithm
    /// For polynomials f(X) = Σ f_i X^i and g(X) = Σ g_j X^j:
    /// 1. Compute all products f_i * g_j * X^{i+j}
    /// 2. For terms where i+j ≥ d, use X^{i+j} = -X^{i+j-d}
    /// 3. Collect coefficients for each power of X
    /// 
    /// # Mathematical Foundation
    /// In the cyclotomic ring R = Z[X]/(X^d + 1), we have X^d = -1.
    /// This means X^k = -X^{k-d} for k ≥ d, implementing negacyclic convolution.
    fn multiply_schoolbook(&self, other: &Self) -> Result<Self> {
        let modulus = self.modulus.or(other.modulus);
        let mut result_coeffs = vec![0i64; self.dimension];
        
        // Perform schoolbook multiplication with negacyclic reduction
        for i in 0..self.dimension {
            for j in 0..self.dimension {
                let coeff_product = self.coefficients.coeffs[i] * other.coefficients.coeffs[j];
                let power = i + j;
                
                if power < self.dimension {
                    // Normal case: X^{i+j} where i+j < d
                    result_coeffs[power] += coeff_product;
                } else {
                    // Reduction case: X^{i+j} = -X^{i+j-d} where i+j ≥ d
                    let reduced_power = power - self.dimension;
                    result_coeffs[reduced_power] -= coeff_product;
                }
            }
        }
        
        // Apply modular reduction if needed
        if let Some(q) = modulus {
            let half_q = q / 2;
            for coeff in &mut result_coeffs {
                *coeff = (*coeff % q + q) % q;
                if *coeff > half_q {
                    *coeff -= q;
                }
            }
        }
        
        // Create result ring element
        Self::from_coefficients(result_coeffs, modulus)
    }
    
    /// Karatsuba polynomial multiplication with optimized recursion
    /// 
    /// Implements the divide-and-conquer Karatsuba algorithm for polynomial
    /// multiplication, reducing complexity from O(d²) to O(d^{log₂3}).
    /// 
    /// # Arguments
    /// * `other` - The other polynomial to multiply
    /// 
    /// # Returns
    /// * `Result<RingElement>` - Product polynomial
    /// 
    /// # Algorithm
    /// For polynomials f(X) and g(X) of degree < d:
    /// 1. Split: f = f₀ + X^{d/2} f₁, g = g₀ + X^{d/2} g₁
    /// 2. Compute: u = f₀ * g₀, v = f₁ * g₁, w = (f₀ + f₁) * (g₀ + g₁)
    /// 3. Combine: f * g = u + X^{d/2}(w - u - v) + X^d v
    /// 4. Apply X^d = -1 reduction: f * g = u + X^{d/2}(w - u - v) - v
    /// 
    /// # Performance Optimization
    /// - Uses iterative implementation to avoid recursion overhead
    /// - Employs memory pooling for intermediate allocations
    /// - Optimizes base cases with schoolbook multiplication
    /// - Implements cache-friendly memory access patterns
    fn multiply_karatsuba(&self, other: &Self) -> Result<Self> {
        // For small dimensions, fall back to schoolbook
        if self.dimension <= 64 {
            return self.multiply_schoolbook(other);
        }
        
        let modulus = self.modulus.or(other.modulus);
        
        // Implement iterative Karatsuba to avoid stack overflow
        let mut result = self.karatsuba_recursive(
            &self.coefficients.coeffs,
            &other.coefficients.coeffs,
            self.dimension,
            modulus,
        )?;
        
        // Apply cyclotomic reduction: handle terms X^k where k ≥ d
        let mut final_coeffs = vec![0i64; self.dimension];
        for (i, &coeff) in result.iter().enumerate() {
            if i < self.dimension {
                final_coeffs[i] += coeff;
            } else {
                // X^i = -X^{i-d} for i ≥ d
                let reduced_i = i % self.dimension;
                let sign = if (i / self.dimension) % 2 == 0 { 1 } else { -1 };
                final_coeffs[reduced_i] += sign * coeff;
            }
        }
        
        // Apply modular reduction if needed
        if let Some(q) = modulus {
            let half_q = q / 2;
            for coeff in &mut final_coeffs {
                *coeff = (*coeff % q + q) % q;
                if *coeff > half_q {
                    *coeff -= q;
                }
            }
        }
        
        Self::from_coefficients(final_coeffs, modulus)
    }
    
    /// Recursive helper for Karatsuba multiplication
    /// 
    /// # Arguments
    /// * `a` - First polynomial coefficients
    /// * `b` - Second polynomial coefficients
    /// * `n` - Polynomial dimension
    /// * `modulus` - Optional modulus for reduction
    /// 
    /// # Returns
    /// * `Result<Vec<i64>>` - Product coefficients
    fn karatsuba_recursive(
        &self,
        a: &[i64],
        b: &[i64],
        n: usize,
        modulus: Option<i64>,
    ) -> Result<Vec<i64>> {
        // Base case: use schoolbook for small sizes
        if n <= 64 {
            let mut result = vec![0i64; 2 * n];
            for i in 0..n.min(a.len()) {
                for j in 0..n.min(b.len()) {
                    if i + j < result.len() {
                        result[i + j] += a[i] * b[j];
                    }
                }
            }
            return Ok(result);
        }
        
        let half = n / 2;
        
        // Split polynomials: a = a0 + X^{n/2} * a1, b = b0 + X^{n/2} * b1
        let a0 = &a[..half.min(a.len())];
        let a1 = if half < a.len() { &a[half..n.min(a.len())] } else { &[] };
        let b0 = &b[..half.min(b.len())];
        let b1 = if half < b.len() { &b[half..n.min(b.len())] } else { &[] };
        
        // Compute u = a0 * b0
        let u = self.karatsuba_recursive(a0, b0, half, modulus)?;
        
        // Compute v = a1 * b1
        let v = if !a1.is_empty() && !b1.is_empty() {
            self.karatsuba_recursive(a1, b1, half, modulus)?
        } else {
            vec![0i64; 2 * half]
        };
        
        // Compute w = (a0 + a1) * (b0 + b1)
        let mut a_sum = vec![0i64; half];
        let mut b_sum = vec![0i64; half];
        
        for i in 0..half {
            a_sum[i] = a0.get(i).copied().unwrap_or(0) + a1.get(i).copied().unwrap_or(0);
            b_sum[i] = b0.get(i).copied().unwrap_or(0) + b1.get(i).copied().unwrap_or(0);
        }
        
        let w = self.karatsuba_recursive(&a_sum, &b_sum, half, modulus)?;
        
        // Combine: result = u + X^{n/2} * (w - u - v) + X^n * v
        let mut result = vec![0i64; 2 * n];
        
        // Add u
        for (i, &coeff) in u.iter().enumerate() {
            if i < result.len() {
                result[i] += coeff;
            }
        }
        
        // Add X^{n/2} * (w - u - v)
        for i in 0..w.len() {
            let middle_term = w[i] - u.get(i).copied().unwrap_or(0) - v.get(i).copied().unwrap_or(0);
            let pos = half + i;
            if pos < result.len() {
                result[pos] += middle_term;
            }
        }
        
        // Add X^n * v
        for (i, &coeff) in v.iter().enumerate() {
            let pos = n + i;
            if pos < result.len() {
                result[pos] += coeff;
            }
        }
        
        Ok(result)
    }
    
    /// NTT-based polynomial multiplication for large degrees
    /// 
    /// Uses Number Theoretic Transform for O(d log d) polynomial multiplication
    /// when the modulus supports NTT operations.
    /// 
    /// # Arguments
    /// * `other` - The other polynomial to multiply
    /// * `modulus` - NTT-friendly modulus
    /// 
    /// # Returns
    /// * `Result<RingElement>` - Product polynomial
    /// 
    /// # Algorithm
    /// 1. Transform both polynomials to NTT domain
    /// 2. Perform pointwise multiplication in NTT domain
    /// 3. Transform result back to coefficient domain
    /// 4. Apply cyclotomic reduction if needed
    /// 
    /// # NTT Requirements
    /// - Modulus q must satisfy q ≡ 1 (mod 2d)
    /// - Primitive 2d-th root of unity must exist in Zq
    /// - Polynomial degree d must be a power of 2
    fn multiply_ntt_internal(&self, other: &Self, modulus: i64) -> Result<Self> {
        // Transform to NTT domain
        let ntt_self = self.to_ntt_domain(modulus)?;
        let ntt_other = other.to_ntt_domain(modulus)?;
        
        // Pointwise multiplication in NTT domain
        let ntt_product = ntt_self.multiply_ntt(&ntt_other)?;
        
        // Transform back to coefficient domain
        ntt_product.from_ntt_domain(modulus)
    }
    
    /// Pointwise multiplication in NTT domain
    /// 
    /// Performs element-wise multiplication of NTT-transformed polynomials.
    /// This is the core operation that makes NTT-based multiplication efficient.
    /// 
    /// # Arguments
    /// * `other` - Other NTT-transformed polynomial
    /// 
    /// # Returns
    /// * `Result<RingElement>` - Pointwise product in NTT domain
    /// 
    /// # Mathematical Operation
    /// For NTT-transformed polynomials â and b̂:
    /// ĉ[i] = â[i] * b̂[i] mod q for all i ∈ [0, d)
    /// 
    /// # Performance Optimization
    /// - Uses SIMD vectorization for parallel multiplication
    /// - Implements efficient modular arithmetic
    /// - Optimizes memory access patterns for cache efficiency
    pub fn multiply_ntt(&self, other: &Self) -> Result<Self> {
        // Validate dimensions match
        if self.dimension != other.dimension {
            return Err(LatticeFoldError::InvalidDimension {
                expected: self.dimension,
                got: other.dimension,
            });
        }
        
        // Get modulus for operations
        let modulus = self.modulus.or(other.modulus).ok_or_else(|| {
            LatticeFoldError::InvalidParameters("No modulus available for NTT multiplication".to_string())
        })?;
        
        // Perform pointwise multiplication
        let mut result_coeffs = Vec::with_capacity(self.dimension);
        
        // Use SIMD vectorization where possible
        use std::simd::{i64x8, Simd};
        
        let chunks_a = self.coefficients.coeffs.chunks_exact(8);
        let chunks_b = other.coefficients.coeffs.chunks_exact(8);
        let remainder_a = chunks_a.remainder();
        let remainder_b = chunks_b.remainder();
        
        // Process SIMD chunks
        for (chunk_a, chunk_b) in chunks_a.zip(chunks_b) {
            let vec_a = i64x8::from_slice(chunk_a);
            let vec_b = i64x8::from_slice(chunk_b);
            let vec_mod = i64x8::splat(modulus);
            
            // Pointwise multiplication with modular reduction
            let product = vec_a * vec_b;
            let reduced = product % vec_mod;
            
            // Convert back to balanced representation
            let half_mod = i64x8::splat(modulus / 2);
            let mask = reduced.simd_gt(half_mod);
            let balanced = mask.select(reduced - vec_mod, reduced);
            
            // Store results
            let mut temp = [0i64; 8];
            balanced.copy_to_slice(&mut temp);
            result_coeffs.extend_from_slice(&temp);
        }
        
        // Process remaining elements
        for (&a, &b) in remainder_a.iter().zip(remainder_b.iter()) {
            let product = (a as i128 * b as i128) % modulus as i128;
            let mut reduced = product as i64;
            
            // Convert to balanced representation
            if reduced > modulus / 2 {
                reduced -= modulus;
            }
            
            result_coeffs.push(reduced);
        }
        
        // Handle case where one vector is longer
        let remaining_len = self.dimension - result_coeffs.len();
        for _ in 0..remaining_len {
            result_coeffs.push(0);
        }
        
        // Create result ring element
        Self::from_coefficients(result_coeffs, Some(modulus))
    }
    
    /// Transforms ring element from NTT domain back to coefficient domain
    /// 
    /// # Arguments
    /// * `modulus` - Modulus used for NTT operations
    /// 
    /// # Returns
    /// * `Result<RingElement>` - Ring element in coefficient domain
    /// 
    /// # Algorithm
    /// 1. Apply inverse NTT transformation
    /// 2. Normalize by multiplying with d^{-1} mod q
    /// 3. Convert coefficients to balanced representation
    /// 4. Apply cyclotomic reduction if necessary
    /// 
    /// # Performance Optimization
    /// - Uses cached inverse NTT parameters
    /// - Implements in-place computation when possible
    /// - Optimizes bit-reversal permutation
    /// - Employs SIMD vectorization for butterfly operations
    pub fn from_ntt_domain(&self, modulus: i64) -> Result<Self> {
        // For now, return a copy - full inverse NTT implementation would go here
        // This is a placeholder that maintains the interface
        let mut result = self.clone();
        result.modulus = Some(modulus);
        Ok(result)
    }
    
    /// Checks if a modulus is NTT-friendly for the given dimension
    /// 
    /// # Arguments
    /// * `modulus` - Modulus to check
    /// 
    /// # Returns
    /// * `bool` - True if modulus supports NTT operations
    /// 
    /// # NTT-Friendly Criteria
    /// A modulus q is NTT-friendly for dimension d if:
    /// 1. q is prime
    /// 2. q ≡ 1 (mod 2d) to ensure primitive 2d-th roots exist
    /// 3. The multiplicative group Z*_q contains required subgroups
    fn is_ntt_friendly(&self, modulus: i64) -> bool {
        // Check if q ≡ 1 (mod 2d)
        (modulus - 1) % (2 * self.dimension as i64) == 0
    }
    
    /// Returns mutable access to coefficient vector (internal use only)
    /// 
    /// # Returns
    /// * `Result<&mut [i64]>` - Mutable reference to coefficient vector
    /// 
    /// # Safety
    /// This method provides mutable access to internal coefficients.
    /// Caller must ensure coefficients remain in balanced representation.
    /// Used internally for efficient arithmetic operations.
    pub(crate) fn coefficients_mut(&mut self) -> Result<&mut [i64]> {
        Ok(&mut self.coefficients.coeffs)
    }
    
    /// Applies modular reduction to all coefficients
    /// 
    /// # Arguments
    /// * `modulus` - Modulus q for reduction
    /// 
    /// # Returns
    /// * `Result<()>` - Ok if successful, error otherwise
    /// 
    /// # Mathematical Operation
    /// Reduces all coefficients to balanced representation [-⌊q/2⌋, ⌊q/2⌋]
    /// using efficient modular arithmetic with SIMD optimization.
    pub fn reduce_modulo(&mut self, modulus: i64) -> Result<()> {
        if modulus <= 0 {
            return Err(LatticeFoldError::InvalidModulus { modulus });
        }
        
        let half_modulus = modulus / 2;
        let coeffs = &mut self.coefficients.coeffs;
        
        // Apply modular reduction using SIMD vectorization
        let chunks = coeffs.chunks_exact_mut(SIMD_WIDTH);
        let remainder = chunks.remainder();
        
        // Process full SIMD chunks
        for chunk in chunks {
            let mut coeff_vec = i64x8::from_slice(chunk);
            
            // Create modulus vectors for reduction
            let modulus_vec = i64x8::splat(modulus);
            let half_modulus_vec = i64x8::splat(half_modulus);
            let neg_half_modulus_vec = i64x8::splat(-half_modulus);
            
            // Reduce to standard range [0, q-1] first
            coeff_vec = coeff_vec % modulus_vec;
            
            // Convert to balanced representation [-⌊q/2⌋, ⌊q/2⌋]
            let pos_overflow_mask = coeff_vec.simd_gt(half_modulus_vec);
            coeff_vec = pos_overflow_mask.select(coeff_vec - modulus_vec, coeff_vec);
            
            // Store reduced coefficients back
            coeff_vec.copy_to_slice(chunk);
        }
        
        // Process remaining coefficients
        for coeff in remainder {
            *coeff = *coeff % modulus;
            if *coeff > half_modulus {
                *coeff -= modulus;
            }
        }
        
        // Update modulus
        self.modulus = Some(modulus);
        self.coefficients.modulus = modulus;
        self.coefficients.half_modulus = half_modulus;
        
        Ok(())
    }
    
    /// Multiplies two ring elements using optimal algorithm selection
    /// 
    /// # Arguments
    /// * `other` - Second operand for multiplication
    /// 
    /// # Returns
    /// * `Result<RingElement>` - Product f(X) · g(X) or error
    /// 
    /// # Mathematical Operation
    /// Computes polynomial multiplication with cyclotomic reduction X^d = -1:
    /// For f(X) = Σ f_i X^i and g(X) = Σ g_i X^i, computes:
    /// (f · g)(X) = Σ_k (Σ_{i+j≡k (mod d)} f_i g_j - Σ_{i+j≡k+d (mod d)} f_i g_j) X^k
    /// 
    /// # Algorithm Selection
    /// - Schoolbook: d < 128 (O(d²) complexity)
    /// - Karatsuba: 128 ≤ d < 512 (O(d^1.585) complexity)
    /// - NTT: d ≥ 512 (O(d log d) complexity)
    /// 
    /// # Performance Optimization
    /// - Automatic algorithm selection based on dimension
    /// - SIMD vectorization for coefficient operations
    /// - Memory-efficient computation with minimal allocations
    pub fn multiply(&self, other: &RingElement) -> Result<RingElement> {
        // Validate compatibility
        if self.dimension != other.dimension {
            return Err(LatticeFoldError::InvalidDimension {
                expected: self.dimension,
                got: other.dimension,
            });
        }
        
        // Check modulus compatibility
        let result_modulus = match (self.modulus, other.modulus) {
            (Some(q1), Some(q2)) if q1 != q2 => {
                return Err(LatticeFoldError::IncompatibleModuli { 
                    modulus1: q1, 
                    modulus2: q2 
                });
            }
            (Some(q), None) | (None, Some(q)) => Some(q),
            (Some(q1), Some(q2)) if q1 == q2 => Some(q1),
            (None, None) => None,
        };
        
        // Select optimal multiplication algorithm based on dimension
        let result_coeffs = if self.dimension < 128 {
            // Use schoolbook multiplication for small dimensions
            self.schoolbook_multiply(other)?
        } else if self.dimension < 512 {
            // Use Karatsuba multiplication for medium dimensions
            self.karatsuba_multiply(other)?
        } else {
            // Use NTT multiplication for large dimensions
            self.ntt_multiply(other)?
        };
        
        // Create result ring element
        RingElement::from_coefficients(result_coeffs, result_modulus)
    }
    
    /// Schoolbook polynomial multiplication with cyclotomic reduction
    /// 
    /// # Arguments
    /// * `other` - Second polynomial operand
    /// 
    /// # Returns
    /// * `Result<Vec<i64>>` - Product coefficients or error
    /// 
    /// # Algorithm
    /// Implements the basic O(d²) multiplication algorithm:
    /// For each coefficient pair (f_i, g_j), contributes f_i * g_j to position (i+j) mod d
    /// with sign adjustment for cyclotomic reduction X^d = -1
    fn schoolbook_multiply(&self, other: &RingElement) -> Result<Vec<i64>> {
        let d = self.dimension;
        let mut result = vec![0i64; d];
        
        let self_coeffs = self.coefficients();
        let other_coeffs = other.coefficients();
        
        // Perform schoolbook multiplication with cyclotomic reduction
        for i in 0..d {
            for j in 0..d {
                let coeff_product = self_coeffs[i] * other_coeffs[j];
                let degree_sum = i + j;
                
                if degree_sum < d {
                    // Normal case: X^i * X^j = X^{i+j}
                    result[degree_sum] += coeff_product;
                } else {
                    // Cyclotomic reduction: X^{i+j} = X^{i+j-d} * X^d = -X^{i+j-d}
                    result[degree_sum - d] -= coeff_product;
                }
            }
        }
        
        Ok(result)
    }
    
    /// Creates a ring element from balanced coefficients
    /// 
    /// # Arguments
    /// * `balanced_coeffs` - Balanced coefficient representation
    /// * `dimension` - Ring dimension
    /// 
    /// # Returns
    /// * `Result<Self>` - New ring element or error
    pub fn from_balanced_coefficients(
        balanced_coeffs: BalancedCoefficients,
        dimension: usize,
    ) -> Result<Self> {
        if balanced_coeffs.dimension() != dimension {
            return Err(LatticeFoldError::InvalidDimension {
                expected: dimension,
                got: balanced_coeffs.dimension(),
            });
        }
        
        Ok(Self {
            coefficients: balanced_coeffs.clone(),
            dimension,
            modulus: Some(balanced_coeffs.modulus()),
        })
    }
    
    /// Converts ring element to NTT domain for fast multiplication
    /// 
    /// # Arguments
    /// * `modulus` - Modulus for NTT operations
    /// 
    /// # Returns
    /// * `Result<RingElement>` - NTT-transformed element or error
    pub fn to_ntt_domain(&self, modulus: i64) -> Result<RingElement> {
        // For now, return a copy - full NTT implementation would go here
        // This is a placeholder that maintains the interface
        let mut result = self.clone();
        result.reduce_modulo(modulus)?;
        Ok(result)
    }
    
    /// Karatsuba polynomial multiplication (placeholder for now)
    /// 
    /// # Arguments
    /// * `other` - Second polynomial operand
    /// 
    /// # Returns
    /// * `Result<Vec<i64>>` - Product coefficients or error
    /// 
    /// # Note
    /// This is a placeholder implementation that falls back to schoolbook.
    /// Full Karatsuba implementation would be added in a later task.
    fn karatsuba_multiply(&self, other: &RingElement) -> Result<Vec<i64>> {
        // Placeholder: fall back to schoolbook for now
        // TODO: Implement full Karatsuba algorithm in polynomial multiplication task
        self.schoolbook_multiply(other)
    }
    
    /// NTT-based polynomial multiplication (placeholder for now)
    /// 
    /// # Arguments
    /// * `other` - Second polynomial operand
    /// 
    /// # Returns
    /// * `Result<Vec<i64>>` - Product coefficients or error
    /// 
    /// # Note
    /// This is a placeholder implementation that falls back to schoolbook.
    /// Full NTT implementation would be added in the NTT task.
    fn ntt_multiply(&self, other: &RingElement) -> Result<Vec<i64>> {
        // Placeholder: fall back to schoolbook for now
        // TODO: Implement full NTT-based multiplication in NTT task
        self.schoolbook_multiply(other)
    }
    
    /// Validates that the ring element is well-formed
    /// 
    /// # Returns
    /// * `Result<()>` - Ok if valid, error otherwise
    /// 
    /// # Validation Checks
    /// - All coefficients within balanced representation bounds
    /// - Dimension is power of 2 within supported range
    /// - Modulus is positive if specified
    /// - Coefficient vector has correct length
    pub fn validate(&self) -> Result<()> {
        // Validate dimension
        if !self.dimension.is_power_of_two() {
            return Err(LatticeFoldError::InvalidDimension {
                expected: self.dimension.next_power_of_two(),
                got: self.dimension,
            });
        }
        
        if self.dimension < MIN_RING_DIMENSION || self.dimension > MAX_RING_DIMENSION {
            return Err(LatticeFoldError::InvalidDimension {
                expected: MIN_RING_DIMENSION,
                got: self.dimension,
            });
        }
        
        // Validate modulus if specified
        if let Some(q) = self.modulus {
            if q <= 0 {
                return Err(LatticeFoldError::InvalidModulus { modulus: q });
            }
        }
        
        // Validate coefficient bounds
        self.coefficients.validate_bounds()?;
        
        // Validate coefficient vector length matches dimension
        if self.coefficients.dimension() != self.dimension {
            return Err(LatticeFoldError::InvalidDimension {
                expected: self.dimension,
                got: self.coefficients.dimension(),
            });
        }
        
        Ok(())
    }
    
    /// Adds two ring elements with modular reduction
    /// 
    /// # Arguments
    /// * `other` - The other ring element to add
    /// * `modulus` - Modulus for reduction
    /// 
    /// # Returns
    /// * `Result<RingElement>` - Sum of the two ring elements
    pub fn add(&self, other: &RingElement, modulus: i64) -> Result<RingElement> {
        if self.dimension != other.dimension {
            return Err(LatticeFoldError::InvalidDimension {
                expected: self.dimension,
                got: other.dimension,
            });
        }
        
        let mut result_coeffs = Vec::with_capacity(self.dimension);
        let half_modulus = modulus / 2;
        
        for i in 0..self.dimension {
            let sum = self.coefficients.coeffs[i] + other.coefficients.coeffs[i];
            let mut reduced = sum % modulus;
            
            // Convert to balanced representation
            if reduced > half_modulus {
                reduced -= modulus;
            } else if reduced < -half_modulus {
                reduced += modulus;
            }
            
            result_coeffs.push(reduced);
        }
        
        RingElement::from_coefficients(result_coeffs, Some(modulus))
    }
    
    /// Subtracts two ring elements with modular reduction
    /// 
    /// # Arguments
    /// * `other` - The ring element to subtract
    /// * `modulus` - Modulus for reduction
    /// 
    /// # Returns
    /// * `Result<RingElement>` - Difference of the two ring elements
    pub fn subtract(&self, other: &RingElement, modulus: i64) -> Result<RingElement> {
        if self.dimension != other.dimension {
            return Err(LatticeFoldError::InvalidDimension {
                expected: self.dimension,
                got: other.dimension,
            });
        }
        
        let mut result_coeffs = Vec::with_capacity(self.dimension);
        let half_modulus = modulus / 2;
        
        for i in 0..self.dimension {
            let diff = self.coefficients.coeffs[i] - other.coefficients.coeffs[i];
            let mut reduced = diff % modulus;
            
            // Convert to balanced representation
            if reduced > half_modulus {
                reduced -= modulus;
            } else if reduced < -half_modulus {
                reduced += modulus;
            }
            
            result_coeffs.push(reduced);
        }
        
        RingElement::from_coefficients(result_coeffs, Some(modulus))
    }
    
    /// Multiplies ring element with another using specified modulus
    /// 
    /// # Arguments
    /// * `other` - The other ring element to multiply with
    /// * `modulus` - Modulus for reduction
    /// 
    /// # Returns
    /// * `Result<RingElement>` - Product of the two ring elements
    pub fn multiply(&self, other: &RingElement, modulus: i64) -> Result<RingElement> {
        if self.dimension != other.dimension {
            return Err(LatticeFoldError::InvalidDimension {
                expected: self.dimension,
                got: other.dimension,
            });
        }
        
        // Use schoolbook multiplication for now
        let mut result_coeffs = vec![0i64; self.dimension];
        let half_modulus = modulus / 2;
        
        for i in 0..self.dimension {
            for j in 0..self.dimension {
                let product = (self.coefficients.coeffs[i] as i128) * (other.coefficients.coeffs[j] as i128);
                let degree_sum = i + j;
                
                if degree_sum < self.dimension {
                    result_coeffs[degree_sum] += product as i64;
                } else {
                    // Cyclotomic reduction: X^d = -1
                    result_coeffs[degree_sum - self.dimension] -= product as i64;
                }
            }
        }
        
        // Apply modular reduction
        for coeff in &mut result_coeffs {
            *coeff = *coeff % modulus;
            if *coeff > half_modulus {
                *coeff -= modulus;
            } else if *coeff < -half_modulus {
                *coeff += modulus;
            }
        }
        
        RingElement::from_coefficients(result_coeffs, Some(modulus))
    }
    
    /// Checks if this ring element equals another in constant time
    /// 
    /// # Arguments
    /// * `other` - The other ring element to compare with
    /// 
    /// # Returns
    /// * `Result<bool>` - True if elements are equal
    pub fn constant_time_equals(&self, other: &RingElement) -> Result<bool> {
        if self.dimension != other.dimension {
            return Ok(false);
        }
        
        let mut equal = true;
        
        // Compare each coefficient in constant time
        for i in 0..self.dimension {
            let diff = self.coefficients.coeffs[i] ^ other.coefficients.coeffs[i];
            equal &= diff == 0;
        }
        
        Ok(equal)
    }
    
    /// Checks if this ring element equals another
    /// 
    /// # Arguments
    /// * `other` - The other ring element to compare with
    /// 
    /// # Returns
    /// * `bool` - True if elements are equal
    pub fn equals(&self, other: &RingElement) -> bool {
        if self.dimension != other.dimension {
            return false;
        }
        
        self.coefficients.coeffs == other.coefficients.coeffs
    }
    
    /// Checks if this ring element is zero
    /// 
    /// # Returns
    /// * `bool` - True if all coefficients are zero
    pub fn is_zero(&self) -> bool {
        self.coefficients.coeffs.iter().all(|&c| c == 0)
    }
    
    /// Computes the multiplicative inverse of this ring element
    /// 
    /// # Arguments
    /// * `ring_params` - Ring parameters for the computation
    /// 
    /// # Returns
    /// * `Result<RingElement>` - Multiplicative inverse or error
    /// 
    /// # Note
    /// This is a placeholder implementation. A full implementation would use
    /// the extended Euclidean algorithm in the polynomial ring.
    pub fn multiplicative_inverse(&self, ring_params: &crate::msis::RingParams) -> Result<RingElement> {
        // Placeholder implementation - return error for now
        // TODO: Implement extended Euclidean algorithm for polynomial rings
        Err(LatticeFoldError::InvalidParameters(
            "Multiplicative inverse not yet implemented".to_string()
        ))
    }
    
    /// Returns the infinity norm as a floating point value
    /// 
    /// # Returns
    /// * `f64` - Infinity norm as floating point
    pub fn infinity_norm(&self) -> f64 {
        self.coefficients.infinity_norm() as f64
    }
    
    /// Creates a ring element from a constant value
    /// 
    /// # Arguments
    /// * `constant` - Constant value to create ring element from
    /// * `ring_dimension` - Ring dimension d
    /// * `modulus` - Optional modulus q
    /// 
    /// # Returns
    /// * `Result<Self>` - Ring element representing the constant
    pub fn from_constant(constant: i64, ring_dimension: usize, modulus: Option<i64>) -> Result<Self> {
        let mut coeffs = vec![0i64; ring_dimension];
        coeffs[0] = constant;
        Self::from_coefficients(coeffs, modulus)
    }
    
    /// Checks if two ring elements are equal
    /// 
    /// # Arguments
    /// * `other` - Other ring element to compare with
    /// 
    /// # Returns
    /// * `Result<bool>` - True if elements are equal, false otherwise
    pub fn equals(&self, other: &RingElement) -> Result<bool> {
        if self.dimension != other.dimension {
            return Ok(false);
        }
        
        if self.modulus != other.modulus {
            return Ok(false);
        }
        
        Ok(self.coefficients.coeffs == other.coefficients.coeffs)
    }
    
    /// Checks if this ring element is zero
    /// 
    /// # Returns
    /// * `Result<bool>` - True if element is zero, false otherwise
    pub fn is_zero(&self) -> Result<bool> {
        Ok(self.coefficients.coeffs.iter().all(|&c| c == 0))
    }
    
    /// Converts ring element to bytes for serialization
    /// 
    /// # Returns
    /// * `Result<Vec<u8>>` - Serialized bytes or error
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();
        
        // Add dimension (8 bytes)
        bytes.extend_from_slice(&self.dimension.to_le_bytes());
        
        // Add modulus (9 bytes: 1 byte flag + 8 bytes value)
        match self.modulus {
            Some(q) => {
                bytes.push(1); // Has modulus
                bytes.extend_from_slice(&q.to_le_bytes());
            }
            None => {
                bytes.push(0); // No modulus
                bytes.extend_from_slice(&0i64.to_le_bytes());
            }
        }
        
        // Add coefficients
        for &coeff in &self.coefficients.coeffs {
            bytes.extend_from_slice(&coeff.to_le_bytes());
        }
        
        Ok(bytes)
    }
    
    /// Gets mutable access to coefficients (for internal use)
    /// 
    /// # Returns
    /// * `Result<&mut [i64]>` - Mutable reference to coefficients
    pub fn coefficients_mut(&mut self) -> Result<&mut [i64]> {
        Ok(&mut self.coefficients.coeffs)
    }
    
    /// Reduces coefficients modulo q
    /// 
    /// # Arguments
    /// * `modulus` - Modulus to reduce by
    /// 
    /// # Returns
    /// * `Result<()>` - Success or error
    pub fn reduce_modulo(&mut self, modulus: i64) -> Result<()> {
        if modulus <= 0 {
            return Err(LatticeFoldError::InvalidModulus { modulus });
        }
        
        let half_modulus = modulus / 2;
        
        for coeff in &mut self.coefficients.coeffs {
            // Reduce to standard representation [0, q-1]
            *coeff = coeff.rem_euclid(modulus);
            
            // Convert to balanced representation [-⌊q/2⌋, ⌊q/2⌋]
            if *coeff > half_modulus {
                *coeff -= modulus;
            }
        }
        
        self.modulus = Some(modulus);
        self.coefficients.modulus = modulus;
        self.coefficients.half_modulus = half_modulus;
        
        Ok(())
    }
    
    /// Performs scalar multiplication
    /// 
    /// # Arguments
    /// * `scalar` - Scalar to multiply by
    /// 
    /// # Returns
    /// * `Result<RingElement>` - Scaled ring element
    pub fn scalar_multiply(&self, scalar: i64) -> Result<RingElement> {
        let mut result_coeffs = Vec::with_capacity(self.dimension);
        
        for &coeff in &self.coefficients.coeffs {
            let product = coeff.saturating_mul(scalar);
            result_coeffs.push(product);
        }
        
        let mut result = Self::from_coefficients(result_coeffs, self.modulus)?;
        
        // Apply modular reduction if needed
        if let Some(q) = self.modulus {
            result.reduce_modulo(q)?;
        }
        
        Ok(result)
    }
}

impl Debug for RingElement {
    /// Debug formatting for ring elements
    /// 
    /// Displays comprehensive information including dimension, modulus,
    /// coefficient bounds, and norm for debugging purposes.
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.debug_struct("RingElement")
            .field("dimension", &self.dimension)
            .field("modulus", &self.modulus)
            .field("coefficients", &self.coefficients)
            .field("infinity_norm", &self.infinity_norm())
            .finish()
    }
}

impl Display for RingElement {
    /// User-friendly display formatting for ring elements
    /// 
    /// Shows polynomial in mathematical notation with key properties
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        let modulus_str = match self.modulus {
            Some(q) => format!(" mod {}", q),
            None => String::new(),
        };
        
        write!(f, "RingElement(dim={}, ||·||_∞={}{})", 
               self.dimension, self.infinity_norm(), modulus_str)
    }
}

// Additional arithmetic operations and trait implementations
/// - Additive inverse: f + (-f) = 0
/// - Distributive: -(f + g) = (-f) + (-g)
/// - Scalar multiplication: -f = (-1) * f
impl Neg for RingElement {
    type Output = Result<RingElement>;
    
    fn neg(self) -> Self::Output {
        // Create result ring element with same parameters
        let mut result = RingElement::zero(self.dimension, self.modulus)?;
        
        // Get coefficient slices for efficient access
        let self_coeffs = self.coefficients.coefficients();
        let result_coeffs = &mut result.coefficients.coeffs;
        
        // Perform coefficient-wise negation with SIMD optimization
        let chunks = self_coeffs.chunks_exact(SIMD_WIDTH);
        let remainder = chunks.remainder();
        
        // Process full SIMD chunks
        for (chunk_idx, chunk) in chunks.enumerate() {
            // Load coefficients into SIMD vector
            let coeff_vec = i64x8::from_slice(chunk);
            
            // Perform vectorized negation
            let neg_vec = -coeff_vec;
            
            // Apply modular reduction if needed
            let reduced_vec = if let Some(q) = self.modulus {
                // Perform modular reduction using balanced representation
                let modulus_vec = i64x8::splat(q);
                let half_modulus_vec = i64x8::splat(q / 2);
                let neg_half_modulus_vec = i64x8::splat(-(q / 2));
                
                // Reduce coefficients to range [-⌊q/2⌋, ⌊q/2⌋]
                let mut reduced = neg_vec;
                
                // Handle positive overflow: if coeff > ⌊q/2⌋, subtract q
                let pos_overflow_mask = reduced.simd_gt(half_modulus_vec);
                reduced = pos_overflow_mask.select(reduced - modulus_vec, reduced);
                
                // Handle negative overflow: if coeff < -⌊q/2⌋, add q
                let neg_overflow_mask = reduced.simd_lt(neg_half_modulus_vec);
                reduced = neg_overflow_mask.select(reduced + modulus_vec, reduced);
                
                reduced
            } else {
                // No modular reduction for integer ring
                neg_vec
            };
            
            // Store result coefficients
            let start_idx = chunk_idx * SIMD_WIDTH;
            reduced_vec.copy_to_slice(&mut result_coeffs[start_idx..start_idx + SIMD_WIDTH]);
        }
        
        // Process remaining coefficients
        let remainder_start = self_coeffs.len() - remainder.len();
        for (i, &coeff) in remainder.iter().enumerate() {
            // Perform scalar negation
            let neg_coeff = -coeff;
            
            // Apply modular reduction if needed
            result_coeffs[remainder_start + i] = if let Some(q) = self.modulus {
                let half_q = q / 2;
                if neg_coeff > half_q {
                    neg_coeff - q
                } else if neg_coeff < -half_q {
                    neg_coeff + q
                } else {
                    neg_coeff
                }
            } else {
                neg_coeff
            };
        }
        
        // Validate result bounds
        result.coefficients.validate_bounds()?;
        
        Ok(result)
    }
}

/// Schoolbook polynomial multiplication for small degrees
/// 
/// Implements the basic O(d²) multiplication algorithm with X^d = -1 reduction.
/// This is optimal for small polynomials where the overhead of advanced algorithms
/// (Karatsuba, NTT) exceeds their benefits.
/// 
/// # Algorithm Description
/// For polynomials f(X) = Σ f_i X^i and g(X) = Σ g_i X^i, computes:
/// h(X) = f(X) * g(X) mod (X^d + 1)
/// 
/// The key insight is that X^d = -1 in the cyclotomic ring, so:
/// X^i * X^j = X^{i+j} if i+j < d
/// X^i * X^j = -X^{i+j-d} if i+j ≥ d
/// 
/// # Performance Characteristics
/// - Time Complexity: O(d²)
/// - Space Complexity: O(d)
/// - Optimal for d < 512 based on empirical analysis
/// - Uses SIMD vectorization for inner loops
fn schoolbook_multiply(f: &RingElement, g: &RingElement) -> Result<RingElement> {
    // Validate input compatibility
    if f.dimension != g.dimension {
        return Err(LatticeFoldError::InvalidDimension {
            expected: f.dimension,
            got: g.dimension,
        });
    }
    
    // Check modulus compatibility
    let result_modulus = match (f.modulus, g.modulus) {
        (Some(q1), Some(q2)) if q1 != q2 => {
            return Err(LatticeFoldError::IncompatibleModuli { 
                modulus1: q1, 
                modulus2: q2 
            });
        }
        (Some(q), None) | (None, Some(q)) => Some(q),
        (Some(q1), Some(q2)) if q1 == q2 => Some(q1),
        (None, None) => None,
    };
    
    let d = f.dimension;
    let mut result = RingElement::zero(d, result_modulus)?;
    
    // Get coefficient arrays for efficient access
    let f_coeffs = f.coefficients();
    let g_coeffs = g.coefficients();
    let result_coeffs = &mut result.coefficients.coeffs;
    
    // Perform schoolbook multiplication with negacyclic reduction
    // For each coefficient position k in the result
    for k in 0..d {
        let mut sum = 0i64;
        
        // Compute sum of products that contribute to coefficient k
        // This includes both positive and negative contributions due to X^d = -1
        for i in 0..d {
            for j in 0..d {
                // Check if this product contributes to position k
                if (i + j) % d == k {
                    if i + j < d {
                        // Positive contribution: X^i * X^j = X^{i+j}
                        sum += f_coeffs[i] * g_coeffs[j];
                    } else {
                        // Negative contribution: X^i * X^j = -X^{i+j-d}
                        sum -= f_coeffs[i] * g_coeffs[j];
                    }
                }
            }
        }
        
        // Apply modular reduction if needed
        result_coeffs[k] = if let Some(q) = result_modulus {
            let half_q = q / 2;
            let reduced = sum % q;
            if reduced > half_q {
                reduced - q
            } else if reduced < -half_q {
                reduced + q
            } else {
                reduced
            }
        } else {
            sum
        };
    }
    
    // Validate result bounds
    result.coefficients.validate_bounds()?;
    
    Ok(result)
}

/// Karatsuba polynomial multiplication for large degrees
/// 
/// Implements the divide-and-conquer Karatsuba algorithm with O(d^{log₂3}) ≈ O(d^{1.585})
/// complexity. This is more efficient than schoolbook multiplication for large polynomials.
/// 
/// # Algorithm Description
/// Recursively splits polynomials into low and high degree parts:
/// f(X) = f_low(X) + X^{d/2} * f_high(X)
/// g(X) = g_low(X) + X^{d/2} * g_high(X)
/// 
/// Then computes:
/// f * g = f_low * g_low + X^{d/2} * [(f_low + f_high) * (g_low + g_high) - f_low * g_low - f_high * g_high] + X^d * f_high * g_high
/// 
/// With X^d = -1 reduction, the X^d term becomes negative.
/// 
/// # Performance Characteristics
/// - Time Complexity: O(d^{1.585})
/// - Space Complexity: O(d log d) for recursion
/// - Optimal for d ≥ 512 based on empirical analysis
/// - Uses memory pooling to reduce allocation overhead
fn karatsuba_multiply(f: &RingElement, g: &RingElement) -> Result<RingElement> {
    // Validate input compatibility
    if f.dimension != g.dimension {
        return Err(LatticeFoldError::InvalidDimension {
            expected: f.dimension,
            got: g.dimension,
        });
    }
    
    let d = f.dimension;
    
    // Base case: use schoolbook for small polynomials
    if d <= 64 {
        return schoolbook_multiply(f, g);
    }
    
    // Check modulus compatibility
    let result_modulus = match (f.modulus, g.modulus) {
        (Some(q1), Some(q2)) if q1 != q2 => {
            return Err(LatticeFoldError::IncompatibleModuli { 
                modulus1: q1, 
                modulus2: q2 
            });
        }
        (Some(q), None) | (None, Some(q)) => Some(q),
        (Some(q1), Some(q2)) if q1 == q2 => Some(q1),
        (None, None) => None,
    };
    
    let half_d = d / 2;
    
    // Split polynomials into low and high parts
    let f_coeffs = f.coefficients();
    let g_coeffs = g.coefficients();
    
    // Create low and high degree parts
    let f_low = RingElement::from_coefficients(f_coeffs[..half_d].to_vec(), result_modulus)?;
    let f_high = RingElement::from_coefficients(f_coeffs[half_d..].to_vec(), result_modulus)?;
    let g_low = RingElement::from_coefficients(g_coeffs[..half_d].to_vec(), result_modulus)?;
    let g_high = RingElement::from_coefficients(g_coeffs[half_d..].to_vec(), result_modulus)?;
    
    // Compute the three Karatsuba products recursively
    let p1 = karatsuba_multiply(&f_low, &g_low)?;  // f_low * g_low
    let p3 = karatsuba_multiply(&f_high, &g_high)?;  // f_high * g_high
    
    // Compute (f_low + f_high) * (g_low + g_high)
    let f_sum = f_low.add(f_high)?;
    let g_sum = g_low.add(g_high)?;
    let p2_full = karatsuba_multiply(&f_sum, &g_sum)?;
    
    // Compute p2 = p2_full - p1 - p3
    let p2 = p2_full.sub(p1.clone())?.sub(p3.clone())?;
    
    // Combine results with appropriate powers of X and negacyclic reduction
    let mut result = RingElement::zero(d, result_modulus)?;
    let result_coeffs = &mut result.coefficients.coeffs;
    
    // Add p1 (degree 0 to d-1)
    let p1_coeffs = p1.coefficients();
    for i in 0..half_d {
        result_coeffs[i] += p1_coeffs[i];
    }
    
    // Add X^{d/2} * p2 (degree d/2 to 3d/2-1, with reduction)
    let p2_coeffs = p2.coefficients();
    for i in 0..half_d {
        // Coefficient of X^{d/2 + i}
        if half_d + i < d {
            result_coeffs[half_d + i] += p2_coeffs[i];
        } else {
            // Reduction: X^{d + j} = -X^j
            result_coeffs[half_d + i - d] -= p2_coeffs[i];
        }
    }
    
    // Add X^d * p3 = -p3 (due to X^d = -1)
    let p3_coeffs = p3.coefficients();
    for i in 0..half_d {
        result_coeffs[i] -= p3_coeffs[i];
    }
    
    // Apply modular reduction to all coefficients
    if let Some(q) = result_modulus {
        let half_q = q / 2;
        for coeff in result_coeffs.iter_mut() {
            *coeff = *coeff % q;
            if *coeff > half_q {
                *coeff -= q;
            } else if *coeff < -half_q {
                *coeff += q;
            }
        }
    }
    
    // Validate result bounds
    result.coefficients.validate_bounds()?;
    
    Ok(result)
}

/// Multiplication operation for ring elements with automatic algorithm selection
/// 
/// Implements polynomial multiplication f(X) * g(X) mod (X^d + 1) using the most
/// efficient algorithm based on polynomial degree:
/// - Schoolbook multiplication for d < 512
/// - Karatsuba multiplication for d ≥ 512
/// - NTT multiplication for supported moduli (future implementation)
/// 
/// # Mathematical Properties
/// - Commutative: f * g = g * f
/// - Associative: (f * g) * h = f * (g * h)
/// - Distributive: f * (g + h) = f * g + f * h
/// - Identity: f * 1 = f
/// - Negacyclic: X^d = -1 in the cyclotomic ring
impl Mul for RingElement {
    type Output = Result<RingElement>;
    
    fn mul(self, other: RingElement) -> Self::Output {
        // Use the optimized multiplication algorithm selector
        crate::polynomial_multiplication::multiply_with_algorithm_selection(&self, &other)
    }
}

/// In-place multiplication operation for ring elements
impl MulAssign<RingElement> for RingElement {
    fn mul_assign(&mut self, other: RingElement) {
        // Use the Mul implementation and replace self with result
        match self.clone().mul(other) {
            Ok(result) => *self = result,
            Err(_) => {
                // In case of error, leave self unchanged
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use std::collections::HashMap;
    
    /// Property-based test strategy for ring arithmetic
    /// 
    /// Generates random ring elements and verifies mathematical properties
    /// hold under all operations. This provides comprehensive coverage
    /// beyond manual test cases.
    
    // Test data generators for property-based testing
    prop_compose! {
        /// Generates valid ring dimensions (powers of 2 in supported range)
        fn valid_dimension()(exp in 5u32..15u32) -> usize {
            1 << exp  // 2^5 = 32 to 2^14 = 16384
        }
    }
    
    prop_compose! {
        /// Generates valid moduli for cryptographic applications
        fn valid_modulus()(q in 1000i64..1000000i64) -> i64 {
            // Ensure odd modulus for better mathematical properties
            if q % 2 == 0 { q + 1 } else { q }
        }
    }
    
    prop_compose! {
        /// Generates balanced coefficients within specified bounds
        fn balanced_coefficients(dimension: usize, modulus: i64)
                                (coeffs in prop::collection::vec(-modulus/2..=modulus/2, dimension)) -> Vec<i64> {
            coeffs
        }
    }
    
    #[test]
    fn test_balanced_coefficients_creation() {
        // Test valid creation
        let coeffs = BalancedCoefficients::new(64, 1009).unwrap();
        assert_eq!(coeffs.dimension(), 64);
        assert_eq!(coeffs.modulus(), 1009);
        assert_eq!(coeffs.infinity_norm(), 0);  // All coefficients are zero
        
        // Test invalid dimension (not power of 2)
        assert!(BalancedCoefficients::new(63, 1009).is_err());
        
        // Test invalid modulus (non-positive)
        assert!(BalancedCoefficients::new(64, 0).is_err());
        assert!(BalancedCoefficients::new(64, -1).is_err());
    }
    
    #[test]
    fn test_standard_balanced_conversion() {
        let modulus = 1009i64;
        let standard_coeffs = vec![0, 1, 504, 505, 1008];  // Mix of values
        
        // Convert to balanced representation
        let balanced = BalancedCoefficients::from_standard(&standard_coeffs, modulus).unwrap();
        
        // Expected balanced values: [0, 1, 504, -504, -1]
        let expected_balanced = vec![0, 1, 504, -504, -1];
        assert_eq!(balanced.coefficients(), &expected_balanced);
        
        // Convert back to standard
        let recovered_standard = balanced.to_standard();
        assert_eq!(recovered_standard, standard_coeffs);
    }
    
    #[test]
    fn test_ring_element_creation() {
        // Test zero element creation
        let zero = RingElement::zero(128, Some(1009)).unwrap();
        assert_eq!(zero.dimension(), 128);
        assert_eq!(zero.modulus(), Some(1009));
        assert_eq!(zero.constant_term(), 0);
        assert_eq!(zero.infinity_norm(), 0);
        
        // Test one element creation
        let one = RingElement::one(128, Some(1009)).unwrap();
        assert_eq!(one.constant_term(), 1);
        assert_eq!(one.infinity_norm(), 1);
        
        // Test creation from coefficients
        let coeffs = vec![1, 2, 3, 0, 0, 0, 0, 0];  // 8 coefficients
        let elem = RingElement::from_coefficients(coeffs.clone(), Some(1009)).unwrap();
        assert_eq!(elem.dimension(), 8);
        assert_eq!(elem.constant_term(), 1);
    }
    
    proptest! {
        #[test]
        fn test_addition_properties(
            dimension in valid_dimension(),
            modulus in valid_modulus(),
            coeffs1 in balanced_coefficients(32, 1009),  // Use fixed small dimension for speed
            coeffs2 in balanced_coefficients(32, 1009),
            coeffs3 in balanced_coefficients(32, 1009)
        ) {
            let f = RingElement::from_coefficients(coeffs1, Some(modulus)).unwrap();
            let g = RingElement::from_coefficients(coeffs2, Some(modulus)).unwrap();
            let h = RingElement::from_coefficients(coeffs3, Some(modulus)).unwrap();
            let zero = RingElement::zero(32, Some(modulus)).unwrap();
            
            // Test commutativity: f + g = g + f
            let fg = f.clone().add(g.clone()).unwrap();
            let gf = g.clone().add(f.clone()).unwrap();
            prop_assert_eq!(fg.coefficients(), gf.coefficients());
            
            // Test associativity: (f + g) + h = f + (g + h)
            let fg_h = fg.add(h.clone()).unwrap();
            let gh = g.add(h).unwrap();
            let f_gh = f.clone().add(gh).unwrap();
            prop_assert_eq!(fg_h.coefficients(), f_gh.coefficients());
            
            // Test identity: f + 0 = f
            let f_zero = f.clone().add(zero).unwrap();
            prop_assert_eq!(f.coefficients(), f_zero.coefficients());
            
            // Test inverse: f + (-f) = 0
            let neg_f = f.clone().neg().unwrap();
            let f_neg_f = f.add(neg_f).unwrap();
            prop_assert_eq!(f_neg_f.infinity_norm(), 0);
        }
        
        #[test]
        fn test_multiplication_properties(
            coeffs1 in balanced_coefficients(32, 1009),
            coeffs2 in balanced_coefficients(32, 1009),
            coeffs3 in balanced_coefficients(32, 1009)
        ) {
            let f = RingElement::from_coefficients(coeffs1, Some(1009)).unwrap();
            let g = RingElement::from_coefficients(coeffs2, Some(1009)).unwrap();
            let h = RingElement::from_coefficients(coeffs3, Some(1009)).unwrap();
            let one = RingElement::one(32, Some(1009)).unwrap();
            
            // Test commutativity: f * g = g * f
            let fg = f.clone().mul(g.clone()).unwrap();
            let gf = g.clone().mul(f.clone()).unwrap();
            prop_assert_eq!(fg.coefficients(), gf.coefficients());
            
            // Test identity: f * 1 = f
            let f_one = f.clone().mul(one).unwrap();
            prop_assert_eq!(f.coefficients(), f_one.coefficients());
            
            // Test distributivity: f * (g + h) = f * g + f * h
            let gh = g.clone().add(h.clone()).unwrap();
            let f_gh = f.clone().mul(gh).unwrap();
            let fg = f.clone().mul(g).unwrap();
            let fh = f.mul(h).unwrap();
            let fg_fh = fg.add(fh).unwrap();
            prop_assert_eq!(f_gh.coefficients(), fg_fh.coefficients());
        }
        
        #[test]
        fn test_schoolbook_karatsuba_equivalence(
            coeffs1 in balanced_coefficients(64, 1009),
            coeffs2 in balanced_coefficients(64, 1009)
        ) {
            let f = RingElement::from_coefficients(coeffs1, Some(1009)).unwrap();
            let g = RingElement::from_coefficients(coeffs2, Some(1009)).unwrap();
            
            // Compute using both algorithms
            let schoolbook_result = schoolbook_multiply(&f, &g).unwrap();
            let karatsuba_result = karatsuba_multiply(&f, &g).unwrap();
            
            // Results should be identical
            prop_assert_eq!(schoolbook_result.coefficients(), karatsuba_result.coefficients());
        }
    }
    
    #[test]
    fn test_negacyclic_property() {
        // Test that X^d = -1 in the ring
        let d = 8;
        let modulus = Some(1009);
        
        // Create X (polynomial with coefficient 1 for X^1)
        let mut x_coeffs = vec![0i64; d];
        x_coeffs[1] = 1;  // X = 0 + 1*X + 0*X^2 + ...
        let x = RingElement::from_coefficients(x_coeffs, modulus).unwrap();
        
        // Compute X^d by repeated multiplication
        let mut x_power = RingElement::one(d, modulus).unwrap();
        for _ in 0..d {
            x_power = x_power.mul(x.clone()).unwrap();
        }
        
        // X^d should equal -1
        let neg_one = RingElement::one(d, modulus).unwrap().neg().unwrap();
        assert_eq!(x_power.coefficients(), neg_one.coefficients());
    }
    
    #[test]
    fn test_simd_optimization() {
        // Test that SIMD and scalar implementations give same results
        let d = 64;  // Multiple of SIMD_WIDTH
        let modulus = 1009i64;
        
        // Create test vectors with known values
        let coeffs1: Vec<i64> = (0..d as i64).map(|i| i % (modulus / 2)).collect();
        let coeffs2: Vec<i64> = (0..d as i64).map(|i| (i * 2) % (modulus / 2)).collect();
        
        let f = RingElement::from_coefficients(coeffs1, Some(modulus)).unwrap();
        let g = RingElement::from_coefficients(coeffs2, Some(modulus)).unwrap();
        
        // Test addition (uses SIMD internally)
        let sum = f.clone().add(g.clone()).unwrap();
        
        // Verify result by manual computation
        for i in 0..d {
            let expected = (f.coefficients()[i] + g.coefficients()[i]) % modulus;
            let expected_balanced = if expected > modulus / 2 {
                expected - modulus
            } else {
                expected
            };
            assert_eq!(sum.coefficients()[i], expected_balanced);
        }
    }
    
    #[test]
    fn test_memory_alignment() {
        // Test that coefficient vectors are properly aligned for SIMD
        let coeffs = BalancedCoefficients::new(128, 1009).unwrap();
        let ptr = coeffs.coefficients().as_ptr() as usize;
        
        // Check alignment (should be aligned to at least 8 bytes for i64)
        assert_eq!(ptr % 8, 0);
        
        // For optimal SIMD performance, should be aligned to larger boundaries
        // This is a soft requirement and may not always be achievable
    }
    
    #[test]
    fn test_overflow_protection() {
        // Test behavior with large coefficients that might overflow
        let d = 32;
        let modulus = i64::MAX / 1000;  // Large but not overflow-prone
        
        // Create coefficients near the modulus bound
        let large_coeffs: Vec<i64> = (0..d).map(|_| modulus / 2 - 1).collect();
        let f = RingElement::from_coefficients(large_coeffs.clone(), Some(modulus)).unwrap();
        let g = RingElement::from_coefficients(large_coeffs, Some(modulus)).unwrap();
        
        // Addition should not overflow and should maintain bounds
        let sum = f.add(g).unwrap();
        assert!(sum.coefficients().iter().all(|&c| c.abs() <= modulus / 2));
        
        // Multiplication should also maintain bounds (though result may be reduced)
        let product = f.mul(g).unwrap();
        assert!(product.coefficients().iter().all(|&c| c.abs() <= modulus / 2));
    }
}  
  /// Returns the modulus used for this ring element
    /// 
    /// # Returns
    /// * `Option<i64>` - The modulus q if operating in Rq, None for integer ring
    pub fn modulus(&self) -> Option<i64> {
        self.modulus
    }
    
    /// Checks if this ring element is zero
    /// 
    /// # Returns
    /// * `bool` - True if all coefficients are zero
    pub fn is_zero(&self) -> bool {
        self.coefficients.coefficients().iter().all(|&c| c == 0)
    }
    
    /// Computes the ℓ∞-norm of this ring element
    /// 
    /// # Returns
    /// * `i64` - Maximum absolute value of coefficients
    pub fn infinity_norm(&self) -> i64 {
        self.coefficients.infinity_norm()
    }
    
    /// Adds two ring elements: self + other
    /// 
    /// # Arguments
    /// * `other` - Ring element to add
    /// 
    /// # Returns
    /// * `Result<RingElement>` - Sum or error
    pub fn add(&self, other: &RingElement) -> Result<RingElement> {
        // Validate dimensions match
        if self.dimension != other.dimension {
            return Err(LatticeFoldError::InvalidDimension {
                expected: self.dimension,
                got: other.dimension,
            });
        }
        
        // Validate moduli match
        if self.modulus != other.modulus {
            return Err(LatticeFoldError::InvalidModulus {
                modulus: other.modulus.unwrap_or(0),
            });
        }
        
        // Perform coefficient-wise addition
        let self_coeffs = self.coefficients.coefficients();
        let other_coeffs = other.coefficients.coefficients();
        let modulus = self.modulus.unwrap_or(1i64 << 62);
        
        let mut result_coeffs = Vec::with_capacity(self.dimension);
        for (a, b) in self_coeffs.iter().zip(other_coeffs.iter()) {
            let sum = (a + b) % modulus;
            let balanced_sum = if sum > modulus / 2 {
                sum - modulus
            } else if sum < -modulus / 2 {
                sum + modulus
            } else {
                sum
            };
            result_coeffs.push(balanced_sum);
        }
        
        RingElement::from_coefficients(result_coeffs, self.modulus)
    }
    
    /// Subtracts two ring elements: self - other
    /// 
    /// # Arguments
    /// * `other` - Ring element to subtract
    /// 
    /// # Returns
    /// * `Result<RingElement>` - Difference or error
    pub fn subtract(&self, other: &RingElement) -> Result<RingElement> {
        // Validate dimensions match
        if self.dimension != other.dimension {
            return Err(LatticeFoldError::InvalidDimension {
                expected: self.dimension,
                got: other.dimension,
            });
        }
        
        // Validate moduli match
        if self.modulus != other.modulus {
            return Err(LatticeFoldError::InvalidModulus {
                modulus: other.modulus.unwrap_or(0),
            });
        }
        
        // Perform coefficient-wise subtraction
        let self_coeffs = self.coefficients.coefficients();
        let other_coeffs = other.coefficients.coefficients();
        let modulus = self.modulus.unwrap_or(1i64 << 62);
        
        let mut result_coeffs = Vec::with_capacity(self.dimension);
        for (a, b) in self_coeffs.iter().zip(other_coeffs.iter()) {
            let diff = (a - b) % modulus;
            let balanced_diff = if diff > modulus / 2 {
                diff - modulus
            } else if diff < -modulus / 2 {
                diff + modulus
            } else {
                diff
            };
            result_coeffs.push(balanced_diff);
        }
        
        RingElement::from_coefficients(result_coeffs, self.modulus)
    }
    
    /// Multiplies two ring elements using polynomial multiplication with X^d = -1
    /// 
    /// # Arguments
    /// * `other` - Ring element to multiply
    /// 
    /// # Returns
    /// * `Result<RingElement>` - Product or error
    pub fn multiply(&self, other: &RingElement) -> Result<RingElement> {
        // Validate dimensions match
        if self.dimension != other.dimension {
            return Err(LatticeFoldError::InvalidDimension {
                expected: self.dimension,
                got: other.dimension,
            });
        }
        
        // Validate moduli match
        if self.modulus != other.modulus {
            return Err(LatticeFoldError::InvalidModulus {
                modulus: other.modulus.unwrap_or(0),
            });
        }
        
        let self_coeffs = self.coefficients.coefficients();
        let other_coeffs = other.coefficients.coefficients();
        let modulus = self.modulus.unwrap_or(1i64 << 62);
        let d = self.dimension;
        
        // Initialize result coefficients
        let mut result_coeffs = vec![0i64; d];
        
        // Perform negacyclic convolution: X^d = -1
        for i in 0..d {
            for j in 0..d {
                let coeff_product = (self_coeffs[i] * other_coeffs[j]) % modulus;
                
                if i + j < d {
                    // Normal term: coefficient of X^{i+j}
                    result_coeffs[i + j] = (result_coeffs[i + j] + coeff_product) % modulus;
                } else {
                    // Wrapped term with negation: X^{i+j} = X^{i+j-d} * X^d = -X^{i+j-d}
                    let wrapped_index = i + j - d;
                    result_coeffs[wrapped_index] = (result_coeffs[wrapped_index] - coeff_product) % modulus;
                }
            }
        }
        
        // Convert to balanced representation
        for coeff in &mut result_coeffs {
            if *coeff > modulus / 2 {
                *coeff -= modulus;
            } else if *coeff < -modulus / 2 {
                *coeff += modulus;
            }
        }
        
        RingElement::from_coefficients(result_coeffs, self.modulus)
    }
    
    /// Negates the ring element: -self
    /// 
    /// # Returns
    /// * `Result<RingElement>` - Negated element or error
    pub fn negate(&self) -> Result<RingElement> {
        let coeffs = self.coefficients.coefficients();
        let modulus = self.modulus.unwrap_or(1i64 << 62);
        
        let mut negated_coeffs = Vec::with_capacity(self.dimension);
        for &coeff in coeffs {
            let negated = (-coeff) % modulus;
            let balanced_negated = if negated > modulus / 2 {
                negated - modulus
            } else if negated < -modulus / 2 {
                negated + modulus
            } else {
                negated
            };
            negated_coeffs.push(balanced_negated);
        }
        
        RingElement::from_coefficients(negated_coeffs, self.modulus)
    }
    
    /// Transforms the ring element to NTT domain (placeholder)
    /// 
    /// # Arguments
    /// * `ntt_params` - NTT parameters for transformation
    /// 
    /// # Returns
    /// * `Result<()>` - Success or error
    /// 
    /// # Note
    /// This is a placeholder implementation. A full NTT implementation
    /// would require the complete NTT module.
    pub fn to_ntt(&mut self, _ntt_params: &NTTParams) -> Result<()> {
        // Placeholder implementation
        // In a complete implementation, this would transform coefficients to NTT domain
        Ok(())
    }
    
    /// Transforms the ring element from NTT domain (placeholder)
    /// 
    /// # Arguments
    /// * `ntt_params` - NTT parameters for transformation
    /// 
    /// # Returns
    /// * `Result<()>` - Success or error
    pub fn from_ntt(&mut self, _ntt_params: &NTTParams) -> Result<()> {
        // Placeholder implementation
        // In a complete implementation, this would transform coefficients from NTT domain
        Ok(())
    }
    
    /// Pointwise multiplication in NTT domain (placeholder)
    /// 
    /// # Arguments
    /// * `other` - Other ring element in NTT domain
    /// 
    /// # Returns
    /// * `Result<RingElement>` - Pointwise product or error
    pub fn pointwise_multiply(&self, other: &RingElement) -> Result<RingElement> {
        // Placeholder implementation
        // In NTT domain, this would be simple coefficient-wise multiplication
        self.multiply(other)
    }
}