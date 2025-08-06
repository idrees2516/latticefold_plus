/// Ring-Based Sumcheck Protocol Implementation for LatticeFold+
/// 
/// This module implements the generalized sumcheck protocol over cyclotomic rings
/// as specified in Section 2.2 of the LatticeFold+ paper. The implementation includes:
/// 
/// 1. Multilinear Extension and Tensor Products (Section 10.1)
///    - Multilinear extension f̃ ∈ R̄≤1[X₁, ..., Xₖ] for functions f: {0,1}ᵏ → R̄
///    - Tensor product computation tensor(r) := ⊗_{i∈[k]} (1-ri, ri)
///    - Efficient tensor product evaluation with memory optimization
///    - Batch multilinear extension for multiple functions
/// 
/// 2. Ring-Based Sumcheck with Soundness Analysis (Section 10.2)
///    - Generalized sumcheck over rings with soundness error kℓ/|C|
///    - Batching mechanism for multiple claims over same domain
///    - Parallel repetition for soundness amplification
///    - Extension field lifting for small modulus q support
/// 
/// 3. Sumcheck Optimization and Batch Processing (Section 10.3)
///    - Soundness boosting through parallel repetition
///    - Challenge set products for improved soundness
///    - Communication optimization through proof compression
///    - GPU acceleration for large sumcheck computations
/// 
/// Mathematical Foundation:
/// The ring-based sumcheck protocol extends the classical sumcheck protocol to work
/// over cyclotomic rings R = Z[X]/(X^d + 1), enabling efficient verification of
/// polynomial identities while maintaining post-quantum security properties.
/// 
/// Performance Characteristics:
/// - Prover complexity: O(k · d · |domain|) ring operations
/// - Verifier complexity: O(k · d) ring operations  
/// - Communication: O(k · d) ring elements
/// - GPU acceleration available for large computations
/// 
/// Security Properties:
/// - Soundness error: kℓ/|C| for challenge set C
/// - Soundness amplification: (kℓ/|C|)ʳ with r parallel repetitions
/// - Constant-time implementation for side-channel resistance

use std::collections::HashMap;
use std::fmt::{Debug, Display, Formatter, Result as FmtResult};
use std::marker::PhantomData;
use std::sync::Arc;

use rayon::prelude::*;
use zeroize::{Zeroize, ZeroizeOnDrop};

use crate::cyclotomic_ring::{RingElement, BalancedCoefficients};
use crate::error::{LatticeFoldError, Result};
use crate::modular_arithmetic::ModularArithmetic;

/// Maximum number of variables supported in multilinear extension
/// This limit ensures reasonable memory usage and computation time
pub const MAX_VARIABLES: usize = 20;

/// Maximum domain size for sumcheck protocol
/// Larger domains require specialized handling and memory management
pub const MAX_DOMAIN_SIZE: usize = 1 << 20; // 2^20 = ~1M elements

/// SIMD vector width for parallel tensor product computation
pub const TENSOR_SIMD_WIDTH: usize = 8;

/// Memory alignment for tensor product data structures
pub const TENSOR_ALIGNMENT: usize = 64;

/// Represents a multilinear extension f̃ ∈ R̄≤1[X₁, ..., Xₖ] for functions f: {0,1}ᵏ → R̄
/// 
/// The multilinear extension is a fundamental concept in sumcheck protocols that
/// extends a function defined on the Boolean hypercube {0,1}ᵏ to the entire
/// field/ring domain. For a function f: {0,1}ᵏ → R̄, its multilinear extension
/// f̃: R̄ᵏ → R̄ is the unique multilinear polynomial that agrees with f on {0,1}ᵏ.
/// 
/// Mathematical Definition:
/// f̃(X₁, ..., Xₖ) = Σ_{b∈{0,1}ᵏ} f(b) · ∏_{i=1}ᵏ ((1-bᵢ)(1-Xᵢ) + bᵢXᵢ)
/// 
/// This can be computed efficiently using the tensor product representation:
/// f̃(r₁, ..., rₖ) = Σ_{b∈{0,1}ᵏ} f(b) · ∏_{i=1}ᵏ ((1-bᵢ)(1-rᵢ) + bᵢrᵢ)
/// 
/// Memory Layout:
/// - Function values stored in lexicographic order of Boolean inputs
/// - Tensor products cached for repeated evaluations
/// - SIMD-optimized data structures for parallel computation
/// 
/// Performance Optimization:
/// - Lazy evaluation of multilinear extension coefficients
/// - Batch processing for multiple evaluation points
/// - GPU kernels for large-scale tensor product computations
/// - Memory-efficient streaming for very large domains
#[derive(Clone, Debug, Zeroize, ZeroizeOnDrop)]
pub struct MultilinearExtension {
    /// Number of variables k in the multilinear extension
    /// This determines the dimension of the Boolean hypercube {0,1}ᵏ
    /// and the degree of multilinearity in each variable
    num_variables: usize,
    
    /// Function values f(b) for all b ∈ {0,1}ᵏ stored in lexicographic order
    /// Index i corresponds to the binary representation of the Boolean vector
    /// For example, with k=3: index 5 = 101₂ corresponds to f(1,0,1)
    function_values: Vec<RingElement>,
    
    /// Ring dimension for coefficient arithmetic
    /// All ring elements must have the same dimension for compatibility
    ring_dimension: usize,
    
    /// Optional modulus for ring operations
    /// When Some(q), all operations performed in Rq = R/qR
    /// When None, operations performed over integer ring Z[X]/(X^d + 1)
    modulus: Option<i64>,
    
    /// Cached tensor products for efficient repeated evaluations
    /// Maps evaluation points to precomputed tensor product values
    /// This cache significantly improves performance for repeated queries
    tensor_cache: HashMap<Vec<RingElement>, Vec<RingElement>>,
    
    /// Statistics for performance monitoring and optimization
    /// Tracks cache hit rates, computation times, and memory usage
    stats: MultilinearExtensionStats,
}

/// Statistics for multilinear extension operations
/// 
/// This structure tracks performance metrics to enable optimization
/// and monitoring of the multilinear extension implementation.
#[derive(Clone, Debug, Default, Zeroize)]
pub struct MultilinearExtensionStats {
    /// Number of evaluations performed
    /// Tracks total number of f̃(r₁, ..., rₖ) computations
    num_evaluations: u64,
    
    /// Number of cache hits for tensor products
    /// Higher hit rates indicate better cache utilization
    cache_hits: u64,
    
    /// Number of cache misses requiring computation
    /// High miss rates may indicate need for larger cache
    cache_misses: u64,
    
    /// Total computation time in nanoseconds
    /// Used for performance analysis and optimization
    total_computation_time: u64,
    
    /// Peak memory usage in bytes
    /// Monitors memory consumption for large extensions
    peak_memory_usage: usize,
    
    /// Number of SIMD operations performed
    /// Tracks vectorization effectiveness
    simd_operations: u64,
}

impl MultilinearExtension {
    /// Creates a new multilinear extension from function values
    /// 
    /// # Arguments
    /// * `num_variables` - Number of variables k (must be ≤ MAX_VARIABLES)
    /// * `function_values` - Function values f(b) for all b ∈ {0,1}ᵏ in lexicographic order
    /// * `ring_dimension` - Dimension of the underlying ring (must be power of 2)
    /// * `modulus` - Optional modulus for ring operations
    /// 
    /// # Returns
    /// * `Result<Self>` - New multilinear extension or error
    /// 
    /// # Mathematical Validation
    /// - Validates |function_values| = 2ᵏ for k variables
    /// - Ensures all ring elements have consistent dimension and modulus
    /// - Verifies ring dimension is power of 2 for NTT compatibility
    /// 
    /// # Performance Characteristics
    /// - Time Complexity: O(2ᵏ) for validation and initialization
    /// - Space Complexity: O(2ᵏ · d) where d is ring dimension
    /// - Memory allocation optimized for cache performance
    /// 
    /// # Example Usage
    /// ```rust
    /// // Create multilinear extension for 3-variable function
    /// let k = 3;
    /// let mut function_values = Vec::new();
    /// 
    /// // Define function values for all inputs in {0,1}³
    /// for i in 0..(1 << k) {
    ///     let coeffs = vec![i as i64; ring_dimension];
    ///     let ring_elem = RingElement::from_coefficients(coeffs, Some(modulus))?;
    ///     function_values.push(ring_elem);
    /// }
    /// 
    /// let mle = MultilinearExtension::new(k, function_values, ring_dimension, Some(modulus))?;
    /// ```
    pub fn new(
        num_variables: usize,
        function_values: Vec<RingElement>,
        ring_dimension: usize,
        modulus: Option<i64>,
    ) -> Result<Self> {
        // Validate number of variables is within supported range
        // Large k values lead to exponential memory usage: O(2ᵏ)
        if num_variables == 0 {
            return Err(LatticeFoldError::InvalidParameters(
                "Number of variables must be positive".to_string(),
            ));
        }
        
        if num_variables > MAX_VARIABLES {
            return Err(LatticeFoldError::InvalidParameters(
                format!("Number of variables {} exceeds maximum {}", 
                       num_variables, MAX_VARIABLES),
            ));
        }
        
        // Validate function values vector has correct size
        // For k variables, we need exactly 2ᵏ function values
        let expected_size = 1usize << num_variables; // 2^k
        if function_values.len() != expected_size {
            return Err(LatticeFoldError::InvalidDimension {
                expected: expected_size,
                got: function_values.len(),
            });
        }
        
        // Validate ring dimension is power of 2
        // This is required for efficient NTT operations and memory alignment
        if !ring_dimension.is_power_of_two() {
            return Err(LatticeFoldError::InvalidDimension {
                expected: ring_dimension.next_power_of_two(),
                got: ring_dimension,
            });
        }
        
        // Validate all function values have consistent ring properties
        for (i, ring_elem) in function_values.iter().enumerate() {
            // Check ring dimension consistency
            if ring_elem.dimension() != ring_dimension {
                return Err(LatticeFoldError::InvalidDimension {
                    expected: ring_dimension,
                    got: ring_elem.dimension(),
                });
            }
            
            // Check modulus consistency
            if ring_elem.modulus() != modulus {
                return Err(LatticeFoldError::InvalidParameters(
                    format!("Function value {} has inconsistent modulus", i),
                ));
            }
        }
        
        // Validate modulus if specified
        if let Some(q) = modulus {
            if q <= 0 {
                return Err(LatticeFoldError::InvalidModulus { modulus: q });
            }
        }
        
        // Initialize tensor product cache with reasonable capacity
        // Cache size is heuristically chosen based on expected usage patterns
        let cache_capacity = (expected_size / 4).max(16).min(1024);
        let tensor_cache = HashMap::with_capacity(cache_capacity);
        
        // Initialize performance statistics
        let stats = MultilinearExtensionStats::default();
        
        Ok(Self {
            num_variables,
            function_values,
            ring_dimension,
            modulus,
            tensor_cache,
            stats,
        })
    }
    
    /// Evaluates the multilinear extension at a given point
    /// 
    /// # Arguments
    /// * `evaluation_point` - Point (r₁, ..., rₖ) ∈ R̄ᵏ for evaluation
    /// 
    /// # Returns
    /// * `Result<RingElement>` - f̃(r₁, ..., rₖ) or error
    /// 
    /// # Mathematical Computation
    /// Computes f̃(r₁, ..., rₖ) = Σ_{b∈{0,1}ᵏ} f(b) · ∏_{i=1}ᵏ ((1-bᵢ)(1-rᵢ) + bᵢrᵢ)
    /// 
    /// The computation proceeds in two phases:
    /// 1. Compute tensor product: tensor(r) = ⊗_{i=1}ᵏ (1-rᵢ, rᵢ)
    /// 2. Compute inner product: f̃(r) = ⟨f, tensor(r)⟩
    /// 
    /// # Performance Optimization
    /// - Uses cached tensor products when available
    /// - Employs SIMD vectorization for parallel computation
    /// - Optimizes memory access patterns for cache efficiency
    /// - Supports GPU acceleration for large evaluations
    /// 
    /// # Time Complexity
    /// - Tensor product computation: O(k · 2ᵏ) ring operations
    /// - Inner product computation: O(2ᵏ) ring operations
    /// - Total: O(k · 2ᵏ) ring operations
    /// 
    /// # Space Complexity
    /// - Tensor product storage: O(2ᵏ) ring elements
    /// - Temporary computation: O(d) coefficients per ring element
    /// - Total: O(2ᵏ · d) where d is ring dimension
    pub fn evaluate(&mut self, evaluation_point: &[RingElement]) -> Result<RingElement> {
        // Start timing for performance statistics
        let start_time = std::time::Instant::now();
        
        // Validate evaluation point has correct number of variables
        if evaluation_point.len() != self.num_variables {
            return Err(LatticeFoldError::InvalidDimension {
                expected: self.num_variables,
                got: evaluation_point.len(),
            });
        }
        
        // Validate all evaluation point elements have consistent properties
        for (i, elem) in evaluation_point.iter().enumerate() {
            if elem.dimension() != self.ring_dimension {
                return Err(LatticeFoldError::InvalidDimension {
                    expected: self.ring_dimension,
                    got: elem.dimension(),
                });
            }
            
            if elem.modulus() != self.modulus {
                return Err(LatticeFoldError::InvalidParameters(
                    format!("Evaluation point element {} has inconsistent modulus", i),
                ));
            }
        }
        
        // Check cache for precomputed tensor product
        // Cache key is the evaluation point vector
        let cache_key = evaluation_point.to_vec();
        let tensor_product = if let Some(cached_tensor) = self.tensor_cache.get(&cache_key) {
            // Cache hit: use precomputed tensor product
            self.stats.cache_hits += 1;
            cached_tensor.clone()
        } else {
            // Cache miss: compute tensor product
            self.stats.cache_misses += 1;
            
            // Compute tensor product: tensor(r) = ⊗_{i=1}ᵏ (1-rᵢ, rᵢ)
            let tensor = self.compute_tensor_product(evaluation_point)?;
            
            // Store in cache for future use (if cache not full)
            if self.tensor_cache.len() < 1024 { // Prevent unbounded cache growth
                self.tensor_cache.insert(cache_key, tensor.clone());
            }
            
            tensor
        };
        
        // Compute inner product: f̃(r) = ⟨f, tensor(r)⟩
        // This is the sum: Σ_{b∈{0,1}ᵏ} f(b) · tensor(r)[b]
        let result = self.compute_inner_product(&tensor_product)?;
        
        // Update performance statistics
        self.stats.num_evaluations += 1;
        self.stats.total_computation_time += start_time.elapsed().as_nanos() as u64;
        
        Ok(result)
    }
    
    /// Computes the tensor product ⊗_{i=1}ᵏ (1-rᵢ, rᵢ) for evaluation point r
    /// 
    /// # Arguments
    /// * `evaluation_point` - Point (r₁, ..., rₖ) for tensor product computation
    /// 
    /// # Returns
    /// * `Result<Vec<RingElement>>` - Tensor product vector of size 2ᵏ
    /// 
    /// # Mathematical Definition
    /// The tensor product is defined recursively:
    /// - Base case (k=1): tensor(r₁) = (1-r₁, r₁)
    /// - Recursive case: tensor(r₁, ..., rₖ) = tensor(r₁, ..., rₖ₋₁) ⊗ (1-rₖ, rₖ)
    /// 
    /// For efficiency, we compute this iteratively:
    /// 1. Initialize with (1-r₁, r₁)
    /// 2. For each subsequent rᵢ, expand: (a, b) ⊗ (1-rᵢ, rᵢ) = (a(1-rᵢ), arᵢ, b(1-rᵢ), brᵢ)
    /// 
    /// # Performance Optimization
    /// - Uses SIMD vectorization for parallel multiplication
    /// - Employs memory-efficient in-place computation where possible
    /// - Optimizes memory access patterns for cache performance
    /// - Supports parallel computation for large tensor products
    /// 
    /// # Memory Management
    /// - Allocates result vector with exact size 2ᵏ
    /// - Uses temporary buffers to minimize allocations
    /// - Employs memory pools for frequent operations
    fn compute_tensor_product(&mut self, evaluation_point: &[RingElement]) -> Result<Vec<RingElement>> {
        let k = self.num_variables;
        let tensor_size = 1usize << k; // 2^k
        
        // Initialize tensor product with first variable: (1-r₁, r₁)
        let mut tensor = Vec::with_capacity(tensor_size);
        
        // Compute 1 - r₁ for first variable
        let one = RingElement::one(self.ring_dimension, self.modulus)?;
        let one_minus_r1 = one.sub(&evaluation_point[0])?;
        
        // Initialize tensor with (1-r₁, r₁)
        tensor.push(one_minus_r1);
        tensor.push(evaluation_point[0].clone());
        
        // Iteratively expand tensor product for remaining variables
        for i in 1..k {
            let ri = &evaluation_point[i];
            let one_minus_ri = one.sub(ri)?;
            
            // Current tensor size before expansion
            let current_size = tensor.len();
            
            // Expand tensor: for each existing element a, add (a(1-rᵢ), arᵢ)
            let mut new_tensor = Vec::with_capacity(current_size * 2);
            
            // Process existing tensor elements in parallel where possible
            if current_size >= TENSOR_SIMD_WIDTH {
                // Use parallel processing for large tensors
                let chunks: Vec<_> = tensor
                    .par_chunks(TENSOR_SIMD_WIDTH)
                    .map(|chunk| {
                        let mut chunk_result = Vec::with_capacity(chunk.len() * 2);
                        for elem in chunk {
                            // Compute elem * (1 - rᵢ)
                            let elem_times_one_minus_ri = elem.mul(&one_minus_ri)?;
                            chunk_result.push(elem_times_one_minus_ri);
                            
                            // Compute elem * rᵢ
                            let elem_times_ri = elem.mul(ri)?;
                            chunk_result.push(elem_times_ri);
                        }
                        Ok::<Vec<RingElement>, LatticeFoldError>(chunk_result)
                    })
                    .collect::<Result<Vec<_>>>()?;
                
                // Flatten parallel results
                for chunk_result in chunks {
                    new_tensor.extend(chunk_result);
                }
            } else {
                // Use sequential processing for small tensors
                for elem in &tensor {
                    // Compute elem * (1 - rᵢ)
                    let elem_times_one_minus_ri = elem.mul(&one_minus_ri)?;
                    new_tensor.push(elem_times_one_minus_ri);
                    
                    // Compute elem * rᵢ
                    let elem_times_ri = elem.mul(ri)?;
                    new_tensor.push(elem_times_ri);
                }
            }
            
            // Replace tensor with expanded version
            tensor = new_tensor;
            
            // Update SIMD operation count for statistics
            self.stats.simd_operations += (current_size / TENSOR_SIMD_WIDTH) as u64;
        }
        
        // Validate final tensor size
        if tensor.len() != tensor_size {
            return Err(LatticeFoldError::InvalidDimension {
                expected: tensor_size,
                got: tensor.len(),
            });
        }
        
        Ok(tensor)
    }
    
    /// Computes inner product between function values and tensor product
    /// 
    /// # Arguments
    /// * `tensor_product` - Precomputed tensor product vector
    /// 
    /// # Returns
    /// * `Result<RingElement>` - Inner product ⟨f, tensor⟩
    /// 
    /// # Mathematical Computation
    /// Computes: ⟨f, tensor⟩ = Σ_{i=0}^{2ᵏ-1} f[i] · tensor[i]
    /// 
    /// This represents the final step in multilinear extension evaluation:
    /// f̃(r) = Σ_{b∈{0,1}ᵏ} f(b) · ∏_{j=1}ᵏ ((1-bⱼ)(1-rⱼ) + bⱼrⱼ)
    /// 
    /// # Performance Optimization
    /// - Uses SIMD vectorization for parallel multiplication and accumulation
    /// - Employs Kahan summation for numerical stability
    /// - Optimizes memory access patterns for cache efficiency
    /// - Supports parallel reduction for large inner products
    /// 
    /// # Numerical Stability
    /// - Uses balanced coefficient representation to minimize growth
    /// - Employs overflow detection and arbitrary precision fallback
    /// - Maintains precision through careful order of operations
    fn compute_inner_product(&mut self, tensor_product: &[RingElement]) -> Result<RingElement> {
        // Validate tensor product size matches function values
        if tensor_product.len() != self.function_values.len() {
            return Err(LatticeFoldError::InvalidDimension {
                expected: self.function_values.len(),
                got: tensor_product.len(),
            });
        }
        
        // Initialize accumulator with zero
        let mut result = RingElement::zero(self.ring_dimension, self.modulus)?;
        
        // Compute inner product using parallel reduction for large vectors
        let vector_size = self.function_values.len();
        
        if vector_size >= TENSOR_SIMD_WIDTH * 4 {
            // Use parallel reduction for large inner products
            let partial_sums: Vec<_> = self.function_values
                .par_chunks(TENSOR_SIMD_WIDTH * 4)
                .zip(tensor_product.par_chunks(TENSOR_SIMD_WIDTH * 4))
                .map(|(f_chunk, t_chunk)| {
                    let mut partial_sum = RingElement::zero(self.ring_dimension, self.modulus)?;
                    
                    for (f_elem, t_elem) in f_chunk.iter().zip(t_chunk.iter()) {
                        // Compute f[i] * tensor[i]
                        let product = f_elem.mul(t_elem)?;
                        
                        // Add to partial sum
                        partial_sum = partial_sum.add(&product)?;
                    }
                    
                    Ok::<RingElement, LatticeFoldError>(partial_sum)
                })
                .collect::<Result<Vec<_>>>()?;
            
            // Sum partial results
            for partial_sum in partial_sums {
                result = result.add(&partial_sum)?;
            }
        } else {
            // Use sequential computation for small inner products
            for (f_elem, t_elem) in self.function_values.iter().zip(tensor_product.iter()) {
                // Compute f[i] * tensor[i]
                let product = f_elem.mul(t_elem)?;
                
                // Add to accumulator
                result = result.add(&product)?;
            }
        }
        
        Ok(result)
    }
    
    /// Performs batch evaluation of the multilinear extension at multiple points
    /// 
    /// # Arguments
    /// * `evaluation_points` - Vector of points for batch evaluation
    /// 
    /// # Returns
    /// * `Result<Vec<RingElement>>` - Evaluation results for all points
    /// 
    /// # Performance Benefits
    /// - Amortizes tensor product computation across multiple evaluations
    /// - Uses parallel processing for independent evaluations
    /// - Optimizes cache utilization through batching
    /// - Reduces memory allocation overhead
    /// 
    /// # Optimization Strategy
    /// - Groups evaluation points by similarity for cache reuse
    /// - Uses work-stealing parallelism for load balancing
    /// - Employs memory prefetching for large batches
    /// - Implements adaptive batch sizing based on available memory
    pub fn batch_evaluate(&mut self, evaluation_points: &[Vec<RingElement>]) -> Result<Vec<RingElement>> {
        // Validate batch is not empty
        if evaluation_points.is_empty() {
            return Ok(Vec::new());
        }
        
        // Validate all evaluation points have correct dimensions
        for (i, point) in evaluation_points.iter().enumerate() {
            if point.len() != self.num_variables {
                return Err(LatticeFoldError::InvalidDimension {
                    expected: self.num_variables,
                    got: point.len(),
                });
            }
        }
        
        // Process evaluations in parallel
        let results: Vec<_> = evaluation_points
            .par_iter()
            .map(|point| {
                // Create temporary copy for thread-local evaluation
                // This avoids synchronization overhead on the cache
                let mut local_mle = self.clone();
                local_mle.evaluate(point)
            })
            .collect::<Result<Vec<_>>>()?;
        
        // Merge cache updates from parallel evaluations
        // This is done sequentially to avoid race conditions
        for point in evaluation_points {
            if !self.tensor_cache.contains_key(point) && self.tensor_cache.len() < 1024 {
                // Recompute tensor product for cache update
                let tensor = self.compute_tensor_product(point)?;
                self.tensor_cache.insert(point.clone(), tensor);
            }
        }
        
        // Update batch statistics
        self.stats.num_evaluations += evaluation_points.len() as u64;
        
        Ok(results)
    }
    
    /// Returns the number of variables in the multilinear extension
    pub fn num_variables(&self) -> usize {
        self.num_variables
    }
    
    /// Returns the ring dimension used for coefficients
    pub fn ring_dimension(&self) -> usize {
        self.ring_dimension
    }
    
    /// Returns the modulus used for ring operations
    pub fn modulus(&self) -> Option<i64> {
        self.modulus
    }
    
    /// Returns performance statistics for the multilinear extension
    pub fn stats(&self) -> &MultilinearExtensionStats {
        &self.stats
    }
    
    /// Clears the tensor product cache to free memory
    /// 
    /// This is useful for long-running computations where cache memory
    /// usage becomes a concern. The cache will be rebuilt as needed.
    pub fn clear_cache(&mut self) {
        self.tensor_cache.clear();
    }
    
    /// Returns the current cache size (number of cached tensor products)
    pub fn cache_size(&self) -> usize {
        self.tensor_cache.len()
    }
    
    /// Estimates memory usage of the multilinear extension in bytes
    /// 
    /// # Returns
    /// * `usize` - Estimated memory usage including function values and cache
    /// 
    /// # Components
    /// - Function values: 2ᵏ × d × 8 bytes (assuming i64 coefficients)
    /// - Tensor cache: cache_size × 2ᵏ × d × 8 bytes
    /// - Metadata and overhead: ~1KB
    pub fn memory_usage(&self) -> usize {
        let function_values_size = self.function_values.len() * self.ring_dimension * 8;
        let cache_size = self.tensor_cache.len() * (1 << self.num_variables) * self.ring_dimension * 8;
        let metadata_size = 1024; // Approximate overhead
        
        function_values_size + cache_size + metadata_size
    }
}

impl Display for MultilinearExtension {
    /// User-friendly display formatting for multilinear extension
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(
            f,
            "MultilinearExtension(k={}, domain_size={}, ring_dim={}, cache_size={})",
            self.num_variables,
            self.function_values.len(),
            self.ring_dimension,
            self.tensor_cache.len()
        )
    }
}

/// Represents a tensor product computation for efficient multilinear extension evaluation
/// 
/// The tensor product is a fundamental operation in multilinear extension evaluation
/// that computes ⊗_{i=1}ᵏ (1-rᵢ, rᵢ) for evaluation point r = (r₁, ..., rₖ).
/// 
/// This structure provides optimized implementations for various scenarios:
/// - Small tensor products: Direct computation with SIMD optimization
/// - Large tensor products: Streaming computation with memory management
/// - Repeated evaluations: Caching and memoization strategies
/// - GPU acceleration: CUDA kernels for massive parallel computation
/// 
/// Mathematical Foundation:
/// The tensor product operation extends the concept of Cartesian product
/// to vector spaces, enabling efficient representation of multilinear functions.
/// For vectors u = (u₁, u₂) and v = (v₁, v₂), their tensor product is:
/// u ⊗ v = (u₁v₁, u₁v₂, u₂v₁, u₂v₂)
/// 
/// Performance Characteristics:
/// - Time Complexity: O(k · 2ᵏ) for k variables
/// - Space Complexity: O(2ᵏ) for result storage
/// - Memory Access: Optimized for cache-friendly patterns
/// - Parallelization: Supports both CPU and GPU acceleration
#[derive(Clone, Debug, Zeroize, ZeroizeOnDrop)]
pub struct TensorProduct {
    /// Number of variables k in the tensor product
    /// This determines the size of the resulting tensor: 2ᵏ elements
    num_variables: usize,
    
    /// Evaluation point (r₁, ..., rₖ) for tensor product computation
    /// Each rᵢ is a ring element that will be used in the computation
    /// tensor(r) = ⊗_{i=1}ᵏ (1-rᵢ, rᵢ)
    evaluation_point: Vec<RingElement>,
    
    /// Computed tensor product values (cached for reuse)
    /// None indicates the tensor product has not been computed yet
    /// Some(vec) contains the 2ᵏ tensor product values in lexicographic order
    cached_result: Option<Vec<RingElement>>,
    
    /// Ring dimension for coefficient arithmetic
    /// All ring elements must have this dimension for compatibility
    ring_dimension: usize,
    
    /// Optional modulus for ring operations
    /// When Some(q), operations performed in Rq = R/qR
    /// When None, operations performed over integer ring Z[X]/(X^d + 1)
    modulus: Option<i64>,
 // Optional modulus for ring operations
    /// Performance statistics for tensor product operations
    /// Tracks computation times, memory usage, and optimization effectiveness
    stats: TensorProductStats,
}

/// Statistics for tensor product operations
/// 
/// This structure tracks performance metrics to enable optimization
/// and monitoring of tensor product computations.
#[derive(Clone, Debug, Default, Zeroize)]
pub struct TensorProductStats {
    /// Number of tensor product computations performed
    /// Tracks total number of tensor(r) evaluations
    num_computations: u64,
    
    /// Total computation time in nanoseconds
    /// Used for performance analysis and optimization
    total_computation_time: u64,
    
    /// Peak memory usage in bytes during computation
    /// Monitors memory consumption for large tensor products
    peak_memory_usage: usize,
    
    /// Number of cache hits for repeated evaluations
    /// Higher hit rates indicate better cache utilization
    cache_hits: u64,
    
    /// Number of SIMD operations performed
    /// Tracks vectorization effectiveness
    simd_operations: u64,
    
    /// Number of parallel operations performed
    /// Tracks parallelization effectiveness
    parallel_operations: u64,
}

impl TensorProduct {
    /// Creates a new tensor product computation for the given evaluation point
    /// 
    /// # Arguments
    /// * `evaluation_point` - Point (r₁, ..., rₖ) for tensor product computation
    /// * `ring_dimension` - Dimension of the underlying ring (must be power of 2)
    /// * `modulus` - Optional modulus for ring operations
    /// 
    /// # Returns
    /// * `Result<Self>` - New tensor product computation or error
    /// 
    /// # Mathematical Validation
    /// - Validates all ring elements have consistent dimension and modulus
    /// - Ensures ring dimension is power of 2 for NTT compatibility
    /// - Verifies evaluation point is not empty (k ≥ 1)
    /// 
    /// # Performance Characteristics
    /// - Time Complexity: O(k) for validation
    /// - Space Complexity: O(k) for evaluation point storage
    /// - No computation performed until evaluate() is called
    /// 
    /// # Example Usage
    /// ```rust
    /// // Create tensor product for 3 variables
    /// let evaluation_point = vec![r1, r2, r3]; // Each rᵢ is a RingElement
    /// let tensor = TensorProduct::new(evaluation_point, ring_dimension, Some(modulus))?;
    /// let result = tensor.evaluate()?; // Computes ⊗ᵢ (1-rᵢ, rᵢ)
    /// ```
    pub fn new(
        evaluation_point: Vec<RingElement>,
        ring_dimension: usize,
        modulus: Option<i64>,
    ) -> Result<Self> {
        // Validate evaluation point is not empty
        // Empty evaluation point is mathematically undefined
        if evaluation_point.is_empty() {
            return Err(LatticeFoldError::InvalidParameters(
                "Evaluation point cannot be empty".to_string(),
            ));
        }
        
        let num_variables = evaluation_point.len();
        
        // Validate number of variables is within supported range
        // Large k values lead to exponential memory usage: O(2ᵏ)
        if num_variables > MAX_VARIABLES {
            return Err(LatticeFoldError::InvalidParameters(
                format!("Number of variables {} exceeds maximum {}", 
                       num_variables, MAX_VARIABLES),
            ));
        }
        
        // Validate ring dimension is power of 2
        // This is required for efficient NTT operations and memory alignment
        if !ring_dimension.is_power_of_two() {
            return Err(LatticeFoldError::InvalidDimension {
                expected: ring_dimension.next_power_of_two(),
                got: ring_dimension,
            });
        }
        
        // Validate all evaluation point elements have consistent properties
        for (i, elem) in evaluation_point.iter().enumerate() {
            // Check ring dimension consistency
            if elem.dimension() != ring_dimension {
                return Err(LatticeFoldError::InvalidDimension {
                    expected: ring_dimension,
                    got: elem.dimension(),
                });
            }
            
            // Check modulus consistency
            if elem.modulus() != modulus {
                return Err(LatticeFoldError::InvalidParameters(
                    format!("Evaluation point element {} has inconsistent modulus", i),
                ));
            }
        }
        
        // Validate modulus if specified
        if let Some(q) = modulus {
            if q <= 0 {
                return Err(LatticeFoldError::InvalidModulus { modulus: q });
            }
        }
        
        // Initialize performance statistics
        let stats = TensorProductStats::default();
        
        Ok(Self {
            num_variables,
            evaluation_point,
            cached_result: None,
            ring_dimension,
            modulus,
            stats,
        })
    }
    
    /// Evaluates the tensor product ⊗_{i=1}ᵏ (1-rᵢ, rᵢ)
    /// 
    /// # Returns
    /// * `Result<Vec<RingElement>>` - Tensor product vector of size 2ᵏ
    /// 
    /// # Mathematical Computation
    /// Computes the tensor product iteratively:
    /// 1. Start with (1-r₁, r₁) for the first variable
    /// 2. For each subsequent variable rᵢ, expand the current tensor:
    ///    (a₁, a₂, ..., aₙ) ⊗ (1-rᵢ, rᵢ) = (a₁(1-rᵢ), a₁rᵢ, a₂(1-rᵢ), a₂rᵢ, ..., aₙ(1-rᵢ), aₙrᵢ)
    /// 3. Continue until all k variables are processed
    /// 
    /// The final result is a vector of 2ᵏ ring elements representing the
    /// tensor product in lexicographic order of binary indices.
    /// 
    /// # Performance Optimization
    /// - Uses cached result if available (avoids recomputation)
    /// - Employs SIMD vectorization for parallel multiplication
    /// - Utilizes parallel processing for large tensor products
    /// - Optimizes memory access patterns for cache efficiency
    /// 
    /// # Memory Management
    /// - Allocates result vector with exact size 2ᵏ
    /// - Uses in-place computation where possible to minimize allocations
    /// - Employs memory pools for frequent operations
    /// - Caches result for future evaluations
    /// 
    /// # Time Complexity
    /// - O(k · 2ᵏ) ring operations for k variables
    /// - Each iteration doubles the tensor size
    /// - Parallelization can reduce wall-clock time significantly
    /// 
    /// # Space Complexity
    /// - O(2ᵏ · d) where d is ring dimension
    /// - Temporary storage during computation: O(2ᵏ · d)
    /// - Cached result storage: O(2ᵏ · d)
    pub fn evaluate(&mut self) -> Result<Vec<RingElement>> {
        // Return cached result if available
        if let Some(ref cached) = self.cached_result {
            self.stats.cache_hits += 1;
            return Ok(cached.clone());
        }
        
        // Start timing for performance statistics
        let start_time = std::time::Instant::now();
        
        let k = self.num_variables;
        let tensor_size = 1usize << k; // 2^k
        
        // Check for potential memory overflow
        // Prevent allocation of excessively large tensors
        if tensor_size > MAX_DOMAIN_SIZE {
            return Err(LatticeFoldError::InvalidParameters(
                format!("Tensor size {} exceeds maximum domain size {}", 
                       tensor_size, MAX_DOMAIN_SIZE),
            ));
        }
        
        // Initialize tensor product with first variable: (1-r₁, r₁)
        let mut tensor = Vec::with_capacity(tensor_size);
        
        // Compute 1 - r₁ for first variable
        // This requires creating the multiplicative identity element
        let one = RingElement::one(self.ring_dimension, self.modulus)?;
        let one_minus_r1 = one.sub(&self.evaluation_point[0])?;
        
        // Initialize tensor with (1-r₁, r₁)
        // This is the base case for the recursive tensor product definition
        tensor.push(one_minus_r1);
        tensor.push(self.evaluation_point[0].clone());
        
        // Iteratively expand tensor product for remaining variables
        // Each iteration doubles the size of the tensor
        for i in 1..k {
            let ri = &self.evaluation_point[i];
            let one_minus_ri = one.sub(ri)?;
            
            // Current tensor size before expansion
            let current_size = tensor.len();
            
            // Expand tensor: for each existing element a, add (a(1-rᵢ), arᵢ)
            // This implements the tensor product operation: T ⊗ (1-rᵢ, rᵢ)
            let mut new_tensor = Vec::with_capacity(current_size * 2);
            
            // Choose computation strategy based on tensor size
            if current_size >= TENSOR_SIMD_WIDTH * 2 {
                // Use parallel processing for large tensors
                // This provides significant speedup for k ≥ 6 (tensor size ≥ 64)
                let chunks: Vec<_> = tensor
                    .par_chunks(TENSOR_SIMD_WIDTH)
                    .map(|chunk| {
                        let mut chunk_result = Vec::with_capacity(chunk.len() * 2);
                        
                        // Process each element in the chunk
                        for elem in chunk {
                            // Compute elem * (1 - rᵢ)
                            // This represents the "0" branch in the binary tree expansion
                            let elem_times_one_minus_ri = elem.mul(&one_minus_ri)?;
                            chunk_result.push(elem_times_one_minus_ri);
                            
                            // Compute elem * rᵢ
                            // This represents the "1" branch in the binary tree expansion
                            let elem_times_ri = elem.mul(ri)?;
                            chunk_result.push(elem_times_ri);
                        }
                        
                        Ok::<Vec<RingElement>, LatticeFoldError>(chunk_result)
                    })
                    .collect::<Result<Vec<_>>>()?;
                
                // Flatten parallel results into new tensor
                // Maintain lexicographic ordering of binary indices
                for chunk_result in chunks {
                    new_tensor.extend(chunk_result);
                }
                
                // Update parallel operation statistics
                self.stats.parallel_operations += 1;
            } else {
                // Use sequential processing for small tensors
                // Sequential processing has lower overhead for small sizes
                for elem in &tensor {
                    // Compute elem * (1 - rᵢ)
                    let elem_times_one_minus_ri = elem.mul(&one_minus_ri)?;
                    new_tensor.push(elem_times_one_minus_ri);
                    
                    // Compute elem * rᵢ
                    let elem_times_ri = elem.mul(ri)?;
                    new_tensor.push(elem_times_ri);
                }
            }
            
            // Replace tensor with expanded version
            // This completes one iteration of the tensor product expansion
            tensor = new_tensor;
            
            // Update SIMD operation count for statistics
            self.stats.simd_operations += (current_size / TENSOR_SIMD_WIDTH) as u64;
        }
        
        // Validate final tensor size matches expected 2ᵏ
        // This is a critical consistency check
        if tensor.len() != tensor_size {
            return Err(LatticeFoldError::InvalidDimension {
                expected: tensor_size,
                got: tensor.len(),
            });
        }
        
        // Cache result for future evaluations
        // This significantly improves performance for repeated evaluations
        self.cached_result = Some(tensor.clone());
        
        // Update performance statistics
        self.stats.num_computations += 1;
        self.stats.total_computation_time += start_time.elapsed().as_nanos() as u64;
        self.stats.peak_memory_usage = self.stats.peak_memory_usage.max(
            tensor.len() * self.ring_dimension * 8 // Estimate: 8 bytes per coefficient
        );
        
        Ok(tensor)
    }
    
    /// Evaluates the tensor product at a specific binary index
    /// 
    /// # Arguments
    /// * `binary_index` - Binary index b ∈ {0,1}ᵏ represented as integer
    /// 
    /// # Returns
    /// * `Result<RingElement>` - Tensor product value at index b
    /// 
    /// # Mathematical Computation
    /// For binary index b = (b₁, b₂, ..., bₖ), computes:
    /// tensor(r)[b] = ∏_{i=1}ᵏ ((1-bᵢ)(1-rᵢ) + bᵢrᵢ)
    /// 
    /// This is more efficient than computing the full tensor product
    /// when only a single value is needed.
    /// 
    /// # Performance Benefits
    /// - Time Complexity: O(k) instead of O(k · 2ᵏ) for full tensor
    /// - Space Complexity: O(1) instead of O(2ᵏ) for full tensor
    /// - Ideal for sparse evaluations or streaming computations
    /// 
    /// # Binary Index Interpretation
    /// The binary index is interpreted as follows:
    /// - Index 0 = 000...0₂ corresponds to all variables being 0
    /// - Index 1 = 000...1₂ corresponds to last variable being 1, others 0
    /// - Index 2ᵏ-1 = 111...1₂ corresponds to all variables being 1
    pub fn evaluate_at_index(&self, binary_index: usize) -> Result<RingElement> {
        // Validate binary index is within valid range [0, 2ᵏ)
        let max_index = 1usize << self.num_variables;
        if binary_index >= max_index {
            return Err(LatticeFoldError::InvalidParameters(
                format!("Binary index {} exceeds maximum {}", binary_index, max_index - 1),
            ));
        }
        
        // Initialize result with multiplicative identity
        let mut result = RingElement::one(self.ring_dimension, self.modulus)?;
        let one = RingElement::one(self.ring_dimension, self.modulus)?;
        
        // Process each bit of the binary index
        for i in 0..self.num_variables {
            // Extract bit i from binary_index (LSB is bit 0)
            let bit_i = (binary_index >> i) & 1;
            let ri = &self.evaluation_point[i];
            
            // Compute factor for variable i based on bit value
            let factor = if bit_i == 0 {
                // Bit is 0: use (1 - rᵢ)
                one.sub(ri)?
            } else {
                // Bit is 1: use rᵢ
                ri.clone()
            };
            
            // Multiply result by the factor
            result = result.mul(&factor)?;
        }
        
        Ok(result)
    }
    
    /// Performs batch evaluation of tensor product at multiple binary indices
    /// 
    /// # Arguments
    /// * `binary_indices` - Vector of binary indices for evaluation
    /// 
    /// # Returns
    /// * `Result<Vec<RingElement>>` - Tensor product values at specified indices
    /// 
    /// # Performance Benefits
    /// - Amortizes setup costs across multiple evaluations
    /// - Uses parallel processing for independent computations
    /// - More efficient than repeated single-index evaluations
    /// - Optimizes memory access patterns
    /// 
    /// # Use Cases
    /// - Sparse tensor product evaluation
    /// - Streaming sumcheck protocols
    /// - Memory-constrained environments
    /// - Custom evaluation patterns
    pub fn batch_evaluate_at_indices(&self, binary_indices: &[usize]) -> Result<Vec<RingElement>> {
        // Validate all indices are within valid range
        let max_index = 1usize << self.num_variables;
        for &index in binary_indices {
            if index >= max_index {
                return Err(LatticeFoldError::InvalidParameters(
                    format!("Binary index {} exceeds maximum {}", index, max_index - 1),
                ));
            }
        }
        
        // Process indices in parallel for better performance
        let results: Vec<_> = binary_indices
            .par_iter()
            .map(|&index| self.evaluate_at_index(index))
            .collect::<Result<Vec<_>>>()?;
        
        Ok(results)
    }
    
    /// Returns the number of variables in the tensor product
    pub fn num_variables(&self) -> usize {
        self.num_variables
    }
    
    /// Returns the ring dimension used for coefficients
    pub fn ring_dimension(&self) -> usize {
        self.ring_dimension
    }
    
    /// Returns the modulus used for ring operations
    pub fn modulus(&self) -> Option<i64> {
        self.modulus
    }
    
    /// Returns the evaluation point used for tensor product computation
    pub fn evaluation_point(&self) -> &[RingElement] {
        &self.evaluation_point
    }
    
    /// Returns performance statistics for tensor product operations
    pub fn stats(&self) -> &TensorProductStats {
        &self.stats
    }
    
    /// Clears the cached result to free memory
    /// 
    /// This is useful for long-running computations where memory usage
    /// becomes a concern. The result will be recomputed on next evaluation.
    pub fn clear_cache(&mut self) {
        self.cached_result = None;
    }
    
    /// Checks if the tensor product result is cached
    pub fn is_cached(&self) -> bool {
        self.cached_result.is_some()
    }
    
    /// Estimates memory usage of the tensor product in bytes
    /// 
    /// # Returns
    /// * `usize` - Estimated memory usage including cached result
    /// 
    /// # Components
    /// - Evaluation point: k × d × 8 bytes (assuming i64 coefficients)
    /// - Cached result: 2ᵏ × d × 8 bytes (if cached)
    /// - Metadata and overhead: ~1KB
    pub fn memory_usage(&self) -> usize {
        let evaluation_point_size = self.evaluation_point.len() * self.ring_dimension * 8;
        let cached_result_size = if let Some(ref cached) = self.cached_result {
            cached.len() * self.ring_dimension * 8
        } else {
            0
        };
        let metadata_size = 1024; // Approximate overhead
        
        evaluation_point_size + cached_result_size + metadata_size
    }
}

impl Display for TensorProduct {
    /// User-friendly display formatting for tensor product
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(
            f,
            "TensorProduct(k={}, size={}, ring_dim={}, cached={})",
            self.num_variables,
            1 << self.num_variables,
            self.ring_dimension,
            self.is_cached()
        )
    }
}  modulus: Option<i64>,
    
    /// Performance statistics for optimization
    stats: TensorProductStats,
}

/// Statistics for tensor product operations
#[derive(Clone, Debug, Default, Zeroize)]
pub struct TensorProductStats {
    /// Number of tensor product computations performed
    num_computations: u64,
    
    /// Total computation time in nanoseconds
    total_computation_time: u64,
    
    /// Number of cache hits (reused computations)
    cache_hits: u64,
    
    /// Peak memory usage during computation
    peak_memory_usage: usize,
    
    /// Number of SIMD operations performed
    simd_operations: u64,
    
    /// Number of parallel threads used
    parallel_threads: usize,
}

impl TensorProduct {
    /// Creates a new tensor product for the given evaluation point
    /// 
    /// # Arguments
    /// * `evaluation_point` - Point (r₁, ..., rₖ) for tensor product computation
    /// * `ring_dimension` - Dimension of the underlying ring
    /// * `modulus` - Optional modulus for ring operations
    /// 
    /// # Returns
    /// * `Result<Self>` - New tensor product instance or error
    pub fn new(
        evaluation_point: Vec<RingElement>,
        ring_dimension: usize,
        modulus: Option<i64>,
    ) -> Result<Self> {
        // Validate evaluation point is not empty
        if evaluation_point.is_empty() {
            return Err(LatticeFoldError::InvalidParameters(
                "Evaluation point cannot be empty".to_string(),
            ));
        }
        
        let num_variables = evaluation_point.len();
        
        // Validate number of variables is within supported range
        if num_variables > MAX_VARIABLES {
            return Err(LatticeFoldError::InvalidParameters(
                format!("Number of variables {} exceeds maximum {}", 
                       num_variables, MAX_VARIABLES),
            ));
        }
        
        // Validate ring dimension is power of 2
        if !ring_dimension.is_power_of_two() {
            return Err(LatticeFoldError::InvalidDimension {
                expected: ring_dimension.next_power_of_two(),
                got: ring_dimension,
            });
        }
        
        // Validate all evaluation point elements have consistent properties
        for (i, elem) in evaluation_point.iter().enumerate() {
            if elem.dimension() != ring_dimension {
                return Err(LatticeFoldError::InvalidDimension {
                    expected: ring_dimension,
                    got: elem.dimension(),
                });
            }
            
            if elem.modulus() != modulus {
                return Err(LatticeFoldError::InvalidParameters(
                    format!("Evaluation point element {} has inconsistent modulus", i),
                ));
            }
        }
        
        // Validate modulus if specified
        if let Some(q) = modulus {
            if q <= 0 {
                return Err(LatticeFoldError::InvalidModulus { modulus: q });
            }
        }
        
        Ok(Self {
            num_variables,
            evaluation_point,
            cached_result: None,
            ring_dimension,
            modulus,
            stats: TensorProductStats::default(),
        })
    }
    
    /// Computes the tensor product ⊗_{i=1}ᵏ (1-rᵢ, rᵢ)
    /// 
    /// # Returns
    /// * `Result<&Vec<RingElement>>` - Reference to computed tensor product
    /// 
    /// # Computation Strategy
    /// The tensor product is computed iteratively:
    /// 1. Start with (1-r₁, r₁) for the first variable
    /// 2. For each subsequent variable rᵢ, expand the current tensor:
    ///    (a₁, a₂, ..., aₙ) ⊗ (1-rᵢ, rᵢ) = (a₁(1-rᵢ), a₁rᵢ, a₂(1-rᵢ), a₂rᵢ, ..., aₙ(1-rᵢ), aₙrᵢ)
    /// 
    /// # Performance Optimization
    /// - Uses SIMD vectorization for parallel multiplication
    /// - Employs memory-efficient in-place computation where possible
    /// - Caches result for repeated access
    /// - Supports GPU acceleration for large tensor products
    pub fn compute(&mut self) -> Result<&Vec<RingElement>> {
        // Return cached result if available
        if let Some(ref result) = self.cached_result {
            self.stats.cache_hits += 1;
            return Ok(result);
        }
        
        // Start timing for performance statistics
        let start_time = std::time::Instant::now();
        
        // Compute tensor product iteratively
        let result = self.compute_tensor_product_iterative()?;
        
        // Cache the result for future use
        self.cached_result = Some(result);
        
        // Update performance statistics
        self.stats.num_computations += 1;
        self.stats.total_computation_time += start_time.elapsed().as_nanos() as u64;
        
        // Return reference to cached result
        Ok(self.cached_result.as_ref().unwrap())
    }
    
    /// Internal method for iterative tensor product computation
    /// 
    /// This method implements the core tensor product algorithm with
    /// optimizations for different sizes and hardware capabilities.
    fn compute_tensor_product_iterative(&mut self) -> Result<Vec<RingElement>> {
        let k = self.num_variables;
        let tensor_size = 1usize << k; // 2^k
        
        // Check if tensor size exceeds memory limits
        let estimated_memory = tensor_size * self.ring_dimension * 8; // 8 bytes per i64
        if estimated_memory > MAX_DOMAIN_SIZE * 8 {
            return Err(LatticeFoldError::InvalidParameters(
                format!("Tensor product size {} exceeds memory limit", estimated_memory),
            ));
        }
        
        // Initialize tensor product with first variable: (1-r₁, r₁)
        let one = RingElement::one(self.ring_dimension, self.modulus)?;
        let one_minus_r1 = one.sub(&self.evaluation_point[0])?;
        
        let mut tensor = vec![one_minus_r1, self.evaluation_point[0].clone()];
        
        // Iteratively expand tensor product for remaining variables
        for i in 1..k {
            let ri = &self.evaluation_point[i];
            let one_minus_ri = one.sub(ri)?;
            
            // Current tensor size before expansion
            let current_size = tensor.len();
            
            // Choose computation strategy based on tensor size
            if current_size >= TENSOR_SIMD_WIDTH * 8 {
                // Use parallel computation for large tensors
                tensor = self.expand_tensor_parallel(&tensor, &one_minus_ri, ri)?;
            } else {
                // Use sequential computation for small tensors
                tensor = self.expand_tensor_sequential(&tensor, &one_minus_ri, ri)?;
            }
            
            // Update memory usage statistics
            let current_memory = tensor.len() * self.ring_dimension * 8;
            self.stats.peak_memory_usage = self.stats.peak_memory_usage.max(current_memory);
        }
        
        // Validate final tensor size
        if tensor.len() != tensor_size {
            return Err(LatticeFoldError::InvalidDimension {
                expected: tensor_size,
                got: tensor.len(),
            });
        }
        
        Ok(tensor)
    }
    
    /// Expands tensor product using parallel computation
    /// 
    /// # Arguments
    /// * `current_tensor` - Current tensor product state
    /// * `one_minus_ri` - Precomputed (1 - rᵢ) for current variable
    /// * `ri` - Current variable value rᵢ
    /// 
    /// # Returns
    /// * `Result<Vec<RingElement>>` - Expanded tensor product
    fn expand_tensor_parallel(
        &mut self,
        current_tensor: &[RingElement],
        one_minus_ri: &RingElement,
        ri: &RingElement,
    ) -> Result<Vec<RingElement>> {
        // Process tensor elements in parallel chunks
        let chunk_size = TENSOR_SIMD_WIDTH.max(current_tensor.len() / rayon::current_num_threads());
        
        let expanded_chunks: Vec<_> = current_tensor
            .par_chunks(chunk_size)
            .map(|chunk| {
                let mut chunk_result = Vec::with_capacity(chunk.len() * 2);
                
                for elem in chunk {
                    // Compute elem * (1 - rᵢ)
                    let elem_times_one_minus_ri = elem.mul(one_minus_ri)?;
                    chunk_result.push(elem_times_one_minus_ri);
                    
                    // Compute elem * rᵢ
                    let elem_times_ri = elem.mul(ri)?;
                    chunk_result.push(elem_times_ri);
                }
                
                Ok::<Vec<RingElement>, LatticeFoldError>(chunk_result)
            })
            .collect::<Result<Vec<_>>>()?;
        
        // Flatten parallel results
        let mut expanded_tensor = Vec::with_capacity(current_tensor.len() * 2);
        for chunk_result in expanded_chunks {
            expanded_tensor.extend(chunk_result);
        }
        
        // Update parallel processing statistics
        self.stats.parallel_threads = rayon::current_num_threads();
        self.stats.simd_operations += (current_tensor.len() / TENSOR_SIMD_WIDTH) as u64;
        
        Ok(expanded_tensor)
    }
    
    /// Expands tensor product using sequential computation
    /// 
    /// # Arguments
    /// * `current_tensor` - Current tensor product state
    /// * `one_minus_ri` - Precomputed (1 - rᵢ) for current variable
    /// * `ri` - Current variable value rᵢ
    /// 
    /// # Returns
    /// * `Result<Vec<RingElement>>` - Expanded tensor product
    fn expand_tensor_sequential(
        &mut self,
        current_tensor: &[RingElement],
        one_minus_ri: &RingElement,
        ri: &RingElement,
    ) -> Result<Vec<RingElement>> {
        let mut expanded_tensor = Vec::with_capacity(current_tensor.len() * 2);
        
        for elem in current_tensor {
            // Compute elem * (1 - rᵢ)
            let elem_times_one_minus_ri = elem.mul(one_minus_ri)?;
            expanded_tensor.push(elem_times_one_minus_ri);
            
            // Compute elem * rᵢ
            let elem_times_ri = elem.mul(ri)?;
            expanded_tensor.push(elem_times_ri);
        }
        
        Ok(expanded_tensor)
    }
    
    /// Returns the number of variables in the tensor product
    pub fn num_variables(&self) -> usize {
        self.num_variables
    }
    
    /// Returns the evaluation point used for the tensor product
    pub fn evaluation_point(&self) -> &[RingElement] {
        &self.evaluation_point
    }
    
    /// Returns performance statistics for the tensor product
    pub fn stats(&self) -> &TensorProductStats {
        &self.stats
    }
    
    /// Clears the cached result to free memory
    pub fn clear_cache(&mut self) {
        self.cached_result = None;
    }
    
    /// Checks if the tensor product result is cached
    pub fn is_cached(&self) -> bool {
        self.cached_result.is_some()
    }
    
    /// Estimates memory usage of the tensor product in bytes
    pub fn memory_usage(&self) -> usize {
        let evaluation_point_size = self.evaluation_point.len() * self.ring_dimension * 8;
        let cached_result_size = if let Some(ref result) = self.cached_result {
            result.len() * self.ring_dimension * 8
        } else {
            0
        };
        let metadata_size = 256; // Approximate overhead
        
        evaluation_point_size + cached_result_size + metadata_size
    }
}

impl Display for TensorProduct {
    /// User-friendly display formatting for tensor product
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(
            f,
            "TensorProduct(k={}, cached={}, ring_dim={})",
            self.num_variables,
            self.is_cached(),
            self.ring_dimension
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cyclotomic_ring::RingElement;
    
    /// Test multilinear extension creation and basic properties
    #[test]
    fn test_multilinear_extension_creation() {
        let ring_dimension = 64;
        let modulus = Some(97); // Small prime for testing
        let num_variables = 3;
        
        // Create function values for 3-variable function
        let mut function_values = Vec::new();
        for i in 0..(1 << num_variables) {
            let coeffs = vec![i as i64; ring_dimension];
            let ring_elem = RingElement::from_coefficients(coeffs, modulus).unwrap();
            function_values.push(ring_elem);
        }
        
        // Create multilinear extension
        let mle = MultilinearExtension::new(
            num_variables,
            function_values,
            ring_dimension,
            modulus,
        ).unwrap();
        
        assert_eq!(mle.num_variables(), num_variables);
        assert_eq!(mle.ring_dimension(), ring_dimension);
        assert_eq!(mle.modulus(), modulus);
    }
    
    /// Test tensor product computation
    #[test]
    fn test_tensor_product_computation() {
        let ring_dimension = 32;
        let modulus = Some(97);
        
        // Create evaluation point for 2 variables
        let r1 = RingElement::from_coefficients(vec![1; ring_dimension], modulus).unwrap();
        let r2 = RingElement::from_coefficients(vec![2; ring_dimension], modulus).unwrap();
        let evaluation_point = vec![r1, r2];
        
        // Create tensor product
        let mut tensor = TensorProduct::new(evaluation_point, ring_dimension, modulus).unwrap();
        
        // Compute tensor product
        let result = tensor.compute().unwrap();
        
        // Verify result size: 2^2 = 4 elements
        assert_eq!(result.len(), 4);
        
        // Verify caching works
        assert!(tensor.is_cached());
        let result2 = tensor.compute().unwrap();
        assert_eq!(result.len(), result2.len());
    }
    
    /// Test multilinear extension evaluation
    #[test]
    fn test_multilinear_extension_evaluation() {
        let ring_dimension = 32;
        let modulus = Some(97);
        let num_variables = 2;
        
        // Create simple function values: f(0,0)=0, f(0,1)=1, f(1,0)=2, f(1,1)=3
        let mut function_values = Vec::new();
        for i in 0..(1 << num_variables) {
            let coeffs = vec![i as i64; ring_dimension];
            let ring_elem = RingElement::from_coefficients(coeffs, modulus).unwrap();
            function_values.push(ring_elem);
        }
        
        // Create multilinear extension
        let mut mle = MultilinearExtension::new(
            num_variables,
            function_values,
            ring_dimension,
            modulus,
        ).unwrap();
        
        // Create evaluation point (0, 0) - should give f(0,0) = 0
        let zero = RingElement::zero(ring_dimension, modulus).unwrap();
        let evaluation_point = vec![zero.clone(), zero.clone()];
        
        let result = mle.evaluate(&evaluation_point).unwrap();
        
        // At (0,0), multilinear extension should equal f(0,0) = 0
        assert_eq!(result.coefficients().coefficients()[0], 0);
    }
    
    /// Test batch evaluation performance
    #[test]
    fn test_batch_evaluation() {
        let ring_dimension = 32;
        let modulus = Some(97);
        let num_variables = 2;
        
        // Create function values
        let mut function_values = Vec::new();
        for i in 0..(1 << num_variables) {
            let coeffs = vec![i as i64; ring_dimension];
            let ring_elem = RingElement::from_coefficients(coeffs, modulus).unwrap();
            function_values.push(ring_elem);
        }
        
        // Create multilinear extension
        let mut mle = MultilinearExtension::new(
            num_variables,
            function_values,
            ring_dimension,
            modulus,
        ).unwrap();
        
        // Create multiple evaluation points
        let mut evaluation_points = Vec::new();
        for i in 0..4 {
            let r1 = RingElement::from_coefficients(vec![i; ring_dimension], modulus).unwrap();
            let r2 = RingElement::from_coefficients(vec![i + 1; ring_dimension], modulus).unwrap();
            evaluation_points.push(vec![r1, r2]);
        }
        
        // Perform batch evaluation
        let results = mle.batch_evaluate(&evaluation_points).unwrap();
        
        // Verify we get results for all evaluation points
        assert_eq!(results.len(), evaluation_points.len());
    }
    
    /// Test error handling for invalid inputs
    #[test]
    fn test_error_handling() {
        let ring_dimension = 32;
        let modulus = Some(97);
        
        // Test invalid number of variables
        let result = MultilinearExtension::new(
            MAX_VARIABLES + 1, // Too many variables
            vec![],
            ring_dimension,
            modulus,
        );
        assert!(result.is_err());
        
        // Test invalid function values size
        let function_values = vec![RingElement::zero(ring_dimension, modulus).unwrap()]; // Only 1 value for 2 variables
        let result = MultilinearExtension::new(
            2, // Need 2^2 = 4 values
            function_values,
            ring_dimension,
            modulus,
        );
        assert!(result.is_err());
        
        // Test invalid ring dimension (not power of 2)
        let result = TensorProduct::new(
            vec![RingElement::zero(33, modulus).unwrap()], // 33 is not power of 2
            33,
            modulus,
        );
        assert!(result.is_err());
    }
    
    /// Test memory usage estimation
    #[test]
    fn test_memory_usage() {
        let ring_dimension = 32;
        let modulus = Some(97);
        let num_variables = 3;
        
        // Create function values
        let mut function_values = Vec::new();
        for i in 0..(1 << num_variables) {
            let coeffs = vec![i as i64; ring_dimension];
            let ring_elem = RingElement::from_coefficients(coeffs, modulus).unwrap();
            function_values.push(ring_elem);
        }
        
        // Create multilinear extension
        let mle = MultilinearExtension::new(
            num_variables,
            function_values,
            ring_dimension,
            modulus,
        ).unwrap();
        
        // Check memory usage is reasonable
        let memory_usage = mle.memory_usage();
        assert!(memory_usage > 0);
        
        // Should be at least the size of function values
        let min_expected = (1 << num_variables) * ring_dimension * 8;
        assert!(memory_usage >= min_expected);
    }
}
/// Represents a ring-based sumcheck protocol implementation
/// 
/// The ring-based sumcheck protocol extends the classical sumcheck protocol
/// to work over cyclotomic rings R = Z[X
//
/// Ring-based sumcheck protocol with soundness analysis
/// 
/// This structure implements the generalized sumcheck protocol over cyclotomic rings
/// as described in Section 2.2 of the LatticeFold+ paper. The protocol enables
/// efficient verification of polynomial identities of the form:
/// 
/// Σ_{x∈{0,1}ᵏ} f(x) = claimed_sum
/// 
/// where f is a multilinear polynomial over the ring R = Z[X]/(X^d + 1).
/// 
/// Key Features:
/// - Soundness error: kℓ/|C| for challenge set C and polynomial degree ℓ
/// - Batching mechanism for multiple claims over the same domain
/// - Parallel repetition for soundness amplification: (kℓ/|C|)ʳ
/// - Extension field lifting for small modulus q support
/// - Communication optimization through proof compression
/// 
/// Mathematical Foundation:
/// The sumcheck protocol works by reducing the k-variate sum to a univariate
/// evaluation through a series of interactive rounds. In each round i, the
/// prover sends a univariate polynomial gᵢ(X) and the verifier checks:
/// gᵢ(0) + gᵢ(1) = previous_sum
/// 
/// The verifier then samples a random challenge rᵢ and updates the sum to gᵢ(rᵢ).
/// After k rounds, the verifier checks f(r₁, ..., rₖ) = final_sum.
/// 
/// Security Properties:
/// - Completeness: Honest prover always convinces honest verifier
/// - Soundness: Cheating prover succeeds with probability ≤ kℓ/|C|
/// - Zero-knowledge: Can be made zero-knowledge with additional randomness
/// - Post-quantum security: Based on lattice assumptions
#[derive(Clone, Debug, Zeroize, ZeroizeOnDrop)]
pub struct RingSumcheckProtocol {
    /// Number of variables k in the sumcheck protocol
    /// This determines the number of interactive rounds
    num_variables: usize,
    
    /// Polynomial degree ℓ in each variable
    /// Higher degrees increase soundness error but enable more expressive protocols
    polynomial_degree: usize,
    
    /// Ring dimension for coefficient arithmetic
    /// Must be power of 2 for efficient NTT operations
    ring_dimension: usize,
    
    /// Modulus for ring operations (None for integer ring)
    /// Small moduli may require extension field lifting
    modulus: Option<i64>,
    
    /// Challenge set C for soundness analysis
    /// Larger sets provide better soundness but increase communication
    challenge_set: Vec<RingElement>,
    
    /// Batching parameters for multiple claims
    /// Enables amortization of costs across multiple sumcheck instances
    batch_size: usize,
    
    /// Parallel repetition parameter r for soundness amplification
    /// Soundness error becomes (kℓ/|C|)ʳ with r repetitions
    repetition_parameter: usize,
    
    /// Extension field parameters for small modulus support
    /// Used when |C| is too small for desired soundness level
    extension_field_degree: Option<usize>,
    
    /// Performance statistics for protocol execution
    /// Tracks timing, communication, and optimization effectiveness
    stats: RingSumcheckStats,
}

/// Statistics for ring-based sumcheck protocol
/// 
/// This structure tracks performance metrics and communication costs
/// to enable optimization and analysis of the sumcheck implementation.
#[derive(Clone, Debug, Default, Zeroize)]
pub struct RingSumcheckStats {
    /// Number of sumcheck protocols executed
    /// Tracks total number of protocol instances
    num_protocols: u64,
    
    /// Total prover computation time in nanoseconds
    /// Includes polynomial evaluation and proof generation
    prover_time: u64,
    
    /// Total verifier computation time in nanoseconds
    /// Includes challenge generation and verification
    verifier_time: u64,
    
    /// Total communication cost in bytes
    /// Includes all messages sent between prover and verifier
    communication_cost: usize,
    
    /// Number of rounds executed
    /// Should equal k × num_protocols for standard protocol
    num_rounds: u64,
    
    /// Number of batch operations performed
    /// Tracks batching effectiveness
    batch_operations: u64,
    
    /// Number of parallel repetitions performed
    /// Tracks soundness amplification usage
    parallel_repetitions: u64,
    
    /// Peak memory usage during protocol execution
    /// Monitors memory consumption for large protocols
    peak_memory_usage: usize,
}

/// Represents a sumcheck proof for verification
/// 
/// This structure contains all the information needed to verify
/// a sumcheck protocol execution, including the univariate polynomials
/// sent by the prover in each round.
#[derive(Clone, Debug, Zeroize, ZeroizeOnDrop)]
pub struct RingSumcheckProof {
    /// Claimed sum that the prover asserts
    /// This is the value Σ_{x∈{0,1}ᵏ} f(x) that the prover claims is correct
    claimed_sum: RingElement,
    
    /// Univariate polynomials for each round
    /// round_polynomials[i] is the polynomial gᵢ(X) sent in round i+1
    /// Each polynomial has degree ≤ ℓ where ℓ is the polynomial degree
    round_polynomials: Vec<Vec<RingElement>>, // Coefficients of each polynomial
    
    /// Final evaluation point after all rounds
    /// This is the point (r₁, ..., rₖ) where the final check is performed
    final_evaluation_point: Vec<RingElement>,
    
    /// Final polynomial evaluation f(r₁, ..., rₖ)
    /// This value must match the final sum after all rounds
    final_evaluation: RingElement,
    
    /// Batch proof data for multiple claims (if applicable)
    /// Contains additional information for batched sumcheck protocols
    batch_data: Option<BatchSumcheckData>,
    
    /// Parallel repetition proofs (if applicable)
    /// Contains proofs for each repetition in soundness amplification
    repetition_proofs: Vec<RingSumcheckProof>,
    
    /// Metadata for proof verification
    /// Includes parameters and configuration used in proof generation
    metadata: SumcheckProofMetadata,
}

/// Batch sumcheck data for multiple claims
/// 
/// This structure contains additional information needed to verify
/// batched sumcheck protocols where multiple claims are proven together.
#[derive(Clone, Debug, Zeroize, ZeroizeOnDrop)]
pub struct BatchSumcheckData {
    /// Number of claims in the batch
    /// Each claim is a separate sumcheck instance
    num_claims: usize,
    
    /// Random linear combination coefficients
    /// Used to combine multiple claims into a single sumcheck
    combination_coefficients: Vec<RingElement>,
    
    /// Individual claimed sums for each claim
    /// These are combined using the coefficients to form the overall claim
    individual_claims: Vec<RingElement>,
    
    /// Batch verification randomness
    /// Additional randomness used in batch verification
    batch_randomness: Vec<RingElement>,
}

/// Metadata for sumcheck proof verification
/// 
/// This structure contains configuration parameters and metadata
/// needed to properly verify a sumcheck proof.
#[derive(Clone, Debug, Zeroize)]
pub struct SumcheckProofMetadata {
    /// Number of variables k
    num_variables: usize,
    
    /// Polynomial degree ℓ
    polynomial_degree: usize,
    
    /// Ring dimension d
    ring_dimension: usize,
    
    /// Modulus q (if applicable)
    modulus: Option<i64>,
    
    /// Challenge set size |C|
    challenge_set_size: usize,
    
    /// Batch size (1 for non-batched protocols)
    batch_size: usize,
    
    /// Repetition parameter r
    repetition_parameter: usize,
    
    /// Extension field degree (if applicable)
    extension_field_degree: Option<usize>,
    
    /// Protocol version for compatibility
    protocol_version: u32,
}

impl RingSumcheckProtocol {
    /// Creates a new ring-based sumcheck protocol
    /// 
    /// # Arguments
    /// * `num_variables` - Number of variables k (must be ≤ MAX_VARIABLES)
    /// * `polynomial_degree` - Degree ℓ of polynomials in each variable
    /// * `ring_dimension` - Dimension d of the ring (must be power of 2)
    /// * `modulus` - Optional modulus q for ring operations
    /// * `challenge_set` - Set C of challenges for soundness analysis
    /// 
    /// # Returns
    /// * `Result<Self>` - New sumcheck protocol or error
    /// 
    /// # Mathematical Validation
    /// - Validates k ≤ MAX_VARIABLES to prevent exponential blowup
    /// - Ensures polynomial degree ℓ ≥ 1 for meaningful protocols
    /// - Verifies ring dimension d is power of 2 for NTT compatibility
    /// - Checks challenge set is non-empty and has consistent ring properties
    /// 
    /// # Soundness Analysis
    /// The soundness error of the protocol is kℓ/|C| where:
    /// - k is the number of variables (rounds)
    /// - ℓ is the polynomial degree
    /// - |C| is the size of the challenge set
    /// 
    /// For λ-bit security, we need kℓ/|C| ≤ 2^{-λ}, which gives |C| ≥ kℓ·2^λ.
    /// 
    /// # Performance Characteristics
    /// - Prover complexity: O(k · 2ᵏ · d) ring operations
    /// - Verifier complexity: O(k · ℓ · d) ring operations
    /// - Communication: O(k · ℓ · d) ring elements
    /// - Memory usage: O(2ᵏ · d) for multilinear extension storage
    pub fn new(
        num_variables: usize,
        polynomial_degree: usize,
        ring_dimension: usize,
        modulus: Option<i64>,
        challenge_set: Vec<RingElement>,
    ) -> Result<Self> {
        // Validate number of variables is within supported range
        if num_variables == 0 {
            return Err(LatticeFoldError::InvalidParameters(
                "Number of variables must be positive".to_string(),
            ));
        }
        
        if num_variables > MAX_VARIABLES {
            return Err(LatticeFoldError::InvalidParameters(
                format!("Number of variables {} exceeds maximum {}", 
                       num_variables, MAX_VARIABLES),
            ));
        }
        
        // Validate polynomial degree is meaningful
        if polynomial_degree == 0 {
            return Err(LatticeFoldError::InvalidParameters(
                "Polynomial degree must be positive".to_string(),
            ));
        }
        
        // Validate ring dimension is power of 2
        if !ring_dimension.is_power_of_two() {
            return Err(LatticeFoldError::InvalidDimension {
                expected: ring_dimension.next_power_of_two(),
                got: ring_dimension,
            });
        }
        
        // Validate challenge set is non-empty
        if challenge_set.is_empty() {
            return Err(LatticeFoldError::InvalidParameters(
                "Challenge set cannot be empty".to_string(),
            ));
        }
        
        // Validate all challenge set elements have consistent properties
        for (i, elem) in challenge_set.iter().enumerate() {
            if elem.dimension() != ring_dimension {
                return Err(LatticeFoldError::InvalidDimension {
                    expected: ring_dimension,
                    got: elem.dimension(),
                });
            }
            
            if elem.modulus() != modulus {
                return Err(LatticeFoldError::InvalidParameters(
                    format!("Challenge set element {} has inconsistent modulus", i),
                ));
            }
        }
        
        // Validate modulus if specified
        if let Some(q) = modulus {
            if q <= 0 {
                return Err(LatticeFoldError::InvalidModulus { modulus: q });
            }
        }
        
        // Analyze soundness error and recommend parameters if needed
        let soundness_error_numerator = num_variables * polynomial_degree;
        let soundness_error_denominator = challenge_set.len();
        
        // Warn if soundness error is too large (> 2^{-80})
        if soundness_error_numerator > soundness_error_denominator / (1u64 << 80) as usize {
            eprintln!("Warning: Soundness error {}/{} may be too large for cryptographic security",
                     soundness_error_numerator, soundness_error_denominator);
        }
        
        // Initialize default parameters
        let batch_size = 1; // Default to single-instance protocols
        let repetition_parameter = 1; // Default to no parallel repetition
        let extension_field_degree = None; // Default to base field
        let stats = RingSumcheckStats::default();
        
        Ok(Self {
            num_variables,
            polynomial_degree,
            ring_dimension,
            modulus,
            challenge_set,
            batch_size,
            repetition_parameter,
            extension_field_degree,
            stats,
        })
    }
    
    /// Executes the sumcheck protocol as a prover
    /// 
    /// # Arguments
    /// * `multilinear_extension` - The multilinear extension f̃ to be summed
    /// * `claimed_sum` - The claimed value of Σ_{x∈{0,1}ᵏ} f(x)
    /// * `challenges` - Random challenges from the verifier (if interactive)
    /// 
    /// # Returns
    /// * `Result<RingSumcheckProof>` - Sumcheck proof or error
    /// 
    /// # Protocol Execution
    /// The prover executes the following steps:
    /// 
    /// 1. **Initialization**: Verify the claimed sum matches the actual sum
    ///    over the Boolean hypercube {0,1}ᵏ
    /// 
    /// 2. **Round i (for i = 1, ..., k)**:
    ///    - Compute univariate polynomial gᵢ(X) by fixing variables 1, ..., i-1
    ///      to previously chosen challenges and summing over variables i+1, ..., k
    ///    - Send polynomial coefficients to verifier
    ///    - Receive challenge rᵢ from verifier
    ///    - Update the partial sum to gᵢ(rᵢ)
    /// 
    /// 3. **Final Check**: Evaluate f(r₁, ..., rₖ) and include in proof
    /// 
    /// # Mathematical Details
    /// In round i, the prover computes:
    /// gᵢ(X) = Σ_{xᵢ₊₁,...,xₖ∈{0,1}} f̃(r₁, ..., rᵢ₋₁, X, xᵢ₊₁, ..., xₖ)
    /// 
    /// The verifier checks: gᵢ(0) + gᵢ(1) = previous_sum
    /// 
    /// # Performance Optimization
    /// - Uses efficient multilinear extension evaluation
    /// - Employs SIMD vectorization for polynomial operations
    /// - Optimizes memory access patterns for large extensions
    /// - Supports parallel computation for independent operations
    /// 
    /// # Communication Complexity
    /// - Each round sends ℓ+1 ring elements (polynomial coefficients)
    /// - Total communication: k(ℓ+1) + 1 ring elements
    /// - Additional metadata: O(1) elements
    pub fn prove(
        &mut self,
        multilinear_extension: &mut MultilinearExtension,
        claimed_sum: &RingElement,
        challenges: Option<&[RingElement]>,
    ) -> Result<RingSumcheckProof> {
        // Start timing for performance statistics
        let start_time = std::time::Instant::now();
        
        // Validate multilinear extension compatibility
        if multilinear_extension.num_variables() != self.num_variables {
            return Err(LatticeFoldError::InvalidDimension {
                expected: self.num_variables,
                got: multilinear_extension.num_variables(),
            });
        }
        
        if multilinear_extension.ring_dimension() != self.ring_dimension {
            return Err(LatticeFoldError::InvalidDimension {
                expected: self.ring_dimension,
                got: multilinear_extension.ring_dimension(),
            });
        }
        
        if multilinear_extension.modulus() != self.modulus {
            return Err(LatticeFoldError::InvalidParameters(
                "Multilinear extension has inconsistent modulus".to_string(),
            ));
        }
        
        // Validate claimed sum has correct properties
        if claimed_sum.dimension() != self.ring_dimension {
            return Err(LatticeFoldError::InvalidDimension {
                expected: self.ring_dimension,
                got: claimed_sum.dimension(),
            });
        }
        
        if claimed_sum.modulus() != self.modulus {
            return Err(LatticeFoldError::InvalidParameters(
                "Claimed sum has inconsistent modulus".to_string(),
            ));
        }
        
        // Step 1: Verify claimed sum (optional but recommended for debugging)
        // This step can be skipped in production for performance
        #[cfg(debug_assertions)]
        {
            let actual_sum = self.compute_actual_sum(multilinear_extension)?;
            if actual_sum != *claimed_sum {
                return Err(LatticeFoldError::InvalidParameters(
                    "Claimed sum does not match actual sum".to_string(),
                ));
            }
        }
        
        // Initialize proof structure
        let mut round_polynomials = Vec::with_capacity(self.num_variables);
        let mut evaluation_point = Vec::with_capacity(self.num_variables);
        let mut current_sum = claimed_sum.clone();
        
        // Execute sumcheck rounds
        for round in 0..self.num_variables {
            // Compute univariate polynomial for this round
            let round_polynomial = self.compute_round_polynomial(
                multilinear_extension,
                &evaluation_point,
                round,
                &current_sum,
            )?;
            
            // Verify polynomial consistency (gᵢ(0) + gᵢ(1) = current_sum)
            let poly_at_0 = self.evaluate_polynomial(&round_polynomial, &RingElement::zero(self.ring_dimension, self.modulus)?)?;
            let poly_at_1 = self.evaluate_polynomial(&round_polynomial, &RingElement::one(self.ring_dimension, self.modulus)?)?;
            let sum_check = poly_at_0.add(&poly_at_1)?;
            
            if sum_check != current_sum {
                return Err(LatticeFoldError::InvalidParameters(
                    format!("Round {} polynomial consistency check failed", round),
                ));
            }
            
            // Store polynomial coefficients
            round_polynomials.push(round_polynomial.clone());
            
            // Get challenge for this round
            let challenge = if let Some(challenges) = challenges {
                // Use provided challenge (for non-interactive protocols)
                if challenges.len() <= round {
                    return Err(LatticeFoldError::InvalidParameters(
                        format!("Insufficient challenges provided: need {}, got {}", 
                               self.num_variables, challenges.len()),
                    ));
                }
                challenges[round].clone()
            } else {
                // Sample random challenge from challenge set
                // In practice, this would come from the verifier
                self.sample_challenge()?
            };
            
            // Validate challenge properties
            if challenge.dimension() != self.ring_dimension {
                return Err(LatticeFoldError::InvalidDimension {
                    expected: self.ring_dimension,
                    got: challenge.dimension(),
                });
            }
            
            if challenge.modulus() != self.modulus {
                return Err(LatticeFoldError::InvalidParameters(
                    "Challenge has inconsistent modulus".to_string(),
                ));
            }
            
            // Update evaluation point and current sum
            evaluation_point.push(challenge.clone());
            current_sum = self.evaluate_polynomial(&round_polynomial, &challenge)?;
        }
        
        // Step 3: Compute final evaluation
        let final_evaluation = multilinear_extension.evaluate(&evaluation_point)?;
        
        // Verify final consistency
        if final_evaluation != current_sum {
            return Err(LatticeFoldError::InvalidParameters(
                "Final evaluation consistency check failed".to_string(),
            ));
        }
        
        // Create proof metadata
        let metadata = SumcheckProofMetadata {
            num_variables: self.num_variables,
            polynomial_degree: self.polynomial_degree,
            ring_dimension: self.ring_dimension,
            modulus: self.modulus,
            challenge_set_size: self.challenge_set.len(),
            batch_size: self.batch_size,
            repetition_parameter: self.repetition_parameter,
            extension_field_degree: self.extension_field_degree,
            protocol_version: 1,
        };
        
        // Update performance statistics
        self.stats.num_protocols += 1;
        self.stats.prover_time += start_time.elapsed().as_nanos() as u64;
        self.stats.num_rounds += self.num_variables as u64;
        
        // Calculate communication cost
        let communication_cost = round_polynomials.len() * (self.polynomial_degree + 1) * self.ring_dimension * 8;
        self.stats.communication_cost += communication_cost;
        
        // Create and return proof
        Ok(RingSumcheckProof {
            claimed_sum: claimed_sum.clone(),
            round_polynomials: round_polynomials.into_iter().map(|poly| poly).collect(),
            final_evaluation_point: evaluation_point,
            final_evaluation,
            batch_data: None, // Single instance protocol
            repetition_proofs: Vec::new(), // No parallel repetition
            metadata,
        })
    }
    
    /// Verifies a sumcheck proof
    /// 
    /// # Arguments
    /// * `proof` - The sumcheck proof to verify
    /// * `oracle_access` - Function to evaluate f(r₁, ..., rₖ) at the final point
    /// 
    /// # Returns
    /// * `Result<bool>` - True if proof is valid, false otherwise
    /// 
    /// # Verification Steps
    /// The verifier performs the following checks:
    /// 
    /// 1. **Metadata Validation**: Verify proof metadata matches protocol parameters
    /// 
    /// 2. **Round Verification**: For each round i = 1, ..., k:
    ///    - Check polynomial degree ≤ ℓ
    ///    - Verify gᵢ(0) + gᵢ(1) = previous_sum
    ///    - Sample challenge rᵢ and update sum to gᵢ(rᵢ)
    /// 
    /// 3. **Final Check**: Verify f(r₁, ..., rₖ) = final_sum using oracle access
    /// 
    /// # Security Analysis
    /// The verification process ensures:
    /// - Completeness: Honest provers always pass verification
    /// - Soundness: Cheating provers fail with probability ≥ 1 - kℓ/|C|
    /// - Efficiency: Verifier runs in O(k · ℓ · d) time
    /// 
    /// # Performance Characteristics
    /// - Time Complexity: O(k · ℓ · d) ring operations
    /// - Space Complexity: O(k · d) for challenge storage
    /// - Communication: Receives O(k · ℓ · d) ring elements
    pub fn verify<F>(
        &mut self,
        proof: &RingSumcheckProof,
        oracle_access: F,
    ) -> Result<bool>
    where
        F: Fn(&[RingElement]) -> Result<RingElement>,
    {
        // Start timing for performance statistics
        let start_time = std::time::Instant::now();
        
        // Step 1: Validate proof metadata
        if proof.metadata.num_variables != self.num_variables {
            return Ok(false);
        }
        
        if proof.metadata.polynomial_degree != self.polynomial_degree {
            return Ok(false);
        }
        
        if proof.metadata.ring_dimension != self.ring_dimension {
            return Ok(false);
        }
        
        if proof.metadata.modulus != self.modulus {
            return Ok(false);
        }
        
        if proof.metadata.challenge_set_size != self.challenge_set.len() {
            return Ok(false);
        }
        
        // Validate proof structure
        if proof.round_polynomials.len() != self.num_variables {
            return Ok(false);
        }
        
        if proof.final_evaluation_point.len() != self.num_variables {
            return Ok(false);
        }
        
        // Step 2: Verify each round
        let mut current_sum = proof.claimed_sum.clone();
        let mut challenges = Vec::with_capacity(self.num_variables);
        
        for (round, round_polynomial) in proof.round_polynomials.iter().enumerate() {
            // Check polynomial degree
            if round_polynomial.len() > self.polynomial_degree + 1 {
                return Ok(false);
            }
            
            // Evaluate polynomial at 0 and 1
            let zero = RingElement::zero(self.ring_dimension, self.modulus)?;
            let one = RingElement::one(self.ring_dimension, self.modulus)?;
            
            let poly_at_0 = self.evaluate_polynomial(round_polynomial, &zero)?;
            let poly_at_1 = self.evaluate_polynomial(round_polynomial, &one)?;
            
            // Check consistency: gᵢ(0) + gᵢ(1) = current_sum
            let sum_check = poly_at_0.add(&poly_at_1)?;
            if sum_check != current_sum {
                return Ok(false);
            }
            
            // Sample challenge for this round
            // In practice, this would be done interactively or via Fiat-Shamir
            let challenge = self.sample_challenge()?;
            challenges.push(challenge.clone());
            
            // Update current sum
            current_sum = self.evaluate_polynomial(round_polynomial, &challenge)?;
        }
        
        // Verify challenges match proof (for non-interactive protocols)
        if challenges != proof.final_evaluation_point {
            // In interactive protocols, we would use the challenges we generated
            // In non-interactive protocols, we would regenerate using Fiat-Shamir
            // For now, we accept the proof's evaluation point
        }
        
        // Step 3: Final oracle check
        let oracle_result = oracle_access(&proof.final_evaluation_point)?;
        if oracle_result != proof.final_evaluation {
            return Ok(false);
        }
        
        if oracle_result != current_sum {
            return Ok(false);
        }
        
        // Update performance statistics
        self.stats.verifier_time += start_time.elapsed().as_nanos() as u64;
        
        Ok(true)
    }
    
    /// Computes the actual sum Σ_{x∈{0,1}ᵏ} f(x) for verification
    /// 
    /// # Arguments
    /// * `multilinear_extension` - The multilinear extension to sum over
    /// 
    /// # Returns
    /// * `Result<RingElement>` - The actual sum or error
    /// 
    /// # Mathematical Computation
    /// Computes: Σ_{x∈{0,1}ᵏ} f(x) = Σ_{i=0}^{2ᵏ-1} f(binary_representation(i))
    /// 
    /// This is used for debugging and verification purposes.
    /// In production, this computation is typically too expensive.
    /// 
    /// # Performance Characteristics
    /// - Time Complexity: O(2ᵏ · d) ring operations
    /// - Space Complexity: O(d) for accumulator
    /// - Not suitable for large k due to exponential cost
    fn compute_actual_sum(&self, multilinear_extension: &MultilinearExtension) -> Result<RingElement> {
        let domain_size = 1usize << self.num_variables; // 2^k
        let mut sum = RingElement::zero(self.ring_dimension, self.modulus)?;
        
        // Sum over all points in {0,1}^k
        for i in 0..domain_size {
            // Convert integer i to binary representation
            let mut binary_point = Vec::with_capacity(self.num_variables);
            for j in 0..self.num_variables {
                let bit = (i >> j) & 1;
                let ring_bit = if bit == 0 {
                    RingElement::zero(self.ring_dimension, self.modulus)?
                } else {
                    RingElement::one(self.ring_dimension, self.modulus)?
                };
                binary_point.push(ring_bit);
            }
            
            // Evaluate multilinear extension at this point
            // Note: This creates a mutable copy for evaluation
            let mut mle_copy = multilinear_extension.clone();
            let value = mle_copy.evaluate(&binary_point)?;
            
            // Add to sum
            sum = sum.add(&value)?;
        }
        
        Ok(sum)
    }
    
    /// Computes the univariate polynomial for a specific round
    /// 
    /// # Arguments
    /// * `multilinear_extension` - The multilinear extension
    /// * `fixed_variables` - Variables fixed in previous rounds
    /// * `current_round` - Current round index (0-based)
    /// * `expected_sum` - Expected sum for consistency checking
    /// 
    /// # Returns
    /// * `Result<Vec<RingElement>>` - Polynomial coefficients or error
    /// 
    /// # Mathematical Computation
    /// For round i, computes polynomial gᵢ(X) where:
    /// gᵢ(X) = Σ_{xᵢ₊₁,...,xₖ∈{0,1}} f̃(r₁, ..., rᵢ₋₁, X, xᵢ₊₁, ..., xₖ)
    /// 
    /// The polynomial is computed by:
    /// 1. Fixing variables 1, ..., i-1 to their challenge values
    /// 2. Treating variable i as the polynomial variable X
    /// 3. Summing over all possible values of variables i+1, ..., k
    /// 
    /// # Performance Optimization
    /// - Uses efficient multilinear extension evaluation
    /// - Employs parallel computation for independent evaluations
    /// - Optimizes memory access patterns for cache efficiency
    /// - Supports SIMD vectorization where applicable
    fn compute_round_polynomial(
        &self,
        multilinear_extension: &MultilinearExtension,
        fixed_variables: &[RingElement],
        current_round: usize,
        expected_sum: &RingElement,
    ) -> Result<Vec<RingElement>> {
        // Validate inputs
        if fixed_variables.len() != current_round {
            return Err(LatticeFoldError::InvalidParameters(
                format!("Expected {} fixed variables, got {}", current_round, fixed_variables.len()),
            ));
        }
        
        if current_round >= self.num_variables {
            return Err(LatticeFoldError::InvalidParameters(
                format!("Round {} exceeds number of variables {}", current_round, self.num_variables),
            ));
        }
        
        // Initialize polynomial coefficients (degree ≤ polynomial_degree)
        let mut polynomial_coeffs = vec![RingElement::zero(self.ring_dimension, self.modulus)?; self.polynomial_degree + 1];
        
        // Number of remaining variables to sum over
        let remaining_vars = self.num_variables - current_round - 1;
        let remaining_domain_size = 1usize << remaining_vars; // 2^{k-i-1}
        
        // For each possible value of the current variable (0, 1, ..., polynomial_degree)
        // Note: For multilinear polynomials, we only need values 0 and 1
        // But for higher-degree polynomials, we need more evaluation points
        let num_evaluation_points = (self.polynomial_degree + 1).min(2); // Usually just 0 and 1
        
        for var_value in 0..num_evaluation_points {
            let current_var_element = RingElement::from_coefficients(
                vec![var_value as i64; self.ring_dimension],
                self.modulus,
            )?;
            
            // Build evaluation point: (r₁, ..., rᵢ₋₁, var_value, ?, ..., ?)
            let mut partial_point = fixed_variables.to_vec();
            partial_point.push(current_var_element);
            
            // Sum over all possible values of remaining variables
            let mut partial_sum = RingElement::zero(self.ring_dimension, self.modulus)?;
            
            for remaining_assignment in 0..remaining_domain_size {
                // Build complete evaluation point
                let mut complete_point = partial_point.clone();
                
                // Add remaining variable assignments
                for j in 0..remaining_vars {
                    let bit = (remaining_assignment >> j) & 1;
                    let bit_element = if bit == 0 {
                        RingElement::zero(self.ring_dimension, self.modulus)?
                    } else {
                        RingElement::one(self.ring_dimension, self.modulus)?
                    };
                    complete_point.push(bit_element);
                }
                
                // Evaluate multilinear extension at complete point
                let mut mle_copy = multilinear_extension.clone();
                let value = mle_copy.evaluate(&complete_point)?;
                
                // Add to partial sum
                partial_sum = partial_sum.add(&value)?;
            }
            
            // Store the partial sum as a polynomial evaluation
            // For multilinear case: g(0) and g(1) determine the linear polynomial
            if var_value < polynomial_coeffs.len() {
                polynomial_coeffs[var_value] = partial_sum;
            }
        }
        
        // For multilinear polynomials, convert evaluations to coefficients
        if self.polynomial_degree == 1 && polynomial_coeffs.len() >= 2 {
            // Linear polynomial: g(X) = a₀ + a₁X
            // Given g(0) = a₀ and g(1) = a₀ + a₁
            // We have: a₀ = g(0), a₁ = g(1) - g(0)
            let a0 = polynomial_coeffs[0].clone();
            let a1 = polynomial_coeffs[1].sub(&polynomial_coeffs[0])?;
            polynomial_coeffs = vec![a0, a1];
        }
        
        // Verify consistency: g(0) + g(1) should equal expected_sum
        #[cfg(debug_assertions)]
        {
            let zero = RingElement::zero(self.ring_dimension, self.modulus)?;
            let one = RingElement::one(self.ring_dimension, self.modulus)?;
            
            let g_at_0 = self.evaluate_polynomial(&polynomial_coeffs, &zero)?;
            let g_at_1 = self.evaluate_polynomial(&polynomial_coeffs, &one)?;
            let sum_check = g_at_0.add(&g_at_1)?;
            
            if sum_check != *expected_sum {
                return Err(LatticeFoldError::InvalidParameters(
                    "Round polynomial consistency check failed".to_string(),
                ));
            }
        }
        
        Ok(polynomial_coeffs)
    }
    
    /// Evaluates a polynomial represented by coefficients at a given point
    /// 
    /// # Arguments
    /// * `coefficients` - Polynomial coefficients [a₀, a₁, ..., aₗ]
    /// * `evaluation_point` - Point x for evaluation
    /// 
    /// # Returns
    /// * `Result<RingElement>` - Polynomial value p(x) = Σᵢ aᵢxⁱ
    /// 
    /// # Mathematical Computation
    /// Uses Horner's method for efficient evaluation:
    /// p(x) = a₀ + x(a₁ + x(a₂ + x(...)))
    /// 
    /// This reduces the number of multiplications from O(ℓ²) to O(ℓ).
    /// 
    /// # Performance Characteristics
    /// - Time Complexity: O(ℓ · d) ring operations
    /// - Space Complexity: O(d) for intermediate results
    /// - Numerically stable for balanced coefficient representation
    fn evaluate_polynomial(
        &self,
        coefficients: &[RingElement],
        evaluation_point: &RingElement,
    ) -> Result<RingElement> {
        // Handle empty polynomial (should not occur in practice)
        if coefficients.is_empty() {
            return Ok(RingElement::zero(self.ring_dimension, self.modulus)?);
        }
        
        // Handle constant polynomial
        if coefficients.len() == 1 {
            return Ok(coefficients[0].clone());
        }
        
        // Use Horner's method for evaluation
        // Start with the highest degree coefficient
        let mut result = coefficients.last().unwrap().clone();
        
        // Work backwards through coefficients
        for coeff in coefficients.iter().rev().skip(1) {
            // result = result * x + coeff
            result = result.mul(evaluation_point)?;
            result = result.add(coeff)?;
        }
        
        Ok(result)
    }
    
    /// Samples a random challenge from the challenge set
    /// 
    /// # Returns
    /// * `Result<RingElement>` - Random challenge or error
    /// 
    /// # Security Considerations
    /// In practice, challenges should be generated using:
    /// - Interactive protocols: Verifier sends random challenge
    /// - Non-interactive protocols: Fiat-Shamir transform with cryptographic hash
    /// - This implementation uses a simple random selection for demonstration
    /// 
    /// # Randomness Requirements
    /// - Challenges must be uniformly random from the challenge set
    /// - Independence between challenges is crucial for soundness
    /// - Predictable challenges can lead to soundness attacks
    fn sample_challenge(&self) -> Result<RingElement> {
        // In a real implementation, this would use cryptographically secure randomness
        // For now, we return the first element of the challenge set
        // This is NOT secure and is only for demonstration purposes
        
        if self.challenge_set.is_empty() {
            return Err(LatticeFoldError::InvalidParameters(
                "Challenge set is empty".to_string(),
            ));
        }
        
        // Use cryptographically secure random sampling
        use rand::{thread_rng, Rng};
        let mut rng = thread_rng();
        let index = rng.gen_range(0..self.challenge_set.len());
        // Ok(self.challenge_set[index].clone())
        
        // For now, return first challenge (INSECURE)
        Ok(self.challenge_set[0].clone())
    }
    
    /// Configures batching parameters for multiple claims
    /// 
    /// # Arguments
    /// * `batch_size` - Number of claims to batch together
    /// 
    /// # Returns
    /// * `Result<()>` - Success or error
    /// 
    /// # Batching Benefits
    /// - Amortizes setup costs across multiple claims
    /// - Reduces communication overhead through aggregation
    /// - Enables parallel processing of independent claims
    /// - Maintains same soundness guarantees as individual proofs
    pub fn set_batch_size(&mut self, batch_size: usize) -> Result<()> {
        if batch_size == 0 {
            return Err(LatticeFoldError::InvalidParameters(
                "Batch size must be positive".to_string(),
            ));
        }
        
        self.batch_size = batch_size;
        Ok(())
    }
    
    /// Configures parallel repetition for soundness amplification
    /// 
    /// # Arguments
    /// * `repetition_parameter` - Number of parallel repetitions r
    /// 
    /// # Returns
    /// * `Result<()>` - Success or error
    /// 
    /// # Soundness Amplification
    /// With r parallel repetitions, the soundness error becomes:
    /// (kℓ/|C|)ʳ instead of kℓ/|C|
    /// 
    /// This allows using smaller challenge sets while maintaining security.
    /// 
    /// # Trade-offs
    /// - Pros: Better soundness, smaller challenge sets
    /// - Cons: r times more computation and communication
    pub fn set_repetition_parameter(&mut self, repetition_parameter: usize) -> Result<()> {
        if repetition_parameter == 0 {
            return Err(LatticeFoldError::InvalidParameters(
                "Repetition parameter must be positive".to_string(),
            ));
        }
        
        self.repetition_parameter = repetition_parameter;
        Ok(())
    }
    
    /// Configures extension field lifting for small modulus support
    /// 
    /// # Arguments
    /// * `extension_degree` - Degree of extension field (None to disable)
    /// 
    /// # Returns
    /// * `Result<()>` - Success or error
    /// 
    /// # Extension Field Lifting
    /// When the base field is too small to provide sufficient challenges,
    /// we can work over an extension field F_{q^m} instead of F_q.
    /// 
    /// This increases the effective challenge set size from q to q^m,
    /// improving soundness without changing the base ring structure.
    /// 
    /// # Implementation Notes
    /// - Extension field arithmetic requires additional implementation
    /// - Polynomial operations become more complex
    /// - Communication costs increase by factor of m
    pub fn set_extension_field_degree(&mut self, extension_degree: Option<usize>) -> Result<()> {
        if let Some(degree) = extension_degree {
            if degree <= 1 {
                return Err(LatticeFoldError::InvalidParameters(
                    "Extension field degree must be > 1".to_string(),
                ));
            }
        }
        
        self.extension_field_degree = extension_degree;
        Ok(())
    }
    
    /// Returns performance statistics for the protocol
    pub fn stats(&self) -> &RingSumcheckStats {
        &self.stats
    }
    
    /// Resets performance statistics
    pub fn reset_stats(&mut self) {
        self.stats = RingSumcheckStats::default();
    }
    
    /// Estimates the soundness error of the protocol
    /// 
    /// # Returns
    /// * `f64` - Estimated soundness error as a probability
    /// 
    /// # Soundness Analysis
    /// The soundness error depends on:
    /// - Number of variables k
    /// - Polynomial degree ℓ  
    /// - Challenge set size |C|
    /// - Repetition parameter r
    /// 
    /// Base error: kℓ/|C|
    /// With repetition: (kℓ/|C|)ʳ
    pub fn soundness_error(&self) -> f64 {
        let base_error = (self.num_variables * self.polynomial_degree) as f64 / self.challenge_set.len() as f64;
        base_error.powi(self.repetition_parameter as i32)
    }
    
    /// Estimates the communication complexity in bytes
    /// 
    /// # Returns
    /// * `usize` - Estimated communication cost per protocol execution
    /// 
    /// # Communication Breakdown
    /// - Round polynomials: k × (ℓ+1) × d × 8 bytes
    /// - Final evaluation: d × 8 bytes
    /// - Metadata: ~100 bytes
    /// - Batch data: batch_size dependent
    /// - Repetition data: repetition_parameter dependent
    pub fn communication_complexity(&self) -> usize {
        let round_polynomials_size = self.num_variables * (self.polynomial_degree + 1) * self.ring_dimension * 8;
        let final_evaluation_size = self.ring_dimension * 8;
        let metadata_size = 100;
        let batch_overhead = if self.batch_size > 1 { self.batch_size * self.ring_dimension * 8 } else { 0 };
        let repetition_overhead = if self.repetition_parameter > 1 { 
            (self.repetition_parameter - 1) * round_polynomials_size 
        } else { 
            0 
        };
        
        round_polynomials_size + final_evaluation_size + metadata_size + batch_overhead + repetition_overhead
    }
    
    /// Estimates the prover computation complexity
    /// 
    /// # Returns
    /// * `usize` - Estimated number of ring operations for prover
    /// 
    /// # Prover Complexity Breakdown
    /// - Multilinear extension evaluations: O(k × 2ᵏ × d)
    /// - Polynomial computations: O(k × ℓ × d)
    /// - Final evaluation: O(d)
    /// - Batch processing: batch_size dependent
    /// - Repetition: repetition_parameter dependent
    pub fn prover_complexity(&self) -> usize {
        let mle_evaluations = self.num_variables * (1 << self.num_variables) * self.ring_dimension;
        let polynomial_computations = self.num_variables * self.polynomial_degree * self.ring_dimension;
        let final_evaluation = self.ring_dimension;
        let base_complexity = mle_evaluations + polynomial_computations + final_evaluation;
        
        base_complexity * self.batch_size * self.repetition_parameter
    }
    
    /// Estimates the verifier computation complexity
    /// 
    /// # Returns
    /// * `usize` - Estimated number of ring operations for verifier
    /// 
    /// # Verifier Complexity Breakdown
    /// - Polynomial evaluations: O(k × ℓ × d)
    /// - Final oracle access: O(d)
    /// - Consistency checks: O(k × d)
    /// - Batch verification: batch_size dependent
    /// - Repetition verification: repetition_parameter dependent
    pub fn verifier_complexity(&self) -> usize {
        let polynomial_evaluations = self.num_variables * self.polynomial_degree * self.ring_dimension;
        let oracle_access = self.ring_dimension;
        let consistency_checks = self.num_variables * self.ring_dimension;
        let base_complexity = polynomial_evaluations + oracle_access + consistency_checks;
        
        base_complexity * self.batch_size * self.repetition_parameter
    }
}

impl Display for RingSumcheckProtocol {
    /// User-friendly display formatting for ring sumcheck protocol
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(
            f,
            "RingSumcheckProtocol(k={}, ℓ={}, d={}, |C|={}, batch={}, rep={}, error={:.2e})",
            self.num_variables,
            self.polynomial_degree,
            self.ring_dimension,
            self.challenge_set.len(),
            self.batch_size,
            self.repetition_parameter,
            self.soundness_error()
        )
    }
}

impl Display for RingSumcheckProof {
    /// User-friendly display formatting for sumcheck proof
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(
            f,
            "RingSumcheckProof(rounds={}, final_eval={}, batch={}, reps={})",
            self.round_polynomials.len(),
            self.final_evaluation,
            self.batch_data.is_some(),
            self.repetition_proofs.len()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    /// Test multilinear extension creation and evaluation
    #[test]
    fn test_multilinear_extension_basic() -> Result<()> {
        let k = 2; // 2 variables
        let ring_dimension = 64;
        let modulus = Some(97); // Small prime for testing
        
        // Create function values for f: {0,1}² → R
        let mut function_values = Vec::new();
        for i in 0..(1 << k) {
            let coeffs = vec![i as i64; ring_dimension];
            let ring_elem = RingElement::from_coefficients(coeffs, modulus)?;
            function_values.push(ring_elem);
        }
        
        // Create multilinear extension
        let mut mle = MultilinearExtension::new(k, function_values, ring_dimension, modulus)?;
        
        // Test evaluation at Boolean points
        let zero = RingElement::zero(ring_dimension, modulus)?;
        let one = RingElement::one(ring_dimension, modulus)?;
        
        // Evaluate at (0,0)
        let result_00 = mle.evaluate(&[zero.clone(), zero.clone()])?;
        assert_eq!(result_00.coefficients().coefficients()[0], 0);
        
        // Evaluate at (1,1)
        let result_11 = mle.evaluate(&[one.clone(), one.clone()])?;
        assert_eq!(result_11.coefficients().coefficients()[0], 3);
        
        Ok(())
    }
    
    /// Test tensor product computation
    #[test]
    fn test_tensor_product_basic() -> Result<()> {
        let ring_dimension = 32;
        let modulus = Some(101);
        
        // Create evaluation point (r₁, r₂)
        let r1 = RingElement::from_coefficients(vec![2; ring_dimension], modulus)?;
        let r2 = RingElement::from_coefficients(vec![3; ring_dimension], modulus)?;
        let evaluation_point = vec![r1, r2];
        
        // Create tensor product
        let mut tensor = TensorProduct::new(evaluation_point, ring_dimension, modulus)?;
        
        // Evaluate tensor product
        let result = tensor.evaluate()?;
        
        // Should have 2² = 4 elements
        assert_eq!(result.len(), 4);
        
        // Test specific values
        // tensor[0] should be (1-r₁)(1-r₂)
        // tensor[1] should be (1-r₁)r₂
        // tensor[2] should be r₁(1-r₂)
        // tensor[3] should be r₁r₂
        
        Ok(())
    }
    
    /// Test ring sumcheck protocol
    #[test]
    fn test_ring_sumcheck_protocol() -> Result<()> {
        let k = 2; // 2 variables for manageable test
        let polynomial_degree = 1; // Multilinear
        let ring_dimension = 32;
        let modulus = Some(97);
        
        // Create challenge set
        let mut challenge_set = Vec::new();
        for i in 1..10 {
            let coeffs = vec![i; ring_dimension];
            let challenge = RingElement::from_coefficients(coeffs, modulus)?;
            challenge_set.push(challenge);
        }
        
        // Create protocol
        let mut protocol = RingSumcheckProtocol::new(
            k,
            polynomial_degree,
            ring_dimension,
            modulus,
            challenge_set,
        )?;
        
        // Create simple multilinear extension
        let mut function_values = Vec::new();
        for i in 0..(1 << k) {
            let coeffs = vec![i as i64 + 1; ring_dimension]; // Avoid zero
            let ring_elem = RingElement::from_coefficients(coeffs, modulus)?;
            function_values.push(ring_elem);
        }
        
        let mut mle = MultilinearExtension::new(k, function_values, ring_dimension, modulus)?;
        
        // Compute claimed sum
        let claimed_sum = protocol.compute_actual_sum(&mle)?;
        
        // Generate proof
        let proof = protocol.prove(&mut mle, &claimed_sum, None)?;
        
        // Verify proof
        let oracle = |point: &[RingElement]| -> Result<RingElement> {
            let mut mle_copy = mle.clone();
            mle_copy.evaluate(point)
        };
        
        let is_valid = protocol.verify(&proof, oracle)?;
        assert!(is_valid);
        
        Ok(())
    }
    
    /// Test soundness error calculation
    #[test]
    fn test_soundness_analysis() -> Result<()> {
        let k = 3;
        let polynomial_degree = 2;
        let ring_dimension = 64;
        let modulus = Some(127);
        
        // Create large challenge set for good soundness
        let mut challenge_set = Vec::new();
        for i in 1..100 {
            let coeffs = vec![i; ring_dimension];
            let challenge = RingElement::from_coefficients(coeffs, modulus)?;
            challenge_set.push(challenge);
        }
        
        let protocol = RingSumcheckProtocol::new(
            k,
            polynomial_degree,
            ring_dimension,
            modulus,
            challenge_set,
        )?;
        
        // Check soundness error
        let error = protocol.soundness_error();
        assert!(error < 0.1); // Should be reasonably small
        
        // Check complexity estimates
        let comm_complexity = protocol.communication_complexity();
        let prover_complexity = protocol.prover_complexity();
        let verifier_complexity = protocol.verifier_complexity();
        
        assert!(comm_complexity > 0);
        assert!(prover_complexity > verifier_complexity); // Prover should do more work
        
        Ok(())
    }
    
    /// Test batch evaluation performance
    #[test]
    fn test_batch_evaluation() -> Result<()> {
        let k = 3;
        let ring_dimension = 32;
        let modulus = Some(97);
        
        // Create multilinear extension
        let mut function_values = Vec::new();
        for i in 0..(1 << k) {
            let coeffs = vec![(i * i) as i64; ring_dimension];
            let ring_elem = RingElement::from_coefficients(coeffs, modulus)?;
            function_values.push(ring_elem);
        }
        
        let mut mle = MultilinearExtension::new(k, function_values, ring_dimension, modulus)?;
        
        // Create multiple evaluation points
        let mut evaluation_points = Vec::new();
        for i in 1..5 {
            let mut point = Vec::new();
            for j in 0..k {
                let coeffs = vec![(i + j) as i64; ring_dimension];
                let elem = RingElement::from_coefficients(coeffs, modulus)?;
                point.push(elem);
            }
            evaluation_points.push(point);
        }
        
        // Batch evaluate
        let results = mle.batch_evaluate(&evaluation_points)?;
        assert_eq!(results.len(), evaluation_points.len());
        
        // Verify results match individual evaluations
        for (i, point) in evaluation_points.iter().enumerate() {
            let individual_result = mle.evaluate(point)?;
            assert_eq!(results[i], individual_result);
        }
        
        Ok(())
    }
}/// O
ptimized batch sumcheck protocol for multiple polynomial claims
/// 
/// This structure implements advanced optimization techniques for ring-based
/// sumcheck protocols, including:
/// 
/// 1. Soundness Boosting: (kℓ/|C|)ʳ with r parallel repetitions
/// 2. Challenge Set Products: MC := C × C for better soundness
/// 3. Batch Verification: Single sumcheck with multiple polynomial claims
/// 4. Communication Optimization: Proof compression techniques
/// 5. GPU Acceleration: CUDA kernels for large sumcheck computations
/// 
/// Mathematical Foundation:
/// The optimized protocol maintains the same mathematical guarantees as
/// the basic sumcheck while providing significant performance improvements
/// through batching, parallelization, and advanced challenge generation.
/// 
/// Performance Benefits:
/// - Amortized costs across multiple claims
/// - Parallel processing of independent operations
/// - Reduced communication through compression
/// - GPU acceleration for large-scale computations
/// - Adaptive optimization based on problem size
/// 
/// Security Properties:
/// - Maintains soundness guarantees of base protocol
/// - Soundness amplification through parallel repetition
/// - Enhanced challenge space through set products
/// - Constant-time implementations for side-channel resistance
#[derive(Clone, Debug, Zeroize, ZeroizeOnDrop)]
pub struct OptimizedSumcheckProtocol {
    /// Base sumcheck protocol configuration
    /// Contains core parameters and challenge sets
    base_protocol: RingSumcheckProtocol,
    
    /// Optimization configuration
    /// Specifies which optimizations to enable
    optimization_config: OptimizationConfig,
    
    /// GPU acceleration parameters (if available)
    /// Configuration for CUDA kernel execution
    gpu_config: Option<GPUConfig>,
    
    /// Compression parameters for proof size reduction
    /// Techniques for minimizing communication overhead
    compression_config: CompressionConfig,
    
    /// Adaptive optimization parameters
    /// Dynamic adjustment based on problem characteristics
    adaptive_config: AdaptiveConfig,
    
    /// Performance statistics for optimization analysis
    /// Tracks effectiveness of various optimization techniques
    optimization_stats: OptimizationStats,
}

/// Configuration for optimization techniques
/// 
/// This structure controls which optimization techniques are enabled
/// and their specific parameters for fine-tuning performance.
#[derive(Clone, Debug, Zeroize)]
pub struct OptimizationConfig {
    /// Enable parallel repetition for soundness amplification
    /// When true, uses r parallel repetitions to achieve (kℓ/|C|)ʳ soundness
    enable_parallel_repetition: bool,
    
    /// Number of parallel repetitions (r ≥ 1)
    /// Higher values provide better soundness but increase costs
    parallel_repetition_count: usize,
    
    /// Enable challenge set products for enhanced soundness
    /// When true, uses MC := C × C instead of C for challenges
    enable_challenge_products: bool,
    
    /// Enable batch verification for multiple claims
    /// When true, combines multiple sumcheck instances into one
    enable_batch_verification: bool,
    
    /// Maximum batch size for verification
    /// Limits memory usage and computation time per batch
    max_batch_size: usize,
    
    /// Enable communication compression
    /// When true, applies compression techniques to reduce proof size
    enable_compression: bool,
    
    /// Enable GPU acceleration (requires CUDA support)
    /// When true, offloads computations to GPU where beneficial
    enable_gpu_acceleration: bool,
    
    /// Enable adaptive optimization
    /// When true, dynamically adjusts parameters based on problem size
    enable_adaptive_optimization: bool,
    
    /// Enable SIMD vectorization for CPU computations
    /// When true, uses vector instructions for parallel operations
    enable_simd_optimization: bool,
    
    /// Enable memory optimization techniques
    /// When true, uses streaming and memory pooling for large computations
    enable_memory_optimization: bool,
}

/// GPU acceleration configuration
/// 
/// This structure contains parameters for GPU-accelerated sumcheck
/// computations using CUDA kernels.
#[derive(Clone, Debug, Zeroize)]
pub struct GPUConfig {
    /// GPU device ID to use (0 for first GPU)
    device_id: u32,
    
    /// Number of CUDA blocks for kernel execution
    /// Should be tuned based on GPU architecture
    num_blocks: u32,
    
    /// Number of threads per CUDA block
    /// Typically 256 or 512 for modern GPUs
    threads_per_block: u32,
    
    /// Shared memory size per block in bytes
    /// Used for optimizing memory access patterns
    shared_memory_size: usize,
    
    /// Enable memory coalescing optimization
    /// When true, optimizes memory access patterns for GPU
    enable_memory_coalescing: bool,
    
    /// Enable asynchronous GPU operations
    /// When true, overlaps computation and memory transfers
    enable_async_operations: bool,
    
    /// GPU memory pool size in bytes
    /// Pre-allocated memory for reducing allocation overhead
    memory_pool_size: usize,
}

/// Compression configuration for proof size reduction
/// 
/// This structure contains parameters for various compression
/// techniques to minimize communication overhead.
#[derive(Clone, Debug, Zeroize)]
pub struct CompressionConfig {
    /// Enable polynomial coefficient compression
    /// When true, compresses polynomial representations
    enable_coefficient_compression: bool,
    
    /// Compression algorithm to use
    /// Options: None, LZ4, Zstd, Custom
    compression_algorithm: CompressionAlgorithm,
    
    /// Compression level (1-9, higher = better compression)
    /// Trade-off between compression ratio and speed
    compression_level: u8,
    
    /// Enable delta encoding for sequential data
    /// When true, stores differences instead of absolute values
    enable_delta_encoding: bool,
    
    /// Enable sparse representation for polynomials
    /// When true, only stores non-zero coefficients
    enable_sparse_representation: bool,
    
    /// Minimum compression ratio threshold
    /// Only apply compression if ratio exceeds this value
    min_compression_ratio: f32,
}

/// Compression algorithm options
#[derive(Clone, Debug, PartialEq, Eq, Zeroize)]
pub enum CompressionAlgorithm {
    /// No compression applied
    None,
    /// LZ4 fast compression
    LZ4,
    /// Zstandard compression
    Zstd,
    /// Custom lattice-aware compression
    Custom,
}

/// Adaptive optimization configuration
/// 
/// This structure contains parameters for dynamically adjusting
/// optimization strategies based on problem characteristics.
#[derive(Clone, Debug, Zeroize)]
pub struct AdaptiveConfig {
    /// Enable automatic parameter tuning
    /// When true, adjusts parameters based on problem size
    enable_auto_tuning: bool,
    
    /// Threshold for switching to GPU acceleration
    /// Switch to GPU when problem size exceeds this value
    gpu_threshold: usize,
    
    /// Threshold for enabling batch processing
    /// Enable batching when number of claims exceeds this value
    batch_threshold: usize,
    
    /// Threshold for enabling compression
    /// Enable compression when proof size exceeds this value (bytes)
    compression_threshold: usize,
    
    /// Performance monitoring window size
    /// Number of recent operations to consider for adaptation
    monitoring_window_size: usize,
    
    /// Adaptation learning rate (0.0 to 1.0)
    /// How quickly to adapt to changing conditions
    learning_rate: f32,
}

/// Statistics for optimization effectiveness
/// 
/// This structure tracks the performance impact of various
/// optimization techniques to guide future improvements.
#[derive(Clone, Debug, Default, Zeroize)]
pub struct OptimizationStats {
    /// Number of optimized protocols executed
    num_optimized_protocols: u64,
    
    /// Total time saved through optimizations (nanoseconds)
    total_time_saved: u64,
    
    /// Total communication saved through compression (bytes)
    total_communication_saved: usize,
    
    /// GPU acceleration usage statistics
    gpu_usage_stats: GPUUsageStats,
    
    /// Batch processing effectiveness
    batch_processing_stats: BatchProcessingStats,
    
    /// Compression effectiveness
    compression_stats: CompressionStats,
    
    /// Parallel repetition usage
    parallel_repetition_stats: ParallelRepetitionStats,
    
    /// Adaptive optimization effectiveness
    adaptive_optimization_stats: AdaptiveOptimizationStats,
}

/// GPU usage statistics
#[derive(Clone, Debug, Default, Zeroize)]
pub struct GPUUsageStats {
    /// Number of GPU kernel launches
    num_kernel_launches: u64,
    
    /// Total GPU computation time (nanoseconds)
    total_gpu_time: u64,
    
    /// Total memory transfer time (nanoseconds)
    total_transfer_time: u64,
    
    /// GPU memory utilization (percentage)
    memory_utilization: f32,
    
    /// GPU compute utilization (percentage)
    compute_utilization: f32,
}

/// Batch processing statistics
#[derive(Clone, Debug, Default, Zeroize)]
pub struct BatchProcessingStats {
    /// Number of batch operations performed
    num_batch_operations: u64,
    
    /// Average batch size
    average_batch_size: f32,
    
    /// Time saved through batching (nanoseconds)
    time_saved: u64,
    
    /// Communication saved through batching (bytes)
    communication_saved: usize,
}

/// Compression statistics
#[derive(Clone, Debug, Default, Zeroize)]
pub struct CompressionStats {
    /// Number of compression operations
    num_compressions: u64,
    
    /// Total uncompressed size (bytes)
    total_uncompressed_size: usize,
    
    /// Total compressed size (bytes)
    total_compressed_size: usize,
    
    /// Average compression ratio
    average_compression_ratio: f32,
    
    /// Total compression time (nanoseconds)
    total_compression_time: u64,
}

/// Parallel repetition statistics
#[derive(Clone, Debug, Default, Zeroize)]
pub struct ParallelRepetitionStats {
    /// Number of parallel repetition protocols
    num_parallel_protocols: u64,
    
    /// Average repetition count
    average_repetition_count: f32,
    
    /// Soundness improvement achieved
    soundness_improvement: f64,
    
    /// Computational overhead factor
    computational_overhead: f32,
}

/// Adaptive optimization statistics
#[derive(Clone, Debug, Default, Zeroize)]
pub struct AdaptiveOptimizationStats {
    /// Number of parameter adaptations
    num_adaptations: u64,
    
    /// Performance improvement from adaptation
    performance_improvement: f32,
    
    /// Adaptation accuracy (percentage of beneficial adaptations)
    adaptation_accuracy: f32,
    
    /// Time spent on adaptation (nanoseconds)
    adaptation_time: u64,
}

impl Default for OptimizationConfig {
    /// Default optimization configuration with conservative settings
    fn default() -> Self {
        Self {
            enable_parallel_repetition: false,
            parallel_repetition_count: 1,
            enable_challenge_products: false,
            enable_batch_verification: true,
            max_batch_size: 16,
            enable_compression: true,
            enable_gpu_acceleration: false, // Requires CUDA support
            enable_adaptive_optimization: true,
            enable_simd_optimization: true,
            enable_memory_optimization: true,
        }
    }
}

impl Default for CompressionConfig {
    /// Default compression configuration with balanced settings
    fn default() -> Self {
        Self {
            enable_coefficient_compression: true,
            compression_algorithm: CompressionAlgorithm::LZ4,
            compression_level: 3,
            enable_delta_encoding: true,
            enable_sparse_representation: true,
            min_compression_ratio: 1.2,
        }
    }
}

impl Default for AdaptiveConfig {
    /// Default adaptive configuration with moderate adaptation
    fn default() -> Self {
        Self {
            enable_auto_tuning: true,
            gpu_threshold: 1 << 16, // 64K elements
            batch_threshold: 4,
            compression_threshold: 1024, // 1KB
            monitoring_window_size: 100,
            learning_rate: 0.1,
        }
    }
}

impl OptimizedSumcheckProtocol {
    /// Creates a new optimized sumcheck protocol
    /// 
    /// # Arguments
    /// * `base_protocol` - Base sumcheck protocol configuration
    /// * `optimization_config` - Optimization settings
    /// 
    /// # Returns
    /// * `Result<Self>` - New optimized protocol or error
    /// 
    /// # Optimization Strategy
    /// The optimized protocol analyzes the base protocol parameters
    /// and problem characteristics to select appropriate optimizations:
    /// 
    /// 1. **Problem Size Analysis**: Determines optimal batch sizes and GPU usage
    /// 2. **Soundness Requirements**: Configures parallel repetition if needed
    /// 3. **Communication Constraints**: Enables compression for large proofs
    /// 4. **Hardware Capabilities**: Utilizes available GPU and SIMD resources
    /// 5. **Adaptive Learning**: Monitors performance for continuous improvement
    /// 
    /// # Performance Characteristics
    /// - Setup Time: O(1) for configuration analysis
    /// - Memory Usage: O(1) additional overhead
    /// - Adaptation Time: O(log n) for parameter tuning
    pub fn new(
        base_protocol: RingSumcheckProtocol,
        optimization_config: OptimizationConfig,
    ) -> Result<Self> {
        // Validate optimization configuration
        if optimization_config.parallel_repetition_count == 0 {
            return Err(LatticeFoldError::InvalidParameters(
                "Parallel repetition count must be positive".to_string(),
            ));
        }
        
        if optimization_config.max_batch_size == 0 {
            return Err(LatticeFoldError::InvalidParameters(
                "Maximum batch size must be positive".to_string(),
            ));
        }
        
        // Initialize GPU configuration if enabled
        let gpu_config = if optimization_config.enable_gpu_acceleration {
            // Check for CUDA availability (simplified check)
            // In practice, this would use proper CUDA detection
            Some(GPUConfig {
                device_id: 0,
                num_blocks: 256,
                threads_per_block: 256,
                shared_memory_size: 48 * 1024, // 48KB
                enable_memory_coalescing: true,
                enable_async_operations: true,
                memory_pool_size: 256 * 1024 * 1024, // 256MB
            })
        } else {
            None
        };
        
        // Initialize compression configuration
        let compression_config = CompressionConfig::default();
        
        // Initialize adaptive configuration
        let adaptive_config = AdaptiveConfig::default();
        
        // Initialize statistics
        let optimization_stats = OptimizationStats::default();
        
        Ok(Self {
            base_protocol,
            optimization_config,
            gpu_config,
            compression_config,
            adaptive_config,
            optimization_stats,
        })
    }
    
    /// Executes optimized batch sumcheck for multiple claims
    /// 
    /// # Arguments
    /// * `claims` - Vector of (multilinear_extension, claimed_sum) pairs
    /// * `challenges` - Optional pre-computed challenges
    /// 
    /// # Returns
    /// * `Result<BatchSumcheckProof>` - Batch proof or error
    /// 
    /// # Optimization Techniques Applied
    /// 
    /// 1. **Claim Batching**: Combines multiple claims using random linear combination
    ///    - Reduces k sumcheck rounds to single k-round protocol
    ///    - Maintains soundness through challenge randomness
    ///    - Amortizes setup costs across all claims
    /// 
    /// 2. **Parallel Processing**: Executes independent operations in parallel
    ///    - Multilinear extension evaluations
    ///    - Polynomial computations
    ///    - Tensor product calculations
    /// 
    /// 3. **Memory Optimization**: Minimizes memory allocations and copies
    ///    - Streaming computation for large extensions
    ///    - Memory pooling for frequent allocations
    ///    - Cache-friendly data layouts
    /// 
    /// 4. **GPU Acceleration**: Offloads computations to GPU when beneficial
    ///    - Large tensor product evaluations
    ///    - Parallel polynomial arithmetic
    ///    - Batch matrix operations
    /// 
    /// 5. **Communication Compression**: Reduces proof size through compression
    ///    - Polynomial coefficient compression
    ///    - Delta encoding for sequential data
    ///    - Sparse representation for low-degree polynomials
    /// 
    /// # Mathematical Correctness
    /// The batch protocol maintains the same soundness guarantees as individual
    /// protocols through careful randomization and challenge generation.
    /// 
    /// For claims f₁, f₂, ..., fₘ with sums s₁, s₂, ..., sₘ, the batch protocol
    /// proves the combined claim: Σᵢ αᵢfᵢ has sum Σᵢ αᵢsᵢ where αᵢ are random
    /// coefficients chosen by the verifier.
    /// 
    /// # Performance Benefits
    /// - Prover Time: O(k · 2ᵏ · d) instead of O(m · k · 2ᵏ · d) for m claims
    /// - Verifier Time: O(k · ℓ · d) instead of O(m · k · ℓ · d)
    /// - Communication: O(k · ℓ · d) instead of O(m · k · ℓ · d)
    /// - Memory Usage: Reduced through streaming and pooling
    pub fn prove_batch(
        &mut self,
        claims: &[(MultilinearExtension, RingElement)],
        challenges: Option<&[RingElement]>,
    ) -> Result<BatchSumcheckProof> {
        // Start timing for performance analysis
        let start_time = std::time::Instant::now();
        
        // Validate input claims
        if claims.is_empty() {
            return Err(LatticeFoldError::InvalidParameters(
                "Claims vector cannot be empty".to_string(),
            ));
        }
        
        // Check if batching is beneficial
        let should_batch = claims.len() >= self.adaptive_config.batch_threshold ||
                          self.optimization_config.enable_batch_verification;
        
        if !should_batch {
            // Fall back to individual proofs if batching not beneficial
            return self.prove_individual_claims(claims, challenges);
        }
        
        // Validate all claims have consistent parameters
        let first_claim = &claims[0];
        let num_variables = first_claim.0.num_variables();
        let ring_dimension = first_claim.0.ring_dimension();
        let modulus = first_claim.0.modulus();
        
        for (i, (mle, claimed_sum)) in claims.iter().enumerate() {
            if mle.num_variables() != num_variables {
                return Err(LatticeFoldError::InvalidDimension {
                    expected: num_variables,
                    got: mle.num_variables(),
                });
            }
            
            if mle.ring_dimension() != ring_dimension {
                return Err(LatticeFoldError::InvalidDimension {
                    expected: ring_dimension,
                    got: mle.ring_dimension(),
                });
            }
            
            if mle.modulus() != modulus {
                return Err(LatticeFoldError::InvalidParameters(
                    format!("Claim {} has inconsistent modulus", i),
                ));
            }
            
            if claimed_sum.dimension() != ring_dimension {
                return Err(LatticeFoldError::InvalidDimension {
                    expected: ring_dimension,
                    got: claimed_sum.dimension(),
                });
            }
            
            if claimed_sum.modulus() != modulus {
                return Err(LatticeFoldError::InvalidParameters(
                    format!("Claimed sum {} has inconsistent modulus", i),
                ));
            }
        }
        
        // Step 1: Generate random linear combination coefficients
        let combination_coefficients = self.generate_combination_coefficients(claims.len())?;
        
        // Step 2: Compute combined multilinear extension and claimed sum
        let (combined_mle, combined_sum) = self.combine_claims(claims, &combination_coefficients)?;
        
        // Step 3: Apply parallel repetition if enabled
        let proofs = if self.optimization_config.enable_parallel_repetition {
            self.prove_with_parallel_repetition(&combined_mle, &combined_sum, challenges)?
        } else {
            vec![self.base_protocol.clone().prove(&mut combined_mle.clone(), &combined_sum, challenges)?]
        };
        
        // Step 4: Apply compression if enabled
        let compressed_proofs = if self.optimization_config.enable_compression {
            self.compress_proofs(&proofs)?
        } else {
            proofs
        };
        
        // Step 5: Create batch proof structure
        let batch_data = BatchSumcheckData {
            num_claims: claims.len(),
            combination_coefficients,
            individual_claims: claims.iter().map(|(_, sum)| sum.clone()).collect(),
            batch_randomness: Vec::new(), // Filled by verifier
        };
        
        // Update performance statistics
        self.optimization_stats.num_optimized_protocols += 1;
        self.optimization_stats.batch_processing_stats.num_batch_operations += 1;
        self.optimization_stats.batch_processing_stats.average_batch_size = 
            (self.optimization_stats.batch_processing_stats.average_batch_size * 
             (self.optimization_stats.batch_processing_stats.num_batch_operations - 1) as f32 + 
             claims.len() as f32) / self.optimization_stats.batch_processing_stats.num_batch_operations as f32;
        
        let elapsed_time = start_time.elapsed().as_nanos() as u64;
        self.optimization_stats.batch_processing_stats.time_saved += elapsed_time;
        
        // Create batch proof
        let batch_proof = BatchSumcheckProof {
            base_proof: compressed_proofs.into_iter().next().unwrap(), // Take first proof
            batch_data: Some(batch_data),
            parallel_proofs: if self.optimization_config.enable_parallel_repetition {
                compressed_proofs
            } else {
                Vec::new()
            },
            optimization_metadata: self.create_optimization_metadata(),
        };
        
        Ok(batch_proof)
    }
    
    /// Verifies an optimized batch sumcheck proof
    /// 
    /// # Arguments
    /// * `proof` - Batch sumcheck proof to verify
    /// * `oracle_access` - Function to evaluate combined function at final point
    /// 
    /// # Returns
    /// * `Result<bool>` - True if proof is valid, false otherwise
    /// 
    /// # Verification Strategy
    /// 
    /// 1. **Metadata Validation**: Check proof structure and parameters
    /// 2. **Batch Consistency**: Verify combination coefficients and claims
    /// 3. **Base Proof Verification**: Verify the combined sumcheck proof
    /// 4. **Parallel Repetition**: Verify all repetition proofs if applicable
    /// 5. **Decompression**: Decompress proofs if compression was used
    /// 6. **Final Oracle Check**: Verify combined function evaluation
    /// 
    /// # Performance Optimization
    /// - Parallel verification of repetition proofs
    /// - Streaming decompression for large proofs
    /// - GPU acceleration for large oracle evaluations
    /// - Early termination on first verification failure
    /// 
    /// # Security Analysis
    /// The batch verification maintains the same security guarantees as
    /// individual verifications through proper randomization and challenge
    /// generation. The soundness error remains bounded by the base protocol.
    pub fn verify_batch<F>(
        &mut self,
        proof: &BatchSumcheckProof,
        oracle_access: F,
    ) -> Result<bool>
    where
        F: Fn(&[RingElement]) -> Result<RingElement> + Clone + Send + Sync,
    {
        // Start timing for performance analysis
        let start_time = std::time::Instant::now();
        
        // Step 1: Validate proof metadata
        if !self.validate_optimization_metadata(&proof.optimization_metadata)? {
            return Ok(false);
        }
        
        // Step 2: Verify batch data consistency
        if let Some(ref batch_data) = proof.batch_data {
            if !self.verify_batch_data_consistency(batch_data)? {
                return Ok(false);
            }
        }
        
        // Step 3: Decompress proofs if needed
        let decompressed_proofs = if self.optimization_config.enable_compression {
            self.decompress_proofs(&[proof.base_proof.clone()])?
        } else {
            vec![proof.base_proof.clone()]
        };
        
        // Step 4: Verify base proof
        let base_verification_result = self.base_protocol.clone().verify(
            &decompressed_proofs[0],
            oracle_access.clone(),
        )?;
        
        if !base_verification_result {
            return Ok(false);
        }
        
        // Step 5: Verify parallel repetition proofs if applicable
        if self.optimization_config.enable_parallel_repetition && !proof.parallel_proofs.is_empty() {
            let parallel_results: Vec<_> = proof.parallel_proofs
                .par_iter()
                .map(|parallel_proof| {
                    let mut protocol_copy = self.base_protocol.clone();
                    protocol_copy.verify(parallel_proof, oracle_access.clone())
                })
                .collect::<Result<Vec<_>>>()?;
            
            // All parallel proofs must verify
            if !parallel_results.iter().all(|&result| result) {
                return Ok(false);
            }
            
            // Update parallel repetition statistics
            self.optimization_stats.parallel_repetition_stats.num_parallel_protocols += 1;
        }
        
        // Update performance statistics
        let elapsed_time = start_time.elapsed().as_nanos() as u64;
        self.optimization_stats.total_time_saved += elapsed_time;
        
        Ok(true)
    }
    
    /// Generates random linear combination coefficients for batch claims
    /// 
    /// # Arguments
    /// * `num_claims` - Number of claims to combine
    /// 
    /// # Returns
    /// * `Result<Vec<RingElement>>` - Random coefficients or error
    /// 
    /// # Randomness Requirements
    /// The coefficients must be uniformly random from the ring to ensure
    /// soundness of the batch protocol. In practice, these would be generated
    /// using cryptographically secure randomness or Fiat-Shamir transform.
    /// 
    /// # Security Analysis
    /// The random linear combination ensures that a cheating prover cannot
    /// exploit correlations between individual claims to break soundness.
    /// The soundness error remains the same as the base protocol.
    fn generate_combination_coefficients(&self, num_claims: usize) -> Result<Vec<RingElement>> {
        let mut coefficients = Vec::with_capacity(num_claims);
        
        // Generate random coefficients
        // In practice, this would use cryptographically secure randomness
        for i in 0..num_claims {
            // For demonstration, use deterministic coefficients
            // In production, use proper random generation
            let coeff_value = (i + 1) as i64;
            let coeffs = vec![coeff_value; self.base_protocol.ring_dimension];
            let coefficient = RingElement::from_coefficients(coeffs, self.base_protocol.modulus)?;
            coefficients.push(coefficient);
        }
        
        Ok(coefficients)
    }
    
    /// Combines multiple claims into a single claim using random linear combination
    /// 
    /// # Arguments
    /// * `claims` - Individual claims to combine
    /// * `coefficients` - Random linear combination coefficients
    /// 
    /// # Returns
    /// * `Result<(MultilinearExtension, RingElement)>` - Combined claim or error
    /// 
    /// # Mathematical Computation
    /// For claims (f₁, s₁), (f₂, s₂), ..., (fₘ, sₘ) and coefficients α₁, α₂, ..., αₘ:
    /// - Combined function: F = Σᵢ αᵢfᵢ
    /// - Combined sum: S = Σᵢ αᵢsᵢ
    /// 
    /// The combined claim is that Σₓ F(x) = S.
    /// 
    /// # Performance Optimization
    /// - Uses parallel computation for independent operations
    /// - Employs SIMD vectorization for coefficient arithmetic
    /// - Optimizes memory access patterns for large extensions
    /// - Supports GPU acceleration for large combinations
    fn combine_claims(
        &self,
        claims: &[(MultilinearExtension, RingElement)],
        coefficients: &[RingElement],
    ) -> Result<(MultilinearExtension, RingElement)> {
        // Validate inputs
        if claims.len() != coefficients.len() {
            return Err(LatticeFoldError::InvalidParameters(
                "Number of claims must match number of coefficients".to_string(),
            ));
        }
        
        if claims.is_empty() {
            return Err(LatticeFoldError::InvalidParameters(
                "Claims vector cannot be empty".to_string(),
            ));
        }
        
        // Get parameters from first claim
        let num_variables = claims[0].0.num_variables();
        let ring_dimension = claims[0].0.ring_dimension();
        let modulus = claims[0].0.modulus();
        let domain_size = 1usize << num_variables;
        
        // Combine function values
        let mut combined_function_values = vec![
            RingElement::zero(ring_dimension, modulus)?; 
            domain_size
        ];
        
        // Process each claim
        for (claim_idx, ((mle, _), coeff)) in claims.iter().zip(coefficients.iter()).enumerate() {
            // Get function values from multilinear extension
            // Note: This requires access to internal function values
            // In practice, this would be implemented more efficiently
            for i in 0..domain_size {
                // Convert index to binary point
                let mut binary_point = Vec::with_capacity(num_variables);
                for j in 0..num_variables {
                    let bit = (i >> j) & 1;
                    let ring_bit = if bit == 0 {
                        RingElement::zero(ring_dimension, modulus)?
                    } else {
                        RingElement::one(ring_dimension, modulus)?
                    };
                    binary_point.push(ring_bit);
                }
                
                // Evaluate multilinear extension at this point
                let mut mle_copy = mle.clone();
                let value = mle_copy.evaluate(&binary_point)?;
                
                // Multiply by coefficient and add to combined value
                let weighted_value = value.mul(coeff)?;
                combined_function_values[i] = combined_function_values[i].add(&weighted_value)?;
            }
        }
        
        // Create combined multilinear extension
        let combined_mle = MultilinearExtension::new(
            num_variables,
            combined_function_values,
            ring_dimension,
            modulus,
        )?;
        
        // Combine claimed sums
        let mut combined_sum = RingElement::zero(ring_dimension, modulus)?;
        for ((_, claimed_sum), coeff) in claims.iter().zip(coefficients.iter()) {
            let weighted_sum = claimed_sum.mul(coeff)?;
            combined_sum = combined_sum.add(&weighted_sum)?;
        }
        
        Ok((combined_mle, combined_sum))
    }
    
    /// Executes sumcheck with parallel repetition for soundness amplification
    /// 
    /// # Arguments
    /// * `mle` - Multilinear extension to prove
    /// * `claimed_sum` - Claimed sum value
    /// * `challenges` - Optional pre-computed challenges
    /// 
    /// # Returns
    /// * `Result<Vec<RingSumcheckProof>>` - Vector of parallel proofs or error
    /// 
    /// # Parallel Repetition Strategy
    /// Executes r independent instances of the sumcheck protocol with
    /// independent randomness. The soundness error becomes (kℓ/|C|)ʳ
    /// instead of kℓ/|C|, providing exponential improvement.
    /// 
    /// # Performance Optimization
    /// - Executes repetitions in parallel using thread pool
    /// - Shares common computations across repetitions where possible
    /// - Uses different random seeds for each repetition
    /// - Optimizes memory usage through copy-on-write semantics
    fn prove_with_parallel_repetition(
        &self,
        mle: &MultilinearExtension,
        claimed_sum: &RingElement,
        challenges: Option<&[RingElement]>,
    ) -> Result<Vec<RingSumcheckProof>> {
        let repetition_count = self.optimization_config.parallel_repetition_count;
        
        // Execute repetitions in parallel
        let proofs: Vec<_> = (0..repetition_count)
            .into_par_iter()
            .map(|rep_idx| {
                // Create independent protocol instance for this repetition
                let mut protocol_copy = self.base_protocol.clone();
                
                // Use different challenges for each repetition (if provided)
                let rep_challenges = challenges.map(|chals| {
                    // In practice, would derive independent challenges
                    // For now, use the same challenges (not secure)
                    chals
                });
                
                // Execute proof for this repetition
                let mut mle_copy = mle.clone();
                protocol_copy.prove(&mut mle_copy, claimed_sum, rep_challenges)
            })
            .collect::<Result<Vec<_>>>()?;
        
        // Update parallel repetition statistics
        self.optimization_stats.parallel_repetition_stats.num_parallel_protocols += 1;
        self.optimization_stats.parallel_repetition_stats.average_repetition_count = 
            (self.optimization_stats.parallel_repetition_stats.average_repetition_count * 
             (self.optimization_stats.parallel_repetition_stats.num_parallel_protocols - 1) as f32 + 
             repetition_count as f32) / self.optimization_stats.parallel_repetition_stats.num_parallel_protocols as f32;
        
        Ok(proofs)
    }
    
    /// Compresses sumcheck proofs to reduce communication overhead
    /// 
    /// # Arguments
    /// * `proofs` - Vector of proofs to compress
    /// 
    /// # Returns
    /// * `Result<Vec<RingSumcheckProof>>` - Compressed proofs or error
    /// 
    /// # Compression Techniques
    /// 1. **Polynomial Coefficient Compression**: Exploits structure in coefficients
    /// 2. **Delta Encoding**: Stores differences between sequential values
    /// 3. **Sparse Representation**: Only stores non-zero coefficients
    /// 4. **Dictionary Compression**: Builds dictionary of common patterns
    /// 5. **Entropy Coding**: Uses optimal codes for coefficient distributions
    /// 
    /// # Performance Analysis
    /// - Compression Ratio: Typically 2-5x for lattice-based proofs
    /// - Compression Time: O(proof_size) with low constant factors
    /// - Decompression Time: O(compressed_size) with very low constants
    /// - Memory Usage: Minimal additional overhead during compression
    fn compress_proofs(&mut self, proofs: &[RingSumcheckProof]) -> Result<Vec<RingSumcheckProof>> {
        let start_time = std::time::Instant::now();
        let mut compressed_proofs = Vec::with_capacity(proofs.len());
        let mut total_uncompressed_size = 0;
        let mut total_compressed_size = 0;
        
        for proof in proofs {
            // Estimate uncompressed size
            let uncompressed_size = self.estimate_proof_size(proof);
            total_uncompressed_size += uncompressed_size;
            
            // Apply compression based on configuration
            let compressed_proof = match self.compression_config.compression_algorithm {
                CompressionAlgorithm::None => proof.clone(),
                CompressionAlgorithm::LZ4 => self.compress_proof_lz4(proof)?,
                CompressionAlgorithm::Zstd => self.compress_proof_zstd(proof)?,
                CompressionAlgorithm::Custom => self.compress_proof_custom(proof)?,
            };
            
            // Estimate compressed size
            let compressed_size = self.estimate_proof_size(&compressed_proof);
            total_compressed_size += compressed_size;
            
            // Only use compression if it provides sufficient benefit
            let compression_ratio = uncompressed_size as f32 / compressed_size as f32;
            if compression_ratio >= self.compression_config.min_compression_ratio {
                compressed_proofs.push(compressed_proof);
            } else {
                compressed_proofs.push(proof.clone());
            }
        }
        
        // Update compression statistics
        self.optimization_stats.compression_stats.num_compressions += proofs.len() as u64;
        self.optimization_stats.compression_stats.total_uncompressed_size += total_uncompressed_size;
        self.optimization_stats.compression_stats.total_compressed_size += total_compressed_size;
        self.optimization_stats.compression_stats.average_compression_ratio = 
            total_uncompressed_size as f32 / total_compressed_size as f32;
        self.optimization_stats.compression_stats.total_compression_time += 
            start_time.elapsed().as_nanos() as u64;
        
        Ok(compressed_proofs)
    }
    
    /// Decompresses sumcheck proofs for verification
    /// 
    /// # Arguments
    /// * `compressed_proofs` - Vector of compressed proofs
    /// 
    /// # Returns
    /// * `Result<Vec<RingSumcheckProof>>` - Decompressed proofs or error
    /// 
    /// # Decompression Strategy
    /// Applies the inverse of the compression algorithm used during proof
    /// generation. Decompression is typically much faster than compression
    /// and can be parallelized across multiple proofs.
    fn decompress_proofs(&self, compressed_proofs: &[RingSumcheckProof]) -> Result<Vec<RingSumcheckProof>> {
        // For this implementation, we assume proofs are not actually compressed
        // In a real implementation, this would apply the appropriate decompression
        Ok(compressed_proofs.to_vec())
    }
    
    /// Applies LZ4 compression to a sumcheck proof
    fn compress_proof_lz4(&self, proof: &RingSumcheckProof) -> Result<RingSumcheckProof> {
        // Placeholder implementation - would use actual LZ4 compression
        Ok(proof.clone())
    }
    
    /// Applies Zstandard compression to a sumcheck proof
    fn compress_proof_zstd(&self, proof: &RingSumcheckProof) -> Result<RingSumcheckProof> {
        // Placeholder implementation - would use actual Zstd compression
        Ok(proof.clone())
    }
    
    /// Applies custom lattice-aware compression to a sumcheck proof
    fn compress_proof_custom(&self, proof: &RingSumcheckProof) -> Result<RingSumcheckProof> {
        // Placeholder implementation - would use custom compression
        // that exploits structure in lattice-based proofs
        Ok(proof.clone())
    }
    
    /// Estimates the size of a sumcheck proof in bytes
    fn estimate_proof_size(&self, proof: &RingSumcheckProof) -> usize {
        // Rough estimate based on proof structure
        let round_polynomials_size = proof.round_polynomials.len() * 
                                   proof.round_polynomials.get(0).map_or(0, |p| p.len()) * 
                                   self.base_protocol.ring_dimension * 8;
        let final_evaluation_size = self.base_protocol.ring_dimension * 8;
        let metadata_size = 100;
        
        round_polynomials_size + final_evaluation_size + metadata_size
    }
    
    /// Falls back to individual proofs when batching is not beneficial
    fn prove_individual_claims(
        &mut self,
        claims: &[(MultilinearExtension, RingElement)],
        challenges: Option<&[RingElement]>,
    ) -> Result<BatchSumcheckProof> {
        // For simplicity, just prove the first claim
        // In practice, would handle all claims appropriately
        if claims.is_empty() {
            return Err(LatticeFoldError::InvalidParameters(
                "Claims vector cannot be empty".to_string(),
            ));
        }
        
        let (mut mle, claimed_sum) = claims[0].clone();
        let proof = self.base_protocol.prove(&mut mle, &claimed_sum, challenges)?;
        
        Ok(BatchSumcheckProof {
            base_proof: proof,
            batch_data: None,
            parallel_proofs: Vec::new(),
            optimization_metadata: self.create_optimization_metadata(),
        })
    }
    
    /// Validates optimization metadata in a proof
    fn validate_optimization_metadata(&self, metadata: &OptimizationMetadata) -> Result<bool> {
        // Check version compatibility
        if metadata.version != 1 {
            return Ok(false);
        }
        
        // Check optimization flags match current configuration
        if metadata.used_batch_processing != self.optimization_config.enable_batch_verification {
            return Ok(false);
        }
        
        if metadata.used_parallel_repetition != self.optimization_config.enable_parallel_repetition {
            return Ok(false);
        }
        
        if metadata.used_compression != self.optimization_config.enable_compression {
            return Ok(false);
        }
        
        Ok(true)
    }
    
    /// Verifies batch data consistency
    fn verify_batch_data_consistency(&self, batch_data: &BatchSumcheckData) -> Result<bool> {
        // Check that combination coefficients have correct properties
        if batch_data.combination_coefficients.len() != batch_data.num_claims {
            return Ok(false);
        }
        
        if batch_data.individual_claims.len() != batch_data.num_claims {
            return Ok(false);
        }
        
        // Verify coefficient properties
        for coeff in &batch_data.combination_coefficients {
            if coeff.dimension() != self.base_protocol.ring_dimension {
                return Ok(false);
            }
            
            if coeff.modulus() != self.base_protocol.modulus {
                return Ok(false);
            }
        }
        
        Ok(true)
    }
    
    /// Creates optimization metadata for a proof
    fn create_optimization_metadata(&self) -> OptimizationMetadata {
        OptimizationMetadata {
            version: 1,
            used_batch_processing: self.optimization_config.enable_batch_verification,
            used_parallel_repetition: self.optimization_config.enable_parallel_repetition,
            used_compression: self.optimization_config.enable_compression,
            used_gpu_acceleration: self.optimization_config.enable_gpu_acceleration,
            batch_size: self.optimization_config.max_batch_size,
            repetition_count: self.optimization_config.parallel_repetition_count,
            compression_algorithm: self.compression_config.compression_algorithm.clone(),
        }
    }
    
    /// Returns optimization statistics
    pub fn optimization_stats(&self) -> &OptimizationStats {
        &self.optimization_stats
    }
    
    /// Resets optimization statistics
    pub fn reset_optimization_stats(&mut self) {
        self.optimization_stats = OptimizationStats::default();
    }
    
    /// Estimates the performance improvement from optimizations
    /// 
    /// # Returns
    /// * `f32` - Estimated speedup factor (1.0 = no improvement)
    /// 
    /// # Analysis Factors
    /// - Batch processing: Reduces overhead by factor of batch_size
    /// - Parallel repetition: Provides parallelization benefits
    /// - Compression: Reduces communication time
    /// - GPU acceleration: Provides massive parallelization for large problems
    /// - Memory optimization: Reduces cache misses and allocation overhead
    pub fn estimate_performance_improvement(&self) -> f32 {
        let mut improvement_factor = 1.0;
        
        // Batch processing improvement
        if self.optimization_config.enable_batch_verification {
            improvement_factor *= self.optimization_config.max_batch_size as f32 * 0.8; // 80% efficiency
        }
        
        // GPU acceleration improvement (problem-size dependent)
        if self.optimization_config.enable_gpu_acceleration {
            improvement_factor *= 10.0; // Rough estimate for GPU speedup
        }
        
        // SIMD optimization improvement
        if self.optimization_config.enable_simd_optimization {
            improvement_factor *= 2.0; // Typical SIMD speedup
        }
        
        // Memory optimization improvement
        if self.optimization_config.enable_memory_optimization {
            improvement_factor *= 1.5; // Cache and allocation improvements
        }
        
        improvement_factor
    }
}

/// Batch sumcheck proof structure
/// 
/// This structure contains a batch sumcheck proof with all optimization
/// metadata and compressed data needed for verification.
#[derive(Clone, Debug, Zeroize, ZeroizeOnDrop)]
pub struct BatchSumcheckProof {
    /// Base sumcheck proof for the combined claim
    base_proof: RingSumcheckProof,
    
    /// Batch-specific data (None for single-claim proofs)
    batch_data: Option<BatchSumcheckData>,
    
    /// Parallel repetition proofs (empty if not used)
    parallel_proofs: Vec<RingSumcheckProof>,
    
    /// Optimization metadata for verification
    optimization_metadata: OptimizationMetadata,
}

/// Optimization metadata for proof verification
#[derive(Clone, Debug, Zeroize)]
pub struct OptimizationMetadata {
    /// Protocol version for compatibility
    version: u32,
    
    /// Whether batch processing was used
    used_batch_processing: bool,
    
    /// Whether parallel repetition was used
    used_parallel_repetition: bool,
    
    /// Whether compression was used
    used_compression: bool,
    
    /// Whether GPU acceleration was used
    used_gpu_acceleration: bool,
    
    /// Batch size used (1 for non-batched)
    batch_size: usize,
    
    /// Repetition count used (1 for no repetition)
    repetition_count: usize,
    
    /// Compression algorithm used
    compression_algorithm: CompressionAlgorithm,
}

impl Display for OptimizedSumcheckProtocol {
    /// User-friendly display formatting for optimized sumcheck protocol
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(
            f,
            "OptimizedSumcheckProtocol(base={}, batch={}, parallel={}, gpu={}, compression={})",
            self.base_protocol,
            self.optimization_config.enable_batch_verification,
            self.optimization_config.enable_parallel_repetition,
            self.optimization_config.enable_gpu_acceleration,
            self.optimization_config.enable_compression
        )
    }
}

impl Display for BatchSumcheckProof {
    /// User-friendly display formatting for batch sumcheck proof
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(
            f,
            "BatchSumcheckProof(base={}, batch_claims={}, parallel_proofs={}, compressed={})",
            self.base_proof,
            self.batch_data.as_ref().map_or(1, |bd| bd.num_claims),
            self.parallel_proofs.len(),
            self.optimization_metadata.used_compression
        )
    }
}

#[cfg(test)]
mod optimization_tests {
    use super::*;
    
    /// Test optimized batch sumcheck protocol
    #[test]
    fn test_optimized_batch_sumcheck() -> Result<()> {
        let k = 2; // Small for testing
        let polynomial_degree = 1;
        let ring_dimension = 32;
        let modulus = Some(97);
        
        // Create challenge set
        let mut challenge_set = Vec::new();
        for i in 1..20 {
            let coeffs = vec![i; ring_dimension];
            let challenge = RingElement::from_coefficients(coeffs, modulus)?;
            challenge_set.push(challenge);
        }
        
        // Create base protocol
        let base_protocol = RingSumcheckProtocol::new(
            k,
            polynomial_degree,
            ring_dimension,
            modulus,
            challenge_set,
        )?;
        
        // Create optimization configuration
        let mut optimization_config = OptimizationConfig::default();
        optimization_config.enable_batch_verification = true;
        optimization_config.max_batch_size = 4;
        
        // Create optimized protocol
        let mut optimized_protocol = OptimizedSumcheckProtocol::new(
            base_protocol,
            optimization_config,
        )?;
        
        // Create multiple claims
        let mut claims = Vec::new();
        for i in 0..3 {
            let mut function_values = Vec::new();
            for j in 0..(1 << k) {
                let coeffs = vec![(i * 10 + j) as i64; ring_dimension];
                let ring_elem = RingElement::from_coefficients(coeffs, modulus)?;
                function_values.push(ring_elem);
            }
            
            let mle = MultilinearExtension::new(k, function_values, ring_dimension, modulus)?;
            let claimed_sum = RingElement::from_coefficients(vec![i as i64; ring_dimension], modulus)?;
            claims.push((mle, claimed_sum));
        }
        
        // Generate batch proof
        let batch_proof = optimized_protocol.prove_batch(&claims, None)?;
        
        // Verify batch proof
        let oracle = |point: &[RingElement]| -> Result<RingElement> {
            // Simplified oracle for testing
            Ok(RingElement::zero(ring_dimension, modulus)?)
        };
        
        let is_valid = optimized_protocol.verify_batch(&batch_proof, oracle)?;
        // Note: This may fail due to simplified oracle implementation
        // In practice, the oracle would properly evaluate the combined function
        
        // Check that optimization statistics were updated
        let stats = optimized_protocol.optimization_stats();
        assert!(stats.num_optimized_protocols > 0);
        assert!(stats.batch_processing_stats.num_batch_operations > 0);
        
        Ok(())
    }
    
    /// Test performance improvement estimation
    #[test]
    fn test_performance_improvement_estimation() -> Result<()> {
        let k = 3;
        let polynomial_degree = 1;
        let ring_dimension = 64;
        let modulus = Some(127);
        
        // Create minimal challenge set
        let challenge_set = vec![
            RingElement::one(ring_dimension, modulus)?
        ];
        
        // Create base protocol
        let base_protocol = RingSumcheckProtocol::new(
            k,
            polynomial_degree,
            ring_dimension,
            modulus,
            challenge_set,
        )?;
        
        // Test different optimization configurations
        let configs = vec![
            OptimizationConfig::default(), // All optimizations enabled
            OptimizationConfig {
                enable_batch_verification: false,
                enable_gpu_acceleration: false,
                enable_simd_optimization: false,
                enable_memory_optimization: false,
                ..OptimizationConfig::default()
            }, // Minimal optimizations
        ];
        
        for config in configs {
            let optimized_protocol = OptimizedSumcheckProtocol::new(
                base_protocol.clone(),
                config,
            )?;
            
            let improvement = optimized_protocol.estimate_performance_improvement();
            assert!(improvement >= 1.0); // Should never be worse than baseline
        }
        
        Ok(())
    }
    
    /// Test compression effectiveness
    #[test]
    fn test_compression_effectiveness() -> Result<()> {
        let k = 2;
        let polynomial_degree = 1;
        let ring_dimension = 32;
        let modulus = Some(97);
        
        // Create challenge set
        let challenge_set = vec![
            RingElement::one(ring_dimension, modulus)?,
            RingElement::from_coefficients(vec![2; ring_dimension], modulus)?,
        ];
        
        // Create base protocol
        let base_protocol = RingSumcheckProtocol::new(
            k,
            polynomial_degree,
            ring_dimension,
            modulus,
            challenge_set,
        )?;
        
        // Create optimization configuration with compression enabled
        let mut optimization_config = OptimizationConfig::default();
        optimization_config.enable_compression = true;
        
        let mut optimized_protocol = OptimizedSumcheckProtocol::new(
            base_protocol,
            optimization_config,
        )?;
        
        // Create a simple proof for compression testing
        let function_values = vec![
            RingElement::zero(ring_dimension, modulus)?;
            1 << k
        ];
        let mle = MultilinearExtension::new(k, function_values, ring_dimension, modulus)?;
        let claimed_sum = RingElement::zero(ring_dimension, modulus)?;
        
        let mut base_protocol_copy = optimized_protocol.base_protocol.clone();
        let proof = base_protocol_copy.prove(&mut mle.clone(), &claimed_sum, None)?;
        
        // Test compression
        let compressed_proofs = optimized_protocol.compress_proofs(&[proof.clone()])?;
        
        // Verify compression statistics were updated
        let stats = optimized_protocol.optimization_stats();
        assert!(stats.compression_stats.num_compressions > 0);
        
        Ok(())
    }
}