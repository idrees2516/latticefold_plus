# Number Theoretic Transform (NTT) API

The NTT module provides optimized Number Theoretic Transform implementations for fast polynomial multiplication in cyclotomic rings, supporting both CPU and GPU acceleration with comprehensive parameter validation and security analysis.

## Mathematical Foundation

The Number Theoretic Transform (NTT) is the discrete Fourier transform over finite fields, enabling fast polynomial multiplication with O(d log d) complexity instead of O(d²) schoolbook multiplication.

### NTT Requirements

For NTT to work over Rq = Zq[X]/(X^d + 1), we need:
- **Prime Modulus**: q must be prime
- **Primitive Root Condition**: q ≡ 1 + 2^e (mod 4^e) where e | d
- **Primitive Root of Unity**: ω ∈ Zq with ω^{2d} ≡ 1 (mod q) and ω^d ≡ -1 (mod q)

### Mathematical Operations

**Forward NTT**: â[i] = Σ_{j=0}^{d-1} a[j] · ω^{ij} mod q for i ∈ [d]
**Inverse NTT**: a[j] = d^{-1} · Σ_{i=0}^{d-1} â[i] · ω^{-ij} mod q for j ∈ [d]
**Pointwise Multiplication**: ĉ[i] = â[i] · b̂[i] mod q in NTT domain

## Core Types

### NTTParams

```rust
/// Parameters for NTT computation with precomputed values
#[derive(Clone, Debug, PartialEq)]
pub struct NTTParams {
    /// Ring dimension d (must be power of 2)
    pub dimension: usize,
    /// Prime modulus q ≡ 1 + 2^e (mod 4^e)
    pub modulus: i64,
    /// Primitive 2d-th root of unity ω
    pub root_of_unity: i64,
    /// Precomputed twiddle factors ω^i for i ∈ [d]
    pub twiddle_factors: Vec<i64>,
    /// Bit-reversal permutation table for in-place computation
    pub bit_reversal_table: Vec<usize>,
    /// Inverse of dimension d^{-1} mod q
    pub dimension_inverse: i64,
}
```

**Mathematical Invariants**:
- `dimension` is a power of 2
- `modulus` is prime and satisfies q ≡ 1 + 2^e (mod 4^e)
- `root_of_unity^{2*dimension} ≡ 1 (mod modulus)`
- `root_of_unity^dimension ≡ -1 (mod modulus)`
- `twiddle_factors[i] = root_of_unity^i mod modulus`
- `dimension_inverse * dimension ≡ 1 (mod modulus)`

### NTTEngine

```rust
/// High-performance NTT computation engine with GPU acceleration
#[derive(Debug)]
pub struct NTTEngine {
    /// NTT parameters with precomputed values
    params: NTTParams,
    /// GPU context for acceleration (if available)
    gpu_context: Option<GPUContext>,
    /// Performance statistics
    stats: NTTStats,
}
```

## Core Operations

### Parameter Generation

```rust
impl NTTParams {
    /// Generate NTT parameters for given dimension and modulus
    /// 
    /// # Mathematical Validation
    /// Verifies all NTT requirements:
    /// 1. q is prime
    /// 2. q ≡ 1 + 2^e (mod 4^e) for some e | d
    /// 3. Finds primitive 2d-th root of unity ω
    /// 4. Precomputes all twiddle factors and bit-reversal table
    /// 
    /// # Parameters
    /// - `dimension`: Ring dimension d (must be power of 2)
    /// - `modulus`: Prime modulus q
    /// 
    /// # Returns
    /// Result containing validated NTT parameters or error
    /// 
    /// # Complexity
    /// - Time: O(d log d) for precomputation
    /// - Space: O(d) for twiddle factors and bit-reversal table
    /// 
    /// # Example
    /// ```rust
    /// use latticefold_plus::ntt::NTTParams;
    /// 
    /// // Generate parameters for dimension 1024 with modulus 12289
    /// // 12289 = 1 + 3 * 2^12, so it supports NTT for d = 1024
    /// let params = NTTParams::new(1024, 12289)?;
    /// assert_eq!(params.dimension, 1024);
    /// assert_eq!(params.modulus, 12289);
    /// ```
    pub fn new(dimension: usize, modulus: i64) -> Result<Self, NTTError> {
        // Validate dimension is power of 2
        if !dimension.is_power_of_two() {
            return Err(NTTError::InvalidDimension {
                dimension,
                reason: "Dimension must be a power of 2".to_string(),
            });
        }
        
        // Validate dimension is within supported range
        if dimension < MIN_NTT_DIMENSION || dimension > MAX_NTT_DIMENSION {
            return Err(NTTError::InvalidDimension {
                dimension,
                reason: format!("Dimension must be between {} and {}", 
                    MIN_NTT_DIMENSION, MAX_NTT_DIMENSION),
            });
        }
        
        // Validate modulus is prime
        if !is_prime(modulus) {
            return Err(NTTError::InvalidModulus {
                modulus,
                reason: "Modulus must be prime for NTT".to_string(),
            });
        }
        
        // Find primitive 2d-th root of unity
        let root_of_unity = find_primitive_root(dimension, modulus)?;
        
        // Precompute twiddle factors: ω^i for i ∈ [d]
        let mut twiddle_factors = Vec::with_capacity(dimension);
        let mut current_power = 1i64;
        
        for i in 0..dimension {
            twiddle_factors.push(current_power);
            // current_power = (current_power * root_of_unity) % modulus
            current_power = modular_multiply(current_power, root_of_unity, modulus);
        }
        
        // Generate bit-reversal permutation table
        let bit_reversal_table = generate_bit_reversal_table(dimension);
        
        // Compute modular inverse of dimension
        let dimension_inverse = modular_inverse(dimension as i64, modulus)?;
        
        Ok(NTTParams {
            dimension,
            modulus,
            root_of_unity,
            twiddle_factors,
            bit_reversal_table,
            dimension_inverse,
        })
    }
}
```
/// Find primitive 2d-th root of unity ω with ω^{2d} ≡ 1 and ω^d ≡ -1 (mod q)
fn find_primitive_root(dimension: usize, modulus: 