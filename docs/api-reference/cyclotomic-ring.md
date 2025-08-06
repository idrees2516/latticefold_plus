# Cyclotomic Ring Arithmetic API

The cyclotomic ring module provides the foundational arithmetic operations for LatticeFold+, implementing the cyclotomic ring R := Z[X]/(X^d + 1) with optimized polynomial operations, norm computations, and Number Theoretic Transform (NTT) support.

## Mathematical Foundation

The cyclotomic ring R = Z[X]/(X^d + 1) forms the algebraic foundation of LatticeFold+. Elements are represented as polynomials f(X) = Σ_{i=0}^{d-1} f_i X^i where:

- **Ring Dimension**: d must be a power of 2 (32, 64, 128, ..., 16384)
- **Coefficient Representation**: Coefficients f_i ∈ Z in balanced form [-⌊q/2⌋, ⌊q/2⌋]
- **Reduction Rule**: X^d = -1, enabling negacyclic convolution
- **Quotient Ring**: Rq = R/qR for prime modulus q

## Core Types

### RingElement

```rust
/// Represents an element in the cyclotomic ring R = Z[X]/(X^d + 1)
#[derive(Clone, Debug, PartialEq)]
pub struct RingElement {
    /// Coefficient vector (f_0, f_1, ..., f_{d-1}) representing f(X) = Σ f_i X^i
    coefficients: Vec<i64>,
    /// Ring dimension d (must be power of 2)
    dimension: usize,
    /// Optional modulus for Rq = R/qR operations
    modulus: Option<i64>,
}
```

**Mathematical Representation**: Each `RingElement` represents a polynomial f(X) = f_0 + f_1·X + f_2·X² + ... + f_{d-1}·X^{d-1} in the ring R.

**Memory Layout**: Coefficients are stored in a cache-aligned vector for optimal SIMD performance.

**Invariants**:
- `coefficients.len() == dimension`
- `dimension` is a power of 2
- If `modulus.is_some()`, all coefficients are in balanced representation [-⌊q/2⌋, ⌊q/2⌋]

### BalancedCoefficients

```rust
/// Coefficient representation in balanced form for Rq = R/qR
#[derive(Clone, Debug, PartialEq)]
pub struct BalancedCoefficients {
    /// Coefficients in range [-⌊q/2⌋, ⌊q/2⌋]
    coeffs: Vec<i64>,
    /// Prime modulus q
    modulus: i64,
}
```

**Mathematical Purpose**: Maintains coefficients in symmetric representation around zero, which is optimal for lattice cryptography operations and reduces coefficient growth.

**Conversion Functions**:
- `to_standard(x)`: Converts balanced coefficient to standard representation [0, q-1]
- `from_standard(x)`: Converts standard coefficient to balanced representation

## Core Operations

### Ring Arithmetic

#### Addition

```rust
impl RingElement {
    /// Add two ring elements coefficient-wise
    /// 
    /// # Mathematical Operation
    /// (f + g)(X) = Σ_{i=0}^{d-1} (f_i + g_i) X^i
    /// 
    /// # Parameters
    /// - `other`: The ring element to add
    /// 
    /// # Returns
    /// Result containing the sum or an error if dimensions/moduli don't match
    /// 
    /// # Complexity
    /// - Time: O(d)
    /// - Space: O(d)
    /// - GPU: O(d/p) where p is number of parallel processors
    /// 
    /// # Example
    /// ```rust
    /// use latticefold_plus::RingElement;
    /// 
    /// let f = RingElement::new(vec![1, 2, 3, 0], None)?;
    /// let g = RingElement::new(vec![4, 5, 6, 7], None)?;
    /// let sum = f.add(&g)?;
    /// 
    /// assert_eq!(sum.coefficients(), &[5, 7, 9, 7]);
    /// ```
    pub fn add(&self, other: &Self) -> Result<Self, LatticeFoldError> {
        // Validate dimensions match
        if self.dimension != other.dimension {
            return Err(LatticeFoldError::InvalidDimension {
                expected: self.dimension,
                got: other.dimension,
            });
        }
        
        // Validate moduli compatibility
        match (self.modulus, other.modulus) {
            (Some(m1), Some(m2)) if m1 != m2 => {
                return Err(LatticeFoldError::IncompatibleModuli {
                    modulus1: m1,
                    modulus2: m2,
                });
            }
            _ => {}
        }
        
        // Perform coefficient-wise addition with overflow checking
        let mut result_coeffs = Vec::with_capacity(self.dimension);
        for i in 0..self.dimension {
            // Check for overflow before addition
            let sum = self.coefficients[i]
                .checked_add(other.coefficients[i])
                .ok_or_else(|| LatticeFoldError::ArithmeticOverflow(
                    format!("Addition overflow at position {}", i)
                ))?;
            
            // Apply modular reduction if modulus is present
            let reduced_sum = if let Some(q) = self.modulus.or(other.modulus) {
                // Reduce to balanced representation [-⌊q/2⌋, ⌊q/2⌋]
                let half_q = q / 2;
                let mut reduced = sum % q;
                if reduced > half_q {
                    reduced -= q;
                } else if reduced < -half_q {
                    reduced += q;
                }
                reduced
            } else {
                sum
            };
            
            result_coeffs.push(reduced_sum);
        }
        
        Ok(RingElement {
            coefficients: result_coeffs,
            dimension: self.dimension,
            modulus: self.modulus.or(other.modulus),
        })
    }
}
```

#### Multiplication

```rust
impl RingElement {
    /// Multiply two ring elements using optimal algorithm selection
    /// 
    /// # Mathematical Operation
    /// (f · g)(X) = f(X) · g(X) mod (X^d + 1)
    /// 
    /// The multiplication uses the negacyclic convolution property:
    /// h_k = Σ_{i+j≡k (mod d)} f_i g_j - Σ_{i+j≡k+d (mod d)} f_i g_j
    /// 
    /// # Algorithm Selection
    /// - Schoolbook: d < 64 (O(d²) complexity)
    /// - Karatsuba: 64 ≤ d < 512 (O(d^1.585) complexity)  
    /// - NTT: d ≥ 512 and q ≡ 1 + 2^e (mod 4^e) (O(d log d) complexity)
    /// 
    /// # Parameters
    /// - `other`: The ring element to multiply
    /// 
    /// # Returns
    /// Result containing the product or an error
    /// 
    /// # Complexity
    /// - Time: O(d log d) for NTT, O(d^1.585) for Karatsuba, O(d²) for schoolbook
    /// - Space: O(d)
    /// - GPU: Significant speedup for NTT with d ≥ 1024
    /// 
    /// # Example
    /// ```rust
    /// use latticefold_plus::RingElement;
    /// 
    /// // Multiply X² by X³ = X⁵ in R = Z[X]/(X⁸ + 1)
    /// let f = RingElement::new(vec![0, 0, 1, 0, 0, 0, 0, 0], None)?;
    /// let g = RingElement::new(vec![0, 0, 0, 1, 0, 0, 0, 0], None)?;
    /// let product = f.multiply(&g)?;
    /// 
    /// // Result should be X⁵
    /// assert_eq!(product.coefficients(), &[0, 0, 0, 0, 0, 1, 0, 0]);
    /// ```
    pub fn multiply(&self, other: &Self) -> Result<Self, LatticeFoldError> {
        // Validate dimensions and moduli (same as addition)
        if self.dimension != other.dimension {
            return Err(LatticeFoldError::InvalidDimension {
                expected: self.dimension,
                got: other.dimension,
            });
        }
        
        match (self.modulus, other.modulus) {
            (Some(m1), Some(m2)) if m1 != m2 => {
                return Err(LatticeFoldError::IncompatibleModuli {
                    modulus1: m1,
                    modulus2: m2,
                });
            }
            _ => {}
        }
        
        // Select optimal multiplication algorithm based on dimension and modulus
        let result_coeffs = if self.dimension >= 512 && self.can_use_ntt() {
            // Use NTT-based multiplication for large dimensions
            self.ntt_multiply(other)?
        } else if self.dimension >= 64 {
            // Use Karatsuba multiplication for medium dimensions
            self.karatsuba_multiply(other)?
        } else {
            // Use schoolbook multiplication for small dimensions
            self.schoolbook_multiply(other)?
        };
        
        Ok(RingElement {
            coefficients: result_coeffs,
            dimension: self.dimension,
            modulus: self.modulus.or(other.modulus),
        })
    }
    
    /// Check if NTT can be used (requires q ≡ 1 + 2^e (mod 4^e))
    fn can_use_ntt(&self) -> bool {
        if let Some(q) = self.modulus {
            // Check if q ≡ 1 + 2^e (mod 4^e) for some e dividing d
            let d = self.dimension;
            for e in 1..=64 {
                if d % (1 << e) == 0 {
                    let four_to_e = 1i64 << (2 * e);
                    let two_to_e = 1i64 << e;
                    if q % four_to_e == 1 + two_to_e {
                        return true;
                    }
                }
            }
        }
        false
    }
    
    /// Schoolbook multiplication with negacyclic convolution
    fn schoolbook_multiply(&self, other: &Self) -> Result<Vec<i64>, LatticeFoldError> {
        let d = self.dimension;
        let mut result = vec![0i64; d];
        
        // Compute negacyclic convolution: h_k = Σ f_i g_j where i+j ≡ k (mod d)
        // with sign flip for i+j ≥ d due to X^d = -1
        for i in 0..d {
            for j in 0..d {
                let coeff_product = self.coefficients[i]
                    .checked_mul(other.coefficients[j])
                    .ok_or_else(|| LatticeFoldError::ArithmeticOverflow(
                        format!("Multiplication overflow at positions {} and {}", i, j)
                    ))?;
                
                let sum_indices = i + j;
                if sum_indices < d {
                    // Normal case: add to result[sum_indices]
                    result[sum_indices] = result[sum_indices]
                        .checked_add(coeff_product)
                        .ok_or_else(|| LatticeFoldError::ArithmeticOverflow(
                            format!("Addition overflow at result position {}", sum_indices)
                        ))?;
                } else {
                    // Reduction case: subtract from result[sum_indices - d] due to X^d = -1
                    let reduced_index = sum_indices - d;
                    result[reduced_index] = result[reduced_index]
                        .checked_sub(coeff_product)
                        .ok_or_else(|| LatticeFoldError::ArithmeticOverflow(
                            format!("Subtraction overflow at result position {}", reduced_index)
                        ))?;
                }
            }
        }
        
        // Apply modular reduction if modulus is present
        if let Some(q) = self.modulus.or(other.modulus) {
            let half_q = q / 2;
            for coeff in &mut result {
                *coeff = *coeff % q;
                if *coeff > half_q {
                    *coeff -= q;
                } else if *coeff < -half_q {
                    *coeff += q;
                }
            }
        }
        
        Ok(result)
    }
    
    /// Karatsuba multiplication with negacyclic convolution
    fn karatsuba_multiply(&self, other: &Self) -> Result<Vec<i64>, LatticeFoldError> {
        // Implementation of Karatsuba algorithm adapted for negacyclic convolution
        // This is a recursive divide-and-conquer approach that reduces O(d²) to O(d^1.585)
        
        let d = self.dimension;
        if d <= 32 {
            // Base case: use schoolbook multiplication
            return self.schoolbook_multiply(other);
        }
        
        let half_d = d / 2;
        
        // Split polynomials: f = f_low + X^{d/2} * f_high
        let f_low = &self.coefficients[0..half_d];
        let f_high = &self.coefficients[half_d..d];
        let g_low = &other.coefficients[0..half_d];
        let g_high = &other.coefficients[half_d..d];
        
        // Create temporary ring elements for recursive calls
        let f_low_elem = RingElement {
            coefficients: f_low.to_vec(),
            dimension: half_d,
            modulus: self.modulus,
        };
        let f_high_elem = RingElement {
            coefficients: f_high.to_vec(),
            dimension: half_d,
            modulus: self.modulus,
        };
        let g_low_elem = RingElement {
            coefficients: g_low.to_vec(),
            dimension: half_d,
            modulus: other.modulus,
        };
        let g_high_elem = RingElement {
            coefficients: g_high.to_vec(),
            dimension: half_d,
            modulus: other.modulus,
        };
        
        // Recursive multiplications
        let p0 = f_low_elem.karatsuba_multiply(&g_low_elem)?;  // f_low * g_low
        let p2 = f_high_elem.karatsuba_multiply(&g_high_elem)?; // f_high * g_high
        
        // Compute (f_low + f_high) * (g_low + g_high)
        let f_sum = f_low_elem.add(&f_high_elem)?;
        let g_sum = g_low_elem.add(&g_high_elem)?;
        let p1_full = f_sum.karatsuba_multiply(&g_sum)?;
        
        // p1 = p1_full - p0 - p2
        let mut p1 = vec![0i64; half_d];
        for i in 0..half_d {
            p1[i] = p1_full[i] - p0[i] - p2[i];
        }
        
        // Combine results with negacyclic convolution
        let mut result = vec![0i64; d];
        
        // Add p0 (low part)
        for i in 0..half_d {
            result[i] += p0[i];
        }
        
        // Add p1 * X^{d/2} (middle part)
        for i in 0..half_d {
            result[i + half_d] += p1[i];
        }
        
        // Add p2 * X^d = -p2 (high part with sign flip due to X^d = -1)
        for i in 0..half_d {
            result[i] -= p2[i];
        }
        
        // Apply modular reduction if needed
        if let Some(q) = self.modulus.or(other.modulus) {
            let half_q = q / 2;
            for coeff in &mut result {
                *coeff = *coeff % q;
                if *coeff > half_q {
                    *coeff -= q;
                } else if *coeff < -half_q {
                    *coeff += q;
                }
            }
        }
        
        Ok(result)
    }
    
    /// NTT-based multiplication for large dimensions
    fn ntt_multiply(&self, other: &Self) -> Result<Vec<i64>, LatticeFoldError> {
        use crate::ntt::{NTTEngine, get_ntt_params};
        
        let q = self.modulus.or(other.modulus)
            .ok_or_else(|| LatticeFoldError::InvalidParameters(
                "NTT multiplication requires a modulus".to_string()
            ))?;
        
        // Get NTT parameters for this dimension and modulus
        let ntt_params = get_ntt_params(self.dimension, q)?;
        let mut ntt_engine = NTTEngine::new(ntt_params);
        
        // Convert to NTT domain
        let mut f_ntt = self.coefficients.clone();
        let mut g_ntt = other.coefficients.clone();
        
        ntt_engine.forward_ntt(&mut f_ntt)?;
        ntt_engine.forward_ntt(&mut g_ntt)?;
        
        // Pointwise multiplication in NTT domain
        for i in 0..self.dimension {
            f_ntt[i] = (f_ntt[i] * g_ntt[i]) % q;
        }
        
        // Convert back to coefficient domain
        ntt_engine.inverse_ntt(&mut f_ntt)?;
        
        Ok(f_ntt)
    }
}
```

### Norm Computations

```rust
impl RingElement {
    /// Compute the ℓ∞-norm of the ring element
    /// 
    /// # Mathematical Definition
    /// ||f||_∞ = max_{i∈[d]} |f_i|
    /// 
    /// This is the maximum absolute value among all coefficients.
    /// 
    /// # Returns
    /// The infinity norm as a non-negative integer
    /// 
    /// # Complexity
    /// - Time: O(d)
    /// - Space: O(1)
    /// - GPU: O(d/p) with parallel reduction
    /// 
    /// # Example
    /// ```rust
    /// use latticefold_plus::RingElement;
    /// 
    /// let f = RingElement::new(vec![1, -5, 3, 2], None)?;
    /// assert_eq!(f.infinity_norm(), 5);
    /// ```
    pub fn infinity_norm(&self) -> i64 {
        // Find maximum absolute value among all coefficients
        // Use parallel reduction for large dimensions
        if self.dimension >= 1024 {
            self.infinity_norm_parallel()
        } else {
            self.infinity_norm_sequential()
        }
    }
    
    /// Sequential computation of infinity norm
    fn infinity_norm_sequential(&self) -> i64 {
        let mut max_abs = 0i64;
        
        // Iterate through all coefficients to find maximum absolute value
        for &coeff in &self.coefficients {
            // Handle potential overflow in abs() for i64::MIN
            let abs_coeff = if coeff == i64::MIN {
                // i64::MIN.abs() would overflow, so handle specially
                i64::MAX
            } else {
                coeff.abs()
            };
            
            // Update maximum if current coefficient is larger
            if abs_coeff > max_abs {
                max_abs = abs_coeff;
            }
        }
        
        max_abs
    }
    
    /// Parallel computation of infinity norm using SIMD
    fn infinity_norm_parallel(&self) -> i64 {
        use crate::simd::vectorized_arithmetic::parallel_max_abs;
        
        // Use SIMD-optimized parallel reduction to find maximum absolute value
        parallel_max_abs(&self.coefficients)
    }
    
    /// Compute the operator norm ||f||_op = sup_{g≠0} ||f·g||_∞ / ||g||_∞
    /// 
    /// # Mathematical Definition
    /// For f ∈ R, the operator norm is the maximum factor by which f can amplify
    /// the infinity norm of another ring element under multiplication.
    /// 
    /// # Approximation
    /// Uses the bound ||f||_op ≤ d · ||f||_∞ from Lemma 2.5 for efficiency.
    /// For exact computation, use `operator_norm_exact()`.
    /// 
    /// # Returns
    /// Upper bound on the operator norm
    /// 
    /// # Example
    /// ```rust
    /// use latticefold_plus::RingElement;
    /// 
    /// let f = RingElement::new(vec![1, 2, 1, 0], None)?;
    /// let op_norm_bound = f.operator_norm_bound();
    /// assert_eq!(op_norm_bound, 4 * 2); // d * ||f||_∞
    /// ```
    pub fn operator_norm_bound(&self) -> i64 {
        // Use Lemma 2.5: ||f||_op ≤ d · ||f||_∞
        let infinity_norm = self.infinity_norm();
        
        // Check for potential overflow in multiplication
        if let Some(result) = (self.dimension as i64).checked_mul(infinity_norm) {
            result
        } else {
            // If overflow would occur, return maximum possible value
            i64::MAX
        }
    }
}
```

### Coefficient Operations

```rust
impl RingElement {
    /// Extract coefficient vector cf(f) = (f_0, f_1, ..., f_{d-1})
    /// 
    /// # Mathematical Purpose
    /// Provides access to the coefficient representation of the polynomial.
    /// This is used extensively in commitment schemes and proof systems.
    /// 
    /// # Returns
    /// Reference to the coefficient vector
    /// 
    /// # Example
    /// ```rust
    /// use latticefold_plus::RingElement;
    /// 
    /// let f = RingElement::new(vec![1, 2, 3, 4], None)?;
    /// assert_eq!(f.coefficients(), &[1, 2, 3, 4]);
    /// ```
    pub fn coefficients(&self) -> &[i64] {
        &self.coefficients
    }
    
    /// Extract constant term ct(f) = f_0
    /// 
    /// # Mathematical Purpose
    /// The constant term is crucial for range proofs and polynomial evaluations.
    /// In range proofs, we verify ct(b·ψ) = a for range checking.
    /// 
    /// # Returns
    /// The constant coefficient f_0
    /// 
    /// # Example
    /// ```rust
    /// use latticefold_plus::RingElement;
    /// 
    /// let f = RingElement::new(vec![42, 1, 2, 3], None)?;
    /// assert_eq!(f.constant_term(), 42);
    /// ```
    pub fn constant_term(&self) -> i64 {
        // Return first coefficient (constant term)
        // Safe because dimension > 0 is guaranteed by constructor
        self.coefficients[0]
    }
    
    /// Evaluate polynomial at a given point
    /// 
    /// # Mathematical Operation
    /// f(α) = Σ_{i=0}^{d-1} f_i α^i
    /// 
    /// Uses Horner's method for numerical stability and efficiency.
    /// 
    /// # Parameters
    /// - `point`: The evaluation point α
    /// 
    /// # Returns
    /// The polynomial evaluated at the given point
    /// 
    /// # Complexity
    /// - Time: O(d)
    /// - Space: O(1)
    /// 
    /// # Example
    /// ```rust
    /// use latticefold_plus::RingElement;
    /// 
    /// // f(X) = 1 + 2X + 3X²
    /// let f = RingElement::new(vec![1, 2, 3, 0], None)?;
    /// // f(2) = 1 + 2*2 + 3*4 = 1 + 4 + 12 = 17
    /// assert_eq!(f.evaluate(2), 17);
    /// ```
    pub fn evaluate(&self, point: i64) -> i64 {
        // Use Horner's method: f(x) = f_0 + x(f_1 + x(f_2 + x(...)))
        // This is numerically stable and requires only d-1 multiplications
        
        if self.coefficients.is_empty() {
            return 0;
        }
        
        // Start with highest degree coefficient
        let mut result = self.coefficients[self.dimension - 1];
        
        // Work backwards through coefficients
        for i in (0..self.dimension - 1).rev() {
            // result = result * point + f_i
            result = result.saturating_mul(point).saturating_add(self.coefficients[i]);
            
            // Apply modular reduction if modulus is present
            if let Some(q) = self.modulus {
                result = result % q;
                let half_q = q / 2;
                if result > half_q {
                    result -= q;
                } else if result < -half_q {
                    result += q;
                }
            }
        }
        
        result
    }
}
```

## Constructor and Validation

```rust
impl RingElement {
    /// Create a new ring element with validation
    /// 
    /// # Parameters
    /// - `coefficients`: Coefficient vector of length d (must be power of 2)
    /// - `modulus`: Optional modulus for Rq operations
    /// 
    /// # Returns
    /// Result containing the ring element or validation error
    /// 
    /// # Validation
    /// - Dimension must be a power of 2 between MIN_RING_DIMENSION and MAX_RING_DIMENSION
    /// - If modulus is provided, coefficients must be in balanced representation
    /// - Coefficient vector must not be empty
    /// 
    /// # Example
    /// ```rust
    /// use latticefold_plus::RingElement;
    /// 
    /// // Create element in Z[X]/(X⁴ + 1)
    /// let f = RingElement::new(vec![1, 2, 3, 4], None)?;
    /// 
    /// // Create element in Z₁₇[X]/(X⁴ + 1) with balanced coefficients
    /// let g = RingElement::new(vec![1, -2, 3, -4], Some(17))?;
    /// ```
    pub fn new(coefficients: Vec<i64>, modulus: Option<i64>) -> Result<Self, LatticeFoldError> {
        // Validate coefficient vector is not empty
        if coefficients.is_empty() {
            return Err(LatticeFoldError::InvalidParameters(
                "Coefficient vector cannot be empty".to_string()
            ));
        }
        
        let dimension = coefficients.len();
        
        // Validate dimension is a power of 2
        if !dimension.is_power_of_two() {
            return Err(LatticeFoldError::InvalidParameters(
                format!("Dimension {} must be a power of 2", dimension)
            ));
        }
        
        // Validate dimension is within supported range
        if dimension < MIN_RING_DIMENSION || dimension > MAX_RING_DIMENSION {
            return Err(LatticeFoldError::InvalidParameters(
                format!("Dimension {} must be between {} and {}", 
                    dimension, MIN_RING_DIMENSION, MAX_RING_DIMENSION)
            ));
        }
        
        // Validate modulus if provided
        if let Some(q) = modulus {
            // Modulus must be positive and odd (for NTT compatibility)
            if q <= 1 || q % 2 == 0 {
                return Err(LatticeFoldError::InvalidModulus { modulus: q });
            }
            
            // Validate coefficients are in balanced representation
            let half_q = q / 2;
            for (i, &coeff) in coefficients.iter().enumerate() {
                if coeff < -half_q || coeff > half_q {
                    return Err(LatticeFoldError::CoefficientOutOfRange {
                        coefficient: coeff,
                        min_bound: -half_q,
                        max_bound: half_q,
                        position: i,
                    });
                }
            }
        }
        
        Ok(RingElement {
            coefficients,
            dimension,
            modulus,
        })
    }
    
    /// Create a zero element in the ring
    /// 
    /// # Parameters
    /// - `dimension`: Ring dimension (must be power of 2)
    /// - `modulus`: Optional modulus
    /// 
    /// # Returns
    /// Zero ring element with all coefficients set to 0
    /// 
    /// # Example
    /// ```rust
    /// use latticefold_plus::RingElement;
    /// 
    /// let zero = RingElement::zero(8, Some(17))?;
    /// assert_eq!(zero.coefficients(), &[0, 0, 0, 0, 0, 0, 0, 0]);
    /// ```
    pub fn zero(dimension: usize, modulus: Option<i64>) -> Result<Self, LatticeFoldError> {
        Self::new(vec![0; dimension], modulus)
    }
    
    /// Create a one element (multiplicative identity) in the ring
    /// 
    /// # Parameters
    /// - `dimension`: Ring dimension (must be power of 2)
    /// - `modulus`: Optional modulus
    /// 
    /// # Returns
    /// One ring element representing the polynomial 1
    /// 
    /// # Example
    /// ```rust
    /// use latticefold_plus::RingElement;
    /// 
    /// let one = RingElement::one(4, None)?;
    /// assert_eq!(one.coefficients(), &[1, 0, 0, 0]);
    /// ```
    pub fn one(dimension: usize, modulus: Option<i64>) -> Result<Self, LatticeFoldError> {
        let mut coeffs = vec![0; dimension];
        coeffs[0] = 1;
        Self::new(coeffs, modulus)
    }
    
    /// Create a monomial X^degree
    /// 
    /// # Parameters
    /// - `degree`: The degree of the monomial (must be < dimension)
    /// - `dimension`: Ring dimension
    /// - `modulus`: Optional modulus
    /// 
    /// # Returns
    /// Ring element representing X^degree
    /// 
    /// # Example
    /// ```rust
    /// use latticefold_plus::RingElement;
    /// 
    /// // Create X³ in Z[X]/(X⁸ + 1)
    /// let x_cubed = RingElement::monomial(3, 8, None)?;
    /// assert_eq!(x_cubed.coefficients(), &[0, 0, 0, 1, 0, 0, 0, 0]);
    /// ```
    pub fn monomial(degree: usize, dimension: usize, modulus: Option<i64>) -> Result<Self, LatticeFoldError> {
        if degree >= dimension {
            return Err(LatticeFoldError::InvalidParameters(
                format!("Degree {} must be less than dimension {}", degree, dimension)
            ));
        }
        
        let mut coeffs = vec![0; dimension];
        coeffs[degree] = 1;
        Self::new(coeffs, modulus)
    }
}
```

## Performance Characteristics

### Complexity Analysis

| Operation | Sequential | Parallel (CPU) | GPU | Memory |
|-----------|------------|----------------|-----|---------|
| Addition | O(d) | O(d/p) | O(d/p) | O(d) |
| Schoolbook Multiply | O(d²) | O(d²/p) | O(d²/p) | O(d) |
| Karatsuba Multiply | O(d^1.585) | O(d^1.585/p) | O(d^1.585/p) | O(d log d) |
| NTT Multiply | O(d log d) | O(d log d/p) | O(d log d/p) | O(d) |
| Infinity Norm | O(d) | O(d/p) | O(d/p) | O(1) |
| Evaluation | O(d) | O(1) | O(1) | O(1) |

Where:
- `d` = ring dimension
- `p` = number of parallel processors

### Memory Layout

Ring elements use cache-aligned memory layout for optimal SIMD performance:

```rust
/// Memory-aligned coefficient storage for SIMD optimization
#[repr(align(64))] // Align to cache line boundary
struct AlignedCoefficients {
    data: Vec<i64>,
}
```

### GPU Acceleration

For large dimensions (d ≥ 1024), operations automatically use GPU acceleration when available:

```rust
/// GPU-accelerated ring operations
impl RingElement {
    /// Check if GPU acceleration is available and beneficial
    fn should_use_gpu(&self) -> bool {
        self.dimension >= 1024 && crate::gpu::is_available()
    }
    
    /// GPU-accelerated addition
    fn gpu_add(&self, other: &Self) -> Result<Vec<i64>, LatticeFoldError> {
        use crate::gpu::kernels::ring_add;
        ring_add(&self.coefficients, &other.coefficients, self.modulus)
    }
    
    /// GPU-accelerated NTT multiplication
    fn gpu_ntt_multiply(&self, other: &Self) -> Result<Vec<i64>, LatticeFoldError> {
        use crate::gpu::kernels::ntt_multiply;
        ntt_multiply(&self.coefficients, &other.coefficients, 
                    self.dimension, self.modulus.unwrap())
    }
}
```

## Error Handling

The cyclotomic ring module provides comprehensive error handling:

```rust
/// Ring-specific error types
#[derive(Debug, Error)]
pub enum RingError {
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    
    #[error("Invalid dimension {dimension}: must be power of 2 between {min} and {max}")]
    InvalidDimension { dimension: usize, min: usize, max: usize },
    
    #[error("Coefficient {coefficient} out of range [{min}, {max}] at position {position}")]
    CoefficientOutOfRange { coefficient: i64, min: i64, max: i64, position: usize },
    
    #[error("Incompatible moduli: {modulus1} and {modulus2}")]
    IncompatibleModuli { modulus1: i64, modulus2: i64 },
    
    #[error("Arithmetic overflow in {operation}")]
    ArithmeticOverflow { operation: String },
    
    #[error("NTT not supported for modulus {modulus} and dimension {dimension}")]
    NTTNotSupported { modulus: i64, dimension: usize },
}
```

## Thread Safety

All ring operations are thread-safe:

- **Immutable Operations**: Addition, multiplication, and norm computations don't modify inputs
- **Send + Sync**: `RingElement` implements both traits for safe concurrent access
- **Lock-Free**: No internal mutability or shared state

## Examples

### Basic Ring Arithmetic

```rust
use latticefold_plus::{RingElement, LatticeFoldError};

fn basic_arithmetic_example() -> Result<(), LatticeFoldError> {
    // Create ring elements in Z₁₇[X]/(X⁸ + 1)
    let f = RingElement::new(vec![1, 2, 3, 4, 0, 0, 0, 0], Some(17))?;
    let g = RingElement::new(vec![5, 6, 7, 8, 0, 0, 0, 0], Some(17))?;
    
    // Addition
    let sum = f.add(&g)?;
    println!("f + g = {:?}", sum.coefficients());
    
    // Multiplication (automatically selects optimal algorithm)
    let product = f.multiply(&g)?;
    println!("f * g = {:?}", product.coefficients());
    
    // Norm computation
    println!("||f||_∞ = {}", f.infinity_norm());
    println!("||g||_∞ = {}", g.infinity_norm());
    
    // Polynomial evaluation
    println!("f(2) = {}", f.evaluate(2));
    
    Ok(())
}
```

### NTT-Based Fast Multiplication

```rust
use latticefold_plus::{RingElement, LatticeFoldError};

fn ntt_multiplication_example() -> Result<(), LatticeFoldError> {
    // Use a large dimension and NTT-friendly modulus
    let dimension = 1024;
    let modulus = 12289; // 12289 = 1 + 12 * 2^10, supports NTT for d=1024
    
    // Create random-looking polynomials
    let mut f_coeffs = vec![0i64; dimension];
    let mut g_coeffs = vec![0i64; dimension];
    
    for i in 0..dimension {
        f_coeffs[i] = (i as i64) % (modulus / 2);
        g_coeffs[i] = ((i * i) as i64) % (modulus / 2);
    }
    
    let f = RingElement::new(f_coeffs, Some(modulus))?;
    let g = RingElement::new(g_coeffs, Some(modulus))?;
    
    // This will automatically use NTT multiplication
    let start = std::time::Instant::now();
    let product = f.multiply(&g)?;
    let duration = start.elapsed();
    
    println!("NTT multiplication of degree {} completed in {:?}", 
             dimension, duration);
    println!("Product norm: {}", product.infinity_norm());
    
    Ok(())
}
```

### GPU-Accelerated Operations

```rust
use latticefold_plus::{RingElement, LatticeFoldError};

fn gpu_acceleration_example() -> Result<(), LatticeFoldError> {
    // Large dimension that benefits from GPU acceleration
    let dimension = 4096;
    let modulus = 40961; // NTT-friendly modulus
    
    // Create large polynomials
    let f_coeffs: Vec<i64> = (0..dimension).map(|i| (i as i64) % 1000).collect();
    let g_coeffs: Vec<i64> = (0..dimension).map(|i| ((i * 7) as i64) % 1000).collect();
    
    let f = RingElement::new(f_coeffs, Some(modulus))?;
    let g = RingElement::new(g_coeffs, Some(modulus))?;
    
    // Check if GPU is available
    if crate::gpu::is_available() {
        println!("GPU acceleration available");
        
        // Time GPU multiplication
        let start = std::time::Instant::now();
        let product = f.multiply(&g)?;
        let gpu_duration = start.elapsed();
        
        println!("GPU multiplication completed in {:?}", gpu_duration);
        println!("Result norm: {}", product.infinity_norm());
    } else {
        println!("GPU not available, using CPU");
        let product = f.multiply(&g)?;
        println!("CPU result norm: {}", product.infinity_norm());
    }
    
    Ok(())
}
```

## Constants

```rust
/// Minimum supported ring dimension
pub const MIN_RING_DIMENSION: usize = 32;

/// Maximum supported ring dimension
pub const MAX_RING_DIMENSION: usize = 16384;

/// Threshold for switching from schoolbook to Karatsuba multiplication
pub const KARATSUBA_THRESHOLD: usize = 64;

/// Threshold for switching to NTT multiplication
pub const NTT_THRESHOLD: usize = 512;

/// Threshold for GPU acceleration
pub const GPU_THRESHOLD: usize = 1024;
```

This comprehensive API documentation provides complete coverage of the cyclotomic ring arithmetic module, including mathematical foundations, detailed implementation explanations, performance characteristics, error handling, and practical examples.