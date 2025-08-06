# LatticeFold+ Quick Start Guide

This guide will get you up and running with LatticeFold+ in minutes.

## Installation

Add LatticeFold+ to your `Cargo.toml`:

```toml
[dependencies]
latticefold-plus = "0.1.0"
ark-bls12-381 = "0.4"
ark-ff = "0.4"
ark-std = "0.4"
```

## Basic Usage

### 1. Setup Protocol Parameters

```rust
use latticefold_plus::{setup_lattice_fold, SecurityLevel};
use ark_bls12_381::{Bls12_381, Fr};
use ark_std::test_rng;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize random number generator
    let mut rng = test_rng();
    
    // Setup LatticeFold+ with 128-bit security
    let params = setup_lattice_fold::<Bls12_381, Fr, _>(
        SecurityLevel::High, 
        &mut rng
    )?;
    
    println!("Protocol setup complete!");
    println!("Ring dimension: {}", params.dimension);
    println!("Security level: {:?}", params.security_param);
    
    Ok(())
}
```

### 2. Create and Prove Instances

```rust
use latticefold_plus::{
    setup_lattice_fold, prove_lattice_fold, verify_lattice_fold,
    LatticeFoldInstance, SecurityLevel
};
use ark_bls12_381::{Bls12_381, Fr};
use ark_ff::UniformRand;
use ark_std::test_rng;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = test_rng();
    
    // Setup protocol
    let params = setup_lattice_fold::<Bls12_381, Fr, _>(
        SecurityLevel::Medium, 
        &mut rng
    )?;
    
    // Create test instances
    let mut instances = Vec::new();
    for i in 0..3 {
        let witness = (0..params.dimension)
            .map(|_| Fr::rand(&mut rng))
            .collect();
        let public_input = (0..params.dimension)
            .map(|_| Fr::rand(&mut rng))
            .collect();
            
        instances.push(LatticeFoldInstance {
            witness,
            public_input,
        });
        
        println!("Created instance {}", i + 1);
    }
    
    // Generate proof
    println!("Generating proof...");
    let proof = prove_lattice_fold(&params, &instances, &mut rng)?;
    println!("Proof generated successfully!");
    
    // Verify proof
    println!("Verifying proof...");
    let is_valid = verify_lattice_fold(&params, &proof)?;
    
    if is_valid {
        println!("✓ Proof verification successful!");
    } else {
        println!("✗ Proof verification failed!");
    }
    
    Ok(())
}
```

### 3. Working with Cyclotomic Rings

```rust
use latticefold_plus::{RingElement, BalancedCoefficients};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create ring elements in R = Z[X]/(X^64 + 1)
    let dimension = 64;
    let modulus = 2147483647; // Large prime
    
    // Create first polynomial: 1 + 2X + 3X²
    let coeffs1 = vec![1, 2, 3, 0, 0, 0]; // Pad with zeros
    let mut coeffs1_full = vec![0i64; dimension];
    coeffs1_full[..coeffs1.len()].copy_from_slice(&coeffs1);
    
    let poly1 = RingElement::new(
        BalancedCoefficients::new(coeffs1_full, modulus)?,
        dimension
    )?;
    
    // Create second polynomial: 2 + X
    let coeffs2 = vec![2, 1];
    let mut coeffs2_full = vec![0i64; dimension];
    coeffs2_full[..coeffs2.len()].copy_from_slice(&coeffs2);
    
    let poly2 = RingElement::new(
        BalancedCoefficients::new(coeffs2_full, modulus)?,
        dimension
    )?;
    
    // Perform ring operations
    let sum = &poly1 + &poly2;
    let product = &poly1 * &poly2;
    
    println!("Polynomial 1: {}", poly1);
    println!("Polynomial 2: {}", poly2);
    println!("Sum: {}", sum);
    println!("Product: {}", product);
    
    // Compute norms
    println!("||poly1||_∞ = {}", poly1.infinity_norm());
    println!("||poly2||_∞ = {}", poly2.infinity_norm());
    
    Ok(())
}
```

### 4. Using Monomial Sets for Range Proofs

```rust
use latticefold_plus::{Monomial, MonomialSet, MonomialMembershipTester};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dimension = 64;
    
    // Create monomial set M = {0, 1, X, X², ..., X^{d-1}}
    let monomial_set = MonomialSet::new(dimension)?;
    
    // Create some monomials
    let monomial_x3 = Monomial::new(3)?; // X³
    let monomial_neg_x5 = Monomial::new(5)?.negate(); // -X⁵
    
    println!("Created monomial X³: {}", monomial_x3);
    println!("Created monomial -X⁵: {}", monomial_neg_x5);
    
    // Test membership in monomial set
    let membership_tester = MonomialMembershipTester::new(dimension, 2147483647)?;
    
    // Convert monomials to ring elements for testing
    let ring_elem_x3 = monomial_x3.to_ring_element(dimension, 2147483647)?;
    let ring_elem_neg_x5 = monomial_neg_x5.to_ring_element(dimension, 2147483647)?;
    
    let is_monomial_1 = membership_tester.is_monomial(&ring_elem_x3)?;
    let is_monomial_2 = membership_tester.is_monomial(&ring_elem_neg_x5)?;
    
    println!("X³ is monomial: {}", is_monomial_1);
    println!("-X⁵ is monomial: {}", is_monomial_2);
    
    // Create a non-monomial polynomial for comparison
    let coeffs = vec![1, 2, 3]; // 1 + 2X + 3X² (not a monomial)
    let mut coeffs_full = vec![0i64; dimension];
    coeffs_full[..coeffs.len()].copy_from_slice(&coeffs);
    
    let non_monomial = RingElement::new(
        BalancedCoefficients::new(coeffs_full, 2147483647)?,
        dimension
    )?;
    
    let is_monomial_3 = membership_tester.is_monomial(&non_monomial)?;
    println!("1 + 2X + 3X² is monomial: {}", is_monomial_3);
    
    Ok(())
}
```

### 5. NTT-Accelerated Polynomial Multiplication

```rust
use latticefold_plus::{NTTParams, NTTEngine, get_ntt_params};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dimension = 64;
    let modulus = 2013265921; // NTT-friendly prime: 15 * 2^27 + 1
    
    // Get NTT parameters for this (dimension, modulus) pair
    let ntt_params = get_ntt_params(dimension, modulus)?;
    
    // Create NTT engine
    let ntt_engine = NTTEngine::new(ntt_params.clone())?;
    
    // Create test polynomials
    let poly1_coeffs = (0..dimension).map(|i| (i as i64) % 100).collect::<Vec<_>>();
    let poly2_coeffs = (0..dimension).map(|i| ((i * 2) as i64) % 100).collect::<Vec<_>>();
    
    // Perform NTT-based multiplication
    println!("Performing NTT-based multiplication...");
    let start = std::time::Instant::now();
    
    let product = ntt_engine.multiply(&poly1_coeffs, &poly2_coeffs)?;
    
    let duration = start.elapsed();
    println!("NTT multiplication completed in {:?}", duration);
    
    // Compare with schoolbook multiplication for small polynomials
    if dimension <= 128 {
        use latticefold_plus::schoolbook_multiply_optimized;
        
        println!("Comparing with schoolbook multiplication...");
        let start_schoolbook = std::time::Instant::now();
        
        let product_schoolbook = schoolbook_multiply_optimized(
            &poly1_coeffs, 
            &poly2_coeffs, 
            modulus
        )?;
        
        let duration_schoolbook = start_schoolbook.elapsed();
        println!("Schoolbook multiplication completed in {:?}", duration_schoolbook);
        
        // Verify results match
        let results_match = product == product_schoolbook;
        println!("Results match: {}", results_match);
        
        if duration.as_nanos() > 0 && duration_schoolbook.as_nanos() > 0 {
            let speedup = duration_schoolbook.as_nanos() as f64 / duration.as_nanos() as f64;
            println!("NTT speedup: {:.2}x", speedup);
        }
    }
    
    Ok(())
}
```

## Next Steps

Now that you've seen the basics, explore these advanced topics:

1. **[Range Proofs](./examples/range-proofs.md)** - Learn about algebraic range proofs without bit decomposition
2. **[Double Commitments](./examples/double-commitments.md)** - Understand compact matrix commitments
3. **[Multi-Instance Folding](./examples/multi-instance-folding.md)** - Scale to multiple instances efficiently
4. **[GPU Acceleration](./gpu-acceleration.md)** - Leverage GPU computing for maximum performance
5. **[R1CS Integration](./examples/r1cs-integration.md)** - Work with constraint systems

## Common Patterns

### Error Handling

```rust
use latticefold_plus::{LatticeFoldError, Result};

fn handle_errors() -> Result<()> {
    match some_operation() {
        Ok(result) => {
            println!("Success: {:?}", result);
            Ok(())
        }
        Err(LatticeFoldError::InvalidDimension { expected, got }) => {
            eprintln!("Dimension mismatch: expected {}, got {}", expected, got);
            Err(LatticeFoldError::InvalidDimension { expected, got })
        }
        Err(LatticeFoldError::InvalidModulus { modulus }) => {
            eprintln!("Invalid modulus: {}", modulus);
            Err(LatticeFoldError::InvalidModulus { modulus })
        }
        Err(e) => {
            eprintln!("Other error: {}", e);
            Err(e)
        }
    }
}
```

### Performance Monitoring

```rust
use std::time::Instant;

fn benchmark_operation() -> Result<()> {
    let start = Instant::now();
    
    // Your operation here
    let result = expensive_computation()?;
    
    let duration = start.elapsed();
    println!("Operation completed in {:?}", duration);
    
    // Log performance metrics
    if duration.as_millis() > 1000 {
        println!("Warning: Operation took longer than expected");
    }
    
    Ok(())
}
```

### Memory Management

```rust
use latticefold_plus::RingElement;

fn efficient_memory_usage() -> Result<()> {
    // Pre-allocate vectors when size is known
    let mut coefficients = Vec::with_capacity(1024);
    
    // Use references to avoid unnecessary clones
    let ring_elem = RingElement::new(/* ... */)?;
    process_ring_element(&ring_elem)?; // Pass by reference
    
    // Clear sensitive data when done
    coefficients.zeroize();
    
    Ok(())
}
```

This quick start guide covers the essential patterns for using LatticeFold+. For more detailed information, consult the full API documentation and examples.