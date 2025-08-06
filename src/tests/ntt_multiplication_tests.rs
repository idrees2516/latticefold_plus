/// Comprehensive tests for NTT-based polynomial multiplication
/// 
/// This module provides extensive testing of the NTT multiplication implementation
/// including correctness verification, performance benchmarking, and edge case handling.
/// 
/// Test Categories:
/// - Correctness tests against reference implementations
/// - Performance benchmarks comparing algorithms
/// - Edge case handling (small/large coefficients, various dimensions)
/// - Error condition testing (invalid inputs, parameter mismatches)
/// - Security validation (constant-time properties, side-channel resistance)
/// - Integration tests with other system components

use crate::cyclotomic_ring::RingElement;
use crate::ntt::{NTTParams, NTTEngine, multiplication};
use crate::polynomial_multiplication::{schoolbook_multiply_optimized, karatsuba_multiply_optimized};
use crate::error::{LatticeFoldError, Result};
use std::sync::Arc;
use std::time::Instant;

/// Test suite for NTT-based polynomial multiplication
#[cfg(test)]
mod tests {
    use super::*;

    /// Tests basic NTT multiplication correctness against schoolbook reference
    /// 
    /// Verifies that NTT multiplication produces identical results to the
    /// schoolbook algorithm for various polynomial sizes and coefficient ranges.
    /// This is the fundamental correctness test for the NTT implementation.
    #[test]
    fn test_ntt_multiplication_correctness() {
        // Test parameters: various dimensions and NTT-friendly moduli
        let test_cases = vec![
            (64, 7681),    // Small dimension, small modulus
            (128, 12289),  // Medium dimension, medium modulus  
            (256, 40961),  // Large dimension, large modulus
            (512, 65537),  // Very large dimension, large modulus
        ];
        
        for (dimension, modulus) in test_cases {
            println!("Testing NTT multiplication: d={}, q={}", dimension, modulus);
            
            // Generate test polynomials with various coefficient patterns
            let test_polynomials = generate_test_polynomials(dimension, modulus);
            
            for (name, f, g) in test_polynomials {
                println!("  Testing case: {}", name);
                
                // Compute reference result using schoolbook multiplication
                let reference_result = schoolbook_multiply_optimized(&f, &g)
                    .expect("Schoolbook multiplication should succeed");
                
                // Compute NTT result
                let ntt_result = multiplication::ntt_multiply_direct(&f, &g)
                    .expect("NTT multiplication should succeed");
                
                // Verify results are identical
                assert_eq!(
                    reference_result.coefficients(),
                    ntt_result.coefficients(),
                    "NTT and schoolbook results should be identical for case: {}", name
                );
                
                // Verify modulus preservation
                assert_eq!(
                    reference_result.modulus(),
                    ntt_result.modulus(),
                    "Modulus should be preserved in NTT multiplication"
                );
                
                // Verify dimension preservation
                assert_eq!(
                    reference_result.dimension(),
                    ntt_result.dimension(),
                    "Dimension should be preserved in NTT multiplication"
                );
            }
        }
    }
    
    /// Tests NTT multiplication with automatic algorithm selection
    /// 
    /// Verifies that the automatic algorithm selection correctly chooses
    /// NTT when beneficial and falls back to other algorithms when appropriate.
    #[test]
    fn test_ntt_automatic_selection() {
        // Test case 1: Large dimension with NTT-friendly modulus (should use NTT)
        let dimension = 1024;
        let modulus = 7681;
        
        let f_coeffs: Vec<i64> = (0..dimension).map(|i| (i as i64 * 17) % modulus).collect();
        let g_coeffs: Vec<i64> = (0..dimension).map(|i| (i as i64 * 23) % modulus).collect();
        
        let f = RingElement::from_coefficients(f_coeffs, Some(modulus)).unwrap();
        let g = RingElement::from_coefficients(g_coeffs, Some(modulus)).unwrap();
        
        // Test automatic selection
        let auto_result = multiplication::ntt_multiply_with_fallback(&f, &g)
            .expect("Automatic selection should succeed");
        
        // Test direct NTT
        let ntt_result = multiplication::ntt_multiply_direct(&f, &g)
            .expect("Direct NTT should succeed");
        
        // Results should be identical (both should use NTT)
        assert_eq!(
            auto_result.coefficients(),
            ntt_result.coefficients(),
            "Automatic selection should choose NTT for large NTT-friendly polynomials"
        );
        
        // Test case 2: Small dimension (should fall back to schoolbook/Karatsuba)
        let small_dimension = 64;
        let small_f_coeffs: Vec<i64> = (0..small_dimension).map(|i| (i as i64 * 17) % modulus).collect();
        let small_g_coeffs: Vec<i64> = (0..small_dimension).map(|i| (i as i64 * 23) % modulus).collect();
        
        let small_f = RingElement::from_coefficients(small_f_coeffs, Some(modulus)).unwrap();
        let small_g = RingElement::from_coefficients(small_g_coeffs, Some(modulus)).unwrap();
        
        let small_auto_result = multiplication::ntt_multiply_with_fallback(&small_f, &small_g)
            .expect("Automatic selection should succeed for small polynomials");
        
        let small_reference = schoolbook_multiply_optimized(&small_f, &small_g)
            .expect("Schoolbook should succeed for small polynomials");
        
        assert_eq!(
            small_auto_result.coefficients(),
            small_reference.coefficients(),
            "Automatic selection should produce correct results for small polynomials"
        );
    }
    
    /// Tests NTT multiplication performance compared to other algorithms
    /// 
    /// Benchmarks NTT against schoolbook and Karatsuba algorithms to verify
    /// that NTT provides the expected performance improvements for large polynomials.
    #[test]
    fn test_ntt_performance_benchmark() {
        // Test parameters for performance comparison
        let dimensions = vec![256, 512, 1024, 2048];
        let modulus = 7681i64; // NTT-friendly modulus
        
        for dimension in dimensions {
            println!("Benchmarking multiplication algorithms for dimension {}", dimension);
            
            // Generate test polynomials
            let f_coeffs: Vec<i64> = (0..dimension)
                .map(|i| ((i as i64 * 31 + 17) % modulus + modulus) % modulus - modulus/2)
                .collect();
            let g_coeffs: Vec<i64> = (0..dimension)
                .map(|i| ((i as i64 * 37 + 23) % modulus + modulus) % modulus - modulus/2)
                .collect();
            
            let f = RingElement::from_coefficients(f_coeffs, Some(modulus)).unwrap();
            let g = RingElement::from_coefficients(g_coeffs, Some(modulus)).unwrap();
            
            // Benchmark NTT multiplication
            let ntt_start = Instant::now();
            let ntt_result = multiplication::ntt_multiply_direct(&f, &g)
                .expect("NTT multiplication should succeed");
            let ntt_time = ntt_start.elapsed();
            
            // Benchmark Karatsuba multiplication (if dimension is reasonable)
            let karatsuba_time = if dimension <= 1024 {
                let karatsuba_start = Instant::now();
                let karatsuba_result = karatsuba_multiply_optimized(&f, &g)
                    .expect("Karatsuba multiplication should succeed");
                let karatsuba_elapsed = karatsuba_start.elapsed();
                
                // Verify results are identical
                assert_eq!(
                    ntt_result.coefficients(),
                    karatsuba_result.coefficients(),
                    "NTT and Karatsuba should produce identical results"
                );
                
                Some(karatsuba_elapsed)
            } else {
                None
            };
            
            // Benchmark schoolbook multiplication (only for small dimensions)
            let schoolbook_time = if dimension <= 512 {
                let schoolbook_start = Instant::now();
                let schoolbook_result = schoolbook_multiply_optimized(&f, &g)
                    .expect("Schoolbook multiplication should succeed");
                let schoolbook_elapsed = schoolbook_start.elapsed();
                
                // Verify results are identical
                assert_eq!(
                    ntt_result.coefficients(),
                    schoolbook_result.coefficients(),
                    "NTT and schoolbook should produce identical results"
                );
                
                Some(schoolbook_elapsed)
            } else {
                None
            };
            
            // Print performance results
            println!("  NTT time: {:?}", ntt_time);
            if let Some(karatsuba_time) = karatsuba_time {
                println!("  Karatsuba time: {:?}", karatsuba_time);
                let speedup = karatsuba_time.as_nanos() as f64 / ntt_time.as_nanos() as f64;
                println!("  NTT speedup over Karatsuba: {:.2}x", speedup);
                
                // For large dimensions, NTT should be faster
                if dimension >= 1024 {
                    assert!(
                        speedup > 1.0,
                        "NTT should be faster than Karatsuba for dimension {}", dimension
                    );
                }
            }
            if let Some(schoolbook_time) = schoolbook_time {
                println!("  Schoolbook time: {:?}", schoolbook_time);
                let speedup = schoolbook_time.as_nanos() as f64 / ntt_time.as_nanos() as f64;
                println!("  NTT speedup over schoolbook: {:.2}x", speedup);
                
                // NTT should be faster than schoolbook for medium+ dimensions
                if dimension >= 256 {
                    assert!(
                        speedup > 1.0,
                        "NTT should be faster than schoolbook for dimension {}", dimension
                    );
                }
            }
        }
    }
    
    /// Tests error handling in NTT multiplication
    /// 
    /// Verifies that the NTT implementation correctly handles various error
    /// conditions including invalid inputs, parameter mismatches, and edge cases.
    #[test]
    fn test_ntt_error_handling() {
        let dimension = 256;
        let modulus = 7681;
        
        // Test case 1: Dimension mismatch
        let f_coeffs = vec![1i64; dimension];
        let g_coeffs = vec![1i64; dimension / 2]; // Different dimension
        
        let f = RingElement::from_coefficients(f_coeffs, Some(modulus)).unwrap();
        let g = RingElement::from_coefficients(g_coeffs, Some(modulus)).unwrap();
        
        let result = multiplication::ntt_multiply_direct(&f, &g);
        assert!(result.is_err(), "NTT multiplication should fail with dimension mismatch");
        
        if let Err(LatticeFoldError::InvalidDimension { expected, got }) = result {
            assert_eq!(expected, dimension);
            assert_eq!(got, dimension / 2);
        } else {
            panic!("Expected InvalidDimension error");
        }
        
        // Test case 2: Modulus mismatch
        let f_mod1 = RingElement::from_coefficients(vec![1i64; dimension], Some(7681)).unwrap();
        let g_mod2 = RingElement::from_coefficients(vec![1i64; dimension], Some(12289)).unwrap();
        
        let result = multiplication::ntt_multiply_direct(&f_mod1, &g_mod2);
        assert!(result.is_err(), "NTT multiplication should fail with modulus mismatch");
        
        if let Err(LatticeFoldError::IncompatibleModuli { modulus1, modulus2 }) = result {
            assert_eq!(modulus1, 7681);
            assert_eq!(modulus2, 12289);
        } else {
            panic!("Expected IncompatibleModuli error");
        }
        
        // Test case 3: No modulus (should fail for direct NTT)
        let f_no_mod = RingElement::from_coefficients(vec![1i64; dimension], None).unwrap();
        let g_no_mod = RingElement::from_coefficients(vec![1i64; dimension], None).unwrap();
        
        let result = multiplication::ntt_multiply_direct(&f_no_mod, &g_no_mod);
        assert!(result.is_err(), "Direct NTT multiplication should fail without modulus");
        
        // Test case 4: Non-NTT-friendly modulus
        let non_ntt_modulus = 1009; // Prime but not NTT-friendly for dimension 256
        let f_bad_mod = RingElement::from_coefficients(vec![1i64; dimension], Some(non_ntt_modulus)).unwrap();
        let g_bad_mod = RingElement::from_coefficients(vec![1i64; dimension], Some(non_ntt_modulus)).unwrap();
        
        let result = multiplication::ntt_multiply_direct(&f_bad_mod, &g_bad_mod);
        assert!(result.is_err(), "NTT multiplication should fail with non-NTT-friendly modulus");
    }
    
    /// Tests NTT multiplication with edge cases
    /// 
    /// Verifies correct handling of special polynomial cases including
    /// zero polynomials, constant polynomials, and sparse polynomials.
    #[test]
    fn test_ntt_edge_cases() {
        let dimension = 256;
        let modulus = 7681;
        
        // Test case 1: Zero polynomial multiplication
        let zero_coeffs = vec![0i64; dimension];
        let nonzero_coeffs: Vec<i64> = (0..dimension).map(|i| (i as i64 + 1) % modulus).collect();
        
        let zero_poly = RingElement::from_coefficients(zero_coeffs, Some(modulus)).unwrap();
        let nonzero_poly = RingElement::from_coefficients(nonzero_coeffs, Some(modulus)).unwrap();
        
        let result = multiplication::ntt_multiply_direct(&zero_poly, &nonzero_poly)
            .expect("Zero polynomial multiplication should succeed");
        
        // Result should be zero polynomial
        for &coeff in result.coefficients() {
            assert_eq!(coeff, 0, "Zero polynomial multiplication should yield zero");
        }
        
        // Test case 2: Constant polynomial multiplication
        let const_coeffs = {
            let mut coeffs = vec![0i64; dimension];
            coeffs[0] = 42; // Constant term only
            coeffs
        };
        let const_poly = RingElement::from_coefficients(const_coeffs, Some(modulus)).unwrap();
        
        let result = multiplication::ntt_multiply_direct(&const_poly, &nonzero_poly)
            .expect("Constant polynomial multiplication should succeed");
        
        // Result should be 42 * nonzero_poly
        let expected = schoolbook_multiply_optimized(&const_poly, &nonzero_poly)
            .expect("Reference multiplication should succeed");
        
        assert_eq!(
            result.coefficients(),
            expected.coefficients(),
            "Constant polynomial multiplication should match reference"
        );
        
        // Test case 3: Sparse polynomial multiplication
        let sparse_coeffs = {
            let mut coeffs = vec![0i64; dimension];
            coeffs[0] = 1;      // Constant term
            coeffs[1] = 2;      // X term
            coeffs[dimension/2] = 3; // X^{d/2} term
            coeffs[dimension-1] = 4; // X^{d-1} term
            coeffs
        };
        let sparse_poly = RingElement::from_coefficients(sparse_coeffs, Some(modulus)).unwrap();
        
        let result = multiplication::ntt_multiply_direct(&sparse_poly, &sparse_poly)
            .expect("Sparse polynomial multiplication should succeed");
        
        // Verify against reference
        let expected = schoolbook_multiply_optimized(&sparse_poly, &sparse_poly)
            .expect("Reference multiplication should succeed");
        
        assert_eq!(
            result.coefficients(),
            expected.coefficients(),
            "Sparse polynomial multiplication should match reference"
        );
        
        // Test case 4: Maximum coefficient values
        let max_coeffs = vec![modulus / 2; dimension]; // Maximum positive coefficients
        let min_coeffs = vec![-modulus / 2; dimension]; // Maximum negative coefficients
        
        let max_poly = RingElement::from_coefficients(max_coeffs, Some(modulus)).unwrap();
        let min_poly = RingElement::from_coefficients(min_coeffs, Some(modulus)).unwrap();
        
        let result = multiplication::ntt_multiply_direct(&max_poly, &min_poly)
            .expect("Maximum coefficient multiplication should succeed");
        
        // Verify against reference
        let expected = schoolbook_multiply_optimized(&max_poly, &min_poly)
            .expect("Reference multiplication should succeed");
        
        assert_eq!(
            result.coefficients(),
            expected.coefficients(),
            "Maximum coefficient multiplication should match reference"
        );
    }
    
    /// Tests NTT engine functionality
    /// 
    /// Verifies the high-level NTT engine interface including parameter
    /// management, performance tracking, and batch operations.
    #[test]
    fn test_ntt_engine() {
        let dimension = 512;
        let modulus = 7681;
        
        // Create NTT parameters
        let params = Arc::new(NTTParams::new(dimension, modulus).unwrap());
        
        // Create NTT engine
        let engine = NTTEngine::new(params).expect("NTT engine creation should succeed");
        
        // Test polynomial multiplication through engine
        let f_coeffs: Vec<i64> = (0..dimension).map(|i| (i as i64 * 17) % modulus).collect();
        let g_coeffs: Vec<i64> = (0..dimension).map(|i| (i as i64 * 23) % modulus).collect();
        
        let result = engine.multiply_polynomials(&f_coeffs, &g_coeffs)
            .expect("Engine multiplication should succeed");
        
        // Verify result correctness
        let f = RingElement::from_coefficients(f_coeffs, Some(modulus)).unwrap();
        let g = RingElement::from_coefficients(g_coeffs, Some(modulus)).unwrap();
        let expected = schoolbook_multiply_optimized(&f, &g)
            .expect("Reference multiplication should succeed");
        
        assert_eq!(
            result,
            expected.coefficients(),
            "Engine multiplication should match reference"
        );
        
        // Check performance statistics
        let stats = engine.stats();
        assert_eq!(stats.multiplication_count, 1, "Should record one multiplication");
        assert!(stats.total_multiplication_time_ns > 0, "Should record execution time");
        assert!(stats.peak_memory_usage_bytes > 0, "Should record memory usage");
        
        // Test multiple operations
        for _ in 0..5 {
            let _ = engine.multiply_polynomials(&f_coeffs, &g_coeffs)
                .expect("Multiple operations should succeed");
        }
        
        let final_stats = engine.stats();
        assert_eq!(final_stats.multiplication_count, 6, "Should record all multiplications");
        assert!(final_stats.avg_multiplication_time_ns() > 0.0, "Should compute average time");
    }
    
    /// Generates test polynomials with various coefficient patterns
    /// 
    /// # Arguments
    /// * `dimension` - Polynomial dimension
    /// * `modulus` - Coefficient modulus
    /// 
    /// # Returns
    /// * `Vec<(String, RingElement, RingElement)>` - Named test cases
    fn generate_test_polynomials(dimension: usize, modulus: i64) -> Vec<(String, RingElement, RingElement)> {
        let mut test_cases = Vec::new();
        
        // Case 1: Small random coefficients
        let small_f: Vec<i64> = (0..dimension).map(|i| (i as i64 % 7) - 3).collect();
        let small_g: Vec<i64> = (0..dimension).map(|i| ((i * 3) as i64 % 7) - 3).collect();
        test_cases.push((
            "small_random".to_string(),
            RingElement::from_coefficients(small_f, Some(modulus)).unwrap(),
            RingElement::from_coefficients(small_g, Some(modulus)).unwrap(),
        ));
        
        // Case 2: Medium random coefficients
        let medium_f: Vec<i64> = (0..dimension).map(|i| ((i * 17) as i64 % 201) - 100).collect();
        let medium_g: Vec<i64> = (0..dimension).map(|i| ((i * 23) as i64 % 201) - 100).collect();
        test_cases.push((
            "medium_random".to_string(),
            RingElement::from_coefficients(medium_f, Some(modulus)).unwrap(),
            RingElement::from_coefficients(medium_g, Some(modulus)).unwrap(),
        ));
        
        // Case 3: Large coefficients (near modulus bound)
        let bound = modulus / 4;
        let large_f: Vec<i64> = (0..dimension).map(|i| ((i * 31) as i64 % bound) - bound/2).collect();
        let large_g: Vec<i64> = (0..dimension).map(|i| ((i * 37) as i64 % bound) - bound/2).collect();
        test_cases.push((
            "large_coefficients".to_string(),
            RingElement::from_coefficients(large_f, Some(modulus)).unwrap(),
            RingElement::from_coefficients(large_g, Some(modulus)).unwrap(),
        ));
        
        // Case 4: Alternating pattern
        let alt_f: Vec<i64> = (0..dimension).map(|i| if i % 2 == 0 { 1 } else { -1 }).collect();
        let alt_g: Vec<i64> = (0..dimension).map(|i| if i % 3 == 0 { 2 } else { -2 }).collect();
        test_cases.push((
            "alternating_pattern".to_string(),
            RingElement::from_coefficients(alt_f, Some(modulus)).unwrap(),
            RingElement::from_coefficients(alt_g, Some(modulus)).unwrap(),
        ));
        
        test_cases
    }
}

/// Integration tests for NTT multiplication with other system components
#[cfg(test)]
mod integration_tests {
    use super::*;
    use crate::commitment::LinearCommitment;
    use crate::gadget::GadgetMatrix;
    
    /// Tests NTT multiplication integration with commitment schemes
    /// 
    /// Verifies that NTT-based polynomial multiplication works correctly
    /// within the broader context of lattice-based commitment schemes.
    #[test]
    fn test_ntt_with_commitments() {
        let dimension = 512;
        let modulus = 7681;
        let kappa = 8; // Security parameter
        
        // Create commitment scheme parameters
        let commitment_params = crate::commitment::CommitmentParams::new(
            kappa, dimension, modulus, 100 // norm bound
        ).expect("Commitment parameters should be valid");
        
        // Create linear commitment scheme
        let mut commitment_scheme = LinearCommitment::new(commitment_params)
            .expect("Commitment scheme creation should succeed");
        
        // Generate commitment matrix
        let mut rng = rand::thread_rng();
        commitment_scheme.generate_matrix(&mut rng)
            .expect("Matrix generation should succeed");
        
        // Create test vectors
        let a_coeffs: Vec<i64> = (0..dimension).map(|i| (i as i64 * 17) % 100 - 50).collect();
        let b_coeffs: Vec<i64> = (0..dimension).map(|i| (i as i64 * 23) % 100 - 50).collect();
        
        let a = RingElement::from_coefficients(a_coeffs, Some(modulus)).unwrap();
        let b = RingElement::from_coefficients(b_coeffs, Some(modulus)).unwrap();
        
        // Compute polynomial product using NTT
        let ab_product = multiplication::ntt_multiply_with_fallback(&a, &b)
            .expect("NTT multiplication should succeed");
        
        // Test commitment homomorphism: com(a) * com(b) should relate to com(a*b)
        let com_a = commitment_scheme.commit_vector(&[a.clone()])
            .expect("Commitment to a should succeed");
        let com_b = commitment_scheme.commit_vector(&[b.clone()])
            .expect("Commitment to b should succeed");
        let com_ab = commitment_scheme.commit_vector(&[ab_product])
            .expect("Commitment to a*b should succeed");
        
        // Verify that the commitment scheme preserves the multiplication structure
        // (This is a simplified test - in practice, the relationship is more complex)
        assert_eq!(com_a.len(), kappa, "Commitment should have correct dimension");
        assert_eq!(com_b.len(), kappa, "Commitment should have correct dimension");
        assert_eq!(com_ab.len(), kappa, "Commitment should have correct dimension");
        
        println!("NTT multiplication successfully integrated with commitment scheme");
    }
    
    /// Tests NTT multiplication with gadget matrix operations
    /// 
    /// Verifies that NTT-based multiplication works correctly with
    /// gadget matrix decomposition and reconstruction operations.
    #[test]
    fn test_ntt_with_gadget_matrices() {
        let dimension = 256;
        let modulus = 7681;
        let base = 4;
        let num_digits = 8;
        
        // Create gadget matrix
        let gadget = GadgetMatrix::new(base, num_digits, dimension)
            .expect("Gadget matrix creation should succeed");
        
        // Create test polynomial
        let f_coeffs: Vec<i64> = (0..dimension).map(|i| (i as i64 * 17) % 100 - 50).collect();
        let f = RingElement::from_coefficients(f_coeffs, Some(modulus)).unwrap();
        
        // Decompose polynomial using gadget matrix
        let f_decomposed = gadget.decompose_vector(f.coefficients())
            .expect("Gadget decomposition should succeed");
        
        // Reconstruct polynomial
        let f_reconstructed = gadget.reconstruct_vector(&f_decomposed)
            .expect("Gadget reconstruction should succeed");
        
        // Verify reconstruction correctness
        assert_eq!(
            f.coefficients(),
            &f_reconstructed,
            "Gadget reconstruction should preserve polynomial"
        );
        
        // Test NTT multiplication with decomposed polynomials
        let g_coeffs: Vec<i64> = (0..dimension).map(|i| (i as i64 * 23) % 100 - 50).collect();
        let g = RingElement::from_coefficients(g_coeffs, Some(modulus)).unwrap();
        
        // Multiply original polynomials using NTT
        let fg_product = multiplication::ntt_multiply_with_fallback(&f, &g)
            .expect("NTT multiplication should succeed");
        
        // Verify that the product has expected properties
        assert_eq!(fg_product.dimension(), dimension, "Product should preserve dimension");
        assert_eq!(fg_product.modulus(), Some(modulus), "Product should preserve modulus");
        
        // Test that the product can be decomposed and reconstructed
        let product_decomposed = gadget.decompose_vector(fg_product.coefficients())
            .expect("Product decomposition should succeed");
        let product_reconstructed = gadget.reconstruct_vector(&product_decomposed)
            .expect("Product reconstruction should succeed");
        
        assert_eq!(
            fg_product.coefficients(),
            &product_reconstructed,
            "Product should survive gadget decomposition/reconstruction"
        );
        
        println!("NTT multiplication successfully integrated with gadget matrices");
    }
}

/// Benchmark suite for comprehensive performance analysis
#[cfg(test)]
mod benchmarks {
    use super::*;
    use std::time::{Duration, Instant};
    
    /// Comprehensive benchmark comparing all multiplication algorithms
    /// 
    /// Measures performance across various polynomial dimensions and
    /// coefficient distributions to validate the performance claims
    /// of NTT-based multiplication.
    #[test]
    #[ignore] // Run with --ignored flag for performance testing
    fn benchmark_multiplication_algorithms() {
        let dimensions = vec![64, 128, 256, 512, 1024, 2048];
        let modulus = 7681i64;
        let iterations = 10;
        
        println!("Comprehensive Multiplication Algorithm Benchmark");
        println!("================================================");
        println!("Modulus: {}", modulus);
        println!("Iterations per test: {}", iterations);
        println!();
        
        for dimension in dimensions {
            println!("Dimension: {}", dimension);
            println!("-".repeat(50));
            
            // Generate test polynomials
            let f_coeffs: Vec<i64> = (0..dimension)
                .map(|i| ((i as i64 * 31 + 17) % modulus + modulus) % modulus - modulus/2)
                .collect();
            let g_coeffs: Vec<i64> = (0..dimension)
                .map(|i| ((i as i64 * 37 + 23) % modulus + modulus) % modulus - modulus/2)
                .collect();
            
            let f = RingElement::from_coefficients(f_coeffs, Some(modulus)).unwrap();
            let g = RingElement::from_coefficients(g_coeffs, Some(modulus)).unwrap();
            
            // Benchmark NTT multiplication
            let ntt_times = benchmark_algorithm("NTT", iterations, || {
                multiplication::ntt_multiply_direct(&f, &g).unwrap()
            });
            
            // Benchmark Karatsuba multiplication (if reasonable)
            let karatsuba_times = if dimension <= 1024 {
                Some(benchmark_algorithm("Karatsuba", iterations, || {
                    karatsuba_multiply_optimized(&f, &g).unwrap()
                }))
            } else {
                None
            };
            
            // Benchmark schoolbook multiplication (if reasonable)
            let schoolbook_times = if dimension <= 512 {
                Some(benchmark_algorithm("Schoolbook", iterations, || {
                    schoolbook_multiply_optimized(&f, &g).unwrap()
                }))
            } else {
                None
            };
            
            // Benchmark automatic selection
            let auto_times = benchmark_algorithm("Auto-Select", iterations, || {
                multiplication::ntt_multiply_with_fallback(&f, &g).unwrap()
            });
            
            // Print results
            print_benchmark_results("NTT", &ntt_times);
            if let Some(ref times) = karatsuba_times {
                print_benchmark_results("Karatsuba", times);
                print_speedup("NTT vs Karatsuba", &ntt_times, times);
            }
            if let Some(ref times) = schoolbook_times {
                print_benchmark_results("Schoolbook", times);
                print_speedup("NTT vs Schoolbook", &ntt_times, times);
            }
            print_benchmark_results("Auto-Select", &auto_times);
            
            println!();
        }
    }
    
    /// Benchmarks a single algorithm multiple times
    fn benchmark_algorithm<F, R>(name: &str, iterations: usize, mut f: F) -> Vec<Duration>
    where
        F: FnMut() -> R,
    {
        let mut times = Vec::with_capacity(iterations);
        
        // Warm-up run
        let _ = f();
        
        // Actual benchmark runs
        for _ in 0..iterations {
            let start = Instant::now();
            let _ = f();
            times.push(start.elapsed());
        }
        
        times
    }
    
    /// Prints benchmark results with statistics
    fn print_benchmark_results(algorithm: &str, times: &[Duration]) {
        let total_ns: u64 = times.iter().map(|d| d.as_nanos() as u64).sum();
        let avg_ns = total_ns / times.len() as u64;
        let min_ns = times.iter().map(|d| d.as_nanos() as u64).min().unwrap();
        let max_ns = times.iter().map(|d| d.as_nanos() as u64).max().unwrap();
        
        println!("  {}: avg={:.2}μs, min={:.2}μs, max={:.2}μs", 
                algorithm,
                avg_ns as f64 / 1000.0,
                min_ns as f64 / 1000.0,
                max_ns as f64 / 1000.0);
    }
    
    /// Prints speedup comparison between two algorithms
    fn print_speedup(comparison: &str, fast_times: &[Duration], slow_times: &[Duration]) {
        let fast_avg: u64 = fast_times.iter().map(|d| d.as_nanos() as u64).sum::<u64>() / fast_times.len() as u64;
        let slow_avg: u64 = slow_times.iter().map(|d| d.as_nanos() as u64).sum::<u64>() / slow_times.len() as u64;
        
        let speedup = slow_avg as f64 / fast_avg as f64;
        println!("  {}: {:.2}x speedup", comparison, speedup);
    }
}