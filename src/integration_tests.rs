/// Integration tests for LatticeFold+ core mathematical infrastructure
/// 
/// This module provides comprehensive integration tests that verify the correct
/// interaction between all implemented components of the core mathematical
/// infrastructure for LatticeFold+.
/// 
/// Test Coverage:
/// - Cyclotomic ring arithmetic operations and properties
/// - Polynomial multiplication algorithm correctness and performance
/// - Modular arithmetic operations and security properties
/// - Norm computations and bounds verification
/// - Cross-component integration and consistency
/// - Performance benchmarks and regression testing
/// 
/// The tests are designed to validate both mathematical correctness and
/// implementation efficiency, ensuring the infrastructure meets the
/// requirements for the LatticeFold+ protocol implementation.

#[cfg(test)]
mod tests {
    use crate::cyclotomic_ring::{RingElement, BalancedCoefficients};
    use crate::modular_arithmetic::{ModularArithmetic, BarrettParams, MontgomeryParams};
    use crate::norm_computation::{InfinityNorm, EuclideanNorm, OperatorNorm};
    use crate::polynomial_multiplication::{
        schoolbook_multiply_optimized, karatsuba_multiply_optimized, multiply_with_algorithm_selection
    };
    use crate::error::LatticeFoldError;
    
    /// Test that all components work together for basic ring operations
    #[test]
    fn test_ring_arithmetic_integration() {
        let dimension = 64;
        let modulus = Some(1009i64);
        
        // Create test ring elements
        let coeffs1 = vec![1, 2, -1, 0, 3, -2, 1, 0, 2, -1, 0, 1, -3, 2, 0, -1,
                          0, 1, -2, 3, 0, -1, 2, 0, 1, -3, 0, 2, -1, 0, 3, -2,
                          1, 0, -2, 1, 3, 0, -1, 2, 0, -3, 1, 0, 2, -1, 0, 3,
                          -2, 1, 0, -1, 2, 0, 3, -1, 0, 2, -3, 0, 1, -2, 0, 1];
        let coeffs2 = vec![2, -1, 3, 0, -2, 1, 0, 3, -1, 2, 0, -3, 1, 0, 2, -1,
                          3, 0, -2, 1, 0, -1, 3, 2, 0, -2, 1, 0, 3, -1, 0, 2,
                          -3, 0, 1, -2, 0, 3, -1, 0, 2, -1, 3, 0, -2, 1, 0, -3,
                          2, 0, -1, 3, 0, -2, 1, 0, 3, -1, 2, 0, -3, 1, 0, 2];
        
        let f = RingElement::from_coefficients(coeffs1, modulus).unwrap();
        let g = RingElement::from_coefficients(coeffs2, modulus).unwrap();
        
        // Test addition
        let sum = f.clone().add(g.clone()).unwrap();
        assert_eq!(sum.dimension(), dimension);
        assert_eq!(sum.modulus(), modulus);
        
        // Test subtraction
        let diff = f.clone().sub(g.clone()).unwrap();
        assert_eq!(diff.dimension(), dimension);
        
        // Test multiplication using different algorithms
        let product_auto = f.clone().mul(g.clone()).unwrap();
        let product_schoolbook = schoolbook_multiply_optimized(&f, &g).unwrap();
        let product_karatsuba = karatsuba_multiply_optimized(&f, &g).unwrap();
        let product_selection = multiply_with_algorithm_selection(&f, &g).unwrap();
        
        // All multiplication algorithms should give the same result
        assert_eq!(product_auto.coefficients(), product_schoolbook.coefficients());
        assert_eq!(product_auto.coefficients(), product_karatsuba.coefficients());
        assert_eq!(product_auto.coefficients(), product_selection.coefficients());
        
        // Test negation
        let neg_f = f.clone().neg().unwrap();
        let zero_sum = f.add(neg_f).unwrap();
        assert_eq!(zero_sum.infinity_norm(), 0);
        
        println!("✓ Ring arithmetic integration test passed");
    }
    
    /// Test modular arithmetic integration with ring operations
    #[test]
    fn test_modular_arithmetic_integration() {
        let modulus = 1009i64;
        let arith = ModularArithmetic::new(modulus).unwrap();
        let barrett = BarrettParams::new(modulus).unwrap();
        
        // Test that modular arithmetic is consistent with ring operations
        let a = 123i64;
        let b = 456i64;
        
        // Test addition consistency
        let mod_sum = arith.add_mod(a, b);
        let expected_sum = barrett.to_balanced(barrett.reduce_barrett((a + b) as u128));
        assert_eq!(mod_sum, expected_sum);
        
        // Test multiplication consistency
        let mod_product = arith.mul_mod(a, b);
        let expected_product = barrett.to_balanced(barrett.reduce_barrett((a as u128) * (b as u128)));
        assert_eq!(mod_product, expected_product);
        
        // Test Montgomery arithmetic for odd modulus
        if modulus % 2 == 1 {
            let montgomery = MontgomeryParams::new(modulus).unwrap();
            
            let a_mont = montgomery.to_montgomery(a);
            let b_mont = montgomery.to_montgomery(b);
            let product_mont = montgomery.montgomery_multiply(a_mont, b_mont);
            let product_std = montgomery.from_montgomery(product_mont);
            let product_balanced = montgomery.to_balanced(product_std);
            
            assert_eq!(product_balanced, mod_product);
        }
        
        // Test batch operations
        let a_batch = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let b_batch = vec![8, 7, 6, 5, 4, 3, 2, 1];
        let mut sum_results = vec![0i64; 8];
        let mut mul_results = vec![0i64; 8];
        
        arith.add_mod_batch(&a_batch, &b_batch, &mut sum_results);
        arith.mul_mod_batch(&a_batch, &b_batch, &mut mul_results);
        
        // Verify batch results match individual operations
        for i in 0..8 {
            assert_eq!(sum_results[i], arith.add_mod(a_batch[i], b_batch[i]));
            assert_eq!(mul_results[i], arith.mul_mod(a_batch[i], b_batch[i]));
        }
        
        println!("✓ Modular arithmetic integration test passed");
    }
    
    /// Test norm computations across different components
    #[test]
    fn test_norm_computation_integration() {
        let dimension = 32;
        let modulus = Some(1009i64);
        
        // Create test ring element with known properties
        let coeffs = vec![1, -5, 3, 0, -2, 4, 0, -1, 2, -3, 0, 1, -4, 2, 0, -1,
                         3, 0, -2, 1, 0, -3, 2, 0, 1, -2, 0, 3, -1, 0, 2, -1];
        let element = RingElement::from_coefficients(coeffs.clone(), modulus).unwrap();
        
        // Test infinity norm consistency
        let ring_inf_norm = element.infinity_norm();
        let direct_inf_norm = InfinityNorm::compute_vector(&coeffs);
        let ring_element_norm = InfinityNorm::compute_ring_element(&element);
        
        assert_eq!(ring_inf_norm, direct_inf_norm);
        assert_eq!(ring_inf_norm, ring_element_norm);
        assert_eq!(ring_inf_norm, 5); // Maximum absolute value in coeffs
        
        // Test Euclidean norm
        let euclidean_norm = EuclideanNorm::compute_ring_element(&element);
        let direct_euclidean = EuclideanNorm::compute_vector_f64(&coeffs);
        assert!((euclidean_norm - direct_euclidean).abs() < 1e-10);
        
        // Test operator norm bounds
        let operator_bound = OperatorNorm::compute_upper_bound(&element);
        assert_eq!(operator_bound, (dimension as i64) * ring_inf_norm);
        
        // Test norm properties under arithmetic operations
        let coeffs2 = vec![2, -1, 0, 3, -1, 0, 2, -3, 1, 0, -2, 1, 0, 3, -1, 0,
                          2, -1, 3, 0, -2, 1, 0, -1, 2, 0, -3, 1, 0, 2, -1, 3];
        let element2 = RingElement::from_coefficients(coeffs2, modulus).unwrap();
        
        let sum = element.clone().add(element2.clone()).unwrap();
        let sum_norm = sum.infinity_norm();
        
        // Triangle inequality: ||a + b||_∞ ≤ ||a||_∞ + ||b||_∞
        assert!(sum_norm <= element.infinity_norm() + element2.infinity_norm());
        
        // Test bound checking
        let bound_check = InfinityNorm::compute_with_bound_check(&coeffs, 5);
        assert_eq!(bound_check, Some(5));
        
        let bound_check_fail = InfinityNorm::compute_with_bound_check(&coeffs, 4);
        assert_eq!(bound_check_fail, None);
        
        println!("✓ Norm computation integration test passed");
    }
    
    /// Test polynomial multiplication correctness across all algorithms
    #[test]
    fn test_polynomial_multiplication_correctness() {
        let dimensions = vec![32, 64, 128, 256, 512];
        let modulus = Some(1009i64);
        
        for d in dimensions {
            // Create test polynomials
            let coeffs1: Vec<i64> = (0..d).map(|i| (i % 10) as i64 - 5).collect();
            let coeffs2: Vec<i64> = (0..d).map(|i| ((i * 3) % 10) as i64 - 5).collect();
            
            let f = RingElement::from_coefficients(coeffs1, modulus).unwrap();
            let g = RingElement::from_coefficients(coeffs2, modulus).unwrap();
            
            // Test all multiplication algorithms
            let result_schoolbook = schoolbook_multiply_optimized(&f, &g).unwrap();
            let result_karatsuba = karatsuba_multiply_optimized(&f, &g).unwrap();
            let result_selection = multiply_with_algorithm_selection(&f, &g).unwrap();
            let result_ring_mul = f.mul(g).unwrap();
            
            // All algorithms should produce identical results
            assert_eq!(result_schoolbook.coefficients(), result_karatsuba.coefficients(),
                      "Schoolbook vs Karatsuba mismatch for dimension {}", d);
            assert_eq!(result_schoolbook.coefficients(), result_selection.coefficients(),
                      "Schoolbook vs Selection mismatch for dimension {}", d);
            assert_eq!(result_schoolbook.coefficients(), result_ring_mul.coefficients(),
                      "Schoolbook vs Ring multiplication mismatch for dimension {}", d);
            
            // Verify negacyclic property: X^d = -1
            if d >= 8 {
                let mut x_coeffs = vec![0i64; d];
                x_coeffs[1] = 1; // X
                let x = RingElement::from_coefficients(x_coeffs, modulus).unwrap();
                
                let mut x_power = RingElement::one(d, modulus).unwrap();
                for _ in 0..d {
                    x_power = x_power.mul(x.clone()).unwrap();
                }
                
                let neg_one = RingElement::one(d, modulus).unwrap().neg().unwrap();
                assert_eq!(x_power.coefficients(), neg_one.coefficients(),
                          "Negacyclic property failed for dimension {}", d);
            }
        }
        
        println!("✓ Polynomial multiplication correctness test passed");
    }
    
    /// Test balanced coefficient representation consistency
    #[test]
    fn test_balanced_representation_consistency() {
        let modulus = 1009i64;
        let barrett = BarrettParams::new(modulus).unwrap();
        
        // Test round-trip conversion for various values
        let test_values = vec![0, 1, 504, 505, 1008, 100, 900];
        
        for &val in &test_values {
            // Standard -> Balanced -> Standard
            let balanced = barrett.to_balanced(val as u128);
            let recovered = barrett.from_balanced(balanced);
            assert_eq!(recovered, val as u128, "Round-trip failed for value {}", val);
            
            // Verify balanced representation bounds
            assert!(balanced >= -barrett.half_modulus() && balanced <= barrett.half_modulus(),
                   "Balanced value {} out of bounds for modulus {}", balanced, modulus);
        }
        
        // Test with BalancedCoefficients structure
        let dimension = 16;
        let standard_coeffs = vec![0, 1, 504, 505, 1008, 100, 900, 200, 
                                  300, 400, 500, 600, 700, 800, 50, 950];
        
        let balanced_coeffs = BalancedCoefficients::from_standard(&standard_coeffs, modulus).unwrap();
        let recovered_standard = balanced_coeffs.to_standard();
        
        assert_eq!(recovered_standard, standard_coeffs);
        
        // Test that all coefficients are within bounds
        for &coeff in balanced_coeffs.coefficients() {
            assert!(coeff >= -barrett.half_modulus() && coeff <= barrett.half_modulus());
        }
        
        println!("✓ Balanced representation consistency test passed");
    }
    
    /// Test performance characteristics and algorithm selection
    #[test]
    fn test_performance_characteristics() {
        use std::time::Instant;
        
        let dimensions = vec![64, 128, 256, 512, 1024];
        let modulus = Some(1009i64);
        
        for d in dimensions {
            // Create test polynomials
            let coeffs1: Vec<i64> = (0..d).map(|i| (i % 20) as i64 - 10).collect();
            let coeffs2: Vec<i64> = (0..d).map(|i| ((i * 7) % 20) as i64 - 10).collect();
            
            let f = RingElement::from_coefficients(coeffs1, modulus).unwrap();
            let g = RingElement::from_coefficients(coeffs2, modulus).unwrap();
            
            // Measure schoolbook multiplication time
            let start = Instant::now();
            let _result_schoolbook = schoolbook_multiply_optimized(&f, &g).unwrap();
            let schoolbook_time = start.elapsed();
            
            // Measure Karatsuba multiplication time
            let start = Instant::now();
            let _result_karatsuba = karatsuba_multiply_optimized(&f, &g).unwrap();
            let karatsuba_time = start.elapsed();
            
            // Measure automatic selection time
            let start = Instant::now();
            let _result_selection = multiply_with_algorithm_selection(&f, &g).unwrap();
            let selection_time = start.elapsed();
            
            println!("Dimension {}: Schoolbook={:?}, Karatsuba={:?}, Selection={:?}",
                    d, schoolbook_time, karatsuba_time, selection_time);
            
            // For larger dimensions, Karatsuba should generally be faster
            if d >= 512 {
                // This is a heuristic check - actual performance depends on system
                println!("  Large dimension {} - Karatsuba vs Schoolbook ratio: {:.2}",
                        d, schoolbook_time.as_nanos() as f64 / karatsuba_time.as_nanos() as f64);
            }
        }
        
        println!("✓ Performance characteristics test completed");
    }
    
    /// Test error handling and edge cases
    #[test]
    fn test_error_handling() {
        // Test invalid dimensions
        assert!(RingElement::zero(63, Some(1009)).is_err()); // Not power of 2
        assert!(RingElement::zero(16, Some(1009)).is_err());  // Too small
        assert!(RingElement::zero(32768, Some(1009)).is_err()); // Too large
        
        // Test invalid modulus
        assert!(BarrettParams::new(0).is_err());
        assert!(BarrettParams::new(-1).is_err());
        assert!(MontgomeryParams::new(0).is_err());
        assert!(MontgomeryParams::new(2).is_err()); // Even modulus
        
        // Test incompatible operations
        let f1 = RingElement::zero(64, Some(1009)).unwrap();
        let f2 = RingElement::zero(128, Some(1009)).unwrap();
        assert!(f1.add(f2).is_err()); // Different dimensions
        
        let f3 = RingElement::zero(64, Some(1009)).unwrap();
        let f4 = RingElement::zero(64, Some(2017)).unwrap();
        assert!(f3.add(f4).is_err()); // Different moduli
        
        // Test coefficient bounds validation
        let invalid_coeffs = vec![1000000i64; 64]; // Too large for modulus 1009
        assert!(RingElement::from_coefficients(invalid_coeffs, Some(1009)).is_err());
        
        println!("✓ Error handling test passed");
    }
    
    /// Test constant-time operations for cryptographic security
    #[test]
    fn test_constant_time_operations() {
        use crate::norm_computation::constant_time;
        use crate::modular_arithmetic::constant_time::ConstantTimeModular;
        use subtle::Choice;
        
        let modulus = 1009i64;
        let ct_arith = ConstantTimeModular::new(modulus).unwrap();
        
        // Test constant-time modular operations
        let a = 123i64;
        let b = 456i64;
        
        let ct_sum = ct_arith.add_mod_ct(a, b);
        let ct_product = ct_arith.mul_mod_ct(a, b);
        
        // Results should match regular operations
        let regular_arith = ModularArithmetic::new(modulus).unwrap();
        assert_eq!(ct_sum, regular_arith.add_mod(a, b));
        assert_eq!(ct_product, regular_arith.mul_mod(a, b));
        
        // Test constant-time norm computation
        let coeffs = vec![1, -5, 3, -2, 4, 0, -1, 2];
        let ct_norm = constant_time::infinity_norm_ct(&coeffs);
        let regular_norm = InfinityNorm::compute_vector(&coeffs);
        assert_eq!(ct_norm, regular_norm);
        
        // Test constant-time bound checking
        let within_bound = constant_time::infinity_norm_bound_check_ct(&coeffs, 5);
        assert_eq!(within_bound.unwrap_u8(), 1);
        
        let exceeds_bound = constant_time::infinity_norm_bound_check_ct(&coeffs, 4);
        assert_eq!(exceeds_bound.unwrap_u8(), 0);
        
        // Test conditional selection
        let choice_true = Choice::from(1);
        let choice_false = Choice::from(0);
        
        assert_eq!(ConstantTimeModular::conditional_select(choice_true, 100, 200), 100);
        assert_eq!(ConstantTimeModular::conditional_select(choice_false, 100, 200), 200);
        
        println!("✓ Constant-time operations test passed");
    }
    
    /// Comprehensive integration test combining all components
    #[test]
    fn test_comprehensive_integration() {
        let dimension = 128;
        let modulus = Some(1009i64);
        
        // Create complex test scenario
        let coeffs1: Vec<i64> = (0..dimension).map(|i| ((i * 17) % 100) as i64 - 50).collect();
        let coeffs2: Vec<i64> = (0..dimension).map(|i| ((i * 23) % 100) as i64 - 50).collect();
        let coeffs3: Vec<i64> = (0..dimension).map(|i| ((i * 31) % 100) as i64 - 50).collect();
        
        let f = RingElement::from_coefficients(coeffs1, modulus).unwrap();
        let g = RingElement::from_coefficients(coeffs2, modulus).unwrap();
        let h = RingElement::from_coefficients(coeffs3, modulus).unwrap();
        
        // Test complex arithmetic expression: (f + g) * h - f * g
        let sum_fg = f.clone().add(g.clone()).unwrap();
        let product_sum_h = sum_fg.mul(h.clone()).unwrap();
        let product_fg = f.mul(g).unwrap();
        let final_result = product_sum_h.sub(product_fg).unwrap();
        
        // This should equal f * h (by distributivity)
        let expected = f.mul(h).unwrap();
        assert_eq!(final_result.coefficients(), expected.coefficients());
        
        // Test norm properties
        let result_norm = final_result.infinity_norm();
        let expected_norm = expected.infinity_norm();
        assert_eq!(result_norm, expected_norm);
        
        // Test modular arithmetic consistency
        let mod_arith = ModularArithmetic::new(modulus.unwrap()).unwrap();
        let test_a = final_result.coefficients()[0];
        let test_b = expected.coefficients()[0];
        assert_eq!(test_a, test_b);
        
        // Verify the result is within expected bounds
        let operator_bound = OperatorNorm::compute_upper_bound(&final_result);
        assert!(result_norm <= operator_bound);
        
        println!("✓ Comprehensive integration test passed");
        println!("  Final result norm: {}", result_norm);
        println!("  Operator bound: {}", operator_bound);
        println!("  Dimension: {}", dimension);
        println!("  Modulus: {:?}", modulus);
    }
    
    /// Run all integration tests
    #[test]
    fn run_all_integration_tests() {
        println!("Running LatticeFold+ Core Mathematical Infrastructure Integration Tests");
        println!("================================================================");
        
        test_ring_arithmetic_integration();
        test_modular_arithmetic_integration();
        test_norm_computation_integration();
        test_polynomial_multiplication_correctness();
        test_balanced_representation_consistency();
        test_performance_characteristics();
        test_error_handling();
        test_constant_time_operations();
        test_comprehensive_integration();
        
        println!("================================================================");
        println!("✅ All integration tests passed successfully!");
        println!("Core mathematical infrastructure is ready for LatticeFold+ implementation.");
    }
}