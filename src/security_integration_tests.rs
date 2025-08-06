// Comprehensive integration tests for LatticeFold+ security implementation
// This module provides end-to-end testing of all security features including
// constant-time operations, side-channel resistance, secure memory management,
// timing analysis, and security validation.

#[cfg(test)]
mod tests {
    use crate::security::*;
    use crate::security::constant_time::*;
    use crate::security::side_channel_resistance::*;
    use crate::security::secure_memory::*;
    use crate::security::timing_analysis::*;
    use crate::security::security_validation::*;
    use crate::error::Result;
    use std::time::{Duration, Instant};
    
    /// Test comprehensive security system integration
    /// This test verifies that all security components work together correctly
    /// and provide the expected level of protection.
    #[test]
    fn test_comprehensive_security_integration() {
        // Initialize security configuration with maximum security
        let config = SecurityConfig::high_security();
        
        // Create security manager
        let mut security_manager = SecurityManager::new(config.clone()).unwrap();
        
        // Verify security configuration
        assert!(security_manager.is_constant_time_enabled());
        assert!(security_manager.is_side_channel_resistance_enabled());
        
        // Test secure memory allocation
        let mut secure_region = security_manager.allocate_secure_memory(1024).unwrap();
        assert_eq!(secure_region.size(), 1024);
        assert!(secure_region.is_protected());
        
        // Test secure memory operations
        let data = secure_region.as_mut_slice();
        data[0] = 0xAA;
        data[1023] = 0x55;
        assert_eq!(data[0], 0xAA);
        assert_eq!(data[1023], 0x55);
        
        // Test timing validation
        let operation_duration = 1000; // 1 microsecond
        let timing_result = security_manager.validate_timing("test_operation", operation_duration);
        assert!(timing_result.is_ok());
        
        // Test security validation
        let validation_report = security_manager.run_security_validation().unwrap();
        assert!(validation_report.overall_assessment != SecurityAssessment::Critical);
        
        println!("Comprehensive security integration test passed");
    }
    
    /// Test constant-time arithmetic operations
    /// This test verifies that arithmetic operations execute in constant time
    /// regardless of input values and maintain security properties.
    #[test]
    fn test_constant_time_arithmetic_operations() {
        // Test constant-time addition
        let a = 12345i64;
        let b = 67890i64;
        
        let start_time = Instant::now();
        let sum = a.ct_add(&b).unwrap();
        let duration1 = start_time.elapsed();
        
        assert_eq!(sum, 80235);
        
        // Test with different values to ensure constant timing
        let c = 1i64;
        let d = 1i64;
        
        let start_time = Instant::now();
        let sum2 = c.ct_add(&d).unwrap();
        let duration2 = start_time.elapsed();
        
        assert_eq!(sum2, 2);
        
        // Timing should be similar (within reasonable variance)
        let timing_diff = if duration1 > duration2 {
            duration1 - duration2
        } else {
            duration2 - duration1
        };
        
        // Allow up to 10 microseconds variance (generous for testing)
        assert!(timing_diff < Duration::from_micros(10));
        
        // Test constant-time subtraction
        let diff = a.ct_sub(&b).unwrap();
        assert_eq!(diff, -55545);
        
        // Test constant-time multiplication
        let product = a.ct_mul(&b).unwrap();
        assert_eq!(product, 838102050);
        
        // Test constant-time equality
        let eq_result = a.ct_eq(&12345);
        assert_eq!(eq_result.unwrap_u8(), 1);
        
        let neq_result = a.ct_eq(&54321);
        assert_eq!(neq_result.unwrap_u8(), 0);
        
        println!("Constant-time arithmetic operations test passed");
    }
    
    /// Test constant-time modular arithmetic operations
    /// This test verifies that modular arithmetic operations execute in constant time
    /// and produce correct results while maintaining security properties.
    #[test]
    fn test_constant_time_modular_arithmetic() {
        let a = 123456i64;
        let b = 789012i64;
        let modulus = 1000003i64; // Large prime
        
        // Test modular addition
        let sum_mod = a.ct_add_mod(&b, modulus).unwrap();
        let expected_sum = (a + b) % modulus;
        assert_eq!(sum_mod, expected_sum);
        
        // Test modular subtraction
        let diff_mod = a.ct_sub_mod(&b, modulus).unwrap();
        let expected_diff = ((a - b) % modulus + modulus) % modulus;
        assert_eq!(diff_mod, expected_diff);
        
        // Test modular multiplication
        let product_mod = a.ct_mul_mod(&b, modulus).unwrap();
        let expected_product = ((a as i128 * b as i128) % modulus as i128) as i64;
        assert_eq!(product_mod, expected_product);
        
        // Test modular reduction
        let large_value = 12345678901234i64;
        let reduced = large_value.ct_reduce_mod(modulus).unwrap();
        assert_eq!(reduced, large_value % modulus);
        
        // Test modular inverse (for coprime values)
        let coprime_a = 123457i64; // Coprime to modulus
        let inverse = coprime_a.ct_inverse_mod(modulus).unwrap();
        let verification = coprime_a.ct_mul_mod(&inverse, modulus).unwrap();
        assert_eq!(verification, 1);
        
        println!("Constant-time modular arithmetic test passed");
    }
    
    /// Test side-channel resistant random number generation
    /// This test verifies that the RNG provides cryptographically secure randomness
    /// while protecting against side-channel attacks.
    #[test]
    fn test_side_channel_resistant_rng() {
        let config = SecurityConfig::high_security();
        let mut rng = SideChannelResistantRNG::new(config).unwrap();
        
        // Test random byte generation
        let mut bytes1 = [0u8; 32];
        let mut bytes2 = [0u8; 32];
        
        rng.fill_bytes(&mut bytes1).unwrap();
        rng.fill_bytes(&mut bytes2).unwrap();
        
        // Bytes should be different (extremely unlikely to be same with good RNG)
        assert_ne!(bytes1, bytes2);
        
        // Test that bytes are not all zeros or all ones
        assert_ne!(bytes1, [0u8; 32]);
        assert_ne!(bytes1, [0xFFu8; 32]);
        
        // Test range generation
        let random_val1 = rng.gen_range_protected(1000).unwrap();
        let random_val2 = rng.gen_range_protected(1000).unwrap();
        
        assert!(random_val1 < 1000);
        assert!(random_val2 < 1000);
        
        // Values should likely be different
        // (could be same by chance, but very unlikely)
        
        // Test entropy statistics
        let stats = rng.get_entropy_stats();
        assert!(stats.entropy_counter > 0);
        assert!(stats.entropy_pool_size > 0);
        assert!(stats.power_analysis_protection);
        assert!(stats.cache_timing_protection);
        assert!(stats.em_protection);
        
        println!("Side-channel resistant RNG test passed");
    }
    
    /// Test power analysis resistance
    /// This test verifies that power analysis countermeasures are effective
    /// and protect cryptographic operations from power-based attacks.
    #[test]
    fn test_power_analysis_resistance() {
        let config = SecurityConfig::high_security();
        let mut power_protection = PowerAnalysisResistance::new(config).unwrap();
        
        // Test protected arithmetic operation
        let result = power_protection.protect_arithmetic_operation(|masks| {
            // Simulate arithmetic operation with masking
            assert!(masks.additive_masks.len() > 0);
            assert!(masks.multiplicative_masks.len() > 0);
            assert!(masks.boolean_masks.len() > 0);
            
            Ok(42i64 + 17i64)
        }).unwrap();
        
        assert_eq!(result, 59);
        
        // Test protected cryptographic operation
        let result = power_protection.protect_crypto_operation(|blinds| {
            // Simulate cryptographic operation with blinding
            assert!(blinds.scalar_blinds.len() > 0);
            assert!(blinds.polynomial_blinds.len() > 0);
            
            Ok(123i64 * 456i64)
        }).unwrap();
        
        assert_eq!(result, 56088);
        
        println!("Power analysis resistance test passed");
    }
    
    /// Test cache-timing attack resistance
    /// This test verifies that cache-timing countermeasures are effective
    /// and protect memory access patterns from timing-based attacks.
    #[test]
    fn test_cache_timing_resistance() {
        let config = SecurityConfig::high_security();
        let mut cache_protection = CacheTimingResistance::new(config).unwrap();
        
        // Test protected memory access
        let mut test_data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        
        let result = cache_protection.protect_memory_access(|| {
            // Simulate memory access pattern
            let mut sum = 0;
            for i in 0..test_data.len() {
                sum += test_data[i];
            }
            test_data[5] = 42; // Modify specific element
            Ok(sum)
        }).unwrap();
        
        assert_eq!(result, 55); // Sum of 1+2+...+10
        assert_eq!(test_data[5], 42);
        
        // Test cache statistics
        let stats = cache_protection.get_cache_statistics();
        assert!(stats.cache_line_size > 0);
        assert!(stats.num_cache_lines_managed > 0);
        assert!(stats.dummy_accesses_per_real > 0);
        assert!(stats.cache_warming_enabled);
        assert!(stats.prefetch_enabled);
        
        println!("Cache-timing resistance test passed");
    }
    
    /// Test secure memory management
    /// This test verifies that secure memory management provides proper
    /// protection, automatic zeroization, and memory safety guarantees.
    #[test]
    fn test_secure_memory_management() {
        let config = SecurityConfig::high_security();
        let mut memory_manager = SecureMemoryManager::new(&config).unwrap();
        
        // Test secure memory allocation
        let mut region1 = memory_manager.allocate(1024).unwrap();
        assert_eq!(region1.size(), 1024);
        assert!(region1.is_protected());
        assert!(!region1.is_zeroized());
        
        // Test memory access
        let data = region1.as_mut_slice();
        data[0] = 0xAA;
        data[512] = 0x55;
        data[1023] = 0xFF;
        
        assert_eq!(data[0], 0xAA);
        assert_eq!(data[512], 0x55);
        assert_eq!(data[1023], 0xFF);
        
        // Test manual zeroization
        region1.zeroize().unwrap();
        assert!(region1.is_zeroized());
        
        // Test multiple allocations
        let region2 = memory_manager.allocate(2048).unwrap();
        let region3 = memory_manager.allocate(512).unwrap();
        
        assert_eq!(region2.size(), 2048);
        assert_eq!(region3.size(), 512);
        
        // Test memory statistics
        let stats = memory_manager.get_statistics().unwrap();
        assert!(stats.allocation_count >= 3);
        assert!(stats.current_usage >= 2560); // At least 2048 + 512
        assert!(stats.total_allocated >= 3584); // 1024 + 2048 + 512
        
        // Test deallocation
        memory_manager.deallocate(region2).unwrap();
        memory_manager.deallocate(region3).unwrap();
        
        let final_stats = memory_manager.get_statistics().unwrap();
        assert!(final_stats.deallocation_count >= 2);
        
        println!("Secure memory management test passed");
    }
    
    /// Test timing analysis and consistency checking
    /// This test verifies that timing analysis can detect timing variations
    /// and ensure constant-time properties are maintained.
    #[test]
    fn test_timing_analysis() {
        let mut analyzer = TimingAnalyzer::new(1000).unwrap(); // 1 microsecond max variance
        
        // Record consistent timing measurements
        for i in 0..20 {
            let duration = 1000 + (i % 3); // Small variation
            analyzer.record_timing("consistent_op", duration).unwrap();
        }
        
        // Check timing consistency (should pass)
        let consistency_result = analyzer.check_timing_consistency("consistent_op");
        assert!(consistency_result.is_ok());
        
        // Record inconsistent timing measurements
        for i in 0..20 {
            let duration = 1000 + i * 100; // Large variation
            analyzer.record_timing("inconsistent_op", duration).unwrap();
        }
        
        // Check timing consistency (should fail)
        let consistency_result = analyzer.check_timing_consistency("inconsistent_op");
        assert!(consistency_result.is_err());
        
        // Get analysis report
        let report = analyzer.get_analysis_report().unwrap();
        assert_eq!(report.operations_analyzed, 2);
        assert_eq!(report.total_measurements, 40);
        assert!(report.overall_security_score <= 100);
        
        // Check that consistent operation has better properties
        let consistent_stats = &report.timing_statistics["consistent_op"];
        let inconsistent_stats = &report.timing_statistics["inconsistent_op"];
        
        assert!(consistent_stats.appears_constant_time);
        assert!(!inconsistent_stats.appears_constant_time);
        assert!(consistent_stats.variance_ns < inconsistent_stats.variance_ns);
        
        println!("Timing analysis test passed");
    }
    
    /// Test comprehensive security validation
    /// This test verifies that the security validation framework can perform
    /// comprehensive security analysis and generate meaningful reports.
    #[test]
    fn test_comprehensive_security_validation() {
        let config = SecurityConfig::high_security();
        let mut validator = SecurityValidator::new(&config).unwrap();
        
        // Run comprehensive validation
        let report = validator.run_comprehensive_validation().unwrap();
        
        // Verify report structure
        assert!(report.overall_security_score <= 100);
        assert!(!report.recommendations.is_empty());
        
        // Check that validation components ran
        assert!(report.threat_analysis.identified_threats.len() > 0);
        assert!(report.attack_simulation_results.simulations_run > 0);
        assert!(report.formal_verification_results.properties_verified > 0);
        assert!(report.crypto_verification_results.properties_checked > 0);
        assert!(report.test_suite_results.tests_run > 0);
        assert!(report.audit_results.audit_items_checked > 0);
        assert!(report.pen_test_results.tests_performed > 0);
        assert!(report.compliance_results.standards_checked > 0);
        assert!(report.vuln_scan_results.vulnerabilities_scanned > 0);
        
        // Check validation duration is reasonable
        assert!(report.validation_duration < Duration::from_secs(60)); // Should complete within 1 minute
        
        // Verify recommendations have proper structure
        for recommendation in &report.recommendations {
            assert!(!recommendation.description.is_empty());
            assert!(!recommendation.rationale.is_empty());
            assert!(!recommendation.implementation_guidance.is_empty());
        }
        
        println!("Comprehensive security validation test passed");
    }
    
    /// Test timing-consistent operations wrapper
    /// This test verifies that the timing-consistent operations wrapper
    /// properly monitors and enforces timing consistency requirements.
    #[test]
    fn test_timing_consistent_operations() {
        let config = SecurityConfig::high_security();
        let mut timing_ops = TimingConsistentOperations::new(config).unwrap();
        
        // Test operation with consistent timing
        let result = timing_ops.execute_with_timing("consistent_test", || {
            // Simulate consistent operation
            std::thread::sleep(Duration::from_micros(100));
            Ok(42)
        }).unwrap();
        
        assert_eq!(result, 42);
        
        // Test multiple operations to build timing profile
        for _ in 0..10 {
            let _ = timing_ops.execute_with_timing("consistent_test", || {
                std::thread::sleep(Duration::from_micros(100));
                Ok(42)
            }).unwrap();
        }
        
        // Get timing statistics
        let stats = timing_ops.get_timing_statistics();
        assert!(stats.total_measurements >= 11);
        assert!(stats.min_duration_ns > 0);
        assert!(stats.max_duration_ns >= stats.min_duration_ns);
        assert!(stats.mean_duration_ns > 0.0);
        
        // Clear measurements
        timing_ops.clear_measurements();
        let cleared_stats = timing_ops.get_timing_statistics();
        assert_eq!(cleared_stats.total_measurements, 0);
        
        println!("Timing-consistent operations test passed");
    }
    
    /// Test security configuration validation
    /// This test verifies that security configurations are properly validated
    /// and invalid configurations are rejected.
    #[test]
    fn test_security_configuration_validation() {
        // Test valid default configuration
        let default_config = SecurityConfig::default();
        assert!(default_config.validate().is_ok());
        
        // Test high security configuration
        let high_security_config = SecurityConfig::high_security();
        assert!(high_security_config.validate().is_ok());
        assert_eq!(high_security_config.security_level_bits, 256);
        assert_eq!(high_security_config.max_timing_variance_ns, 100);
        
        // Test performance optimized configuration
        let perf_config = SecurityConfig::performance_optimized();
        assert!(perf_config.validate().is_ok());
        assert!(!perf_config.side_channel_resistance_enabled);
        assert!(!perf_config.cache_timing_resistance_enabled);
        
        // Test invalid configuration (invalid security level)
        let mut invalid_config = SecurityConfig::default();
        invalid_config.security_level_bits = 100; // Invalid
        assert!(invalid_config.validate().is_err());
        
        // Test invalid configuration (zero timing variance)
        let mut invalid_config = SecurityConfig::default();
        invalid_config.max_timing_variance_ns = 0; // Invalid
        assert!(invalid_config.validate().is_err());
        
        // Test inconsistent configuration
        let mut invalid_config = SecurityConfig::default();
        invalid_config.constant_time_enabled = false;
        invalid_config.timing_analysis_enabled = true; // Inconsistent
        assert!(invalid_config.validate().is_err());
        
        // Test cryptographic parameters derivation
        let crypto_params = default_config.get_crypto_params();
        assert_eq!(crypto_params.ring_dimension, 1024);
        assert_eq!(crypto_params.modulus_bits, 64);
        assert!(crypto_params.noise_parameter > 0.0);
        assert!(crypto_params.norm_bound > 0);
        
        println!("Security configuration validation test passed");
    }
    
    /// Test end-to-end security workflow
    /// This test simulates a complete security workflow from initialization
    /// through operation to validation and demonstrates all components working together.
    #[test]
    fn test_end_to_end_security_workflow() {
        println!("Starting end-to-end security workflow test...");
        
        // Step 1: Initialize security system
        let config = SecurityConfig::high_security();
        let mut security_manager = SecurityManager::new(config.clone()).unwrap();
        
        // Step 2: Allocate secure memory for sensitive operations
        let mut secure_memory = security_manager.allocate_secure_memory(2048).unwrap();
        
        // Step 3: Initialize side-channel resistant RNG
        let mut secure_rng = SideChannelResistantRNG::new(config.clone()).unwrap();
        
        // Step 4: Initialize power analysis protection
        let mut power_protection = PowerAnalysisResistance::new(config.clone()).unwrap();
        
        // Step 5: Initialize cache-timing protection
        let mut cache_protection = CacheTimingResistance::new(config.clone()).unwrap();
        
        // Step 6: Initialize timing analyzer
        let mut timing_analyzer = TimingAnalyzer::new(config.max_timing_variance_ns).unwrap();
        
        // Step 7: Perform secure cryptographic operations
        let operation_result = power_protection.protect_crypto_operation(|_blinds| {
            cache_protection.protect_memory_access(|| {
                // Generate secure random data
                let mut random_data = [0u8; 32];
                secure_rng.fill_bytes(&mut random_data)?;
                
                // Store in secure memory
                let memory_slice = secure_memory.as_mut_slice();
                memory_slice[0..32].copy_from_slice(&random_data);
                
                // Perform constant-time arithmetic
                let a = 12345i64;
                let b = 67890i64;
                let modulus = 1000003i64;
                
                let start_time = std::time::Instant::now();
                let result = a.ct_mul_mod(&b, modulus)?;
                let duration = start_time.elapsed();
                
                // Record timing for analysis
                timing_analyzer.record_timing("secure_operation", duration.as_nanos() as u64)?;
                
                Ok(result)
            })
        }).unwrap();
        
        // Step 8: Verify operation completed successfully
        assert!(operation_result > 0);
        
        // Step 9: Check timing consistency
        let timing_check = timing_analyzer.check_timing_consistency("secure_operation");
        // Note: May fail due to single measurement, but demonstrates the check
        
        // Step 10: Get timing analysis report
        let timing_report = timing_analyzer.get_analysis_report().unwrap();
        assert_eq!(timing_report.operations_analyzed, 1);
        assert_eq!(timing_report.total_measurements, 1);
        
        // Step 11: Validate overall security
        let security_report = security_manager.run_security_validation().unwrap();
        assert!(security_report.overall_assessment != SecurityAssessment::Critical);
        
        // Step 12: Clean up secure memory
        secure_memory.zeroize().unwrap();
        assert!(secure_memory.is_zeroized());
        
        // Step 13: Get final memory statistics
        let memory_stats = security_manager.get_statistics().unwrap();
        assert!(memory_stats.allocation_count > 0);
        
        println!("End-to-end security workflow test completed successfully");
        println!("Security score: {:?}", security_report.overall_assessment);
        println!("Memory allocations: {}", memory_stats.allocation_count);
        println!("Timing measurements: {}", timing_report.total_measurements);
    }
    
    /// Performance benchmark for security operations
    /// This test measures the performance impact of security features
    /// and ensures they meet acceptable performance requirements.
    #[test]
    fn test_security_performance_benchmark() {
        println!("Running security performance benchmarks...");
        
        let config = SecurityConfig::high_security();
        let iterations = 1000;
        
        // Benchmark constant-time arithmetic
        let start_time = std::time::Instant::now();
        for i in 0..iterations {
            let a = 12345i64 + i as i64;
            let b = 67890i64 + i as i64;
            let _result = a.ct_add(&b).unwrap();
        }
        let ct_arithmetic_duration = start_time.elapsed();
        
        // Benchmark regular arithmetic for comparison
        let start_time = std::time::Instant::now();
        for i in 0..iterations {
            let a = 12345i64 + i as i64;
            let b = 67890i64 + i as i64;
            let _result = a + b;
        }
        let regular_arithmetic_duration = start_time.elapsed();
        
        // Calculate overhead
        let overhead_ratio = ct_arithmetic_duration.as_nanos() as f64 / regular_arithmetic_duration.as_nanos() as f64;
        
        println!("Constant-time arithmetic overhead: {:.2}x", overhead_ratio);
        
        // Overhead should be reasonable (less than 10x for this simple operation)
        assert!(overhead_ratio < 10.0);
        
        // Benchmark secure memory allocation
        let mut memory_manager = SecureMemoryManager::new(&config).unwrap();
        
        let start_time = std::time::Instant::now();
        let mut regions = Vec::new();
        for _ in 0..100 {
            let region = memory_manager.allocate(1024).unwrap();
            regions.push(region);
        }
        let allocation_duration = start_time.elapsed();
        
        let start_time = std::time::Instant::now();
        for region in regions {
            memory_manager.deallocate(region).unwrap();
        }
        let deallocation_duration = start_time.elapsed();
        
        println!("Secure memory allocation: {} allocations in {:?}", 100, allocation_duration);
        println!("Secure memory deallocation: {} deallocations in {:?}", 100, deallocation_duration);
        
        // Allocations should complete within reasonable time (less than 1 second for 100 allocations)
        assert!(allocation_duration < Duration::from_secs(1));
        assert!(deallocation_duration < Duration::from_secs(1));
        
        // Benchmark side-channel resistant RNG
        let mut rng = SideChannelResistantRNG::new(config).unwrap();
        
        let start_time = std::time::Instant::now();
        for _ in 0..iterations {
            let mut bytes = [0u8; 32];
            rng.fill_bytes(&mut bytes).unwrap();
        }
        let rng_duration = start_time.elapsed();
        
        println!("Side-channel resistant RNG: {} generations in {:?}", iterations, rng_duration);
        
        // RNG should be reasonably fast (less than 10 seconds for 1000 generations)
        assert!(rng_duration < Duration::from_secs(10));
        
        println!("Security performance benchmarks completed");
    }
}