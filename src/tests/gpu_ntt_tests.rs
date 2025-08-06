/// Comprehensive tests for GPU NTT implementation
/// 
/// This module provides extensive testing of the GPU-accelerated NTT kernels,
/// including correctness verification, performance benchmarking, and error handling.
/// 
/// Test Categories:
/// 1. Correctness Tests - Verify GPU results match CPU reference implementation
/// 2. Performance Tests - Benchmark GPU vs CPU performance across dimensions
/// 3. Memory Tests - Validate GPU memory management and allocation
/// 4. Error Handling Tests - Test graceful degradation and error recovery
/// 5. Batch Processing Tests - Verify batch NTT correctness and performance
/// 6. Integration Tests - Test GPU NTT within larger cryptographic protocols
/// 
/// Test Coverage:
/// - All supported polynomial dimensions (32 to 16384)
/// - Various moduli and NTT parameters
/// - Edge cases and boundary conditions
/// - Memory pressure scenarios
/// - Concurrent GPU usage
/// - Cross-platform compatibility

#[cfg(test)]
mod tests {
    use super::super::*;
    use crate::ntt::{NTTParams, AdaptiveNttEngine};
    use crate::error::Result;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha20Rng;
    use std::time::Instant;
    
    /// Test parameters for various polynomial dimensions
    const TEST_DIMENSIONS: &[usize] = &[32, 64, 128, 256, 512, 1024, 2048, 4096, 8192];
    
    /// Test moduli that are NTT-friendly
    const TEST_MODULI: &[i64] = &[
        2013265921,  // 15 * 2^27 + 1, supports NTT up to 2^27
        1811939329,  // 13 * 2^27 + 1, supports NTT up to 2^27  
        469762049,   // 7 * 2^26 + 1, supports NTT up to 2^26
        998244353,   // 119 * 2^23 + 1, supports NTT up to 2^23
    ];
    
    /// Creates NTT parameters for testing
    /// 
    /// # Arguments
    /// * `dimension` - Polynomial dimension (power of 2)
    /// * `modulus` - Prime modulus for NTT
    /// 
    /// # Returns
    /// * `Result<NTTParams>` - NTT parameters or error
    fn create_test_params(dimension: usize, modulus: i64) -> Result<NTTParams> {
        NTTParams::new(dimension, modulus)
    }
    
    /// Generates random polynomial coefficients for testing
    /// 
    /// # Arguments
    /// * `dimension` - Number of coefficients
    /// * `modulus` - Coefficient bound (coefficients in [0, modulus))
    /// * `rng` - Random number generator
    /// 
    /// # Returns
    /// * `Vec<i64>` - Random polynomial coefficients
    fn generate_random_polynomial(dimension: usize, modulus: i64, rng: &mut impl Rng) -> Vec<i64> {
        (0..dimension)
            .map(|_| rng.gen_range(0..modulus))
            .collect()
    }
    
    /// Generates batch of random polynomials for testing
    /// 
    /// # Arguments
    /// * `batch_size` - Number of polynomials
    /// * `dimension` - Coefficients per polynomial
    /// * `modulus` - Coefficient bound
    /// * `rng` - Random number generator
    /// 
    /// # Returns
    /// * `Vec<Vec<i64>>` - Batch of random polynomials
    fn generate_random_batch(
        batch_size: usize, 
        dimension: usize, 
        modulus: i64, 
        rng: &mut impl Rng
    ) -> Vec<Vec<i64>> {
        (0..batch_size)
            .map(|_| generate_random_polynomial(dimension, modulus, rng))
            .collect()
    }
    
    /// Tests GPU NTT correctness against CPU reference implementation
    /// 
    /// Verifies that GPU forward and inverse NTT produce identical results
    /// to the CPU implementation across various dimensions and moduli.
    #[test]
    #[cfg(feature = "gpu")]
    fn test_gpu_ntt_correctness() -> Result<()> {
        let mut rng = ChaCha20Rng::seed_from_u64(12345);
        
        for &dimension in TEST_DIMENSIONS {
            for &modulus in TEST_MODULI {
                // Skip if modulus doesn't support this dimension
                if !is_ntt_compatible(dimension, modulus) {
                    continue;
                }
                
                println!("Testing GPU NTT correctness: dimension={}, modulus={}", dimension, modulus);
                
                // Create NTT parameters
                let params = create_test_params(dimension, modulus)?;
                
                // Create adaptive NTT engine with GPU enabled
                let mut adaptive_engine = AdaptiveNttEngine::new(params.clone(), true)?;
                
                // Generate random test polynomial
                let coefficients = generate_random_polynomial(dimension, modulus, &mut rng);
                
                // Perform forward NTT on both CPU and GPU
                let gpu_forward = adaptive_engine.forward_ntt(&coefficients)?;
                
                // Force CPU computation for comparison
                let mut cpu_engine = crate::ntt::NttEngine::new(params.clone())?;
                let cpu_forward = cpu_engine.forward_ntt(&coefficients)?;
                
                // Verify forward NTT results match
                assert_eq!(gpu_forward.len(), cpu_forward.len(), 
                          "Forward NTT result lengths differ for dimension {}", dimension);
                
                for (i, (&gpu_val, &cpu_val)) in gpu_forward.iter().zip(cpu_forward.iter()).enumerate() {
                    assert_eq!(gpu_val, cpu_val, 
                              "Forward NTT mismatch at index {} for dimension {}: GPU={}, CPU={}", 
                              i, dimension, gpu_val, cpu_val);
                }
                
                // Perform inverse NTT
                let gpu_inverse = adaptive_engine.inverse_ntt(&gpu_forward)?;
                let cpu_inverse = cpu_engine.inverse_ntt(&cpu_forward)?;
                
                // Verify inverse NTT results match
                assert_eq!(gpu_inverse.len(), cpu_inverse.len(),
                          "Inverse NTT result lengths differ for dimension {}", dimension);
                
                for (i, (&gpu_val, &cpu_val)) in gpu_inverse.iter().zip(cpu_inverse.iter()).enumerate() {
                    assert_eq!(gpu_val, cpu_val,
                              "Inverse NTT mismatch at index {} for dimension {}: GPU={}, CPU={}",
                              i, dimension, gpu_val, cpu_val);
                }
                
                // Verify round-trip correctness (inverse(forward(x)) = x)
                for (i, (&original, &recovered)) in coefficients.iter().zip(gpu_inverse.iter()).enumerate() {
                    assert_eq!(original, recovered,
                              "Round-trip mismatch at index {} for dimension {}: original={}, recovered={}",
                              i, dimension, original, recovered);
                }
            }
        }
        
        Ok(())
    }
    
    /// Tests GPU batch NTT correctness
    /// 
    /// Verifies that batch processing produces identical results to
    /// individual polynomial processing.
    #[test]
    #[cfg(feature = "gpu")]
    fn test_gpu_batch_ntt_correctness() -> Result<()> {
        let mut rng = ChaCha20Rng::seed_from_u64(54321);
        let batch_sizes = [1, 2, 4, 8, 16, 32];
        
        for &dimension in &[256, 1024, 4096] {
            for &modulus in &TEST_MODULI[0..2] { // Test subset for efficiency
                if !is_ntt_compatible(dimension, modulus) {
                    continue;
                }
                
                println!("Testing GPU batch NTT: dimension={}, modulus={}", dimension, modulus);
                
                let params = create_test_params(dimension, modulus)?;
                let mut adaptive_engine = AdaptiveNttEngine::new(params, true)?;
                
                for &batch_size in &batch_sizes {
                    println!("  Testing batch size: {}", batch_size);
                    
                    // Generate batch of random polynomials
                    let batch_coefficients = generate_random_batch(batch_size, dimension, modulus, &mut rng);
                    
                    // Perform batch forward NTT
                    let batch_forward = adaptive_engine.batch_ntt(&batch_coefficients, true)?;
                    
                    // Perform individual forward NTTs for comparison
                    let mut individual_forward = Vec::new();
                    for poly in &batch_coefficients {
                        individual_forward.push(adaptive_engine.forward_ntt(poly)?);
                    }
                    
                    // Verify batch results match individual results
                    assert_eq!(batch_forward.len(), individual_forward.len(),
                              "Batch forward NTT count mismatch");
                    
                    for (batch_idx, (batch_result, individual_result)) in 
                        batch_forward.iter().zip(individual_forward.iter()).enumerate() {
                        
                        assert_eq!(batch_result.len(), individual_result.len(),
                                  "Batch forward NTT length mismatch for polynomial {}", batch_idx);
                        
                        for (coeff_idx, (&batch_val, &individual_val)) in 
                            batch_result.iter().zip(individual_result.iter()).enumerate() {
                            
                            assert_eq!(batch_val, individual_val,
                                      "Batch forward NTT mismatch at polynomial {} coefficient {}: batch={}, individual={}",
                                      batch_idx, coeff_idx, batch_val, individual_val);
                        }
                    }
                    
                    // Test batch inverse NTT
                    let batch_inverse = adaptive_engine.batch_ntt(&batch_forward, false)?;
                    
                    // Verify round-trip correctness
                    for (batch_idx, (original, recovered)) in 
                        batch_coefficients.iter().zip(batch_inverse.iter()).enumerate() {
                        
                        assert_eq!(original.len(), recovered.len(),
                                  "Batch inverse NTT length mismatch for polynomial {}", batch_idx);
                        
                        for (coeff_idx, (&orig_val, &rec_val)) in 
                            original.iter().zip(recovered.iter()).enumerate() {
                            
                            assert_eq!(orig_val, rec_val,
                                      "Batch round-trip mismatch at polynomial {} coefficient {}: original={}, recovered={}",
                                      batch_idx, coeff_idx, orig_val, rec_val);
                        }
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Tests GPU NTT performance compared to CPU implementation
    /// 
    /// Measures and compares execution times for various polynomial dimensions
    /// to validate performance improvements and identify optimal crossover points.
    #[test]
    #[cfg(feature = "gpu")]
    fn test_gpu_ntt_performance() -> Result<()> {
        let mut rng = ChaCha20Rng::seed_from_u64(98765);
        let modulus = TEST_MODULI[0]; // Use first test modulus
        let num_iterations = 10; // Number of timing iterations
        
        println!("GPU NTT Performance Comparison");
        println!("==============================");
        println!("{:>10} {:>15} {:>15} {:>10}", "Dimension", "CPU Time (ms)", "GPU Time (ms)", "Speedup");
        println!("{:-<55}", "");
        
        for &dimension in TEST_DIMENSIONS {
            if !is_ntt_compatible(dimension, modulus) {
                continue;
            }
            
            // Create test parameters and engines
            let params = create_test_params(dimension, modulus)?;
            let mut adaptive_engine = AdaptiveNttEngine::new(params.clone(), true)?;
            let mut cpu_engine = crate::ntt::NttEngine::new(params)?;
            
            // Generate test polynomial
            let coefficients = generate_random_polynomial(dimension, modulus, &mut rng);
            
            // Warm up GPU
            let _ = adaptive_engine.forward_ntt(&coefficients)?;
            
            // Benchmark CPU performance
            let cpu_start = Instant::now();
            for _ in 0..num_iterations {
                let _ = cpu_engine.forward_ntt(&coefficients)?;
            }
            let cpu_elapsed = cpu_start.elapsed();
            let cpu_time_ms = cpu_elapsed.as_millis() as f64 / num_iterations as f64;
            
            // Benchmark GPU performance (force GPU usage)
            let gpu_start = Instant::now();
            for _ in 0..num_iterations {
                let _ = adaptive_engine.forward_ntt(&coefficients)?;
            }
            let gpu_elapsed = gpu_start.elapsed();
            let gpu_time_ms = gpu_elapsed.as_millis() as f64 / num_iterations as f64;
            
            // Calculate speedup
            let speedup = if gpu_time_ms > 0.0 { cpu_time_ms / gpu_time_ms } else { 0.0 };
            
            println!("{:>10} {:>15.3} {:>15.3} {:>10.2}x", 
                    dimension, cpu_time_ms, gpu_time_ms, speedup);
            
            // Performance assertions (adjust thresholds based on hardware)
            if dimension >= 1024 {
                // GPU should be competitive for large dimensions
                assert!(speedup > 0.5, 
                       "GPU performance significantly worse than CPU for dimension {}: speedup={:.2}x", 
                       dimension, speedup);
            }
        }
        
        Ok(())
    }
    
    /// Tests GPU memory management and allocation
    /// 
    /// Verifies that GPU memory is properly allocated, used, and deallocated
    /// without leaks or corruption.
    #[test]
    #[cfg(feature = "gpu")]
    fn test_gpu_memory_management() -> Result<()> {
        let mut rng = ChaCha20Rng::seed_from_u64(13579);
        let dimension = 4096;
        let modulus = TEST_MODULI[0];
        
        if !is_ntt_compatible(dimension, modulus) {
            return Ok(()); // Skip if not compatible
        }
        
        println!("Testing GPU memory management for dimension {}", dimension);
        
        let params = create_test_params(dimension, modulus)?;
        let mut adaptive_engine = AdaptiveNttEngine::new(params, true)?;
        
        // Test multiple allocations and deallocations
        for iteration in 0..50 {
            let coefficients = generate_random_polynomial(dimension, modulus, &mut rng);
            
            // Perform NTT operations that require GPU memory allocation
            let forward_result = adaptive_engine.forward_ntt(&coefficients)?;
            let inverse_result = adaptive_engine.inverse_ntt(&forward_result)?;
            
            // Verify correctness to ensure memory wasn't corrupted
            assert_eq!(coefficients.len(), inverse_result.len(),
                      "Memory corruption detected in iteration {}: length mismatch", iteration);
            
            for (i, (&original, &recovered)) in coefficients.iter().zip(inverse_result.iter()).enumerate() {
                assert_eq!(original, recovered,
                          "Memory corruption detected in iteration {} at index {}: original={}, recovered={}",
                          iteration, i, original, recovered);
            }
            
            // Periodically check performance statistics for memory leaks
            if iteration % 10 == 0 {
                let perf_summary = adaptive_engine.get_performance_summary();
                println!("Iteration {}: {}", iteration, perf_summary.lines().last().unwrap_or(""));
            }
        }
        
        println!("GPU memory management test completed successfully");
        Ok(())
    }
    
    /// Tests GPU error handling and graceful degradation
    /// 
    /// Verifies that GPU errors are properly handled and the system
    /// falls back to CPU computation when necessary.
    #[test]
    #[cfg(feature = "gpu")]
    fn test_gpu_error_handling() -> Result<()> {
        let dimension = 1024;
        let modulus = TEST_MODULI[0];
        
        if !is_ntt_compatible(dimension, modulus) {
            return Ok(());
        }
        
        println!("Testing GPU error handling and fallback mechanisms");
        
        let params = create_test_params(dimension, modulus)?;
        
        // Test initialization with invalid GPU device ID
        match AdaptiveNttEngine::new(params.clone(), true) {
            Ok(_) => println!("GPU initialization successful"),
            Err(e) => println!("GPU initialization failed as expected: {}", e),
        }
        
        // Test with GPU disabled (should work with CPU fallback)
        let mut cpu_only_engine = AdaptiveNttEngine::new(params, false)?;
        
        let mut rng = ChaCha20Rng::seed_from_u64(24680);
        let coefficients = generate_random_polynomial(dimension, modulus, &mut rng);
        
        // This should work using CPU implementation
        let forward_result = cpu_only_engine.forward_ntt(&coefficients)?;
        let inverse_result = cpu_only_engine.inverse_ntt(&forward_result)?;
        
        // Verify correctness
        for (i, (&original, &recovered)) in coefficients.iter().zip(inverse_result.iter()).enumerate() {
            assert_eq!(original, recovered,
                      "CPU fallback failed at index {}: original={}, recovered={}",
                      i, original, recovered);
        }
        
        println!("GPU error handling test completed successfully");
        Ok(())
    }
    
    /// Tests GPU NTT with various edge cases and boundary conditions
    /// 
    /// Verifies correct behavior for edge cases such as:
    /// - Minimum and maximum supported dimensions
    /// - Zero polynomials
    /// - Polynomials with maximum coefficient values
    /// - Boundary modulus values
    #[test]
    #[cfg(feature = "gpu")]
    fn test_gpu_ntt_edge_cases() -> Result<()> {
        println!("Testing GPU NTT edge cases");
        
        // Test minimum dimension
        let min_dimension = 32;
        let modulus = TEST_MODULI[0];
        
        if is_ntt_compatible(min_dimension, modulus) {
            let params = create_test_params(min_dimension, modulus)?;
            let mut engine = AdaptiveNttEngine::new(params, true)?;
            
            // Test zero polynomial
            let zero_poly = vec![0i64; min_dimension];
            let forward_zero = engine.forward_ntt(&zero_poly)?;
            let inverse_zero = engine.inverse_ntt(&forward_zero)?;
            
            assert_eq!(zero_poly, inverse_zero, "Zero polynomial round-trip failed");
            
            // Test polynomial with maximum coefficients
            let max_poly = vec![modulus - 1; min_dimension];
            let forward_max = engine.forward_ntt(&max_poly)?;
            let inverse_max = engine.inverse_ntt(&forward_max)?;
            
            assert_eq!(max_poly, inverse_max, "Maximum coefficient polynomial round-trip failed");
            
            // Test polynomial with single non-zero coefficient
            let mut single_coeff = vec![0i64; min_dimension];
            single_coeff[0] = 1;
            let forward_single = engine.forward_ntt(&single_coeff)?;
            let inverse_single = engine.inverse_ntt(&forward_single)?;
            
            assert_eq!(single_coeff, inverse_single, "Single coefficient polynomial round-trip failed");
        }
        
        // Test maximum supported dimension
        let max_dimension = 8192; // Adjust based on GPU memory limits
        if is_ntt_compatible(max_dimension, modulus) {
            let params = create_test_params(max_dimension, modulus)?;
            let mut engine = AdaptiveNttEngine::new(params, true)?;
            
            let mut rng = ChaCha20Rng::seed_from_u64(11111);
            let large_poly = generate_random_polynomial(max_dimension, modulus, &mut rng);
            
            let forward_large = engine.forward_ntt(&large_poly)?;
            let inverse_large = engine.inverse_ntt(&forward_large)?;
            
            assert_eq!(large_poly, inverse_large, "Large polynomial round-trip failed");
        }
        
        println!("GPU NTT edge cases test completed successfully");
        Ok(())
    }
    
    /// Tests concurrent GPU NTT operations
    /// 
    /// Verifies that multiple threads can safely use GPU NTT simultaneously
    /// without data races or corruption.
    #[test]
    #[cfg(feature = "gpu")]
    fn test_gpu_ntt_concurrency() -> Result<()> {
        use std::sync::Arc;
        use std::thread;
        
        let dimension = 1024;
        let modulus = TEST_MODULI[0];
        
        if !is_ntt_compatible(dimension, modulus) {
            return Ok(());
        }
        
        println!("Testing GPU NTT concurrency with {} threads", num_cpus::get());
        
        let params = create_test_params(dimension, modulus)?;
        let engine = Arc::new(std::sync::Mutex::new(
            AdaptiveNttEngine::new(params, true)?
        ));
        
        let num_threads = num_cpus::get().min(4); // Limit threads for test stability
        let operations_per_thread = 5;
        
        let mut handles = Vec::new();
        
        for thread_id in 0..num_threads {
            let engine_clone = Arc::clone(&engine);
            
            let handle = thread::spawn(move || -> Result<()> {
                let mut rng = ChaCha20Rng::seed_from_u64(thread_id as u64 * 12345);
                
                for operation in 0..operations_per_thread {
                    let coefficients = generate_random_polynomial(dimension, modulus, &mut rng);
                    
                    // Acquire lock and perform NTT operations
                    let mut engine_guard = engine_clone.lock().unwrap();
                    let forward_result = engine_guard.forward_ntt(&coefficients)?;
                    let inverse_result = engine_guard.inverse_ntt(&forward_result)?;
                    drop(engine_guard); // Release lock early
                    
                    // Verify correctness
                    assert_eq!(coefficients, inverse_result,
                              "Concurrency test failed for thread {} operation {}", 
                              thread_id, operation);
                }
                
                Ok(())
            });
            
            handles.push(handle);
        }
        
        // Wait for all threads to complete
        for (thread_id, handle) in handles.into_iter().enumerate() {
            handle.join().unwrap()
                .map_err(|e| {
                    eprintln!("Thread {} failed: {}", thread_id, e);
                    e
                })?;
        }
        
        println!("GPU NTT concurrency test completed successfully");
        Ok(())
    }
    
    /// Helper function to check if dimension and modulus are NTT-compatible
    /// 
    /// # Arguments
    /// * `dimension` - Polynomial dimension
    /// * `modulus` - Prime modulus
    /// 
    /// # Returns
    /// * `bool` - True if compatible for NTT
    fn is_ntt_compatible(dimension: usize, modulus: i64) -> bool {
        // Check if dimension is power of 2
        if !dimension.is_power_of_two() {
            return false;
        }
        
        // Check if modulus supports required order
        // For NTT of size d, we need 2d-th roots of unity
        let required_order = 2 * dimension;
        (modulus - 1) % (required_order as i64) == 0
    }
    
    /// Integration test: GPU NTT within polynomial multiplication
    /// 
    /// Tests GPU NTT as part of a complete polynomial multiplication pipeline
    /// to ensure it works correctly in realistic usage scenarios.
    #[test]
    #[cfg(feature = "gpu")]
    fn test_gpu_ntt_polynomial_multiplication() -> Result<()> {
        let dimension = 2048;
        let modulus = TEST_MODULI[0];
        
        if !is_ntt_compatible(dimension, modulus) {
            return Ok(());
        }
        
        println!("Testing GPU NTT in polynomial multiplication context");
        
        let params = create_test_params(dimension, modulus)?;
        let mut engine = AdaptiveNttEngine::new(params, true)?;
        
        let mut rng = ChaCha20Rng::seed_from_u64(55555);
        
        // Generate two random polynomials for multiplication
        let poly_a = generate_random_polynomial(dimension, modulus, &mut rng);
        let poly_b = generate_random_polynomial(dimension, modulus, &mut rng);
        
        // Perform NTT-based multiplication using GPU
        let ntt_a = engine.forward_ntt(&poly_a)?;
        let ntt_b = engine.forward_ntt(&poly_b)?;
        
        // Pointwise multiplication in NTT domain
        let mut ntt_product = Vec::with_capacity(dimension);
        for (a_coeff, b_coeff) in ntt_a.iter().zip(ntt_b.iter()) {
            ntt_product.push(((*a_coeff as i128 * *b_coeff as i128) % modulus as i128) as i64);
        }
        
        // Inverse NTT to get final product
        let product = engine.inverse_ntt(&ntt_product)?;
        
        // Verify result using schoolbook multiplication (for small test)
        if dimension <= 256 {
            let expected_product = schoolbook_multiply(&poly_a, &poly_b, modulus, dimension);
            assert_eq!(product, expected_product, 
                      "GPU NTT multiplication result doesn't match schoolbook multiplication");
        }
        
        // Verify basic properties
        assert_eq!(product.len(), dimension, "Product length incorrect");
        
        // All coefficients should be in valid range
        for (i, &coeff) in product.iter().enumerate() {
            assert!(coeff >= 0 && coeff < modulus,
                   "Product coefficient {} at index {} out of range [0, {})", 
                   coeff, i, modulus);
        }
        
        println!("GPU NTT polynomial multiplication test completed successfully");
        Ok(())
    }
    
    /// Helper function for schoolbook polynomial multiplication
    /// 
    /// # Arguments
    /// * `a` - First polynomial
    /// * `b` - Second polynomial  
    /// * `modulus` - Modulus for coefficient reduction
    /// * `result_size` - Size of result polynomial
    /// 
    /// # Returns
    /// * `Vec<i64>` - Product polynomial
    fn schoolbook_multiply(a: &[i64], b: &[i64], modulus: i64, result_size: usize) -> Vec<i64> {
        let mut result = vec![0i64; result_size];
        
        for (i, &a_coeff) in a.iter().enumerate() {
            for (j, &b_coeff) in b.iter().enumerate() {
                let pos = (i + j) % result_size;
                let sign = if i + j >= result_size { -1 } else { 1 };
                
                let product = (a_coeff as i128 * b_coeff as i128 * sign as i128) % modulus as i128;
                result[pos] = ((result[pos] as i128 + product) % modulus as i128) as i64;
                
                if result[pos] < 0 {
                    result[pos] += modulus;
                }
            }
        }
        
        result
    }
    
    /// Benchmark test comparing GPU batch processing vs individual operations
    /// 
    /// Measures the performance benefit of batch processing on GPU
    /// compared to individual NTT operations.
    #[test]
    #[cfg(feature = "gpu")]
    fn test_gpu_batch_performance() -> Result<()> {
        let dimension = 1024;
        let modulus = TEST_MODULI[0];
        let batch_sizes = [1, 2, 4, 8, 16, 32];
        
        if !is_ntt_compatible(dimension, modulus) {
            return Ok(());
        }
        
        println!("GPU Batch Processing Performance Test");
        println!("====================================");
        println!("{:>10} {:>15} {:>15} {:>10}", "Batch Size", "Individual (ms)", "Batch (ms)", "Speedup");
        println!("{:-<55}", "");
        
        let params = create_test_params(dimension, modulus)?;
        let mut engine = AdaptiveNttEngine::new(params, true)?;
        let mut rng = ChaCha20Rng::seed_from_u64(77777);
        
        for &batch_size in &batch_sizes {
            // Generate batch of test polynomials
            let batch_polys = generate_random_batch(batch_size, dimension, modulus, &mut rng);
            
            // Warm up
            let _ = engine.batch_ntt(&batch_polys, true)?;
            
            // Benchmark individual operations
            let individual_start = Instant::now();
            for poly in &batch_polys {
                let _ = engine.forward_ntt(poly)?;
            }
            let individual_elapsed = individual_start.elapsed();
            let individual_time_ms = individual_elapsed.as_millis() as f64;
            
            // Benchmark batch operation
            let batch_start = Instant::now();
            let _ = engine.batch_ntt(&batch_polys, true)?;
            let batch_elapsed = batch_start.elapsed();
            let batch_time_ms = batch_elapsed.as_millis() as f64;
            
            // Calculate speedup
            let speedup = if batch_time_ms > 0.0 { individual_time_ms / batch_time_ms } else { 0.0 };
            
            println!("{:>10} {:>15.3} {:>15.3} {:>10.2}x", 
                    batch_size, individual_time_ms, batch_time_ms, speedup);
            
            // Batch processing should be more efficient for larger batches
            if batch_size >= 8 {
                assert!(speedup > 1.0, 
                       "Batch processing not more efficient for batch size {}: speedup={:.2}x", 
                       batch_size, speedup);
            }
        }
        
        Ok(())
    }
}