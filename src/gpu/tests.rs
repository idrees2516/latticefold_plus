/// Comprehensive tests for GPU acceleration and performance optimization
/// 
/// This module provides extensive testing for all GPU and SIMD implementations
/// to ensure correctness, performance, and compatibility across different
/// hardware configurations.

#[cfg(test)]
mod tests {
    use crate::gpu::*;
    use crate::simd::*;
    use crate::memory::*;
    use crate::error::Result;
    
    #[test]
    fn test_gpu_device_detection() {
        // Test GPU device detection and initialization
        let result = initialize_gpu();
        assert!(result.is_ok(), "GPU initialization should not fail");
        
        // Print device information for debugging
        print_device_info();
        
        // Check if any GPU devices are available
        let gpu_available = is_gpu_available();
        println!("GPU acceleration available: {}", gpu_available);
        
        // Get current device info
        if let Some(device_info) = get_current_device_info() {
            println!("Current device: {} ({})", device_info.name, 
                match device_info.device_type {
                    GpuDeviceType::Cuda => "CUDA",
                    GpuDeviceType::OpenCL => "OpenCL", 
                    GpuDeviceType::Cpu => "CPU",
                });
        }
    }
    
    #[test]
    fn test_simd_capability_detection() {
        // Test SIMD capability detection
        let dispatcher = get_simd_dispatcher();
        
        println!("SIMD capability: {:?}", dispatcher.capability());
        println!("Vector width: {} elements", dispatcher.vector_width());
        println!("Memory alignment: {} bytes", dispatcher.alignment());
        
        // Print detailed SIMD information
        print_simd_info();
        
        // Verify dispatcher is functional
        assert!(dispatcher.vector_width() >= 1);
        assert!(dispatcher.alignment() >= 8);
    }
    
    #[test]
    fn test_memory_allocator() -> Result<()> {
        // Test aligned memory allocation
        initialize_allocator(true)?;
        let allocator = get_allocator()?;
        
        // Test basic allocation and deallocation
        let ptr = allocator.allocate(1024, 64)?;
        
        // Verify alignment
        assert_eq!((ptr.as_ptr() as usize) % 64, 0, "Memory not properly aligned");
        
        // Test writing to allocated memory
        unsafe {
            for i in 0..128 {
                *ptr.as_ptr().cast::<i64>().add(i) = i as i64;
            }
            
            // Verify data integrity
            for i in 0..128 {
                assert_eq!(*ptr.as_ptr().cast::<i64>().add(i), i as i64);
            }
            
            // Deallocate
            allocator.deallocate(ptr, 1024, 64);
        }
        
        // Check for memory leaks
        let leaks = allocator.check_leaks();
        assert!(leaks.is_empty(), "Memory leaks detected: {:?}", leaks);
        
        // Print memory statistics
        let stats = allocator.stats();
        stats.print_stats();
        
        Ok(())
    }
    
    #[test]
    fn test_simd_operations() -> Result<()> {
        let dispatcher = get_simd_dispatcher();
        
        // Test data
        let a = vec![1i64, 2, 3, 4, 5, 6, 7, 8];
        let b = vec![8i64, 7, 6, 5, 4, 3, 2, 1];
        let mut result = vec![0i64; 8];
        let modulus = 1000000007i64;
        
        // Test vectorized addition
        dispatcher.add_mod(&a, &b, &mut result, modulus)?;
        
        // Verify results (all should be 9)
        for (i, &val) in result.iter().enumerate() {
            assert_eq!(val, 9, "Addition failed at index {}: expected 9, got {}", i, val);
        }
        
        // Test vectorized subtraction
        dispatcher.sub_mod(&a, &b, &mut result, modulus)?;
        
        // Verify results
        let expected_sub = vec![-7i64, -5, -3, -1, 1, 3, 5, 7];
        for (i, (&actual, &expected)) in result.iter().zip(expected_sub.iter()).enumerate() {
            assert_eq!(actual, expected, "Subtraction failed at index {}: expected {}, got {}", i, expected, actual);
        }
        
        // Test vectorized multiplication
        dispatcher.mul_mod(&a, &b, &mut result, modulus)?;
        
        // Verify results
        let expected_mul = vec![8i64, 14, 18, 20, 20, 18, 14, 8];
        for (i, (&actual, &expected)) in result.iter().zip(expected_mul.iter()).enumerate() {
            assert_eq!(actual, expected, "Multiplication failed at index {}: expected {}, got {}", i, expected, actual);
        }
        
        // Test scalar multiplication
        let scalar = 3i64;
        dispatcher.scale_mod(&a, scalar, &mut result, modulus)?;
        
        // Verify results
        let expected_scale = vec![3i64, 6, 9, 12, 15, 18, 21, 24];
        for (i, (&actual, &expected)) in result.iter().zip(expected_scale.iter()).enumerate() {
            assert_eq!(actual, expected, "Scaling failed at index {}: expected {}, got {}", i, expected, actual);
        }
        
        Ok(())
    }
    
    #[test]
    fn test_norm_computations() -> Result<()> {
        let dispatcher = get_simd_dispatcher();
        
        // Test vector: [3, -4, 0, 5, -2]
        let vector = vec![3i64, -4, 0, 5, -2];
        
        // Test infinity norm
        let inf_norm = dispatcher.infinity_norm(&vector)?;
        assert_eq!(inf_norm, 5, "Infinity norm should be 5, got {}", inf_norm);
        
        // Test squared Euclidean norm
        let euclidean_squared = dispatcher.euclidean_norm_squared(&vector)?;
        let expected_squared = 3*3 + 4*4 + 0*0 + 5*5 + 2*2; // 9 + 16 + 0 + 25 + 4 = 54
        assert_eq!(euclidean_squared, expected_squared, 
                  "Squared Euclidean norm should be {}, got {}", expected_squared, euclidean_squared);
        
        Ok(())
    }
    
    #[test]
    fn test_dot_product() -> Result<()> {
        let dispatcher = get_simd_dispatcher();
        
        let a = vec![1i64, 2, 3, 4];
        let b = vec![5i64, 6, 7, 8];
        
        let dot_product = dispatcher.dot_product(&a, &b)?;
        let expected = 1*5 + 2*6 + 3*7 + 4*8; // 5 + 12 + 21 + 32 = 70
        
        assert_eq!(dot_product, expected, "Dot product should be {}, got {}", expected, dot_product);
        
        Ok(())
    }
    
    #[test]
    fn test_linear_combination() -> Result<()> {
        let dispatcher = get_simd_dispatcher();
        
        let a = vec![1i64, 2, 3, 4];
        let b = vec![5i64, 6, 7, 8];
        let mut result = vec![0i64; 4];
        let alpha = 2i64;
        let beta = 3i64;
        let modulus = 1000000007i64;
        
        dispatcher.linear_combination(&a, &b, alpha, beta, &mut result, modulus)?;
        
        // Expected: result[i] = 2*a[i] + 3*b[i]
        let expected = vec![17i64, 22, 27, 32]; // [2*1+3*5, 2*2+3*6, 2*3+3*7, 2*4+3*8]
        
        for (i, (&actual, &expected)) in result.iter().zip(expected.iter()).enumerate() {
            assert_eq!(actual, expected, "Linear combination failed at index {}: expected {}, got {}", i, expected, actual);
        }
        
        Ok(())
    }
    
    #[test]
    fn test_performance_comparison() -> Result<()> {
        use std::time::Instant;
        
        let dispatcher = get_simd_dispatcher();
        
        // Large test arrays for performance comparison
        let size = 10000;
        let a = vec![1i64; size];
        let b = vec![2i64; size];
        let mut result_simd = vec![0i64; size];
        let mut result_scalar = vec![0i64; size];
        let modulus = 1000000007i64;
        
        // Time SIMD implementation
        let start_simd = Instant::now();
        for _ in 0..100 {
            dispatcher.add_mod(&a, &b, &mut result_simd, modulus)?;
        }
        let simd_time = start_simd.elapsed();
        
        // Time scalar implementation (using the scalar fallback directly)
        let start_scalar = Instant::now();
        for _ in 0..100 {
            unsafe {
                crate::simd::scalar_fallback::add_mod_scalar(
                    a.as_ptr(), 
                    b.as_ptr(), 
                    result_scalar.as_mut_ptr(), 
                    modulus, 
                    size
                );
            }
        }
        let scalar_time = start_scalar.elapsed();
        
        // Verify results are identical
        assert_eq!(result_simd, result_scalar, "SIMD and scalar results should be identical");
        
        // Print performance comparison
        println!("Performance Comparison (100 iterations, {} elements):", size);
        println!("SIMD time: {:?}", simd_time);
        println!("Scalar time: {:?}", scalar_time);
        
        if scalar_time > simd_time {
            let speedup = scalar_time.as_nanos() as f64 / simd_time.as_nanos() as f64;
            println!("SIMD speedup: {:.2}x", speedup);
        } else {
            println!("No speedup achieved (possibly due to small array size or overhead)");
        }
        
        Ok(())
    }
    
    #[test]
    fn test_memory_system_info() {
        // Test system memory information
        let (total, available) = get_system_memory_info();
        
        println!("System memory: {:.2} GB total, {:.2} GB available", 
                total as f64 / (1024.0 * 1024.0 * 1024.0),
                available as f64 / (1024.0 * 1024.0 * 1024.0));
        
        // Basic sanity checks
        assert!(total > 0, "Total memory should be positive");
        assert!(available > 0, "Available memory should be positive");
        assert!(available <= total, "Available memory should not exceed total");
        
        // Should have at least 1GB total memory
        assert!(total >= 1024 * 1024 * 1024, "Should have at least 1GB total memory");
        
        // Print detailed memory information
        print_memory_info();
    }
    
    #[test]
    fn test_array_allocation_helpers() -> Result<()> {
        // Test typed array allocation helpers
        let ptr = allocate_array::<i64>(100)?;
        
        unsafe {
            // Write test data
            for i in 0..100 {
                *ptr.as_ptr().add(i) = i as i64;
            }
            
            // Verify data
            for i in 0..100 {
                assert_eq!(*ptr.as_ptr().add(i), i as i64);
            }
            
            // Deallocate
            deallocate_array(ptr, 100);
        }
        
        Ok(())
    }
    
    #[test]
    fn test_edge_cases() -> Result<()> {
        let dispatcher = get_simd_dispatcher();
        
        // Test empty arrays
        let empty: Vec<i64> = vec![];
        let inf_norm = dispatcher.infinity_norm(&empty)?;
        assert_eq!(inf_norm, 0, "Infinity norm of empty array should be 0");
        
        let euclidean_squared = dispatcher.euclidean_norm_squared(&empty)?;
        assert_eq!(euclidean_squared, 0, "Squared Euclidean norm of empty array should be 0");
        
        let dot_product = dispatcher.dot_product(&empty, &empty)?;
        assert_eq!(dot_product, 0, "Dot product of empty arrays should be 0");
        
        // Test single element arrays
        let single_a = vec![42i64];
        let single_b = vec![17i64];
        let mut single_result = vec![0i64];
        let modulus = 1000000007i64;
        
        dispatcher.add_mod(&single_a, &single_b, &mut single_result, modulus)?;
        assert_eq!(single_result[0], 59, "Single element addition failed");
        
        let single_dot = dispatcher.dot_product(&single_a, &single_b)?;
        assert_eq!(single_dot, 42 * 17, "Single element dot product failed");
        
        Ok(())
    }
    
    #[test]
    fn test_large_arrays() -> Result<()> {
        let dispatcher = get_simd_dispatcher();
        
        // Test with large arrays to verify scalability
        let size = 100000;
        let a = vec![1i64; size];
        let b = vec![2i64; size];
        let mut result = vec![0i64; size];
        let modulus = 1000000007i64;
        
        // This should not crash or cause memory issues
        dispatcher.add_mod(&a, &b, &mut result, modulus)?;
        
        // Verify all results are correct
        for &val in &result {
            assert_eq!(val, 3, "Large array addition failed");
        }
        
        // Test norm computations on large arrays
        let large_vector = vec![1i64; size];
        let inf_norm = dispatcher.infinity_norm(&large_vector)?;
        assert_eq!(inf_norm, 1, "Large array infinity norm failed");
        
        let euclidean_squared = dispatcher.euclidean_norm_squared(&large_vector)?;
        assert_eq!(euclidean_squared, size as i64, "Large array Euclidean norm failed");
        
        Ok(())
    }
    
    #[test]
    fn test_modular_arithmetic_correctness() -> Result<()> {
        let dispatcher = get_simd_dispatcher();
        
        // Test with various moduli to ensure correctness
        let test_moduli = vec![17i64, 97, 1009, 1000000007];
        
        for &modulus in &test_moduli {
            let a = vec![modulus - 1, modulus / 2, 1, 0];
            let b = vec![1, modulus / 2, modulus - 1, modulus / 2];
            let mut result = vec![0i64; 4];
            
            // Test addition
            dispatcher.add_mod(&a, &b, &mut result, modulus)?;
            
            // Verify results are in balanced representation
            let half_modulus = modulus / 2;
            for &val in &result {
                assert!(val >= -half_modulus && val <= half_modulus, 
                       "Result {} not in balanced representation for modulus {}", val, modulus);
            }
            
            // Test subtraction
            dispatcher.sub_mod(&a, &b, &mut result, modulus)?;
            
            // Verify results are in balanced representation
            for &val in &result {
                assert!(val >= -half_modulus && val <= half_modulus, 
                       "Result {} not in balanced representation for modulus {}", val, modulus);
            }
        }
        
        Ok(())
    }
}

/// Integration tests that require multiple components
#[cfg(test)]
mod integration_tests {
    use super::*;
    
    #[test]
    fn test_gpu_simd_memory_integration() -> Result<()> {
        // Test that GPU, SIMD, and memory systems work together
        
        // Initialize all systems
        initialize_gpu()?;
        initialize_allocator(false)?;
        
        let gpu_manager = get_gpu_manager()?;
        let simd_dispatcher = get_simd_dispatcher();
        let allocator = get_allocator()?;
        
        // Allocate aligned memory
        let size = 1024;
        let alignment = simd_dispatcher.alignment();
        let ptr = allocator.allocate(size * 8, alignment)?;
        
        // Use SIMD operations on aligned memory
        let test_data = vec![1i64; size];
        unsafe {
            std::ptr::copy_nonoverlapping(
                test_data.as_ptr(),
                ptr.as_ptr() as *mut i64,
                size
            );
            
            // Perform SIMD operation
            let vector_slice = std::slice::from_raw_parts(ptr.as_ptr() as *const i64, size);
            let norm = simd_dispatcher.infinity_norm(vector_slice)?;
            assert_eq!(norm, 1);
            
            // Clean up
            allocator.deallocate(ptr, size * 8, alignment);
        }
        
        println!("GPU-SIMD-Memory integration test passed");
        
        Ok(())
    }
    
    #[test]
    fn test_cross_platform_compatibility() {
        // Test that the implementation works across different platforms
        
        println!("Testing cross-platform compatibility...");
        
        // Test SIMD detection
        let dispatcher = get_simd_dispatcher();
        println!("SIMD capability: {:?}", dispatcher.capability());
        
        // Test GPU detection
        let _ = initialize_gpu();
        let gpu_available = is_gpu_available();
        println!("GPU available: {}", gpu_available);
        
        // Test memory allocation
        let result = initialize_allocator(false);
        assert!(result.is_ok(), "Memory allocator should initialize on all platforms");
        
        println!("Cross-platform compatibility test passed");
    }
}