/// Performance benchmarks for GPU acceleration and SIMD optimization
/// 
/// This benchmark suite measures the performance improvements achieved
/// through GPU acceleration, SIMD vectorization, and memory optimization
/// compared to baseline scalar implementations.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use latticefold_plus::simd::*;
use latticefold_plus::memory::*;
use latticefold_plus::gpu::*;
use std::time::Duration;

/// Benchmark SIMD vs scalar operations
fn bench_simd_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_operations");
    
    // Test different array sizes
    let sizes = vec![64, 256, 1024, 4096, 16384];
    
    for size in sizes {
        group.throughput(Throughput::Elements(size as u64));
        
        // Prepare test data
        let a = vec![1i64; size];
        let b = vec![2i64; size];
        let mut result = vec![0i64; size];
        let modulus = 1000000007i64;
        
        let dispatcher = get_simd_dispatcher();
        
        // Benchmark SIMD addition
        group.bench_with_input(
            BenchmarkId::new("simd_add", size),
            &size,
            |bench, _| {
                bench.iter(|| {
                    dispatcher.add_mod(
                        black_box(&a),
                        black_box(&b),
                        black_box(&mut result),
                        black_box(modulus),
                    ).unwrap();
                });
            },
        );
        
        // Benchmark scalar addition for comparison
        group.bench_with_input(
            BenchmarkId::new("scalar_add", size),
            &size,
            |bench, _| {
                bench.iter(|| {
                    unsafe {
                        crate::simd::scalar_fallback::add_mod_scalar(
                            black_box(a.as_ptr()),
                            black_box(b.as_ptr()),
                            black_box(result.as_mut_ptr()),
                            black_box(modulus),
                            black_box(size),
                        );
                    }
                });
            },
        );
        
        // Benchmark SIMD multiplication
        group.bench_with_input(
            BenchmarkId::new("simd_mul", size),
            &size,
            |bench, _| {
                bench.iter(|| {
                    dispatcher.mul_mod(
                        black_box(&a),
                        black_box(&b),
                        black_box(&mut result),
                        black_box(modulus),
                    ).unwrap();
                });
            },
        );
        
        // Benchmark scalar multiplication for comparison
        group.bench_with_input(
            BenchmarkId::new("scalar_mul", size),
            &size,
            |bench, _| {
                bench.iter(|| {
                    unsafe {
                        crate::simd::scalar_fallback::mul_mod_scalar(
                            black_box(a.as_ptr()),
                            black_box(b.as_ptr()),
                            black_box(result.as_mut_ptr()),
                            black_box(modulus),
                            black_box(size),
                        );
                    }
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark norm computations
fn bench_norm_computations(c: &mut Criterion) {
    let mut group = c.benchmark_group("norm_computations");
    
    let sizes = vec![256, 1024, 4096, 16384];
    
    for size in sizes {
        group.throughput(Throughput::Elements(size as u64));
        
        // Create test vector with varying values
        let vector: Vec<i64> = (0..size).map(|i| (i as i64) % 1000 - 500).collect();
        
        let dispatcher = get_simd_dispatcher();
        
        // Benchmark SIMD infinity norm
        group.bench_with_input(
            BenchmarkId::new("simd_inf_norm", size),
            &size,
            |bench, _| {
                bench.iter(|| {
                    black_box(dispatcher.infinity_norm(black_box(&vector)).unwrap());
                });
            },
        );
        
        // Benchmark scalar infinity norm
        group.bench_with_input(
            BenchmarkId::new("scalar_inf_norm", size),
            &size,
            |bench, _| {
                bench.iter(|| {
                    unsafe {
                        black_box(crate::simd::scalar_fallback::infinity_norm_scalar(
                            black_box(vector.as_ptr()),
                            black_box(size),
                        ));
                    }
                });
            },
        );
        
        // Benchmark SIMD Euclidean norm squared
        group.bench_with_input(
            BenchmarkId::new("simd_euclidean_norm_sq", size),
            &size,
            |bench, _| {
                bench.iter(|| {
                    black_box(dispatcher.euclidean_norm_squared(black_box(&vector)).unwrap());
                });
            },
        );
        
        // Benchmark scalar Euclidean norm squared
        group.bench_with_input(
            BenchmarkId::new("scalar_euclidean_norm_sq", size),
            &size,
            |bench, _| {
                bench.iter(|| {
                    unsafe {
                        black_box(crate::simd::scalar_fallback::euclidean_norm_squared_scalar(
                            black_box(vector.as_ptr()),
                            black_box(size),
                        ));
                    }
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark dot product computations
fn bench_dot_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("dot_product");
    
    let sizes = vec![256, 1024, 4096, 16384];
    
    for size in sizes {
        group.throughput(Throughput::Elements(size as u64));
        
        let a: Vec<i64> = (0..size).map(|i| i as i64).collect();
        let b: Vec<i64> = (0..size).map(|i| (size - i) as i64).collect();
        
        let dispatcher = get_simd_dispatcher();
        
        // Benchmark SIMD dot product
        group.bench_with_input(
            BenchmarkId::new("simd_dot_product", size),
            &size,
            |bench, _| {
                bench.iter(|| {
                    black_box(dispatcher.dot_product(black_box(&a), black_box(&b)).unwrap());
                });
            },
        );
        
        // Benchmark scalar dot product
        group.bench_with_input(
            BenchmarkId::new("scalar_dot_product", size),
            &size,
            |bench, _| {
                bench.iter(|| {
                    unsafe {
                        black_box(crate::simd::scalar_fallback::dot_product_scalar(
                            black_box(a.as_ptr()),
                            black_box(b.as_ptr()),
                            black_box(size),
                        ));
                    }
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark memory allocation strategies
fn bench_memory_allocation(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_allocation");
    
    // Initialize allocator
    initialize_allocator(false).unwrap();
    let allocator = get_allocator().unwrap();
    
    let sizes = vec![1024, 4096, 16384, 65536];
    
    for size in sizes {
        group.throughput(Throughput::Bytes(size as u64));
        
        // Benchmark aligned allocation
        group.bench_with_input(
            BenchmarkId::new("aligned_alloc", size),
            &size,
            |bench, &size| {
                bench.iter(|| {
                    let ptr = allocator.allocate(black_box(size), black_box(64)).unwrap();
                    unsafe {
                        allocator.deallocate(ptr, size, 64);
                    }
                });
            },
        );
        
        // Benchmark standard allocation for comparison
        group.bench_with_input(
            BenchmarkId::new("standard_alloc", size),
            &size,
            |bench, &size| {
                bench.iter(|| {
                    let vec = vec![0u8; black_box(size)];
                    black_box(vec);
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark linear combinations (common in lattice operations)
fn bench_linear_combination(c: &mut Criterion) {
    let mut group = c.benchmark_group("linear_combination");
    
    let sizes = vec![256, 1024, 4096];
    
    for size in sizes {
        group.throughput(Throughput::Elements(size as u64));
        
        let a = vec![1i64; size];
        let b = vec![2i64; size];
        let mut result = vec![0i64; size];
        let alpha = 3i64;
        let beta = 5i64;
        let modulus = 1000000007i64;
        
        let dispatcher = get_simd_dispatcher();
        
        // Benchmark SIMD linear combination
        group.bench_with_input(
            BenchmarkId::new("simd_linear_comb", size),
            &size,
            |bench, _| {
                bench.iter(|| {
                    dispatcher.linear_combination(
                        black_box(&a),
                        black_box(&b),
                        black_box(alpha),
                        black_box(beta),
                        black_box(&mut result),
                        black_box(modulus),
                    ).unwrap();
                });
            },
        );
        
        // Benchmark scalar linear combination
        group.bench_with_input(
            BenchmarkId::new("scalar_linear_comb", size),
            &size,
            |bench, _| {
                bench.iter(|| {
                    unsafe {
                        crate::simd::scalar_fallback::linear_combination_scalar(
                            black_box(a.as_ptr()),
                            black_box(b.as_ptr()),
                            black_box(alpha),
                            black_box(beta),
                            black_box(result.as_mut_ptr()),
                            black_box(modulus),
                            black_box(size),
                        );
                    }
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark GPU operations (if available)
fn bench_gpu_operations(c: &mut Criterion) {
    // Only run GPU benchmarks if GPU is available
    if !is_gpu_available() {
        println!("Skipping GPU benchmarks - no GPU available");
        return;
    }
    
    let mut group = c.benchmark_group("gpu_operations");
    group.measurement_time(Duration::from_secs(10)); // Longer measurement time for GPU
    
    // Initialize GPU
    initialize_gpu().unwrap();
    
    let sizes = vec![1024, 4096, 16384];
    
    for size in sizes {
        group.throughput(Throughput::Elements(size as u64));
        
        let a = vec![1i64; size];
        let b = vec![2i64; size];
        
        // Note: Actual GPU benchmarks would require implementing the GPU operations
        // For now, we'll benchmark the GPU initialization and memory management
        
        group.bench_with_input(
            BenchmarkId::new("gpu_memory_alloc", size),
            &size,
            |bench, &size| {
                bench.iter(|| {
                    if let Ok(manager) = get_gpu_manager() {
                        if let Some(memory_manager) = manager.selected_memory_manager() {
                            // Simulate GPU memory allocation
                            let byte_size = size * std::mem::size_of::<i64>();
                            // Note: This would actually allocate GPU memory in a real implementation
                            black_box(byte_size);
                        }
                    }
                });
            },
        );
    }
    
    group.finish();
}

/// Comprehensive benchmark comparing all optimization levels
fn bench_optimization_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimization_comparison");
    
    let size = 4096;
    group.throughput(Throughput::Elements(size as u64));
    
    let a = vec![1i64; size];
    let b = vec![2i64; size];
    let mut result = vec![0i64; size];
    let modulus = 1000000007i64;
    
    let dispatcher = get_simd_dispatcher();
    
    // Benchmark different optimization levels for the same operation (addition)
    
    // Level 0: Pure scalar
    group.bench_function("level0_pure_scalar", |bench| {
        bench.iter(|| {
            for i in 0..size {
                let sum = black_box(a[i]) + black_box(b[i]);
                let reduced = ((sum % modulus) + modulus) % modulus;
                let half_modulus = modulus / 2;
                result[i] = if reduced > half_modulus {
                    reduced - modulus
                } else {
                    reduced
                };
            }
            black_box(&result);
        });
    });
    
    // Level 1: Optimized scalar with loop unrolling
    group.bench_function("level1_optimized_scalar", |bench| {
        bench.iter(|| {
            unsafe {
                crate::simd::scalar_fallback::add_mod_scalar(
                    black_box(a.as_ptr()),
                    black_box(b.as_ptr()),
                    black_box(result.as_mut_ptr()),
                    black_box(modulus),
                    black_box(size),
                );
            }
        });
    });
    
    // Level 2: SIMD vectorized
    group.bench_function("level2_simd_vectorized", |bench| {
        bench.iter(|| {
            dispatcher.add_mod(
                black_box(&a),
                black_box(&b),
                black_box(&mut result),
                black_box(modulus),
            ).unwrap();
        });
    });
    
    // Level 3: Parallel SIMD (using Rayon)
    group.bench_function("level3_parallel_simd", |bench| {
        bench.iter(|| {
            crate::simd::scalar_fallback::add_mod_scalar_parallel(
                black_box(&a),
                black_box(&b),
                black_box(&mut result),
                black_box(modulus),
            ).unwrap();
        });
    });
    
    group.finish();
}

/// Benchmark memory access patterns
fn bench_memory_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_patterns");
    
    let size = 4096;
    group.throughput(Throughput::Elements(size as u64));
    
    // Test different memory access patterns
    let data = vec![1i64; size];
    
    // Sequential access
    group.bench_function("sequential_access", |bench| {
        bench.iter(|| {
            let mut sum = 0i64;
            for i in 0..size {
                sum = sum.wrapping_add(black_box(data[i]));
            }
            black_box(sum);
        });
    });
    
    // Strided access (every 8th element)
    group.bench_function("strided_access", |bench| {
        bench.iter(|| {
            let mut sum = 0i64;
            for i in (0..size).step_by(8) {
                sum = sum.wrapping_add(black_box(data[i]));
            }
            black_box(sum);
        });
    });
    
    // Random access (pseudo-random pattern)
    group.bench_function("random_access", |bench| {
        let indices: Vec<usize> = (0..size).map(|i| (i * 7919) % size).collect();
        bench.iter(|| {
            let mut sum = 0i64;
            for &i in &indices {
                sum = sum.wrapping_add(black_box(data[i]));
            }
            black_box(sum);
        });
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_simd_operations,
    bench_norm_computations,
    bench_dot_product,
    bench_memory_allocation,
    bench_linear_combination,
    bench_gpu_operations,
    bench_optimization_comparison,
    bench_memory_patterns
);

criterion_main!(benches);