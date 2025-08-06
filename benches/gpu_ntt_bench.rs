/// Comprehensive benchmarks for GPU NTT implementation
/// 
/// This benchmark suite measures the performance of GPU-accelerated NTT operations
/// compared to CPU implementations across various dimensions and use cases.
/// 
/// Benchmark Categories:
/// 1. Single NTT Operations - Forward and inverse NTT performance
/// 2. Batch Processing - Performance benefits of batch operations
/// 3. Memory Transfer Overhead - Impact of CPU-GPU data transfers
/// 4. Algorithm Selection - Adaptive algorithm selection performance
/// 5. Scalability Analysis - Performance scaling with problem size
/// 6. Real-world Workloads - Performance in realistic cryptographic scenarios
/// 
/// Performance Metrics:
/// - Throughput (operations per second)
/// - Latency (time per operation)
/// - Memory bandwidth utilization
/// - GPU occupancy and efficiency
/// - Power consumption (when available)
/// - Scalability characteristics

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use latticefold_plus::ntt::{NTTParams, AdaptiveNttEngine};
use latticefold_plus::error::Result;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use std::time::Duration;

/// Benchmark configuration parameters
const BENCHMARK_DIMENSIONS: &[usize] = &[256, 512, 1024, 2048, 4096, 8192];
const BENCHMARK_BATCH_SIZES: &[usize] = &[1, 2, 4, 8, 16, 32, 64];
const BENCHMARK_MODULUS: i64 = 2013265921; // 15 * 2^27 + 1, NTT-friendly
const WARMUP_ITERATIONS: usize = 3;
const MEASUREMENT_TIME: Duration = Duration::from_secs(10);

/// Creates NTT parameters for benchmarking
/// 
/// # Arguments
/// * `dimension` - Polynomial dimension
/// 
/// # Returns
/// * `Result<NTTParams>` - NTT parameters for benchmarking
fn create_benchmark_params(dimension: usize) -> Result<NTTParams> {
    NTTParams::new(dimension, BENCHMARK_MODULUS)
}

/// Generates random polynomial for benchmarking
/// 
/// # Arguments
/// * `dimension` - Number of coefficients
/// * `rng` - Random number generator
/// 
/// # Returns
/// * `Vec<i64>` - Random polynomial coefficients
fn generate_benchmark_polynomial(dimension: usize, rng: &mut impl Rng) -> Vec<i64> {
    (0..dimension)
        .map(|_| rng.gen_range(0..BENCHMARK_MODULUS))
        .collect()
}

/// Generates batch of random polynomials for benchmarking
/// 
/// # Arguments
/// * `batch_size` - Number of polynomials
/// * `dimension` - Coefficients per polynomial
/// * `rng` - Random number generator
/// 
/// # Returns
/// * `Vec<Vec<i64>>` - Batch of random polynomials
fn generate_benchmark_batch(
    batch_size: usize, 
    dimension: usize, 
    rng: &mut impl Rng
) -> Vec<Vec<i64>> {
    (0..batch_size)
        .map(|_| generate_benchmark_polynomial(dimension, rng))
        .collect()
}

/// Benchmarks single forward NTT operations
/// 
/// Measures the performance of forward NTT across different polynomial dimensions
/// for both CPU and GPU implementations.
fn bench_single_forward_ntt(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_forward_ntt");
    group.measurement_time(MEASUREMENT_TIME);
    
    let mut rng = ChaCha20Rng::seed_from_u64(12345);
    
    for &dimension in BENCHMARK_DIMENSIONS {
        // Skip if dimension not supported
        if let Err(_) = create_benchmark_params(dimension) {
            continue;
        }
        
        let params = create_benchmark_params(dimension).unwrap();
        let coefficients = generate_benchmark_polynomial(dimension, &mut rng);
        
        // Set throughput for meaningful comparison
        group.throughput(Throughput::Elements(dimension as u64));
        
        // Benchmark CPU implementation
        group.bench_with_input(
            BenchmarkId::new("cpu", dimension),
            &dimension,
            |b, &_dim| {
                let mut cpu_engine = latticefold_plus::ntt::NttEngine::new(params.clone()).unwrap();
                b.iter(|| {
                    black_box(cpu_engine.forward_ntt(black_box(&coefficients)).unwrap())
                });
            },
        );
        
        // Benchmark GPU implementation (if available)
        #[cfg(feature = "gpu")]
        {
            if let Ok(mut gpu_engine) = AdaptiveNttEngine::new(params.clone(), true) {
                // Warm up GPU
                for _ in 0..WARMUP_ITERATIONS {
                    let _ = gpu_engine.forward_ntt(&coefficients);
                }
                
                group.bench_with_input(
                    BenchmarkId::new("gpu", dimension),
                    &dimension,
                    |b, &_dim| {
                        b.iter(|| {
                            black_box(gpu_engine.forward_ntt(black_box(&coefficients)).unwrap())
                        });
                    },
                );
            }
        }
        
        // Benchmark adaptive engine (automatic selection)
        if let Ok(mut adaptive_engine) = AdaptiveNttEngine::new(params.clone(), true) {
            // Warm up
            for _ in 0..WARMUP_ITERATIONS {
                let _ = adaptive_engine.forward_ntt(&coefficients);
            }
            
            group.bench_with_input(
                BenchmarkId::new("adaptive", dimension),
                &dimension,
                |b, &_dim| {
                    b.iter(|| {
                        black_box(adaptive_engine.forward_ntt(black_box(&coefficients)).unwrap())
                    });
                },
            );
        }
    }
    
    group.finish();
}

/// Benchmarks single inverse NTT operations
/// 
/// Measures the performance of inverse NTT across different polynomial dimensions.
fn bench_single_inverse_ntt(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_inverse_ntt");
    group.measurement_time(MEASUREMENT_TIME);
    
    let mut rng = ChaCha20Rng::seed_from_u64(23456);
    
    for &dimension in BENCHMARK_DIMENSIONS {
        if let Err(_) = create_benchmark_params(dimension) {
            continue;
        }
        
        let params = create_benchmark_params(dimension).unwrap();
        let coefficients = generate_benchmark_polynomial(dimension, &mut rng);
        
        // Pre-compute NTT coefficients for inverse operation
        let mut setup_engine = AdaptiveNttEngine::new(params.clone(), false).unwrap();
        let ntt_coefficients = setup_engine.forward_ntt(&coefficients).unwrap();
        
        group.throughput(Throughput::Elements(dimension as u64));
        
        // Benchmark CPU inverse NTT
        group.bench_with_input(
            BenchmarkId::new("cpu", dimension),
            &dimension,
            |b, &_dim| {
                let mut cpu_engine = latticefold_plus::ntt::NttEngine::new(params.clone()).unwrap();
                b.iter(|| {
                    black_box(cpu_engine.inverse_ntt(black_box(&ntt_coefficients)).unwrap())
                });
            },
        );
        
        // Benchmark GPU inverse NTT
        #[cfg(feature = "gpu")]
        {
            if let Ok(mut gpu_engine) = AdaptiveNttEngine::new(params.clone(), true) {
                // Warm up
                for _ in 0..WARMUP_ITERATIONS {
                    let _ = gpu_engine.inverse_ntt(&ntt_coefficients);
                }
                
                group.bench_with_input(
                    BenchmarkId::new("gpu", dimension),
                    &dimension,
                    |b, &_dim| {
                        b.iter(|| {
                            black_box(gpu_engine.inverse_ntt(black_box(&ntt_coefficients)).unwrap())
                        });
                    },
                );
            }
        }
        
        // Benchmark adaptive inverse NTT
        if let Ok(mut adaptive_engine) = AdaptiveNttEngine::new(params.clone(), true) {
            for _ in 0..WARMUP_ITERATIONS {
                let _ = adaptive_engine.inverse_ntt(&ntt_coefficients);
            }
            
            group.bench_with_input(
                BenchmarkId::new("adaptive", dimension),
                &dimension,
                |b, &_dim| {
                    b.iter(|| {
                        black_box(adaptive_engine.inverse_ntt(black_box(&ntt_coefficients)).unwrap())
                    });
                },
            );
        }
    }
    
    group.finish();
}

/// Benchmarks batch NTT operations
/// 
/// Measures the performance benefits of batch processing for multiple polynomials.
fn bench_batch_ntt(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_ntt");
    group.measurement_time(MEASUREMENT_TIME);
    
    let mut rng = ChaCha20Rng::seed_from_u64(34567);
    let dimension = 1024; // Fixed dimension for batch testing
    
    if let Err(_) = create_benchmark_params(dimension) {
        return; // Skip if dimension not supported
    }
    
    let params = create_benchmark_params(dimension).unwrap();
    
    for &batch_size in BENCHMARK_BATCH_SIZES {
        let batch_coefficients = generate_benchmark_batch(batch_size, dimension, &mut rng);
        
        // Set throughput based on total elements processed
        group.throughput(Throughput::Elements((batch_size * dimension) as u64));
        
        // Benchmark individual operations (baseline)
        group.bench_with_input(
            BenchmarkId::new("individual", batch_size),
            &batch_size,
            |b, &_size| {
                let mut engine = AdaptiveNttEngine::new(params.clone(), false).unwrap();
                b.iter(|| {
                    let mut results = Vec::new();
                    for poly in black_box(&batch_coefficients) {
                        results.push(engine.forward_ntt(poly).unwrap());
                    }
                    black_box(results)
                });
            },
        );
        
        // Benchmark batch operations (GPU)
        #[cfg(feature = "gpu")]
        {
            if let Ok(mut gpu_engine) = AdaptiveNttEngine::new(params.clone(), true) {
                // Warm up
                for _ in 0..WARMUP_ITERATIONS {
                    let _ = gpu_engine.batch_ntt(&batch_coefficients, true);
                }
                
                group.bench_with_input(
                    BenchmarkId::new("batch_gpu", batch_size),
                    &batch_size,
                    |b, &_size| {
                        b.iter(|| {
                            black_box(gpu_engine.batch_ntt(black_box(&batch_coefficients), true).unwrap())
                        });
                    },
                );
            }
        }
        
        // Benchmark adaptive batch operations
        if let Ok(mut adaptive_engine) = AdaptiveNttEngine::new(params.clone(), true) {
            for _ in 0..WARMUP_ITERATIONS {
                let _ = adaptive_engine.batch_ntt(&batch_coefficients, true);
            }
            
            group.bench_with_input(
                BenchmarkId::new("batch_adaptive", batch_size),
                &batch_size,
                |b, &_size| {
                    b.iter(|| {
                        black_box(adaptive_engine.batch_ntt(black_box(&batch_coefficients), true).unwrap())
                    });
                },
            );
        }
    }
    
    group.finish();
}

/// Benchmarks NTT-based polynomial multiplication
/// 
/// Measures the performance of complete polynomial multiplication using NTT,
/// which is the primary use case in cryptographic applications.
fn bench_ntt_multiplication(c: &mut Criterion) {
    let mut group = c.benchmark_group("ntt_multiplication");
    group.measurement_time(MEASUREMENT_TIME);
    
    let mut rng = ChaCha20Rng::seed_from_u64(45678);
    
    for &dimension in BENCHMARK_DIMENSIONS {
        if let Err(_) = create_benchmark_params(dimension) {
            continue;
        }
        
        let params = create_benchmark_params(dimension).unwrap();
        let poly_a = generate_benchmark_polynomial(dimension, &mut rng);
        let poly_b = generate_benchmark_polynomial(dimension, &mut rng);
        
        group.throughput(Throughput::Elements(dimension as u64));
        
        // Benchmark CPU NTT multiplication
        group.bench_with_input(
            BenchmarkId::new("cpu_ntt_mult", dimension),
            &dimension,
            |b, &_dim| {
                let mut cpu_engine = latticefold_plus::ntt::NttEngine::new(params.clone()).unwrap();
                b.iter(|| {
                    // Forward NTT
                    let ntt_a = cpu_engine.forward_ntt(black_box(&poly_a)).unwrap();
                    let ntt_b = cpu_engine.forward_ntt(black_box(&poly_b)).unwrap();
                    
                    // Pointwise multiplication
                    let mut ntt_product = Vec::with_capacity(dimension);
                    for (a_coeff, b_coeff) in ntt_a.iter().zip(ntt_b.iter()) {
                        ntt_product.push(((*a_coeff as i128 * *b_coeff as i128) % BENCHMARK_MODULUS as i128) as i64);
                    }
                    
                    // Inverse NTT
                    black_box(cpu_engine.inverse_ntt(&ntt_product).unwrap())
                });
            },
        );
        
        // Benchmark GPU NTT multiplication
        #[cfg(feature = "gpu")]
        {
            if let Ok(mut gpu_engine) = AdaptiveNttEngine::new(params.clone(), true) {
                // Warm up
                for _ in 0..WARMUP_ITERATIONS {
                    let ntt_a = gpu_engine.forward_ntt(&poly_a).unwrap();
                    let ntt_b = gpu_engine.forward_ntt(&poly_b).unwrap();
                    let mut ntt_product = Vec::with_capacity(dimension);
                    for (a_coeff, b_coeff) in ntt_a.iter().zip(ntt_b.iter()) {
                        ntt_product.push(((*a_coeff as i128 * *b_coeff as i128) % BENCHMARK_MODULUS as i128) as i64);
                    }
                    let _ = gpu_engine.inverse_ntt(&ntt_product);
                }
                
                group.bench_with_input(
                    BenchmarkId::new("gpu_ntt_mult", dimension),
                    &dimension,
                    |b, &_dim| {
                        b.iter(|| {
                            // Forward NTT
                            let ntt_a = gpu_engine.forward_ntt(black_box(&poly_a)).unwrap();
                            let ntt_b = gpu_engine.forward_ntt(black_box(&poly_b)).unwrap();
                            
                            // Pointwise multiplication
                            let mut ntt_product = Vec::with_capacity(dimension);
                            for (a_coeff, b_coeff) in ntt_a.iter().zip(ntt_b.iter()) {
                                ntt_product.push(((*a_coeff as i128 * *b_coeff as i128) % BENCHMARK_MODULUS as i128) as i64);
                            }
                            
                            // Inverse NTT
                            black_box(gpu_engine.inverse_ntt(&ntt_product).unwrap())
                        });
                    },
                );
            }
        }
    }
    
    group.finish();
}

/// Benchmarks memory transfer overhead
/// 
/// Measures the impact of CPU-GPU memory transfers on overall performance.
fn bench_memory_transfer_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_transfer_overhead");
    group.measurement_time(MEASUREMENT_TIME);
    
    let mut rng = ChaCha20Rng::seed_from_u64(56789);
    
    for &dimension in &[1024, 4096, 16384] { // Focus on larger dimensions
        if let Err(_) = create_benchmark_params(dimension) {
            continue;
        }
        
        let coefficients = generate_benchmark_polynomial(dimension, &mut rng);
        
        group.throughput(Throughput::Bytes((dimension * std::mem::size_of::<i64>()) as u64));
        
        // Benchmark pure computation (no transfer overhead)
        group.bench_with_input(
            BenchmarkId::new("cpu_no_transfer", dimension),
            &dimension,
            |b, &_dim| {
                let params = create_benchmark_params(dimension).unwrap();
                let mut cpu_engine = latticefold_plus::ntt::NttEngine::new(params).unwrap();
                b.iter(|| {
                    black_box(cpu_engine.forward_ntt(black_box(&coefficients)).unwrap())
                });
            },
        );
        
        // Benchmark with simulated transfer overhead
        #[cfg(feature = "gpu")]
        {
            group.bench_with_input(
                BenchmarkId::new("gpu_with_transfer", dimension),
                &dimension,
                |b, &_dim| {
                    let params = create_benchmark_params(dimension).unwrap();
                    if let Ok(mut gpu_engine) = AdaptiveNttEngine::new(params, true) {
                        b.iter(|| {
                            // This includes actual GPU memory transfer overhead
                            black_box(gpu_engine.forward_ntt(black_box(&coefficients)).unwrap())
                        });
                    }
                },
            );
        }
    }
    
    group.finish();
}

/// Benchmarks algorithm selection overhead
/// 
/// Measures the performance impact of adaptive algorithm selection.
fn bench_algorithm_selection(c: &mut Criterion) {
    let mut group = c.benchmark_group("algorithm_selection");
    group.measurement_time(MEASUREMENT_TIME);
    
    let mut rng = ChaCha20Rng::seed_from_u64(67890);
    let dimension = 2048; // Fixed dimension for selection testing
    
    if let Err(_) = create_benchmark_params(dimension) {
        return;
    }
    
    let params = create_benchmark_params(dimension).unwrap();
    let coefficients = generate_benchmark_polynomial(dimension, &mut rng);
    
    group.throughput(Throughput::Elements(dimension as u64));
    
    // Benchmark fixed CPU selection
    group.bench_function("fixed_cpu", |b| {
        let mut cpu_engine = latticefold_plus::ntt::NttEngine::new(params.clone()).unwrap();
        b.iter(|| {
            black_box(cpu_engine.forward_ntt(black_box(&coefficients)).unwrap())
        });
    });
    
    // Benchmark adaptive selection (with decision overhead)
    group.bench_function("adaptive_selection", |b| {
        let mut adaptive_engine = AdaptiveNttEngine::new(params.clone(), true).unwrap();
        b.iter(|| {
            black_box(adaptive_engine.forward_ntt(black_box(&coefficients)).unwrap())
        });
    });
    
    // Benchmark adaptive selection after learning (should be faster)
    group.bench_function("adaptive_learned", |b| {
        let mut adaptive_engine = AdaptiveNttEngine::new(params.clone(), true).unwrap();
        
        // Train the adaptive engine with several operations
        for _ in 0..20 {
            let _ = adaptive_engine.forward_ntt(&coefficients);
        }
        
        b.iter(|| {
            black_box(adaptive_engine.forward_ntt(black_box(&coefficients)).unwrap())
        });
    });
    
    group.finish();
}

/// Benchmarks scalability across different problem sizes
/// 
/// Measures how performance scales with polynomial dimension to identify
/// optimal operating ranges for different implementations.
fn bench_scalability_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("scalability_analysis");
    group.measurement_time(MEASUREMENT_TIME);
    
    let mut rng = ChaCha20Rng::seed_from_u64(78901);
    
    // Extended dimension range for scalability analysis
    let scalability_dimensions = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384];
    
    for &dimension in &scalability_dimensions {
        if let Err(_) = create_benchmark_params(dimension) {
            continue;
        }
        
        let params = create_benchmark_params(dimension).unwrap();
        let coefficients = generate_benchmark_polynomial(dimension, &mut rng);
        
        // Measure operations per second for scalability analysis
        group.throughput(Throughput::Elements(dimension as u64));
        
        // CPU scalability
        group.bench_with_input(
            BenchmarkId::new("cpu_scalability", dimension),
            &dimension,
            |b, &_dim| {
                let mut cpu_engine = latticefold_plus::ntt::NttEngine::new(params.clone()).unwrap();
                b.iter(|| {
                    black_box(cpu_engine.forward_ntt(black_box(&coefficients)).unwrap())
                });
            },
        );
        
        // GPU scalability (if available)
        #[cfg(feature = "gpu")]
        {
            if let Ok(mut gpu_engine) = AdaptiveNttEngine::new(params.clone(), true) {
                // Warm up for consistent measurements
                for _ in 0..WARMUP_ITERATIONS {
                    let _ = gpu_engine.forward_ntt(&coefficients);
                }
                
                group.bench_with_input(
                    BenchmarkId::new("gpu_scalability", dimension),
                    &dimension,
                    |b, &_dim| {
                        b.iter(|| {
                            black_box(gpu_engine.forward_ntt(black_box(&coefficients)).unwrap())
                        });
                    },
                );
            }
        }
    }
    
    group.finish();
}

/// Benchmarks real-world cryptographic workloads
/// 
/// Simulates realistic usage patterns in cryptographic protocols
/// such as commitment schemes and zero-knowledge proofs.
fn bench_cryptographic_workloads(c: &mut Criterion) {
    let mut group = c.benchmark_group("cryptographic_workloads");
    group.measurement_time(MEASUREMENT_TIME);
    
    let mut rng = ChaCha20Rng::seed_from_u64(89012);
    let dimension = 2048; // Typical cryptographic dimension
    
    if let Err(_) = create_benchmark_params(dimension) {
        return;
    }
    
    let params = create_benchmark_params(dimension).unwrap();
    
    // Simulate commitment scheme workload (multiple polynomial operations)
    group.bench_function("commitment_scheme_workload", |b| {
        let mut adaptive_engine = AdaptiveNttEngine::new(params.clone(), true).unwrap();
        
        b.iter(|| {
            // Simulate commitment to multiple polynomials
            let mut commitments = Vec::new();
            for _ in 0..8 { // Typical number of commitments in a round
                let poly = generate_benchmark_polynomial(dimension, &mut rng);
                let ntt_poly = adaptive_engine.forward_ntt(&poly).unwrap();
                commitments.push(ntt_poly);
            }
            
            // Simulate verification operations (inverse NTTs)
            let mut results = Vec::new();
            for commitment in commitments {
                results.push(adaptive_engine.inverse_ntt(&commitment).unwrap());
            }
            
            black_box(results)
        });
    });
    
    // Simulate zero-knowledge proof workload (batch operations)
    group.bench_function("zkp_batch_workload", |b| {
        let mut adaptive_engine = AdaptiveNttEngine::new(params.clone(), true).unwrap();
        
        b.iter(|| {
            // Generate batch of witness polynomials
            let witness_batch = generate_benchmark_batch(16, dimension, &mut rng);
            
            // Forward NTT for all witnesses (prover computation)
            let ntt_batch = adaptive_engine.batch_ntt(&witness_batch, true).unwrap();
            
            // Simulate some processing in NTT domain
            let mut processed_batch = Vec::new();
            for ntt_poly in ntt_batch {
                let mut processed = Vec::with_capacity(dimension);
                for coeff in ntt_poly {
                    processed.push((coeff * 2) % BENCHMARK_MODULUS); // Simple processing
                }
                processed_batch.push(processed);
            }
            
            // Inverse NTT for verification
            let final_batch = adaptive_engine.batch_ntt(&processed_batch, false).unwrap();
            
            black_box(final_batch)
        });
    });
    
    // Simulate folding protocol workload (mixed operations)
    group.bench_function("folding_protocol_workload", |b| {
        let mut adaptive_engine = AdaptiveNttEngine::new(params.clone(), true).unwrap();
        
        b.iter(|| {
            // Simulate folding of multiple instances
            let mut folded_result = generate_benchmark_polynomial(dimension, &mut rng);
            
            for _ in 0..4 { // Fold 4 instances
                let instance = generate_benchmark_polynomial(dimension, &mut rng);
                
                // Forward NTT for both polynomials
                let ntt_folded = adaptive_engine.forward_ntt(&folded_result).unwrap();
                let ntt_instance = adaptive_engine.forward_ntt(&instance).unwrap();
                
                // Combine in NTT domain (simulated folding)
                let mut ntt_combined = Vec::with_capacity(dimension);
                for (f_coeff, i_coeff) in ntt_folded.iter().zip(ntt_instance.iter()) {
                    ntt_combined.push((f_coeff + i_coeff) % BENCHMARK_MODULUS);
                }
                
                // Inverse NTT to get new folded result
                folded_result = adaptive_engine.inverse_ntt(&ntt_combined).unwrap();
            }
            
            black_box(folded_result)
        });
    });
    
    group.finish();
}

// Define benchmark groups
criterion_group!(
    benches,
    bench_single_forward_ntt,
    bench_single_inverse_ntt,
    bench_batch_ntt,
    bench_ntt_multiplication,
    bench_memory_transfer_overhead,
    bench_algorithm_selection,
    bench_scalability_analysis,
    bench_cryptographic_workloads
);

criterion_main!(benches);