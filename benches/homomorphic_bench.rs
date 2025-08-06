/// Performance benchmarks for homomorphic commitment operations
/// 
/// This benchmark suite measures the performance of various homomorphic operations
/// on commitment schemes, comparing batch vs individual operations and analyzing
/// the efficiency gains from the optimized implementations.
/// 
/// Benchmark Categories:
/// 1. Individual Operations: Basic add, scale, and linear combination operations
/// 2. Batch Operations: Batch processing of multiple commitments
/// 3. Zero-Knowledge Operations: ZK-preserving homomorphic operations
/// 4. Scalability Tests: Performance with varying commitment counts and dimensions
/// 
/// Performance Targets (based on LatticeFold+ paper claims):
/// - 5x speedup over baseline implementations
/// - Linear scaling with number of commitments
/// - Constant overhead for batch operations
/// - SIMD acceleration benefits for large dimensions

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use latticefold_plus::{
    commitment::{CommitmentParams, SISCommitmentScheme, HomomorphicCommitmentScheme, CommitmentScheme},
    lattice::LatticeParams,
    error::Result,
};
use rand::{thread_rng, Rng};
use std::time::Duration;

/// Benchmark configuration parameters
struct BenchConfig {
    /// Lattice dimension for testing
    dimension: usize,
    /// Modulus for arithmetic operations
    modulus: i64,
    /// Number of commitments for batch operations
    batch_sizes: Vec<usize>,
    /// Number of iterations for statistical significance
    iterations: usize,
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            dimension: 256,
            modulus: 2147483647, // 2^31 - 1
            batch_sizes: vec![1, 10, 50, 100, 500, 1000],
            iterations: 100,
        }
    }
}

/// Setup function to create commitment scheme and test data
fn setup_commitment_scheme(config: &BenchConfig) -> Result<(SISCommitmentScheme, Vec<Vec<u8>>, Vec<crate::lattice::LatticePoint>)> {
    let mut rng = thread_rng();
    
    // Create lattice parameters
    let lattice_params = LatticeParams {
        dimension: config.dimension,
        modulus: config.modulus,
        gaussian_width: 3.0,
    };
    
    // Create commitment parameters
    let commitment_params = CommitmentParams {
        lattice_params: lattice_params.clone(),
        dimension_factor: 2,
        hiding: true,
        security_param: 128,
    };
    
    // Create commitment scheme
    let scheme = SISCommitmentScheme::new(commitment_params)?;
    
    // Generate test messages
    let max_batch_size = *config.batch_sizes.iter().max().unwrap_or(&1000);
    let mut messages = Vec::with_capacity(max_batch_size);
    let mut randomness_values = Vec::with_capacity(max_batch_size);
    
    for _ in 0..max_batch_size {
        // Generate random message
        let message: Vec<u8> = (0..config.dimension * 8)
            .map(|_| rng.gen::<u8>())
            .collect();
        messages.push(message);
        
        // Generate random randomness
        let randomness = scheme.random_randomness(&mut rng)?;
        randomness_values.push(randomness);
    }
    
    Ok((scheme, messages, randomness_values))
}

/// Benchmark individual homomorphic addition operations
fn bench_individual_addition(c: &mut Criterion) {
    let config = BenchConfig::default();
    let (scheme, messages, randomness_values) = setup_commitment_scheme(&config).unwrap();
    
    // Create test commitments
    let commitment1 = scheme.commit(&messages[0], &randomness_values[0]).unwrap();
    let commitment2 = scheme.commit(&messages[1], &randomness_values[1]).unwrap();
    
    let mut group = c.benchmark_group("individual_addition");
    group.throughput(Throughput::Elements(1));
    group.measurement_time(Duration::from_secs(10));
    
    group.bench_function("add_commitments", |b| {
        b.iter(|| {
            black_box(scheme.add_commitments(
                black_box(&commitment1),
                black_box(&commitment2)
            ).unwrap())
        })
    });
    
    group.finish();
}

/// Benchmark individual homomorphic scaling operations
fn bench_individual_scaling(c: &mut Criterion) {
    let config = BenchConfig::default();
    let (scheme, messages, randomness_values) = setup_commitment_scheme(&config).unwrap();
    
    // Create test commitment
    let commitment = scheme.commit(&messages[0], &randomness_values[0]).unwrap();
    let scalar = 12345i64;
    
    let mut group = c.benchmark_group("individual_scaling");
    group.throughput(Throughput::Elements(1));
    group.measurement_time(Duration::from_secs(10));
    
    group.bench_function("scale_commitment", |b| {
        b.iter(|| {
            black_box(scheme.scale_commitment(
                black_box(&commitment),
                black_box(scalar)
            ).unwrap())
        })
    });
    
    group.finish();
}

/// Benchmark optimized add-scaled operations
fn bench_add_scaled_operations(c: &mut Criterion) {
    let config = BenchConfig::default();
    let (scheme, messages, randomness_values) = setup_commitment_scheme(&config).unwrap();
    
    // Create test commitments
    let commitment1 = scheme.commit(&messages[0], &randomness_values[0]).unwrap();
    let commitment2 = scheme.commit(&messages[1], &randomness_values[1]).unwrap();
    let scalar = 12345i64;
    
    let mut group = c.benchmark_group("add_scaled_operations");
    group.throughput(Throughput::Elements(1));
    group.measurement_time(Duration::from_secs(10));
    
    // Benchmark optimized add-scaled operation
    group.bench_function("add_scaled_commitment", |b| {
        b.iter(|| {
            black_box(scheme.add_scaled_commitment(
                black_box(&commitment1),
                black_box(&commitment2),
                black_box(scalar)
            ).unwrap())
        })
    });
    
    // Benchmark separate scale-then-add for comparison
    group.bench_function("separate_scale_then_add", |b| {
        b.iter(|| {
            let scaled = scheme.scale_commitment(black_box(&commitment2), black_box(scalar)).unwrap();
            black_box(scheme.add_commitments(black_box(&commitment1), &scaled).unwrap())
        })
    });
    
    group.finish();
}

/// Benchmark linear combination operations with varying numbers of commitments
fn bench_linear_combinations(c: &mut Criterion) {
    let config = BenchConfig::default();
    let (scheme, messages, randomness_values) = setup_commitment_scheme(&config).unwrap();
    
    let mut group = c.benchmark_group("linear_combinations");
    group.measurement_time(Duration::from_secs(15));
    
    for &size in &[2, 5, 10, 20, 50] {
        // Create commitments for linear combination
        let mut commitments = Vec::with_capacity(size);
        for i in 0..size {
            let commitment = scheme.commit(&messages[i], &randomness_values[i]).unwrap();
            commitments.push(commitment);
        }
        
        // Create random scalars
        let mut rng = thread_rng();
        let scalars: Vec<i64> = (0..size).map(|_| rng.gen_range(1..1000)).collect();
        
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::new("linear_combination", size),
            &size,
            |b, _| {
                b.iter(|| {
                    black_box(scheme.linear_combination(
                        black_box(&commitments),
                        black_box(&scalars)
                    ).unwrap())
                })
            }
        );
    }
    
    group.finish();
}

/// Benchmark batch addition operations
fn bench_batch_addition(c: &mut Criterion) {
    let config = BenchConfig::default();
    let (scheme, messages, randomness_values) = setup_commitment_scheme(&config).unwrap();
    
    let mut group = c.benchmark_group("batch_addition");
    group.measurement_time(Duration::from_secs(15));
    
    for &batch_size in &config.batch_sizes {
        if batch_size > messages.len() { continue; }
        
        // Create two sets of commitments for batch addition
        let mut commitments1 = Vec::with_capacity(batch_size);
        let mut commitments2 = Vec::with_capacity(batch_size);
        
        for i in 0..batch_size {
            let c1 = scheme.commit(&messages[i], &randomness_values[i]).unwrap();
            let c2 = scheme.commit(&messages[(i + batch_size) % messages.len()], 
                                 &randomness_values[(i + batch_size) % randomness_values.len()]).unwrap();
            commitments1.push(c1);
            commitments2.push(c2);
        }
        
        group.throughput(Throughput::Elements(batch_size as u64));
        
        // Benchmark batch addition
        group.bench_with_input(
            BenchmarkId::new("batch_add", batch_size),
            &batch_size,
            |b, _| {
                b.iter(|| {
                    black_box(scheme.batch_add_commitments(
                        black_box(&commitments1),
                        black_box(&commitments2)
                    ).unwrap())
                })
            }
        );
        
        // Benchmark individual additions for comparison
        group.bench_with_input(
            BenchmarkId::new("individual_adds", batch_size),
            &batch_size,
            |b, _| {
                b.iter(|| {
                    let mut results = Vec::with_capacity(batch_size);
                    for i in 0..batch_size {
                        let result = scheme.add_commitments(&commitments1[i], &commitments2[i]).unwrap();
                        results.push(result);
                    }
                    black_box(results)
                })
            }
        );
    }
    
    group.finish();
}

/// Benchmark batch scaling operations
fn bench_batch_scaling(c: &mut Criterion) {
    let config = BenchConfig::default();
    let (scheme, messages, randomness_values) = setup_commitment_scheme(&config).unwrap();
    
    let mut group = c.benchmark_group("batch_scaling");
    group.measurement_time(Duration::from_secs(15));
    
    for &batch_size in &config.batch_sizes {
        if batch_size > messages.len() { continue; }
        
        // Create commitments for batch scaling
        let mut commitments = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let commitment = scheme.commit(&messages[i], &randomness_values[i]).unwrap();
            commitments.push(commitment);
        }
        
        // Create random scalars
        let mut rng = thread_rng();
        let scalars: Vec<i64> = (0..batch_size).map(|_| rng.gen_range(1..1000)).collect();
        
        group.throughput(Throughput::Elements(batch_size as u64));
        
        // Benchmark batch scaling
        group.bench_with_input(
            BenchmarkId::new("batch_scale", batch_size),
            &batch_size,
            |b, _| {
                b.iter(|| {
                    black_box(scheme.batch_scale_commitments(
                        black_box(&commitments),
                        black_box(&scalars)
                    ).unwrap())
                })
            }
        );
        
        // Benchmark individual scaling for comparison
        group.bench_with_input(
            BenchmarkId::new("individual_scales", batch_size),
            &batch_size,
            |b, _| {
                b.iter(|| {
                    let mut results = Vec::with_capacity(batch_size);
                    for i in 0..batch_size {
                        let result = scheme.scale_commitment(&commitments[i], scalars[i]).unwrap();
                        results.push(result);
                    }
                    black_box(results)
                })
            }
        );
    }
    
    group.finish();
}

/// Benchmark zero-knowledge homomorphic operations
fn bench_zero_knowledge_operations(c: &mut Criterion) {
    let config = BenchConfig::default();
    let (scheme, messages, randomness_values) = setup_commitment_scheme(&config).unwrap();
    
    // Create test commitments with known randomness
    let commitment1 = scheme.commit(&messages[0], &randomness_values[0]).unwrap();
    let commitment2 = scheme.commit(&messages[1], &randomness_values[1]).unwrap();
    let scalar = 12345i64;
    
    let mut group = c.benchmark_group("zero_knowledge_operations");
    group.throughput(Throughput::Elements(1));
    group.measurement_time(Duration::from_secs(10));
    
    // Benchmark zero-knowledge addition
    group.bench_function("zk_add_commitments", |b| {
        b.iter(|| {
            black_box(scheme.zk_add_commitments(
                black_box(&commitment1),
                black_box(&randomness_values[0]),
                black_box(&commitment2),
                black_box(&randomness_values[1])
            ).unwrap())
        })
    });
    
    // Benchmark zero-knowledge scaling
    group.bench_function("zk_scale_commitment", |b| {
        b.iter(|| {
            black_box(scheme.zk_scale_commitment(
                black_box(&commitment1),
                black_box(&randomness_values[0]),
                black_box(scalar)
            ).unwrap())
        })
    });
    
    group.finish();
}

/// Benchmark scalability with different lattice dimensions
fn bench_dimension_scalability(c: &mut Criterion) {
    let dimensions = vec![64, 128, 256, 512, 1024];
    
    let mut group = c.benchmark_group("dimension_scalability");
    group.measurement_time(Duration::from_secs(20));
    
    for &dimension in &dimensions {
        let config = BenchConfig {
            dimension,
            ..Default::default()
        };
        
        let (scheme, messages, randomness_values) = setup_commitment_scheme(&config).unwrap();
        
        // Create test commitments
        let commitment1 = scheme.commit(&messages[0], &randomness_values[0]).unwrap();
        let commitment2 = scheme.commit(&messages[1], &randomness_values[1]).unwrap();
        
        group.throughput(Throughput::Elements(dimension as u64));
        
        // Benchmark addition with different dimensions
        group.bench_with_input(
            BenchmarkId::new("add_by_dimension", dimension),
            &dimension,
            |b, _| {
                b.iter(|| {
                    black_box(scheme.add_commitments(
                        black_box(&commitment1),
                        black_box(&commitment2)
                    ).unwrap())
                })
            }
        );
        
        // Benchmark scaling with different dimensions
        group.bench_with_input(
            BenchmarkId::new("scale_by_dimension", dimension),
            &dimension,
            |b, _| {
                b.iter(|| {
                    black_box(scheme.scale_commitment(
                        black_box(&commitment1),
                        black_box(12345i64)
                    ).unwrap())
                })
            }
        );
    }
    
    group.finish();
}

/// Benchmark memory efficiency of batch operations
fn bench_memory_efficiency(c: &mut Criterion) {
    let config = BenchConfig::default();
    let (scheme, messages, randomness_values) = setup_commitment_scheme(&config).unwrap();
    
    let batch_size = 100;
    let mut commitments = Vec::with_capacity(batch_size);
    
    for i in 0..batch_size {
        let commitment = scheme.commit(&messages[i], &randomness_values[i]).unwrap();
        commitments.push(commitment);
    }
    
    let scalars: Vec<i64> = (0..batch_size).map(|i| (i as i64) + 1).collect();
    
    let mut group = c.benchmark_group("memory_efficiency");
    group.throughput(Throughput::Elements(batch_size as u64));
    group.measurement_time(Duration::from_secs(10));
    
    // Benchmark memory-efficient linear combination
    group.bench_function("linear_combination_memory_efficient", |b| {
        b.iter(|| {
            black_box(scheme.linear_combination(
                black_box(&commitments),
                black_box(&scalars)
            ).unwrap())
        })
    });
    
    // Benchmark naive approach with intermediate allocations
    group.bench_function("linear_combination_naive", |b| {
        b.iter(|| {
            let mut result = scheme.scale_commitment(&commitments[0], scalars[0]).unwrap();
            for i in 1..batch_size {
                let scaled = scheme.scale_commitment(&commitments[i], scalars[i]).unwrap();
                result = scheme.add_commitments(&result, &scaled).unwrap();
            }
            black_box(result)
        })
    });
    
    group.finish();
}

/// Performance analysis and reporting
fn bench_performance_analysis(c: &mut Criterion) {
    let config = BenchConfig::default();
    let (scheme, messages, randomness_values) = setup_commitment_scheme(&config).unwrap();
    
    // Create comprehensive test data
    let test_sizes = vec![1, 10, 100, 1000];
    
    let mut group = c.benchmark_group("performance_analysis");
    group.measurement_time(Duration::from_secs(30));
    
    for &size in &test_sizes {
        if size > messages.len() { continue; }
        
        let mut commitments = Vec::with_capacity(size);
        for i in 0..size {
            let commitment = scheme.commit(&messages[i], &randomness_values[i]).unwrap();
            commitments.push(commitment);
        }
        
        let scalars: Vec<i64> = (0..size).map(|i| (i as i64) + 1).collect();
        
        group.throughput(Throughput::Elements(size as u64));
        
        // Measure throughput for different operation types
        group.bench_with_input(
            BenchmarkId::new("throughput_analysis", size),
            &size,
            |b, _| {
                b.iter(|| {
                    // Perform a mix of operations to simulate real usage
                    if size >= 2 {
                        let _ = scheme.add_commitments(&commitments[0], &commitments[1]);
                    }
                    if size >= 1 {
                        let _ = scheme.scale_commitment(&commitments[0], scalars[0]);
                    }
                    if size >= 3 {
                        let _ = scheme.linear_combination(&commitments[0..3], &scalars[0..3]);
                    }
                    black_box(())
                })
            }
        );
    }
    
    group.finish();
}

// Define benchmark groups
criterion_group!(
    homomorphic_benches,
    bench_individual_addition,
    bench_individual_scaling,
    bench_add_scaled_operations,
    bench_linear_combinations,
    bench_batch_addition,
    bench_batch_scaling,
    bench_zero_knowledge_operations,
    bench_dimension_scalability,
    bench_memory_efficiency,
    bench_performance_analysis
);

criterion_main!(homomorphic_benches);