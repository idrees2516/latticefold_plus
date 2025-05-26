use criterion::{black_box, criterion_group, criterion_main, Criterion};
use latticefold_plus::*;
use ark_bls12_381::{Bls12_381, Fr};
use ark_std::test_rng;

fn bench_lattice_fold(c: &mut Criterion) {
    let mut rng = test_rng();
    let dimension = 10;
    let security_param = 128;

    // Setup
    let params = setup_lattice_fold::<Bls12_381, Fr, _>(dimension, security_param, &mut rng);

    // Generate test instances
    let mut instances = Vec::new();
    for _ in 0..3 {
        let witness = vec![Fr::rand(&mut rng); dimension];
        let public_input = vec![Fr::rand(&mut rng); dimension];
        instances.push(LatticeFoldInstance {
            witness,
            public_input,
        });
    }

    // Benchmark proving
    c.bench_function("prove_lattice_fold", |b| {
        b.iter(|| {
            let proof = prove_lattice_fold(black_box(&params), black_box(&instances), &mut rng);
            black_box(proof);
        });
    });

    // Benchmark verification
    let proof = prove_lattice_fold(&params, &instances, &mut rng);
    c.bench_function("verify_lattice_fold", |b| {
        b.iter(|| {
            let result = verify_lattice_fold(black_box(&params), black_box(&proof));
            black_box(result);
        });
    });
}

criterion_group!(benches, bench_lattice_fold);
criterion_main!(benches); 