use criterion::{black_box, criterion_group, criterion_main, Criterion};
use latticefold_plus::{LatticePoint, LatticeParams, LatticeMatrix, FoldingScheme};
use rand::thread_rng;
use merlin::Transcript;

fn bench_lattice_operations(c: &mut Criterion) {
    let params = LatticeParams {
        q: 1031, // Larger prime for realistic benchmarking
        n: 128,  // Realistic dimension
        sigma: 3.0,
        beta: 2.0,
    };

    let mut rng = thread_rng();
    let point1 = LatticePoint::random(&params, &mut rng);
    let point2 = LatticePoint::random(&params, &mut rng);

    c.bench_function("lattice_add", |b| {
        b.iter(|| {
            black_box(point1.clone().add_mod(&point2, params.q))
        })
    });

    c.bench_function("lattice_sub", |b| {
        b.iter(|| {
            black_box(point1.clone().sub_mod(&point2, params.q))
        })
    });

    c.bench_function("lattice_scalar_mul", |b| {
        b.iter(|| {
            black_box(point1.clone().scalar_mul_mod(5, params.q))
        })
    });
}

fn bench_folding_scheme(c: &mut Criterion) {
    let params = LatticeParams {
        q: 1031,
        n: 128,
        sigma: 3.0,
        beta: 2.0,
    };

    let basis_matrix = LatticeMatrix::random(
        params.n,
        params.n,
        &params,
        &mut thread_rng(),
    );

    let mut scheme = FoldingScheme::new(params.clone(), basis_matrix);
    let mut rng = thread_rng();

    // Generate test points
    let points: Vec<LatticePoint> = (0..10)
        .map(|_| LatticePoint::random(&params, &mut rng))
        .collect();

    c.bench_function("folding_prove", |b| {
        b.iter(|| {
            let mut transcript = Transcript::new(b"bench_folding");
            black_box(scheme.prove(&points, &mut transcript, &mut rng))
        })
    });

    let mut transcript = Transcript::new(b"bench_folding");
    let proof = scheme.prove(&points, &mut transcript, &mut rng);

    c.bench_function("folding_verify", |b| {
        b.iter(|| {
            let mut transcript = Transcript::new(b"bench_folding");
            black_box(scheme.verify(&proof, &mut transcript))
        })
    });
}

fn bench_gram_schmidt(c: &mut Criterion) {
    let params = LatticeParams {
        q: 1031,
        n: 128,
        sigma: 3.0,
        beta: 2.0,
    };

    let matrix = LatticeMatrix::random(
        params.n,
        params.n,
        &params,
        &mut thread_rng(),
    );

    c.bench_function("gram_schmidt", |b| {
        b.iter(|| {
            black_box(matrix.gram_schmidt())
        })
    });
}

criterion_group!(
    benches,
    bench_lattice_operations,
    bench_folding_scheme,
    bench_gram_schmidt
);
criterion_main!(benches);
