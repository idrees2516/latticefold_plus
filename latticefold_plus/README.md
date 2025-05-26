# LatticeFold+

A Rust implementation of LatticeFold+: Faster, Simpler, Shorter Lattice-Based Folding for Succinct Proof Systems.

## Features

- Efficient lattice operations with modular arithmetic
- Secure folding scheme for succinct proof systems
- Optimized Gram-Schmidt orthogonalization
- Comprehensive test suite and benchmarks
- Thread-safe and zero-copy where possible
- Constant-time operations for cryptographic security

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
latticefold_plus = "0.1.0"
```

Basic example:

```rust
use latticefold_plus::{LatticeParams, LatticePoint, LatticeMatrix, FoldingScheme};
use rand::thread_rng;
use merlin::Transcript;

fn main() {
    // Initialize parameters
    let params = LatticeParams {
        q: 1031,  // Modulus (prime)
        n: 128,   // Dimension
        sigma: 3.0, // Gaussian parameter
        beta: 2.0,  // Smoothing parameter
    };

    // Create random points
    let mut rng = thread_rng();
    let point1 = LatticePoint::random(&params, &mut rng);
    let point2 = LatticePoint::random(&params, &mut rng);

    // Initialize folding scheme
    let basis_matrix = LatticeMatrix::random(
        params.n,
        params.n,
        &params,
        &mut rng,
    );
    let mut scheme = FoldingScheme::new(params, basis_matrix);

    // Generate and verify proof
    let points = vec![point1, point2];
    let mut transcript = Transcript::new(b"example");
    let proof = scheme.prove(&points, &mut transcript, &mut rng);

    let mut verify_transcript = Transcript::new(b"example");
    assert!(scheme.verify(&proof, &mut verify_transcript));
}
```

## Security Features

- Constant-time operations to prevent timing attacks
- Secure random number generation using `rand`
- Zeroization of sensitive data
- Side-channel resistant implementations

## Performance

Run the benchmarks:

```bash
cargo bench
```

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
