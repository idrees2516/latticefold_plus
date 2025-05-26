# LatticeFold+

Implementation of LatticeFold+: Faster, Simpler, Shorter Lattice-Based Folding for Succinct Proof Systems.

## Overview

LatticeFold+ is a novel lattice-based folding scheme that enables efficient zero-knowledge proofs for lattice-based cryptography. This implementation provides:

- Lattice-based folding operations
- Zero-knowledge proof generation and verification
- Commitment schemes for vectors and polynomials
- Efficient proof composition and verification

## Features

- **Lattice Operations**: Efficient lattice basis operations and point arithmetic
- **Folding Protocol**: Implementation of the LatticeFold+ folding protocol
- **Zero-Knowledge Proofs**: Zero-knowledge proof generation and verification
- **Commitment Schemes**: Vector and polynomial commitment schemes
- **Performance**: Optimized implementation with parallel processing support

## Dependencies

- ark-ec: Elliptic curve operations
- ark-ff: Finite field operations
- ark-poly: Polynomial operations
- ark-std: Standard library extensions
- ark-bls12-381: BLS12-381 curve implementation
- merlin: Transcript for Fiat-Shamir
- rand: Random number generation
- zeroize: Secure memory zeroing

## Usage

```rust
use latticefold_plus::*;
use ark_bls12_381::{Bls12_381, Fr};
use ark_std::test_rng;

fn main() {
    let mut rng = test_rng();
    let dimension = 10;
    let security_param = 128;

    // Setup
    let params = setup_lattice_fold::<Bls12_381, Fr, _>(dimension, security_param, &mut rng);

    // Generate instances
    let mut instances = Vec::new();
    for _ in 0..3 {
        let witness = vec![Fr::rand(&mut rng); dimension];
        let public_input = vec![Fr::rand(&mut rng); dimension];
        instances.push(LatticeFoldInstance {
            witness,
            public_input,
        });
    }

    // Prove
    let proof = prove_lattice_fold(&params, &instances, &mut rng);

    // Verify
    assert!(verify_lattice_fold(&params, &proof));
}
```

## Building

```bash
cargo build --release
```

## Testing

```bash
cargo test
```

## Benchmarking

```bash
cargo bench
```

## Security

This implementation is for research purposes only and has not been audited for production use. Use at your own risk.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Based on the paper "LatticeFold+: Faster, Simpler, Shorter Lattice-Based Folding for Succinct Proof Systems"
- Uses the Arkworks ecosystem for cryptographic primitives 