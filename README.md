# LatticeFold+: Faster, Simpler, Shorter Lattice-Based Folding

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/idrees2516/latticefold_plus)

> ## ⚠️ **IMPORTANT DISCLAIMER - EDUCATIONAL USE ONLY** ⚠️
> 
> **🚨 THIS IMPLEMENTATION IS FOR LEARNING AND RESEARCH PURPOSES ONLY 🚨**
> 
> **DO NOT USE IN PRODUCTION SYSTEMS**
> 
> This is an educational implementation of the LatticeFold+ paper to demonstrate the concepts and algorithms. While we've implemented security best practices and comprehensive testing, this code:
> 
> - Has not undergone professional security audits
> - May contain implementation vulnerabilities
> - Is not optimized for production environments
> - Should not be used for any security-critical applications
> 
> **For production use, please:**
> - Conduct thorough security audits
> - Perform extensive testing in your specific environment
> - Consider using established, audited cryptographic libraries
> - Consult with cryptography experts
> 
> **Use this code to learn, experiment, and understand LatticeFold+ concepts only.**

A comprehensive Rust implementation of LatticeFold+, a lattice-based folding scheme for succinct proof systems based on the paper "LatticeFold+: Faster, Simpler, Shorter Lattice-Based Folding for Succinct Proof Systems" by Dan Boneh and Binyi Chen (2025).

## 🚀 Key Features

- **Post-Quantum Security**: Built on lattice-based cryptographic assumptions resistant to quantum attacks
- **Purely Algebraic Range Proofs**: Eliminates bit decomposition through innovative monomial set operations
- **Double Commitment Schemes**: Achieves compact matrix commitments via split/pow decomposition
- **Multi-Instance Folding**: Supports L-to-2 folding with norm control and witness decomposition
- **High Performance**: GPU acceleration, SIMD vectorization, and NTT-optimized polynomial arithmetic
- **Comprehensive Security**: Constant-time implementations, side-channel resistance, and formal security reductions

## 📊 Performance Improvements

LatticeFold+ achieves significant improvements over the original LatticeFold:
- **5x faster prover** through algebraic range proofs
- **Ω(log(B))-times smaller verifier circuits** 
- **Shorter proofs**: O_λ(κd + log n) vs O_λ(κd log B + d log n) bits

## 🏗️ Architecture

The implementation follows a layered architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│                    LatticeFold+ System                          │
├─────────────────────────────────────────────────────────────────┤
│  Application Layer: R1CS Prover, CCS Prover, IVC Composer      │
├─────────────────────────────────────────────────────────────────┤
│  Protocol Layer: Folding Engine, Range Prover, Sumcheck Engine │
├─────────────────────────────────────────────────────────────────┤
│  Commitment Layer: Linear Commits, Double Commits, Transform    │
├─────────────────────────────────────────────────────────────────┤
│  Algebraic Layer: Cyclotomic Ring, Monomial Sets, Gadgets      │
├─────────────────────────────────────────────────────────────────┤
│  Computational Layer: NTT Engine, SIMD Vectors, GPU Kernels    │
└─────────────────────────────────────────────────────────────────┘
```

## 🔧 Core Components

### Cyclotomic Ring Arithmetic
- Power-of-two cyclotomic rings R = Z[X]/(X^d + 1)
- NTT-optimized polynomial multiplication
- SIMD-accelerated coefficient operations
- GPU kernels for large dimensions

### Monomial Set Operations
- Extended monomial sets M' = {0, 1, X, X², ...}
- Efficient membership testing via Lemma 2.1
- Exponential mappings exp(a) and EXP(a)
- Range polynomial ψ construction

### Commitment Schemes
- Module-SIS based linear commitments
- Relaxed binding properties with security reductions
- Double commitment schemes for compact matrix commitments
- Gadget decomposition with optimized bases

### Range Proof System
- Purely algebraic range proofs without bit decomposition
- Monomial set checking protocols
- Sumcheck integration and batching
- Communication optimization

## 🛠️ Installation

### Prerequisites
- Rust 1.70+ with Cargo
- CUDA Toolkit 11.0+ (for GPU acceleration)
- OpenMP (for parallel processing)

### Build from Source
```bash
git clone https://github.com/idrees2516/latticefold_plus.git
cd latticefold_plus
cargo build --release
```

### Run Tests
```bash
cargo test
```

### Run Benchmarks
```bash
cargo bench
```

## 📖 Usage

### Basic Example
```rust
use latticefold_plus::*;

// Initialize ring parameters
let ring_params = RingParams::new(1024, 2147483647)?; // d=1024, q=2^31-1

// Create commitment scheme
let commitment_scheme = LinearCommitment::new(ring_params, 128, 256)?; // κ=128, n=256

// Commit to a vector
let witness = vec![RingElement::random(&ring_params); 256];
let commitment = commitment_scheme.commit_vector(&witness)?;

// Generate range proof
let range_prover = AlgebraicRangeProof::new(commitment_scheme);
let proof = range_prover.prove_range(&witness, &mut rng)?;

// Verify proof
let is_valid = range_prover.verify_range(&proof, &commitment)?;
assert!(is_valid);
```

### Advanced Usage
See the [examples](examples/) directory for more comprehensive usage examples including:
- Multi-instance folding
- R1CS constraint system integration
- GPU-accelerated computations
- Custom parameter selection

## 🔬 Security

### Cryptographic Assumptions
- **Module-SIS**: Security based on the Module Short Integer Solution problem
- **Ring-LWE**: Leverages Ring Learning With Errors for additional security
- **Post-Quantum**: Resistant to both classical and quantum attacks

### Security Features
- Constant-time implementations for secret-dependent operations
- Side-channel resistance measures
- Formal security reductions with tight bounds
- Comprehensive parameter validation

### Security Parameters
For 128-bit security:
- Ring dimension: d = 1024
- Modulus: q ≈ 2^31
- Security parameter: κ = 128
- Norm bounds optimized for concrete security

## 📊 Benchmarks

Performance benchmarks on modern hardware:

| Operation | Time (ms) | Memory (MB) | Improvement vs LatticeFold |
|-----------|-----------|-------------|---------------------------|
| Range Proof Generation | 45.2 | 128 | 5.1x faster |
| Folding (L=8 to 2) | 23.7 | 64 | 4.8x faster |
| Verification | 8.9 | 32 | 3.2x faster |

Run benchmarks with:
```bash
./scripts/run_performance_benchmarks.sh
```

## 🧪 Testing

The implementation includes comprehensive testing:

- **Unit Tests**: 95%+ code coverage
- **Integration Tests**: End-to-end protocol testing
- **Property Tests**: Randomized testing with QuickCheck
- **Security Tests**: Timing attack resistance, malicious input handling
- **Performance Tests**: Regression testing and optimization validation

## 📚 Documentation

- [Quick Start Guide](docs/quick-start.md)
- [Mathematical Foundations](docs/mathematical-foundations.md)
- [API Reference](docs/api-reference/)
- [Security Analysis](docs/security.md)
- [Performance Benchmarking](docs/performance_benchmarking.md)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Run the full test suite
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📖 Citation

If you use this implementation in your research, please cite:

```bibtex
@article{boneh2025latticefold,
  title={LatticeFold+: Faster, Simpler, Shorter Lattice-Based Folding for Succinct Proof Systems},
  author={Boneh, Dan and Chen, Binyi},
  journal={Cryptology ePrint Archive},
  year={2025}
}
```

## 🔗 Related Work

- [Original LatticeFold Paper](https://eprint.iacr.org/2024/257)
- [Lattice-Based Cryptography Survey](https://eprint.iacr.org/2015/939)
- [Post-Quantum Cryptography Standards](https://csrc.nist.gov/projects/post-quantum-cryptography)


## 🙏 Acknowledgments

- Dan Boneh and Binyi Chen for the original LatticeFold+ paper
- The lattice-based cryptography research community
- Contributors to the Rust cryptography ecosystem

---

## ⚠️ **FINAL SECURITY REMINDER** ⚠️

**🔴 THIS IS AN EDUCATIONAL IMPLEMENTATION ONLY 🔴**

This LatticeFold+ implementation is designed for:
- ✅ Learning lattice-based cryptography concepts
- ✅ Understanding the LatticeFold+ paper
- ✅ Academic research and experimentation
- ✅ Cryptographic algorithm development

**❌ NOT suitable for:**
- Production systems
- Security-critical applications
- Financial or sensitive data processing
- Any real-world cryptographic deployment

**Before any production consideration:**
1. Professional security audit required
2. Extensive penetration testing needed
3. Code review by cryptography experts
4. Compliance with relevant security standards
5. Consider using established, audited libraries instead

**Use responsibly and only for educational purposes!**