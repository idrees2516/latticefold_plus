# LatticeFold+ API Reference

This comprehensive API reference provides detailed documentation for all public interfaces in the LatticeFold+ library. Each module is documented with mathematical foundations, usage examples, performance characteristics, and security considerations.

## Table of Contents

### Core Mathematical Components
- [Cyclotomic Ring Arithmetic](./cyclotomic-ring.md) - Ring operations and polynomial arithmetic
- [Modular Arithmetic](./modular-arithmetic.md) - Barrett and Montgomery reduction implementations
- [NTT Engine](./ntt.md) - Number Theoretic Transform for fast polynomial multiplication
- [Norm Computation](./norm-computation.md) - Infinity, Euclidean, and operator norm calculations

### Algebraic Structures
- [Monomial Operations](./monomial.md) - Monomial sets and membership testing
- [Gadget Matrices](./gadget.md) - Gadget decomposition and reconstruction
- [Lattice Structures](./lattice.md) - Lattice basis and point operations

### Commitment Schemes
- [Linear Commitments](./commitment.md) - SIS-based linear commitment schemes
- [Double Commitments](./double-commitment.md) - Compact matrix commitments
- [Commitment Transformation](./commitment-transformation.md) - Protocol for transforming commitments

### Proof Systems
- [Range Check Protocol](./range-check.md) - Algebraic range proofs without bit decomposition
- [Sumcheck Protocol](./sumcheck.md) - Ring-based sumcheck with batching
- [Multi-Instance Folding](./multi-instance-folding.md) - L-to-2 folding with norm control
- [End-to-End Folding](./end-to-end-folding.md) - Complete folding system integration

### Constraint Systems
- [R1CS Integration](./r1cs.md) - Rank-1 Constraint System support
- [Protocol Interface](./protocol.md) - High-level protocol operations

### Security and Performance
- [Security Analysis](./security-analysis.md) - Formal security verification tools
- [Parameter Generation](./parameter-generation.md) - Automated parameter selection
- [Quantum Resistance](./quantum-resistance.md) - Post-quantum security analysis
- [Side Channel Protection](./side-channel.md) - Constant-time implementations

### GPU Acceleration
- [GPU Operations](./gpu.md) - CUDA and OpenCL implementations
- [SIMD Vectorization](./simd.md) - CPU vectorization optimizations
- [Memory Management](./memory.md) - Efficient memory allocation strategies

### Error Handling and Types
- [Error Types](./error.md) - Comprehensive error handling system
- [Core Types](./types.md) - Fundamental data structures and type definitions

## API Design Principles

The LatticeFold+ API follows these key design principles:

### 1. Mathematical Correctness
Every operation is mathematically sound and implements the exact algorithms described in the LatticeFold+ paper. All implementations include:
- Formal mathematical specifications
- Correctness proofs and invariants
- Comprehensive test coverage with property-based testing
- Cross-validation against reference implementations

### 2. Performance-First Design
All components are optimized for maximum performance:
- GPU acceleration for computationally intensive operations
- SIMD vectorization for CPU operations
- Cache-optimized memory layouts
- Parallel processing where mathematically feasible
- Automatic algorithm selection based on input size

### 3. Security by Design
Security is built into every component:
- Constant-time implementations for secret-dependent operations
- Side-channel resistance measures
- Formal security reductions from well-established assumptions
- Comprehensive parameter validation
- Secure random number generation

### 4. Composability and Modularity
The API is designed for easy composition:
- Clean separation of concerns between layers
- Minimal dependencies between components
- Consistent error handling across all modules
- Uniform naming conventions and patterns
- Extensive trait-based abstractions

### 5. Developer Experience
The API prioritizes ease of use:
- Comprehensive documentation with examples
- Clear error messages with actionable guidance
- Sensible defaults for common use cases
- Progressive disclosure of complexity
- Extensive debugging and profiling support

## Common Usage Patterns

### Basic Setup Pattern
```rust
use latticefold_plus::{setup_lattice_fold, SecurityLevel};
use ark_bls12_381::{Bls12_381, Fr};
use ark_std::test_rng;

// Standard setup for most applications
let mut rng = test_rng();
let params = setup_lattice_fold::<Bls12_381, Fr, _>(
    SecurityLevel::High, 
    &mut rng
)?;
```

### Error Handling Pattern
```rust
use latticefold_plus::{LatticeFoldError, Result};

match operation() {
    Ok(result) => process_result(result),
    Err(LatticeFoldError::InvalidDimension { expected, got }) => {
        // Handle dimension mismatch
        eprintln!("Expected dimension {}, got {}", expected, got);
    }
    Err(e) => {
        // Handle other errors
        eprintln!("Operation failed: {}", e);
    }
}
```

### Performance Monitoring Pattern
```rust
use std::time::Instant;

let start = Instant::now();
let result = expensive_operation()?;
let duration = start.elapsed();

println!("Operation completed in {:?}", duration);
if duration.as_millis() > threshold {
    println!("Warning: Operation exceeded expected time");
}
```

### GPU Acceleration Pattern
```rust
use latticefold_plus::gpu::{GPUEngine, DeviceType};

// Automatic GPU detection and fallback
let gpu_engine = GPUEngine::new(DeviceType::Auto)?;
let result = if gpu_engine.is_available() {
    gpu_engine.compute_intensive_operation(data)?
} else {
    cpu_fallback_operation(data)?
};
```

## Performance Characteristics

### Complexity Analysis

| Operation | CPU Complexity | GPU Complexity | Memory Usage |
|-----------|---------------|----------------|--------------|
| Ring Addition | O(d) | O(d/p) | O(d) |
| Ring Multiplication (NTT) | O(d log d) | O(d log d / p) | O(d) |
| Matrix Commitment | O(κnd) | O(κnd / p) | O(κnd) |
| Range Proof Generation | O(nκd) | O(nκd / p) | O(nκd) |
| Sumcheck Verification | O(kd) | O(kd / p) | O(kd) |
| Multi-Instance Folding | O(Lnκd) | O(Lnκd / p) | O(Lnκd) |

Where:
- `d` = ring dimension
- `κ` = security parameter
- `n` = vector/matrix dimension
- `k` = sumcheck rounds
- `L` = number of instances to fold
- `p` = number of parallel processors

### Memory Requirements

| Component | Memory Usage | GPU Memory | Notes |
|-----------|--------------|------------|-------|
| Ring Element | 8d bytes | 8d bytes | Coefficient storage |
| Commitment Matrix | 8κnd bytes | 8κnd bytes | Dense matrix storage |
| NTT Parameters | 16d bytes | 16d bytes | Twiddle factors + bit-reversal |
| Proof Data | O(κd + log n) bytes | - | Compressed proof format |

### Recommended Parameters

| Security Level | Ring Dimension | Modulus Size | Memory Usage | Performance |
|----------------|----------------|--------------|--------------|-------------|
| Low (80-bit) | 512 | 32-bit | ~2 MB | ~10ms proof |
| Medium (128-bit) | 1024 | 64-bit | ~8 MB | ~50ms proof |
| High (256-bit) | 2048 | 64-bit | ~32 MB | ~200ms proof |

## Thread Safety

All LatticeFold+ components are designed with thread safety in mind:

- **Immutable Operations**: Most mathematical operations are pure functions that don't modify their inputs
- **Interior Mutability**: Components that require mutable state use appropriate synchronization primitives
- **Send + Sync**: All public types implement `Send` and `Sync` where mathematically sound
- **Lock-Free Algorithms**: Performance-critical paths use lock-free data structures where possible

## Memory Safety

The library provides strong memory safety guarantees:

- **No Unsafe Code**: The public API contains no unsafe code blocks
- **Bounds Checking**: All array and vector accesses are bounds-checked
- **Overflow Protection**: Arithmetic operations include overflow detection
- **Secure Memory**: Sensitive data is zeroized when dropped
- **Memory Pools**: Large allocations use memory pools to prevent fragmentation

## Versioning and Compatibility

LatticeFold+ follows semantic versioning:

- **Major Version**: Breaking API changes
- **Minor Version**: New features, backward compatible
- **Patch Version**: Bug fixes, performance improvements

### API Stability Guarantees

- **Public API**: Stable across minor versions
- **Mathematical Correctness**: Never changes within major versions
- **Performance**: May improve, never intentionally degrades
- **Security**: Parameters may be updated for new attack discoveries

## Getting Help

If you need assistance with the API:

1. **Documentation**: Start with this API reference and the examples
2. **Error Messages**: Read error messages carefully - they include actionable guidance
3. **Performance Issues**: Consult the performance guide and profiling tools
4. **Security Questions**: Review the security analysis documentation
5. **Bug Reports**: Include minimal reproduction cases and system information

## Contributing to the API

When contributing new API components:

1. **Follow Design Principles**: Maintain consistency with existing patterns
2. **Mathematical Rigor**: Include formal specifications and proofs
3. **Performance Testing**: Benchmark against existing implementations
4. **Security Review**: Ensure constant-time implementations where required
5. **Documentation**: Provide comprehensive documentation with examples
6. **Testing**: Include unit tests, integration tests, and property-based tests

This API reference provides the foundation for understanding and effectively using the LatticeFold+ library. Each linked section provides detailed information about specific components and their usage patterns.