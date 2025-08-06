# LatticeFold+ API Documentation

Welcome to the comprehensive API documentation for LatticeFold+, a lattice-based folding scheme for succinct proof systems.

## Table of Contents

1. [Quick Start Guide](./quick-start.md)
2. [API Reference](./api-reference/README.md)
3. [Mathematical Foundations](./mathematical-foundations.md)
4. [Examples and Tutorials](./examples/README.md)
5. [Error Handling Guide](./error-handling.md)
6. [Performance Guide](./performance.md)
7. [Security Considerations](./security.md)
8. [GPU Acceleration](./gpu-acceleration.md)
9. [Troubleshooting](./troubleshooting.md)
10. [API Design Guidelines](./api-design-guidelines.md)

## Overview

LatticeFold+ is a revolutionary advancement in post-quantum succinct proof systems, introducing several key innovations:

- **Purely Algebraic Range Proofs**: Eliminates bit decomposition through monomial set operations
- **Double Commitment Schemes**: Achieves compact matrix commitments through split/pow decomposition  
- **Commitment Transformation Protocols**: Enables folding of non-homomorphic commitments
- **Multi-Instance Folding**: Supports L-to-2 folding with norm control
- **Ring-Based Sumcheck**: Optimized sumcheck protocols over cyclotomic rings

## Performance Claims

LatticeFold+ achieves significant improvements over previous approaches:
- **5x faster prover** compared to LatticeFold
- **Ω(log(B))-times smaller verifier circuits**
- **O_λ(κd + log n) vs O_λ(κd log B + d log n) bit proof sizes**

## Architecture

The system follows a layered architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Application Layer                            │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   R1CS Prover   │  │   CCS Prover    │  │  IVC Composer   │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│                    Protocol Layer                               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ Folding Engine  │  │ Range Prover    │  │ Sumcheck Engine │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│                   Commitment Layer                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ Linear Commits  │  │ Double Commits  │  │ Transform Proto │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│                   Algebraic Layer                               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ Cyclotomic Ring │  │ Monomial Sets   │  │ Gadget Matrices │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│                 Computational Layer                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   NTT Engine    │  │  SIMD Vectors   │  │  GPU Kernels    │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Getting Started

For a quick introduction to using LatticeFold+, see our [Quick Start Guide](./quick-start.md).

For detailed API documentation, browse the [API Reference](./api-reference/README.md).

For working examples, check out our [Examples and Tutorials](./examples/README.md).

## Support

If you encounter issues or need help:

1. Check the [Troubleshooting Guide](./troubleshooting.md)
2. Review the [Error Handling Guide](./error-handling.md)
3. Consult the [Performance Guide](./performance.md) for optimization tips
4. Review the [Security Considerations](./security.md) for deployment guidance

## Contributing

When extending the API, please follow our [API Design Guidelines](./api-design-guidelines.md) to maintain consistency and usability.