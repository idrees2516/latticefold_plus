/// Test modules for LatticeFold+ implementation
/// 
/// This module organizes all test suites for the LatticeFold+ implementation,
/// including unit tests, integration tests, and performance benchmarks.
/// 
/// Test Organization:
/// - Unit tests for individual components and functions
/// - Integration tests for complete protocol execution
/// - Performance tests and benchmarks
/// - GPU-specific tests when GPU features are enabled
/// - Cross-platform compatibility tests
/// - Security and correctness validation tests

pub mod ntt_multiplication_tests;

#[cfg(feature = "gpu")]
pub mod gpu_ntt_tests;

// Re-export test utilities for use in other test modules
pub use gpu_ntt_tests::*;