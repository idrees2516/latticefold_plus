/// Basic test for NTT implementation
/// 
/// This test verifies that the NTT implementation works correctly by testing
/// the forward and inverse transforms, as well as polynomial multiplication.

use std::sync::Arc;

// Mock error types for testing
#[derive(Debug)]
pub enum LatticeFoldError {
    InvalidDimension { expected: usize, got: usize },
    InvalidModulus { modulus: i64 },
    InvalidParameters(String),
    GPUError(String),
}

pub type Result<T> = std::result::Result<T, LatticeFoldError>;

// Mock modular arithmetic for testing
#[derive(Clone, Debug)]
pub struct ModularArithmetic {
    modulus: i64,
}

impl ModularArithmetic {
    pub fn new(modulus: i64) -> Result<Self> {
        Ok(Self { modulus })
    }
    
    pub fn add_mod(&self, a: i64, b: i64) -> i64 {
        (a + b) % self.modulus
    }
    
    pub fn sub_mod(&self, a: i64, b: i64) -> i64 {
        (a - b + self.modulus) % self.modulus
    }
    
    pub fn mul_mod(&self, a: i64, b: i64) -> i64 {
        ((a as i128 * b as i128) % self.modulus as i128) as i64
    }
    
    pub fn neg_mod(&self, a: i64) -> i64 {
        (self.modulus - a) % self.modulus
    }
}

// Include the NTT implementation (simplified version for testing)
include!("src/ntt.rs");

fn main() {
    println!("Testing NTT implementation...");
    
    // Test NTT parameter generation
    test_ntt_params();
    
    // Test forward and inverse NTT
    test_forward_inverse_ntt();
    
    // Test polynomial multiplication
    test_polynomial_multiplication();
    
    println!("All tests passed!");
}

fn test_ntt_params() {
    println!("Testing NTT parameter generation...");
    
    // Test with small parameters
    let params = NTTParams::new(64, 193).unwrap(); // 193 ≡ 1 + 64 (mod 128)
    assert_eq!(params.dimension(), 64);
    assert_eq!(params.modulus(), 193);
    assert!(params.security_level() >= 80);
    
    println!("✓ NTT parameter generation works");
}

fn test_forward_inverse_ntt() {
    println!("Testing forward and inverse NTT...");
    
    let dimension = 64;
    let modulus = 193;
    
    let params = Arc::new(NTTParams::new(dimension, modulus).unwrap());
    let mut ntt_engine = NTTEngine::new(params);
    
    // Generate test polynomial
    let mut coefficients: Vec<i64> = (0..dimension)
        .map(|i| (i as i64 * 123 + 456) % modulus)
        .collect();
    let original_coefficients = coefficients.clone();
    
    // Apply forward NTT
    ntt_engine.forward_ntt(&mut coefficients).unwrap();
    
    // Apply inverse NTT
    ntt_engine.inverse_ntt(&mut coefficients).unwrap();
    
    // Verify we get back the original polynomial
    assert_eq!(coefficients, original_coefficients);
    
    println!("✓ Forward and inverse NTT work correctly");
}

fn test_polynomial_multiplication() {
    println!("Testing polynomial multiplication...");
    
    let dimension = 64;
    let modulus = 193;
    
    let params = Arc::new(NTTParams::new(dimension, modulus).unwrap());
    let mut multiplier = NTTMultiplier::new(params);
    
    // Test multiplication by zero
    let f = vec![1, 2, 3, 4, 5];
    let g = vec![0; dimension];
    let mut result = vec![0i64; dimension];
    
    multiplier.multiply(&f, &g, &mut result).unwrap();
    assert!(result.iter().all(|&x| x == 0));
    
    // Test multiplication by constant
    let f = vec![1, 2, 3, 4];
    let g = vec![5]; // Constant polynomial g(X) = 5
    let mut result = vec![0i64; dimension];
    
    multiplier.multiply(&f, &g, &mut result).unwrap();
    
    // Result should be [5, 10, 15, 20, 0, 0, ...]
    assert_eq!(result[0], 5);
    assert_eq!(result[1], 10);
    assert_eq!(result[2], 15);
    assert_eq!(result[3], 20);
    
    println!("✓ Polynomial multiplication works correctly");
}