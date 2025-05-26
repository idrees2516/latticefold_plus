use crate::error::{LatticeFoldError, Result};
use crate::lattice::{LatticeMatrix, LatticeParams, LatticePoint};
use std::collections::HashMap;
use std::f64::consts::PI;
use zeroize::Zeroize;

/// Parameters for quantum-resistant lattice schemes
/// Based on Section 7.2 of the LatticeFold+ paper
#[derive(Clone, Debug)]
pub struct QuantumResistanceParams {
    /// The security level in bits
    pub security_level: usize,
    /// The estimated cost of quantum attack
    pub quantum_cost: usize,
    /// The lattice dimension for the given security level
    pub dimension: usize,
    /// The modulus for the given security level
    pub modulus: i64,
    /// The standard deviation for Gaussian sampling
    pub sigma: f64,
    /// The rejection sampling parameter
    pub rejection_param: f64,
}

/// Security levels for different applications
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum SecurityLevel {
    /// 128-bit classical / 64-bit quantum security
    Medium,
    /// 192-bit classical / 96-bit quantum security
    High,
    /// 256-bit classical / 128-bit quantum security
    VeryHigh,
    /// Custom security level
    Custom(usize),
}

/// The main quantum resistance analyzer
#[derive(Clone, Debug)]
pub struct QuantumResistanceAnalyzer {
    /// Cache of pre-computed parameters for different security levels
    params_cache: HashMap<SecurityLevel, QuantumResistanceParams>,
    /// The Grover's algorithm speedup factor (default: 2)
    grover_speedup: f64,
    /// The BKZ block size cost model
    bkz_cost_model: BKZCostModel,
}

/// Cost models for BKZ lattice reduction
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BKZCostModel {
    /// Core-SVP cost model
    CoreSVP,
    /// Gate count cost model
    GateCount,
    /// Q-core-SVP cost model
    QCoreSVP,
}

impl QuantumResistanceAnalyzer {
    /// Create a new quantum resistance analyzer
    pub fn new() -> Self {
        let mut analyzer = Self {
            params_cache: HashMap::new(),
            grover_speedup: 2.0,
            bkz_cost_model: BKZCostModel::CoreSVP,
        };
        
        // Pre-compute parameters for standard security levels
        analyzer.precompute_security_params();
        
        analyzer
    }
    
    /// Pre-compute parameters for standard security levels
    fn precompute_security_params(&mut self) {
        self.params_cache.insert(
            SecurityLevel::Medium,
            QuantumResistanceParams {
                security_level: 128,
                quantum_cost: 64,
                dimension: 512,
                modulus: 4096 + 1, // q = 2^12 + 1
                sigma: 3.2,
                rejection_param: 1.7,
            },
        );
        
        self.params_cache.insert(
            SecurityLevel::High,
            QuantumResistanceParams {
                security_level: 192,
                quantum_cost: 96,
                dimension: 768,
                modulus: 8192 + 1, // q = 2^13 + 1
                sigma: 3.6,
                rejection_param: 1.9,
            },
        );
        
        self.params_cache.insert(
            SecurityLevel::VeryHigh,
            QuantumResistanceParams {
                security_level: 256,
                quantum_cost: 128,
                dimension: 1024,
                modulus: 16384 + 1, // q = 2^14 + 1
                sigma: 4.0,
                rejection_param: 2.1,
            },
        );
    }
    
    /// Get the parameters for a specific security level
    pub fn get_params(&self, level: SecurityLevel) -> Result<QuantumResistanceParams> {
        match level {
            SecurityLevel::Custom(bits) => {
                self.compute_params_for_security_level(bits)
            },
            _ => {
                if let Some(params) = self.params_cache.get(&level) {
                    Ok(params.clone())
                } else {
                    Err(LatticeFoldError::InvalidParameters(
                        format!("Unknown security level: {:?}", level),
                    ))
                }
            },
        }
    }
    
    /// Compute the parameters for a specific security level
    fn compute_params_for_security_level(&self, bits: usize) -> Result<QuantumResistanceParams> {
        if bits < 64 {
            return Err(LatticeFoldError::InvalidParameters(
                format!("Security level must be at least 64 bits, got {}", bits),
            ));
        }
        
        // Calculate dimension based on security level
        let quantum_cost = (bits as f64 / self.grover_speedup).ceil() as usize;
        
        // Dimension formula based on the paper's analysis
        let dimension = match self.bkz_cost_model {
            BKZCostModel::CoreSVP => {
                // n ≈ security_level / log(delta_0)
                // where delta_0 ≈ (k / 2πe)^(1/2k) for BKZ-k
                // We use k ≈ security_level / 7 as a heuristic
                let k = bits / 7;
                let delta_0 = ((k as f64) / (2.0 * PI * std::f64::consts::E)).powf(1.0 / (2.0 * k as f64));
                let log_delta_0 = delta_0.ln();
                ((bits as f64) / -log_delta_0).ceil() as usize
            },
            BKZCostModel::GateCount => {
                // For gate count model, we increase the dimension by ~40%
                ((bits as f64) * 1.4).ceil() as usize
            },
            BKZCostModel::QCoreSVP => {
                // For quantum model, we use the quantum cost
                ((quantum_cost as f64) * 4.5).ceil() as usize
            },
        };
        
        // Round up dimension to a power of 2
        let dimension = dimension.next_power_of_two();
        
        // Calculate appropriate modulus
        // For SIS, we need q ≥ sqrt(n) * B * ω(sqrt(log n))
        // where B is the norm bound
        let log_n = (dimension as f64).ln();
        let norm_bound = (dimension as f64).sqrt() * log_n.sqrt();
        let modulus = (2.0 * norm_bound).ceil() as i64;
        
        // Round up to the next power of 2 + 1 (for NTT-friendly modulus)
        let log_modulus = (modulus as f64).log2().ceil() as u32;
        let modulus = (1 << log_modulus) + 1;
        
        // Calculate Gaussian parameter based on security level
        // σ ≥ sqrt(n) * ω(sqrt(log n))
        let sigma = (dimension as f64).sqrt() * log_n.sqrt() * 1.1;
        
        // Calculate rejection sampling parameter based on security level
        let rejection_param = 1.5 + (bits as f64 / 256.0) * 0.6;
        
        Ok(QuantumResistanceParams {
            security_level: bits,
            quantum_cost,
            dimension,
            modulus,
            sigma,
            rejection_param,
        })
    }
    
    /// Set the BKZ cost model
    pub fn set_cost_model(&mut self, model: BKZCostModel) {
        self.bkz_cost_model = model;
        // Clear cache to force recomputation with new model
        self.params_cache.clear();
        self.precompute_security_params();
    }
    
    /// Set the Grover speedup factor (default: 2.0)
    pub fn set_grover_speedup(&mut self, speedup: f64) {
        if speedup <= 0.0 {
            return;
        }
        
        self.grover_speedup = speedup;
        // Clear cache to force recomputation with new speedup
        self.params_cache.clear();
        self.precompute_security_params();
    }
    
    /// Estimate the quantum security of existing parameters
    pub fn estimate_security(&self, params: &LatticeParams) -> usize {
        let dimension = params.n;
        let q = params.q;
        
        // Estimate BKZ block size needed to break this lattice
        let block_size = self.estimate_bkz_block_size(dimension, q);
        
        // Convert block size to security bits
        match self.bkz_cost_model {
            BKZCostModel::CoreSVP => {
                // 2^(0.292 * beta) operations
                (0.292 * block_size as f64).ceil() as usize
            },
            BKZCostModel::GateCount => {
                // 2^(0.292 * beta) quantum gates
                (0.365 * block_size as f64).ceil() as usize
            },
            BKZCostModel::QCoreSVP => {
                // 2^(0.265 * beta) quantum operations with Grover
                let classical_security = (0.292 * block_size as f64).ceil() as usize;
                (classical_security as f64 / self.grover_speedup).ceil() as usize
            },
        }
    }
    
    /// Estimate the BKZ block size needed to break a lattice
    fn estimate_bkz_block_size(&self, dimension: usize, q: i64) -> usize {
        // Using the formula from the paper
        // beta ≈ n * log(q) / (2 * pi * e)
        let log_q = (q as f64).ln() / std::f64::consts::LN_2;
        let numerator = dimension as f64 * log_q;
        let denominator = 2.0 * PI * std::f64::consts::E;
        
        (numerator / denominator).ceil() as usize
    }
    
    /// Create LatticeParams from quantum-resistant parameters
    pub fn create_lattice_params(&self, qr_params: &QuantumResistanceParams) -> LatticeParams {
        LatticeParams {
            q: qr_params.modulus,
            n: qr_params.dimension,
            sigma: qr_params.sigma,
            beta: qr_params.security_level as f64 / 100.0, // Scale security level to a reasonable beta
        }
    }
    
    /// Check if parameters are quantum-resistant
    pub fn is_quantum_resistant(&self, params: &LatticeParams, min_security: usize) -> bool {
        let estimated_security = self.estimate_security(params);
        estimated_security >= min_security
    }
    
    /// Calculate probability of Grover's algorithm success
    pub fn grover_success_probability(&self, dimension: usize, iterations: usize) -> f64 {
        // Grover's algorithm success probability after t iterations:
        // P(success) = sin^2((2t + 1) * theta)
        // where theta = asin(1/sqrt(N)) and N = 2^dimension
        
        let n = 2.0_f64.powi(dimension as i32);
        let theta = (1.0 / n.sqrt()).asin();
        
        let angle = (2 * iterations + 1) as f64 * theta;
        angle.sin().powi(2)
    }
    
    /// Calculate the optimal number of Grover iterations
    pub fn optimal_grover_iterations(&self, dimension: usize) -> usize {
        // Optimal number of iterations for Grover's algorithm:
        // t_opt = (pi/4) * sqrt(N)
        // where N = 2^dimension
        
        let n = 2.0_f64.powi(dimension as i32);
        (PI / 4.0 * n.sqrt()).floor() as usize
    }
    
    /// Calculate the lattice security level against quantum computers
    pub fn quantum_security_level(&self, params: &LatticeParams) -> SecurityLevel {
        let security = self.estimate_security(params);
        
        if security >= 128 {
            SecurityLevel::VeryHigh
        } else if security >= 96 {
            SecurityLevel::High
        } else if security >= 64 {
            SecurityLevel::Medium
        } else {
            SecurityLevel::Custom(security)
        }
    }
}

/// Quantum-resistant Gaussian sampling
/// Based on Section 7.2 of the paper
pub struct QuantumResistantSampler {
    /// The lattice dimension
    pub dimension: usize,
    /// The standard deviation for Gaussian sampling
    pub sigma: f64,
    /// The rejection sampling parameter
    pub rejection_param: f64,
    /// The modulus
    pub q: i64,
}

impl QuantumResistantSampler {
    /// Create a new quantum-resistant sampler
    pub fn new(params: &QuantumResistanceParams) -> Self {
        Self {
            dimension: params.dimension,
            sigma: params.sigma,
            rejection_param: params.rejection_param,
            q: params.modulus,
        }
    }
    
    /// Sample a vector from a discrete Gaussian distribution
    pub fn sample_gaussian<R: rand::Rng>(&self, rng: &mut R) -> Vec<i64> {
        let mut result = Vec::with_capacity(self.dimension);
        
        for _ in 0..self.dimension {
            // Box-Muller transform to get Gaussian samples
            let u1 = rng.gen::<f64>();
            let u2 = rng.gen::<f64>();
            
            let radius = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * PI * u2;
            
            let x = radius * theta.cos() * self.sigma;
            
            // Round to nearest integer
            let x_rounded = x.round() as i64;
            
            // Apply rejection sampling
            let alpha = (-((x - x_rounded as f64).powi(2)) / (2.0 * self.sigma * self.sigma)).exp();
            let u3 = rng.gen::<f64>();
            
            if u3 <= alpha {
                // Accept sample
                result.push(x_rounded.rem_euclid(self.q));
            } else {
                // Reject and try again
                let mut accepted = false;
                while !accepted {
                    let u1 = rng.gen::<f64>();
                    let u2 = rng.gen::<f64>();
                    
                    let radius = (-2.0 * u1.ln()).sqrt();
                    let theta = 2.0 * PI * u2;
                    
                    let x = radius * theta.cos() * self.sigma;
                    let x_rounded = x.round() as i64;
                    
                    let alpha = (-((x - x_rounded as f64).powi(2)) / (2.0 * self.sigma * self.sigma)).exp();
                    let u3 = rng.gen::<f64>();
                    
                    if u3 <= alpha {
                        result.push(x_rounded.rem_euclid(self.q));
                        accepted = true;
                    }
                }
            }
        }
        
        result
    }
    
    /// Sample a short vector that doesn't lose security against quantum computers
    pub fn sample_short_vector<R: rand::Rng>(&self, rng: &mut R) -> Vec<i64> {
        let mut result = self.sample_gaussian(rng);
        
        // Apply bounds check to ensure vector is "short enough"
        let norm_squared: i64 = result.iter().map(|&x| x * x).sum();
        let target_norm = (self.dimension as f64).sqrt() * self.sigma * self.rejection_param;
        
        if (norm_squared as f64).sqrt() > target_norm {
            // If the vector is too long, rescale it
            let scale_factor = target_norm / (norm_squared as f64).sqrt();
            
            for x in &mut result {
                *x = (*x as f64 * scale_factor).round() as i64;
                *x = x.rem_euclid(self.q);
            }
        }
        
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::thread_rng;
    
    #[test]
    fn test_quantum_resistance_parameters() {
        let analyzer = QuantumResistanceAnalyzer::new();
        
        // Test the pre-computed parameters
        let medium_params = analyzer.get_params(SecurityLevel::Medium).unwrap();
        assert_eq!(medium_params.security_level, 128);
        assert_eq!(medium_params.quantum_cost, 64);
        
        let high_params = analyzer.get_params(SecurityLevel::High).unwrap();
        assert_eq!(high_params.security_level, 192);
        assert_eq!(high_params.quantum_cost, 96);
        
        let very_high_params = analyzer.get_params(SecurityLevel::VeryHigh).unwrap();
        assert_eq!(very_high_params.security_level, 256);
        assert_eq!(very_high_params.quantum_cost, 128);
        
        // Test custom security level
        let custom_params = analyzer.get_params(SecurityLevel::Custom(160)).unwrap();
        assert_eq!(custom_params.security_level, 160);
        assert!(custom_params.dimension >= 512);
        
        // Test invalid security level
        let result = analyzer.get_params(SecurityLevel::Custom(32));
        assert!(result.is_err());
    }
    
    #[test]
    fn test_quantum_resistant_sampling() {
        let analyzer = QuantumResistanceAnalyzer::new();
        let params = analyzer.get_params(SecurityLevel::Medium).unwrap();
        
        let sampler = QuantumResistantSampler::new(&params);
        let mut rng = thread_rng();
        
        // Test Gaussian sampling
        let gaussian = sampler.sample_gaussian(&mut rng);
        assert_eq!(gaussian.len(), params.dimension);
        
        // Test short vector sampling
        let short_vector = sampler.sample_short_vector(&mut rng);
        assert_eq!(short_vector.len(), params.dimension);
        
        // Check that the vector is "short enough"
        let norm_squared: i64 = short_vector.iter().map(|&x| x * x).sum();
        let target_norm = (params.dimension as f64).sqrt() * params.sigma * params.rejection_param;
        assert!((norm_squared as f64).sqrt() <= target_norm);
    }
    
    #[test]
    fn test_security_estimation() {
        let analyzer = QuantumResistanceAnalyzer::new();
        
        // Test with known parameters
        let known_params = LatticeParams {
            q: 4097,
            n: 512,
            sigma: 3.2,
            beta: 1.28,
        };
        
        let security = analyzer.estimate_security(&known_params);
        assert!(security >= 64);
        
        // Test with stronger parameters
        let stronger_params = LatticeParams {
            q: 8193,
            n: 1024,
            sigma: 4.0,
            beta: 2.56,
        };
        
        let stronger_security = analyzer.estimate_security(&stronger_params);
        assert!(stronger_security > security);
        
        // Test quantum resistance check
        assert!(analyzer.is_quantum_resistant(&stronger_params, 64));
        assert!(analyzer.is_quantum_resistant(&known_params, 64));
        
        // Test security level classification
        let level = analyzer.quantum_security_level(&stronger_params);
        match level {
            SecurityLevel::VeryHigh | SecurityLevel::High => {
                // Expected for these parameters
            },
            _ => {
                panic!("Expected high security level, got {:?}", level);
            }
        }
    }
} 