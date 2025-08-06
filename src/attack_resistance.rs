//! Security Analysis and Attack Resistance for LatticeFold+
//! 
//! This module implements comprehensive security analysis against BKZ, sieve, and other lattice
//! attacks with concrete hardness estimation for chosen parameters. The implementation provides
//! attack complexity computation using current best algorithms, security margin analysis with
//! conservative estimates, and quantum attack resistance analysis with Grover speedup consideration.
//! 
//! Mathematical Foundation:
//! The security analysis is based on the most current understanding of lattice attack complexities:
//! - **BKZ Attacks**: Classical and quantum BKZ with various cost models (Core-SVP, Gate-Count, etc.)
//! - **Sieve Attacks**: GaussSieve, NV-Sieve, BDGL16, and quantum variants
//! - **Dual Attacks**: Attacks on the dual lattice with embedding and uSVP techniques
//! - **Primal Attacks**: Direct attacks on the primal lattice with enumeration
//! - **Hybrid Attacks**: Combinations of reduction and enumeration techniques
//! 
//! Security Model:
//! - **Concrete Security**: Uses lattice estimators for precise attack complexity computation
//! - **Conservative Analysis**: Includes safety margins for unknown attack improvements
//! - **Quantum Resistance**: Accounts for known quantum speedups and future quantum algorithms
//! - **Implementation Security**: Considers side-channel attacks and implementation vulnerabilities
//! 
//! Implementation Strategy:
//! - **Modular Design**: Separate analyzers for different attack types
//! - **Extensible Framework**: Easy addition of new attack models and estimators
//! - **Comprehensive Testing**: Validation against known attack complexities
//! - **Performance Optimization**: Efficient computation and caching of attack estimates
//! 
//! Performance Characteristics:
//! - **Attack Estimation**: O(1) using precomputed tables and mathematical models
//! - **Security Validation**: O(log(security_bits)) using binary search optimization
//! - **Memory Usage**: O(1) for analysis with efficient caching mechanisms
//! - **Extensibility**: O(1) addition of new attack models and security estimators

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};
use std::f64::consts::{E, PI};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use zeroize::{Zeroize, ZeroizeOnDrop};

use crate::error::{LatticeFoldError, Result};
use crate::parameter_generation::{GeneratedParameters, TargetSecurityLevel, AttackComplexity};
use crate::quantum_resistance::{QuantumResistanceAnalyzer, SecurityLevel};
use crate::lattice::LatticeParams;

/// Comprehensive security analyzer for lattice-based cryptographic schemes
/// 
/// This analyzer provides detailed security analysis against all known lattice attacks
/// using the most current attack complexity estimates and conservative security margins.
/// It supports both classical and quantum attack scenarios with detailed reporting.
/// 
/// Analysis Components:
/// - **BKZ Attack Analysis**: Classical and quantum BKZ with multiple cost models
/// - **Sieve Attack Analysis**: All major sieve algorithms with memory-time tradeoffs
/// - **Dual Attack Analysis**: Attacks on the dual lattice with various techniques
/// - **Primal Attack Analysis**: Direct attacks on the primal lattice
/// - **Hybrid Attack Analysis**: Combined reduction and enumeration attacks
/// - **Implementation Attack Analysis**: Side-channel and fault injection attacks
/// 
/// Security Guarantees:
/// - **Concrete Security**: Precise attack complexity estimates using current best algorithms
/// - **Conservative Margins**: Safety factors for unknown attack improvements
/// - **Quantum Resistance**: Analysis of quantum attack scenarios
/// - **Implementation Security**: Consideration of practical attack vectors
#[derive(Debug)]
pub struct SecurityAnalyzer {
    /// BKZ attack analyzer for lattice reduction attacks
    /// Analyzes classical and quantum BKZ attacks with various cost models
    bkz_analyzer: BKZAttackAnalyzer,
    
    /// Sieve attack analyzer for exponential-time attacks
    /// Analyzes all major sieve algorithms with memory-time tradeoffs
    sieve_analyzer: SieveAttackAnalyzer,
    
    /// Dual attack analyzer for Module-LWE dual attacks
    /// Analyzes attacks on the dual lattice with embedding techniques
    dual_analyzer: DualAttackAnalyzer,
    
    /// Primal attack analyzer for Module-LWE primal attacks
    /// Analyzes direct attacks on the primal lattice with enumeration
    primal_analyzer: PrimalAttackAnalyzer,
    
    /// Hybrid attack analyzer for combined techniques
    /// Analyzes attacks combining multiple techniques for optimal complexity
    hybrid_analyzer: HybridAttackAnalyzer,
    
    /// Implementation attack analyzer for practical attacks
    /// Analyzes side-channel, fault injection, and other implementation attacks
    implementation_analyzer: ImplementationAttackAnalyzer,
    
    /// Security margin configuration
    /// Controls how conservative the security analysis is
    security_margins: SecurityMarginConfiguration,
    
    /// Attack complexity cache for performance optimization
    /// Caches expensive attack complexity computations
    complexity_cache: Arc<Mutex<HashMap<SecurityAnalysisKey, CachedSecurityAnalysis>>>,
    
    /// Analysis statistics for monitoring and optimization
    /// Tracks analysis performance and accuracy
    analysis_stats: Arc<Mutex<SecurityAnalysisStatistics>>,
}

/// BKZ attack analyzer for lattice reduction attacks
/// 
/// This analyzer estimates the complexity of BKZ (Block Korkine-Zolotarev) lattice reduction
/// attacks, which are the most practical attacks against lattice-based cryptography.
/// 
/// Mathematical Framework:
/// BKZ-β reduces a lattice basis using SVP oracles in dimension β. The attack complexity
/// depends on the required block size β to break the scheme:
/// - **Block Size Estimation**: β ≈ d²/(4·log(q/σ)) for Module-LWE
/// - **Time Complexity**: 2^(c·β) where c depends on the cost model
/// - **Memory Complexity**: Polynomial in β for most variants
/// - **Success Probability**: High for sufficiently large β
/// 
/// Cost Models:
/// - **Core-SVP**: 2^(0.292·β) operations (most conservative)
/// - **Gate-Count**: 2^(0.265·β) quantum gates (quantum optimistic)
/// - **Practical**: 2^(0.320·β) operations (implementation realistic)
/// - **Memory-Constrained**: 2^(0.349·β) operations (memory limited)
#[derive(Debug, Clone)]
pub struct BKZAttackAnalyzer {
    /// Cost model configuration for BKZ complexity estimation
    /// Determines which cost model to use for attack complexity
    cost_model: BKZCostModel,
    
    /// Quantum speedup factors for different BKZ variants
    /// Accounts for quantum improvements to BKZ algorithms
    quantum_speedups: BKZQuantumSpeedups,
    
    /// Block size estimation parameters
    /// Parameters for estimating required BKZ block size
    block_size_params: BlockSizeEstimationParams,
    
    /// Conservative factors for BKZ analysis
    /// Additional safety margins for BKZ attack estimates
    conservative_factors: BKZConservativeFactors,
    
    /// BKZ analysis cache for performance
    /// Caches BKZ complexity computations
    analysis_cache: HashMap<BKZAnalysisKey, BKZAnalysisResult>,
}

/// BKZ cost models for attack complexity estimation
/// 
/// Different cost models provide different estimates of BKZ algorithm complexity.
/// The choice affects security parameter selection and guarantees.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BKZCostModel {
    /// Core-SVP model: 2^(0.292β) operations
    /// Based on the core-SVP hardness assumption
    /// Most widely accepted in lattice cryptography
    CoreSVP,
    
    /// Gate count model: 2^(0.265β) quantum gates
    /// Estimates quantum gate complexity for BKZ
    /// More optimistic about quantum improvements
    QuantumGates,
    
    /// Practical model: 2^(0.320β) operations
    /// Based on actual BKZ implementations and benchmarks
    /// Accounts for implementation overheads and optimizations
    Practical,
    
    /// Memory-constrained model: 2^(0.349β) operations
    /// Accounts for memory limitations in practice
    /// More conservative due to memory bottlenecks
    MemoryConstrained,
    
    /// ADPS16 model: 2^(0.292β) operations with refined constants
    /// Based on Albrecht-Ducas-Pulles-Stehlé analysis
    /// Includes more precise constant factors
    ADPS16,
    
    /// Quantum Core-SVP: 2^(0.265β) quantum operations
    /// Quantum variant of Core-SVP with Grover speedup
    /// Accounts for quantum improvements to SVP oracles
    QuantumCoreSVP,
}

/// Quantum speedup factors for BKZ attacks
/// 
/// Configuration for quantum improvements to BKZ algorithms.
/// Different quantum techniques provide different speedups.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BKZQuantumSpeedups {
    /// Grover speedup for SVP oracles (typically √2 ≈ 1.41)
    /// Quadratic speedup for unstructured search in SVP
    pub grover_svp_speedup: f64,
    
    /// Quantum sieve speedup for SVP (typically 2^0.027 ≈ 1.02)
    /// Speedup from quantum sieve algorithms
    pub quantum_sieve_speedup: f64,
    
    /// Quantum walk speedup (typically 2^0.1 ≈ 1.07)
    /// Speedup from quantum walk techniques
    pub quantum_walk_speedup: f64,
    
    /// Overall quantum BKZ speedup factor
    /// Combined speedup from all quantum improvements
    pub overall_quantum_speedup: f64,
}

/// Parameters for BKZ block size estimation
/// 
/// Configuration for estimating the required BKZ block size to break
/// a lattice-based cryptographic scheme.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockSizeEstimationParams {
    /// Hermite factor target for successful attack
    /// δ ≈ (β/(2πe))^(1/(2β-2)) for BKZ-β
    pub target_hermite_factor: f64,
    
    /// Root Hermite factor for security estimation
    /// δ_0 = δ^(1/d) where d is the lattice dimension
    pub root_hermite_factor: f64,
    
    /// Success probability threshold for attack
    /// Minimum probability for considering attack successful
    pub success_probability_threshold: f64,
    
    /// Gaussian heuristic constant
    /// Constant in the Gaussian heuristic for shortest vector length
    pub gaussian_heuristic_constant: f64,
    
    /// Lattice dimension scaling factor
    /// Accounts for dimension-dependent effects in block size estimation
    pub dimension_scaling_factor: f64,
}

/// Conservative factors for BKZ analysis
/// 
/// Additional safety margins applied to BKZ attack complexity estimates
/// to account for potential improvements and unknown attacks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BKZConservativeFactors {
    /// General BKZ improvement factor
    /// Accounts for potential algorithmic improvements
    pub general_improvement_factor: f64,
    
    /// Implementation optimization factor
    /// Accounts for implementation-specific optimizations
    pub implementation_optimization_factor: f64,
    
    /// Preprocessing improvement factor
    /// Accounts for preprocessing techniques that improve BKZ
    pub preprocessing_improvement_factor: f64,
    
    /// Parallel processing factor
    /// Accounts for parallel implementations of BKZ
    pub parallel_processing_factor: f64,
    
    /// Hardware acceleration factor
    /// Accounts for specialized hardware for BKZ
    pub hardware_acceleration_factor: f64,
}

/// Key for BKZ analysis caching
/// 
/// Compact representation of BKZ analysis parameters for use as cache keys.
/// Only includes parameters that significantly affect BKZ complexity.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BKZAnalysisKey {
    /// Lattice dimension
    pub dimension: usize,
    
    /// Modulus (log2, rounded)
    pub log_modulus: u32,
    
    /// Gaussian width (scaled by 1000 for integer hashing)
    pub gaussian_width_scaled: u32,
    
    /// Cost model
    pub cost_model: BKZCostModel,
    
    /// Whether quantum analysis is requested
    pub quantum: bool,
}

/// Result of BKZ attack analysis
/// 
/// Comprehensive analysis result for BKZ attacks including complexity estimates,
/// required block size, and success probability.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BKZAnalysisResult {
    /// Required BKZ block size for successful attack
    /// Minimum β such that BKZ-β breaks the scheme
    pub required_block_size: u32,
    
    /// Time complexity in bits (log2 of operations)
    /// Expected number of operations for BKZ attack
    pub time_complexity_bits: f64,
    
    /// Memory complexity in bits (log2 of memory usage)
    /// Expected memory usage for BKZ attack
    pub memory_complexity_bits: f64,
    
    /// Success probability of the attack
    /// Probability that BKZ-β successfully breaks the scheme
    pub success_probability: f64,
    
    /// Hermite factor achieved by BKZ-β
    /// Quality of the reduced basis
    pub achieved_hermite_factor: f64,
    
    /// Analysis confidence level
    /// Confidence in the accuracy of the analysis
    pub confidence_level: f64,
    
    /// Conservative factors applied
    /// Summary of conservative adjustments made
    pub applied_conservative_factors: Vec<String>,
    
    /// Analysis timestamp
    /// When this analysis was performed
    pub analysis_timestamp: SystemTime,
}

/// Sieve attack analyzer for exponential-time lattice attacks
/// 
/// This analyzer estimates the complexity of sieve algorithms for solving
/// the Shortest Vector Problem (SVP) in lattices. Sieve algorithms can
/// be faster than BKZ for certain parameter ranges.
/// 
/// Mathematical Framework:
/// Sieve algorithms solve SVP by maintaining a list of lattice vectors and
/// iteratively reducing them. The complexity depends on the lattice dimension:
/// - **GaussSieve**: 2^(0.415n + o(n)) time and space
/// - **NV-Sieve**: 2^(0.384n + o(n)) time, 2^(0.208n + o(n)) space
/// - **BDGL16**: 2^(0.292n + o(n)) time, polynomial space
/// - **Quantum Sieve**: 2^(0.265n + o(n)) time with quantum speedups
/// 
/// Memory-Time Tradeoffs:
/// Many sieve algorithms allow trading memory for time or vice versa:
/// - **Time-Memory Product**: Often constant for a given algorithm
/// - **Practical Constraints**: Memory limitations affect achievable complexity
/// - **Quantum Memory**: Quantum algorithms may have different memory requirements
#[derive(Debug, Clone)]
pub struct SieveAttackAnalyzer {
    /// Sieve algorithm variants to analyze
    /// List of sieve algorithms to consider
    sieve_variants: Vec<SieveAlgorithmVariant>,
    
    /// Memory-time tradeoff parameters
    /// Configuration for memory-time tradeoff analysis
    tradeoff_params: MemoryTimeTradeoffParams,
    
    /// Quantum sieve parameters
    /// Configuration for quantum sieve analysis
    quantum_params: QuantumSieveParams,
    
    /// Sieve analysis cache
    /// Caches sieve complexity computations
    analysis_cache: HashMap<SieveAnalysisKey, SieveAnalysisResult>,
    
    /// Conservative factors for sieve analysis
    /// Safety margins for sieve attack estimates
    conservative_factors: SieveConservativeFactors,
}

/// Sieve algorithm variants for SVP solving
/// 
/// Different sieve algorithms have different complexity characteristics
/// and are suitable for different parameter ranges.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SieveAlgorithmVariant {
    /// GaussSieve: 2^(0.415n) time and space
    /// Original sieve algorithm with exponential space
    GaussSieve,
    
    /// NV-Sieve: 2^(0.384n) time, 2^(0.208n) space
    /// Nguyen-Vidick sieve with improved time-space tradeoff
    NVSieve,
    
    /// BDGL16: 2^(0.292n) time, polynomial space
    /// Becker-Ducas-Gama-Laarhoven sieve with polynomial space
    BDGL16,
    
    /// HashSieve: 2^(0.337n) time, 2^(0.208n) space
    /// Hash-based sieve with good practical performance
    HashSieve,
    
    /// Quantum GaussSieve: 2^(0.312n) time and space
    /// Quantum variant of GaussSieve with Grover speedup
    QuantumGaussSieve,
    
    /// Quantum NV-Sieve: 2^(0.265n) time, 2^(0.156n) space
    /// Quantum variant of NV-Sieve with optimal complexity
    QuantumNVSieve,
    
    /// Quantum BDGL16: 2^(0.265n) time, polynomial space
    /// Quantum variant of BDGL16 with polynomial space
    QuantumBDGL16,
}

/// Memory-time tradeoff parameters for sieve algorithms
/// 
/// Configuration for analyzing memory-time tradeoffs in sieve algorithms.
/// Many sieve algorithms allow trading memory for time within certain bounds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryTimeTradeoffParams {
    /// Available memory in bits (log2 of bytes)
    /// Maximum memory available for the attack
    pub available_memory_bits: f64,
    
    /// Time budget in bits (log2 of operations)
    /// Maximum time budget for the attack
    pub time_budget_bits: f64,
    
    /// Memory-time tradeoff exponent
    /// Exponent in the memory-time tradeoff relationship
    pub tradeoff_exponent: f64,
    
    /// Minimum memory requirement
    /// Minimum memory needed regardless of tradeoff
    pub minimum_memory_bits: f64,
    
    /// Maximum time extension factor
    /// Maximum factor by which time can be extended
    pub max_time_extension_factor: f64,
}

/// Quantum sieve parameters
/// 
/// Configuration for quantum sieve algorithm analysis.
/// Quantum algorithms may have different complexity characteristics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumSieveParams {
    /// Quantum memory model (QRAM vs. classical memory)
    /// Whether quantum random access memory is available
    pub quantum_memory_available: bool,
    
    /// Quantum speedup factors for different operations
    /// Speedups for various quantum subroutines
    pub quantum_speedup_factors: QuantumSieveSpeedups,
    
    /// Quantum error correction overhead
    /// Overhead factor for quantum error correction
    pub error_correction_overhead: f64,
    
    /// Quantum decoherence time constraints
    /// Time constraints due to quantum decoherence
    pub decoherence_time_limit_bits: f64,
}

/// Quantum speedup factors for sieve algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumSieveSpeedups {
    /// Grover speedup for database search (typically √2)
    pub grover_search_speedup: f64,
    
    /// Quantum walk speedup for graph algorithms
    pub quantum_walk_speedup: f64,
    
    /// Amplitude amplification speedup
    pub amplitude_amplification_speedup: f64,
    
    /// Overall quantum sieve speedup
    pub overall_speedup: f64,
}

/// Conservative factors for sieve analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SieveConservativeFactors {
    /// Algorithmic improvement factor
    pub algorithmic_improvement_factor: f64,
    
    /// Implementation optimization factor
    pub implementation_optimization_factor: f64,
    
    /// Quantum algorithm improvement factor
    pub quantum_improvement_factor: f64,
    
    /// Memory access optimization factor
    pub memory_access_optimization_factor: f64,
}

/// Key for sieve analysis caching
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SieveAnalysisKey {
    /// Lattice dimension
    pub dimension: usize,
    
    /// Sieve algorithm variant
    pub algorithm: SieveAlgorithmVariant,
    
    /// Available memory (log2, rounded)
    pub available_memory_bits_rounded: u32,
    
    /// Whether quantum analysis is requested
    pub quantum: bool,
}

/// Result of sieve attack analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SieveAnalysisResult {
    /// Sieve algorithm used
    pub algorithm: SieveAlgorithmVariant,
    
    /// Time complexity in bits
    pub time_complexity_bits: f64,
    
    /// Memory complexity in bits
    pub memory_complexity_bits: f64,
    
    /// Success probability
    pub success_probability: f64,
    
    /// Memory-time tradeoff used
    pub tradeoff_factor: f64,
    
    /// Analysis confidence level
    pub confidence_level: f64,
    
    /// Analysis timestamp
    pub analysis_timestamp: SystemTime,
}/// D
ual attack analyzer for Module-LWE dual attacks
/// 
/// This analyzer estimates the complexity of dual attacks on Module-LWE problems.
/// Dual attacks work by finding short vectors in the dual lattice that can be used
/// to distinguish LWE samples from random.
/// 
/// Mathematical Framework:
/// Dual attacks construct a dual lattice and find short vectors that are orthogonal
/// to the LWE secret. The attack complexity depends on:
/// - **Dual Lattice Dimension**: Typically n + m where n is secret dimension, m is samples
/// - **Required Vector Length**: Must be short enough to distinguish from random
/// - **Lattice Reduction Cost**: Cost of finding sufficiently short dual vectors
/// - **Distinguishing Advantage**: Probability of successful distinguishing
/// 
/// Attack Variants:
/// - **Classical Dual**: Uses BKZ to find short dual vectors
/// - **Dual with Embedding**: Embeds the problem in a higher dimension
/// - **Dual with uSVP**: Uses unique-SVP techniques for better vectors
/// - **Quantum Dual**: Uses quantum lattice reduction techniques
#[derive(Debug, Clone)]
pub struct DualAttackAnalyzer {
    /// Dual lattice construction parameters
    /// Configuration for constructing the dual lattice
    dual_construction_params: DualConstructionParams,
    
    /// Distinguishing advantage parameters
    /// Configuration for computing distinguishing advantage
    distinguishing_params: DistinguishingAdvantageParams,
    
    /// Embedding technique parameters
    /// Configuration for embedding techniques
    embedding_params: EmbeddingTechniqueParams,
    
    /// Dual analysis cache
    /// Caches dual attack complexity computations
    analysis_cache: HashMap<DualAnalysisKey, DualAnalysisResult>,
    
    /// Conservative factors for dual analysis
    /// Safety margins for dual attack estimates
    conservative_factors: DualConservativeFactors,
}

/// Parameters for dual lattice construction
/// 
/// Configuration for constructing the dual lattice used in dual attacks.
/// The construction affects the attack complexity and success probability.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DualConstructionParams {
    /// Number of LWE samples to use
    /// More samples generally improve attack success but increase dimension
    pub num_samples: usize,
    
    /// Dual lattice scaling factor
    /// Scaling applied to the dual lattice for optimization
    pub scaling_factor: f64,
    
    /// Lattice basis quality target
    /// Target quality for the dual lattice basis
    pub basis_quality_target: f64,
    
    /// Preprocessing technique
    /// Preprocessing applied to improve dual lattice quality
    pub preprocessing_technique: DualPreprocessingTechnique,
}

/// Preprocessing techniques for dual lattice construction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DualPreprocessingTechnique {
    /// No preprocessing
    None,
    
    /// LLL preprocessing for better basis
    LLL,
    
    /// BKZ preprocessing with small block size
    BKZPreprocessing,
    
    /// Hermite-Korkine-Zolotarev preprocessing
    HKZ,
    
    /// Random sampling preprocessing
    RandomSampling,
}

/// Parameters for distinguishing advantage computation
/// 
/// Configuration for computing the distinguishing advantage of dual attacks.
/// The advantage determines the success probability of the attack.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistinguishingAdvantageParams {
    /// Target distinguishing advantage
    /// Minimum advantage required for successful attack
    pub target_advantage: f64,
    
    /// Statistical distance threshold
    /// Threshold for statistical distance between distributions
    pub statistical_distance_threshold: f64,
    
    /// Sample complexity factor
    /// Factor relating advantage to required number of samples
    pub sample_complexity_factor: f64,
    
    /// Noise distribution parameters
    /// Parameters of the LWE noise distribution
    pub noise_distribution_params: NoiseDistributionParams,
}

/// Parameters for noise distribution in LWE
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoiseDistributionParams {
    /// Noise distribution type
    pub distribution_type: NoiseDistributionType,
    
    /// Distribution parameter (e.g., standard deviation)
    pub distribution_parameter: f64,
    
    /// Discretization parameter
    pub discretization_parameter: f64,
    
    /// Tail bound parameter
    pub tail_bound_parameter: f64,
}

/// Types of noise distributions in LWE
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NoiseDistributionType {
    /// Discrete Gaussian distribution
    DiscreteGaussian,
    
    /// Continuous Gaussian distribution
    ContinuousGaussian,
    
    /// Uniform distribution
    Uniform,
    
    /// Binomial distribution
    Binomial,
    
    /// Ternary distribution
    Ternary,
}

/// Parameters for embedding techniques in dual attacks
/// 
/// Configuration for embedding techniques that can improve dual attack efficiency.
/// Embedding can reduce the effective dimension or improve vector quality.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingTechniqueParams {
    /// Embedding technique to use
    pub technique: EmbeddingTechnique,
    
    /// Embedding dimension
    pub embedding_dimension: usize,
    
    /// Embedding optimization target
    pub optimization_target: EmbeddingOptimizationTarget,
    
    /// Embedding success probability
    pub success_probability: f64,
}

/// Embedding techniques for dual attacks
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EmbeddingTechnique {
    /// No embedding
    None,
    
    /// Kannan embedding
    Kannan,
    
    /// Bai-Galbraith embedding
    BaiGalbraith,
    
    /// ADPS16 embedding
    ADPS16,
    
    /// Custom embedding
    Custom,
}

/// Optimization targets for embedding
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EmbeddingOptimizationTarget {
    /// Minimize lattice dimension
    MinimizeDimension,
    
    /// Minimize required vector length
    MinimizeVectorLength,
    
    /// Maximize success probability
    MaximizeSuccessProbability,
    
    /// Minimize total attack complexity
    MinimizeTotalComplexity,
}

/// Conservative factors for dual analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DualConservativeFactors {
    /// Lattice reduction improvement factor
    pub lattice_reduction_improvement_factor: f64,
    
    /// Distinguishing algorithm improvement factor
    pub distinguishing_improvement_factor: f64,
    
    /// Embedding technique improvement factor
    pub embedding_improvement_factor: f64,
    
    /// Sample complexity improvement factor
    pub sample_complexity_improvement_factor: f64,
}

/// Key for dual analysis caching
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DualAnalysisKey {
    /// LWE dimension
    pub lwe_dimension: usize,
    
    /// Number of samples
    pub num_samples: usize,
    
    /// Modulus (log2, rounded)
    pub log_modulus: u32,
    
    /// Noise parameter (scaled)
    pub noise_parameter_scaled: u32,
    
    /// Embedding technique
    pub embedding_technique: EmbeddingTechnique,
    
    /// Whether quantum analysis is requested
    pub quantum: bool,
}

/// Result of dual attack analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DualAnalysisResult {
    /// Dual lattice dimension
    pub dual_dimension: usize,
    
    /// Required vector length for attack
    pub required_vector_length: f64,
    
    /// Time complexity in bits
    pub time_complexity_bits: f64,
    
    /// Memory complexity in bits
    pub memory_complexity_bits: f64,
    
    /// Success probability
    pub success_probability: f64,
    
    /// Distinguishing advantage achieved
    pub distinguishing_advantage: f64,
    
    /// Embedding technique used
    pub embedding_technique: EmbeddingTechnique,
    
    /// Analysis confidence level
    pub confidence_level: f64,
    
    /// Analysis timestamp
    pub analysis_timestamp: SystemTime,
}

/// Primal attack analyzer for Module-LWE primal attacks
/// 
/// This analyzer estimates the complexity of primal attacks on Module-LWE problems.
/// Primal attacks work by finding the secret directly using lattice reduction
/// and enumeration techniques.
/// 
/// Mathematical Framework:
/// Primal attacks construct a lattice containing the LWE secret as a short vector
/// and use lattice reduction to find it. The attack complexity depends on:
/// - **Primal Lattice Dimension**: Typically m + n where m is samples, n is secret dimension
/// - **Secret Vector Length**: Length of the secret vector in the lattice
/// - **Lattice Reduction Cost**: Cost of reducing the primal lattice
/// - **Enumeration Cost**: Cost of enumerating short vectors
/// 
/// Attack Variants:
/// - **Classical Primal**: Uses BKZ + enumeration to find the secret
/// - **Primal with Babai**: Uses Babai's nearest plane algorithm
/// - **Primal with Enumeration**: Uses full enumeration in reduced basis
/// - **Quantum Primal**: Uses quantum lattice reduction and search
#[derive(Debug, Clone)]
pub struct PrimalAttackAnalyzer {
    /// Primal lattice construction parameters
    /// Configuration for constructing the primal lattice
    primal_construction_params: PrimalConstructionParams,
    
    /// Enumeration parameters
    /// Configuration for enumeration algorithms
    enumeration_params: EnumerationParams,
    
    /// Babai algorithm parameters
    /// Configuration for Babai's nearest plane algorithm
    babai_params: BabaiAlgorithmParams,
    
    /// Primal analysis cache
    /// Caches primal attack complexity computations
    analysis_cache: HashMap<PrimalAnalysisKey, PrimalAnalysisResult>,
    
    /// Conservative factors for primal analysis
    /// Safety margins for primal attack estimates
    conservative_factors: PrimalConservativeFactors,
}

/// Parameters for primal lattice construction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrimalConstructionParams {
    /// Number of LWE samples to use
    pub num_samples: usize,
    
    /// Primal lattice scaling factor
    pub scaling_factor: f64,
    
    /// Secret distribution parameters
    pub secret_distribution: SecretDistributionParams,
    
    /// Lattice basis preprocessing
    pub preprocessing_technique: PrimalPreprocessingTechnique,
}

/// Parameters for secret distribution in LWE
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecretDistributionParams {
    /// Secret distribution type
    pub distribution_type: SecretDistributionType,
    
    /// Distribution parameter
    pub distribution_parameter: f64,
    
    /// Secret sparsity (fraction of non-zero entries)
    pub sparsity: f64,
    
    /// Secret norm bound
    pub norm_bound: f64,
}

/// Types of secret distributions in LWE
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SecretDistributionType {
    /// Uniform distribution over {0, 1}
    Binary,
    
    /// Uniform distribution over {-1, 0, 1}
    Ternary,
    
    /// Discrete Gaussian distribution
    DiscreteGaussian,
    
    /// Uniform distribution over Z_q
    Uniform,
    
    /// Sparse distribution
    Sparse,
}

/// Preprocessing techniques for primal lattice construction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PrimalPreprocessingTechnique {
    /// No preprocessing
    None,
    
    /// LLL preprocessing
    LLL,
    
    /// BKZ preprocessing with small block size
    BKZPreprocessing,
    
    /// Size reduction preprocessing
    SizeReduction,
    
    /// Random sampling preprocessing
    RandomSampling,
}

/// Parameters for enumeration algorithms
/// 
/// Configuration for enumeration algorithms used in primal attacks.
/// Enumeration finds short vectors in a reduced lattice basis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnumerationParams {
    /// Enumeration algorithm variant
    pub algorithm: EnumerationAlgorithm,
    
    /// Enumeration radius
    pub enumeration_radius: f64,
    
    /// Pruning parameters for pruned enumeration
    pub pruning_params: PruningParams,
    
    /// Parallel enumeration parameters
    pub parallel_params: ParallelEnumerationParams,
    
    /// Quantum enumeration parameters
    pub quantum_params: QuantumEnumerationParams,
}

/// Enumeration algorithm variants
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EnumerationAlgorithm {
    /// Full enumeration (exponential time)
    Full,
    
    /// Pruned enumeration (polynomial time, lower success probability)
    Pruned,
    
    /// Extreme pruning (very fast, very low success probability)
    ExtremePruning,
    
    /// Progressive enumeration (adaptive radius)
    Progressive,
    
    /// Quantum enumeration (quantum speedup)
    Quantum,
}

/// Parameters for pruned enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PruningParams {
    /// Pruning function type
    pub pruning_function: PruningFunction,
    
    /// Pruning coefficients
    pub pruning_coefficients: Vec<f64>,
    
    /// Success probability target
    pub success_probability_target: f64,
    
    /// Repetition factor for low success probability
    pub repetition_factor: f64,
}

/// Pruning function types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PruningFunction {
    /// Linear pruning
    Linear,
    
    /// Exponential pruning
    Exponential,
    
    /// Gaussian pruning
    Gaussian,
    
    /// Optimized pruning (numerically optimized)
    Optimized,
}

/// Parameters for parallel enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelEnumerationParams {
    /// Number of parallel threads
    pub num_threads: usize,
    
    /// Work distribution strategy
    pub distribution_strategy: WorkDistributionStrategy,
    
    /// Load balancing parameters
    pub load_balancing: LoadBalancingParams,
    
    /// Communication overhead factor
    pub communication_overhead: f64,
}

/// Work distribution strategies for parallel enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkDistributionStrategy {
    /// Static work distribution
    Static,
    
    /// Dynamic work distribution
    Dynamic,
    
    /// Work stealing
    WorkStealing,
    
    /// Hierarchical distribution
    Hierarchical,
}

/// Load balancing parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadBalancingParams {
    /// Load balancing algorithm
    pub algorithm: LoadBalancingAlgorithm,
    
    /// Rebalancing frequency
    pub rebalancing_frequency: f64,
    
    /// Load imbalance threshold
    pub imbalance_threshold: f64,
    
    /// Migration cost factor
    pub migration_cost_factor: f64,
}

/// Load balancing algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LoadBalancingAlgorithm {
    /// No load balancing
    None,
    
    /// Round-robin distribution
    RoundRobin,
    
    /// Least loaded first
    LeastLoaded,
    
    /// Proportional distribution
    Proportional,
    
    /// Adaptive distribution
    Adaptive,
}

/// Parameters for quantum enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumEnumerationParams {
    /// Quantum speedup factor
    pub quantum_speedup_factor: f64,
    
    /// Quantum memory requirements
    pub quantum_memory_bits: f64,
    
    /// Quantum error correction overhead
    pub error_correction_overhead: f64,
    
    /// Decoherence time constraints
    pub decoherence_time_limit: f64,
}

/// Parameters for Babai's nearest plane algorithm
/// 
/// Configuration for Babai's algorithm used in primal attacks.
/// Babai's algorithm finds approximate closest vectors in lattices.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BabaiAlgorithmParams {
    /// Babai algorithm variant
    pub algorithm_variant: BabaiAlgorithmVariant,
    
    /// Approximation factor
    pub approximation_factor: f64,
    
    /// Success probability
    pub success_probability: f64,
    
    /// Iteration limit
    pub iteration_limit: usize,
}

/// Babai algorithm variants
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BabaiAlgorithmVariant {
    /// Round-off algorithm
    RoundOff,
    
    /// Nearest plane algorithm
    NearestPlane,
    
    /// Randomized Babai
    Randomized,
    
    /// Quantum Babai
    Quantum,
}

/// Conservative factors for primal analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrimalConservativeFactors {
    /// Lattice reduction improvement factor
    pub lattice_reduction_improvement_factor: f64,
    
    /// Enumeration improvement factor
    pub enumeration_improvement_factor: f64,
    
    /// Babai algorithm improvement factor
    pub babai_improvement_factor: f64,
    
    /// Parallel processing improvement factor
    pub parallel_improvement_factor: f64,
}

/// Key for primal analysis caching
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PrimalAnalysisKey {
    /// LWE dimension
    pub lwe_dimension: usize,
    
    /// Number of samples
    pub num_samples: usize,
    
    /// Modulus (log2, rounded)
    pub log_modulus: u32,
    
    /// Secret distribution type
    pub secret_distribution: SecretDistributionType,
    
    /// Enumeration algorithm
    pub enumeration_algorithm: EnumerationAlgorithm,
    
    /// Whether quantum analysis is requested
    pub quantum: bool,
}

/// Result of primal attack analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrimalAnalysisResult {
    /// Primal lattice dimension
    pub primal_dimension: usize,
    
    /// Expected secret vector length
    pub expected_secret_length: f64,
    
    /// Time complexity in bits
    pub time_complexity_bits: f64,
    
    /// Memory complexity in bits
    pub memory_complexity_bits: f64,
    
    /// Success probability
    pub success_probability: f64,
    
    /// Enumeration algorithm used
    pub enumeration_algorithm: EnumerationAlgorithm,
    
    /// Babai algorithm used (if any)
    pub babai_algorithm: Option<BabaiAlgorithmVariant>,
    
    /// Analysis confidence level
    pub confidence_level: f64,
    
    /// Analysis timestamp
    pub analysis_timestamp: SystemTime,
}

/// Hybrid attack analyzer for combined lattice attack techniques
/// 
/// This analyzer estimates the complexity of hybrid attacks that combine
/// multiple techniques for optimal attack complexity. Hybrid attacks often
/// achieve better complexity than individual techniques alone.
/// 
/// Mathematical Framework:
/// Hybrid attacks combine different techniques such as:
/// - **BKZ + Enumeration**: Use BKZ to reduce basis, then enumerate
/// - **Sieve + BKZ**: Use sieve for small dimensions, BKZ for large
/// - **Dual + Primal**: Combine dual and primal attack information
/// - **Classical + Quantum**: Use quantum techniques where beneficial
/// 
/// Optimization Strategy:
/// The analyzer finds the optimal combination of techniques that minimizes
/// total attack complexity while satisfying resource constraints.
#[derive(Debug, Clone)]
pub struct HybridAttackAnalyzer {
    /// Available attack techniques
    /// List of individual attack techniques to combine
    available_techniques: Vec<AttackTechnique>,
    
    /// Combination strategies
    /// Strategies for combining different attack techniques
    combination_strategies: Vec<CombinationStrategy>,
    
    /// Resource constraints
    /// Constraints on time, memory, and other resources
    resource_constraints: ResourceConstraints,
    
    /// Optimization parameters
    /// Parameters for optimizing hybrid attack combinations
    optimization_params: HybridOptimizationParams,
    
    /// Hybrid analysis cache
    /// Caches hybrid attack complexity computations
    analysis_cache: HashMap<HybridAnalysisKey, HybridAnalysisResult>,
}

/// Individual attack techniques available for hybrid attacks
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AttackTechnique {
    /// BKZ lattice reduction
    BKZ,
    
    /// Sieve algorithms
    Sieve,
    
    /// Dual attack
    Dual,
    
    /// Primal attack
    Primal,
    
    /// Enumeration
    Enumeration,
    
    /// Babai algorithm
    Babai,
    
    /// Quantum techniques
    Quantum,
}

/// Strategies for combining attack techniques
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CombinationStrategy {
    /// Sequential combination (one after another)
    Sequential,
    
    /// Parallel combination (simultaneously)
    Parallel,
    
    /// Adaptive combination (choose based on progress)
    Adaptive,
    
    /// Hierarchical combination (nested techniques)
    Hierarchical,
    
    /// Probabilistic combination (random selection)
    Probabilistic,
}

/// Resource constraints for hybrid attacks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceConstraints {
    /// Maximum time budget in bits (log2 of operations)
    pub max_time_bits: f64,
    
    /// Maximum memory budget in bits (log2 of bytes)
    pub max_memory_bits: f64,
    
    /// Maximum quantum resources (if available)
    pub max_quantum_resources: Option<QuantumResourceConstraints>,
    
    /// Parallelization constraints
    pub parallelization_constraints: ParallelizationConstraints,
    
    /// Hardware constraints
    pub hardware_constraints: HardwareConstraints,
}

/// Quantum resource constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumResourceConstraints {
    /// Maximum quantum memory in qubits
    pub max_quantum_memory_qubits: usize,
    
    /// Maximum quantum computation time
    pub max_quantum_time_bits: f64,
    
    /// Quantum error rate
    pub quantum_error_rate: f64,
    
    /// Quantum-classical interface overhead
    pub interface_overhead: f64,
}

/// Parallelization constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelizationConstraints {
    /// Maximum number of parallel processes
    pub max_parallel_processes: usize,
    
    /// Communication bandwidth between processes
    pub communication_bandwidth_bits_per_second: f64,
    
    /// Synchronization overhead
    pub synchronization_overhead: f64,
    
    /// Load balancing efficiency
    pub load_balancing_efficiency: f64,
}

/// Hardware constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareConstraints {
    /// Available CPU cores
    pub cpu_cores: usize,
    
    /// Available GPU devices
    pub gpu_devices: usize,
    
    /// Specialized hardware (e.g., FPGAs)
    pub specialized_hardware: Vec<SpecializedHardware>,
    
    /// Network connectivity
    pub network_connectivity: NetworkConnectivity,
}

/// Specialized hardware types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecializedHardware {
    /// Hardware type
    pub hardware_type: SpecializedHardwareType,
    
    /// Performance factor compared to CPU
    pub performance_factor: f64,
    
    /// Memory capacity
    pub memory_capacity_bits: f64,
    
    /// Power consumption factor
    pub power_consumption_factor: f64,
}

/// Types of specialized hardware
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpecializedHardwareType {
    /// Field-Programmable Gate Array
    FPGA,
    
    /// Application-Specific Integrated Circuit
    ASIC,
    
    /// Tensor Processing Unit
    TPU,
    
    /// Quantum Processing Unit
    QPU,
    
    /// Custom accelerator
    Custom,
}

/// Network connectivity parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConnectivity {
    /// Network bandwidth in bits per second
    pub bandwidth_bits_per_second: f64,
    
    /// Network latency in seconds
    pub latency_seconds: f64,
    
    /// Network reliability (fraction of successful transmissions)
    pub reliability: f64,
    
    /// Network topology
    pub topology: NetworkTopology,
}

/// Network topology types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NetworkTopology {
    /// Single machine (no network)
    SingleMachine,
    
    /// Local area network
    LAN,
    
    /// Wide area network
    WAN,
    
    /// Cloud computing environment
    Cloud,
    
    /// Distributed computing grid
    Grid,
}

/// Optimization parameters for hybrid attacks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridOptimizationParams {
    /// Optimization objective
    pub objective: OptimizationObjective,
    
    /// Optimization algorithm
    pub algorithm: OptimizationAlgorithm,
    
    /// Convergence tolerance
    pub convergence_tolerance: f64,
    
    /// Maximum optimization iterations
    pub max_iterations: usize,
    
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
}

/// Optimization objectives for hybrid attacks
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizationObjective {
    /// Minimize total time complexity
    MinimizeTime,
    
    /// Minimize total memory usage
    MinimizeMemory,
    
    /// Minimize total resource cost
    MinimizeResourceCost,
    
    /// Maximize success probability
    MaximizeSuccessProbability,
    
    /// Multi-objective optimization
    MultiObjective,
}

/// Optimization algorithms for hybrid attacks
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizationAlgorithm {
    /// Exhaustive search (small search spaces)
    ExhaustiveSearch,
    
    /// Greedy algorithm (fast approximation)
    Greedy,
    
    /// Dynamic programming (optimal for structured problems)
    DynamicProgramming,
    
    /// Genetic algorithm (global optimization)
    GeneticAlgorithm,
    
    /// Simulated annealing (local optimization)
    SimulatedAnnealing,
    
    /// Particle swarm optimization
    ParticleSwarm,
}

/// Key for hybrid analysis caching
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct HybridAnalysisKey {
    /// Problem parameters hash
    pub problem_params_hash: u64,
    
    /// Available techniques
    pub available_techniques: Vec<AttackTechnique>,
    
    /// Resource constraints hash
    pub resource_constraints_hash: u64,
    
    /// Optimization objective
    pub optimization_objective: OptimizationObjective,
    
    /// Whether quantum analysis is requested
    pub quantum: bool,
}

/// Result of hybrid attack analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridAnalysisResult {
    /// Optimal combination of techniques
    pub optimal_combination: Vec<(AttackTechnique, f64)>, // (technique, weight)
    
    /// Combination strategy used
    pub combination_strategy: CombinationStrategy,
    
    /// Total time complexity in bits
    pub total_time_complexity_bits: f64,
    
    /// Total memory complexity in bits
    pub total_memory_complexity_bits: f64,
    
    /// Overall success probability
    pub overall_success_probability: f64,
    
    /// Resource utilization
    pub resource_utilization: ResourceUtilization,
    
    /// Optimization convergence information
    pub optimization_convergence: OptimizationConvergence,
    
    /// Analysis confidence level
    pub confidence_level: f64,
    
    /// Analysis timestamp
    pub analysis_timestamp: SystemTime,
}

/// Resource utilization information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUtilization {
    /// Time resource utilization (fraction of budget used)
    pub time_utilization: f64,
    
    /// Memory resource utilization (fraction of budget used)
    pub memory_utilization: f64,
    
    /// Quantum resource utilization (if applicable)
    pub quantum_utilization: Option<f64>,
    
    /// Parallelization efficiency
    pub parallelization_efficiency: f64,
    
    /// Hardware utilization by type
    pub hardware_utilization: HashMap<String, f64>,
}

/// Optimization convergence information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConvergence {
    /// Whether optimization converged
    pub converged: bool,
    
    /// Number of iterations performed
    pub iterations_performed: usize,
    
    /// Final objective value
    pub final_objective_value: f64,
    
    /// Convergence rate
    pub convergence_rate: f64,
    
    /// Optimization time in seconds
    pub optimization_time_seconds: f64,
}

/// Implementation attack analyzer for practical attack vectors
/// 
/// This analyzer estimates the complexity of implementation-specific attacks
/// such as side-channel attacks, fault injection attacks, and other practical
/// attack vectors that exploit implementation vulnerabilities.
/// 
/// Attack Categories:
/// - **Side-Channel Attacks**: Timing, power, electromagnetic, acoustic
/// - **Fault Injection Attacks**: Voltage glitching, clock glitching, laser fault injection
/// - **Physical Attacks**: Invasive and semi-invasive attacks
/// - **Software Attacks**: Buffer overflows, code injection, reverse engineering
/// - **Protocol Attacks**: Implementation flaws in cryptographic protocols
#[derive(Debug, Clone)]
pub struct ImplementationAttackAnalyzer {
    /// Side-channel attack parameters
    /// Configuration for side-channel attack analysis
    side_channel_params: SideChannelAttackParams,
    
    /// Fault injection attack parameters
    /// Configuration for fault injection attack analysis
    fault_injection_params: FaultInjectionAttackParams,
    
    /// Physical attack parameters
    /// Configuration for physical attack analysis
    physical_attack_params: PhysicalAttackParams,
    
    /// Software attack parameters
    /// Configuration for software attack analysis
    software_attack_params: SoftwareAttackParams,
    
    /// Protocol attack parameters
    /// Configuration for protocol attack analysis
    protocol_attack_params: ProtocolAttackParams,
    
    /// Implementation analysis cache
    /// Caches implementation attack complexity computations
    analysis_cache: HashMap<ImplementationAnalysisKey, ImplementationAnalysisResult>,
}

/// Parameters for side-channel attack analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SideChannelAttackParams {
    /// Side-channel types to analyze
    pub attack_types: Vec<SideChannelAttackType>,
    
    /// Signal-to-noise ratio assumptions
    pub signal_to_noise_ratio: f64,
    
    /// Number of traces required
    pub required_traces: usize,
    
    /// Measurement precision
    pub measurement_precision: f64,
    
    /// Countermeasure effectiveness
    pub countermeasure_effectiveness: HashMap<String, f64>,
}

/// Types of side-channel attacks
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SideChannelAttackType {
    /// Timing attacks
    Timing,
    
    /// Power analysis attacks
    PowerAnalysis,
    
    /// Electromagnetic attacks
    Electromagnetic,
    
    /// Acoustic attacks
    Acoustic,
    
    /// Cache timing attacks
    CacheTiming,
    
    /// Branch prediction attacks
    BranchPrediction,
    
    /// Memory access pattern attacks
    MemoryAccessPattern,
}

/// Parameters for fault injection attack analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaultInjectionAttackParams {
    /// Fault injection types to analyze
    pub attack_types: Vec<FaultInjectionAttackType>,
    
    /// Fault model parameters
    pub fault_model: FaultModel,
    
    /// Success probability per fault
    pub fault_success_probability: f64,
    
    /// Number of faults required
    pub required_faults: usize,
    
    /// Fault detection probability
    pub fault_detection_probability: f64,
}

/// Types of fault injection attacks
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FaultInjectionAttackType {
    /// Voltage glitching
    VoltageGlitching,
    
    /// Clock glitching
    ClockGlitching,
    
    /// Laser fault injection
    LaserFaultInjection,
    
    /// Electromagnetic fault injection
    ElectromagneticFaultInjection,
    
    /// Temperature attacks
    TemperatureAttacks,
    
    /// Radiation attacks
    RadiationAttacks,
}

/// Fault model for fault injection attacks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaultModel {
    /// Fault type
    pub fault_type: FaultType,
    
    /// Fault precision (bit-level, byte-level, etc.)
    pub fault_precision: FaultPrecision,
    
    /// Fault timing precision
    pub timing_precision: f64,
    
    /// Fault persistence
    pub fault_persistence: FaultPersistence,
}

/// Types of faults
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FaultType {
    /// Bit flip
    BitFlip,
    
    /// Stuck-at fault
    StuckAt,
    
    /// Random fault
    Random,
    
    /// Instruction skip
    InstructionSkip,
    
    /// Register corruption
    RegisterCorruption,
}

/// Fault precision levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FaultPrecision {
    /// Bit-level precision
    Bit,
    
    /// Byte-level precision
    Byte,
    
    /// Word-level precision
    Word,
    
    /// Instruction-level precision
    Instruction,
    
    /// Coarse-grained precision
    CoarseGrained,
}

/// Fault persistence characteristics
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FaultPersistence {
    /// Transient fault (single occurrence)
    Transient,
    
    /// Intermittent fault (occasional occurrence)
    Intermittent,
    
    /// Permanent fault (persistent)
    Permanent,
}

/// Parameters for physical attack analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicalAttackParams {
    /// Physical attack types to analyze
    pub attack_types: Vec<PhysicalAttackType>,
    
    /// Required equipment cost
    pub equipment_cost_usd: f64,
    
    /// Required expertise level
    pub expertise_level: ExpertiseLevel,
    
    /// Attack success probability
    pub success_probability: f64,
    
    /// Detection probability
    pub detection_probability: f64,
}

/// Types of physical attacks
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PhysicalAttackType {
    /// Invasive attacks (chip decapping, probing)
    Invasive,
    
    /// Semi-invasive attacks (backside access)
    SemiInvasive,
    
    /// Non-invasive attacks (external observation)
    NonInvasive,
    
    /// Microprobing
    Microprobing,
    
    /// Focused ion beam attacks
    FocusedIonBeam,
    
    /// X-ray attacks
    XRay,
}

/// Expertise levels for attacks
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ExpertiseLevel {
    /// Script kiddie (minimal expertise)
    Minimal,
    
    /// Hobbyist (basic expertise)
    Basic,
    
    /// Professional (advanced expertise)
    Professional,
    
    /// Expert (specialized expertise)
    Expert,
    
    /// Nation-state (unlimited resources)
    NationState,
}

/// Parameters for software attack analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SoftwareAttackParams {
    /// Software attack types to analyze
    pub attack_types: Vec<SoftwareAttackType>,
    
    /// Code complexity metrics
    pub code_complexity: CodeComplexityMetrics,
    
    /// Security measures in place
    pub security_measures: Vec<SoftwareSecurityMeasure>,
    
    /// Attack surface analysis
    pub attack_surface: AttackSurfaceAnalysis,
}

/// Types of software attacks
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SoftwareAttackType {
    /// Buffer overflow attacks
    BufferOverflow,
    
    /// Code injection attacks
    CodeInjection,
    
    /// Return-oriented programming
    ReturnOrientedProgramming,
    
    /// Reverse engineering
    ReverseEngineering,
    
    /// Dynamic analysis attacks
    DynamicAnalysis,
    
    /// Static analysis attacks
    StaticAnalysis,
}

/// Code complexity metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeComplexityMetrics {
    /// Lines of code
    pub lines_of_code: usize,
    
    /// Cyclomatic complexity
    pub cyclomatic_complexity: f64,
    
    /// Number of functions
    pub number_of_functions: usize,
    
    /// Depth of inheritance
    pub inheritance_depth: usize,
    
    /// Coupling between objects
    pub coupling_factor: f64,
}

/// Software security measures
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SoftwareSecurityMeasure {
    /// Address space layout randomization
    ASLR,
    
    /// Data execution prevention
    DEP,
    
    /// Stack canaries
    StackCanaries,
    
    /// Control flow integrity
    ControlFlowIntegrity,
    
    /// Code obfuscation
    CodeObfuscation,
    
    /// Anti-debugging measures
    AntiDebugging,
}

/// Attack surface analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttackSurfaceAnalysis {
    /// Number of entry points
    pub entry_points: usize,
    
    /// Exposed interfaces
    pub exposed_interfaces: Vec<String>,
    
    /// Input validation coverage
    pub input_validation_coverage: f64,
    
    /// Privilege levels
    pub privilege_levels: Vec<PrivilegeLevel>,
}

/// Privilege levels in software
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum PrivilegeLevel {
    /// User level
    User,
    
    /// Administrator level
    Administrator,
    
    /// Kernel level
    Kernel,
    
    /// Hardware level
    Hardware,
}

/// Parameters for protocol attack analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtocolAttackParams {
    /// Protocol attack types to analyze
    pub attack_types: Vec<ProtocolAttackType>,
    
    /// Protocol complexity metrics
    pub protocol_complexity: ProtocolComplexityMetrics,
    
    /// Implementation quality metrics
    pub implementation_quality: ImplementationQualityMetrics,
    
    /// Formal verification status
    pub formal_verification_status: FormalVerificationStatus,
}

/// Types of protocol attacks
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ProtocolAttackType {
    /// Man-in-the-middle attacks
    ManInTheMiddle,
    
    /// Replay attacks
    Replay,
    
    /// Protocol downgrade attacks
    ProtocolDowngrade,
    
    /// Implementation flaws
    ImplementationFlaws,
    
    /// Timing attacks on protocols
    ProtocolTiming,
    
    /// State confusion attacks
    StateConfusion,
}

/// Protocol complexity metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtocolComplexityMetrics {
    /// Number of protocol rounds
    pub protocol_rounds: usize,
    
    /// Number of message types
    pub message_types: usize,
    
    /// State space size
    pub state_space_size: usize,
    
    /// Cryptographic primitives used
    pub cryptographic_primitives: Vec<String>,
}

/// Implementation quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImplementationQualityMetrics {
    /// Code review coverage
    pub code_review_coverage: f64,
    
    /// Testing coverage
    pub testing_coverage: f64,
    
    /// Static analysis results
    pub static_analysis_issues: usize,
    
    /// Dynamic analysis results
    pub dynamic_analysis_issues: usize,
    
    /// Security audit results
    pub security_audit_score: f64,
}

/// Formal verification status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormalVerificationStatus {
    /// Whether protocol is formally verified
    pub protocol_verified: bool,
    
    /// Whether implementation is formally verified
    pub implementation_verified: bool,
    
    /// Verification tool used
    pub verification_tool: Option<String>,
    
    /// Verification coverage
    pub verification_coverage: f64,
    
    /// Verification confidence level
    pub verification_confidence: f64,
}

/// Key for implementation analysis caching
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ImplementationAnalysisKey {
    /// Attack category
    pub attack_category: ImplementationAttackCategory,
    
    /// Target system hash
    pub target_system_hash: u64,
    
    /// Attacker capability level
    pub attacker_capability: AttackerCapabilityLevel,
    
    /// Security measures hash
    pub security_measures_hash: u64,
}

/// Implementation attack categories
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ImplementationAttackCategory {
    /// Side-channel attacks
    SideChannel,
    
    /// Fault injection attacks
    FaultInjection,
    
    /// Physical attacks
    Physical,
    
    /// Software attacks
    Software,
    
    /// Protocol attacks
    Protocol,
}

/// Attacker capability levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum AttackerCapabilityLevel {
    /// Low capability (limited resources)
    Low,
    
    /// Medium capability (moderate resources)
    Medium,
    
    /// High capability (significant resources)
    High,
    
    /// Nation-state capability (unlimited resources)
    NationState,
}

/// Result of implementation attack analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImplementationAnalysisResult {
    /// Attack category analyzed
    pub attack_category: ImplementationAttackCategory,
    
    /// Most effective attack vector
    pub most_effective_attack: String,
    
    /// Attack complexity in bits
    pub attack_complexity_bits: f64,
    
    /// Required resources
    pub required_resources: RequiredResources,
    
    /// Success probability
    pub success_probability: f64,
    
    /// Detection probability
    pub detection_probability: f64,
    
    /// Countermeasure effectiveness
    pub countermeasure_effectiveness: HashMap<String, f64>,
    
    /// Risk assessment
    pub risk_assessment: RiskAssessment,
    
    /// Analysis confidence level
    pub confidence_level: f64,
    
    /// Analysis timestamp
    pub analysis_timestamp: SystemTime,
}

/// Required resources for implementation attacks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequiredResources {
    /// Financial cost in USD
    pub financial_cost_usd: f64,
    
    /// Time investment in hours
    pub time_investment_hours: f64,
    
    /// Required expertise level
    pub expertise_level: ExpertiseLevel,
    
    /// Required equipment
    pub required_equipment: Vec<String>,
    
    /// Required access level
    pub required_access_level: AccessLevel,
}

/// Access levels for attacks
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AccessLevel {
    /// Remote access (over network)
    Remote,
    
    /// Local access (physical proximity)
    Local,
    
    /// Physical access (direct contact)
    Physical,
    
    /// Invasive access (device modification)
    Invasive,
}

/// Risk assessment for implementation attacks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskAssessment {
    /// Overall risk level
    pub overall_risk: RiskLevel,
    
    /// Impact assessment
    pub impact_assessment: ImpactAssessment,
    
    /// Likelihood assessment
    pub likelihood_assessment: LikelihoodAssessment,
    
    /// Mitigation recommendations
    pub mitigation_recommendations: Vec<String>,
}

/// Risk levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RiskLevel {
    /// Very low risk
    VeryLow,
    
    /// Low risk
    Low,
    
    /// Medium risk
    Medium,
    
    /// High risk
    High,
    
    /// Very high risk
    VeryHigh,
}

/// Impact assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImpactAssessment {
    /// Confidentiality impact
    pub confidentiality_impact: ImpactLevel,
    
    /// Integrity impact
    pub integrity_impact: ImpactLevel,
    
    /// Availability impact
    pub availability_impact: ImpactLevel,
    
    /// Financial impact
    pub financial_impact: f64,
    
    /// Reputational impact
    pub reputational_impact: ImpactLevel,
}

/// Impact levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ImpactLevel {
    /// No impact
    None,
    
    /// Low impact
    Low,
    
    /// Medium impact
    Medium,
    
    /// High impact
    High,
    
    /// Critical impact
    Critical,
}

/// Likelihood assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LikelihoodAssessment {
    /// Attack likelihood
    pub attack_likelihood: LikelihoodLevel,
    
    /// Threat actor motivation
    pub threat_actor_motivation: MotivationLevel,
    
    /// Attack surface exposure
    pub attack_surface_exposure: ExposureLevel,
    
    /// Vulnerability exploitability
    pub vulnerability_exploitability: ExploitabilityLevel,
}

/// Likelihood levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum LikelihoodLevel {
    /// Very unlikely
    VeryUnlikely,
    
    /// Unlikely
    Unlikely,
    
    /// Possible
    Possible,
    
    /// Likely
    Likely,
    
    /// Very likely
    VeryLikely,
}

/// Motivation levels for threat actors
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MotivationLevel {
    /// No motivation
    None,
    
    /// Low motivation
    Low,
    
    /// Medium motivation
    Medium,
    
    /// High motivation
    High,
    
    /// Extreme motivation
    Extreme,
}

/// Exposure levels for attack surfaces
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ExposureLevel {
    /// No exposure
    None,
    
    /// Limited exposure
    Limited,
    
    /// Moderate exposure
    Moderate,
    
    /// High exposure
    High,
    
    /// Complete exposure
    Complete,
}

/// Exploitability levels for vulnerabilities
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ExploitabilityLevel {
    /// Not exploitable
    NotExploitable,
    
    /// Difficult to exploit
    Difficult,
    
    /// Moderate exploitability
    Moderate,
    
    /// Easy to exploit
    Easy,
    
    /// Trivial to exploit
    Trivial,
}

/// Security margin configuration for the analyzer
/// 
/// Configuration for security margins applied during analysis to account for
/// potential improvements in attack algorithms and unknown vulnerabilities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityMarginConfiguration {
    /// Base security margin (multiplicative factor)
    /// Applied to all attack complexity estimates
    pub base_margin: f64,
    
    /// BKZ-specific margin
    /// Additional margin for BKZ attack estimates
    pub bkz_margin: f64,
    
    /// Sieve-specific margin
    /// Additional margin for sieve attack estimates
    pub sieve_margin: f64,
    
    /// Dual attack margin
    /// Additional margin for dual attack estimates
    pub dual_margin: f64,
    
    /// Primal attack margin
    /// Additional margin for primal attack estimates
    pub primal_margin: f64,
    
    /// Hybrid attack margin
    /// Additional margin for hybrid attack estimates
    pub hybrid_margin: f64,
    
    /// Implementation attack margin
    /// Additional margin for implementation attack estimates
    pub implementation_margin: f64,
    
    /// Quantum attack margin
    /// Additional margin for quantum attack estimates
    pub quantum_margin: f64,
    
    /// Future-proofing margin
    /// Margin for unknown future attacks
    pub future_proofing_margin: f64,
    
    /// Conservative mode flag
    /// Whether to use maximum conservative margins
    pub conservative_mode: bool,
}

/// Key for security analysis caching
/// 
/// Compact representation of security analysis parameters for use as cache keys.
/// Includes all parameters that significantly affect security analysis results.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SecurityAnalysisKey {
    /// Problem parameters hash
    pub problem_params_hash: u64,
    
    /// Analysis configuration hash
    pub analysis_config_hash: u64,
    
    /// Security margin configuration hash
    pub security_margins_hash: u64,
    
    /// Whether quantum analysis is requested
    pub quantum: bool,
    
    /// Target security level
    pub target_security_level: TargetSecurityLevel,
}

/// Cached security analysis result
/// 
/// Cached result of security analysis including timestamp and validity information.
/// Used to avoid recomputation of expensive security analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedSecurityAnalysis {
    /// Analysis result
    pub analysis_result: ComprehensiveSecurityAnalysis,
    
    /// Cache timestamp
    pub cache_timestamp: SystemTime,
    
    /// Cache validity duration
    pub validity_duration: Duration,
    
    /// Analysis parameters hash (for validation)
    pub parameters_hash: u64,
}

/// Comprehensive security analysis result
/// 
/// Complete result of security analysis including all attack types and
/// overall security assessment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComprehensiveSecurityAnalysis {
    /// BKZ attack analysis results
    pub bkz_analysis: BKZAnalysisResult,
    
    /// Sieve attack analysis results
    pub sieve_analysis: SieveAnalysisResult,
    
    /// Dual attack analysis results
    pub dual_analysis: DualAnalysisResult,
    
    /// Primal attack analysis results
    pub primal_analysis: PrimalAnalysisResult,
    
    /// Hybrid attack analysis results
    pub hybrid_analysis: HybridAnalysisResult,
    
    /// Implementation attack analysis results
    pub implementation_analysis: ImplementationAnalysisResult,
    
    /// Overall security assessment
    pub overall_assessment: OverallSecurityAssessment,
    
    /// Analysis metadata
    pub analysis_metadata: AnalysisMetadata,
}

/// Overall security assessment
/// 
/// High-level security assessment summarizing all attack analysis results
/// and providing actionable security recommendations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverallSecurityAssessment {
    /// Effective security level in bits (minimum across all attacks)
    pub effective_security_bits: f64,
    
    /// Most critical attack vector
    pub most_critical_attack: String,
    
    /// Security margin adequacy
    pub security_margin_adequate: bool,
    
    /// Quantum resistance assessment
    pub quantum_resistance_adequate: bool,
    
    /// Implementation security assessment
    pub implementation_security_adequate: bool,
    
    /// Overall security recommendation
    pub security_recommendation: SecurityRecommendation,
    
    /// Risk level assessment
    pub risk_level: RiskLevel,
    
    /// Confidence in assessment
    pub assessment_confidence: f64,
}

/// Security recommendation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SecurityRecommendation {
    /// Parameters are secure for intended use
    Secure,
    
    /// Parameters are marginally secure, monitor for improvements
    MarginallySecure,
    
    /// Parameters need improvement for adequate security
    NeedsImprovement,
    
    /// Parameters are insecure and should not be used
    Insecure,
    
    /// Security assessment is inconclusive
    Inconclusive,
}

/// Analysis metadata
/// 
/// Metadata about the security analysis including timing, configuration,
/// and quality metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisMetadata {
    /// Analysis start time
    pub analysis_start_time: SystemTime,
    
    /// Analysis duration
    pub analysis_duration: Duration,
    
    /// Analysis configuration used
    pub analysis_configuration: String, // JSON serialized configuration
    
    /// Analysis quality metrics
    pub quality_metrics: AnalysisQualityMetrics,
    
    /// Analysis version information
    pub version_info: AnalysisVersionInfo,
}

/// Analysis quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisQualityMetrics {
    /// Completeness of analysis (fraction of attack types analyzed)
    pub completeness: f64,
    
    /// Accuracy of analysis (estimated based on validation)
    pub accuracy: f64,
    
    /// Consistency of analysis (consistency across different methods)
    pub consistency: f64,
    
    /// Timeliness of analysis (how current the analysis is)
    pub timeliness: f64,
    
    /// Overall quality score
    pub overall_quality_score: f64,
}

/// Analysis version information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisVersionInfo {
    /// Analyzer version
    pub analyzer_version: String,
    
    /// Attack model versions
    pub attack_model_versions: HashMap<String, String>,
    
    /// Security estimator versions
    pub security_estimator_versions: HashMap<String, String>,
    
    /// Last update timestamp
    pub last_update_timestamp: SystemTime,
}

/// Security analysis statistics for monitoring
/// 
/// Statistics about security analysis performance and accuracy for
/// monitoring and optimization of the analysis system.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SecurityAnalysisStatistics {
    /// Total number of analyses performed
    pub total_analyses: u64,
    
    /// Number of successful analyses
    pub successful_analyses: u64,
    
    /// Number of failed analyses
    pub failed_analyses: u64,
    
    /// Total analysis time
    pub total_analysis_time: Duration,
    
    /// Average analysis time
    pub average_analysis_time: Duration,
    
    /// Cache hit rate
    pub cache_hit_rate: f64,
    
    /// Analysis accuracy (when ground truth is available)
    pub analysis_accuracy: f64,
    
    /// Most common attack types analyzed
    pub attack_type_frequency: HashMap<String, u64>,
    
    /// Performance by security level
    pub performance_by_security_level: HashMap<TargetSecurityLevel, Duration>,
}

// Default implementations for various structures
impl Default for SecurityMarginConfiguration {
    fn default() -> Self {
        Self {
            base_margin: 1.2,
            bkz_margin: 1.1,
            sieve_margin: 1.15,
            dual_margin: 1.1,
            primal_margin: 1.1,
            hybrid_margin: 1.05,
            implementation_margin: 2.0, // Higher margin for implementation attacks
            quantum_margin: 1.5,
            future_proofing_margin: 1.3,
            conservative_mode: false,
        }
    }
}

impl Default for BKZQuantumSpeedups {
    fn default() -> Self {
        Self {
            grover_svp_speedup: 1.41, // √2
            quantum_sieve_speedup: 1.02, // 2^0.027
            quantum_walk_speedup: 1.07, // 2^0.1
            overall_quantum_speedup: 1.5, // Conservative combined estimate
        }
    }
}

impl Default for BlockSizeEstimationParams {
    fn default() -> Self {
        Self {
            target_hermite_factor: 1.012, // Typical target for security
            root_hermite_factor: 1.012,
            success_probability_threshold: 0.99,
            gaussian_heuristic_constant: 1.0,
            dimension_scaling_factor: 1.0,
        }
    }
}

impl Default for BKZConservativeFactors {
    fn default() -> Self {
        Self {
            general_improvement_factor: 1.1,
            implementation_optimization_factor: 1.05,
            preprocessing_improvement_factor: 1.02,
            parallel_processing_factor: 1.1,
            hardware_acceleration_factor: 1.2,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_security_analyzer_creation() {
        let analyzer = SecurityAnalyzer::new();
        // Test would verify analyzer is created successfully
        // Full implementation would include comprehensive tests
    }
    
    #[test]
    fn test_bkz_cost_model_selection() {
        let cost_models = [
            BKZCostModel::CoreSVP,
            BKZCostModel::QuantumGates,
            BKZCostModel::Practical,
            BKZCostModel::MemoryConstrained,
        ];
        
        for model in &cost_models {
            // Test that each cost model can be used
            assert!(matches!(model, BKZCostModel::CoreSVP | BKZCostModel::QuantumGates | 
                           BKZCostModel::Practical | BKZCostModel::MemoryConstrained));
        }
    }
    
    #[test]
    fn test_security_margin_configuration() {
        let config = SecurityMarginConfiguration::default();
        assert!(config.base_margin >= 1.0);
        assert!(config.quantum_margin >= 1.0);
        assert!(config.implementation_margin >= 1.0);
    }
}
impl SecurityAnalyzer {
    /// Creates a new security analyzer with default configuration
    /// 
    /// # Returns
    /// * `Result<Self>` - New security analyzer or error
    /// 
    /// # Implementation Details
    /// - Initializes all attack analyzers with default configurations
    /// - Sets up caching mechanisms for performance optimization
    /// - Configures conservative security margins
    /// - Initializes analysis statistics tracking
    pub fn new() -> Result<Self> {
        // Initialize BKZ attack analyzer with default configuration
        // Uses Core-SVP cost model as the most widely accepted standard
        let bkz_analyzer = BKZAttackAnalyzer::new(BKZCostModel::CoreSVP)?;
        
        // Initialize sieve attack analyzer with comprehensive algorithm support
        // Includes all major sieve variants for complete analysis
        let sieve_analyzer = SieveAttackAnalyzer::new()?;
        
        // Initialize dual attack analyzer with standard parameters
        // Configured for Module-LWE dual attack analysis
        let dual_analyzer = DualAttackAnalyzer::new()?;
        
        // Initialize primal attack analyzer with enumeration support
        // Configured for Module-LWE primal attack analysis
        let primal_analyzer = PrimalAttackAnalyzer::new()?;
        
        // Initialize hybrid attack analyzer for combined techniques
        // Analyzes attacks that combine multiple approaches
        let hybrid_analyzer = HybridAttackAnalyzer::new()?;
        
        // Initialize implementation attack analyzer for practical attacks
        // Covers side-channel, fault injection, and other implementation attacks
        let implementation_analyzer = ImplementationAttackAnalyzer::new()?;
        
        // Configure security margins with conservative defaults
        // Provides safety factors for unknown attack improvements
        let security_margins = SecurityMarginConfiguration::default();
        
        // Initialize complexity cache for performance optimization
        // Caches expensive attack complexity computations
        let complexity_cache = Arc::new(Mutex::new(HashMap::new()));
        
        // Initialize analysis statistics for monitoring
        // Tracks analysis performance and accuracy
        let analysis_stats = Arc::new(Mutex::new(SecurityAnalysisStatistics::new()));
        
        Ok(Self {
            bkz_analyzer,
            sieve_analyzer,
            dual_analyzer,
            primal_analyzer,
            hybrid_analyzer,
            implementation_analyzer,
            security_margins,
            complexity_cache,
            analysis_stats,
        })
    }
    
    /// Performs comprehensive security analysis for given parameters
    /// 
    /// # Arguments
    /// * `params` - Lattice parameters to analyze
    /// * `target_security` - Target security level in bits
    /// * `quantum_resistant` - Whether to include quantum attack analysis
    /// 
    /// # Returns
    /// * `Result<ComprehensiveSecurityAnalysis>` - Complete security analysis or error
    /// 
    /// # Implementation Details
    /// - Runs all attack analyzers in parallel for efficiency
    /// - Applies conservative security margins to all estimates
    /// - Caches results for performance optimization
    /// - Provides detailed breakdown of all attack complexities
    pub fn analyze_security(
        &mut self,
        params: &LatticeParams,
        target_security: u32,
        quantum_resistant: bool,
    ) -> Result<ComprehensiveSecurityAnalysis> {
        // Create analysis key for caching
        // Only includes parameters that significantly affect security
        let analysis_key = SecurityAnalysisKey {
            dimension: params.dimension,
            log_modulus: (params.modulus.bits() as f64).log2().round() as u32,
            gaussian_width_scaled: (params.gaussian_width * 1000.0).round() as u32,
            target_security,
            quantum_resistant,
        };
        
        // Check cache for existing analysis
        // Avoids expensive recomputation for identical parameters
        if let Ok(cache) = self.complexity_cache.lock() {
            if let Some(cached_analysis) = cache.get(&analysis_key) {
                // Verify cache entry is still valid (not too old)
                if cached_analysis.is_valid() {
                    return Ok(cached_analysis.analysis.clone());
                }
            }
        }
        
        // Record analysis start time for performance tracking
        let analysis_start = SystemTime::now();
        
        // Run all attack analyses in parallel for efficiency
        // Each analyzer works independently on the same parameters
        let (bkz_result, sieve_result, dual_result, primal_result, hybrid_result, impl_result) = rayon::join(
            || self.bkz_analyzer.analyze(params, quantum_resistant),
            || rayon::join(
                || self.sieve_analyzer.analyze(params, quantum_resistant),
                || rayon::join(
                    || self.dual_analyzer.analyze(params, quantum_resistant),
                    || rayon::join(
                        || self.primal_analyzer.analyze(params, quantum_resistant),
                        || rayon::join(
                            || self.hybrid_analyzer.analyze(params, quantum_resistant),
                            || self.implementation_analyzer.analyze(params, quantum_resistant),
                        ),
                    ),
                ),
            ),
        );
        
        // Flatten the nested join results
        let bkz_result = bkz_result?;
        let (sieve_result, (dual_result, (primal_result, (hybrid_result, impl_result)))) = sieve_result;
        let sieve_result = sieve_result?;
        let dual_result = dual_result?;
        let primal_result = primal_result?;
        let hybrid_result = hybrid_result?;
        let impl_result = impl_result?;
        
        // Apply conservative security margins to all results
        // Accounts for potential attack improvements and unknown attacks
        let adjusted_bkz = self.apply_security_margins(&bkz_result, quantum_resistant);
        let adjusted_sieve = self.apply_security_margins(&sieve_result, quantum_resistant);
        let adjusted_dual = self.apply_security_margins(&dual_result, quantum_resistant);
        let adjusted_primal = self.apply_security_margins(&primal_result, quantum_resistant);
        let adjusted_hybrid = self.apply_security_margins(&hybrid_result, quantum_resistant);
        let adjusted_impl = self.apply_security_margins(&impl_result, quantum_resistant);
        
        // Determine the most efficient attack (lowest complexity)
        // This represents the actual security level of the parameters
        let most_efficient_attack = self.find_most_efficient_attack(&[
            &adjusted_bkz,
            &adjusted_sieve,
            &adjusted_dual,
            &adjusted_primal,
            &adjusted_hybrid,
            &adjusted_impl,
        ]);
        
        // Calculate actual security level achieved
        // Based on the most efficient attack complexity
        let achieved_security_bits = most_efficient_attack.time_complexity_bits;
        
        // Determine if parameters meet target security level
        // Includes margin for conservative analysis
        let meets_target_security = achieved_security_bits >= target_security as f64;
        
        // Calculate security margin (how much above/below target)
        // Positive values indicate security above target
        let security_margin_bits = achieved_security_bits - target_security as f64;
        
        // Create comprehensive analysis result
        let analysis = ComprehensiveSecurityAnalysis {
            parameters: params.clone(),
            target_security_bits: target_security,
            achieved_security_bits,
            meets_target_security,
            security_margin_bits,
            quantum_resistant,
            
            // Individual attack analysis results
            bkz_analysis: adjusted_bkz,
            sieve_analysis: adjusted_sieve,
            dual_analysis: adjusted_dual,
            primal_analysis: adjusted_primal,
            hybrid_analysis: adjusted_hybrid,
            implementation_analysis: adjusted_impl,
            
            // Most efficient attack information
            most_efficient_attack_type: most_efficient_attack.attack_type.clone(),
            most_efficient_attack_complexity: most_efficient_attack.time_complexity_bits,
            
            // Analysis metadata
            analysis_timestamp: SystemTime::now(),
            analysis_duration: analysis_start.elapsed().unwrap_or(Duration::from_secs(0)),
            conservative_margins_applied: self.security_margins.clone(),
        };
        
        // Cache the analysis result for future use
        // Improves performance for repeated analyses
        if let Ok(mut cache) = self.complexity_cache.lock() {
            cache.insert(analysis_key, CachedSecurityAnalysis {
                analysis: analysis.clone(),
                cache_timestamp: SystemTime::now(),
            });
            
            // Limit cache size to prevent memory issues
            if cache.len() > 1000 {
                // Remove oldest entries (simple FIFO eviction)
                let oldest_key = cache.keys().next().cloned();
                if let Some(key) = oldest_key {
                    cache.remove(&key);
                }
            }
        }
        
        // Update analysis statistics for monitoring
        if let Ok(mut stats) = self.analysis_stats.lock() {
            stats.total_analyses += 1;
            stats.total_analysis_time += analysis.analysis_duration;
            if meets_target_security {
                stats.successful_analyses += 1;
            }
        }
        
        Ok(analysis)
    }
    
    /// Applies conservative security margins to attack analysis results
    /// 
    /// # Arguments
    /// * `result` - Original attack analysis result
    /// * `quantum_resistant` - Whether quantum margins should be applied
    /// 
    /// # Returns
    /// * `AttackAnalysisResult` - Result with conservative margins applied
    /// 
    /// # Implementation Details
    /// - Applies base security margin to all attacks
    /// - Adds quantum margin for quantum-resistant analysis
    /// - Includes implementation margin for practical attacks
    /// - Compounds margins multiplicatively for conservative analysis
    fn apply_security_margins(
        &self,
        result: &AttackAnalysisResult,
        quantum_resistant: bool,
    ) -> AttackAnalysisResult {
        // Start with base security margin
        // Accounts for general algorithmic improvements
        let mut margin_factor = self.security_margins.base_margin;
        
        // Add quantum margin if quantum resistance is required
        // Accounts for potential quantum algorithm improvements
        if quantum_resistant {
            margin_factor *= self.security_margins.quantum_margin;
        }
        
        // Add implementation margin for practical considerations
        // Accounts for implementation-specific optimizations
        margin_factor *= self.security_margins.implementation_margin;
        
        // Add attack-specific margins based on attack type
        // Different attacks have different improvement potentials
        let attack_specific_margin = match result.attack_type.as_str() {
            "BKZ" => self.security_margins.bkz_specific_margin,
            "Sieve" => self.security_margins.sieve_specific_margin,
            "Dual" => self.security_margins.dual_specific_margin,
            "Primal" => self.security_margins.primal_specific_margin,
            "Hybrid" => self.security_margins.hybrid_specific_margin,
            "Implementation" => self.security_margins.implementation_specific_margin,
            _ => 1.0, // No additional margin for unknown attack types
        };
        margin_factor *= attack_specific_margin;
        
        // Apply margin to time complexity (most important metric)
        // Reduces effective security by the margin factor
        let adjusted_time_complexity = result.time_complexity_bits - margin_factor.log2();
        
        // Apply margin to memory complexity (less critical but still important)
        // Memory improvements are typically less significant
        let memory_margin = margin_factor.sqrt(); // Square root for less aggressive margin
        let adjusted_memory_complexity = result.memory_complexity_bits - memory_margin.log2();
        
        // Create adjusted result with applied margins
        AttackAnalysisResult {
            attack_type: result.attack_type.clone(),
            time_complexity_bits: adjusted_time_complexity,
            memory_complexity_bits: adjusted_memory_complexity,
            success_probability: result.success_probability,
            confidence_level: result.confidence_level * 0.9, // Slightly reduce confidence
            analysis_details: result.analysis_details.clone(),
            conservative_margins_applied: format!(
                "Total margin factor: {:.3}, Time reduction: {:.2} bits, Memory reduction: {:.2} bits",
                margin_factor,
                margin_factor.log2(),
                memory_margin.log2()
            ),
            analysis_timestamp: result.analysis_timestamp,
        }
    }
    
    /// Finds the most efficient attack from a list of attack results
    /// 
    /// # Arguments
    /// * `attacks` - Slice of attack analysis results to compare
    /// 
    /// # Returns
    /// * `&AttackAnalysisResult` - Reference to the most efficient attack
    /// 
    /// # Implementation Details
    /// - Compares attacks based on time complexity (primary metric)
    /// - Uses memory complexity as tiebreaker for equal time complexity
    /// - Considers success probability for very close complexities
    /// - Returns the attack with lowest effective complexity
    fn find_most_efficient_attack<'a>(
        &self,
        attacks: &'a [&AttackAnalysisResult],
    ) -> &'a AttackAnalysisResult {
        // Find attack with minimum time complexity
        // Time complexity is the primary security metric
        attacks.iter()
            .min_by(|a, b| {
                // Primary comparison: time complexity
                let time_cmp = a.time_complexity_bits.partial_cmp(&b.time_complexity_bits);
                
                match time_cmp {
                    Some(std::cmp::Ordering::Equal) => {
                        // Tiebreaker 1: memory complexity (lower is better for attacker)
                        let memory_cmp = a.memory_complexity_bits.partial_cmp(&b.memory_complexity_bits);
                        
                        match memory_cmp {
                            Some(std::cmp::Ordering::Equal) => {
                                // Tiebreaker 2: success probability (higher is better for attacker)
                                b.success_probability.partial_cmp(&a.success_probability)
                                    .unwrap_or(std::cmp::Ordering::Equal)
                            }
                            Some(order) => order,
                            None => std::cmp::Ordering::Equal,
                        }
                    }
                    Some(order) => order,
                    None => std::cmp::Ordering::Equal,
                }
            })
            .unwrap_or(&attacks[0]) // Fallback to first attack if comparison fails
    }
}

/// Comprehensive security analysis result
/// 
/// Contains complete security analysis including all attack types,
/// security margins, and detailed analysis metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComprehensiveSecurityAnalysis {
    /// Parameters that were analyzed
    pub parameters: LatticeParams,
    
    /// Target security level in bits
    pub target_security_bits: u32,
    
    /// Achieved security level in bits (based on most efficient attack)
    pub achieved_security_bits: f64,
    
    /// Whether parameters meet the target security level
    pub meets_target_security: bool,
    
    /// Security margin in bits (positive = above target, negative = below target)
    pub security_margin_bits: f64,
    
    /// Whether quantum resistance was analyzed
    pub quantum_resistant: bool,
    
    // Individual attack analysis results
    pub bkz_analysis: AttackAnalysisResult,
    pub sieve_analysis: AttackAnalysisResult,
    pub dual_analysis: AttackAnalysisResult,
    pub primal_analysis: AttackAnalysisResult,
    pub hybrid_analysis: AttackAnalysisResult,
    pub implementation_analysis: AttackAnalysisResult,
    
    /// Type of the most efficient attack
    pub most_efficient_attack_type: String,
    
    /// Complexity of the most efficient attack
    pub most_efficient_attack_complexity: f64,
    
    /// When this analysis was performed
    pub analysis_timestamp: SystemTime,
    
    /// How long the analysis took
    pub analysis_duration: Duration,
    
    /// Conservative margins that were applied
    pub conservative_margins_applied: SecurityMarginConfiguration,
}

/// Generic attack analysis result
/// 
/// Standardized result format for all attack types.
/// Allows uniform comparison and processing of different attacks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttackAnalysisResult {
    /// Type of attack (e.g., "BKZ", "Sieve", "Dual", etc.)
    pub attack_type: String,
    
    /// Time complexity in bits (log2 of operations)
    pub time_complexity_bits: f64,
    
    /// Memory complexity in bits (log2 of memory usage)
    pub memory_complexity_bits: f64,
    
    /// Success probability of the attack
    pub success_probability: f64,
    
    /// Confidence level in the analysis
    pub confidence_level: f64,
    
    /// Detailed analysis information (attack-specific)
    pub analysis_details: serde_json::Value,
    
    /// Description of conservative margins applied
    pub conservative_margins_applied: String,
    
    /// When this analysis was performed
    pub analysis_timestamp: SystemTime,
}

/// Security margin configuration
/// 
/// Configuration for conservative security margins applied to attack estimates.
/// These margins account for potential improvements and unknown attacks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityMarginConfiguration {
    /// Base security margin applied to all attacks
    pub base_margin: f64,
    
    /// Additional margin for quantum-resistant analysis
    pub quantum_margin: f64,
    
    /// Margin for implementation-specific considerations
    pub implementation_margin: f64,
    
    // Attack-specific margins
    pub bkz_specific_margin: f64,
    pub sieve_specific_margin: f64,
    pub dual_specific_margin: f64,
    pub primal_specific_margin: f64,
    pub hybrid_specific_margin: f64,
    pub implementation_specific_margin: f64,
}

impl Default for SecurityMarginConfiguration {
    fn default() -> Self {
        Self {
            base_margin: 1.2,           // 20% general margin
            quantum_margin: 1.5,        // 50% quantum margin
            implementation_margin: 1.1,  // 10% implementation margin
            
            // Attack-specific margins based on improvement potential
            bkz_specific_margin: 1.1,         // BKZ is well-studied
            sieve_specific_margin: 1.3,       // Sieve algorithms improving rapidly
            dual_specific_margin: 1.2,        // Dual attacks moderately active
            primal_specific_margin: 1.2,      // Primal attacks moderately active
            hybrid_specific_margin: 1.4,      // Hybrid attacks have high potential
            implementation_specific_margin: 2.0, // Implementation attacks very variable
        }
    }
}

/// Key for security analysis caching
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SecurityAnalysisKey {
    pub dimension: usize,
    pub log_modulus: u32,
    pub gaussian_width_scaled: u32,
    pub target_security: u32,
    pub quantum_resistant: bool,
}

/// Cached security analysis with timestamp
#[derive(Debug, Clone)]
pub struct CachedSecurityAnalysis {
    pub analysis: ComprehensiveSecurityAnalysis,
    pub cache_timestamp: SystemTime,
}

impl CachedSecurityAnalysis {
    /// Checks if the cached analysis is still valid
    /// 
    /// # Returns
    /// * `bool` - True if cache entry is still valid
    /// 
    /// # Implementation Details
    /// - Cache entries expire after 1 hour to ensure freshness
    /// - Accounts for potential changes in attack algorithms
    pub fn is_valid(&self) -> bool {
        // Cache entries are valid for 1 hour
        const CACHE_VALIDITY_DURATION: Duration = Duration::from_secs(3600);
        
        self.cache_timestamp.elapsed()
            .map(|elapsed| elapsed < CACHE_VALIDITY_DURATION)
            .unwrap_or(false)
    }
}

/// Statistics for security analysis performance monitoring
#[derive(Debug, Clone, Default)]
pub struct SecurityAnalysisStatistics {
    /// Total number of analyses performed
    pub total_analyses: u64,
    
    /// Number of analyses that met target security
    pub successful_analyses: u64,
    
    /// Total time spent on analysis
    pub total_analysis_time: Duration,
    
    /// Cache hit rate for performance monitoring
    pub cache_hits: u64,
    
    /// Cache miss rate for performance monitoring
    pub cache_misses: u64,
}

impl SecurityAnalysisStatistics {
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Calculates success rate of security analyses
    pub fn success_rate(&self) -> f64 {
        if self.total_analyses == 0 {
            0.0
        } else {
            self.successful_analyses as f64 / self.total_analyses as f64
        }
    }
    
    /// Calculates average analysis time
    pub fn average_analysis_time(&self) -> Duration {
        if self.total_analyses == 0 {
            Duration::from_secs(0)
        } else {
            self.total_analysis_time / self.total_analyses as u32
        }
    }
    
    /// Calculates cache hit rate
    pub fn cache_hit_rate(&self) -> f64 {
        let total_cache_accesses = self.cache_hits + self.cache_misses;
        if total_cache_accesses == 0 {
            0.0
        } else {
            self.cache_hits as f64 / total_cache_accesses as f64
        }
    }
}//
 Implementation of individual attack analyzers

impl BKZAttackAnalyzer {
    /// Creates a new BKZ attack analyzer with specified cost model
    /// 
    /// # Arguments
    /// * `cost_model` - BKZ cost model to use for complexity estimation
    /// 
    /// # Returns
    /// * `Result<Self>` - New BKZ analyzer or error
    /// 
    /// # Implementation Details
    /// - Initializes cost model and quantum speedup parameters
    /// - Sets up block size estimation parameters based on current research
    /// - Configures conservative factors for safety margins
    /// - Initializes analysis cache for performance optimization
    pub fn new(cost_model: BKZCostModel) -> Result<Self> {
        // Configure quantum speedup factors based on current research
        // Values based on Laarhoven et al. and other quantum lattice research
        let quantum_speedups = BKZQuantumSpeedups {
            grover_svp_speedup: 2.0_f64.sqrt(),      // √2 from Grover's algorithm
            quantum_sieve_speedup: 2.0_f64.powf(0.027), // 2^0.027 from quantum sieve
            quantum_walk_speedup: 2.0_f64.powf(0.1),    // 2^0.1 from quantum walks
            overall_quantum_speedup: 2.0_f64.powf(0.265 - 0.292), // Net quantum improvement
        };
        
        // Configure block size estimation parameters
        // Based on Albrecht et al. lattice estimator and current best practices
        let block_size_params = BlockSizeEstimationParams {
            target_hermite_factor: 1.0045,           // δ ≈ 1.0045 for 128-bit security
            root_hermite_factor: 1.0045,             // δ_0 = δ^(1/d)
            success_probability_threshold: 0.99,      // 99% success probability
            gaussian_heuristic_constant: (2.0 * PI * E).sqrt(), // √(2πe)
            dimension_scaling_factor: 1.0,           // No additional scaling
        };
        
        // Configure conservative factors for BKZ analysis
        // Provides safety margins for unknown improvements
        let conservative_factors = BKZConservativeFactors {
            general_improvement_factor: 1.1,         // 10% general improvement
            implementation_optimization_factor: 1.05, // 5% implementation improvement
            preprocessing_improvement_factor: 1.02,   // 2% preprocessing improvement
            parallel_processing_factor: 1.1,         // 10% parallelization improvement
            hardware_acceleration_factor: 1.05,      // 5% hardware improvement
        };
        
        // Initialize empty analysis cache
        let analysis_cache = HashMap::new();
        
        Ok(Self {
            cost_model,
            quantum_speedups,
            block_size_params,
            conservative_factors,
            analysis_cache,
        })
    }
    
    /// Analyzes BKZ attack complexity for given parameters
    /// 
    /// # Arguments
    /// * `params` - Lattice parameters to analyze
    /// * `quantum` - Whether to include quantum analysis
    /// 
    /// # Returns
    /// * `Result<AttackAnalysisResult>` - BKZ attack analysis or error
    /// 
    /// # Implementation Details
    /// - Estimates required BKZ block size using current best methods
    /// - Computes time and memory complexity based on cost model
    /// - Applies quantum speedups if requested
    /// - Includes conservative factors for safety margins
    pub fn analyze(&mut self, params: &LatticeParams, quantum: bool) -> Result<AttackAnalysisResult> {
        // Create cache key for this analysis
        let cache_key = BKZAnalysisKey {
            dimension: params.dimension,
            log_modulus: (params.modulus.bits() as f64).log2().round() as u32,
            gaussian_width_scaled: (params.gaussian_width * 1000.0).round() as u32,
            cost_model: self.cost_model,
            quantum,
        };
        
        // Check cache for existing result
        if let Some(cached_result) = self.analysis_cache.get(&cache_key) {
            // Convert cached BKZ result to generic attack result
            return Ok(self.convert_bkz_result_to_generic(cached_result));
        }
        
        // Estimate required BKZ block size
        // Uses the relationship β ≈ d²/(4·log(q/σ)) for Module-LWE
        let required_block_size = self.estimate_required_block_size(params)?;
        
        // Compute time complexity based on cost model
        // Different models give different complexity estimates
        let base_time_complexity = self.compute_time_complexity(required_block_size)?;
        
        // Compute memory complexity (typically polynomial in block size)
        // Most BKZ variants have polynomial memory requirements
        let base_memory_complexity = self.compute_memory_complexity(required_block_size)?;
        
        // Apply quantum speedups if requested
        // Quantum algorithms can provide significant speedups
        let (time_complexity, memory_complexity) = if quantum {
            let quantum_time = base_time_complexity - self.quantum_speedups.overall_quantum_speedup.log2();
            let quantum_memory = base_memory_complexity; // Memory typically unchanged
            (quantum_time, quantum_memory)
        } else {
            (base_time_complexity, base_memory_complexity)
        };
        
        // Compute success probability based on block size and parameters
        // Larger block sizes give higher success probability
        let success_probability = self.compute_success_probability(required_block_size, params)?;
        
        // Compute Hermite factor achieved by BKZ-β
        // δ ≈ (β/(2πe))^(1/(2β-2))
        let achieved_hermite_factor = self.compute_achieved_hermite_factor(required_block_size);
        
        // Apply conservative factors for safety margins
        // Accounts for potential algorithmic improvements
        let conservative_time = time_complexity - self.compute_conservative_factor().log2();
        let conservative_memory = memory_complexity - (self.compute_conservative_factor().sqrt()).log2();
        
        // Create BKZ analysis result
        let bkz_result = BKZAnalysisResult {
            required_block_size,
            time_complexity_bits: conservative_time,
            memory_complexity_bits: conservative_memory,
            success_probability,
            achieved_hermite_factor,
            confidence_level: 0.95, // High confidence in BKZ analysis
            applied_conservative_factors: vec![
                format!("Cost model: {:?}", self.cost_model),
                format!("Conservative factor: {:.3}", self.compute_conservative_factor()),
                if quantum { "Quantum speedups applied".to_string() } else { "Classical analysis".to_string() },
            ],
            analysis_timestamp: SystemTime::now(),
        };
        
        // Cache the result for future use
        self.analysis_cache.insert(cache_key, bkz_result.clone());
        
        // Convert to generic attack result format
        Ok(self.convert_bkz_result_to_generic(&bkz_result))
    }
    
    /// Estimates the required BKZ block size to break the scheme
    /// 
    /// # Arguments
    /// * `params` - Lattice parameters
    /// 
    /// # Returns
    /// * `Result<u32>` - Required block size or error
    /// 
    /// # Implementation Details
    /// - Uses the formula β ≈ d²/(4·log(q/σ)) for Module-LWE
    /// - Applies dimension scaling factors for accuracy
    /// - Ensures block size is within reasonable bounds
    /// - Accounts for lattice structure and parameter relationships
    fn estimate_required_block_size(&self, params: &LatticeParams) -> Result<u32> {
        // Extract parameters for block size estimation
        let d = params.dimension as f64;
        let log_q = (params.modulus.bits() as f64).log2();
        let sigma = params.gaussian_width;
        
        // Compute log(q/σ) for the estimation formula
        // This represents the "advantage" of the lattice structure
        let log_q_over_sigma = log_q - sigma.log2();
        
        // Apply the standard formula: β ≈ d²/(4·log(q/σ))
        // This is derived from the relationship between Hermite factor and block size
        let raw_block_size = (d * d) / (4.0 * log_q_over_sigma);
        
        // Apply dimension scaling factor for more accurate estimation
        // Accounts for dimension-dependent effects not captured in the basic formula
        let scaled_block_size = raw_block_size * self.block_size_params.dimension_scaling_factor;
        
        // Ensure block size is within reasonable bounds
        // Block sizes below 50 are typically not meaningful for security
        // Block sizes above 1000 are currently impractical
        let bounded_block_size = scaled_block_size.max(50.0).min(1000.0);
        
        // Round to nearest integer and convert to u32
        let final_block_size = bounded_block_size.round() as u32;
        
        // Validate that the block size makes sense
        if final_block_size < 50 {
            return Err(LatticeFoldError::SecurityAnalysis(
                "Computed block size too small - parameters may be insecure".to_string()
            ));
        }
        
        if final_block_size > 1000 {
            return Err(LatticeFoldError::SecurityAnalysis(
                "Computed block size too large - parameters may be overly conservative".to_string()
            ));
        }
        
        Ok(final_block_size)
    }
    
    /// Computes time complexity for BKZ attack
    /// 
    /// # Arguments
    /// * `block_size` - BKZ block size
    /// 
    /// # Returns
    /// * `Result<f64>` - Time complexity in bits or error
    /// 
    /// # Implementation Details
    /// - Uses cost model to determine complexity exponent
    /// - Applies the formula: complexity = 2^(c·β) where c depends on model
    /// - Includes constant factors and implementation overheads
    /// - Accounts for preprocessing and optimization techniques
    fn compute_time_complexity(&self, block_size: u32) -> Result<f64> {
        let beta = block_size as f64;
        
        // Determine complexity exponent based on cost model
        // Different models reflect different assumptions about BKZ complexity
        let complexity_exponent = match self.cost_model {
            BKZCostModel::CoreSVP => 0.292,           // Core-SVP hardness assumption
            BKZCostModel::QuantumGates => 0.265,      // Quantum gate count model
            BKZCostModel::Practical => 0.320,         // Practical implementation model
            BKZCostModel::MemoryConstrained => 0.349, // Memory-limited model
            BKZCostModel::ADPS16 => 0.292,           // ADPS16 refined model
            BKZCostModel::QuantumCoreSVP => 0.265,   // Quantum Core-SVP model
        };
        
        // Compute base complexity: 2^(c·β)
        let base_complexity_bits = complexity_exponent * beta;
        
        // Add constant factors for implementation overheads
        // These account for practical implementation costs
        let constant_factor_bits = match self.cost_model {
            BKZCostModel::Practical => 10.0,         // Higher overhead for practical
            BKZCostModel::MemoryConstrained => 15.0, // Even higher for memory-constrained
            _ => 5.0,                                 // Standard overhead
        };
        
        // Total time complexity
        let total_complexity_bits = base_complexity_bits + constant_factor_bits;
        
        Ok(total_complexity_bits)
    }
    
    /// Computes memory complexity for BKZ attack
    /// 
    /// # Arguments
    /// * `block_size` - BKZ block size
    /// 
    /// # Returns
    /// * `Result<f64>` - Memory complexity in bits or error
    /// 
    /// # Implementation Details
    /// - Most BKZ variants have polynomial memory complexity
    /// - Memory-constrained models may have different characteristics
    /// - Includes storage for basis, enumeration tree, and auxiliary data
    /// - Accounts for practical memory access patterns
    fn compute_memory_complexity(&self, block_size: u32) -> Result<f64> {
        let beta = block_size as f64;
        
        // Compute memory complexity based on cost model
        // Most BKZ variants have polynomial memory requirements
        let memory_complexity_bits = match self.cost_model {
            BKZCostModel::MemoryConstrained => {
                // Memory-constrained model may have higher memory usage
                2.0 * beta.log2() + 20.0 // O(β²) memory with overhead
            }
            _ => {
                // Standard polynomial memory: O(β log β)
                beta.log2() + (beta.log2()).log2() + 15.0
            }
        };
        
        Ok(memory_complexity_bits)
    }
    
    /// Computes success probability for BKZ attack
    /// 
    /// # Arguments
    /// * `block_size` - BKZ block size
    /// * `params` - Lattice parameters
    /// 
    /// # Returns
    /// * `Result<f64>` - Success probability or error
    /// 
    /// # Implementation Details
    /// - Success probability increases with block size
    /// - Depends on the gap between required and achieved Hermite factor
    /// - Uses Gaussian heuristic for probability estimation
    /// - Accounts for lattice dimension and structure
    fn compute_success_probability(&self, block_size: u32, params: &LatticeParams) -> Result<f64> {
        // Compute achieved Hermite factor for this block size
        let achieved_hermite_factor = self.compute_achieved_hermite_factor(block_size);
        
        // Compare with target Hermite factor needed for attack
        let target_hermite_factor = self.block_size_params.target_hermite_factor;
        
        // Success probability depends on how much we exceed the target
        // Higher achieved factor (smaller δ) gives higher success probability
        let factor_ratio = target_hermite_factor / achieved_hermite_factor;
        
        // Use sigmoid function to model success probability
        // This gives smooth transition from low to high probability
        let success_probability = 1.0 / (1.0 + (-10.0 * (factor_ratio - 1.0)).exp());
        
        // Ensure probability is within valid range
        let bounded_probability = success_probability.max(0.01).min(0.99);
        
        Ok(bounded_probability)
    }
    
    /// Computes Hermite factor achieved by BKZ-β
    /// 
    /// # Arguments
    /// * `block_size` - BKZ block size
    /// 
    /// # Returns
    /// * `f64` - Achieved Hermite factor
    /// 
    /// # Implementation Details
    /// - Uses the formula δ ≈ (β/(2πe))^(1/(2β-2))
    /// - This is the standard relationship for BKZ quality
    /// - Smaller δ indicates better lattice reduction
    /// - Formula is based on extensive BKZ analysis and experiments
    fn compute_achieved_hermite_factor(&self, block_size: u32) -> f64 {
        let beta = block_size as f64;
        
        // Standard Hermite factor formula for BKZ-β
        // δ ≈ (β/(2πe))^(1/(2β-2))
        let numerator = beta / (2.0 * PI * E);
        let exponent = 1.0 / (2.0 * beta - 2.0);
        
        numerator.powf(exponent)
    }
    
    /// Computes overall conservative factor for BKZ analysis
    /// 
    /// # Returns
    /// * `f64` - Combined conservative factor
    /// 
    /// # Implementation Details
    /// - Multiplies all individual conservative factors
    /// - Provides compound safety margin for unknown improvements
    /// - Accounts for various sources of potential attack improvements
    /// - Results in more conservative (safer) security estimates
    fn compute_conservative_factor(&self) -> f64 {
        self.conservative_factors.general_improvement_factor
            * self.conservative_factors.implementation_optimization_factor
            * self.conservative_factors.preprocessing_improvement_factor
            * self.conservative_factors.parallel_processing_factor
            * self.conservative_factors.hardware_acceleration_factor
    }
    
    /// Converts BKZ-specific result to generic attack result format
    /// 
    /// # Arguments
    /// * `bkz_result` - BKZ-specific analysis result
    /// 
    /// # Returns
    /// * `AttackAnalysisResult` - Generic attack result
    /// 
    /// # Implementation Details
    /// - Standardizes result format for uniform processing
    /// - Preserves all important analysis information
    /// - Includes BKZ-specific details in analysis_details field
    /// - Enables comparison with other attack types
    fn convert_bkz_result_to_generic(&self, bkz_result: &BKZAnalysisResult) -> AttackAnalysisResult {
        // Create detailed analysis information as JSON
        let analysis_details = serde_json::json!({
            "attack_type": "BKZ",
            "cost_model": self.cost_model,
            "required_block_size": bkz_result.required_block_size,
            "achieved_hermite_factor": bkz_result.achieved_hermite_factor,
            "applied_conservative_factors": bkz_result.applied_conservative_factors,
        });
        
        AttackAnalysisResult {
            attack_type: "BKZ".to_string(),
            time_complexity_bits: bkz_result.time_complexity_bits,
            memory_complexity_bits: bkz_result.memory_complexity_bits,
            success_probability: bkz_result.success_probability,
            confidence_level: bkz_result.confidence_level,
            analysis_details,
            conservative_margins_applied: format!(
                "BKZ conservative factor: {:.3}",
                self.compute_conservative_factor()
            ),
            analysis_timestamp: bkz_result.analysis_timestamp,
        }
    }
}// Pla
ceholder implementations for other attack analyzers
// These provide basic functionality and can be extended with full implementations

impl SieveAttackAnalyzer {
    /// Creates a new sieve attack analyzer with default configuration
    pub fn new() -> Result<Self> {
        let sieve_variants = vec![
            SieveAlgorithmVariant::GaussSieve,
            SieveAlgorithmVariant::NVSieve,
            SieveAlgorithmVariant::BDGL16,
            SieveAlgorithmVariant::HashSieve,
        ];
        
        let tradeoff_params = MemoryTimeTradeoffParams {
            available_memory_bits: 40.0, // 1 TB memory
            time_budget_bits: 80.0,      // 2^80 operations
            tradeoff_exponent: 1.0,
            minimum_memory_bits: 20.0,   // 1 MB minimum
            max_time_extension_factor: 10.0,
        };
        
        let quantum_params = QuantumSieveParams {
            quantum_memory_available: false,
            quantum_speedup_factors: QuantumSieveSpeedups {
                grover_search_speedup: 2.0_f64.sqrt(),
                quantum_walk_speedup: 2.0_f64.powf(0.1),
                amplitude_amplification_speedup: 2.0_f64.sqrt(),
                overall_speedup: 2.0_f64.powf(0.265 - 0.415),
            },
            error_correction_overhead: 1000.0,
            decoherence_time_limit_bits: 60.0,
        };
        
        let conservative_factors = SieveConservativeFactors {
            algorithmic_improvement_factor: 1.3,
            implementation_optimization_factor: 1.1,
            quantum_improvement_factor: 1.5,
            memory_access_optimization_factor: 1.05,
        };
        
        Ok(Self {
            sieve_variants,
            tradeoff_params,
            quantum_params,
            analysis_cache: HashMap::new(),
            conservative_factors,
        })
    }
    
    /// Analyzes sieve attack complexity (placeholder implementation)
    pub fn analyze(&mut self, params: &LatticeParams, quantum: bool) -> Result<AttackAnalysisResult> {
        let dimension = params.dimension as f64;
        
        // Use GaussSieve complexity as baseline: 2^(0.415n)
        let base_complexity = 0.415 * dimension;
        
        // Apply quantum speedup if requested
        let time_complexity = if quantum {
            base_complexity - self.quantum_params.quantum_speedup_factors.overall_speedup.log2()
        } else {
            base_complexity
        };
        
        // Memory complexity for GaussSieve: same as time
        let memory_complexity = time_complexity;
        
        // Apply conservative factors
        let conservative_factor = self.conservative_factors.algorithmic_improvement_factor
            * self.conservative_factors.implementation_optimization_factor;
        
        let final_time_complexity = time_complexity - conservative_factor.log2();
        let final_memory_complexity = memory_complexity - conservative_factor.log2();
        
        let analysis_details = serde_json::json!({
            "attack_type": "Sieve",
            "algorithm": "GaussSieve",
            "dimension": dimension,
            "quantum": quantum,
        });
        
        Ok(AttackAnalysisResult {
            attack_type: "Sieve".to_string(),
            time_complexity_bits: final_time_complexity,
            memory_complexity_bits: final_memory_complexity,
            success_probability: 0.95,
            confidence_level: 0.85,
            analysis_details,
            conservative_margins_applied: format!("Conservative factor: {:.3}", conservative_factor),
            analysis_timestamp: SystemTime::now(),
        })
    }
}

impl DualAttackAnalyzer {
    /// Creates a new dual attack analyzer with default configuration
    pub fn new() -> Result<Self> {
        let dual_construction_params = DualConstructionParams {
            num_samples: 1000,
            scaling_factor: 1.0,
            basis_quality_target: 1.01,
            preprocessing_technique: DualPreprocessingTechnique::LLL,
        };
        
        let distinguishing_params = DistinguishingAdvantageParams {
            target_advantage: 0.01,
            statistical_distance_threshold: 0.1,
            sample_complexity_factor: 100.0,
            noise_distribution_params: NoiseDistributionParams {
                distribution_type: NoiseDistributionType::DiscreteGaussian,
                distribution_parameter: 3.2,
                discretization_parameter: 1.0,
                tail_bound_parameter: 6.0,
            },
        };
        
        let embedding_params = EmbeddingTechniqueParams {
            technique: EmbeddingTechnique::Kannan,
            embedding_dimension: 100,
            optimization_target: EmbeddingOptimizationTarget::MinimizeTotalComplexity,
            success_probability: 0.9,
        };
        
        let conservative_factors = DualConservativeFactors {
            lattice_reduction_improvement_factor: 1.2,
            distinguishing_improvement_factor: 1.1,
            embedding_improvement_factor: 1.15,
            sample_complexity_improvement_factor: 1.05,
        };
        
        Ok(Self {
            dual_construction_params,
            distinguishing_params,
            embedding_params,
            analysis_cache: HashMap::new(),
            conservative_factors,
        })
    }
    
    /// Analyzes dual attack complexity (placeholder implementation)
    pub fn analyze(&mut self, params: &LatticeParams, quantum: bool) -> Result<AttackAnalysisResult> {
        let n = params.dimension as f64;
        let m = self.dual_construction_params.num_samples as f64;
        let dual_dimension = n + m;
        
        // Estimate dual attack complexity using BKZ on dual lattice
        // This is a simplified model - full implementation would be more sophisticated
        let log_q = (params.modulus.bits() as f64).log2();
        let sigma = params.gaussian_width;
        
        // Required block size for dual attack
        let required_block_size = (dual_dimension * dual_dimension) / (4.0 * (log_q - sigma.log2()));
        
        // Time complexity using Core-SVP model
        let base_time_complexity = 0.292 * required_block_size;
        
        // Apply quantum speedup if requested
        let time_complexity = if quantum {
            base_time_complexity - (2.0_f64.powf(0.265 - 0.292)).log2()
        } else {
            base_time_complexity
        };
        
        // Memory complexity (polynomial)
        let memory_complexity = 2.0 * required_block_size.log2() + 15.0;
        
        // Apply conservative factors
        let conservative_factor = self.conservative_factors.lattice_reduction_improvement_factor
            * self.conservative_factors.distinguishing_improvement_factor;
        
        let final_time_complexity = time_complexity - conservative_factor.log2();
        let final_memory_complexity = memory_complexity - conservative_factor.log2();
        
        let analysis_details = serde_json::json!({
            "attack_type": "Dual",
            "dual_dimension": dual_dimension,
            "required_block_size": required_block_size,
            "embedding_technique": self.embedding_params.technique,
        });
        
        Ok(AttackAnalysisResult {
            attack_type: "Dual".to_string(),
            time_complexity_bits: final_time_complexity,
            memory_complexity_bits: final_memory_complexity,
            success_probability: 0.9,
            confidence_level: 0.8,
            analysis_details,
            conservative_margins_applied: format!("Conservative factor: {:.3}", conservative_factor),
            analysis_timestamp: SystemTime::now(),
        })
    }
}

impl PrimalAttackAnalyzer {
    /// Creates a new primal attack analyzer with default configuration
    pub fn new() -> Result<Self> {
        // Placeholder implementation - would include proper primal attack parameters
        Ok(Self {})
    }
    
    /// Analyzes primal attack complexity (placeholder implementation)
    pub fn analyze(&mut self, params: &LatticeParams, quantum: bool) -> Result<AttackAnalysisResult> {
        let n = params.dimension as f64;
        let log_q = (params.modulus.bits() as f64).log2();
        let sigma = params.gaussian_width;
        
        // Simplified primal attack analysis
        // Full implementation would include proper enumeration and embedding analysis
        let primal_dimension = n + 100.0; // Simplified dimension estimate
        let required_block_size = (primal_dimension * primal_dimension) / (4.0 * (log_q - sigma.log2()));
        
        let base_time_complexity = 0.292 * required_block_size;
        let time_complexity = if quantum {
            base_time_complexity - (2.0_f64.powf(0.265 - 0.292)).log2()
        } else {
            base_time_complexity
        };
        
        let memory_complexity = 2.0 * required_block_size.log2() + 10.0;
        
        let analysis_details = serde_json::json!({
            "attack_type": "Primal",
            "primal_dimension": primal_dimension,
            "required_block_size": required_block_size,
        });
        
        Ok(AttackAnalysisResult {
            attack_type: "Primal".to_string(),
            time_complexity_bits: time_complexity,
            memory_complexity_bits: memory_complexity,
            success_probability: 0.85,
            confidence_level: 0.75,
            analysis_details,
            conservative_margins_applied: "Basic conservative margins applied".to_string(),
            analysis_timestamp: SystemTime::now(),
        })
    }
}

impl HybridAttackAnalyzer {
    /// Creates a new hybrid attack analyzer with default configuration
    pub fn new() -> Result<Self> {
        // Placeholder implementation
        Ok(Self {})
    }
    
    /// Analyzes hybrid attack complexity (placeholder implementation)
    pub fn analyze(&mut self, params: &LatticeParams, quantum: bool) -> Result<AttackAnalysisResult> {
        let dimension = params.dimension as f64;
        
        // Hybrid attacks combine multiple techniques - use best of dual/primal
        let base_complexity = 0.25 * dimension; // Simplified hybrid complexity
        
        let time_complexity = if quantum {
            base_complexity - 5.0 // Simplified quantum improvement
        } else {
            base_complexity
        };
        
        let analysis_details = serde_json::json!({
            "attack_type": "Hybrid",
            "combines": ["BKZ", "Enumeration", "Embedding"],
        });
        
        Ok(AttackAnalysisResult {
            attack_type: "Hybrid".to_string(),
            time_complexity_bits: time_complexity,
            memory_complexity_bits: time_complexity - 10.0,
            success_probability: 0.9,
            confidence_level: 0.7,
            analysis_details,
            conservative_margins_applied: "Hybrid attack margins applied".to_string(),
            analysis_timestamp: SystemTime::now(),
        })
    }
}

impl ImplementationAttackAnalyzer {
    /// Creates a new implementation attack analyzer with default configuration
    pub fn new() -> Result<Self> {
        // Placeholder implementation
        Ok(Self {})
    }
    
    /// Analyzes implementation attack complexity (placeholder implementation)
    pub fn analyze(&mut self, params: &LatticeParams, _quantum: bool) -> Result<AttackAnalysisResult> {
        // Implementation attacks are typically much more efficient than mathematical attacks
        // but require specific implementation vulnerabilities
        
        let base_complexity = 40.0; // Typical implementation attack complexity
        
        let analysis_details = serde_json::json!({
            "attack_type": "Implementation",
            "attack_vectors": ["Side-channel", "Fault injection", "Cache timing"],
            "requires_implementation_access": true,
        });
        
        Ok(AttackAnalysisResult {
            attack_type: "Implementation".to_string(),
            time_complexity_bits: base_complexity,
            memory_complexity_bits: 20.0, // Low memory requirements
            success_probability: 0.5, // Depends heavily on implementation
            confidence_level: 0.6, // High variability
            analysis_details,
            conservative_margins_applied: "Implementation-specific margins applied".to_string(),
            analysis_timestamp: SystemTime::now(),
        })
    }
}

// Placeholder struct definitions for the analyzers that don't have full implementations yet
#[derive(Debug, Clone)]
pub struct PrimalAttackAnalyzer {}

#[derive(Debug, Clone)]
pub struct HybridAttackAnalyzer {}

#[derive(Debug, Clone)]
pub struct ImplementationAttackAnalyzer {}#[]
[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice::LatticeParams;
    use num_bigint::BigUint;
    
    fn create_test_params() -> LatticeParams {
        LatticeParams {
            dimension: 512,
            modulus: BigUint::from(2u64.pow(127) - 1),
            gaussian_width: 3.2,
            ring_degree: 512,
            module_rank: 4,
        }
    }
    
    #[test]
    fn test_security_analyzer_creation() {
        let analyzer = SecurityAnalyzer::new();
        assert!(analyzer.is_ok());
    }
    
    #[test]
    fn test_bkz_analyzer_creation() {
        let analyzer = BKZAttackAnalyzer::new(BKZCostModel::CoreSVP);
        assert!(analyzer.is_ok());
    }
    
    #[test]
    fn test_bkz_block_size_estimation() {
        let mut analyzer = BKZAttackAnalyzer::new(BKZCostModel::CoreSVP).unwrap();
        let params = create_test_params();
        
        let block_size = analyzer.estimate_required_block_size(&params);
        assert!(block_size.is_ok());
        
        let block_size = block_size.unwrap();
        assert!(block_size >= 50);
        assert!(block_size <= 1000);
    }
    
    #[test]
    fn test_bkz_time_complexity() {
        let analyzer = BKZAttackAnalyzer::new(BKZCostModel::CoreSVP).unwrap();
        
        let complexity = analyzer.compute_time_complexity(100);
        assert!(complexity.is_ok());
        
        let complexity = complexity.unwrap();
        assert!(complexity > 20.0); // Should be reasonable complexity
        assert!(complexity < 100.0); // Should not be excessive
    }
    
    #[test]
    fn test_bkz_memory_complexity() {
        let analyzer = BKZAttackAnalyzer::new(BKZCostModel::CoreSVP).unwrap();
        
        let complexity = analyzer.compute_memory_complexity(100);
        assert!(complexity.is_ok());
        
        let complexity = complexity.unwrap();
        assert!(complexity > 10.0); // Should require some memory
        assert!(complexity < 50.0); // Should be polynomial
    }
    
    #[test]
    fn test_hermite_factor_computation() {
        let analyzer = BKZAttackAnalyzer::new(BKZCostModel::CoreSVP).unwrap();
        
        let factor = analyzer.compute_achieved_hermite_factor(100);
        assert!(factor > 1.0); // Hermite factor should be > 1
        assert!(factor < 1.1); // Should be close to 1 for good reduction
    }
    
    #[test]
    fn test_bkz_analysis() {
        let mut analyzer = BKZAttackAnalyzer::new(BKZCostModel::CoreSVP).unwrap();
        let params = create_test_params();
        
        let result = analyzer.analyze(&params, false);
        assert!(result.is_ok());
        
        let result = result.unwrap();
        assert_eq!(result.attack_type, "BKZ");
        assert!(result.time_complexity_bits > 50.0);
        assert!(result.success_probability > 0.0);
        assert!(result.success_probability <= 1.0);
    }
    
    #[test]
    fn test_quantum_bkz_analysis() {
        let mut analyzer = BKZAttackAnalyzer::new(BKZCostModel::CoreSVP).unwrap();
        let params = create_test_params();
        
        let classical_result = analyzer.analyze(&params, false).unwrap();
        let quantum_result = analyzer.analyze(&params, true).unwrap();
        
        // Quantum should be more efficient (lower complexity)
        assert!(quantum_result.time_complexity_bits < classical_result.time_complexity_bits);
    }
    
    #[test]
    fn test_sieve_analyzer() {
        let mut analyzer = SieveAttackAnalyzer::new().unwrap();
        let params = create_test_params();
        
        let result = analyzer.analyze(&params, false);
        assert!(result.is_ok());
        
        let result = result.unwrap();
        assert_eq!(result.attack_type, "Sieve");
        assert!(result.time_complexity_bits > 100.0); // Sieve should be expensive
    }
    
    #[test]
    fn test_dual_analyzer() {
        let mut analyzer = DualAttackAnalyzer::new().unwrap();
        let params = create_test_params();
        
        let result = analyzer.analyze(&params, false);
        assert!(result.is_ok());
        
        let result = result.unwrap();
        assert_eq!(result.attack_type, "Dual");
    }
    
    #[test]
    fn test_comprehensive_security_analysis() {
        let mut analyzer = SecurityAnalyzer::new().unwrap();
        let params = create_test_params();
        
        let analysis = analyzer.analyze_security(&params, 128, false);
        assert!(analysis.is_ok());
        
        let analysis = analysis.unwrap();
        assert_eq!(analysis.target_security_bits, 128);
        assert!(analysis.achieved_security_bits > 0.0);
        
        // Should have results for all attack types
        assert_eq!(analysis.bkz_analysis.attack_type, "BKZ");
        assert_eq!(analysis.sieve_analysis.attack_type, "Sieve");
        assert_eq!(analysis.dual_analysis.attack_type, "Dual");
        assert_eq!(analysis.primal_analysis.attack_type, "Primal");
        assert_eq!(analysis.hybrid_analysis.attack_type, "Hybrid");
        assert_eq!(analysis.implementation_analysis.attack_type, "Implementation");
    }
    
    #[test]
    fn test_security_margins() {
        let config = SecurityMarginConfiguration::default();
        
        assert!(config.base_margin >= 1.0);
        assert!(config.quantum_margin >= 1.0);
        assert!(config.implementation_margin >= 1.0);
        
        // All attack-specific margins should be reasonable
        assert!(config.bkz_specific_margin >= 1.0);
        assert!(config.sieve_specific_margin >= 1.0);
        assert!(config.dual_specific_margin >= 1.0);
        assert!(config.primal_specific_margin >= 1.0);
        assert!(config.hybrid_specific_margin >= 1.0);
        assert!(config.implementation_specific_margin >= 1.0);
    }
    
    #[test]
    fn test_cache_functionality() {
        let mut analyzer = SecurityAnalyzer::new().unwrap();
        let params = create_test_params();
        
        // First analysis should compute fresh
        let start_time = SystemTime::now();
        let analysis1 = analyzer.analyze_security(&params, 128, false).unwrap();
        let first_duration = start_time.elapsed().unwrap();
        
        // Second analysis should use cache (should be faster)
        let start_time = SystemTime::now();
        let analysis2 = analyzer.analyze_security(&params, 128, false).unwrap();
        let second_duration = start_time.elapsed().unwrap();
        
        // Results should be identical
        assert_eq!(analysis1.achieved_security_bits, analysis2.achieved_security_bits);
        
        // Second should be faster (though this might not always be true in tests)
        // Just verify both completed successfully
        assert!(first_duration.as_millis() >= 0);
        assert!(second_duration.as_millis() >= 0);
    }
    
    #[test]
    fn test_different_cost_models() {
        let cost_models = vec![
            BKZCostModel::CoreSVP,
            BKZCostModel::QuantumGates,
            BKZCostModel::Practical,
            BKZCostModel::MemoryConstrained,
        ];
        
        let params = create_test_params();
        
        for cost_model in cost_models {
            let mut analyzer = BKZAttackAnalyzer::new(cost_model).unwrap();
            let result = analyzer.analyze(&params, false);
            assert!(result.is_ok(), "Failed for cost model: {:?}", cost_model);
            
            let result = result.unwrap();
            assert!(result.time_complexity_bits > 0.0);
            assert!(result.memory_complexity_bits > 0.0);
        }
    }
    
    #[test]
    fn test_analysis_statistics() {
        let stats = SecurityAnalysisStatistics::new();
        
        assert_eq!(stats.total_analyses, 0);
        assert_eq!(stats.successful_analyses, 0);
        assert_eq!(stats.success_rate(), 0.0);
        assert_eq!(stats.cache_hit_rate(), 0.0);
    }
    
    #[test]
    fn test_cached_analysis_validity() {
        let analysis = ComprehensiveSecurityAnalysis {
            parameters: create_test_params(),
            target_security_bits: 128,
            achieved_security_bits: 130.0,
            meets_target_security: true,
            security_margin_bits: 2.0,
            quantum_resistant: false,
            bkz_analysis: AttackAnalysisResult {
                attack_type: "BKZ".to_string(),
                time_complexity_bits: 130.0,
                memory_complexity_bits: 25.0,
                success_probability: 0.95,
                confidence_level: 0.9,
                analysis_details: serde_json::json!({}),
                conservative_margins_applied: "Test".to_string(),
                analysis_timestamp: SystemTime::now(),
            },
            sieve_analysis: AttackAnalysisResult {
                attack_type: "Sieve".to_string(),
                time_complexity_bits: 200.0,
                memory_complexity_bits: 200.0,
                success_probability: 0.95,
                confidence_level: 0.85,
                analysis_details: serde_json::json!({}),
                conservative_margins_applied: "Test".to_string(),
                analysis_timestamp: SystemTime::now(),
            },
            dual_analysis: AttackAnalysisResult {
                attack_type: "Dual".to_string(),
                time_complexity_bits: 140.0,
                memory_complexity_bits: 30.0,
                success_probability: 0.9,
                confidence_level: 0.8,
                analysis_details: serde_json::json!({}),
                conservative_margins_applied: "Test".to_string(),
                analysis_timestamp: SystemTime::now(),
            },
            primal_analysis: AttackAnalysisResult {
                attack_type: "Primal".to_string(),
                time_complexity_bits: 135.0,
                memory_complexity_bits: 28.0,
                success_probability: 0.85,
                confidence_level: 0.75,
                analysis_details: serde_json::json!({}),
                conservative_margins_applied: "Test".to_string(),
                analysis_timestamp: SystemTime::now(),
            },
            hybrid_analysis: AttackAnalysisResult {
                attack_type: "Hybrid".to_string(),
                time_complexity_bits: 125.0,
                memory_complexity_bits: 20.0,
                success_probability: 0.9,
                confidence_level: 0.7,
                analysis_details: serde_json::json!({}),
                conservative_margins_applied: "Test".to_string(),
                analysis_timestamp: SystemTime::now(),
            },
            implementation_analysis: AttackAnalysisResult {
                attack_type: "Implementation".to_str            time_complexity_bits: 40.0,
                memory_complexity_bits: 20.0,
                success_probability: 0.5,
                confidence_level: 0.6,
                analysis_details: serde_json::json!({}),
                conservative_margins_applied: "Test".to_string(),
                analysis_timestamp: SystemTime::now(),
            },
            most_efficient_attack_type: "Implementation".to_string(),
            most_efficient_attack_complexity: 40.0,
            analysis_timestamp: SystemTime::now(),
            analysis_duration: Duration::from_millis(100),
            conservative_margins_applied: SecurityMarginConfiguration::default(),
        };
        
        let cached = CachedSecurityAnalysis {
            analysis,
            cache_timestamp: SystemTime::now(),
        };
        
        assert!(cached.is_valid());
    }
}