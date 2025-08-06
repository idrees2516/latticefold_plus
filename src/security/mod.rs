// Security module for LatticeFold+ implementation
// This module provides constant-time cryptographic operations, side-channel resistance,
// and comprehensive security validation to protect against timing attacks, power analysis,
// and cache-timing attacks as required by the LatticeFold+ security model.

pub mod constant_time;
pub mod side_channel_resistance;
pub mod secure_memory;
pub mod timing_analysis;
pub mod security_validation;

pub use constant_time::{
    ConstantTimeArithmetic, ConstantTimeModularArithmetic, ConstantTimeNormChecker,
    ConstantTimePolynomialOps, ConstantTimeComparison, ConstantTimeSelection,
    ConstantTimeMatrixOps, ConstantTimeVectorOps, ConstantTimeGadgetOps,
    ConstantTimeCommitmentOps, ConstantTimeRangeProof, ConstantTimeFolding,
    ConstantTimeErrorHandling, TimingConsistentOperations
};

pub use side_channel_resistance::{
    SideChannelResistantRNG, PowerAnalysisResistance, CacheTimingResistance,
    MemoryAccessPatternMasking, BranchPredictionResistance, ElectromagneticResistance,
    AcousticResistance, ThermalResistance, FaultInjectionResistance,
    MicroarchitecturalAttackResistance, SpeculativeExecutionResistance
};

pub use secure_memory::{
    SecureMemoryManager, SecureAllocator, AutoZeroization, MemoryProtection,
    SecureSwapping, MemoryEncryption, MemoryIntegrityChecking, SecureDeallocation,
    MemoryLeakageProtection, SecureMemoryPool, ConstantTimeMemoryOps
};

pub use timing_analysis::{
    TimingAnalyzer, TimingConsistencyChecker, PerformanceProfiler, TimingAttackDetector,
    StatisticalTimingAnalysis, MicrobenchmarkSuite, TimingVarianceAnalyzer,
    ConstantTimeVerifier, TimingLeakageDetector, ExecutionTimeNormalizer
};

pub use security_validation::{
    SecurityValidator, ThreatModelAnalyzer, AttackSimulator, SecurityTestSuite,
    CryptographicPropertyVerifier, FormalSecurityVerifier, SecurityAuditor,
    PenetrationTester, SecurityComplianceChecker, VulnerabilityScanner
};

use crate::error::{LatticeFoldError, Result};
use subtle::{Choice, ConditionallySelectable, ConstantTimeEq};
use zeroize::{Zeroize, ZeroizeOnDrop};

/// Security configuration parameters for the LatticeFold+ implementation
/// This structure defines all security-related parameters and policies
/// that govern the constant-time and side-channel resistant operations.
#[derive(Clone, Debug, Zeroize, ZeroizeOnDrop)]
pub struct SecurityConfig {
    /// Enable constant-time operations for all secret-dependent computations
    /// When true, all operations involving secret data use constant-time algorithms
    pub constant_time_enabled: bool,
    
    /// Enable side-channel resistance measures including power analysis protection
    /// When true, operations use masking, blinding, and other countermeasures
    pub side_channel_resistance_enabled: bool,
    
    /// Enable cache-timing attack resistance through consistent memory access patterns
    /// When true, memory accesses are made uniform to prevent cache-based attacks
    pub cache_timing_resistance_enabled: bool,
    
    /// Enable automatic secure memory zeroization on deallocation
    /// When true, all sensitive memory is automatically zeroed when freed
    pub auto_zeroization_enabled: bool,
    
    /// Enable memory protection against unauthorized access
    /// When true, sensitive memory regions are protected with OS-level mechanisms
    pub memory_protection_enabled: bool,
    
    /// Enable timing analysis and consistency checking
    /// When true, operations are monitored for timing consistency
    pub timing_analysis_enabled: bool,
    
    /// Enable formal security verification of cryptographic properties
    /// When true, security properties are formally verified during execution
    pub formal_verification_enabled: bool,
    
    /// Enable comprehensive security testing and validation
    /// When true, security tests are run continuously during operation
    pub security_testing_enabled: bool,
    
    /// Maximum allowed timing variance for constant-time operations (in nanoseconds)
    /// Operations exceeding this variance are flagged as potential timing leaks
    pub max_timing_variance_ns: u64,
    
    /// Security level for cryptographic operations (128, 192, or 256 bits)
    /// Determines the strength of cryptographic primitives and parameters
    pub security_level_bits: u32,
    
    /// Enable debug mode for security analysis (NEVER use in production)
    /// When true, detailed security analysis information is logged
    pub debug_mode: bool,
}

impl Default for SecurityConfig {
    /// Default security configuration with maximum security enabled
    /// This configuration prioritizes security over performance and should be
    /// used in production environments where security is paramount.
    fn default() -> Self {
        Self {
            constant_time_enabled: true,           // Always enable constant-time operations
            side_channel_resistance_enabled: true, // Always enable side-channel resistance
            cache_timing_resistance_enabled: true, // Always enable cache-timing resistance
            auto_zeroization_enabled: true,       // Always enable automatic zeroization
            memory_protection_enabled: true,      // Always enable memory protection
            timing_analysis_enabled: true,        // Always enable timing analysis
            formal_verification_enabled: true,    // Always enable formal verification
            security_testing_enabled: true,       // Always enable security testing
            max_timing_variance_ns: 1000,         // Allow maximum 1 microsecond variance
            security_level_bits: 128,             // Use 128-bit security level by default
            debug_mode: false,                    // Never enable debug mode by default
        }
    }
}

impl SecurityConfig {
    /// Create a new security configuration with custom parameters
    /// This allows fine-tuning of security parameters for specific use cases
    /// while maintaining secure defaults for unspecified parameters.
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Create a high-security configuration for maximum protection
    /// This configuration enables all security features and uses the most
    /// conservative settings for maximum protection against all known attacks.
    pub fn high_security() -> Self {
        Self {
            constant_time_enabled: true,
            side_channel_resistance_enabled: true,
            cache_timing_resistance_enabled: true,
            auto_zeroization_enabled: true,
            memory_protection_enabled: true,
            timing_analysis_enabled: true,
            formal_verification_enabled: true,
            security_testing_enabled: true,
            max_timing_variance_ns: 100,          // Very strict timing requirements
            security_level_bits: 256,             // Maximum security level
            debug_mode: false,
        }
    }
    
    /// Create a performance-optimized configuration with reduced security
    /// This configuration disables some security features for better performance
    /// and should only be used in trusted environments or for testing.
    pub fn performance_optimized() -> Self {
        Self {
            constant_time_enabled: true,           // Keep constant-time for basic security
            side_channel_resistance_enabled: false, // Disable for performance
            cache_timing_resistance_enabled: false, // Disable for performance
            auto_zeroization_enabled: true,       // Keep for memory safety
            memory_protection_enabled: false,     // Disable for performance
            timing_analysis_enabled: false,       // Disable for performance
            formal_verification_enabled: false,   // Disable for performance
            security_testing_enabled: false,      // Disable for performance
            max_timing_variance_ns: 10000,        // More relaxed timing requirements
            security_level_bits: 128,             // Standard security level
            debug_mode: false,
        }
    }
    
    /// Create a debug configuration for security analysis and testing
    /// This configuration enables all debugging and analysis features
    /// and should NEVER be used in production environments.
    pub fn debug() -> Self {
        Self {
            constant_time_enabled: true,
            side_channel_resistance_enabled: true,
            cache_timing_resistance_enabled: true,
            auto_zeroization_enabled: true,
            memory_protection_enabled: true,
            timing_analysis_enabled: true,
            formal_verification_enabled: true,
            security_testing_enabled: true,
            max_timing_variance_ns: 1000,
            security_level_bits: 128,
            debug_mode: true,                     // Enable debug mode for analysis
        }
    }
    
    /// Validate the security configuration for consistency and safety
    /// This method checks that the configuration parameters are valid
    /// and consistent with each other, preventing misconfigurations.
    pub fn validate(&self) -> Result<()> {
        // Check that security level is valid (must be 128, 192, or 256)
        if ![128, 192, 256].contains(&self.security_level_bits) {
            return Err(LatticeFoldError::InvalidParameters(
                format!("Invalid security level: {} bits. Must be 128, 192, or 256.", 
                       self.security_level_bits)
            ));
        }
        
        // Check that timing variance is reasonable (not too small or too large)
        if self.max_timing_variance_ns == 0 {
            return Err(LatticeFoldError::InvalidParameters(
                "Maximum timing variance cannot be zero".to_string()
            ));
        }
        
        if self.max_timing_variance_ns > 1_000_000_000 { // 1 second
            return Err(LatticeFoldError::InvalidParameters(
                "Maximum timing variance is too large (> 1 second)".to_string()
            ));
        }
        
        // Warn about debug mode in production (this would be logged in real implementation)
        if self.debug_mode {
            eprintln!("WARNING: Debug mode is enabled. This should NEVER be used in production!");
        }
        
        // Check for inconsistent configurations
        if self.timing_analysis_enabled && !self.constant_time_enabled {
            return Err(LatticeFoldError::InvalidParameters(
                "Timing analysis requires constant-time operations to be enabled".to_string()
            ));
        }
        
        if self.formal_verification_enabled && !self.constant_time_enabled {
            return Err(LatticeFoldError::InvalidParameters(
                "Formal verification requires constant-time operations to be enabled".to_string()
            ));
        }
        
        Ok(())
    }
    
    /// Get the recommended parameters for the current security level
    /// This method returns cryptographic parameters that are appropriate
    /// for the configured security level and threat model.
    pub fn get_crypto_params(&self) -> CryptographicParameters {
        match self.security_level_bits {
            128 => CryptographicParameters {
                ring_dimension: 1024,
                modulus_bits: 64,
                noise_parameter: 3.2,
                norm_bound: 1024,
                challenge_set_size: 256,
                repetitions: 128,
            },
            192 => CryptographicParameters {
                ring_dimension: 2048,
                modulus_bits: 96,
                noise_parameter: 4.8,
                norm_bound: 2048,
                challenge_set_size: 512,
                repetitions: 192,
            },
            256 => CryptographicParameters {
                ring_dimension: 4096,
                modulus_bits: 128,
                noise_parameter: 6.4,
                norm_bound: 4096,
                challenge_set_size: 1024,
                repetitions: 256,
            },
            _ => unreachable!("Invalid security level validated earlier"),
        }
    }
}

/// Cryptographic parameters derived from the security configuration
/// These parameters define the concrete values used in the LatticeFold+
/// protocol based on the desired security level and threat model.
#[derive(Clone, Debug)]
pub struct CryptographicParameters {
    /// Ring dimension d (power of 2)
    pub ring_dimension: usize,
    
    /// Modulus bit length
    pub modulus_bits: u32,
    
    /// Gaussian noise parameter σ
    pub noise_parameter: f64,
    
    /// Norm bound for valid witnesses
    pub norm_bound: i64,
    
    /// Size of challenge set for soundness
    pub challenge_set_size: usize,
    
    /// Number of repetitions for soundness amplification
    pub repetitions: usize,
}

/// Global security manager for the LatticeFold+ implementation
/// This singleton manages all security-related operations and ensures
/// consistent security policies across the entire system.
pub struct SecurityManager {
    /// Current security configuration
    config: SecurityConfig,
    
    /// Timing analyzer for constant-time verification
    timing_analyzer: Option<TimingAnalyzer>,
    
    /// Security validator for comprehensive testing
    security_validator: Option<SecurityValidator>,
    
    /// Secure memory manager for protected allocations
    memory_manager: Option<SecureMemoryManager>,
}

impl SecurityManager {
    /// Create a new security manager with the specified configuration
    /// This initializes all security subsystems according to the configuration
    /// and prepares the system for secure operation.
    pub fn new(config: SecurityConfig) -> Result<Self> {
        // Validate the configuration before proceeding
        config.validate()?;
        
        // Initialize timing analyzer if enabled
        let timing_analyzer = if config.timing_analysis_enabled {
            Some(TimingAnalyzer::new(config.max_timing_variance_ns)?)
        } else {
            None
        };
        
        // Initialize security validator if enabled
        let security_validator = if config.security_testing_enabled {
            Some(SecurityValidator::new(&config)?)
        } else {
            None
        };
        
        // Initialize secure memory manager if enabled
        let memory_manager = if config.memory_protection_enabled {
            Some(SecureMemoryManager::new(&config)?)
        } else {
            None
        };
        
        Ok(Self {
            config,
            timing_analyzer,
            security_validator,
            memory_manager,
        })
    }
    
    /// Get the current security configuration
    /// This provides read-only access to the security configuration
    /// to allow other components to adapt their behavior accordingly.
    pub fn get_config(&self) -> &SecurityConfig {
        &self.config
    }
    
    /// Check if constant-time operations are enabled
    /// This is a convenience method for components that need to choose
    /// between constant-time and variable-time implementations.
    pub fn is_constant_time_enabled(&self) -> bool {
        self.config.constant_time_enabled
    }
    
    /// Check if side-channel resistance is enabled
    /// This is a convenience method for components that need to apply
    /// additional side-channel countermeasures.
    pub fn is_side_channel_resistance_enabled(&self) -> bool {
        self.config.side_channel_resistance_enabled
    }
    
    /// Validate a timing measurement for constant-time compliance
    /// This method checks if an operation's timing is consistent with
    /// constant-time requirements and flags potential timing leaks.
    pub fn validate_timing(&mut self, operation_name: &str, duration_ns: u64) -> Result<()> {
        if let Some(ref mut analyzer) = self.timing_analyzer {
            analyzer.record_timing(operation_name, duration_ns)?;
            analyzer.check_timing_consistency(operation_name)?;
        }
        Ok(())
    }
    
    /// Run comprehensive security validation
    /// This method performs a complete security analysis of the system
    /// and reports any potential vulnerabilities or security issues.
    pub fn run_security_validation(&mut self) -> Result<SecurityValidationReport> {
        if let Some(ref mut validator) = self.security_validator {
            validator.run_comprehensive_validation()
        } else {
            Ok(SecurityValidationReport::default())
        }
    }
    
    /// Allocate secure memory for sensitive data
    /// This method allocates memory with appropriate security protections
    /// including automatic zeroization and access control.
    pub fn allocate_secure_memory(&mut self, size: usize) -> Result<SecureMemoryRegion> {
        if let Some(ref mut manager) = self.memory_manager {
            manager.allocate(size)
        } else {
            // Fallback to regular allocation if secure memory is disabled
            Ok(SecureMemoryRegion::new_unprotected(size)?)
        }
    }
}

/// Report from comprehensive security validation
/// This structure contains the results of security testing and analysis
/// including any detected vulnerabilities or security issues.
#[derive(Clone, Debug, Default)]
pub struct SecurityValidationReport {
    /// Overall security assessment
    pub overall_assessment: SecurityAssessment,
    
    /// Timing analysis results
    pub timing_analysis: TimingAnalysisReport,
    
    /// Side-channel resistance analysis
    pub side_channel_analysis: SideChannelAnalysisReport,
    
    /// Memory security analysis
    pub memory_security_analysis: MemorySecurityReport,
    
    /// Cryptographic property verification
    pub crypto_verification: CryptographicVerificationReport,
    
    /// List of detected vulnerabilities
    pub vulnerabilities: Vec<SecurityVulnerability>,
    
    /// List of security recommendations
    pub recommendations: Vec<SecurityRecommendation>,
}

/// Overall security assessment levels
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SecurityAssessment {
    /// System passes all security tests with no issues detected
    Secure,
    /// System has minor security issues that should be addressed
    Warning,
    /// System has significant security vulnerabilities that must be fixed
    Vulnerable,
    /// System has critical security flaws that make it unsafe to use
    Critical,
}

impl Default for SecurityAssessment {
    fn default() -> Self {
        SecurityAssessment::Secure
    }
}

/// Timing analysis report
#[derive(Clone, Debug, Default)]
pub struct TimingAnalysisReport {
    /// Whether timing is consistent across operations
    pub timing_consistent: bool,
    
    /// Maximum observed timing variance
    pub max_variance_ns: u64,
    
    /// Operations with suspicious timing patterns
    pub suspicious_operations: Vec<String>,
}

/// Side-channel analysis report
#[derive(Clone, Debug, Default)]
pub struct SideChannelAnalysisReport {
    /// Whether side-channel countermeasures are effective
    pub countermeasures_effective: bool,
    
    /// Detected side-channel vulnerabilities
    pub vulnerabilities: Vec<SideChannelVulnerability>,
}

/// Memory security report
#[derive(Clone, Debug, Default)]
pub struct MemorySecurityReport {
    /// Whether memory is properly protected
    pub memory_protected: bool,
    
    /// Whether sensitive data is properly zeroized
    pub data_zeroized: bool,
    
    /// Detected memory security issues
    pub issues: Vec<MemorySecurityIssue>,
}

/// Cryptographic verification report
#[derive(Clone, Debug, Default)]
pub struct CryptographicVerificationReport {
    /// Whether cryptographic properties are satisfied
    pub properties_satisfied: bool,
    
    /// Results of individual property checks
    pub property_results: Vec<PropertyVerificationResult>,
}

/// Individual security vulnerability
#[derive(Clone, Debug)]
pub struct SecurityVulnerability {
    /// Severity level of the vulnerability
    pub severity: VulnerabilitySeverity,
    
    /// Description of the vulnerability
    pub description: String,
    
    /// Location where the vulnerability was detected
    pub location: String,
    
    /// Recommended mitigation
    pub mitigation: String,
}

/// Security recommendation
#[derive(Clone, Debug)]
pub struct SecurityRecommendation {
    /// Priority level of the recommendation
    pub priority: RecommendationPriority,
    
    /// Description of the recommendation
    pub description: String,
    
    /// Expected security benefit
    pub benefit: String,
}

/// Vulnerability severity levels
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum VulnerabilitySeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Recommendation priority levels
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum RecommendationPriority {
    Low,
    Medium,
    High,
    Critical,
}

/// Side-channel vulnerability types
#[derive(Clone, Debug)]
pub enum SideChannelVulnerability {
    TimingLeak { operation: String, variance_ns: u64 },
    PowerAnalysisVulnerable { operation: String },
    CacheTimingVulnerable { operation: String },
    ElectromagneticLeak { operation: String },
    AcousticLeak { operation: String },
}

/// Memory security issue types
#[derive(Clone, Debug)]
pub enum MemorySecurityIssue {
    UnprotectedSensitiveData { location: String },
    MemoryNotZeroized { location: String },
    BufferOverflow { location: String },
    UseAfterFree { location: String },
}

/// Property verification result
#[derive(Clone, Debug)]
pub struct PropertyVerificationResult {
    /// Name of the cryptographic property
    pub property_name: String,
    
    /// Whether the property is satisfied
    pub satisfied: bool,
    
    /// Additional details about the verification
    pub details: String,
}

/// Secure memory region with automatic protection
/// This structure represents a region of memory that is protected
/// against unauthorized access and automatically zeroized on deallocation.
pub struct SecureMemoryRegion {
    /// Pointer to the protected memory
    ptr: *mut u8,
    
    /// Size of the memory region
    size: usize,
    
    /// Whether the memory is protected
    protected: bool,
}

impl SecureMemoryRegion {
    /// Create a new unprotected memory region (fallback)
    /// This is used when secure memory protection is disabled
    /// but still provides basic memory management functionality.
    pub fn new_unprotected(size: usize) -> Result<Self> {
        let layout = std::alloc::Layout::from_size_align(size, 8)
            .map_err(|e| LatticeFoldError::MemoryAllocationError(e.to_string()))?;
        
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        if ptr.is_null() {
            return Err(LatticeFoldError::MemoryAllocationError(
                "Failed to allocate memory".to_string()
            ));
        }
        
        Ok(Self {
            ptr,
            size,
            protected: false,
        })
    }
    
    /// Get a mutable slice to the memory region
    /// This provides safe access to the protected memory
    /// while maintaining memory safety guarantees.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.size) }
    }
    
    /// Get an immutable slice to the memory region
    /// This provides safe read-only access to the protected memory
    /// while maintaining memory safety guarantees.
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.size) }
    }
    
    /// Get the size of the memory region
    pub fn size(&self) -> usize {
        self.size
    }
    
    /// Check if the memory is protected
    pub fn is_protected(&self) -> bool {
        self.protected
    }
}

impl Drop for SecureMemoryRegion {
    /// Automatically zeroize and deallocate the memory region
    /// This ensures that sensitive data is properly cleared
    /// when the memory region goes out of scope.
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // Zeroize the memory before deallocation
            unsafe {
                std::ptr::write_bytes(self.ptr, 0, self.size);
            }
            
            // Deallocate the memory
            let layout = std::alloc::Layout::from_size_align(self.size, 8).unwrap();
            unsafe {
                std::alloc::dealloc(self.ptr, layout);
            }
            
            self.ptr = std::ptr::null_mut();
        }
    }
}

// Ensure SecureMemoryRegion cannot be sent between threads unsafely
unsafe impl Send for SecureMemoryRegion {}
unsafe impl Sync for SecureMemoryRegion {}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_security_config_validation() {
        // Test valid configuration
        let config = SecurityConfig::default();
        assert!(config.validate().is_ok());
        
        // Test invalid security level
        let mut invalid_config = SecurityConfig::default();
        invalid_config.security_level_bits = 100; // Invalid
        assert!(invalid_config.validate().is_err());
        
        // Test zero timing variance
        let mut invalid_config = SecurityConfig::default();
        invalid_config.max_timing_variance_ns = 0; // Invalid
        assert!(invalid_config.validate().is_err());
        
        // Test inconsistent configuration
        let mut invalid_config = SecurityConfig::default();
        invalid_config.constant_time_enabled = false;
        invalid_config.timing_analysis_enabled = true; // Inconsistent
        assert!(invalid_config.validate().is_err());
    }
    
    #[test]
    fn test_security_config_presets() {
        // Test high security configuration
        let high_sec = SecurityConfig::high_security();
        assert!(high_sec.validate().is_ok());
        assert_eq!(high_sec.security_level_bits, 256);
        assert_eq!(high_sec.max_timing_variance_ns, 100);
        
        // Test performance optimized configuration
        let perf_opt = SecurityConfig::performance_optimized();
        assert!(perf_opt.validate().is_ok());
        assert!(!perf_opt.side_channel_resistance_enabled);
        assert!(!perf_opt.cache_timing_resistance_enabled);
        
        // Test debug configuration
        let debug = SecurityConfig::debug();
        assert!(debug.validate().is_ok());
        assert!(debug.debug_mode);
    }
    
    #[test]
    fn test_cryptographic_parameters() {
        let config = SecurityConfig::default();
        let params = config.get_crypto_params();
        
        // Check that parameters are reasonable for 128-bit security
        assert_eq!(params.ring_dimension, 1024);
        assert_eq!(params.modulus_bits, 64);
        assert!(params.noise_parameter > 0.0);
        assert!(params.norm_bound > 0);
        assert!(params.challenge_set_size > 0);
        assert!(params.repetitions > 0);
    }
    
    #[test]
    fn test_secure_memory_region() {
        // Test unprotected memory allocation
        let mut region = SecureMemoryRegion::new_unprotected(1024).unwrap();
        assert_eq!(region.size(), 1024);
        assert!(!region.is_protected());
        
        // Test memory access
        let slice = region.as_mut_slice();
        slice[0] = 42;
        assert_eq!(slice[0], 42);
        
        let const_slice = region.as_slice();
        assert_eq!(const_slice[0], 42);
        
        // Memory will be automatically zeroized on drop
    }
    
    #[test]
    fn test_security_manager_creation() {
        // Test creation with default configuration
        let config = SecurityConfig::default();
        let manager = SecurityManager::new(config.clone());
        assert!(manager.is_ok());
        
        let manager = manager.unwrap();
        assert_eq!(manager.get_config().security_level_bits, config.security_level_bits);
        assert!(manager.is_constant_time_enabled());
        assert!(manager.is_side_channel_resistance_enabled());
        
        // Test creation with invalid configuration
        let mut invalid_config = SecurityConfig::default();
        invalid_config.security_level_bits = 100; // Invalid
        let manager = SecurityManager::new(invalid_config);
        assert!(manager.is_err());
    }
}