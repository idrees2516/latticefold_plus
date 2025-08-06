// Integration Test Runner
//
// This module provides a comprehensive test runner for executing end-to-end
// integration tests, performance benchmarks, security validation, and
// cross-platform compatibility testing for the LatticeFold+ proof system.
//
// The test runner orchestrates the execution of all test categories and
// provides detailed reporting, analysis, and validation capabilities.

use crate::error::LatticeFoldError;
use crate::types::*;
use crate::integration_tests::{
    IntegrationTestRunner, IntegrationTestResult, TestConfiguration, ValidationStrictness,
    end_to_end::EndToEndTestSuite,
    performance::PerformanceBenchmarkSuite,
    security::SecurityTestSuite,
    compatibility::CompatibilityTestSuite,
};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tokio::time::timeout;

/// Main test runner for comprehensive LatticeFold+ integration testing
/// 
/// This is the primary entry point for executing the complete integration
/// test suite including end-to-end protocol testing, performance benchmarking,
/// security validation, and cross-platform compatibility verification.
pub struct ComprehensiveTestRunner {
    /// Test configuration parameters
    config: TestConfiguration,
    
    /// End-to-end test suite
    end_to_end_suite: EndToEndTestSuite,
    
    /// Performance benchmark suite
    performance_suite: PerformanceBenchmarkSuite,
    
    /// Security test suite
    security_suite: SecurityTestSuite,
    
    /// Compatibility test suite
    compatibility_suite: CompatibilityTestSuite,
    
    /// Test execution results
    test_results: Vec<IntegrationTestResult>,
    
    /// Test execution start time
    execution_start_time: Option<Instant>,
}

impl ComprehensiveTestRunner {
    /// Create new comprehensive test runner with configuration
    /// 
    /// Initializes all test suites with the provided configuration and
    /// prepares the runner for comprehensive integration testing.
    /// 
    /// # Arguments
    /// * `config` - Test configuration specifying parameters and validation criteria
    /// 
    /// # Returns
    /// * New ComprehensiveTestRunner instance ready for test execution
    /// 
    /// # Example
    /// ```rust
    /// let config = TestConfiguration {
    ///     ring_dimension: 1024,
    ///     modulus: 2147483647,
    ///     security_parameter: 128,
    ///     norm_bounds: vec![1024, 2048, 4096],
    ///     constraint_counts: vec![64, 128, 256, 512],
    ///     folding_instance_counts: vec![2, 4, 8, 16],
    ///     gpu_acceleration_enabled: true,
    ///     test_iterations: 10,
    ///     test_timeout: Duration::from_secs(300),
    ///     memory_limit: 8 * 1024 * 1024 * 1024, // 8GB
    ///     random_seed: 42,
    ///     validation_strictness: ValidationStrictness::Comprehensive,
    /// };
    /// let runner = ComprehensiveTestRunner::new(config);
    /// ```
    pub fn new(config: TestConfiguration) -> Self {
        // Initialize end-to-end test suite with configuration
        let end_to_end_suite = EndToEndTestSuite::new(config.clone());
        
        // Initialize performance benchmark suite with configuration
        let performance_suite = PerformanceBenchmarkSuite::new(config.clone());
        
        // Initialize security test suite with security-specific configuration
        let security_config = crate::integration_tests::security::SecurityTestConfiguration {
            timing_measurement_count: 1000,
            timing_significance_threshold: 0.05,
            side_channel_precision: 0.001,
            parameter_validation_strictness: match config.validation_strictness {
                ValidationStrictness::Minimal => crate::integration_tests::security::ValidationStrictness::Basic,
                ValidationStrictness::Standard => crate::integration_tests::security::ValidationStrictness::Standard,
                ValidationStrictness::Comprehensive => crate::integration_tests::security::ValidationStrictness::Comprehensive,
                ValidationStrictness::Exhaustive => crate::integration_tests::security::ValidationStrictness::Exhaustive,
            },
            assumption_verification_depth: crate::integration_tests::security::AssumptionVerificationDepth::Comprehensive,
            malicious_attack_scenario_count: 50,
            test_timeout: config.test_timeout,
        };
        let security_suite = SecurityTestSuite::new(security_config);
        
        // Initialize compatibility test suite with platform-specific configuration
        let compatibility_config = crate::integration_tests::compatibility::CompatibilityTestConfiguration {
            target_platforms: vec![
                crate::integration_tests::compatibility::PlatformSpecification {
                    platform_id: "x86_64-linux".to_string(),
                    architecture: "x86_64".to_string(),
                    operating_system: "linux".to_string(),
                    cpu_model: "Intel Core i7".to_string(),
                    cpu_cores: 8,
                    cpu_frequency: 3.2,
                    memory_size: 16 * 1024 * 1024 * 1024, // 16GB
                    memory_type: "DDR4".to_string(),
                    instruction_sets: vec!["AVX2".to_string(), "AVX-512".to_string()],
                    endianness: "little".to_string(),
                    word_size: 64,
                },
                crate::integration_tests::compatibility::PlatformSpecification {
                    platform_id: "aarch64-macos".to_string(),
                    architecture: "aarch64".to_string(),
                    operating_system: "macos".to_string(),
                    cpu_model: "Apple M1".to_string(),
                    cpu_cores: 8,
                    cpu_frequency: 3.2,
                    memory_size: 16 * 1024 * 1024 * 1024, // 16GB
                    memory_type: "LPDDR4X".to_string(),
                    instruction_sets: vec!["NEON".to_string()],
                    endianness: "little".to_string(),
                    word_size: 64,
                },
            ],
            gpu_platforms: vec![
                crate::integration_tests::compatibility::GpuSpecification {
                    gpu_id: "nvidia_rtx_4090".to_string(),
                    vendor: "NVIDIA".to_string(),
                    model: "GeForce RTX 4090".to_string(),
                    compute_capability: "8.9".to_string(),
                    memory_size: 24 * 1024 * 1024 * 1024, // 24GB
                    memory_bandwidth: 1008.0, // GB/s
                    compute_units: 128,
                    supported_apis: vec!["CUDA".to_string(), "OpenCL".to_string()],
                },
            ],
            compiler_configurations: vec![
                crate::integration_tests::compatibility::CompilerConfiguration {
                    compiler_id: "rustc_stable".to_string(),
                    compiler_name: "rustc".to_string(),
                    compiler_version: "1.70.0".to_string(),
                    optimization_level: "-O3".to_string(),
                    target_flags: vec!["--target-cpu=native".to_string()],
                    additional_flags: vec!["-C", "lto=fat"].iter().map(|s| s.to_string()).collect(),
                    lto_enabled: true,
                    debug_info: false,
                },
            ],
            numerical_tolerance: 1e-12,
            performance_tolerance: 10.0, // 10% tolerance
            test_timeout: config.test_timeout,
            test_iterations: config.test_iterations,
        };
        let compatibility_suite = CompatibilityTestSuite::new(compatibility_config);
        
        Self {
            config,
            end_to_end_suite,
            performance_suite,
            security_suite,
            compatibility_suite,
            test_results: Vec::new(),
            execution_start_time: None,
        }
    }
    
    /// Execute comprehensive integration tests with timeout and error handling
    /// 
    /// Runs the complete integration test suite with comprehensive error handling,
    /// timeout management, and detailed progress reporting. This is the main
    /// entry point for executing all integration tests.
    /// 
    /// # Returns
    /// * `Result<IntegrationTestResult, LatticeFoldError>` - Comprehensive test results
    /// 
    /// # Errors
    /// * Returns error if test setup fails or critical system components are unavailable
    /// * Returns timeout error if tests exceed configured timeout duration
    /// * Returns error if memory usage exceeds configured limits
    /// 
    /// # Example
    /// ```rust
    /// let mut runner = ComprehensiveTestRunner::new(config);
    /// match runner.execute_comprehensive_tests().await {
    ///     Ok(results) => {
    ///         println!("Tests completed successfully: {} categories", results.category_results.len());
    ///         println!("Overall success: {}", results.success);
    ///         println!("Total execution time: {:?}", results.total_execution_time);
    ///     },
    ///     Err(e) => {
    ///         eprintln!("Test execution failed: {}", e);
    ///         std::process::exit(1);
    ///     }
    /// }
    /// ```
    pub async fn execute_comprehensive_tests(&mut self) -> Result<IntegrationTestResult, LatticeFoldError> {
        // Record test execution start time for overall timing analysis
        self.execution_start_time = Some(Instant::now());
        let test_start_time = self.execution_start_time.unwrap();
        
        println!("=== Starting Comprehensive LatticeFold+ Integration Testing ===");
        println!("Configuration:");
        println!("  Ring dimension: {}", self.config.ring_dimension);
        println!("  Security parameter: {}", self.config.security_parameter);
        println!("  GPU acceleration: {}", self.config.gpu_acceleration_enabled);
        println!("  Test iterations: {}", self.config.test_iterations);
        println!("  Validation strictness: {:?}", self.config.validation_strictness);
        println!("  Test timeout: {:?}", self.config.test_timeout);
        println!();
        
        // Initialize comprehensive test runner with timeout
        let test_future = self.run_all_test_suites();
        let test_result = match timeout(self.config.test_timeout, test_future).await {
            Ok(result) => result,
            Err(_) => {
                return Err(LatticeFoldError::TestTimeout(format!(
                    "Integration tests exceeded timeout of {:?}", 
                    self.config.test_timeout
                )));
            }
        };
        
        // Calculate total execution time
        let total_execution_time = test_start_time.elapsed();
        
        // Process and validate test results
        let mut final_result = test_result?;
        final_result.total_execution_time = total_execution_time;
        
        // Store results for future analysis and reporting
        self.test_results.push(final_result.clone());
        
        // Generate comprehensive test report
        self.generate_test_report(&final_result).await?;
        
        // Validate test results against success criteria
        self.validate_test_success(&final_result)?;
        
        println!("=== Comprehensive Integration Testing Completed ===");
        println!("Overall success: {}", final_result.success);
        println!("Total execution time: {:?}", final_result.total_execution_time);
        println!("Categories tested: {}", final_result.category_results.len());
        println!("Memory usage: {} MB", final_result.memory_usage.peak_memory_usage / (1024 * 1024));
        
        if !final_result.success {
            println!("Errors encountered: {}", final_result.error_details.len());
            for error in &final_result.error_details {
                println!("  - {}: {}", error.error_category, error.error_message);
            }
        }
        
        Ok(final_result)
    }
    
    /// Run all test suites in sequence with comprehensive error handling
    /// 
    /// Executes all test suites (end-to-end, performance, security, compatibility)
    /// in sequence with proper error handling and result aggregation.
    async fn run_all_test_suites(&mut self) -> Result<IntegrationTestResult, LatticeFoldError> {
        // Create integration test runner for orchestration
        let mut integration_runner = IntegrationTestRunner::new(self.config.clone());
        
        // Execute comprehensive tests through the integration runner
        let test_result = integration_runner.run_comprehensive_tests().await?;
        
        Ok(test_result)
    }
    
    /// Generate comprehensive test report with detailed analysis
    /// 
    /// Generates a detailed test report including performance analysis,
    /// security validation results, compatibility assessment, and
    /// recommendations for optimization and improvement.
    async fn generate_test_report(&self, test_result: &IntegrationTestResult) -> Result<(), LatticeFoldError> {
        println!("\n=== Generating Comprehensive Test Report ===");
        
        // Generate report header with test configuration
        let report_header = format!(
            "LatticeFold+ Integration Test Report\n\
             =====================================\n\
             Test ID: {}\n\
             Execution Time: {:?}\n\
             Test Configuration:\n\
             - Ring Dimension: {}\n\
             - Security Parameter: {}\n\
             - GPU Acceleration: {}\n\
             - Validation Strictness: {:?}\n\
             - Test Iterations: {}\n\n",
            test_result.test_id,
            test_result.total_execution_time,
            test_result.test_configuration.ring_dimension,
            test_result.test_configuration.security_parameter,
            test_result.test_configuration.gpu_acceleration_enabled,
            test_result.test_configuration.validation_strictness,
            test_result.test_configuration.test_iterations
        );
        
        println!("{}", report_header);
        
        // Generate category-specific reports
        for (category_name, category_result) in &test_result.category_results {
            println!("=== {} Test Results ===", category_name.to_uppercase());
            println!("Success: {}", category_result.success);
            println!("Tests Passed: {}/{}", category_result.tests_passed, category_result.total_tests);
            println!("Execution Time: {:?}", category_result.execution_time);
            
            // Display individual test results
            for individual_result in &category_result.individual_results {
                println!("  Test: {}", individual_result.test_name);
                println!("    Success: {}", individual_result.success);
                println!("    Time: {:?}", individual_result.execution_time);
                println!("    Memory: {} MB", individual_result.memory_used / (1024 * 1024));
                
                if let Some(error) = &individual_result.error_message {
                    println!("    Error: {}", error);
                }
                
                // Display key metrics
                for (metric_name, metric_value) in &individual_result.test_metrics {
                    println!("    {}: {:.2}", metric_name, metric_value);
                }
            }
            println!();
        }
        
        // Generate performance analysis report
        self.generate_performance_analysis_report(&test_result.performance_metrics).await?;
        
        // Generate security analysis report
        self.generate_security_analysis_report(&test_result.security_results).await?;
        
        // Generate compatibility analysis report
        self.generate_compatibility_analysis_report(&test_result.compatibility_results).await?;
        
        // Generate recommendations and next steps
        self.generate_recommendations_report(test_result).await?;
        
        Ok(())
    }
    
    /// Generate performance analysis report with detailed metrics
    /// 
    /// Creates a comprehensive performance analysis report including
    /// throughput measurements, scalability analysis, and comparison
    /// with baseline implementations.
    async fn generate_performance_analysis_report(&self, performance_metrics: &crate::integration_tests::PerformanceMetrics) -> Result<(), LatticeFoldError> {
        println!("=== PERFORMANCE ANALYSIS ===");
        println!("Proof Throughput: {:.2} constraints/second", performance_metrics.proof_throughput);
        println!("Verification Throughput: {:.2} proofs/second", performance_metrics.verification_throughput);
        println!("Prover Memory Usage: {} MB", performance_metrics.prover_memory_usage / (1024 * 1024));
        println!("Verifier Memory Usage: {} MB", performance_metrics.verifier_memory_usage / (1024 * 1024));
        println!("Memory Bandwidth Utilization: {:.1}%", performance_metrics.memory_bandwidth_utilization);
        
        // Display prover timing analysis
        if !performance_metrics.prover_times.is_empty() {
            println!("\nProver Timing Analysis:");
            for (constraint_count, duration) in &performance_metrics.prover_times {
                let throughput = *constraint_count as f64 / duration.as_secs_f64();
                println!("  {} constraints: {:?} ({:.2} constraints/sec)", 
                    constraint_count, duration, throughput);
            }
        }
        
        // Display verifier timing analysis
        if !performance_metrics.verifier_times.is_empty() {
            println!("\nVerifier Timing Analysis:");
            for (proof_size, duration) in &performance_metrics.verifier_times {
                println!("  {} byte proof: {:?}", proof_size, duration);
            }
        }
        
        // Display proof size analysis
        if !performance_metrics.proof_sizes.is_empty() {
            println!("\nProof Size Analysis:");
            for (constraint_count, proof_size) in &performance_metrics.proof_sizes {
                let size_per_constraint = *proof_size as f64 / *constraint_count as f64;
                println!("  {} constraints: {} bytes ({:.2} bytes/constraint)", 
                    constraint_count, proof_size, size_per_constraint);
            }
        }
        
        // Display GPU acceleration analysis
        if !performance_metrics.gpu_speedup_factors.is_empty() {
            println!("\nGPU Acceleration Analysis:");
            for (operation, speedup) in &performance_metrics.gpu_speedup_factors {
                println!("  {}: {:.2}x speedup", operation, speedup);
            }
        }
        
        println!();
        Ok(())
    }
    
    /// Generate security analysis report with vulnerability assessment
    /// 
    /// Creates a comprehensive security analysis report including
    /// timing attack resistance, side-channel analysis, and
    /// parameter security validation.
    async fn generate_security_analysis_report(&self, security_results: &crate::integration_tests::SecurityTestResults) -> Result<(), LatticeFoldError> {
        println!("=== SECURITY ANALYSIS ===");
        println!("Timing Attack Resistance: {}", security_results.timing_attack_resistance);
        println!("Parameter Security Validation: {}", security_results.parameter_security_validation);
        
        // Display side-channel resistance analysis
        println!("\nSide-Channel Resistance:");
        for (channel_type, is_resistant) in &security_results.side_channel_resistance {
            println!("  {}: {}", channel_type, if *is_resistant { "RESISTANT" } else { "VULNERABLE" });
        }
        
        // Display malicious prover test results
        if !security_results.malicious_prover_results.is_empty() {
            println!("\nMalicious Prover Attack Results:");
            for attack_result in &security_results.malicious_prover_results {
                println!("  Attack: {}", attack_result.attack_name);
                println!("    Prevented: {}", attack_result.attack_prevented);
                println!("    Detection Time: {:?}", attack_result.detection_time);
                println!("    Vector: {}", attack_result.attack_vector);
            }
        }
        
        // Display cryptographic assumption validation
        println!("\nCryptographic Assumption Validation:");
        for (assumption, is_valid) in &security_results.assumption_validation {
            println!("  {}: {}", assumption, if *is_valid { "VALID" } else { "INVALID" });
        }
        
        // Display randomness quality assessment
        println!("\nRandomness Quality Assessment:");
        println!("  Entropy Source Valid: {}", security_results.randomness_quality.entropy_source_valid);
        println!("  Cryptographic Quality: {}", security_results.randomness_quality.cryptographic_quality);
        println!("  Seed Management: {}", security_results.randomness_quality.seed_management);
        
        // Display constant-time validation
        println!("\nConstant-Time Operation Validation:");
        for (operation, is_constant_time) in &security_results.constant_time_validation {
            println!("  {}: {}", operation, if *is_constant_time { "CONSTANT-TIME" } else { "VARIABLE-TIME" });
        }
        
        // Display memory safety results
        println!("\nMemory Safety Analysis:");
        println!("  Buffer Overflow Protection: {}", security_results.memory_safety_results.buffer_overflow_protection);
        println!("  Bounds Checking: {}", security_results.memory_safety_results.bounds_checking);
        println!("  Memory Leak Detection: {}", security_results.memory_safety_results.memory_leak_detection);
        println!("  Use-After-Free Protection: {}", security_results.memory_safety_results.use_after_free_protection);
        println!("  Secure Zeroization: {}", security_results.memory_safety_results.secure_zeroization);
        
        println!();
        Ok(())
    }
    
    /// Generate compatibility analysis report with platform assessment
    /// 
    /// Creates a comprehensive compatibility analysis report including
    /// cross-platform validation, GPU compatibility, and numerical
    /// consistency verification.
    async fn generate_compatibility_analysis_report(&self, compatibility_results: &crate::integration_tests::CompatibilityResults) -> Result<(), LatticeFoldError> {
        println!("=== COMPATIBILITY ANALYSIS ===");
        println!("Numerical Consistency: {}", compatibility_results.numerical_consistency);
        println!("Serialization Compatibility: {}", compatibility_results.serialization_compatibility);
        
        // Display platform compatibility results
        println!("\nPlatform Compatibility:");
        for (platform_name, platform_result) in &compatibility_results.platform_results {
            println!("  {}: {}", platform_name, if platform_result.success { "COMPATIBLE" } else { "INCOMPATIBLE" });
            if !platform_result.platform_issues.is_empty() {
                for issue in &platform_result.platform_issues {
                    println!("    Issue: {}", issue);
                }
            }
        }
        
        // Display GPU compatibility results
        println!("\nGPU Compatibility:");
        for (gpu_name, gpu_result) in &compatibility_results.gpu_compatibility {
            println!("  {}: {}", gpu_name, if gpu_result.acceleration_available { "AVAILABLE" } else { "UNAVAILABLE" });
            println!("    Device: {}", gpu_result.device_name);
            println!("    Compute Capability: {}", gpu_result.compute_capability);
            println!("    Speedup Factor: {:.2}x", gpu_result.speedup_factor);
            println!("    Memory Efficiency: {:.1}%", gpu_result.memory_efficiency * 100.0);
            println!("    Kernel Correctness: {}", gpu_result.kernel_correctness);
            
            if !gpu_result.gpu_issues.is_empty() {
                for issue in &gpu_result.gpu_issues {
                    println!("    Issue: {}", issue);
                }
            }
        }
        
        // Display operating system compatibility
        println!("\nOperating System Compatibility:");
        for (os_name, is_compatible) in &compatibility_results.os_compatibility {
            println!("  {}: {}", os_name, if *is_compatible { "COMPATIBLE" } else { "INCOMPATIBLE" });
        }
        
        // Display compiler compatibility
        println!("\nCompiler Compatibility:");
        for (compiler_name, compiler_result) in &compatibility_results.compiler_compatibility {
            println!("  {}: {}", compiler_name, if compiler_result.compilation_success { "COMPATIBLE" } else { "INCOMPATIBLE" });
            println!("    Version: {}", compiler_result.compiler_version);
            
            if !compiler_result.optimization_effects.is_empty() {
                println!("    Optimization Effects:");
                for (opt_level, improvement) in &compiler_result.optimization_effects {
                    println!("      {}: {:.2}x improvement", opt_level, improvement);
                }
            }
            
            if !compiler_result.compiler_issues.is_empty() {
                for issue in &compiler_result.compiler_issues {
                    println!("    Issue: {}", issue);
                }
            }
        }
        
        println!();
        Ok(())
    }
    
    /// Generate recommendations report with optimization suggestions
    /// 
    /// Creates a comprehensive recommendations report including
    /// performance optimization suggestions, security improvements,
    /// and compatibility enhancements.
    async fn generate_recommendations_report(&self, test_result: &IntegrationTestResult) -> Result<(), LatticeFoldError> {
        println!("=== RECOMMENDATIONS AND NEXT STEPS ===");
        
        let mut recommendations = Vec::new();
        
        // Analyze test results and generate recommendations
        if !test_result.success {
            recommendations.push("Address failing tests before production deployment".to_string());
        }
        
        // Performance recommendations
        if test_result.performance_metrics.proof_throughput < 1000.0 {
            recommendations.push("Consider optimizing proof generation for better throughput".to_string());
        }
        
        if test_result.performance_metrics.memory_bandwidth_utilization < 50.0 {
            recommendations.push("Optimize memory access patterns to improve bandwidth utilization".to_string());
        }
        
        // Security recommendations
        if !test_result.security_results.timing_attack_resistance {
            recommendations.push("CRITICAL: Implement constant-time algorithms to prevent timing attacks".to_string());
        }
        
        if !test_result.security_results.parameter_security_validation {
            recommendations.push("CRITICAL: Validate and adjust cryptographic parameters for security requirements".to_string());
        }
        
        // Compatibility recommendations
        if !test_result.compatibility_results.numerical_consistency {
            recommendations.push("Address numerical consistency issues across platforms".to_string());
        }
        
        if !test_result.compatibility_results.serialization_compatibility {
            recommendations.push("Fix serialization compatibility issues for cross-platform deployment".to_string());
        }
        
        // GPU acceleration recommendations
        let gpu_available = test_result.compatibility_results.gpu_compatibility
            .values()
            .any(|gpu| gpu.acceleration_available);
        
        if !gpu_available && test_result.test_configuration.gpu_acceleration_enabled {
            recommendations.push("GPU acceleration requested but not available - consider CPU optimizations".to_string());
        }
        
        // Memory usage recommendations
        if test_result.memory_usage.peak_memory_usage > test_result.test_configuration.memory_limit {
            recommendations.push("Peak memory usage exceeds configured limit - optimize memory allocation".to_string());
        }
        
        // Display recommendations
        if recommendations.is_empty() {
            println!("✅ All tests passed successfully - no critical recommendations");
        } else {
            println!("Recommendations for improvement:");
            for (i, recommendation) in recommendations.iter().enumerate() {
                println!("  {}. {}", i + 1, recommendation);
            }
        }
        
        // Display next steps
        println!("\nNext Steps:");
        println!("1. Review detailed test results and address any failures");
        println!("2. Implement recommended optimizations and security improvements");
        println!("3. Re-run tests to validate improvements");
        println!("4. Consider additional testing with larger parameter sets");
        println!("5. Prepare for production deployment validation");
        
        println!();
        Ok(())
    }
    
    /// Validate test success against defined criteria
    /// 
    /// Validates the overall test results against success criteria
    /// and determines if the implementation meets requirements.
    fn validate_test_success(&self, test_result: &IntegrationTestResult) -> Result<(), LatticeFoldError> {
        let mut validation_errors = Vec::new();
        
        // Check overall test success
        if !test_result.success {
            validation_errors.push("Overall test execution failed".to_string());
        }
        
        // Check critical security requirements
        if !test_result.security_results.timing_attack_resistance {
            validation_errors.push("CRITICAL: Timing attack resistance validation failed".to_string());
        }
        
        if !test_result.security_results.parameter_security_validation {
            validation_errors.push("CRITICAL: Parameter security validation failed".to_string());
        }
        
        // Check performance requirements
        if test_result.performance_metrics.proof_throughput < 100.0 {
            validation_errors.push("Performance requirement not met: proof throughput too low".to_string());
        }
        
        // Check memory requirements
        if test_result.memory_usage.peak_memory_usage > test_result.test_configuration.memory_limit {
            validation_errors.push("Memory requirement not met: peak usage exceeds limit".to_string());
        }
        
        // Check compatibility requirements
        if !test_result.compatibility_results.numerical_consistency {
            validation_errors.push("Compatibility requirement not met: numerical inconsistency detected".to_string());
        }
        
        // Return error if validation fails
        if !validation_errors.is_empty() {
            return Err(LatticeFoldError::ValidationFailure(format!(
                "Test validation failed with {} errors: {}",
                validation_errors.len(),
                validation_errors.join("; ")
            )));
        }
        
        Ok(())
    }
}

/// Execute comprehensive integration tests with command-line interface
/// 
/// Main entry point for running comprehensive integration tests from
/// command line or automated testing systems.
/// 
/// # Arguments
/// * `config` - Optional test configuration (uses defaults if None)
/// 
/// # Returns
/// * `Result<IntegrationTestResult, LatticeFoldError>` - Test execution results
/// 
/// # Example
/// ```rust
/// // Run with default configuration
/// let result = execute_integration_tests(None).await?;
/// 
/// // Run with custom configuration
/// let custom_config = TestConfiguration { /* ... */ };
/// let result = execute_integration_tests(Some(custom_config)).await?;
/// ```
pub async fn execute_integration_tests(config: Option<TestConfiguration>) -> Result<IntegrationTestResult, LatticeFoldError> {
    // Use provided configuration or create default
    let test_config = config.unwrap_or_else(|| TestConfiguration {
        ring_dimension: 1024,
        modulus: 2147483647, // 2^31 - 1
        security_parameter: 128,
        norm_bounds: vec![1024, 2048, 4096],
        constraint_counts: vec![64, 128, 256, 512],
        folding_instance_counts: vec![2, 4, 8, 16],
        gpu_acceleration_enabled: true,
        test_iterations: 10,
        test_timeout: Duration::from_secs(1800), // 30 minutes
        memory_limit: 8 * 1024 * 1024 * 1024, // 8GB
        random_seed: 42,
        validation_strictness: ValidationStrictness::Comprehensive,
    });
    
    // Create and execute comprehensive test runner
    let mut runner = ComprehensiveTestRunner::new(test_config);
    let test_result = runner.execute_comprehensive_tests().await?;
    
    Ok(test_result)
}