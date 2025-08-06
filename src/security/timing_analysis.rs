// Timing analysis and constant-time verification for LatticeFold+ implementation
// This module provides comprehensive timing analysis capabilities to detect
// timing-based side-channel vulnerabilities and verify constant-time properties
// of cryptographic operations.

use crate::error::{LatticeFoldError, Result};
use crate::security::SecurityConfig;
use std::collections::HashMap;
use std::time::{Duration, Instant, SystemTime};
use std::sync::{Arc, Mutex};
use std::thread;

/// Timing analyzer for detecting timing-based side-channel vulnerabilities
/// This analyzer monitors operation timing, detects anomalies, and provides
/// comprehensive analysis of timing behavior for security assessment.
#[derive(Debug)]
pub struct TimingAnalyzer {
    /// Configuration for timing analysis
    config: SecurityConfig,
    
    /// Maximum allowed timing variance in nanoseconds
    max_variance_ns: u64,
    
    /// Collected timing measurements
    measurements: Arc<Mutex<HashMap<String, Vec<TimingMeasurement>>>>,
    
    /// Statistical analysis results
    statistics: Arc<Mutex<HashMap<String, TimingStatistics>>>,
    
    /// Detected timing anomalies
    anomalies: Arc<Mutex<Vec<TimingAnomaly>>>,
    
    /// Baseline timing measurements for comparison
    baselines: Arc<Mutex<HashMap<String, BaselineTiming>>>,
    
    /// Whether continuous monitoring is enabled
    continuous_monitoring: bool,
    
    /// Monitoring thread handle
    monitor_thread: Option<thread::JoinHandle<()>>,
}

/// Individual timing measurement record
#[derive(Debug, Clone)]
pub struct TimingMeasurement {
    /// Name of the operation being measured
    pub operation_name: String,
    
    /// Duration of the operation in nanoseconds
    pub duration_ns: u64,
    
    /// Timestamp when measurement was taken
    pub timestamp: SystemTime,
    
    /// Input size or complexity measure
    pub input_size: usize,
    
    /// Additional context information
    pub context: TimingContext,
    
    /// CPU cycle count (if available)
    pub cpu_cycles: Option<u64>,
    
    /// Memory access count (if available)
    pub memory_accesses: Option<u64>,
}

/// Context information for timing measurements
#[derive(Debug, Clone)]
pub struct TimingContext {
    /// Thread ID where measurement was taken
    pub thread_id: u64,
    
    /// CPU core where operation executed
    pub cpu_core: Option<u32>,
    
    /// System load at time of measurement
    pub system_load: f64,
    
    /// Available memory at time of measurement
    pub available_memory: u64,
    
    /// Cache state information
    pub cache_state: CacheState,
    
    /// Whether operation involved secret data
    pub involves_secrets: bool,
}

/// Cache state information
#[derive(Debug, Clone)]
pub struct CacheState {
    /// L1 cache hit rate
    pub l1_hit_rate: f64,
    
    /// L2 cache hit rate
    pub l2_hit_rate: f64,
    
    /// L3 cache hit rate
    pub l3_hit_rate: f64,
    
    /// TLB hit rate
    pub tlb_hit_rate: f64,
    
    /// Branch prediction accuracy
    pub branch_prediction_accuracy: f64,
}

/// Statistical analysis of timing measurements
#[derive(Debug, Clone)]
pub struct TimingStatistics {
    /// Operation name
    pub operation_name: String,
    
    /// Total number of measurements
    pub sample_count: usize,
    
    /// Minimum observed duration in nanoseconds
    pub min_duration_ns: u64,
    
    /// Maximum observed duration in nanoseconds
    pub max_duration_ns: u64,
    
    /// Mean duration in nanoseconds
    pub mean_duration_ns: f64,
    
    /// Median duration in nanoseconds
    pub median_duration_ns: u64,
    
    /// Standard deviation in nanoseconds
    pub std_deviation_ns: f64,
    
    /// Variance in nanoseconds squared
    pub variance_ns: f64,
    
    /// Coefficient of variation (std_dev / mean)
    pub coefficient_of_variation: f64,
    
    /// Skewness of the distribution
    pub skewness: f64,
    
    /// Kurtosis of the distribution
    pub kurtosis: f64,
    
    /// 95th percentile duration
    pub percentile_95_ns: u64,
    
    /// 99th percentile duration
    pub percentile_99_ns: u64,
    
    /// Whether timing appears constant
    pub appears_constant_time: bool,
    
    /// Confidence level for constant-time assessment
    pub constant_time_confidence: f64,
    
    /// Detected timing patterns
    pub timing_patterns: Vec<TimingPattern>,
}

/// Detected timing pattern
#[derive(Debug, Clone)]
pub struct TimingPattern {
    /// Type of pattern detected
    pub pattern_type: PatternType,
    
    /// Confidence level for pattern detection
    pub confidence: f64,
    
    /// Description of the pattern
    pub description: String,
    
    /// Potential security implications
    pub security_implications: Vec<String>,
}

/// Types of timing patterns
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PatternType {
    /// Input-dependent timing variation
    InputDependent,
    
    /// Secret-dependent timing variation
    SecretDependent,
    
    /// Branch-dependent timing variation
    BranchDependent,
    
    /// Memory-access-dependent timing variation
    MemoryAccessDependent,
    
    /// Cache-dependent timing variation
    CacheDependent,
    
    /// Algorithmic complexity variation
    ComplexityDependent,
    
    /// System-load-dependent variation
    SystemLoadDependent,
}

/// Timing anomaly detection
#[derive(Debug, Clone)]
pub struct TimingAnomaly {
    /// Operation where anomaly was detected
    pub operation_name: String,
    
    /// Type of anomaly
    pub anomaly_type: AnomalyType,
    
    /// Severity of the anomaly
    pub severity: AnomalySeverity,
    
    /// Detailed description
    pub description: String,
    
    /// Measurements that triggered the anomaly
    pub triggering_measurements: Vec<TimingMeasurement>,
    
    /// Statistical evidence for the anomaly
    pub statistical_evidence: StatisticalEvidence,
    
    /// Timestamp when anomaly was detected
    pub detected_at: SystemTime,
    
    /// Recommended actions
    pub recommendations: Vec<String>,
}

/// Types of timing anomalies
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AnomalyType {
    /// Excessive timing variance
    ExcessiveVariance,
    
    /// Timing correlation with input data
    InputCorrelation,
    
    /// Timing correlation with secret data
    SecretCorrelation,
    
    /// Suspicious timing distribution
    SuspiciousDistribution,
    
    /// Timing side-channel vulnerability
    SideChannelVulnerability,
    
    /// Performance regression
    PerformanceRegression,
    
    /// Unexpected timing behavior
    UnexpectedBehavior,
}

/// Severity levels for timing anomalies
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AnomalySeverity {
    /// Low severity - informational
    Low,
    
    /// Medium severity - should be investigated
    Medium,
    
    /// High severity - requires immediate attention
    High,
    
    /// Critical severity - security vulnerability
    Critical,
}

/// Statistical evidence for anomaly detection
#[derive(Debug, Clone)]
pub struct StatisticalEvidence {
    /// P-value for statistical test
    pub p_value: f64,
    
    /// Test statistic value
    pub test_statistic: f64,
    
    /// Type of statistical test used
    pub test_type: StatisticalTest,
    
    /// Confidence interval
    pub confidence_interval: (f64, f64),
    
    /// Effect size measure
    pub effect_size: f64,
    
    /// Power of the statistical test
    pub statistical_power: f64,
}

/// Types of statistical tests
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StatisticalTest {
    /// Student's t-test
    TTest,
    
    /// Welch's t-test (unequal variances)
    WelchTTest,
    
    /// Mann-Whitney U test
    MannWhitneyU,
    
    /// Kolmogorov-Smirnov test
    KolmogorovSmirnov,
    
    /// Anderson-Darling test
    AndersonDarling,
    
    /// Chi-square test
    ChiSquare,
    
    /// Correlation analysis
    Correlation,
}

/// Baseline timing measurements for comparison
#[derive(Debug, Clone)]
pub struct BaselineTiming {
    /// Operation name
    pub operation_name: String,
    
    /// Expected mean duration in nanoseconds
    pub expected_mean_ns: f64,
    
    /// Expected standard deviation in nanoseconds
    pub expected_std_dev_ns: f64,
    
    /// Acceptable variance threshold
    pub variance_threshold: f64,
    
    /// When baseline was established
    pub established_at: SystemTime,
    
    /// Number of measurements used for baseline
    pub baseline_sample_count: usize,
    
    /// Confidence level for baseline
    pub confidence_level: f64,
}

impl TimingAnalyzer {
    /// Create a new timing analyzer
    /// This initializes the analyzer with the specified configuration
    /// and prepares it for timing measurement and analysis.
    pub fn new(max_variance_ns: u64) -> Result<Self> {
        Ok(Self {
            config: SecurityConfig::default(),
            max_variance_ns,
            measurements: Arc::new(Mutex::new(HashMap::new())),
            statistics: Arc::new(Mutex::new(HashMap::new())),
            anomalies: Arc::new(Mutex::new(Vec::new())),
            baselines: Arc::new(Mutex::new(HashMap::new())),
            continuous_monitoring: false,
            monitor_thread: None,
        })
    }
    
    /// Record a timing measurement
    /// This method records a timing measurement for the specified operation
    /// and updates the statistical analysis in real-time.
    pub fn record_timing(&mut self, operation_name: &str, duration_ns: u64) -> Result<()> {
        let measurement = TimingMeasurement {
            operation_name: operation_name.to_string(),
            duration_ns,
            timestamp: SystemTime::now(),
            input_size: 0, // Would be provided by caller in real implementation
            context: self.collect_timing_context()?,
            cpu_cycles: self.get_cpu_cycles(),
            memory_accesses: self.get_memory_accesses(),
        };
        
        // Store measurement
        {
            let mut measurements = self.measurements.lock()
                .map_err(|e| LatticeFoldError::CryptoError(
                    format!("Failed to acquire measurements lock: {}", e)
                ))?;
            
            measurements.entry(operation_name.to_string())
                .or_insert_with(Vec::new)
                .push(measurement.clone());
        }
        
        // Update statistics
        self.update_statistics(operation_name)?;
        
        // Check for anomalies
        self.check_for_anomalies(operation_name, &measurement)?;
        
        Ok(())
    }
    
    /// Check timing consistency for an operation
    /// This method analyzes the timing measurements for an operation
    /// and determines if they meet constant-time requirements.
    pub fn check_timing_consistency(&self, operation_name: &str) -> Result<()> {
        let statistics = self.statistics.lock()
            .map_err(|e| LatticeFoldError::CryptoError(
                format!("Failed to acquire statistics lock: {}", e)
            ))?;
        
        if let Some(stats) = statistics.get(operation_name) {
            // Check if variance exceeds threshold
            if stats.variance_ns > (self.max_variance_ns as f64).powi(2) {
                return Err(LatticeFoldError::CryptoError(format!(
                    "Timing variance {} ns² exceeds maximum {} ns² for operation '{}'",
                    stats.variance_ns, (self.max_variance_ns as f64).powi(2), operation_name
                )));
            }
            
            // Check coefficient of variation
            if stats.coefficient_of_variation > 0.1 { // 10% threshold
                return Err(LatticeFoldError::CryptoError(format!(
                    "Timing coefficient of variation {} exceeds 0.1 for operation '{}'",
                    stats.coefficient_of_variation, operation_name
                )));
            }
            
            // Check for constant-time properties
            if !stats.appears_constant_time {
                return Err(LatticeFoldError::CryptoError(format!(
                    "Operation '{}' does not appear to execute in constant time (confidence: {})",
                    operation_name, stats.constant_time_confidence
                )));
            }
        }
        
        Ok(())
    }
    
    /// Collect timing context information
    /// This method gathers system and execution context information
    /// that may affect timing measurements for analysis purposes.
    fn collect_timing_context(&self) -> Result<TimingContext> {
        Ok(TimingContext {
            thread_id: self.get_thread_id(),
            cpu_core: self.get_cpu_core(),
            system_load: self.get_system_load()?,
            available_memory: self.get_available_memory()?,
            cache_state: self.get_cache_state()?,
            involves_secrets: false, // Would be determined by caller
        })
    }
    
    /// Get current thread ID
    fn get_thread_id(&self) -> u64 {
        // In a real implementation, this would get the actual thread ID
        std::thread::current().id().as_u64().get()
    }
    
    /// Get current CPU core
    fn get_cpu_core(&self) -> Option<u32> {
        // In a real implementation, this would use platform-specific APIs
        // to determine which CPU core the thread is running on
        None
    }
    
    /// Get current system load
    fn get_system_load(&self) -> Result<f64> {
        // In a real implementation, this would read system load average
        // For now, return a placeholder value
        Ok(0.5)
    }
    
    /// Get available memory
    fn get_available_memory(&self) -> Result<u64> {
        // In a real implementation, this would read available system memory
        // For now, return a placeholder value
        Ok(1_000_000_000) // 1GB
    }
    
    /// Get cache state information
    fn get_cache_state(&self) -> Result<CacheState> {
        // In a real implementation, this would use performance counters
        // to get actual cache statistics
        Ok(CacheState {
            l1_hit_rate: 0.95,
            l2_hit_rate: 0.85,
            l3_hit_rate: 0.75,
            tlb_hit_rate: 0.99,
            branch_prediction_accuracy: 0.90,
        })
    }
    
    /// Get CPU cycle count
    fn get_cpu_cycles(&self) -> Option<u64> {
        // In a real implementation, this would use RDTSC or similar
        // to get actual CPU cycle counts
        None
    }
    
    /// Get memory access count
    fn get_memory_accesses(&self) -> Option<u64> {
        // In a real implementation, this would use performance counters
        // to get actual memory access counts
        None
    }
    
    /// Update statistical analysis for an operation
    /// This method recalculates statistics for an operation based on
    /// all collected measurements and updates the analysis results.
    fn update_statistics(&mut self, operation_name: &str) -> Result<()> {
        let measurements = self.measurements.lock()
            .map_err(|e| LatticeFoldError::CryptoError(
                format!("Failed to acquire measurements lock: {}", e)
            ))?;
        
        if let Some(measurements) = measurements.get(operation_name) {
            if measurements.is_empty() {
                return Ok(());
            }
            
            // Calculate basic statistics
            let durations: Vec<u64> = measurements.iter()
                .map(|m| m.duration_ns)
                .collect();
            
            let sample_count = durations.len();
            let min_duration_ns = *durations.iter().min().unwrap();
            let max_duration_ns = *durations.iter().max().unwrap();
            let mean_duration_ns = durations.iter().sum::<u64>() as f64 / sample_count as f64;
            
            // Calculate median
            let mut sorted_durations = durations.clone();
            sorted_durations.sort_unstable();
            let median_duration_ns = if sample_count % 2 == 0 {
                (sorted_durations[sample_count / 2 - 1] + sorted_durations[sample_count / 2]) / 2
            } else {
                sorted_durations[sample_count / 2]
            };
            
            // Calculate variance and standard deviation
            let variance_ns = durations.iter()
                .map(|&d| {
                    let diff = d as f64 - mean_duration_ns;
                    diff * diff
                })
                .sum::<f64>() / sample_count as f64;
            
            let std_deviation_ns = variance_ns.sqrt();
            let coefficient_of_variation = if mean_duration_ns > 0.0 {
                std_deviation_ns / mean_duration_ns
            } else {
                0.0
            };
            
            // Calculate percentiles
            let percentile_95_ns = sorted_durations[(sample_count as f64 * 0.95) as usize];
            let percentile_99_ns = sorted_durations[(sample_count as f64 * 0.99) as usize];
            
            // Calculate skewness and kurtosis
            let skewness = self.calculate_skewness(&durations, mean_duration_ns, std_deviation_ns);
            let kurtosis = self.calculate_kurtosis(&durations, mean_duration_ns, std_deviation_ns);
            
            // Assess constant-time properties
            let (appears_constant_time, constant_time_confidence) = 
                self.assess_constant_time_properties(&durations, variance_ns);
            
            // Detect timing patterns
            let timing_patterns = self.detect_timing_patterns(measurements)?;
            
            // Create statistics object
            let stats = TimingStatistics {
                operation_name: operation_name.to_string(),
                sample_count,
                min_duration_ns,
                max_duration_ns,
                mean_duration_ns,
                median_duration_ns,
                std_deviation_ns,
                variance_ns,
                coefficient_of_variation,
                skewness,
                kurtosis,
                percentile_95_ns,
                percentile_99_ns,
                appears_constant_time,
                constant_time_confidence,
                timing_patterns,
            };
            
            // Store statistics
            let mut statistics = self.statistics.lock()
                .map_err(|e| LatticeFoldError::CryptoError(
                    format!("Failed to acquire statistics lock: {}", e)
                ))?;
            
            statistics.insert(operation_name.to_string(), stats);
        }
        
        Ok(())
    }
    
    /// Calculate skewness of timing distribution
    /// This method calculates the skewness (asymmetry) of the timing
    /// distribution to detect potential timing anomalies.
    fn calculate_skewness(&self, durations: &[u64], mean: f64, std_dev: f64) -> f64 {
        if std_dev == 0.0 || durations.len() < 3 {
            return 0.0;
        }
        
        let n = durations.len() as f64;
        let sum_cubed_deviations = durations.iter()
            .map(|&d| {
                let deviation = (d as f64 - mean) / std_dev;
                deviation.powi(3)
            })
            .sum::<f64>();
        
        (n / ((n - 1.0) * (n - 2.0))) * sum_cubed_deviations
    }
    
    /// Calculate kurtosis of timing distribution
    /// This method calculates the kurtosis (tail heaviness) of the timing
    /// distribution to detect potential timing anomalies.
    fn calculate_kurtosis(&self, durations: &[u64], mean: f64, std_dev: f64) -> f64 {
        if std_dev == 0.0 || durations.len() < 4 {
            return 0.0;
        }
        
        let n = durations.len() as f64;
        let sum_fourth_deviations = durations.iter()
            .map(|&d| {
                let deviation = (d as f64 - mean) / std_dev;
                deviation.powi(4)
            })
            .sum::<f64>();
        
        let kurtosis = (n * (n + 1.0) / ((n - 1.0) * (n - 2.0) * (n - 3.0))) * sum_fourth_deviations;
        let correction = 3.0 * (n - 1.0).powi(2) / ((n - 2.0) * (n - 3.0));
        
        kurtosis - correction // Excess kurtosis
    }
    
    /// Assess constant-time properties of timing measurements
    /// This method analyzes timing measurements to determine if an operation
    /// exhibits constant-time behavior with statistical confidence.
    fn assess_constant_time_properties(&self, durations: &[u64], variance: f64) -> (bool, f64) {
        if durations.len() < 10 {
            return (false, 0.0); // Need sufficient samples
        }
        
        // Check variance against threshold
        let variance_threshold = (self.max_variance_ns as f64).powi(2);
        let variance_ok = variance <= variance_threshold;
        
        // Check for outliers using IQR method
        let mut sorted = durations.to_vec();
        sorted.sort_unstable();
        
        let q1_idx = sorted.len() / 4;
        let q3_idx = 3 * sorted.len() / 4;
        let q1 = sorted[q1_idx] as f64;
        let q3 = sorted[q3_idx] as f64;
        let iqr = q3 - q1;
        
        let lower_bound = q1 - 1.5 * iqr;
        let upper_bound = q3 + 1.5 * iqr;
        
        let outlier_count = durations.iter()
            .filter(|&&d| (d as f64) < lower_bound || (d as f64) > upper_bound)
            .count();
        
        let outlier_ratio = outlier_count as f64 / durations.len() as f64;
        let outliers_ok = outlier_ratio <= 0.05; // Allow up to 5% outliers
        
        // Calculate confidence based on multiple factors
        let mut confidence = 0.0;
        
        if variance_ok {
            confidence += 0.4;
        }
        
        if outliers_ok {
            confidence += 0.3;
        }
        
        // Check distribution normality (simplified)
        let mean = durations.iter().sum::<u64>() as f64 / durations.len() as f64;
        let std_dev = variance.sqrt();
        let cv = if mean > 0.0 { std_dev / mean } else { 0.0 };
        
        if cv <= 0.1 { // Low coefficient of variation
            confidence += 0.3;
        }
        
        let appears_constant_time = confidence >= 0.7;
        
        (appears_constant_time, confidence)
    }
    
    /// Detect timing patterns in measurements
    /// This method analyzes timing measurements to detect patterns that
    /// might indicate timing-based side-channel vulnerabilities.
    fn detect_timing_patterns(&self, measurements: &[TimingMeasurement]) -> Result<Vec<TimingPattern>> {
        let mut patterns = Vec::new();
        
        // Check for input-dependent timing
        if let Some(pattern) = self.detect_input_dependent_timing(measurements)? {
            patterns.push(pattern);
        }
        
        // Check for cache-dependent timing
        if let Some(pattern) = self.detect_cache_dependent_timing(measurements)? {
            patterns.push(pattern);
        }
        
        // Check for system-load-dependent timing
        if let Some(pattern) = self.detect_system_load_dependent_timing(measurements)? {
            patterns.push(pattern);
        }
        
        Ok(patterns)
    }
    
    /// Detect input-dependent timing patterns
    /// This method analyzes timing measurements to detect correlations
    /// between input characteristics and execution timing.
    fn detect_input_dependent_timing(&self, measurements: &[TimingMeasurement]) -> Result<Option<TimingPattern>> {
        if measurements.len() < 20 {
            return Ok(None); // Need sufficient samples
        }
        
        // Group measurements by input size
        let mut size_groups: HashMap<usize, Vec<u64>> = HashMap::new();
        for measurement in measurements {
            size_groups.entry(measurement.input_size)
                .or_insert_with(Vec::new)
                .push(measurement.duration_ns);
        }
        
        if size_groups.len() < 2 {
            return Ok(None); // Need multiple input sizes
        }
        
        // Calculate correlation between input size and timing
        let mut size_timing_pairs: Vec<(f64, f64)> = Vec::new();
        for (size, timings) in &size_groups {
            let mean_timing = timings.iter().sum::<u64>() as f64 / timings.len() as f64;
            size_timing_pairs.push(*size as f64, mean_timing);
        }
        
        let correlation = self.calculate_correlation(&size_timing_pairs);
        
        if correlation.abs() > 0.5 { // Significant correlation
            let confidence = correlation.abs();
            let security_implications = vec![
                "Input size may leak through timing".to_string(),
                "Potential timing-based side-channel vulnerability".to_string(),
            ];
            
            return Ok(Some(TimingPattern {
                pattern_type: PatternType::InputDependent,
                confidence,
                description: format!("Strong correlation ({:.3}) between input size and timing", correlation),
                security_implications,
            }));
        }
        
        Ok(None)
    }
    
    /// Detect cache-dependent timing patterns
    /// This method analyzes timing measurements to detect correlations
    /// between cache behavior and execution timing.
    fn detect_cache_dependent_timing(&self, measurements: &[TimingMeasurement]) -> Result<Option<TimingPattern>> {
        if measurements.len() < 20 {
            return Ok(None);
        }
        
        // Analyze correlation between cache hit rates and timing
        let mut cache_timing_pairs: Vec<(f64, f64)> = Vec::new();
        for measurement in measurements {
            let avg_hit_rate = (measurement.context.cache_state.l1_hit_rate +
                               measurement.context.cache_state.l2_hit_rate +
                               measurement.context.cache_state.l3_hit_rate) / 3.0;
            cache_timing_pairs.push(avg_hit_rate, measurement.duration_ns as f64);
        }
        
        let correlation = self.calculate_correlation(&cache_timing_pairs);
        
        if correlation.abs() > 0.4 { // Moderate correlation threshold for cache effects
            let confidence = correlation.abs();
            let security_implications = vec![
                "Cache behavior affects timing".to_string(),
                "Potential cache-timing side-channel vulnerability".to_string(),
            ];
            
            return Ok(Some(TimingPattern {
                pattern_type: PatternType::CacheDependent,
                confidence,
                description: format!("Correlation ({:.3}) between cache hit rate and timing", correlation),
                security_implications,
            }));
        }
        
        Ok(None)
    }
    
    /// Detect system-load-dependent timing patterns
    /// This method analyzes timing measurements to detect correlations
    /// between system load and execution timing.
    fn detect_system_load_dependent_timing(&self, measurements: &[TimingMeasurement]) -> Result<Option<TimingPattern>> {
        if measurements.len() < 20 {
            return Ok(None);
        }
        
        // Analyze correlation between system load and timing
        let mut load_timing_pairs: Vec<(f64, f64)> = Vec::new();
        for measurement in measurements {
            load_timing_pairs.push(measurement.context.system_load, measurement.duration_ns as f64);
        }
        
        let correlation = self.calculate_correlation(&load_timing_pairs);
        
        if correlation.abs() > 0.6 { // Higher threshold for system load effects
            let confidence = correlation.abs();
            let security_implications = vec![
                "System load significantly affects timing".to_string(),
                "Timing measurements may be unreliable under load".to_string(),
            ];
            
            return Ok(Some(TimingPattern {
                pattern_type: PatternType::SystemLoadDependent,
                confidence,
                description: format!("Strong correlation ({:.3}) between system load and timing", correlation),
                security_implications,
            }));
        }
        
        Ok(None)
    }
    
    /// Calculate Pearson correlation coefficient
    /// This method calculates the correlation between two variables
    /// to detect linear relationships in timing data.
    fn calculate_correlation(&self, pairs: &[(f64, f64)]) -> f64 {
        if pairs.len() < 2 {
            return 0.0;
        }
        
        let n = pairs.len() as f64;
        let sum_x = pairs.iter().map(|(x, _)| x).sum::<f64>();
        let sum_y = pairs.iter().map(|(_, y)| y).sum::<f64>();
        let sum_xy = pairs.iter().map(|(x, y)| x * y).sum::<f64>();
        let sum_x2 = pairs.iter().map(|(x, _)| x * x).sum::<f64>();
        let sum_y2 = pairs.iter().map(|(_, y)| y * y).sum::<f64>();
        
        let numerator = n * sum_xy - sum_x * sum_y;
        let denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)).sqrt();
        
        if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator
        }
    }
    
    /// Check for timing anomalies
    /// This method analyzes a new timing measurement against historical
    /// data to detect potential timing-based security vulnerabilities.
    fn check_for_anomalies(&mut self, operation_name: &str, measurement: &TimingMeasurement) -> Result<()> {
        // Check against baseline if available
        if let Ok(baselines) = self.baselines.lock() {
            if let Some(baseline) = baselines.get(operation_name) {
                let deviation = (measurement.duration_ns as f64 - baseline.expected_mean_ns).abs();
                let z_score = deviation / baseline.expected_std_dev_ns;
                
                if z_score > 3.0 { // 3-sigma rule
                    let anomaly = TimingAnomaly {
                        operation_name: operation_name.to_string(),
                        anomaly_type: AnomalyType::ExcessiveVariance,
                        severity: if z_score > 5.0 { AnomalySeverity::Critical } else { AnomalySeverity::High },
                        description: format!("Timing measurement deviates {:.2} standard deviations from baseline", z_score),
                        triggering_measurements: vec![measurement.clone()],
                        statistical_evidence: StatisticalEvidence {
                            p_value: self.calculate_p_value(z_score),
                            test_statistic: z_score,
                            test_type: StatisticalTest::TTest,
                            confidence_interval: (baseline.expected_mean_ns - 2.0 * baseline.expected_std_dev_ns,
                                                baseline.expected_mean_ns + 2.0 * baseline.expected_std_dev_ns),
                            effect_size: z_score / (baseline.baseline_sample_count as f64).sqrt(),
                            statistical_power: 0.8, // Assumed
                        },
                        detected_at: SystemTime::now(),
                        recommendations: vec![
                            "Investigate potential timing side-channel vulnerability".to_string(),
                            "Review implementation for input-dependent branches".to_string(),
                            "Consider constant-time implementation".to_string(),
                        ],
                    };
                    
                    if let Ok(mut anomalies) = self.anomalies.lock() {
                        anomalies.push(anomaly);
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Calculate p-value for z-score
    /// This method calculates the p-value for a given z-score using
    /// the standard normal distribution approximation.
    fn calculate_p_value(&self, z_score: f64) -> f64 {
        // Simplified p-value calculation using complementary error function approximation
        // In a real implementation, this would use a proper statistical library
        let abs_z = z_score.abs();
        if abs_z > 6.0 {
            0.0
        } else {
            2.0 * (1.0 - self.normal_cdf(abs_z))
        }
    }
    
    /// Approximate normal cumulative distribution function
    /// This is a simplified approximation for demonstration purposes.
    fn normal_cdf(&self, x: f64) -> f64 {
        // Abramowitz and Stegun approximation
        let a1 = 0.254829592;
        let a2 = -0.284496736;
        let a3 = 1.421413741;
        let a4 = -1.453152027;
        let a5 = 1.061405429;
        let p = 0.3275911;
        
        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let x = x.abs() / 2.0_f64.sqrt();
        
        let t = 1.0 / (1.0 + p * x);
        let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
        
        0.5 * (1.0 + sign * y)
    }
    
    /// Get comprehensive timing analysis report
    /// This method generates a comprehensive report of all timing analysis
    /// results including statistics, anomalies, and security assessments.
    pub fn get_analysis_report(&self) -> Result<TimingAnalysisReport> {
        let statistics = self.statistics.lock()
            .map_err(|e| LatticeFoldError::CryptoError(
                format!("Failed to acquire statistics lock: {}", e)
            ))?;
        
        let anomalies = self.anomalies.lock()
            .map_err(|e| LatticeFoldError::CryptoError(
                format!("Failed to acquire anomalies lock: {}", e)
            ))?;
        
        let measurements = self.measurements.lock()
            .map_err(|e| LatticeFoldError::CryptoError(
                format!("Failed to acquire measurements lock: {}", e)
            ))?;
        
        let total_measurements = measurements.values()
            .map(|v| v.len())
            .sum();
        
        let operations_analyzed = statistics.len();
        let constant_time_operations = statistics.values()
            .filter(|s| s.appears_constant_time)
            .count();
        
        let critical_anomalies = anomalies.iter()
            .filter(|a| a.severity == AnomalySeverity::Critical)
            .count();
        
        let high_severity_anomalies = anomalies.iter()
            .filter(|a| a.severity == AnomalySeverity::High)
            .count();
        
        let overall_security_score = self.calculate_overall_security_score(&statistics, &anomalies);
        
        Ok(TimingAnalysisReport {
            total_measurements,
            operations_analyzed,
            constant_time_operations,
            timing_statistics: statistics.clone(),
            detected_anomalies: anomalies.clone(),
            overall_security_score,
            critical_anomalies,
            high_severity_anomalies,
            recommendations: self.generate_recommendations(&statistics, &anomalies),
            analysis_timestamp: SystemTime::now(),
        })
    }
    
    /// Calculate overall security score
    /// This method calculates a comprehensive security score based on
    /// timing analysis results and detected vulnerabilities.
    fn calculate_overall_security_score(&self, 
                                       statistics: &HashMap<String, TimingStatistics>,
                                       anomalies: &[TimingAnomaly]) -> u32 {
        let mut score = 100u32;
        
        // Deduct points for non-constant-time operations
        let total_ops = statistics.len();
        if total_ops > 0 {
            let constant_time_ops = statistics.values()
                .filter(|s| s.appears_constant_time)
                .count();
            
            let constant_time_ratio = constant_time_ops as f64 / total_ops as f64;
            score = score.saturating_sub((50.0 * (1.0 - constant_time_ratio)) as u32);
        }
        
        // Deduct points for anomalies
        for anomaly in anomalies {
            let deduction = match anomaly.severity {
                AnomalySeverity::Low => 2,
                AnomalySeverity::Medium => 5,
                AnomalySeverity::High => 15,
                AnomalySeverity::Critical => 30,
            };
            score = score.saturating_sub(deduction);
        }
        
        // Deduct points for timing patterns indicating vulnerabilities
        for stats in statistics.values() {
            for pattern in &stats.timing_patterns {
                let deduction = match pattern.pattern_type {
                    PatternType::SecretDependent => 25,
                    PatternType::InputDependent => 15,
                    PatternType::CacheDependent => 10,
                    PatternType::BranchDependent => 20,
                    _ => 5,
                };
                score = score.saturating_sub(deduction);
            }
        }
        
        score
    }
    
    /// Generate security recommendations
    /// This method analyzes timing analysis results and generates
    /// specific recommendations for improving timing security.
    fn generate_recommendations(&self,
                               statistics: &HashMap<String, TimingStatistics>,
                               anomalies: &[TimingAnomaly]) -> Vec<String> {
        let mut recommendations = Vec::new();
        
        // Check for non-constant-time operations
        for (op_name, stats) in statistics {
            if !stats.appears_constant_time {
                recommendations.push(format!(
                    "Implement constant-time version of operation '{}' (confidence: {:.2})",
                    op_name, stats.constant_time_confidence
                ));
            }
            
            if stats.coefficient_of_variation > 0.1 {
                recommendations.push(format!(
                    "Reduce timing variance for operation '{}' (CV: {:.3})",
                    op_name, stats.coefficient_of_variation
                ));
            }
        }
        
        // Check for critical anomalies
        let critical_count = anomalies.iter()
            .filter(|a| a.severity == AnomalySeverity::Critical)
            .count();
        
        if critical_count > 0 {
            recommendations.push(format!(
                "Address {} critical timing anomalies immediately",
                critical_count
            ));
        }
        
        // Check for timing patterns
        let mut has_secret_dependent = false;
        let mut has_input_dependent = false;
        
        for stats in statistics.values() {
            for pattern in &stats.timing_patterns {
                match pattern.pattern_type {
                    PatternType::SecretDependent => has_secret_dependent = true,
                    PatternType::InputDependent => has_input_dependent = true,
                    _ => {}
                }
            }
        }
        
        if has_secret_dependent {
            recommendations.push("Eliminate secret-dependent timing variations".to_string());
        }
        
        if has_input_dependent {
            recommendations.push("Consider input-independent algorithms".to_string());
        }
        
        // General recommendations
        if recommendations.is_empty() {
            recommendations.push("Continue monitoring timing behavior".to_string());
            recommendations.push("Establish timing baselines for new operations".to_string());
        }
        
        recommendations
    }
}

/// Comprehensive timing analysis report
#[derive(Debug, Clone)]
pub struct TimingAnalysisReport {
    /// Total number of timing measurements collected
    pub total_measurements: usize,
    
    /// Number of operations analyzed
    pub operations_analyzed: usize,
    
    /// Number of operations that appear to be constant-time
    pub constant_time_operations: usize,
    
    /// Detailed timing statistics for each operation
    pub timing_statistics: HashMap<String, TimingStatistics>,
    
    /// List of detected timing anomalies
    pub detected_anomalies: Vec<TimingAnomaly>,
    
    /// Overall security score (0-100)
    pub overall_security_score: u32,
    
    /// Number of critical anomalies
    pub critical_anomalies: usize,
    
    /// Number of high-severity anomalies
    pub high_severity_anomalies: usize,
    
    /// Security recommendations
    pub recommendations: Vec<String>,
    
    /// When this analysis was performed
    pub analysis_timestamp: SystemTime,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_timing_analyzer_creation() {
        let analyzer = TimingAnalyzer::new(1000);
        assert!(analyzer.is_ok());
        
        let analyzer = analyzer.unwrap();
        assert_eq!(analyzer.max_variance_ns, 1000);
    }
    
    #[test]
    fn test_timing_measurement_recording() {
        let mut analyzer = TimingAnalyzer::new(1000).unwrap();
        
        // Record some measurements
        for i in 0..10 {
            let duration = 1000 + i * 10; // Gradually increasing timing
            analyzer.record_timing("test_operation", duration).unwrap();
        }
        
        // Check that measurements were recorded
        let measurements = analyzer.measurements.lock().unwrap();
        assert!(measurements.contains_key("test_operation"));
        assert_eq!(measurements["test_operation"].len(), 10);
    }
    
    #[test]
    fn test_timing_statistics_calculation() {
        let mut analyzer = TimingAnalyzer::new(1000).unwrap();
        
        // Record measurements with known statistics
        let durations = vec![1000, 1010, 1020, 1030, 1040];
        for duration in &durations {
            analyzer.record_timing("test_op", *duration).unwrap();
        }
        
        let statistics = analyzer.statistics.lock().unwrap();
        let stats = statistics.get("test_op").unwrap();
        
        assert_eq!(stats.sample_count, 5);
        assert_eq!(stats.min_duration_ns, 1000);
        assert_eq!(stats.max_duration_ns, 1040);
        assert_eq!(stats.mean_duration_ns, 1020.0);
        assert_eq!(stats.median_duration_ns, 1020);
    }
    
    #[test]
    fn test_constant_time_assessment() {
        let mut analyzer = TimingAnalyzer::new(100).unwrap(); // Low variance threshold
        
        // Record constant-time measurements
        for _ in 0..20 {
            analyzer.record_timing("constant_op", 1000).unwrap(); // Exactly 1000ns each time
        }
        
        let statistics = analyzer.statistics.lock().unwrap();
        let stats = statistics.get("constant_op").unwrap();
        
        assert!(stats.appears_constant_time);
        assert!(stats.constant_time_confidence > 0.7);
        assert!(stats.variance_ns < 100.0);
    }
    
    #[test]
    fn test_timing_consistency_check() {
        let mut analyzer = TimingAnalyzer::new(100).unwrap();
        
        // Record measurements with low variance (should pass)
        for i in 0..10 {
            analyzer.record_timing("good_op", 1000 + i).unwrap();
        }
        
        let result = analyzer.check_timing_consistency("good_op");
        assert!(result.is_ok());
        
        // Record measurements with high variance (should fail)
        for i in 0..10 {
            analyzer.record_timing("bad_op", 1000 + i * 100).unwrap();
        }
        
        let result = analyzer.check_timing_consistency("bad_op");
        assert!(result.is_err());
    }
    
    #[test]
    fn test_correlation_calculation() {
        let analyzer = TimingAnalyzer::new(1000).unwrap();
        
        // Perfect positive correlation
        let perfect_positive = vec![(1.0, 2.0), (2.0, 4.0), (3.0, 6.0), (4.0, 8.0)];
        let correlation = analyzer.calculate_correlation(&perfect_positive);
        assert!((correlation - 1.0).abs() < 0.001);
        
        // Perfect negative correlation
        let perfect_negative = vec![(1.0, 8.0), (2.0, 6.0), (3.0, 4.0), (4.0, 2.0)];
        let correlation = analyzer.calculate_correlation(&perfect_negative);
        assert!((correlation + 1.0).abs() < 0.001);
        
        // No correlation
        let no_correlation = vec![(1.0, 5.0), (2.0, 5.0), (3.0, 5.0), (4.0, 5.0)];
        let correlation = analyzer.calculate_correlation(&no_correlation);
        assert!(correlation.abs() < 0.001);
    }
    
    #[test]
    fn test_statistical_calculations() {
        let analyzer = TimingAnalyzer::new(1000).unwrap();
        
        // Test skewness calculation
        let symmetric_data = vec![1000, 1010, 1020, 1030, 1040];
        let skewness = analyzer.calculate_skewness(&symmetric_data, 1020.0, 15.81);
        assert!(skewness.abs() < 0.1); // Should be close to 0 for symmetric data
        
        // Test kurtosis calculation
        let kurtosis = analyzer.calculate_kurtosis(&symmetric_data, 1020.0, 15.81);
        // For uniform-like distribution, kurtosis should be negative
        assert!(kurtosis < 0.0);
    }
    
    #[test]
    fn test_analysis_report_generation() {
        let mut analyzer = TimingAnalyzer::new(1000).unwrap();
        
        // Record some measurements
        for i in 0..20 {
            analyzer.record_timing("test_op", 1000 + i).unwrap();
        }
        
        let report = analyzer.get_analysis_report().unwrap();
        
        assert_eq!(report.total_measurements, 20);
        assert_eq!(report.operations_analyzed, 1);
        assert!(report.timing_statistics.contains_key("test_op"));
        assert!(report.overall_security_score <= 100);
    }
    
    #[test]
    fn test_p_value_calculation() {
        let analyzer = TimingAnalyzer::new(1000).unwrap();
        
        // Test p-value for different z-scores
        let p_value_0 = analyzer.calculate_p_value(0.0);
        assert!((p_value_0 - 1.0).abs() < 0.1); // Should be close to 1.0
        
        let p_value_2 = analyzer.calculate_p_value(2.0);
        assert!(p_value_2 < 0.1); // Should be small for z=2
        
        let p_value_large = analyzer.calculate_p_value(10.0);
        assert!(p_value_large < 0.001); // Should be very small for large z
    }
}