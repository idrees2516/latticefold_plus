/// Performance Analysis Documentation Generator
/// 
/// This module implements comprehensive performance analysis documentation
/// generation as specified in task 15.2. It provides:
/// 
/// - Automated performance analysis report generation
/// - Bottleneck identification and analysis documentation
/// - Performance optimization recommendations
/// - Comparative analysis documentation against baselines
/// - Scalability analysis and projections
/// - Memory usage analysis and optimization guidance
/// - GPU acceleration analysis and recommendations
/// 
/// The documentation generator creates detailed, actionable reports that
/// help developers understand performance characteristics, identify
/// optimization opportunities, and track performance improvements over time.

use crate::error::{LatticeFoldError, Result};
use std::{
    collections::HashMap,
    fs::{File, create_dir_all},
    io::{Write, BufWriter},
    path::{Path, PathBuf},
    time::{Duration, SystemTime},
};
use serde::{Serialize, Deserialize};

/// Performance analysis documentation configuration
/// 
/// Configures the documentation generation including output formats,
/// analysis depth, visualization options, and report customization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentationConfig {
    /// Output directory for generated documentation
    pub output_directory: PathBuf,
    
    /// Documentation formats to generate
    pub output_formats: Vec<OutputFormat>,
    
    /// Analysis depth level
    pub analysis_depth: AnalysisDepth,
    
    /// Include visualizations and charts
    pub include_visualizations: bool,
    
    /// Include code examples and recommendations
    pub include_code_examples: bool,
    
    /// Include comparative analysis
    pub include_comparative_analysis: bool,
    
    /// Include executive summary
    pub include_executive_summary: bool,
    
    /// Custom report sections to include
    pub custom_sections: Vec<String>,
    
    /// Report branding and styling
    pub branding_config: BrandingConfig,
}

/// Documentation output formats
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutputFormat {
    /// Markdown format for GitHub/GitLab
    Markdown,
    /// HTML format with styling
    Html,
    /// PDF format for formal reports
    Pdf,
    /// JSON format for programmatic access
    Json,
    /// Plain text format
    Text,
}

/// Analysis depth levels
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnalysisDepth {
    /// Basic analysis with key metrics only
    Basic,
    /// Standard analysis with detailed metrics and recommendations
    Standard,
    /// Comprehensive analysis with deep dive into all aspects
    Comprehensive,
    /// Expert analysis with advanced statistical analysis
    Expert,
}

/// Report branding and styling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrandingConfig {
    /// Report title
    pub title: String,
    
    /// Organization name
    pub organization: String,
    
    /// Report author
    pub author: String,
    
    /// Custom CSS for HTML reports
    pub custom_css: Option<String>,
    
    /// Logo path for branded reports
    pub logo_path: Option<PathBuf>,
    
    /// Color scheme for visualizations
    pub color_scheme: ColorScheme,
}

/// Color scheme for visualizations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColorScheme {
    /// Primary color (hex)
    pub primary: String,
    
    /// Secondary color (hex)
    pub secondary: String,
    
    /// Success color (hex)
    pub success: String,
    
    /// Warning color (hex)
    pub warning: String,
    
    /// Error color (hex)
    pub error: String,
}

impl Default for DocumentationConfig {
    fn default() -> Self {
        Self {
            output_directory: PathBuf::from("performance_reports"),
            output_formats: vec![OutputFormat::Markdown, OutputFormat::Html],
            analysis_depth: AnalysisDepth::Standard,
            include_visualizations: true,
            include_code_examples: true,
            include_comparative_analysis: true,
            include_executive_summary: true,
            custom_sections: Vec::new(),
            branding_config: BrandingConfig {
                title: "LatticeFold+ Performance Analysis Report".to_string(),
                organization: "LatticeFold+ Development Team".to_string(),
                author: "Performance Analysis System".to_string(),
                custom_css: None,
                logo_path: None,
                color_scheme: ColorScheme {
                    primary: "#2563eb".to_string(),
                    secondary: "#64748b".to_string(),
                    success: "#16a34a".to_string(),
                    warning: "#d97706".to_string(),
                    error: "#dc2626".to_string(),
                },
            },
        }
    }
}/// Per
formance analysis data structures for documentation
/// 
/// These structures represent the comprehensive performance analysis
/// results that will be documented and reported.

/// Comprehensive performance analysis results
/// 
/// Contains all performance analysis data including benchmarks,
/// comparisons, bottlenecks, and recommendations for documentation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceAnalysisResults {
    /// Analysis metadata
    pub metadata: AnalysisMetadata,
    
    /// Executive summary of key findings
    pub executive_summary: ExecutiveSummary,
    
    /// Detailed benchmark results
    pub benchmark_results: BenchmarkResults,
    
    /// Baseline comparison analysis
    pub baseline_comparisons: BaselineComparisons,
    
    /// Scalability analysis results
    pub scalability_analysis: ScalabilityAnalysis,
    
    /// Memory usage analysis
    pub memory_analysis: MemoryAnalysis,
    
    /// GPU acceleration analysis
    pub gpu_analysis: Option<GpuAnalysis>,
    
    /// Performance bottleneck identification
    pub bottleneck_analysis: BottleneckAnalysis,
    
    /// Optimization recommendations
    pub optimization_recommendations: OptimizationRecommendations,
    
    /// Performance regression analysis
    pub regression_analysis: RegressionAnalysis,
    
    /// Historical performance trends
    pub trend_analysis: TrendAnalysis,
}

/// Analysis metadata and context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisMetadata {
    /// Analysis timestamp
    pub timestamp: SystemTime,
    
    /// Analysis duration
    pub analysis_duration: Duration,
    
    /// Hardware configuration
    pub hardware_config: HardwareConfiguration,
    
    /// Software configuration
    pub software_config: SoftwareConfiguration,
    
    /// Test configuration
    pub test_config: TestConfiguration,
    
    /// Analysis version
    pub analysis_version: String,
    
    /// Data quality metrics
    pub data_quality: DataQuality,
}

/// Hardware configuration details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareConfiguration {
    /// CPU specifications
    pub cpu: CpuSpecs,
    
    /// Memory specifications
    pub memory: MemorySpecs,
    
    /// GPU specifications (if available)
    pub gpu: Option<GpuSpecs>,
    
    /// Storage specifications
    pub storage: StorageSpecs,
    
    /// Network specifications
    pub network: NetworkSpecs,
}

/// CPU specifications
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuSpecs {
    /// CPU model name
    pub model: String,
    
    /// Number of cores
    pub cores: usize,
    
    /// Number of threads
    pub threads: usize,
    
    /// Base frequency in GHz
    pub base_frequency: f64,
    
    /// Boost frequency in GHz
    pub boost_frequency: f64,
    
    /// Cache sizes (L1, L2, L3) in KB
    pub cache_sizes: Vec<usize>,
    
    /// Instruction set extensions
    pub instruction_sets: Vec<String>,
}

/// Memory specifications
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySpecs {
    /// Total memory size in GB
    pub total_size_gb: usize,
    
    /// Memory type (DDR4, DDR5, etc.)
    pub memory_type: String,
    
    /// Memory frequency in MHz
    pub frequency_mhz: usize,
    
    /// Memory bandwidth in GB/s
    pub bandwidth_gbps: f64,
    
    /// Number of memory channels
    pub channels: usize,
}

/// GPU specifications
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuSpecs {
    /// GPU model name
    pub model: String,
    
    /// GPU memory size in GB
    pub memory_gb: usize,
    
    /// GPU memory bandwidth in GB/s
    pub memory_bandwidth_gbps: f64,
    
    /// Compute capability version
    pub compute_capability: String,
    
    /// Number of streaming multiprocessors
    pub sm_count: usize,
    
    /// Base clock frequency in MHz
    pub base_clock_mhz: usize,
    
    /// Memory clock frequency in MHz
    pub memory_clock_mhz: usize,
}

/// Storage specifications
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageSpecs {
    /// Storage type (SSD, NVMe, HDD)
    pub storage_type: String,
    
    /// Storage capacity in GB
    pub capacity_gb: usize,
    
    /// Read bandwidth in MB/s
    pub read_bandwidth_mbps: f64,
    
    /// Write bandwidth in MB/s
    pub write_bandwidth_mbps: f64,
    
    /// Random read IOPS
    pub random_read_iops: usize,
    
    /// Random write IOPS
    pub random_write_iops: usize,
}

/// Network specifications
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkSpecs {
    /// Network interface type
    pub interface_type: String,
    
    /// Network bandwidth in Mbps
    pub bandwidth_mbps: f64,
    
    /// Network latency in ms
    pub latency_ms: f64,
}

/// Software configuration details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SoftwareConfiguration {
    /// Operating system details
    pub operating_system: OperatingSystemInfo,
    
    /// Compiler information
    pub compiler: CompilerInfo,
    
    /// Runtime environment
    pub runtime: RuntimeInfo,
    
    /// Dependencies and versions
    pub dependencies: HashMap<String, String>,
}

/// Operating system information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperatingSystemInfo {
    /// OS name (Linux, Windows, macOS)
    pub name: String,
    
    /// OS version
    pub version: String,
    
    /// Kernel version
    pub kernel_version: String,
    
    /// Architecture (x86_64, aarch64)
    pub architecture: String,
}

/// Compiler information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompilerInfo {
    /// Compiler name (rustc, gcc, clang)
    pub name: String,
    
    /// Compiler version
    pub version: String,
    
    /// Optimization level
    pub optimization_level: String,
    
    /// Compilation flags
    pub flags: Vec<String>,
    
    /// Target triple
    pub target_triple: String,
}

/// Runtime environment information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeInfo {
    /// Runtime name (Rust runtime, JVM, etc.)
    pub name: String,
    
    /// Runtime version
    pub version: String,
    
    /// Runtime configuration
    pub configuration: HashMap<String, String>,
}

/// Test configuration details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestConfiguration {
    /// Test parameters used
    pub parameters: HashMap<String, String>,
    
    /// Test duration
    pub duration: Duration,
    
    /// Number of iterations
    pub iterations: usize,
    
    /// Warm-up iterations
    pub warmup_iterations: usize,
    
    /// Statistical confidence level
    pub confidence_level: f64,
    
    /// Random seed for reproducibility
    pub random_seed: u64,
}

/// Data quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataQuality {
    /// Measurement accuracy score (0.0 to 1.0)
    pub accuracy_score: f64,
    
    /// Measurement precision score (0.0 to 1.0)
    pub precision_score: f64,
    
    /// Data completeness percentage
    pub completeness_percentage: f64,
    
    /// Statistical significance of results
    pub statistical_significance: f64,
    
    /// Confidence intervals for key metrics
    pub confidence_intervals: HashMap<String, (f64, f64)>,
    
    /// Outlier detection results
    pub outliers_detected: usize,
    
    /// Data validation results
    pub validation_results: Vec<ValidationResult>,
}

/// Data validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    /// Validation test name
    pub test_name: String,
    
    /// Validation result (passed/failed)
    pub passed: bool,
    
    /// Validation message
    pub message: String,
    
    /// Validation severity
    pub severity: ValidationSeverity,
}

/// Validation severity levels
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValidationSeverity {
    /// Information only
    Info,
    /// Warning about potential issues
    Warning,
    /// Error that affects analysis quality
    Error,
    /// Critical error that invalidates results
    Critical,
}

/// Executive summary of performance analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutiveSummary {
    /// Overall performance score (0-100)
    pub overall_score: f64,
    
    /// Performance grade (A, B, C, D, F)
    pub performance_grade: PerformanceGrade,
    
    /// Key findings summary
    pub key_findings: Vec<KeyFinding>,
    
    /// Critical issues identified
    pub critical_issues: Vec<CriticalIssue>,
    
    /// Top optimization opportunities
    pub top_optimizations: Vec<OptimizationOpportunity>,
    
    /// Performance vs. targets
    pub target_comparison: TargetComparison,
    
    /// Executive recommendations
    pub executive_recommendations: Vec<ExecutiveRecommendation>,
}

/// Performance grade classification
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PerformanceGrade {
    /// Excellent performance (90-100)
    A,
    /// Good performance (80-89)
    B,
    /// Acceptable performance (70-79)
    C,
    /// Poor performance (60-69)
    D,
    /// Unacceptable performance (<60)
    F,
}

/// Key finding from performance analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyFinding {
    /// Finding title
    pub title: String,
    
    /// Finding description
    pub description: String,
    
    /// Finding impact level
    pub impact: ImpactLevel,
    
    /// Supporting metrics
    pub supporting_metrics: Vec<SupportingMetric>,
    
    /// Confidence level in the finding
    pub confidence: f64,
}

/// Impact level classification
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ImpactLevel {
    /// Low impact
    Low,
    /// Medium impact
    Medium,
    /// High impact
    High,
    /// Critical impact
    Critical,
}

/// Supporting metric for findings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SupportingMetric {
    /// Metric name
    pub name: String,
    
    /// Metric value
    pub value: f64,
    
    /// Metric unit
    pub unit: String,
    
    /// Comparison to baseline/target
    pub comparison: MetricComparison,
}

/// Metric comparison result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricComparison {
    /// Baseline/target value
    pub baseline_value: f64,
    
    /// Improvement factor (>1.0 is better)
    pub improvement_factor: f64,
    
    /// Percentage change
    pub percentage_change: f64,
    
    /// Comparison status
    pub status: ComparisonStatus,
}

/// Comparison status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComparisonStatus {
    /// Significantly better than baseline
    SignificantlyBetter,
    /// Better than baseline
    Better,
    /// Similar to baseline
    Similar,
    /// Worse than baseline
    Worse,
    /// Significantly worse than baseline
    SignificantlyWorse,
}

/// Critical issue identification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CriticalIssue {
    /// Issue title
    pub title: String,
    
    /// Issue description
    pub description: String,
    
    /// Issue severity
    pub severity: IssueSeverity,
    
    /// Affected components
    pub affected_components: Vec<String>,
    
    /// Performance impact
    pub performance_impact: f64,
    
    /// Recommended actions
    pub recommended_actions: Vec<String>,
    
    /// Estimated resolution effort
    pub resolution_effort: ResolutionEffort,
}

/// Issue severity levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum IssueSeverity {
    /// Low severity issue
    Low,
    /// Medium severity issue
    Medium,
    /// High severity issue
    High,
    /// Critical severity issue
    Critical,
}

/// Resolution effort estimation
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResolutionEffort {
    /// Low effort (hours)
    Low,
    /// Medium effort (days)
    Medium,
    /// High effort (weeks)
    High,
    /// Very high effort (months)
    VeryHigh,
}

/// Optimization opportunity identification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationOpportunity {
    /// Opportunity title
    pub title: String,
    
    /// Opportunity description
    pub description: String,
    
    /// Potential performance improvement
    pub potential_improvement: f64,
    
    /// Implementation complexity
    pub complexity: ImplementationComplexity,
    
    /// Estimated implementation time
    pub implementation_time: Duration,
    
    /// Priority level
    pub priority: PriorityLevel,
    
    /// Technical approach
    pub technical_approach: String,
    
    /// Expected benefits
    pub expected_benefits: Vec<String>,
    
    /// Implementation risks
    pub risks: Vec<String>,
}

/// Implementation complexity levels
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImplementationComplexity {
    /// Simple implementation
    Simple,
    /// Moderate implementation complexity
    Moderate,
    /// Complex implementation
    Complex,
    /// Very complex implementation
    VeryComplex,
}

/// Priority levels for optimizations
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum PriorityLevel {
    /// Low priority
    Low,
    /// Medium priority
    Medium,
    /// High priority
    High,
    /// Critical priority
    Critical,
}

/// Performance vs. targets comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetComparison {
    /// Performance targets
    pub targets: HashMap<String, PerformanceTarget>,
    
    /// Actual vs. target results
    pub results: HashMap<String, TargetResult>,
    
    /// Overall target achievement percentage
    pub overall_achievement: f64,
    
    /// Targets met count
    pub targets_met: usize,
    
    /// Total targets count
    pub total_targets: usize,
}

/// Performance target definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTarget {
    /// Target name
    pub name: String,
    
    /// Target value
    pub target_value: f64,
    
    /// Target unit
    pub unit: String,
    
    /// Target type (minimum, maximum, exact)
    pub target_type: TargetType,
    
    /// Target priority
    pub priority: PriorityLevel,
}

/// Target type classification
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TargetType {
    /// Minimum acceptable value
    Minimum,
    /// Maximum acceptable value
    Maximum,
    /// Exact target value
    Exact,
    /// Range target (min-max)
    Range(f64, f64),
}

/// Target achievement result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetResult {
    /// Actual measured value
    pub actual_value: f64,
    
    /// Target achievement status
    pub status: TargetStatus,
    
    /// Achievement percentage
    pub achievement_percentage: f64,
    
    /// Gap to target
    pub gap_to_target: f64,
}

/// Target achievement status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TargetStatus {
    /// Target exceeded
    Exceeded,
    /// Target met
    Met,
    /// Target nearly met (within 10%)
    NearlyMet,
    /// Target missed
    Missed,
    /// Target significantly missed (>50% gap)
    SignificantlyMissed,
}

/// Executive recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutiveRecommendation {
    /// Recommendation title
    pub title: String,
    
    /// Recommendation description
    pub description: String,
    
    /// Business impact
    pub business_impact: String,
    
    /// Technical impact
    pub technical_impact: String,
    
    /// Implementation timeline
    pub timeline: Duration,
    
    /// Resource requirements
    pub resource_requirements: ResourceRequirements,
    
    /// Success metrics
    pub success_metrics: Vec<String>,
}

/// Resource requirements for recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirements {
    /// Engineering effort in person-days
    pub engineering_days: f64,
    
    /// Required skill sets
    pub required_skills: Vec<String>,
    
    /// Hardware requirements
    pub hardware_requirements: Vec<String>,
    
    /// Budget requirements
    pub budget_estimate: Option<f64>,
}

/// Detailed benchmark results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResults {
    /// Individual benchmark measurements
    pub measurements: HashMap<String, BenchmarkMeasurement>,
    
    /// Benchmark categories and summaries
    pub categories: HashMap<String, CategorySummary>,
    
    /// Performance metrics aggregation
    pub aggregated_metrics: AggregatedMetrics,
    
    /// Statistical analysis results
    pub statistical_analysis: StatisticalAnalysisResults,
}

/// Individual benchmark measurement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkMeasurement {
    /// Measurement name
    pub name: String,
    
    /// Measurement value
    pub value: f64,
    
    /// Measurement unit
    pub unit: String,
    
    /// Measurement timestamp
    pub timestamp: SystemTime,
    
    /// Measurement parameters
    pub parameters: HashMap<String, String>,
    
    /// Statistical properties
    pub statistics: MeasurementStatistics,
    
    /// Quality indicators
    pub quality: MeasurementQuality,
}

/// Measurement statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementStatistics {
    /// Mean value
    pub mean: f64,
    
    /// Standard deviation
    pub std_dev: f64,
    
    /// Minimum value
    pub min: f64,
    
    /// Maximum value
    pub max: f64,
    
    /// Median value
    pub median: f64,
    
    /// 95th percentile
    pub p95: f64,
    
    /// 99th percentile
    pub p99: f64,
    
    /// Coefficient of variation
    pub coefficient_of_variation: f64,
}

/// Measurement quality indicators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementQuality {
    /// Measurement reliability score (0.0 to 1.0)
    pub reliability: f64,
    
    /// Measurement precision score (0.0 to 1.0)
    pub precision: f64,
    
    /// Number of samples
    pub sample_count: usize,
    
    /// Confidence interval
    pub confidence_interval: (f64, f64),
    
    /// Outliers detected
    pub outliers: usize,
}

/// Category summary for benchmark results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategorySummary {
    /// Category name
    pub name: String,
    
    /// Category description
    pub description: String,
    
    /// Number of measurements in category
    pub measurement_count: usize,
    
    /// Category performance score
    pub performance_score: f64,
    
    /// Key metrics for the category
    pub key_metrics: Vec<KeyMetric>,
    
    /// Category-specific insights
    pub insights: Vec<String>,
}

/// Key metric definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyMetric {
    /// Metric name
    pub name: String,
    
    /// Metric value
    pub value: f64,
    
    /// Metric unit
    pub unit: String,
    
    /// Metric importance weight
    pub weight: f64,
    
    /// Metric trend (improving, stable, degrading)
    pub trend: MetricTrend,
}

/// Metric trend classification
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MetricTrend {
    /// Metric is improving over time
    Improving,
    /// Metric is stable
    Stable,
    /// Metric is degrading over time
    Degrading,
    /// Insufficient data to determine trend
    Unknown,
}

/// Aggregated performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedMetrics {
    /// Overall throughput metrics
    pub throughput: ThroughputMetrics,
    
    /// Overall latency metrics
    pub latency: LatencyMetrics,
    
    /// Resource utilization metrics
    pub resource_utilization: ResourceUtilizationMetrics,
    
    /// Efficiency metrics
    pub efficiency: EfficiencyMetrics,
}

/// Throughput metrics aggregation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThroughputMetrics {
    /// Operations per second
    pub operations_per_second: f64,
    
    /// Constraints processed per second
    pub constraints_per_second: f64,
    
    /// Proofs generated per second
    pub proofs_per_second: f64,
    
    /// Verifications per second
    pub verifications_per_second: f64,
    
    /// Data processing rate (MB/s)
    pub data_processing_rate: f64,
}

/// Latency metrics aggregation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyMetrics {
    /// Average proof generation time
    pub avg_proof_time: Duration,
    
    /// Average verification time
    pub avg_verification_time: Duration,
    
    /// Average setup time
    pub avg_setup_time: Duration,
    
    /// End-to-end latency
    pub end_to_end_latency: Duration,
    
    /// Latency percentiles
    pub latency_percentiles: HashMap<String, Duration>,
}

/// Resource utilization metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUtilizationMetrics {
    /// CPU utilization percentage
    pub cpu_utilization: f64,
    
    /// Memory utilization percentage
    pub memory_utilization: f64,
    
    /// GPU utilization percentage (if available)
    pub gpu_utilization: Option<f64>,
    
    /// Network utilization percentage
    pub network_utilization: f64,
    
    /// Storage utilization percentage
    pub storage_utilization: f64,
}

/// Efficiency metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EfficiencyMetrics {
    /// Performance per watt (operations/watt)
    pub performance_per_watt: f64,
    
    /// Performance per dollar (operations/dollar)
    pub performance_per_dollar: f64,
    
    /// Memory efficiency (operations/MB)
    pub memory_efficiency: f64,
    
    /// CPU efficiency (operations/core)
    pub cpu_efficiency: f64,
    
    /// Overall system efficiency score
    pub overall_efficiency: f64,
}

/// Statistical analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalAnalysisResults {
    /// Correlation analysis between metrics
    pub correlations: HashMap<String, HashMap<String, f64>>,
    
    /// Regression analysis results
    pub regression_analysis: HashMap<String, RegressionAnalysisResult>,
    
    /// Variance analysis
    pub variance_analysis: VarianceAnalysis,
    
    /// Hypothesis testing results
    pub hypothesis_tests: Vec<HypothesisTestResult>,
    
    /// Confidence intervals for key metrics
    pub confidence_intervals: HashMap<String, (f64, f64)>,
}

/// Regression analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionAnalysisResult {
    /// Dependent variable name
    pub dependent_variable: String,
    
    /// Independent variables
    pub independent_variables: Vec<String>,
    
    /// Regression coefficients
    pub coefficients: Vec<f64>,
    
    /// R-squared value
    pub r_squared: f64,
    
    /// Adjusted R-squared value
    pub adjusted_r_squared: f64,
    
    /// P-values for coefficients
    pub p_values: Vec<f64>,
    
    /// Regression equation
    pub equation: String,
}

/// Variance analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VarianceAnalysis {
    /// Within-group variance
    pub within_group_variance: f64,
    
    /// Between-group variance
    pub between_group_variance: f64,
    
    /// F-statistic
    pub f_statistic: f64,
    
    /// P-value for F-test
    pub p_value: f64,
    
    /// Variance components
    pub variance_components: HashMap<String, f64>,
}

/// Hypothesis test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HypothesisTestResult {
    /// Test name
    pub test_name: String,
    
    /// Null hypothesis
    pub null_hypothesis: String,
    
    /// Alternative hypothesis
    pub alternative_hypothesis: String,
    
    /// Test statistic
    pub test_statistic: f64,
    
    /// P-value
    pub p_value: f64,
    
    /// Critical value
    pub critical_value: f64,
    
    /// Test result (reject/fail to reject null)
    pub result: HypothesisTestResult,
    
    /// Confidence level
    pub confidence_level: f64,
}

/// Hypothesis test conclusion
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum HypothesisTestConclusion {
    /// Reject null hypothesis
    RejectNull,
    /// Fail to reject null hypothesis
    FailToRejectNull,
    /// Inconclusive result
    Inconclusive,
}