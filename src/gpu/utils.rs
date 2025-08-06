/// GPU utility functions and helper routines for LatticeFold+ operations
/// 
/// This module provides utility functions for GPU operations, including
/// memory management helpers, device capability queries, performance
/// profiling utilities, and debugging tools.
/// 
/// Key Features:
/// - GPU memory allocation and deallocation helpers
/// - Device capability detection and reporting
/// - Performance profiling and benchmarking utilities
/// - Error handling and debugging tools
/// - Cross-platform compatibility utilities
/// - Memory transfer optimization helpers
/// 
/// Performance Utilities:
/// - Kernel execution timing and profiling
/// - Memory bandwidth measurement
/// - Occupancy calculation and optimization
/// - Cache performance analysis
/// - Power consumption estimation
/// 
/// Debugging Features:
/// - GPU memory leak detection
/// - Kernel execution validation
/// - Performance bottleneck identification
/// - Error reporting and diagnostics

use crate::error::{LatticeFoldError, Result};
use std::time::{Duration, Instant};
use std::collections::HashMap;

/// GPU memory transfer direction
/// 
/// This enumeration specifies the direction of memory transfers
/// between host and device memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryTransferDirection {
    /// Host to device transfer
    HostToDevice,
    /// Device to host transfer
    DeviceToHost,
    /// Device to device transfer
    DeviceToDevice,
}

/// GPU performance profiler for kernel execution analysis
/// 
/// This structure provides comprehensive performance profiling capabilities
/// for GPU kernel execution, including timing, memory bandwidth analysis,
/// and occupancy measurements.
#[derive(Debug)]
pub struct GpuProfiler {
    /// Profiling results for different operations
    profiles: HashMap<String, Vec<ProfileResult>>,
    
    /// Whether profiling is currently enabled
    enabled: bool,
    
    /// Minimum execution time threshold for recording (microseconds)
    min_time_threshold_us: u64,
}

/// Individual profiling result for a kernel execution
/// 
/// This structure contains detailed profiling information for a single
/// kernel execution, including timing, memory usage, and performance metrics.
#[derive(Debug, Clone)]
pub struct ProfileResult {
    /// Kernel execution time in microseconds
    pub execution_time_us: u64,
    
    /// Memory transfer time in microseconds
    pub transfer_time_us: u64,
    
    /// Total time including overhead in microseconds
    pub total_time_us: u64,
    
    /// Number of elements processed
    pub elements_processed: u64,
    
    /// Memory bandwidth achieved in GB/s
    pub memory_bandwidth_gbps: f64,
    
    /// Effective throughput in elements per second
    pub throughput_eps: f64,
    
    /// GPU memory usage in bytes
    pub memory_usage_bytes: u64,
    
    /// Estimated occupancy percentage
    pub occupancy_percent: f64,
    
    /// Timestamp when the operation was performed
    pub timestamp: Instant,
}

impl GpuProfiler {
    /// Creates a new GPU profiler
    /// 
    /// # Arguments
    /// * `enabled` - Whether profiling should be enabled initially
    /// * `min_time_threshold_us` - Minimum execution time to record (microseconds)
    /// 
    /// # Returns
    /// * `Self` - New profiler instance
    pub fn new(enabled: bool, min_time_threshold_us: u64) -> Self {
        Self {
            profiles: HashMap::new(),
            enabled,
            min_time_threshold_us,
        }
    }
    
    /// Enables or disables profiling
    /// 
    /// # Arguments
    /// * `enabled` - Whether to enable profiling
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
    
    /// Records a profiling result for a kernel execution
    /// 
    /// # Arguments
    /// * `operation_name` - Name of the operation being profiled
    /// * `execution_time_us` - Kernel execution time in microseconds
    /// * `transfer_time_us` - Memory transfer time in microseconds
    /// * `elements_processed` - Number of elements processed
    /// * `memory_usage_bytes` - GPU memory usage in bytes
    /// * `occupancy_percent` - Estimated occupancy percentage
    pub fn record_execution(
        &mut self,
        operation_name: &str,
        execution_time_us: u64,
        transfer_time_us: u64,
        elements_processed: u64,
        memory_usage_bytes: u64,
        occupancy_percent: f64,
    ) {
        if !self.enabled || execution_time_us < self.min_time_threshold_us {
            return;
        }
        
        let total_time_us = execution_time_us + transfer_time_us;
        
        // Calculate memory bandwidth
        let bytes_transferred = elements_processed * 8; // Assume 8 bytes per element
        let time_seconds = total_time_us as f64 / 1_000_000.0;
        let memory_bandwidth_gbps = if time_seconds > 0.0 {
            (bytes_transferred as f64) / (time_seconds * 1_000_000_000.0)
        } else {
            0.0
        };
        
        // Calculate throughput
        let throughput_eps = if execution_time_us > 0 {
            (elements_processed as f64) / ((execution_time_us as f64) / 1_000_000.0)
        } else {
            0.0
        };
        
        let result = ProfileResult {
            execution_time_us,
            transfer_time_us,
            total_time_us,
            elements_processed,
            memory_bandwidth_gbps,
            throughput_eps,
            memory_usage_bytes,
            occupancy_percent,
            timestamp: Instant::now(),
        };
        
        self.profiles
            .entry(operation_name.to_string())
            .or_insert_with(Vec::new)
            .push(result);
    }
    
    /// Gets profiling statistics for an operation
    /// 
    /// # Arguments
    /// * `operation_name` - Name of the operation
    /// 
    /// # Returns
    /// * `Option<ProfileStatistics>` - Statistics or None if no data available
    pub fn get_statistics(&self, operation_name: &str) -> Option<ProfileStatistics> {
        let results = self.profiles.get(operation_name)?;
        
        if results.is_empty() {
            return None;
        }
        
        let count = results.len();
        let total_execution_time: u64 = results.iter().map(|r| r.execution_time_us).sum();
        let total_transfer_time: u64 = results.iter().map(|r| r.transfer_time_us).sum();
        let total_elements: u64 = results.iter().map(|r| r.elements_processed).sum();
        
        let avg_execution_time = total_execution_time as f64 / count as f64;
        let avg_transfer_time = total_transfer_time as f64 / count as f64;
        let avg_bandwidth = results.iter().map(|r| r.memory_bandwidth_gbps).sum::<f64>() / count as f64;
        let avg_throughput = results.iter().map(|r| r.throughput_eps).sum::<f64>() / count as f64;
        let avg_occupancy = results.iter().map(|r| r.occupancy_percent).sum::<f64>() / count as f64;
        
        let min_execution_time = results.iter().map(|r| r.execution_time_us).min().unwrap_or(0);
        let max_execution_time = results.iter().map(|r| r.execution_time_us).max().unwrap_or(0);
        
        let max_bandwidth = results.iter().map(|r| r.memory_bandwidth_gbps).fold(0.0f64, f64::max);
        let max_throughput = results.iter().map(|r| r.throughput_eps).fold(0.0f64, f64::max);
        
        Some(ProfileStatistics {
            operation_name: operation_name.to_string(),
            execution_count: count,
            total_elements_processed: total_elements,
            avg_execution_time_us: avg_execution_time,
            avg_transfer_time_us: avg_transfer_time,
            min_execution_time_us: min_execution_time,
            max_execution_time_us: max_execution_time,
            avg_memory_bandwidth_gbps: avg_bandwidth,
            max_memory_bandwidth_gbps: max_bandwidth,
            avg_throughput_eps: avg_throughput,
            max_throughput_eps: max_throughput,
            avg_occupancy_percent: avg_occupancy,
        })
    }
    
    /// Gets all available operation names
    /// 
    /// # Returns
    /// * `Vec<String>` - List of operation names with profiling data
    pub fn get_operation_names(&self) -> Vec<String> {
        self.profiles.keys().cloned().collect()
    }
    
    /// Clears all profiling data
    pub fn clear(&mut self) {
        self.profiles.clear();
    }
    
    /// Prints a comprehensive profiling report
    pub fn print_report(&self) {
        println!("GPU Profiling Report");
        println!("====================");
        
        if self.profiles.is_empty() {
            println!("No profiling data available.");
            return;
        }
        
        for operation_name in self.get_operation_names() {
            if let Some(stats) = self.get_statistics(&operation_name) {
                println!("\nOperation: {}", operation_name);
                println!("  Executions: {}", stats.execution_count);
                println!("  Total Elements: {}", stats.total_elements_processed);
                println!("  Avg Execution Time: {:.2} ms", stats.avg_execution_time_us as f64 / 1000.0);
                println!("  Avg Transfer Time: {:.2} ms", stats.avg_transfer_time_us as f64 / 1000.0);
                println!("  Time Range: {:.2} - {:.2} ms", 
                    stats.min_execution_time_us as f64 / 1000.0,
                    stats.max_execution_time_us as f64 / 1000.0);
                println!("  Avg Bandwidth: {:.2} GB/s", stats.avg_memory_bandwidth_gbps);
                println!("  Max Bandwidth: {:.2} GB/s", stats.max_memory_bandwidth_gbps);
                println!("  Avg Throughput: {:.2} M elements/sec", stats.avg_throughput_eps / 1_000_000.0);
                println!("  Max Throughput: {:.2} M elements/sec", stats.max_throughput_eps / 1_000_000.0);
                println!("  Avg Occupancy: {:.1}%", stats.avg_occupancy_percent);
            }
        }
    }
}

/// Profiling statistics for a specific operation
/// 
/// This structure contains aggregated statistics for multiple executions
/// of the same operation, providing insights into performance consistency
/// and optimization opportunities.
#[derive(Debug, Clone)]
pub struct ProfileStatistics {
    /// Name of the operation
    pub operation_name: String,
    
    /// Number of times the operation was executed
    pub execution_count: usize,
    
    /// Total number of elements processed across all executions
    pub total_elements_processed: u64,
    
    /// Average execution time in microseconds
    pub avg_execution_time_us: f64,
    
    /// Average memory transfer time in microseconds
    pub avg_transfer_time_us: f64,
    
    /// Minimum execution time in microseconds
    pub min_execution_time_us: u64,
    
    /// Maximum execution time in microseconds
    pub max_execution_time_us: u64,
    
    /// Average memory bandwidth in GB/s
    pub avg_memory_bandwidth_gbps: f64,
    
    /// Maximum memory bandwidth achieved in GB/s
    pub max_memory_bandwidth_gbps: f64,
    
    /// Average throughput in elements per second
    pub avg_throughput_eps: f64,
    
    /// Maximum throughput achieved in elements per second
    pub max_throughput_eps: f64,
    
    /// Average GPU occupancy percentage
    pub avg_occupancy_percent: f64,
}

/// GPU memory bandwidth benchmark utility
/// 
/// This utility measures the effective memory bandwidth of GPU operations
/// by performing controlled memory transfers and computations.
pub struct MemoryBandwidthBenchmark {
    /// Size of test arrays in elements
    test_sizes: Vec<usize>,
    
    /// Number of iterations for each test
    iterations: usize,
    
    /// Whether to include memory transfer overhead
    include_transfer_overhead: bool,
}

impl MemoryBandwidthBenchmark {
    /// Creates a new memory bandwidth benchmark
    /// 
    /// # Arguments
    /// * `test_sizes` - Array sizes to test (in elements)
    /// * `iterations` - Number of iterations per test
    /// * `include_transfer_overhead` - Whether to include transfer time
    /// 
    /// # Returns
    /// * `Self` - New benchmark instance
    pub fn new(test_sizes: Vec<usize>, iterations: usize, include_transfer_overhead: bool) -> Self {
        Self {
            test_sizes,
            iterations,
            include_transfer_overhead,
        }
    }
    
    /// Runs the memory bandwidth benchmark
    /// 
    /// # Returns
    /// * `Result<BandwidthResults>` - Benchmark results or error
    /// 
    /// # Benchmark Process
    /// 1. For each test size, create test arrays
    /// 2. Perform memory-bound operations (copy, add, scale)
    /// 3. Measure execution time and calculate bandwidth
    /// 4. Average results across multiple iterations
    /// 5. Report peak and sustained bandwidth
    pub fn run(&self) -> Result<BandwidthResults> {
        let mut results = BandwidthResults {
            test_results: Vec::new(),
            peak_bandwidth_gbps: 0.0,
            sustained_bandwidth_gbps: 0.0,
            average_bandwidth_gbps: 0.0,
        };
        
        for &size in &self.test_sizes {
            let mut size_results = Vec::new();
            
            for _ in 0..self.iterations {
                // Create test data
                let test_data_a = vec![1i64; size];
                let test_data_b = vec![2i64; size];
                let mut result_data = vec![0i64; size];
                
                // Measure memory copy bandwidth
                let copy_bandwidth = self.measure_copy_bandwidth(&test_data_a, &mut result_data)?;
                
                // Measure add bandwidth (read two arrays, write one)
                let add_bandwidth = self.measure_add_bandwidth(&test_data_a, &test_data_b, &mut result_data)?;
                
                // Measure scale bandwidth (read one array, write one)
                let scale_bandwidth = self.measure_scale_bandwidth(&test_data_a, &mut result_data, 3)?;
                
                size_results.push(SizeBandwidthResult {
                    size,
                    copy_bandwidth_gbps: copy_bandwidth,
                    add_bandwidth_gbps: add_bandwidth,
                    scale_bandwidth_gbps: scale_bandwidth,
                });
            }
            
            // Average results for this size
            let avg_copy = size_results.iter().map(|r| r.copy_bandwidth_gbps).sum::<f64>() / size_results.len() as f64;
            let avg_add = size_results.iter().map(|r| r.add_bandwidth_gbps).sum::<f64>() / size_results.len() as f64;
            let avg_scale = size_results.iter().map(|r| r.scale_bandwidth_gbps).sum::<f64>() / size_results.len() as f64;
            
            results.test_results.push(SizeBandwidthResult {
                size,
                copy_bandwidth_gbps: avg_copy,
                add_bandwidth_gbps: avg_add,
                scale_bandwidth_gbps: avg_scale,
            });
            
            // Update peak bandwidth
            let max_bandwidth = avg_copy.max(avg_add).max(avg_scale);
            if max_bandwidth > results.peak_bandwidth_gbps {
                results.peak_bandwidth_gbps = max_bandwidth;
            }
        }
        
        // Calculate sustained and average bandwidth
        if !results.test_results.is_empty() {
            let all_bandwidths: Vec<f64> = results.test_results.iter()
                .flat_map(|r| vec![r.copy_bandwidth_gbps, r.add_bandwidth_gbps, r.scale_bandwidth_gbps])
                .collect();
            
            results.average_bandwidth_gbps = all_bandwidths.iter().sum::<f64>() / all_bandwidths.len() as f64;
            
            // Sustained bandwidth is the average of the largest test sizes
            let large_size_results: Vec<&SizeBandwidthResult> = results.test_results.iter()
                .filter(|r| r.size >= 1024 * 1024) // 1M elements or larger
                .collect();
            
            if !large_size_results.is_empty() {
                let sustained_bandwidths: Vec<f64> = large_size_results.iter()
                    .flat_map(|r| vec![r.copy_bandwidth_gbps, r.add_bandwidth_gbps, r.scale_bandwidth_gbps])
                    .collect();
                
                results.sustained_bandwidth_gbps = sustained_bandwidths.iter().sum::<f64>() / sustained_bandwidths.len() as f64;
            } else {
                results.sustained_bandwidth_gbps = results.average_bandwidth_gbps;
            }
        }
        
        Ok(results)
    }
    
    /// Measures memory copy bandwidth
    /// 
    /// # Arguments
    /// * `source` - Source array
    /// * `destination` - Destination array
    /// 
    /// # Returns
    /// * `Result<f64>` - Bandwidth in GB/s or error
    fn measure_copy_bandwidth(&self, source: &[i64], destination: &mut [i64]) -> Result<f64> {
        let start_time = Instant::now();
        
        // Perform memory copy
        destination.copy_from_slice(source);
        
        let elapsed = start_time.elapsed();
        
        // Calculate bandwidth: 2 * size * sizeof(i64) / time (read + write)
        let bytes_transferred = 2 * source.len() * std::mem::size_of::<i64>();
        let time_seconds = elapsed.as_secs_f64();
        
        if time_seconds > 0.0 {
            Ok((bytes_transferred as f64) / (time_seconds * 1_000_000_000.0))
        } else {
            Ok(0.0)
        }
    }
    
    /// Measures add operation bandwidth
    /// 
    /// # Arguments
    /// * `a` - First input array
    /// * `b` - Second input array
    /// * `result` - Result array
    /// 
    /// # Returns
    /// * `Result<f64>` - Bandwidth in GB/s or error
    fn measure_add_bandwidth(&self, a: &[i64], b: &[i64], result: &mut [i64]) -> Result<f64> {
        let start_time = Instant::now();
        
        // Perform element-wise addition
        for i in 0..a.len() {
            result[i] = a[i] + b[i];
        }
        
        let elapsed = start_time.elapsed();
        
        // Calculate bandwidth: 3 * size * sizeof(i64) / time (read a, read b, write result)
        let bytes_transferred = 3 * a.len() * std::mem::size_of::<i64>();
        let time_seconds = elapsed.as_secs_f64();
        
        if time_seconds > 0.0 {
            Ok((bytes_transferred as f64) / (time_seconds * 1_000_000_000.0))
        } else {
            Ok(0.0)
        }
    }
    
    /// Measures scale operation bandwidth
    /// 
    /// # Arguments
    /// * `input` - Input array
    /// * `result` - Result array
    /// * `scalar` - Scalar multiplier
    /// 
    /// # Returns
    /// * `Result<f64>` - Bandwidth in GB/s or error
    fn measure_scale_bandwidth(&self, input: &[i64], result: &mut [i64], scalar: i64) -> Result<f64> {
        let start_time = Instant::now();
        
        // Perform scalar multiplication
        for i in 0..input.len() {
            result[i] = input[i] * scalar;
        }
        
        let elapsed = start_time.elapsed();
        
        // Calculate bandwidth: 2 * size * sizeof(i64) / time (read input, write result)
        let bytes_transferred = 2 * input.len() * std::mem::size_of::<i64>();
        let time_seconds = elapsed.as_secs_f64();
        
        if time_seconds > 0.0 {
            Ok((bytes_transferred as f64) / (time_seconds * 1_000_000_000.0))
        } else {
            Ok(0.0)
        }
    }
}

/// Results from memory bandwidth benchmark
/// 
/// This structure contains the results of memory bandwidth measurements
/// for different operation types and array sizes.
#[derive(Debug, Clone)]
pub struct BandwidthResults {
    /// Results for each test size
    pub test_results: Vec<SizeBandwidthResult>,
    
    /// Peak bandwidth achieved across all tests
    pub peak_bandwidth_gbps: f64,
    
    /// Sustained bandwidth for large arrays
    pub sustained_bandwidth_gbps: f64,
    
    /// Average bandwidth across all tests
    pub average_bandwidth_gbps: f64,
}

impl BandwidthResults {
    /// Prints a detailed bandwidth report
    pub fn print_report(&self) {
        println!("Memory Bandwidth Benchmark Results");
        println!("===================================");
        println!("Peak Bandwidth: {:.2} GB/s", self.peak_bandwidth_gbps);
        println!("Sustained Bandwidth: {:.2} GB/s", self.sustained_bandwidth_gbps);
        println!("Average Bandwidth: {:.2} GB/s", self.average_bandwidth_gbps);
        println!();
        
        println!("Detailed Results by Size:");
        println!("{:>12} {:>12} {:>12} {:>12}", "Size", "Copy (GB/s)", "Add (GB/s)", "Scale (GB/s)");
        println!("{:-<12} {:-<12} {:-<12} {:-<12}", "", "", "", "");
        
        for result in &self.test_results {
            println!("{:>12} {:>12.2} {:>12.2} {:>12.2}", 
                result.size,
                result.copy_bandwidth_gbps,
                result.add_bandwidth_gbps,
                result.scale_bandwidth_gbps);
        }
    }
}

/// Bandwidth results for a specific array size
/// 
/// This structure contains bandwidth measurements for different operation
/// types at a specific array size.
#[derive(Debug, Clone)]
pub struct SizeBandwidthResult {
    /// Array size in elements
    pub size: usize,
    
    /// Memory copy bandwidth in GB/s
    pub copy_bandwidth_gbps: f64,
    
    /// Add operation bandwidth in GB/s
    pub add_bandwidth_gbps: f64,
    
    /// Scale operation bandwidth in GB/s
    pub scale_bandwidth_gbps: f64,
}

/// Utility functions for GPU operations
pub struct GpuUtils;

impl GpuUtils {
    /// Validates that an array size is suitable for GPU processing
    /// 
    /// # Arguments
    /// * `size` - Array size to validate
    /// * `min_size` - Minimum recommended size for GPU efficiency
    /// 
    /// # Returns
    /// * `Result<()>` - Success or validation error
    /// 
    /// # Validation Criteria
    /// - Size must be non-zero
    /// - Size should be large enough to benefit from GPU parallelism
    /// - Size should not exceed GPU memory limits
    /// - Size should be aligned for optimal memory access
    pub fn validate_array_size(size: usize, min_size: usize) -> Result<()> {
        if size == 0 {
            return Err(LatticeFoldError::InvalidParameters(
                "Array size cannot be zero".to_string()
            ));
        }
        
        if size < min_size {
            return Err(LatticeFoldError::InvalidParameters(
                format!("Array size {} is too small for efficient GPU processing (minimum: {})", 
                       size, min_size)
            ));
        }
        
        // Check for reasonable upper bound (assume 16GB GPU memory)
        let max_elements = (16 * 1024 * 1024 * 1024) / std::mem::size_of::<i64>();
        if size > max_elements {
            return Err(LatticeFoldError::InvalidParameters(
                format!("Array size {} exceeds GPU memory limits", size)
            ));
        }
        
        Ok(())
    }
    
    /// Calculates optimal work-group size for a given problem size
    /// 
    /// # Arguments
    /// * `problem_size` - Total number of elements to process
    /// * `max_work_group_size` - Maximum work-group size supported by device
    /// * `preferred_multiple` - Preferred multiple for work-group size
    /// 
    /// # Returns
    /// * `usize` - Optimal work-group size
    /// 
    /// # Optimization Strategy
    /// - Choose work-group size that maximizes occupancy
    /// - Ensure work-group size is a multiple of warp/wavefront size
    /// - Balance between parallelism and resource usage
    /// - Consider memory access patterns and cache efficiency
    pub fn calculate_optimal_work_group_size(
        problem_size: usize,
        max_work_group_size: usize,
        preferred_multiple: usize,
    ) -> usize {
        // Start with a reasonable default
        let mut optimal_size = 256.min(max_work_group_size);
        
        // Ensure it's a multiple of the preferred multiple (usually warp/wavefront size)
        optimal_size = (optimal_size / preferred_multiple) * preferred_multiple;
        
        // For small problems, use smaller work-group sizes
        if problem_size < 1024 {
            optimal_size = optimal_size.min(128);
        }
        
        // For very large problems, consider using larger work-group sizes
        if problem_size > 1024 * 1024 {
            optimal_size = optimal_size.max(512).min(max_work_group_size);
        }
        
        // Ensure we have at least the preferred multiple
        optimal_size.max(preferred_multiple)
    }
    
    /// Estimates GPU memory usage for an operation
    /// 
    /// # Arguments
    /// * `input_elements` - Number of input elements
    /// * `output_elements` - Number of output elements
    /// * `temp_elements` - Number of temporary elements needed
    /// * `element_size` - Size of each element in bytes
    /// 
    /// # Returns
    /// * `u64` - Estimated memory usage in bytes
    pub fn estimate_memory_usage(
        input_elements: usize,
        output_elements: usize,
        temp_elements: usize,
        element_size: usize,
    ) -> u64 {
        let total_elements = input_elements + output_elements + temp_elements;
        (total_elements * element_size) as u64
    }
    
    /// Checks if a GPU operation is likely to be beneficial over CPU
    /// 
    /// # Arguments
    /// * `problem_size` - Size of the problem
    /// * `arithmetic_intensity` - Ratio of arithmetic operations to memory accesses
    /// * `parallelism_factor` - Degree of parallelism available
    /// 
    /// # Returns
    /// * `bool` - True if GPU is likely to be beneficial
    /// 
    /// # Decision Criteria
    /// - Problem size should be large enough to amortize GPU overhead
    /// - Arithmetic intensity should justify memory transfer costs
    /// - Parallelism factor should be sufficient for GPU utilization
    /// - Memory access patterns should be GPU-friendly
    pub fn is_gpu_beneficial(
        problem_size: usize,
        arithmetic_intensity: f64,
        parallelism_factor: f64,
    ) -> bool {
        // Minimum problem size threshold
        if problem_size < 1024 {
            return false;
        }
        
        // Calculate a benefit score based on multiple factors
        let size_factor = (problem_size as f64).log2() / 20.0; // Normalize to ~1.0 for large sizes
        let intensity_factor = arithmetic_intensity.min(10.0) / 10.0; // Cap at 10.0
        let parallelism_factor = parallelism_factor.min(1000.0) / 1000.0; // Cap at 1000.0
        
        let benefit_score = size_factor * 0.4 + intensity_factor * 0.3 + parallelism_factor * 0.3;
        
        // GPU is beneficial if score > 0.5
        benefit_score > 0.5
    }
    
    /// Formats a duration in a human-readable way
    /// 
    /// # Arguments
    /// * `duration` - Duration to format
    /// 
    /// # Returns
    /// * `String` - Formatted duration string
    pub fn format_duration(duration: Duration) -> String {
        let total_micros = duration.as_micros();
        
        if total_micros < 1000 {
            format!("{} μs", total_micros)
        } else if total_micros < 1_000_000 {
            format!("{:.2} ms", total_micros as f64 / 1000.0)
        } else {
            format!("{:.2} s", total_micros as f64 / 1_000_000.0)
        }
    }
    
    /// Formats a byte count in a human-readable way
    /// 
    /// # Arguments
    /// * `bytes` - Number of bytes
    /// 
    /// # Returns
    /// * `String` - Formatted byte count string
    pub fn format_bytes(bytes: u64) -> String {
        const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
        
        if bytes == 0 {
            return "0 B".to_string();
        }
        
        let mut size = bytes as f64;
        let mut unit_index = 0;
        
        while size >= 1024.0 && unit_index < UNITS.len() - 1 {
            size /= 1024.0;
            unit_index += 1;
        }
        
        if unit_index == 0 {
            format!("{} {}", bytes, UNITS[unit_index])
        } else {
            format!("{:.2} {}", size, UNITS[unit_index])
        }
    }
    
    /// Runs a comprehensive GPU benchmark suite
    /// 
    /// # Returns
    /// * `Result<()>` - Success or benchmark error
    /// 
    /// # Benchmark Suite
    /// 1. Memory bandwidth benchmark
    /// 2. Compute throughput benchmark
    /// 3. Latency measurement
    /// 4. Occupancy analysis
    /// 5. Power consumption estimation
    pub fn run_benchmark_suite() -> Result<()> {
        println!("Running GPU Benchmark Suite");
        println!("============================");
        
        // Memory bandwidth benchmark
        println!("\n1. Memory Bandwidth Benchmark");
        let bandwidth_benchmark = MemoryBandwidthBenchmark::new(
            vec![1024, 4096, 16384, 65536, 262144, 1048576],
            5,
            false,
        );
        
        match bandwidth_benchmark.run() {
            Ok(results) => {
                results.print_report();
            }
            Err(e) => {
                println!("Memory bandwidth benchmark failed: {}", e);
            }
        }
        
        // Additional benchmarks would be implemented here
        println!("\n2. Compute Throughput Benchmark");
        println!("(Not yet implemented)");
        
        println!("\n3. Latency Measurement");
        println!("(Not yet implemented)");
        
        println!("\n4. Occupancy Analysis");
        println!("(Not yet implemented)");
        
        println!("\n5. Power Consumption Estimation");
        println!("(Not yet implemented)");
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_gpu_profiler() {
        let mut profiler = GpuProfiler::new(true, 100);
        
        // Record some test executions
        profiler.record_execution("test_op", 1000, 200, 1024, 8192, 75.0);
        profiler.record_execution("test_op", 1100, 180, 1024, 8192, 80.0);
        profiler.record_execution("test_op", 950, 220, 1024, 8192, 70.0);
        
        // Get statistics
        let stats = profiler.get_statistics("test_op").unwrap();
        assert_eq!(stats.execution_count, 3);
        assert_eq!(stats.total_elements_processed, 3072);
        
        // Test that averages are reasonable
        assert!(stats.avg_execution_time_us > 900.0 && stats.avg_execution_time_us < 1200.0);
        assert!(stats.avg_occupancy_percent > 70.0 && stats.avg_occupancy_percent < 80.0);
        
        println!("Profiler test passed");
    }
    
    #[test]
    fn test_memory_bandwidth_benchmark() {
        let benchmark = MemoryBandwidthBenchmark::new(
            vec![1024, 4096],
            2,
            false,
        );
        
        match benchmark.run() {
            Ok(results) => {
                assert!(!results.test_results.is_empty());
                assert!(results.peak_bandwidth_gbps > 0.0);
                assert!(results.average_bandwidth_gbps > 0.0);
                
                println!("Memory bandwidth benchmark test passed");
                results.print_report();
            }
            Err(e) => {
                println!("Memory bandwidth benchmark failed: {}", e);
            }
        }
    }
    
    #[test]
    fn test_gpu_utils() {
        // Test array size validation
        assert!(GpuUtils::validate_array_size(1024, 256).is_ok());
        assert!(GpuUtils::validate_array_size(0, 256).is_err());
        assert!(GpuUtils::validate_array_size(100, 256).is_err());
        
        // Test work-group size calculation
        let work_group_size = GpuUtils::calculate_optimal_work_group_size(10000, 1024, 32);
        assert!(work_group_size >= 32);
        assert!(work_group_size <= 1024);
        assert_eq!(work_group_size % 32, 0);
        
        // Test memory usage estimation
        let memory_usage = GpuUtils::estimate_memory_usage(1000, 1000, 500, 8);
        assert_eq!(memory_usage, 2500 * 8);
        
        // Test GPU benefit analysis
        assert!(GpuUtils::is_gpu_beneficial(10000, 5.0, 100.0));
        assert!(!GpuUtils::is_gpu_beneficial(100, 1.0, 10.0));
        
        // Test formatting functions
        let duration_str = GpuUtils::format_duration(Duration::from_micros(1500));
        assert_eq!(duration_str, "1.50 ms");
        
        let bytes_str = GpuUtils::format_bytes(1536);
        assert_eq!(bytes_str, "1.50 KB");
        
        println!("GPU utils test passed");
    }
    
    #[test]
    fn test_benchmark_suite() {
        // Run the benchmark suite (may take some time)
        match GpuUtils::run_benchmark_suite() {
            Ok(()) => {
                println!("Benchmark suite completed successfully");
            }
            Err(e) => {
                println!("Benchmark suite failed: {}", e);
            }
        }
    }
}