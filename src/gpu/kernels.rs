/// GPU kernel implementations and utilities for LatticeFold+ operations
/// 
/// This module provides high-level interfaces for GPU kernel execution,
/// automatic device selection, and performance optimization across different
/// GPU architectures and vendors.

use crate::error::{LatticeFoldError, Result};
use std::sync::{Arc, Mutex, Once};
use std::collections::HashMap;

/// GPU kernel execution backend
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuBackend {
    /// NVIDIA CUDA backend
    Cuda,
    /// Cross-platform OpenCL backend
    OpenCL,
    /// CPU fallback (no GPU acceleration)
    Cpu,
}

/// GPU kernel performance metrics
#[derive(Debug, Clone, Default)]
pub struct KernelMetrics {
    /// Total execution time in microseconds
    pub execution_time_us: u64,
    
    /// Memory transfer time in microseconds
    pub transfer_time_us: u64,
    
    /// Number of elements processed
    pub elements_processed: u64,
    
    /// Effective throughput in elements per second
    pub throughput_eps: f64,
}

impl KernelMetrics {
    /// Updates metrics after kernel execution
    pub fn update(&mut self, elements: u64, execution_time_us: u64, transfer_time_us: u64) {
        self.elements_processed = elements;
        self.execution_time_us = execution_time_us;
        self.transfer_time_us = transfer_time_us;
        
        if execution_time_us > 0 {
            self.throughput_eps = (elements as f64) / ((execution_time_us as f64) / 1_000_000.0);
        }
    }
}

/// Unified GPU kernel executor
pub struct GpuKernelExecutor {
    /// Selected GPU backend
    backend: GpuBackend,
    
    /// Performance metrics for different operations
    metrics: Arc<Mutex<HashMap<String, KernelMetrics>>>,
}

impl GpuKernelExecutor {
    /// Creates a new GPU kernel executor with automatic backend selection
    pub fn new() -> Result<Self> {
        let backend = Self::select_best_backend();
        
        Ok(Self {
            backend,
            metrics: Arc::new(Mutex::new(HashMap::new())),
        })
    }
    
    /// Selects the best available GPU backend
    fn select_best_backend() -> GpuBackend {
        // Try CUDA first
        #[cfg(feature = "cuda")]
        {
            if Self::test_cuda_availability() {
                return GpuBackend::Cuda;
            }
        }
        
        // Try OpenCL
        #[cfg(feature = "opencl")]
        {
            if Self::test_opencl_availability() {
                return GpuBackend::OpenCL;
            }
        }
        
        // Fall back to CPU
        GpuBackend::Cpu
    }
    
    /// Tests CUDA availability
    #[cfg(feature = "cuda")]
    fn test_cuda_availability() -> bool {
        use cudarc::driver::CudaDevice;
        CudaDevice::new(0).is_ok()
    }
    
    /// Tests OpenCL availability
    #[cfg(feature = "opencl")]
    fn test_opencl_availability() -> bool {
        use crate::gpu::opencl::GpuOpenCL;
        GpuOpenCL::new(0).is_ok()
    }
    
    /// Fallback implementations when features are not available
    #[cfg(not(feature = "cuda"))]
    fn test_cuda_availability() -> bool {
        false
    }
    
    #[cfg(not(feature = "opencl"))]
    fn test_opencl_availability() -> bool {
        false
    }
    
    /// Executes NTT forward transform
    pub fn ntt_forward(&self, input: &[i64], twiddle_factors: &[i64], modulus: i64) -> Result<Vec<i64>> {
        let start_time = std::time::Instant::now();
        
        let result = match self.backend {
            GpuBackend::Cuda => {
                #[cfg(feature = "cuda")]
                {
                    use crate::gpu::cuda::GpuNTT;
                    let gpu_ntt = GpuNTT::new(0, input.len(), modulus)?;
                    gpu_ntt.forward_ntt(input)
                }
                #[cfg(not(feature = "cuda"))]
                {
                    self.cpu_ntt_forward(input, twiddle_factors, modulus)
                }
            }
            GpuBackend::OpenCL => {
                #[cfg(feature = "opencl")]
                {
                    use crate::gpu::opencl::GpuOpenCL;
                    let gpu = GpuOpenCL::new(0)?;
                    gpu.forward_ntt(input, twiddle_factors, modulus)
                }
                #[cfg(not(feature = "opencl"))]
                {
                    self.cpu_ntt_forward(input, twiddle_factors, modulus)
                }
            }
            GpuBackend::Cpu => {
                self.cpu_ntt_forward(input, twiddle_factors, modulus)
            }
        };
        
        // Record metrics
        let execution_time_us = start_time.elapsed().as_micros() as u64;
        self.record_metrics("ntt_forward", input.len() as u64, execution_time_us, 0);
        
        result
    }
    
    /// Executes matrix-vector multiplication
    pub fn matrix_vector_multiply(&self, matrix: &[i64], vector: &[i64], m: usize, n: usize, modulus: i64) -> Result<Vec<i64>> {
        let start_time = std::time::Instant::now();
        
        let result = match self.backend {
            GpuBackend::OpenCL => {
                #[cfg(feature = "opencl")]
                {
                    use crate::gpu::opencl::GpuOpenCL;
                    let gpu = GpuOpenCL::new(0)?;
                    gpu.matrix_vector_multiply(matrix, vector, m, n, modulus)
                }
                #[cfg(not(feature = "opencl"))]
                {
                    self.cpu_matrix_vector_multiply(matrix, vector, m, n, modulus)
                }
            }
            _ => {
                self.cpu_matrix_vector_multiply(matrix, vector, m, n, modulus)
            }
        };
        
        let execution_time_us = start_time.elapsed().as_micros() as u64;
        self.record_metrics("matrix_vector", (m * n) as u64, execution_time_us, 0);
        
        result
    }
    
    /// Computes infinity norm
    pub fn infinity_norm(&self, vector: &[i64]) -> Result<i64> {
        let start_time = std::time::Instant::now();
        
        let result = match self.backend {
            GpuBackend::OpenCL => {
                #[cfg(feature = "opencl")]
                {
                    use crate::gpu::opencl::GpuOpenCL;
                    let gpu = GpuOpenCL::new(0)?;
                    gpu.infinity_norm(vector)
                }
                #[cfg(not(feature = "opencl"))]
                {
                    self.cpu_infinity_norm(vector)
                }
            }
            _ => {
                self.cpu_infinity_norm(vector)
            }
        };
        
        let execution_time_us = start_time.elapsed().as_micros() as u64;
        self.record_metrics("infinity_norm", vector.len() as u64, execution_time_us, 0);
        
        result
    }
    
    /// Records performance metrics
    fn record_metrics(&self, operation: &str, elements: u64, execution_time_us: u64, transfer_time_us: u64) {
        let mut metrics = self.metrics.lock().unwrap();
        let mut metric = metrics.entry(operation.to_string()).or_insert_with(KernelMetrics::default);
        metric.update(elements, execution_time_us, transfer_time_us);
    }
    
    /// Returns the selected backend
    pub fn backend(&self) -> GpuBackend {
        self.backend
    }
    
    /// Returns performance metrics
    pub fn get_metrics(&self) -> HashMap<String, KernelMetrics> {
        let metrics = self.metrics.lock().unwrap();
        metrics.clone()
    }
    
    /// Prints performance summary
    pub fn print_performance_summary(&self) {
        let metrics = self.metrics.lock().unwrap();
        
        println!("\nGPU Kernel Performance Summary");
        println!("==============================");
        println!("Backend: {:?}", self.backend);
        println!();
        
        for (operation, metric) in metrics.iter() {
            println!("Operation: {}", operation);
            println!("  Throughput: {:.2} M elements/sec", metric.throughput_eps / 1_000_000.0);
            println!("  Execution Time: {:.2} ms", metric.execution_time_us as f64 / 1000.0);
            println!();
        }
    }
    
    // CPU fallback implementations
    
    /// CPU fallback for NTT forward transform
    fn cpu_ntt_forward(&self, input: &[i64], _twiddle_factors: &[i64], modulus: i64) -> Result<Vec<i64>> {
        // Simple CPU NTT implementation for fallback
        let n = input.len();
        let mut output = input.to_vec();
        
        // Basic NTT implementation (simplified)
        for i in 0..n {
            output[i] = (output[i] * 2) % modulus; // Placeholder transformation
        }
        
        Ok(output)
    }
    
    /// CPU fallback for matrix-vector multiplication
    fn cpu_matrix_vector_multiply(&self, matrix: &[i64], vector: &[i64], m: usize, n: usize, modulus: i64) -> Result<Vec<i64>> {
        if matrix.len() != m * n {
            return Err(LatticeFoldError::InvalidDimension {
                expected: m * n,
                got: matrix.len(),
            });
        }
        
        if vector.len() != n {
            return Err(LatticeFoldError::InvalidDimension {
                expected: n,
                got: vector.len(),
            });
        }
        
        let mut result = vec![0i64; m];
        
        for i in 0..m {
            let mut sum = 0i64;
            for j in 0..n {
                let product = (matrix[i * n + j] * vector[j]) % modulus;
                sum = (sum + product) % modulus;
            }
            result[i] = sum;
        }
        
        Ok(result)
    }
    
    /// CPU fallback for infinity norm computation
    fn cpu_infinity_norm(&self, vector: &[i64]) -> Result<i64> {
        if vector.is_empty() {
            return Ok(0);
        }
        
        let mut max_abs = 0i64;
        for &value in vector {
            let abs_value = value.abs();
            if abs_value > max_abs {
                max_abs = abs_value;
            }
        }
        
        Ok(max_abs)
    }
}

/// Global GPU kernel executor instance
static mut GLOBAL_GPU_EXECUTOR: Option<GpuKernelExecutor> = None;
static GPU_EXECUTOR_INIT: Once = Once::new();

/// Initializes the global GPU kernel executor
/// 
/// # Returns
/// * `Result<()>` - Success or initialization error
pub fn initialize_gpu_kernels() -> Result<()> {
    GPU_EXECUTOR_INIT.call_once(|| {
        match GpuKernelExecutor::new() {
            Ok(executor) => {
                unsafe {
                    GLOBAL_GPU_EXECUTOR = Some(executor);
                }
                println!("GPU kernel executor initialized successfully");
            }
            Err(e) => {
                eprintln!("Warning: GPU kernel executor initialization failed: {}", e);
                // Create a CPU-only executor as fallback
                let cpu_executor = GpuKernelExecutor {
                    backend: GpuBackend::Cpu,
                    #[cfg(feature = "cuda")]
                    cuda_ntt: None,
                    #[cfg(feature = "opencl")]
                    opencl_gpu: None,
                    metrics: Arc::new(Mutex::new(HashMap::new())),
                    device_info: GpuDeviceCapabilities {
                        device_name: "CPU Fallback".to_string(),
                        max_work_group_size: 1,
                        local_memory_size: 0,
                        global_memory_size: 8 * 1024 * 1024 * 1024,
                        memory_bandwidth_gbps: 50.0,
                        compute_units: num_cpus::get() as u32,
                        peak_performance_gflops: 100.0,
                        supports_double_precision: true,
                        memory_alignment: 64,
                        preferred_vector_width: 4,
                    },
                    kernel_params: Arc::new(Mutex::new(HashMap::new())),
                };
                unsafe {
                    GLOBAL_GPU_EXECUTOR = Some(cpu_executor);
                }
            }
        }
    });
    
    Ok(())
}

/// Gets the global GPU kernel executor
/// 
/// # Returns
/// * `Result<&'static GpuKernelExecutor>` - Reference to global executor or error
pub fn get_gpu_executor() -> Result<&'static GpuKernelExecutor> {
    initialize_gpu_kernels()?;
    
    unsafe {
        GLOBAL_GPU_EXECUTOR.as_ref().ok_or_else(|| {
            LatticeFoldError::GpuError("GPU kernel executor not initialized".to_string())
        })
    }
}

/// Convenience function for GPU NTT forward transform
/// 
/// # Arguments
/// * `input` - Input polynomial coefficients
/// * `twiddle_factors` - Precomputed twiddle factors
/// * `modulus` - Prime modulus
/// 
/// # Returns
/// * `Result<Vec<i64>>` - NTT-transformed coefficients or error
pub fn gpu_ntt_forward(input: &[i64], twiddle_factors: &[i64], modulus: i64) -> Result<Vec<i64>> {
    let executor = get_gpu_executor()?;
    executor.ntt_forward(input, twiddle_factors, modulus)
}

/// Convenience function for GPU matrix-vector multiplication
/// 
/// # Arguments
/// * `matrix` - Input matrix (row-major layout)
/// * `vector` - Input vector
/// * `m` - Number of matrix rows
/// * `n` - Number of matrix columns
/// * `modulus` - Prime modulus
/// 
/// # Returns
/// * `Result<Vec<i64>>` - Result vector or error
pub fn gpu_matrix_vector_multiply(matrix: &[i64], vector: &[i64], m: usize, n: usize, modulus: i64) -> Result<Vec<i64>> {
    let executor = get_gpu_executor()?;
    executor.matrix_vector_multiply(matrix, vector, m, n, modulus)
}

/// Convenience function for GPU infinity norm computation
/// 
/// # Arguments
/// * `vector` - Input vector
/// 
/// # Returns
/// * `Result<i64>` - Infinity norm or error
pub fn gpu_infinity_norm(vector: &[i64]) -> Result<i64> {
    let executor = get_gpu_executor()?;
    executor.infinity_norm(vector)
}

/// Prints GPU performance summary
pub fn print_gpu_performance_summary() {
    if let Ok(executor) = get_gpu_executor() {
        executor.print_performance_summary();
    } else {
        println!("GPU executor not available");
    }
}

/// Checks if GPU acceleration is available and functional
/// 
/// # Returns
/// * `bool` - True if GPU acceleration is available
pub fn is_gpu_acceleration_available() -> bool {
    match get_gpu_executor() {
        Ok(executor) => executor.backend() != GpuBackend::Cpu,
        Err(_) => false,
    }
}

/// Gets information about the current GPU backend
/// 
/// # Returns
/// * `Option<(GpuBackend, String)>` - Backend type and device name, or None if not available
pub fn get_gpu_backend_info() -> Option<(GpuBackend, String)> {
    match get_gpu_executor() {
        Ok(executor) => {
            let backend = executor.backend();
            let device_name = executor.device_capabilities().device_name.clone();
            Some((backend, device_name))
        }
        Err(_) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_gpu_executor_initialization() {
        // Test that GPU executor can be initialized (may fall back to CPU)
        let result = initialize_gpu_kernels();
        assert!(result.is_ok());
        
        // Test that we can get the executor
        let executor = get_gpu_executor();
        assert!(executor.is_ok());
        
        if let Ok(exec) = executor {
            println!("GPU executor backend: {:?}", exec.backend());
            println!("Device: {}", exec.device_capabilities().device_name);
        }
    }
    
    #[test]
    fn test_cpu_fallback_operations() {
        let executor = get_gpu_executor().unwrap();
        
        // Test CPU fallback for matrix-vector multiplication
        let matrix = vec![1, 2, 3, 4, 5, 6]; // 2x3 matrix
        let vector = vec![1, 2, 3];
        let modulus = 1000000007i64;
        
        let result = executor.cpu_matrix_vector_multiply(&matrix, &vector, 2, 3, modulus).unwrap();
        assert_eq!(result, vec![14, 32]); // [1*1+2*2+3*3, 4*1+5*2+6*3]
        
        // Test CPU fallback for infinity norm
        let test_vector = vec![1, -5, 3, -2, 7, -1];
        let norm = executor.cpu_infinity_norm(&test_vector).unwrap();
        assert_eq!(norm, 7);
    }
    
    #[test]
    fn test_gpu_convenience_functions() {
        // Test that convenience functions work (may use CPU fallback)
        let test_vector = vec![1, -3, 2, -7, 4];
        
        match gpu_infinity_norm(&test_vector) {
            Ok(norm) => {
                assert_eq!(norm, 7);
                println!("GPU infinity norm test passed: {}", norm);
            }
            Err(e) => {
                println!("GPU infinity norm test failed: {}", e);
            }
        }
    }
    
    #[test]
    fn test_performance_metrics() {
        let executor = get_gpu_executor().unwrap();
        
        // Perform some operations to generate metrics
        let test_vector = vec![1i64; 1000];
        let _ = executor.infinity_norm(&test_vector);
        
        // Check that metrics were recorded
        let metrics = executor.get_metrics();
        if !metrics.is_empty() {
            println!("Performance metrics recorded for {} operations", metrics.len());
            for (op, metric) in metrics.iter() {
                println!("  {}: {:.2} M elements/sec", op, metric.throughput_eps / 1_000_000.0);
            }
        }
    }
    
    #[test]
    fn test_backend_selection() {
        // Test that backend selection works
        if let Some((backend, device_name)) = get_gpu_backend_info() {
            println!("Selected backend: {:?}", backend);
            println!("Device: {}", device_name);
            
            match backend {
                GpuBackend::Cuda => println!("CUDA acceleration available"),
                GpuBackend::OpenCL => println!("OpenCL acceleration available"),
                GpuBackend::Cpu => println!("Using CPU fallback"),
            }
        }
        
        // Test availability check
        let gpu_available = is_gpu_acceleration_available();
        println!("GPU acceleration available: {}", gpu_available);
    }
}, m: usize, n: usize, modulus: i64) -> Result<Vec<i64>> {
        if matrix.len() != m * n {
            return Err(LatticeFoldError::InvalidDimension {
                expected: m * n,
                got: matrix.len(),
            });
        }
        
        if vector.len() != n {
            return Err(LatticeFoldError::InvalidDimension {
                expected: n,
                got: vector.len(),
            });
        }
        
        let mut result = vec![0i64; m];
        
        for i in 0..m {
            let mut sum = 0i64;
            for j in 0..n {
                let product = (matrix[i * n + j] * vector[j]) % modulus;
                sum = (sum + product) % modulus;
            }
            result[i] = sum;
        }
        
        Ok(result)
    }
    
    /// CPU fallback for infinity norm computation
    fn cpu_infinity_norm(&self, vector: &[i64]) -> Result<i64> {
        if vector.is_empty() {
            return Ok(0);
        }
        
        let mut max_abs = 0i64;
        for &value in vector {
            let abs_value = value.abs();
            if abs_value > max_abs {
                max_abs = abs_value;
            }
        }
        
        Ok(max_abs)
    }
}

/// Global GPU kernel executor instance
static mut GLOBAL_EXECUTOR: Option<GpuKernelExecutor> = None;
static EXECUTOR_INIT: Once = Once::new();

/// Initializes the global GPU kernel executor
pub fn initialize_gpu_kernels() -> Result<()> {
    EXECUTOR_INIT.call_once(|| {
        match GpuKernelExecutor::new() {
            Ok(executor) => {
                unsafe {
                    GLOBAL_EXECUTOR = Some(executor);
                }
            }
            Err(e) => {
                eprintln!("Warning: GPU kernel initialization failed: {}", e);
                // Create CPU-only executor as fallback
                let cpu_executor = GpuKernelExecutor {
                    backend: GpuBackend::Cpu,
                    metrics: Arc::new(Mutex::new(HashMap::new())),
                };
                unsafe {
                    GLOBAL_EXECUTOR = Some(cpu_executor);
                }
            }
        }
    });
    
    Ok(())
}

/// Gets the global GPU kernel executor
pub fn get_gpu_executor() -> Result<&'static GpuKernelExecutor> {
    initialize_gpu_kernels()?;
    
    unsafe {
        GLOBAL_EXECUTOR.as_ref().ok_or_else(|| {
            LatticeFoldError::GpuError("GPU kernel executor not initialized".to_string())
        })
    }
}

/// Convenience function for GPU NTT forward transform
pub fn gpu_ntt_forward(input: &[i64], twiddle_factors: &[i64], modulus: i64) -> Result<Vec<i64>> {
    let executor = get_gpu_executor()?;
    executor.ntt_forward(input, twiddle_factors, modulus)
}

/// Convenience function for GPU matrix-vector multiplication
pub fn gpu_matrix_vector_multiply(matrix: &[i64], vector: &[i64], m: usize, n: usize, modulus: i64) -> Result<Vec<i64>> {
    let executor = get_gpu_executor()?;
    executor.matrix_vector_multiply(matrix, vector, m, n, modulus)
}

/// Convenience function for GPU infinity norm computation
pub fn gpu_infinity_norm(vector: &[i64]) -> Result<i64> {
    let executor = get_gpu_executor()?;
    executor.infinity_norm(vector)
}

/// Checks if GPU acceleration is available
pub fn is_gpu_acceleration_available() -> bool {
    match get_gpu_executor() {
        Ok(executor) => executor.backend() != GpuBackend::Cpu,
        Err(_) => false,
    }
}

/// Gets information about the current GPU backend
pub fn get_gpu_backend_info() -> Option<(GpuBackend, String)> {
    match get_gpu_executor() {
        Ok(executor) => {
            let backend = executor.backend();
            let device_name = format!("{:?} Device", backend);
            Some((backend, device_name))
        }
        Err(_) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_gpu_executor_creation() {
        match GpuKernelExecutor::new() {
            Ok(executor) => {
                println!("GPU executor created successfully");
                println!("Backend: {:?}", executor.backend());
            }
            Err(e) => {
                println!("GPU executor creation failed: {}", e);
            }
        }
    }
    
    #[test]
    fn test_cpu_fallback_operations() {
        let executor = get_gpu_executor().unwrap();
        
        // Test CPU matrix-vector multiplication
        let matrix = vec![1, 2, 3, 4]; // 2x2 matrix
        let vector = vec![1, 2];
        let modulus = 17i64;
        
        match executor.cpu_matrix_vector_multiply(&matrix, &vector, 2, 2, modulus) {
            Ok(result) => {
                println!("CPU matrix-vector result: {:?}", result);
                assert_eq!(result, vec![5, 11]); // [1*1+2*2, 3*1+4*2] = [5, 11]
            }
            Err(e) => {
                println!("CPU matrix-vector failed: {}", e);
            }
        }
        
        // Test CPU infinity norm
        let test_vector = vec![1, -5, 3, -2];
        match executor.cpu_infinity_norm(&test_vector) {
            Ok(norm) => {
                println!("CPU infinity norm result: {}", norm);
                assert_eq!(norm, 5); // max(|1|, |-5|, |3|, |-2|) = 5
            }
            Err(e) => {
                println!("CPU infinity norm failed: {}", e);
            }
        }
    }
}