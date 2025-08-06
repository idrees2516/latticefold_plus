use thiserror::Error;

#[derive(Error, Debug)]
pub enum LatticeFoldError {
    #[error("Invalid dimension: expected {expected}, got {got}")]
    InvalidDimension {
        expected: usize,
        got: usize,
    },

    #[error("Invalid modulus: {modulus}")]
    InvalidModulus { modulus: i64 },

    #[error("Point not in lattice")]
    PointNotInLattice,

    #[error("Invalid parameters: {0}")]
    InvalidParameters(String),

    #[error("Verification failed: {0}")]
    VerificationFailed(String),

    #[error("Proof generation failed: {0}")]
    ProofGenerationFailed(String),

    #[error("Invalid commitment: {0}")]
    InvalidCommitment(String),

    #[error("Cryptographic error: {0}")]
    CryptoError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Challenge verification failed")]
    ChallengeVerificationFailed,

    #[error("Response verification failed")]
    ResponseVerificationFailed,

    #[error("Folding verification failed")]
    FoldingVerificationFailed,

    #[error("Invalid witness")]
    InvalidWitness,

    #[error("Invalid public input")]
    InvalidPublicInput,

    #[error("Coefficient {coefficient} out of range [{min_bound}, {max_bound}] at position {position}")]
    CoefficientOutOfRange {
        coefficient: i64,
        min_bound: i64,
        max_bound: i64,
        position: usize,
    },

    #[error("Incompatible moduli: {modulus1} and {modulus2}")]
    IncompatibleModuli {
        modulus1: i64,
        modulus2: i64,
    },

    #[error("GPU error: {0}")]
    GPUError(String),

    #[error("GPU error: {0}")]
    GpuError(String),

    #[error("GPU not available: {0}")]
    GpuNotAvailable(String),

    #[error("GPU memory error: {0}")]
    GpuMemoryError(String),

    #[error("Memory allocation error: {0}")]
    MemoryAllocationError(String),

    #[error("SIMD error: {0}")]
    SimdError(String),

    #[error("Cache error: {0}")]
    CacheError(String),

    #[error("Streaming computation error: {0}")]
    StreamingError(String),

    #[error("NUMA error: {0}")]
    NumaError(String),

    #[error("Arithmetic overflow: {0}")]
    ArithmeticOverflow(String),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("NTT error: {0}")]
    NTTError(String),

    #[error("Norm bound violation: norm {norm} >= bound {bound}")]
    NormBoundViolation {
        norm: i64,
        bound: i64,
    },

    #[error("Verification timeout exceeded")]
    VerificationTimeout,

    #[error("GPU initialization failed: {0}")]
    GpuInitialization(String),

    #[error("GPU computation failed: {0}")]
    GpuComputation(String),

    #[error("GPU configuration error: {0}")]
    GpuConfiguration(String),

    #[error("GPU memory allocation failed: {0}")]
    GpuMemoryAllocation(String),

    #[error("CUDA kernel compilation failed: {0}")]
    CudaKernelCompilation(String),

    #[error("CUDA kernel launch failed: {0}")]
    CudaKernelLaunch(String),
}

pub type Result<T> = std::result::Result<T, LatticeFoldError>;

impl From<ark_ff::PrimeFieldDecodingError> for LatticeFoldError {
    fn from(err: ark_ff::PrimeFieldDecodingError) -> Self {
        Self::CryptoError(format!("Field decoding error: {:?}", err))
    }
}

impl From<merlin::TranscriptError> for LatticeFoldError {
    fn from(err: merlin::TranscriptError) -> Self {
        Self::CryptoError(format!("Transcript error: {:?}", err))
    }
} 