#[derive(Error, Debug)]
pub enum LatticeFoldError {
    #[error("Invalid dimension: expected {expected}, got {got}")]
    InvalidDimension {
        expected: usize,
        got: usize,
    },

    #[error("Invalid modulus: {0}")]
    InvalidModulus(String),

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