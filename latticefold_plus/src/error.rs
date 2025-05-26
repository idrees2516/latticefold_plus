use thiserror::Error;

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

    #[error("Verification failed")]
    VerificationFailed,

    #[error("Proof generation failed: {0}")]
    ProofGenerationFailed(String),

    #[error("Invalid commitment")]
    InvalidCommitment,

    #[error("Cryptographic error: {0}")]
    CryptoError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, LatticeFoldError>;
