 //! LatticeFold+ – protocol orchestrator (Section 4.3)
//! This module glues together commitment, challenge generation and folding
//! to provide a clean `setup`, `prove`, and `verify` API.
//!
//! NOTE: The cryptographic security properties depend on the correctness
//! of sub-modules (`commitment`, `folding`, etc.).  This orchestrator
//! simply sequences them.

use crate::challenge_generation::{ChallengeGenerator, TranscriptChallengeGenerator};
use crate::commitment::{CommitmentParams, FoldingCommitment, HomomorphicCommitmentScheme};
use crate::error::{LatticeFoldError, Result};
use crate::folding::{FoldingProof, FoldingScheme};
use crate::lattice::{LatticeMatrix, LatticeParams, LatticePoint};
use crate::quantum_resistance::{QuantumResistanceAnalyzer, SecurityLevel};
use crate::types::{LatticeFoldInstance, LatticeFoldParams};

use merlin::Transcript;
use rand::{CryptoRng, RngCore};
use zeroize::Zeroize;

/// Combined public parameters for the LatticeFold+ protocol.
#[derive(Clone, Debug, Zeroize)]
pub struct ProtocolParams<C: HomomorphicCommitmentScheme> {
    /// Underlying lattice parameters (dimension, modulus, etc.)
    pub lattice_params: LatticeParams,
    /// Commitment-scheme parameters (vector commitment built from SIS)
    pub commitment_scheme: C,
    /// Maximum number of instances allowed in one proof
    pub max_instances: usize,
    /// Security parameter (in *classical* bits)
    pub security_param: usize,
}

/// Proof object produced by the protocol.
#[derive(Clone, Debug)]
pub struct ProtocolProof<C: HomomorphicCommitmentScheme> {
    /// Folding sub-proof
    pub folding_proof: FoldingProof,
    /// Commitments to folded vector (if hiding-binding split is needed)
    pub final_commitment: FoldingCommitment,
    /// Transcript bytes to allow deterministic re-verification
    pub transcript_bytes: Vec<u8>,
    /// Phantom for commitment scheme
    _phantom: std::marker::PhantomData<C>,
}

impl<C> ProtocolParams<C>
where
    C: HomomorphicCommitmentScheme + Clone,
{
    /// High-level setup selecting parameters via the quantum-resistance analyser.
    pub fn setup<R: RngCore + CryptoRng>(
        rng: &mut R,
        security_level: SecurityLevel,
        commitment_params: CommitmentParams,
    ) -> Result<Self> {
        let analyzer = QuantumResistanceAnalyzer::new();
        let qr = analyzer.get_params(security_level)?;
        let lattice_params = analyzer.create_lattice_params(&qr);

        let commitment_scheme = C::new(commitment_params, &lattice_params, rng)?;

        Ok(Self {
            lattice_params,
            commitment_scheme,
            max_instances: 1 << 16, // arbitrary large default
            security_param: qr.security_level,
        })
    }
}

/// Prover context.
pub struct ProtocolProver<'a, C: HomomorphicCommitmentScheme> {
    params: &'a ProtocolParams<C>,
}

impl<'a, C> ProtocolProver<'a, C>
where
    C: HomomorphicCommitmentScheme + Clone,
{
    pub fn new(params: &'a ProtocolParams<C>) -> Self {
        Self { params }
    }

    pub fn prove<R: CryptoRng + RngCore>(
        &self,
        instances: &[LatticeFoldInstance],
        rng: &mut R,
    ) -> Result<ProtocolProof<C>> {
        if instances.is_empty() {
            return Err(LatticeFoldError::InvalidParameters(
                "no instances provided".to_string(),
            ));
        }
        if instances.len() > self.params.max_instances {
            return Err(LatticeFoldError::InvalidParameters(format!(
                "too many instances ({} > max {})",
                instances.len(), self.params.max_instances
            )));
        }

        // ---- Phase 1: map instances to lattice points ----
        let mut points = Vec::with_capacity(instances.len());
        for inst in instances {
            // simple concatenation; in real implementation convert witness to lattice vector
            let point = LatticePoint::from_scalars(&inst.witness, self.params.lattice_params.q)?;
            points.push(point);
        }

        // ---- Phase 2: run folding scheme ----
        let basis = LatticeMatrix::identity(self.params.lattice_params.n);
        let mut folding_scheme = FoldingScheme::new(self.params.lattice_params.clone(), basis);

        let mut transcript = Transcript::new(b"lattice_fold_plus");
        let mut challenge_gen = TranscriptChallengeGenerator::new_default(&mut transcript);

        let folding_proof = folding_scheme.prove(&points, &mut transcript, rng)?;
        let final_point = folding_scheme.fold_points(&points)?;

        // ---- Phase 3: commit to final point ----
        let final_commitment = self
            .params
            .commitment_scheme
            .commit(&final_point.to_bytes(), rng)?;

        let mut transcript_bytes = vec![0u8; 64];
        transcript.extract_bytes(&mut transcript_bytes);

        Ok(ProtocolProof {
            folding_proof,
            final_commitment,
            transcript_bytes,
            _phantom: std::marker::PhantomData,
        })
    }
}

/// Verifier context.
pub struct ProtocolVerifier<'a, C: HomomorphicCommitmentScheme> {
    params: &'a ProtocolParams<C>,
}

impl<'a, C> ProtocolVerifier<'a, C>
where
    C: HomomorphicCommitmentScheme + Clone,
{
    pub fn new(params: &'a ProtocolParams<C>) -> Self {
        Self { params }
    }

    pub fn verify(&self, proof: &ProtocolProof<C>) -> Result<bool> {
        // Re-create transcript
        let mut transcript = Transcript::new(b"lattice_fold_plus");
        transcript.append_message(b"replay", &proof.transcript_bytes);

        // Re-create basis and folding scheme
        let basis = LatticeMatrix::identity(self.params.lattice_params.n);
        let mut folding_scheme = FoldingScheme::new(self.params.lattice_params.clone(), basis);

        let folding_valid = folding_scheme.verify(&proof.folding_proof, &mut transcript)?;
        if !folding_valid {
            return Ok(false);
        }

        // Verify commitment
        let folded_point = proof.folding_proof.final_point.clone();
        let commit_valid = self
            .params
            .commitment_scheme
            .verify(
                &proof.final_commitment,
                &folded_point.to_bytes(),
            )?;

        Ok(commit_valid)
    }
}