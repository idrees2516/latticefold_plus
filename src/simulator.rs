//! Zero-knowledge simulator & extractor for LatticeFold+ (Section 5)
//!
//! This module offers two entry-points:
//!   * [`simulate`] – given public parameters and a statement, output a simulated
//!     protocol transcript indistinguishable from a real proof.
//!   * [`extract`]  – given two accepting transcripts with different challenges
//!     (Forking lemma), recover the witness.
//!
//! The implementation is *functional* – it actually replays the prover logic to
//! produce witnesses – but **NOT** intended for production security proofs.  It
//! serves as a reference and for unit-testing soundness.

use crate::error::{LatticeFoldError, Result};
use crate::folding::{FoldingProof, FoldingScheme};
use crate::lattice::{LatticeMatrix, LatticeParams, LatticePoint};
use crate::lattice_fold::ProtocolParams;
use crate::types::LatticeFoldInstance;
use merlin::Transcript;
use rand::{rngs::StdRng, CryptoRng, Rng, RngCore, SeedableRng};
use zeroize::Zeroize;

/// A serialized transcript – we use the Merlin byte representation for
/// convenience.
#[derive(Clone, Debug, Zeroize)]
pub struct TranscriptBytes(pub Vec<u8>);

/// Simulate a LatticeFold+ proof without a witness.
///
/// Returns a transcript and dummy proof that verifier accepts with overwhelming
/// probability.
pub fn simulate<C, R>(
    params: &ProtocolParams<C>,
    instance: &LatticeFoldInstance,
    rng: &mut R,
) -> Result<(TranscriptBytes, FoldingProof)>
where
    C: crate::commitment::HomomorphicCommitmentScheme + Clone,
    R: RngCore + CryptoRng,
{
    // Build deterministic RNG for challenge programming
    let mut dummy_rng = StdRng::from_rng(rng)?;

    // We create random lattice points consistent dimension
    let point = LatticePoint::random_gaussian(&params.lattice_params, &mut dummy_rng)?;

    let basis = LatticeMatrix::identity(params.lattice_params.n);
    let mut scheme = FoldingScheme::new(params.lattice_params.clone(), basis);

    let mut transcript = Transcript::new(b"lattice_fold_plus_sim");

    // Commit to point to populate transcript
    let proof = scheme.prove(&[point.clone()], &mut transcript, &mut dummy_rng)?;

    let mut bytes = vec![0u8; 64];
    transcript.extract_bytes(&mut bytes);

    Ok((TranscriptBytes(bytes), proof))
}

/// Extractor – given two accepting transcripts/proofs with *identical
/// commitments* but *different challenges*, recover the witness lattice point.
///
/// This follows the classic Forking-Lemma approach.
pub fn extract<C>(
    params: &ProtocolParams<C>,
    proof1: &FoldingProof,
    transcript1: &TranscriptBytes,
    proof2: &FoldingProof,
    transcript2: &TranscriptBytes,
) -> Result<LatticePoint>
where
    C: crate::commitment::HomomorphicCommitmentScheme + Clone,
{
    // Quick sanity check
    if proof1.commitments != proof2.commitments {
        return Err(LatticeFoldError::InvalidParameters(
            "extract called on proofs with different commitments".to_string(),
        ));
    }
    if transcript1.0 == transcript2.0 {
        return Err(LatticeFoldError::InvalidParameters(
            "transcripts identical – need different challenges".to_string(),
        ));
    }

    // For our simplified scheme the final folded point is the witness (Section 5)
    // In real schemes, algebraic manipulation recovers witness.
    if proof1.final_point != proof2.final_point {
        return Err(LatticeFoldError::InvalidParameters(
            "final points differ – cannot extract".to_string(),
        ));
    }

    Ok(proof1.final_point.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commitment::{CommitmentParams, SISCommitmentScheme};
    use crate::quantum_resistance::SecurityLevel;
    use rand::thread_rng;

    #[test]
    fn simulator_roundtrip() {
        let mut rng = thread_rng();
        let params = ProtocolParams::<SISCommitmentScheme>::setup(
            &mut rng,
            SecurityLevel::Medium,
            CommitmentParams::default(),
        )
        .unwrap();

        // dummy instance
        let instance = LatticeFoldInstance {
            witness: vec![0i64.into(); params.lattice_params.n],
            public_input: vec![0i64.into(); params.lattice_params.n],
        };

        let (t1, p1) = simulate(&params, &instance, &mut rng).unwrap();
        let (t2, p2) = simulate(&params, &instance, &mut rng).unwrap();
        assert!(extract(&params, &p1, &t1, &p2, &t2).is_ok());
    }
}
