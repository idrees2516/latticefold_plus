//! Parallel / batched folding optimisation (Section 6.3 of the paper)
//!
//! Uses Rayon to fold multiple instances concurrently and then reduce.
//! Enabled with the `parallel` feature.

use crate::error::Result;
use crate::folding::{FoldingProof, FoldingScheme};
use crate::lattice::{LatticeMatrix, LatticeParams, LatticePoint};
use merlin::Transcript;
use rand::{CryptoRng, RngCore};
use rayon::prelude::*;

/// A folding scheme that runs commitment & response generation in parallel.
pub struct ParallelFoldingScheme {
    params: LatticeParams,
    basis: LatticeMatrix,
}

impl ParallelFoldingScheme {
    pub fn new(params: LatticeParams, basis: LatticeMatrix) -> Self {
        Self { params, basis }
    }

    /// Parallel prove – folds each point chunk in parallel and merges proofs.
    pub fn prove_parallel<R: CryptoRng + RngCore>(
        &self,
        points: &[LatticePoint],
        transcript: &mut Transcript,
        rng: &mut R,
    ) -> Result<FoldingProof> {
        // Split points into chunks of 4 for better cache locality.
        let chunks: Vec<&[LatticePoint]> = points.chunks(4).collect();

        // Each chunk proves in parallel.
        let partial_proofs: Result<Vec<FoldingProof>> = chunks
            .into_par_iter()
            .map(|chunk| {
                let mut local_scheme = FoldingScheme::new(self.params.clone(), self.basis.clone());
                // We need independent RNG per thread.
                let mut local_rng = rand::rngs::StdRng::from_entropy();
                local_scheme.prove(chunk, &mut transcript.clone(), &mut local_rng)
            })
            .collect();

        let mut proofs = partial_proofs?;
        // Merge proofs sequentially using existing FoldingScheme logic.
        let mut scheme = FoldingScheme::new(self.params.clone(), self.basis.clone());
        let mut final_points = Vec::new();
        for p in &proofs {
            final_points.push(p.final_point.clone());
        }
        let merged_final = scheme.fold_points(&final_points)?;

        // Concatenate commitments & responses
        let mut commitments = Vec::new();
        let mut responses = Vec::new();
        for p in proofs.drain(..) {
            commitments.extend(p.commitments);
            responses.extend(p.challenge_responses);
        }

        Ok(FoldingProof {
            commitments,
            challenge_responses: responses,
            final_point: merged_final,
        })
    }
}
