use ark_ec::{AffineCurve, ProjectiveCurve};
use ark_ff::{Field, PrimeField};
use ark_std::{rand::Rng, UniformRand};
use merlin::Transcript;

use crate::commitment::FoldingCommitment;
use crate::lattice::{LatticeBasis, LatticePoint, LatticeRelation};
use crate::types::{LatticeFoldInstance, LatticeFoldParams};

/// A prover for the lattice folding protocol
pub struct FoldingProver<G: AffineCurve, F: Field> {
    /// The protocol parameters
    params: LatticeFoldParams<G, F>,
    /// The transcript for the protocol
    transcript: Transcript,
}

impl<G: AffineCurve, F: Field> FoldingProver<G, F> {
    /// Create a new folding prover
    pub fn new(params: LatticeFoldParams<G, F>) -> Self {
        Self {
            params,
            transcript: Transcript::new(b"LatticeFold+ Folding"),
        }
    }

    /// Generate a folding proof for the given instances
    pub fn prove<R: Rng>(
        &mut self,
        instances: &[LatticeFoldInstance<F>],
        rng: &mut R,
    ) -> FoldingProof<G, F> {
        // 1. Generate randomness for each instance
        let mut randomness = Vec::new();
        for _ in 0..instances.len() {
            let mut r = vec![F::zero(); self.params.dimension];
            for i in 0..self.params.dimension {
                r[i] = F::rand(rng);
            }
            randomness.push(r);
        }

        // 2. Compute commitments
        let mut commitments = Vec::new();
        for (instance, r) in instances.iter().zip(randomness.iter()) {
            let commitment = self.commit(instance, r);
            commitments.push(commitment);
        }

        // 3. Generate challenge
        let mut challenge_bytes = [0u8; 32];
        self.transcript.challenge_bytes(b"folding_challenge", &mut challenge_bytes);
        let challenge = F::from_random_bytes(&challenge_bytes).unwrap();

        // 4. Compute responses
        let mut responses = Vec::new();
        for (instance, r) in instances.iter().zip(randomness.iter()) {
            let response = self.compute_response(instance, r, &challenge);
            responses.push(response);
        }

        // 5. Fold the instances
        let folded_witness = self.fold_witnesses(instances, &challenge);
        let folded_public = self.fold_public_inputs(instances, &challenge);

        FoldingProof {
            commitments,
            responses,
            folded_witness,
            folded_public,
        }
    }

    /// Commit to an instance using the given randomness
    fn commit(&self, instance: &LatticeFoldInstance<F>, randomness: &[F]) -> G {
        let mut commitment = G::Projective::zero();
        
        // Add witness commitment
        for (i, w) in instance.witness.iter().enumerate() {
            let g = &self.params.commitment_scheme.vector_commitment.params.generators[i];
            commitment += g.mul(w.into_repr());
        }
        
        // Add public input commitment
        for (i, p) in instance.public_input.iter().enumerate() {
            let g = &self.params.commitment_scheme.vector_commitment.params.generators[i];
            commitment += g.mul(p.into_repr());
        }
        
        // Add randomness
        let h = &self.params.commitment_scheme.vector_commitment.params.h;
        for r in randomness {
            commitment += h.mul(r.into_repr());
        }
        
        commitment.into_affine()
    }

    /// Compute the response for an instance
    fn compute_response(&self, instance: &LatticeFoldInstance<F>, randomness: &[F], challenge: &F) -> Vec<F> {
        let mut response = Vec::with_capacity(self.params.dimension);
        
        for (w, r) in instance.witness.iter().zip(randomness.iter()) {
            response.push(*w * *challenge + *r);
        }
        
        response
    }

    /// Fold multiple witnesses using the challenge
    fn fold_witnesses(&self, instances: &[LatticeFoldInstance<F>], challenge: &F) -> Vec<F> {
        let mut folded = vec![F::zero(); self.params.dimension];
        
        for instance in instances {
            for (i, w) in instance.witness.iter().enumerate() {
                folded[i] += *w * *challenge;
            }
        }
        
        folded
    }

    /// Fold multiple public inputs using the challenge
    fn fold_public_inputs(&self, instances: &[LatticeFoldInstance<F>], challenge: &F) -> Vec<F> {
        let mut folded = vec![F::zero(); self.params.dimension];
        
        for instance in instances {
            for (i, p) in instance.public_input.iter().enumerate() {
                folded[i] += *p * *challenge;
            }
        }
        
        folded
    }
}

/// A folding proof containing commitments, responses, and folded values
#[derive(Clone, Debug)]
pub struct FoldingProof<G: AffineCurve, F: Field> {
    /// The commitments to each instance
    pub commitments: Vec<G>,
    /// The responses to the challenge
    pub responses: Vec<Vec<F>>,
    /// The folded witness
    pub folded_witness: Vec<F>,
    /// The folded public input
    pub folded_public: Vec<F>,
} 