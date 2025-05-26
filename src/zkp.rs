use ark_ec::{AffineCurve, ProjectiveCurve};
use ark_ff::{Field, PrimeField};
use ark_poly::{univariate::DensePolynomial, DenseUVPolynomial, Polynomial};
use ark_std::{rand::Rng, UniformRand};
use merlin::Transcript;
use std::collections::HashMap;

use crate::commitment::{FoldingCommitment, PedersenCommitment};
use crate::lattice::{LatticeBasis, LatticePoint, LatticeRelation};

#[derive(Clone, Debug)]
pub struct ZKProof<G: AffineCurve, F: Field> {
    pub commitment: G,
    pub response: Vec<F>,
    pub challenge: F,
}

pub struct ZKProver<G: AffineCurve, F: Field> {
    pub params: ZKParams<G, F>,
    pub transcript: Transcript,
}

#[derive(Clone, Debug)]
pub struct ZKParams<G: AffineCurve, F: Field> {
    pub commitment_scheme: FoldingCommitment<G>,
    pub lattice_relation: LatticeRelation<F>,
}

impl<G: AffineCurve, F: Field> ZKProver<G, F> {
    pub fn new(params: ZKParams<G, F>) -> Self {
        Self {
            params,
            transcript: Transcript::new(b"LatticeFold+ ZKP"),
        }
    }

    pub fn prove<R: Rng>(
        &mut self,
        witness: &LatticePoint<F>,
        rng: &mut R,
    ) -> ZKProof<G, F> {
        // 1. Commit to witness
        let randomness = F::rand(rng);
        let commitment = self.params.commitment_scheme.commit_vector(
            &witness.coordinates,
            randomness,
        );

        self.transcript.append_message(b"commitment", &commitment.to_bytes());

        // 2. Generate challenge
        let mut challenge_bytes = [0u8; 32];
        self.transcript.challenge_bytes(b"challenge", &mut challenge_bytes);
        let challenge = F::from_random_bytes(&challenge_bytes).unwrap();

        // 3. Compute response
        let response = self.compute_response(witness, &challenge, randomness);

        ZKProof {
            commitment,
            response,
            challenge,
        }
    }

    fn compute_response(
        &self,
        witness: &LatticePoint<F>,
        challenge: &F,
        randomness: F,
    ) -> Vec<F> {
        let mut response = witness.coordinates.clone();
        for coord in response.iter_mut() {
            *coord = *coord * *challenge + randomness;
        }
        response
    }
}

pub struct ZKVerifier<G: AffineCurve, F: Field> {
    pub params: ZKParams<G, F>,
    pub transcript: Transcript,
}

impl<G: AffineCurve, F: Field> ZKVerifier<G, F> {
    pub fn new(params: ZKParams<G, F>) -> Self {
        Self {
            params,
            transcript: Transcript::new(b"LatticeFold+ ZKP"),
        }
    }

    pub fn verify(&mut self, proof: &ZKProof<G, F>) -> bool {
        // 1. Verify commitment
        self.transcript.append_message(b"commitment", &proof.commitment.to_bytes());

        // 2. Verify challenge
        let mut challenge_bytes = [0u8; 32];
        self.transcript.challenge_bytes(b"challenge", &mut challenge_bytes);
        let challenge = F::from_random_bytes(&challenge_bytes).unwrap();

        if challenge != proof.challenge {
            return false;
        }

        // 3. Verify response
        self.verify_response(&proof.commitment, &proof.response, &proof.challenge)
    }

    fn verify_response(
        &self,
        commitment: &G,
        response: &[F],
        challenge: &F,
    ) -> bool {
        // 1. Reconstruct the commitment from the response
        let mut reconstructed = G::Projective::zero();
        
        // 2. Compute the expected commitment
        for (i, r) in response.iter().enumerate() {
            let g = &self.params.commitment_scheme.vector_commitment.params.generators[i];
            reconstructed += g.mul(r.into_repr());
        }
        
        // 3. Add the challenge term
        let h = &self.params.commitment_scheme.vector_commitment.params.h;
        reconstructed += h.mul(challenge.into_repr());
        
        // 4. Compare with the provided commitment
        reconstructed.into_affine() == *commitment
    }
}

#[derive(Clone, Debug)]
pub struct FoldingProof<G: AffineCurve, F: Field> {
    pub commitments: Vec<G>,
    pub responses: Vec<Vec<F>>,
    pub folded_witness: Vec<F>,
    pub folded_public: Vec<F>,
    pub challenge: F,
}

pub struct FoldingProver<G: AffineCurve, F: Field> {
    pub params: ZKParams<G, F>,
    pub transcript: Transcript,
}

impl<G: AffineCurve, F: Field> FoldingProver<G, F> {
    pub fn new(params: ZKParams<G, F>) -> Self {
        Self {
            params,
            transcript: Transcript::new(b"LatticeFold+ Folding"),
        }
    }

    pub fn prove<R: Rng>(
        &mut self,
        instances: &[LatticeFoldInstance<F>],
        rng: &mut R,
    ) -> FoldingProof<G, F> {
        // 1. Commit to each instance
        let mut commitments = Vec::new();
        let mut responses = Vec::new();

        for instance in instances {
            let randomness = F::rand(rng);
            let commitment = self.params.commitment_scheme.commit_vector(
                &instance.witness,
                randomness,
            );
            commitments.push(commitment);
            responses.push(instance.witness.clone());

            self.transcript.append_message(b"commitment", &commitment.to_bytes());
        }

        // 2. Generate challenge
        let mut challenge_bytes = [0u8; 32];
        self.transcript.challenge_bytes(b"challenge", &mut challenge_bytes);
        let challenge = F::from_random_bytes(&challenge_bytes).unwrap();

        // 3. Fold witnesses and public inputs
        let folded_witness = self.fold_witnesses(&instances, &challenge);
        let folded_public = self.fold_public_inputs(&instances, &challenge);

        FoldingProof {
            commitments,
            responses,
            folded_witness,
            folded_public,
            challenge,
        }
    }

    fn fold_witnesses(&self, instances: &[LatticeFoldInstance<F>], challenge: &F) -> Vec<F> {
        let mut result = vec![F::zero(); self.params.lattice_relation.basis.dimension];

        for (i, instance) in instances.iter().enumerate() {
            let weight = challenge.pow(&[i as u64]);
            for (j, w) in instance.witness.iter().enumerate() {
                result[j] += *w * weight;
            }
        }

        result
    }

    fn fold_public_inputs(&self, instances: &[LatticeFoldInstance<F>], challenge: &F) -> Vec<F> {
        let mut result = vec![F::zero(); self.params.lattice_relation.basis.dimension];

        for (i, instance) in instances.iter().enumerate() {
            let weight = challenge.pow(&[i as u64]);
            for (j, p) in instance.public_input.iter().enumerate() {
                result[j] += *p * weight;
            }
        }

        result
    }
}

#[derive(Clone, Debug)]
pub struct LatticeFoldInstance<F: Field> {
    pub witness: Vec<F>,
    pub public_input: Vec<F>,
} 