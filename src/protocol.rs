use ark_ec::{AffineCurve, ProjectiveCurve};
use ark_ff::{Field, PrimeField};
use ark_poly::{univariate::DensePolynomial, DenseUVPolynomial, Polynomial};
use ark_std::{rand::Rng, UniformRand};
use merlin::Transcript;
use std::collections::HashMap;

use crate::commitment::{FoldingCommitment, PedersenCommitment};
use crate::lattice::{LatticeBasis, LatticePoint, LatticeRelation};
use crate::types::{CommitmentParams, FinalProof, LatticeFoldInstance, LatticeFoldParams};
use crate::zkp::{FoldingProof, ZKProof};

#[derive(Clone, Debug)]
pub struct ProtocolParams<G: AffineCurve, F: Field> {
    pub dimension: usize,
    pub modulus: F,
    pub security_param: usize,
    pub commitment_scheme: FoldingCommitment<G>,
    pub lattice_relation: LatticeRelation<F>,
}

pub struct ProtocolProver<G: AffineCurve, F: Field> {
    pub params: ProtocolParams<G, F>,
    pub transcript: Transcript,
}

impl<G: AffineCurve, F: Field> ProtocolProver<G, F> {
    pub fn new(params: ProtocolParams<G, F>) -> Self {
        Self {
            params,
            transcript: Transcript::new(b"LatticeFold+ Protocol"),
        }
    }

    pub fn prove<R: Rng>(
        &mut self,
        instances: &[LatticeFoldInstance<F>],
        rng: &mut R,
    ) -> ProtocolProof<G, F> {
        // 1. Generate folding proof
        let folding_proof = self.generate_folding_proof(instances, rng);

        // 2. Generate zero-knowledge proof
        let zk_proof = self.generate_zk_proof(&folding_proof, rng);

        // 3. Generate final proof
        let final_proof = self.generate_final_proof(&folding_proof, &zk_proof, rng);

        ProtocolProof {
            folding_proof,
            zk_proof,
            final_proof,
        }
    }

    fn generate_folding_proof<R: Rng>(
        &mut self,
        instances: &[LatticeFoldInstance<F>],
        rng: &mut R,
    ) -> FoldingProof<G, F> {
        let mut prover = FoldingProver::new(self.params.clone());
        prover.prove(instances, rng)
    }

    fn generate_zk_proof<R: Rng>(
        &mut self,
        folding_proof: &FoldingProof<G, F>,
        rng: &mut R,
    ) -> ZKProof<G, F> {
        let witness = LatticePoint::new(
            folding_proof.folded_witness.clone(),
            self.params.lattice_relation.basis.clone(),
        );

        let mut prover = ZKProver::new(self.params.clone());
        prover.prove(&witness, rng)
    }

    fn generate_final_proof<R: Rng>(
        &mut self,
        folding_proof: &FoldingProof<G, F>,
        zk_proof: &ZKProof<G, F>,
        rng: &mut R,
    ) -> FinalProof<G, F> {
        // 1. Generate randomness
        let randomness = F::rand(rng);
        
        // 2. Compute the final commitment
        let mut final_commitment = G::Projective::zero();
        
        // 3. Add the folded witness commitment
        for (i, w) in folding_proof.folded_witness.iter().enumerate() {
            let g = &self.params.commitment_scheme.vector_commitment.params.generators[i];
            final_commitment += g.mul(w.into_repr());
        }
        
        // 4. Add the zero-knowledge proof commitment
        final_commitment += zk_proof.commitment.mul(F::one().into_repr());
        
        // 5. Add the randomness
        let h = &self.params.commitment_scheme.vector_commitment.params.h;
        final_commitment += h.mul(randomness.into_repr());
        
        // 6. Generate challenge
        let mut challenge_bytes = [0u8; 32];
        self.transcript.challenge_bytes(b"final_challenge", &mut challenge_bytes);
        let challenge = F::from_random_bytes(&challenge_bytes).unwrap();
        
        // 7. Compute response
        let mut response = Vec::new();
        for (w, z) in folding_proof.folded_witness.iter().zip(zk_proof.response.iter()) {
            response.push(*w * challenge + *z + randomness);
        }
        
        FinalProof {
            commitment: final_commitment.into_affine(),
            response,
            challenge,
        }
    }
}

pub struct ProtocolVerifier<G: AffineCurve, F: Field> {
    pub params: ProtocolParams<G, F>,
    pub transcript: Transcript,
}

impl<G: AffineCurve, F: Field> ProtocolVerifier<G, F> {
    pub fn new(params: ProtocolParams<G, F>) -> Self {
        Self {
            params,
            transcript: Transcript::new(b"LatticeFold+ Protocol"),
        }
    }

    pub fn verify(&mut self, proof: &ProtocolProof<G, F>) -> bool {
        // 1. Verify folding proof
        if !self.verify_folding_proof(&proof.folding_proof) {
            return false;
        }

        // 2. Verify zero-knowledge proof
        if !self.verify_zk_proof(&proof.zk_proof) {
            return false;
        }

        // 3. Verify final proof
        self.verify_final_proof(&proof.final_proof)
    }

    fn verify_folding_proof(&mut self, proof: &FoldingProof<G, F>) -> bool {
        // 1. Verify the folded witness
        let mut folded_commitment = G::Projective::zero();
        for (i, w) in proof.folded_witness.iter().enumerate() {
            let g = &self.params.commitment_scheme.vector_commitment.params.generators[i];
            folded_commitment += g.mul(w.into_repr());
        }
        
        // 2. Verify the folded public input
        let mut folded_public = G::Projective::zero();
        for (i, p) in proof.folded_public.iter().enumerate() {
            let g = &self.params.commitment_scheme.vector_commitment.params.generators[i];
            folded_public += g.mul(p.into_repr());
        }
        
        // 3. Verify the challenge responses
        for (i, response) in proof.responses.iter().enumerate() {
            let mut response_commitment = G::Projective::zero();
            for (j, r) in response.iter().enumerate() {
                let g = &self.params.commitment_scheme.vector_commitment.params.generators[j];
                response_commitment += g.mul(r.into_repr());
            }
            
            if response_commitment.into_affine() != proof.commitments[i] {
                return false;
            }
        }
        
        true
    }

    fn verify_zk_proof(&mut self, proof: &ZKProof<G, F>) -> bool {
        // 1. Verify the commitment
        let mut commitment = G::Projective::zero();
        for (i, r) in proof.response.iter().enumerate() {
            let g = &self.params.commitment_scheme.vector_commitment.params.generators[i];
            commitment += g.mul(r.into_repr());
        }
        
        let h = &self.params.commitment_scheme.vector_commitment.params.h;
        commitment += h.mul(proof.challenge.into_repr());
        
        if commitment.into_affine() != proof.commitment {
            return false;
        }
        
        true
    }

    fn verify_final_proof(&mut self, proof: &FinalProof<G, F>) -> bool {
        // 1. Verify the commitment
        let mut commitment = G::Projective::zero();
        for (i, r) in proof.response.iter().enumerate() {
            let g = &self.params.commitment_scheme.vector_commitment.params.generators[i];
            commitment += g.mul(r.into_repr());
        }
        
        let h = &self.params.commitment_scheme.vector_commitment.params.h;
        commitment += h.mul(proof.challenge.into_repr());
        
        if commitment.into_affine() != proof.commitment {
            return false;
        }
        
        true
    }
}

#[derive(Clone, Debug)]
pub struct ProtocolProof<G: AffineCurve, F: Field> {
    pub folding_proof: FoldingProof<G, F>,
    pub zk_proof: ZKProof<G, F>,
    pub final_proof: FinalProof<G, F>,
}

#[derive(Clone, Debug)]
pub struct FinalProof<G: AffineCurve, F: Field> {
    pub commitment: G,
    pub response: Vec<F>,
    pub challenge: F,
}

#[derive(Clone, Debug)]
pub struct LatticeFoldInstance<F: Field> {
    pub witness: Vec<F>,
    pub public_input: Vec<F>,
} 