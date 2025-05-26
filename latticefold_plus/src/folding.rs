use crate::lattice::{LatticePoint, LatticeParams, LatticeMatrix};
use blake3::Hasher;
use merlin::Transcript;
use rand::{CryptoRng, RngCore};
use std::collections::HashMap;
use zeroize::Zeroize;

#[derive(Clone, Debug, Zeroize)]
pub struct FoldingCommitment {
    point: LatticePoint,
    randomness: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct FoldingProof {
    commitments: Vec<FoldingCommitment>,
    challenge_responses: Vec<LatticePoint>,
    final_point: LatticePoint,
}

pub struct FoldingScheme {
    params: LatticeParams,
    basis_matrix: LatticeMatrix,
    hash_to_point_cache: HashMap<Vec<u8>, LatticePoint>,
}

impl FoldingScheme {
    pub fn new(params: LatticeParams, basis_matrix: LatticeMatrix) -> Self {
        Self {
            params,
            basis_matrix,
            hash_to_point_cache: HashMap::new(),
        }
    }

    pub fn hash_to_point(&mut self, input: &[u8]) -> LatticePoint {
        if let Some(point) = self.hash_to_point_cache.get(input) {
            return point.clone();
        }

        let mut hasher = Hasher::new();
        hasher.update(input);
        let hash = hasher.finalize();
        
        let coords: Vec<i64> = hash.as_bytes()
            .chunks(8)
            .take(self.params.n)
            .map(|chunk| {
                let mut buf = [0u8; 8];
                buf.copy_from_slice(chunk);
                let val = u64::from_le_bytes(buf) as i64;
                val % self.params.q
            })
            .collect();

        let point = LatticePoint::new(coords);
        self.hash_to_point_cache.insert(input.to_vec(), point.clone());
        point
    }

    pub fn commit<R: RngCore + CryptoRng>(
        &mut self,
        point: &LatticePoint,
        transcript: &mut Transcript,
        rng: &mut R,
    ) -> FoldingCommitment {
        let mut randomness = vec![0u8; 32];
        rng.fill_bytes(&mut randomness);
        
        transcript.append_message(b"point", &point.to_bytes());
        transcript.append_message(b"randomness", &randomness);
        
        let blinding = self.hash_to_point(&randomness);
        let committed_point = point.add_mod(&blinding, self.params.q);
        
        FoldingCommitment {
            point: committed_point,
            randomness,
        }
    }

    pub fn prove<R: RngCore + CryptoRng>(
        &mut self,
        points: &[LatticePoint],
        transcript: &mut Transcript,
        rng: &mut R,
    ) -> FoldingProof {
        let mut commitments = Vec::new();
        let mut challenge_responses = Vec::new();
        
        // Initial commitment phase
        for point in points {
            let commitment = self.commit(point, transcript, rng);
            commitments.push(commitment);
        }
        
        // Challenge phase
        let mut challenges = Vec::new();
        for _ in 0..points.len() {
            let mut challenge = [0u8; 32];
            transcript.challenge_bytes(b"fold_challenge", &mut challenge);
            challenges.push(challenge);
        }
        
        // Response phase
        for (i, point) in points.iter().enumerate() {
            let challenge_point = self.hash_to_point(&challenges[i]);
            let response = point.add_mod(&challenge_point, self.params.q);
            challenge_responses.push(response);
        }
        
        // Final folding
        let final_point = self.fold_points(&challenge_responses);
        
        FoldingProof {
            commitments,
            challenge_responses,
            final_point,
        }
    }
    
    pub fn verify(
        &mut self,
        proof: &FoldingProof,
        transcript: &mut Transcript,
    ) -> bool {
        // Verify commitments
        for commitment in &proof.commitments {
            transcript.append_message(b"commitment", &commitment.point.to_bytes());
        }
        
        // Verify challenges
        let mut challenges = Vec::new();
        for _ in 0..proof.commitments.len() {
            let mut challenge = [0u8; 32];
            transcript.challenge_bytes(b"fold_challenge", &mut challenge);
            challenges.push(challenge);
        }
        
        // Verify responses
        for (i, response) in proof.challenge_responses.iter().enumerate() {
            let challenge_point = self.hash_to_point(&challenges[i]);
            let expected = proof.commitments[i].point.add_mod(&challenge_point, self.params.q);
            if !response.equals(&expected) {
                return false;
            }
        }
        
        // Verify final folded point
        let folded = self.fold_points(&proof.challenge_responses);
        folded.equals(&proof.final_point)
    }
    
    fn fold_points(&self, points: &[LatticePoint]) -> LatticePoint {
        points.iter()
            .fold(LatticePoint::zero(self.params.n), |acc, point| {
                acc.add_mod(point, self.params.q)
            })
    }
}

// Extension trait for LatticePoint
trait LatticePointExt {
    fn to_bytes(&self) -> Vec<u8>;
    fn equals(&self, other: &Self) -> bool;
    fn zero(dimension: usize) -> Self;
}

impl LatticePointExt for LatticePoint {
    fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        for coord in &self.coordinates {
            bytes.extend_from_slice(&coord.to_le_bytes());
        }
        bytes
    }
    
    fn equals(&self, other: &Self) -> bool {
        self.coordinates == other.coordinates
    }
    
    fn zero(dimension: usize) -> Self {
        LatticePoint::new(vec![0; dimension])
    }
}
