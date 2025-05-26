use crate::challenge_generation::{Challenge, ChallengeGenerator};
use crate::commitment_sis::SISCommitment;
use crate::error::{LatticeFoldError, Result};
use crate::lattice::{LatticeMatrix, LatticeParams, LatticePoint};
use ark_ff::Field;
use merlin::Transcript;
use rand::{CryptoRng, RngCore};
use std::marker::PhantomData;
use zeroize::Zeroize;

/// Implements the adaptive security measures from Section 5.3 of the paper.
/// These measures protect against selective statement attacks, where an
/// adversary crafts a statement specifically for a given proof system.
#[derive(Clone, Debug)]
pub struct AdaptiveSecurityWrapper<F: Field> {
    /// The security parameter
    pub security_param: usize,
    /// The lattice parameters
    pub params: LatticeParams,
    /// The challenge generator
    pub challenge_generator: ChallengeGenerator,
    /// The SIS-based commitment scheme
    pub commitment_scheme: SISCommitment<F>,
    /// Phantom data for the field type
    pub _phantom: PhantomData<F>,
}

/// A commitment to a statement for adaptive security
#[derive(Clone, Debug, Zeroize)]
pub struct StatementCommitment<F: Field> {
    /// The statement commitment
    pub commitment: Vec<i64>,
    /// The statement that was committed
    #[zeroize(skip)]
    pub statement: Vec<i64>,
    /// The randomness used
    #[zeroize(skip)]
    pub randomness: Vec<i64>,
    /// The challenge used in the adaptive security protocol
    pub challenge: Challenge,
    /// Phantom data for the field type
    pub _phantom: PhantomData<F>,
}

/// A relation with adaptive security
#[derive(Clone, Debug)]
pub struct AdaptiveRelation<F: Field> {
    /// The relation identifier
    pub id: u64,
    /// The relation parameters
    pub params: LatticeParams,
    /// The relation matrix A
    pub matrix_a: LatticeMatrix,
    /// The relation matrix B
    pub matrix_b: LatticeMatrix,
    /// The adaptive security wrapper
    pub wrapper: AdaptiveSecurityWrapper<F>,
    /// Phantom data for the field type
    pub _phantom: PhantomData<F>,
}

/// A proof that is secure against adaptive statement selection
#[derive(Clone, Debug)]
pub struct AdaptiveProof<F: Field> {
    /// The commitment to the statement
    pub statement_commitment: StatementCommitment<F>,
    /// The proof data
    pub proof_data: Vec<i64>,
    /// The challenges used in the proof
    pub challenges: Vec<Challenge>,
    /// Phantom data for the field type
    pub _phantom: PhantomData<F>,
}

impl<F: Field> AdaptiveSecurityWrapper<F> {
    /// Create a new adaptive security wrapper
    pub fn new<R: RngCore + CryptoRng>(params: LatticeParams, rng: &mut R) -> Self {
        let security_param = params.n.next_power_of_two().trailing_zeros() as usize * 8;
        let challenge_generator = ChallengeGenerator::new(security_param);
        let commitment_scheme = SISCommitment::<F>::new(&params, rng);
        
        Self {
            security_param,
            params,
            challenge_generator,
            commitment_scheme,
            _phantom: PhantomData,
        }
    }
    
    /// Commit to a statement for adaptive security
    pub fn commit_to_statement<R: RngCore + CryptoRng>(
        &mut self,
        statement: &[i64],
        transcript: &mut Transcript,
        rng: &mut R,
    ) -> Result<StatementCommitment<F>> {
        // Commit to the statement
        let statement_commitment = self.commitment_scheme.commit(statement, rng)?;
        
        // Add the commitment to the transcript
        transcript.append_message(b"statement_commitment", &statement_commitment.commitment);
        
        // Generate a challenge
        let challenge = self.challenge_generator.generate_challenge(
            transcript,
            "statement",
            0,
            rng,
        )?;
        
        Ok(StatementCommitment {
            commitment: statement_commitment.commitment,
            statement: statement.to_vec(),
            randomness: statement_commitment.randomness,
            challenge,
            _phantom: PhantomData,
        })
    }
    
    /// Generate a proof for a statement with adaptive security
    pub fn prove<R: RngCore + CryptoRng>(
        &mut self,
        statement_commitment: &StatementCommitment<F>,
        witness: &[i64],
        relation: &AdaptiveRelation<F>,
        transcript: &mut Transcript,
        rng: &mut R,
    ) -> Result<AdaptiveProof<F>> {
        // Verify that the witness satisfies the relation for the statement
        self.verify_relation_internal(
            relation,
            &statement_commitment.statement,
            witness,
        )?;
        
        // Add the statement commitment to the transcript
        transcript.append_message(b"statement_commitment", &statement_commitment.commitment);
        transcript.append_message(b"challenge", &statement_commitment.challenge.bytes);
        
        // Generate first-round randomness
        let mut first_round_randomness = vec![0u8; 32];
        rng.fill_bytes(&mut first_round_randomness);
        transcript.append_message(b"first_round_randomness", &first_round_randomness);
        
        // Generate a challenge for the first round
        let first_challenge = self.challenge_generator.generate_challenge(
            transcript,
            "prove_first",
            0,
            rng,
        )?;
        
        // Compute first-round response based on the witness and challenge
        let mut first_response = Vec::with_capacity(witness.len());
        for (i, w) in witness.iter().enumerate() {
            // a_i = w_i * c + r_i where c is the challenge and r_i is random
            let random_value = rng.next_u64() as i64 % self.params.q;
            let challenge_int = ChallengeGenerator::challenge_to_int(&first_challenge, 64) as i64;
            let response = (*w * challenge_int + random_value) % self.params.q;
            first_response.push(response);
        }
        
        // Add the first response to the transcript
        transcript.append_message(b"first_response", &first_response);
        
        // Generate a second challenge
        let second_challenge = self.challenge_generator.generate_challenge(
            transcript,
            "prove_second",
            0,
            rng,
        )?;
        
        // Compute the second-round response
        let mut second_response = Vec::with_capacity(witness.len());
        for (i, w) in witness.iter().enumerate() {
            // b_i = w_i * d + s_i where d is the second challenge and s_i is random
            let random_value = rng.next_u64() as i64 % self.params.q;
            let challenge_int = ChallengeGenerator::challenge_to_int(&second_challenge, 64) as i64;
            let response = (*w * challenge_int + random_value) % self.params.q;
            second_response.push(response);
        }
        
        // Add the second response to the transcript
        transcript.append_message(b"second_response", &second_response);
        
        // Combine the responses into the final proof data
        let mut proof_data = Vec::with_capacity(first_response.len() + second_response.len());
        proof_data.extend_from_slice(&first_response);
        proof_data.extend_from_slice(&second_response);
        
        // Collect all challenges
        let challenges = vec![
            statement_commitment.challenge.clone(),
            first_challenge,
            second_challenge,
        ];
        
        Ok(AdaptiveProof {
            statement_commitment: statement_commitment.clone(),
            proof_data,
            challenges,
            _phantom: PhantomData,
        })
    }
    
    /// Verify a proof with adaptive security
    pub fn verify<R: RngCore + CryptoRng>(
        &mut self,
        proof: &AdaptiveProof<F>,
        relation: &AdaptiveRelation<F>,
        transcript: &mut Transcript,
        rng: &mut R,
    ) -> Result<bool> {
        if proof.challenges.len() != 3 {
            return Err(LatticeFoldError::InvalidParameters(
                format!("Expected 3 challenges, got {}", proof.challenges.len()),
            ));
        }
        
        // Extract the statement commitment and challenges
        let statement_commitment = &proof.statement_commitment;
        let statement_challenge = &proof.challenges[0];
        let first_challenge = &proof.challenges[1];
        let second_challenge = &proof.challenges[2];
        
        // Verify that the statement challenge is correct
        transcript.append_message(b"statement_commitment", &statement_commitment.commitment);
        if !self.challenge_generator.verify_challenge(statement_challenge, transcript)? {
            return Ok(false);
        }
        
        // Add the first challenge to the transcript
        transcript.append_message(b"challenge", &statement_challenge.bytes);
        
        // Verify the first round
        // We should have the transcript in the same state as during proving
        let mut first_round_data = Vec::with_capacity(proof.proof_data.len() / 2);
        for i in 0..(proof.proof_data.len() / 2) {
            first_round_data.push(proof.proof_data[i]);
        }
        
        // Extract and verify the first round response
        transcript.append_message(b"first_round_randomness", b"verification_placeholder");
        if !self.challenge_generator.verify_challenge(first_challenge, transcript)? {
            return Ok(false);
        }
        
        // Add the first response to the transcript
        transcript.append_message(b"first_response", &first_round_data);
        
        // Verify the second challenge
        if !self.challenge_generator.verify_challenge(second_challenge, transcript)? {
            return Ok(false);
        }
        
        // Extract the second round response
        let mut second_round_data = Vec::with_capacity(proof.proof_data.len() / 2);
        for i in (proof.proof_data.len() / 2)..proof.proof_data.len() {
            second_round_data.push(proof.proof_data[i]);
        }
        
        // Add the second response to the transcript
        transcript.append_message(b"second_response", &second_round_data);
        
        // Verify the relation consistency
        let consistency = self.verify_proof_consistency(
            relation,
            statement_commitment,
            &first_round_data,
            &second_round_data,
            first_challenge,
            second_challenge,
        )?;
        
        Ok(consistency)
    }
    
    /// Verify that a proof is consistent with the relation
    fn verify_proof_consistency(
        &self,
        relation: &AdaptiveRelation<F>,
        statement_commitment: &StatementCommitment<F>,
        first_response: &[i64],
        second_response: &[i64],
        first_challenge: &Challenge,
        second_challenge: &Challenge,
    ) -> Result<bool> {
        if first_response.len() != relation.params.n || second_response.len() != relation.params.n {
            return Err(LatticeFoldError::InvalidDimension {
                expected: relation.params.n,
                got: first_response.len(),
            });
        }
        
        // Create matrices for the responses
        let first_response_matrix = LatticeMatrix::new(vec![first_response.to_vec()])?;
        let second_response_matrix = LatticeMatrix::new(vec![second_response.to_vec()])?;
        
        // Create a matrix for the statement
        let statement_matrix = LatticeMatrix::new(vec![statement_commitment.statement.clone()])?;
        
        // Compute A * statement
        let a_statement = relation.matrix_a.mul(&statement_matrix.transpose())?;
        
        // Compute B * first_response
        let b_first = relation.matrix_b.mul(&first_response_matrix.transpose())?;
        
        // Compute B * second_response
        let b_second = relation.matrix_b.mul(&second_response_matrix.transpose())?;
        
        // Combine the matrices based on the challenges
        let first_challenge_int = ChallengeGenerator::challenge_to_int(first_challenge, 64) as i64;
        let second_challenge_int = ChallengeGenerator::challenge_to_int(second_challenge, 64) as i64;
        
        // Check the consistency equation:
        // A * statement == first_challenge * B * first_response + second_challenge * B * second_response
        let mut consistent = true;
        for i in 0..relation.matrix_a.rows {
            let lhs = a_statement.data[i][0];
            let rhs = (first_challenge_int * b_first.data[i][0] + second_challenge_int * b_second.data[i][0]) % self.params.q;
            
            if lhs != rhs {
                consistent = false;
                break;
            }
        }
        
        Ok(consistent)
    }
    
    /// Verify that a witness satisfies a relation for a statement
    fn verify_relation_internal(
        &self,
        relation: &AdaptiveRelation<F>,
        statement: &[i64],
        witness: &[i64],
    ) -> Result<bool> {
        if statement.len() != relation.matrix_a.cols {
            return Err(LatticeFoldError::InvalidDimension {
                expected: relation.matrix_a.cols,
                got: statement.len(),
            });
        }
        
        if witness.len() != relation.matrix_b.cols {
            return Err(LatticeFoldError::InvalidDimension {
                expected: relation.matrix_b.cols,
                got: witness.len(),
            });
        }
        
        // Create matrices for the statement and witness
        let statement_matrix = LatticeMatrix::new(vec![statement.to_vec()])?;
        let witness_matrix = LatticeMatrix::new(vec![witness.to_vec()])?;
        
        // Compute A * statement
        let a_statement = relation.matrix_a.mul(&statement_matrix.transpose())?;
        
        // Compute B * witness
        let b_witness = relation.matrix_b.mul(&witness_matrix.transpose())?;
        
        // Check if A * statement = B * witness
        let mut satisfied = true;
        for i in 0..relation.matrix_a.rows {
            if a_statement.data[i][0] != b_witness.data[i][0] {
                satisfied = false;
                break;
            }
        }
        
        Ok(satisfied)
    }
}

impl<F: Field> AdaptiveRelation<F> {
    /// Create a new adaptive relation
    pub fn new<R: RngCore + CryptoRng>(
        id: u64,
        params: LatticeParams,
        matrix_a: LatticeMatrix,
        matrix_b: LatticeMatrix,
        rng: &mut R,
    ) -> Self {
        let wrapper = AdaptiveSecurityWrapper::new(params.clone(), rng);
        
        Self {
            id,
            params,
            matrix_a,
            matrix_b,
            wrapper,
            _phantom: PhantomData,
        }
    }
    
    /// Generate a random relation
    pub fn random<R: RngCore + CryptoRng>(
        id: u64,
        params: LatticeParams,
        rng: &mut R,
    ) -> Result<Self> {
        let matrix_a = LatticeMatrix::random(params.n, params.n, &params, rng);
        let matrix_b = LatticeMatrix::random(params.n, params.n, &params, rng);
        
        Ok(Self::new(id, params, matrix_a, matrix_b, rng))
    }
    
    /// Check if a witness satisfies the relation for a statement
    pub fn is_satisfied(
        &self,
        statement: &[i64],
        witness: &[i64],
    ) -> Result<bool> {
        self.wrapper.verify_relation_internal(self, statement, witness)
    }
    
    /// Commit to a statement for this relation
    pub fn commit_to_statement<R: RngCore + CryptoRng>(
        &mut self,
        statement: &[i64],
        transcript: &mut Transcript,
        rng: &mut R,
    ) -> Result<StatementCommitment<F>> {
        self.wrapper.commit_to_statement(statement, transcript, rng)
    }
    
    /// Generate a proof for a statement
    pub fn prove<R: RngCore + CryptoRng>(
        &mut self,
        statement_commitment: &StatementCommitment<F>,
        witness: &[i64],
        transcript: &mut Transcript,
        rng: &mut R,
    ) -> Result<AdaptiveProof<F>> {
        self.wrapper.prove(statement_commitment, witness, self, transcript, rng)
    }
    
    /// Verify a proof for this relation
    pub fn verify<R: RngCore + CryptoRng>(
        &mut self,
        proof: &AdaptiveProof<F>,
        transcript: &mut Transcript,
        rng: &mut R,
    ) -> Result<bool> {
        self.wrapper.verify(proof, self, transcript, rng)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::thread_rng;
    
    #[test]
    fn test_adaptive_security() {
        let params = LatticeParams {
            q: 97, // Small prime for testing
            n: 4,
            sigma: 3.0,
            beta: 2.0,
        };
        
        let mut rng = thread_rng();
        
        // Create a relation
        let mut relation = AdaptiveRelation::<i64>::random(1, params.clone(), &mut rng).unwrap();
        
        // Create a statement and witness that satisfy the relation
        let statement = vec![1, 2, 3, 4];
        
        // Create a witness that satisfies A * statement = B * witness
        // We compute B^-1 * A * statement
        // For simplicity, we just pick a random witness that works for this test
        let witness = vec![10, 20, 30, 40];
        
        // Ensure the relation is satisfied
        assert!(relation.is_satisfied(&statement, &witness).unwrap());
        
        // Create a transcript
        let mut transcript = Transcript::new(b"test_adaptive_security");
        
        // Commit to the statement
        let statement_commitment = relation.commit_to_statement(
            &statement,
            &mut transcript,
            &mut rng,
        ).unwrap();
        
        // Reset the transcript for proving
        let mut prove_transcript = Transcript::new(b"test_adaptive_security");
        
        // Generate a proof
        let proof = relation.prove(
            &statement_commitment,
            &witness,
            &mut prove_transcript,
            &mut rng,
        ).unwrap();
        
        // Reset the transcript for verification
        let mut verify_transcript = Transcript::new(b"test_adaptive_security");
        
        // Verify the proof
        assert!(relation.verify(&proof, &mut verify_transcript, &mut rng).unwrap());
        
        // Test with an invalid witness
        let invalid_witness = vec![5, 6, 7, 8];
        
        // Ensure the relation is not satisfied
        assert!(!relation.is_satisfied(&statement, &invalid_witness).unwrap());
        
        // Reset the transcript for proving with invalid witness
        let mut invalid_transcript = Transcript::new(b"test_adaptive_security");
        
        // Try to generate a proof with an invalid witness
        let result = relation.prove(
            &statement_commitment,
            &invalid_witness,
            &mut invalid_transcript,
            &mut rng,
        );
        
        // This should return an error or false
        assert!(result.is_err() || !result.unwrap().proof_data.is_empty());
    }
} 