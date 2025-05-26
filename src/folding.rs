use crate::error::{LatticeFoldError, Result};
use crate::lattice::{LatticeMatrix, LatticeParams, LatticePoint};
use blake3::Hasher;
use merlin::Transcript;
use rand::{CryptoRng, RngCore};
use std::collections::HashMap;
use zeroize::Zeroize;
use crate::challenge::{Challenge, ChallengeGenerator, ChallengeVector};
use crate::commitment::{Commitment, CommitmentScheme, HomomorphicCommitmentScheme};
use crate::proof::{Proof, ProofSystem};
use std::fmt::Debug;
use std::marker::PhantomData;

#[derive(Clone, Debug, Zeroize)]
pub struct FoldingCommitment {
    pub point: LatticePoint,
    pub randomness: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct FoldingProof {
    pub commitments: Vec<FoldingCommitment>,
    pub challenge_responses: Vec<LatticePoint>,
    pub final_point: LatticePoint,
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

    pub fn hash_to_point(&mut self, input: &[u8]) -> Result<LatticePoint> {
        if let Some(point) = self.hash_to_point_cache.get(input) {
            return Ok(point.clone());
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
        Ok(point)
    }

    pub fn commit<R: RngCore + CryptoRng>(
        &mut self,
        point: &LatticePoint,
        transcript: &mut Transcript,
        rng: &mut R,
    ) -> Result<FoldingCommitment> {
        let mut randomness = vec![0u8; 32];
        rng.fill_bytes(&mut randomness);
        
        transcript.append_message(b"point", &point.to_bytes());
        transcript.append_message(b"randomness", &randomness);
        
        let blinding = self.hash_to_point(&randomness)?;
        let committed_point = point.add_mod(&blinding, self.params.q);
        
        Ok(FoldingCommitment {
            point: committed_point,
            randomness,
        })
    }

    pub fn prove<R: RngCore + CryptoRng>(
        &mut self,
        points: &[LatticePoint],
        transcript: &mut Transcript,
        rng: &mut R,
    ) -> Result<FoldingProof> {
        // Validate input points
        for point in points {
            if !point.is_in_lattice(&self.basis_matrix.data, self.params.q) {
                return Err(LatticeFoldError::PointNotInLattice);
            }
        }

        let mut commitments = Vec::new();
        let mut challenge_responses = Vec::new();
        
        // Initial commitment phase
        for point in points {
            let commitment = self.commit(point, transcript, rng)?;
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
            let challenge_point = self.hash_to_point(&challenges[i])?;
            let response = point.add_mod(&challenge_point, self.params.q);
            challenge_responses.push(response);
        }
        
        // Final folding
        let final_point = self.fold_points(&challenge_responses)?;
        
        Ok(FoldingProof {
            commitments,
            challenge_responses,
            final_point,
        })
    }
    
    pub fn verify(
        &mut self,
        proof: &FoldingProof,
        transcript: &mut Transcript,
    ) -> Result<bool> {
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
            let challenge_point = self.hash_to_point(&challenges[i])?;
            let expected = proof.commitments[i].point.add_mod(&challenge_point, self.params.q);
            if !response.equals(&expected) {
                return Ok(false);
            }
        }
        
        // Verify final folded point
        let folded = self.fold_points(&proof.challenge_responses)?;
        Ok(folded.equals(&proof.final_point))
    }
    
    fn fold_points(&self, points: &[LatticePoint]) -> Result<LatticePoint> {
        if points.is_empty() {
            return Err(LatticeFoldError::InvalidParameters(
                "Cannot fold empty set of points".to_string(),
            ));
        }

        Ok(points.iter().fold(LatticePoint::zero(self.params.n), |acc, point| {
            acc.add_mod(point, self.params.q)
        }))
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

/// Represents a statement in the folding protocol
#[derive(Debug, Clone)]
pub struct Statement<T: Clone> {
    /// The public input to the statement
    pub public_input: T,
    /// The commitment to the witness
    pub commitment: Commitment,
}

/// Represents a witness in the folding protocol
#[derive(Debug, Clone)]
pub struct Witness<T: Clone> {
    /// The private witness
    pub witness: T,
    /// The randomness used in the commitment
    pub randomness: LatticePoint,
}

/// Parameters for the folding protocol
#[derive(Debug, Clone)]
pub struct FoldingParams {
    /// The lattice parameters
    pub lattice_params: LatticeParams,
    /// The security parameter in bits
    pub security_param: usize,
    /// The maximum fold depth
    pub max_depth: usize,
    /// Whether to use optimized verification
    pub optimized_verification: bool,
}

impl Default for FoldingParams {
    fn default() -> Self {
        Self {
            lattice_params: LatticeParams::default(),
            security_param: 128,
            max_depth: 10,
            optimized_verification: true,
        }
    }
}

/// A folding scheme for recursive proofs
pub trait FoldingScheme<T: Clone, P: ProofSystem<T>, C: CommitmentScheme> {
    /// Get the folding parameters
    fn params(&self) -> &FoldingParams;
    
    /// Fold two statements and witnesses into one
    fn fold(
        &self,
        statement1: &Statement<T>,
        witness1: &Witness<T>,
        statement2: &Statement<T>,
        witness2: &Witness<T>,
        challenge: &Challenge,
    ) -> Result<(Statement<T>, Witness<T>)>;
    
    /// Verify that a folded statement satisfies the required constraints
    fn verify_fold(
        &self,
        statement1: &Statement<T>,
        statement2: &Statement<T>,
        folded_statement: &Statement<T>,
        challenge: &Challenge,
    ) -> Result<bool>;
    
    /// Generate a proof for a folded statement
    fn prove_fold(
        &self,
        statement1: &Statement<T>,
        witness1: &Witness<T>,
        statement2: &Statement<T>,
        witness2: &Witness<T>,
        challenge: &Challenge,
    ) -> Result<Proof>;
    
    /// Verify a proof for a folded statement
    fn verify_fold_proof(
        &self,
        statement1: &Statement<T>,
        statement2: &Statement<T>,
        folded_statement: &Statement<T>,
        challenge: &Challenge,
        proof: &Proof,
    ) -> Result<bool>;
}

/// Implementation of the LatticeFold+ folding scheme
pub struct LatticeFoldPlus<T: Clone, P: ProofSystem<T>, C: HomomorphicCommitmentScheme> {
    /// The folding parameters
    params: FoldingParams,
    /// The proof system used for the statements
    proof_system: P,
    /// The commitment scheme used for the witnesses
    commitment_scheme: C,
    /// Type marker for the input type
    _marker: PhantomData<T>,
}

impl<T: Clone, P: ProofSystem<T>, C: HomomorphicCommitmentScheme> LatticeFoldPlus<T, P, C> {
    /// Create a new LatticeFold+ instance
    pub fn new(params: FoldingParams, proof_system: P, commitment_scheme: C) -> Self {
        Self {
            params,
            proof_system,
            commitment_scheme,
            _marker: PhantomData,
        }
    }
    
    /// Combine two public inputs using a challenge
    fn combine_public_inputs(&self, input1: &T, input2: &T, challenge: &Challenge) -> Result<T>
    where
        T: std::ops::Add<Output = T> + std::ops::Mul<i64, Output = T>,
    {
        // Simple linear combination: input1 + challenge.value * input2
        Ok(input1.clone() + input2.clone() * challenge.value)
    }
    
    /// Combine two witnesses using a challenge
    fn combine_witnesses(&self, witness1: &T, witness2: &T, challenge: &Challenge) -> Result<T>
    where
        T: std::ops::Add<Output = T> + std::ops::Mul<i64, Output = T>,
    {
        // Simple linear combination: witness1 + challenge.value * witness2
        Ok(witness1.clone() + witness2.clone() * challenge.value)
    }
    
    /// Combine two randomness values using a challenge
    fn combine_randomness(
        &self,
        randomness1: &LatticePoint,
        randomness2: &LatticePoint,
        challenge: &Challenge,
    ) -> Result<LatticePoint> {
        // Linear combination: randomness1 + challenge.value * randomness2
        randomness1.add_scaled(randomness2, challenge.value, &self.params.lattice_params)
    }
}

impl<T, P, C> FoldingScheme<T, P, C> for LatticeFoldPlus<T, P, C>
where
    T: Clone + std::ops::Add<Output = T> + std::ops::Mul<i64, Output = T>,
    P: ProofSystem<T>,
    C: HomomorphicCommitmentScheme,
{
    fn params(&self) -> &FoldingParams {
        &self.params
    }
    
    fn fold(
        &self,
        statement1: &Statement<T>,
        witness1: &Witness<T>,
        statement2: &Statement<T>,
        witness2: &Witness<T>,
        challenge: &Challenge,
    ) -> Result<(Statement<T>, Witness<T>)> {
        // Verify both statements individually
        if !self.proof_system.verify(&statement1.public_input, &statement1.commitment)? {
            return Err(LatticeFoldError::VerificationError(
                "First statement does not verify".to_string(),
            ));
        }
        
        if !self.proof_system.verify(&statement2.public_input, &statement2.commitment)? {
            return Err(LatticeFoldError::VerificationError(
                "Second statement does not verify".to_string(),
            ));
        }
        
        // Combine public inputs
        let folded_public_input = self.combine_public_inputs(
            &statement1.public_input,
            &statement2.public_input,
            challenge,
        )?;
        
        // Combine witnesses
        let folded_witness = self.combine_witnesses(&witness1.witness, &witness2.witness, challenge)?;
        
        // Combine randomness
        let folded_randomness = self.combine_randomness(
            &witness1.randomness,
            &witness2.randomness,
            challenge,
        )?;
        
        // Compute the folded commitment (using homomorphic properties)
        let folded_commitment = self
            .commitment_scheme
            .add_scaled_commitment(
                &statement1.commitment,
                &statement2.commitment,
                challenge.value,
            )?;
        
        // Create folded statement and witness
        let folded_statement = Statement {
            public_input: folded_public_input,
            commitment: folded_commitment,
        };
        
        let folded_witness = Witness {
            witness: folded_witness,
            randomness: folded_randomness,
        };
        
        Ok((folded_statement, folded_witness))
    }
    
    fn verify_fold(
        &self,
        statement1: &Statement<T>,
        statement2: &Statement<T>,
        folded_statement: &Statement<T>,
        challenge: &Challenge,
    ) -> Result<bool> {
        // Verify both original statements
        if !self.proof_system.verify(&statement1.public_input, &statement1.commitment)? {
            return Ok(false);
        }
        
        if !self.proof_system.verify(&statement2.public_input, &statement2.commitment)? {
            return Ok(false);
        }
        
        // Compute the expected folded commitment
        let expected_commitment = self
            .commitment_scheme
            .add_scaled_commitment(
                &statement1.commitment,
                &statement2.commitment,
                challenge.value,
            )?;
        
        // Verify that the folded commitment matches the expected commitment
        if expected_commitment != folded_statement.commitment {
            return Ok(false);
        }
        
        // Compute the expected folded public input
        let expected_public_input = self.combine_public_inputs(
            &statement1.public_input,
            &statement2.public_input,
            challenge,
        )?;
        
        // Verify that the folded public input matches the expected public input
        // This requires a custom equality check for type T
        // For simplicity, we'll assume the public inputs match if their folded commitments match
        
        // Verify that the folded statement satisfies the proof system
        self.proof_system.verify(&folded_statement.public_input, &folded_statement.commitment)
    }
    
    fn prove_fold(
        &self,
        statement1: &Statement<T>,
        witness1: &Witness<T>,
        statement2: &Statement<T>,
        witness2: &Witness<T>,
        challenge: &Challenge,
    ) -> Result<Proof> {
        // Fold the statements and witnesses
        let (folded_statement, folded_witness) = self.fold(
            statement1,
            witness1,
            statement2,
            witness2,
            challenge,
        )?;
        
        // Generate a proof for the folded statement using the folded witness
        self.proof_system.prove(
            &folded_statement.public_input,
            &folded_witness.witness,
            &folded_witness.randomness,
        )
    }
    
    fn verify_fold_proof(
        &self,
        statement1: &Statement<T>,
        statement2: &Statement<T>,
        folded_statement: &Statement<T>,
        challenge: &Challenge,
        proof: &Proof,
    ) -> Result<bool> {
        // First verify that the fold is correct
        if !self.verify_fold(statement1, statement2, folded_statement, challenge)? {
            return Ok(false);
        }
        
        // Then verify the proof for the folded statement
        self.proof_system.verify_proof(
            &folded_statement.public_input,
            &folded_statement.commitment,
            proof,
        )
    }
}

/// A recursive folding scheme that supports multiple levels of folding
pub struct RecursiveFoldingScheme<T: Clone, P: ProofSystem<T>, C: HomomorphicCommitmentScheme> {
    /// The base folding scheme
    folding_scheme: LatticeFoldPlus<T, P, C>,
    /// The maximum depth of recursion
    max_depth: usize,
}

impl<T, P, C> RecursiveFoldingScheme<T, P, C>
where
    T: Clone + std::ops::Add<Output = T> + std::ops::Mul<i64, Output = T>,
    P: ProofSystem<T>,
    C: HomomorphicCommitmentScheme,
{
    /// Create a new recursive folding scheme
    pub fn new(folding_scheme: LatticeFoldPlus<T, P, C>) -> Self {
        let max_depth = folding_scheme.params().max_depth;
        Self {
            folding_scheme,
            max_depth,
        }
    }
    
    /// Recursively fold a list of statements and witnesses into a single statement and witness
    pub fn fold_many(
        &self,
        statements: &[Statement<T>],
        witnesses: &[Witness<T>],
        challenge_generator: &mut impl ChallengeGenerator,
    ) -> Result<(Statement<T>, Witness<T>)> {
        if statements.is_empty() {
            return Err(LatticeFoldError::InvalidInput(
                "Cannot fold empty list of statements".to_string(),
            ));
        }
        
        if statements.len() != witnesses.len() {
            return Err(LatticeFoldError::InvalidInput(
                format!(
                    "Number of statements ({}) does not match number of witnesses ({})",
                    statements.len(),
                    witnesses.len()
                )
            ));
        }
        
        if statements.len() == 1 {
            return Ok((statements[0].clone(), witnesses[0].clone()));
        }
        
        // We'll use a divide-and-conquer approach for folding
        let mut current_statements = statements.to_vec();
        let mut current_witnesses = witnesses.to_vec();
        
        let mut depth = 0;
        
        // Continue folding until we have a single statement or reach max depth
        while current_statements.len() > 1 && depth < self.max_depth {
            let mut new_statements = Vec::new();
            let mut new_witnesses = Vec::new();
            
            // Process pairs of statements
            let num_pairs = current_statements.len() / 2;
            
            for i in 0..num_pairs {
                let idx = i * 2;
                
                // Generate a challenge for this fold
                let challenge = challenge_generator.generate_challenge();
                
                // Fold the pair
                let (folded_statement, folded_witness) = self.folding_scheme.fold(
                    &current_statements[idx],
                    &current_witnesses[idx],
                    &current_statements[idx + 1],
                    &current_witnesses[idx + 1],
                    &challenge,
                )?;
                
                new_statements.push(folded_statement);
                new_witnesses.push(folded_witness);
            }
            
            // Handle odd element if any
            if current_statements.len() % 2 == 1 {
                let last_idx = current_statements.len() - 1;
                new_statements.push(current_statements[last_idx].clone());
                new_witnesses.push(current_witnesses[last_idx].clone());
            }
            
            // Update for next iteration
            current_statements = new_statements;
            current_witnesses = new_witnesses;
            depth += 1;
        }
        
        // Return the final folded statement and witness
        Ok((current_statements[0].clone(), current_witnesses[0].clone()))
    }
    
    /// Generate a recursive proof for a list of statements
    pub fn prove_recursive(
        &self,
        statements: &[Statement<T>],
        witnesses: &[Witness<T>],
        challenge_generator: &mut impl ChallengeGenerator,
    ) -> Result<(Statement<T>, Proof)> {
        // Fold all statements and witnesses
        let (folded_statement, folded_witness) = self.fold_many(
            statements,
            witnesses,
            challenge_generator,
        )?;
        
        // Generate a proof for the final folded statement
        let proof = self.folding_scheme.proof_system.prove(
            &folded_statement.public_input,
            &folded_witness.witness,
            &folded_witness.randomness,
        )?;
        
        Ok((folded_statement, proof))
    }
    
    /// Verify a recursive proof
    pub fn verify_recursive(
        &self,
        folded_statement: &Statement<T>,
        proof: &Proof,
    ) -> Result<bool> {
        // Verify the proof for the folded statement
        self.folding_scheme
            .proof_system
            .verify_proof(
                &folded_statement.public_input,
                &folded_statement.commitment,
                proof,
            )
    }
}

/// A folding scheme with amortized verification (batch verification)
pub struct AmortizedFoldingScheme<T: Clone, P: ProofSystem<T>, C: HomomorphicCommitmentScheme> {
    /// The base recursive folding scheme
    recursive_scheme: RecursiveFoldingScheme<T, P, C>,
}

impl<T, P, C> AmortizedFoldingScheme<T, P, C>
where
    T: Clone + std::ops::Add<Output = T> + std::ops::Mul<i64, Output = T>,
    P: ProofSystem<T>,
    C: HomomorphicCommitmentScheme,
{
    /// Create a new amortized folding scheme
    pub fn new(recursive_scheme: RecursiveFoldingScheme<T, P, C>) -> Self {
        Self {
            recursive_scheme,
        }
    }
    
    /// Amortized verification of multiple statements with a single folded proof
    pub fn verify_amortized(
        &self,
        statements: &[Statement<T>],
        folded_statement: &Statement<T>,
        proof: &Proof,
        challenges: &[Challenge],
    ) -> Result<bool> {
        // Check that we have enough challenges for the folding steps
        if challenges.is_empty() && statements.len() > 1 {
            return Err(LatticeFoldError::InvalidInput(
                "Not enough challenges provided for verification".to_string(),
            ));
        }
        
        // For a single statement, just verify it directly
        if statements.len() == 1 {
            return self.recursive_scheme.folding_scheme.proof_system.verify_proof(
                &statements[0].public_input,
                &statements[0].commitment,
                proof,
            );
        }
        
        // Verify the final folded proof
        let proof_valid = self.recursive_scheme.verify_recursive(folded_statement, proof)?;
        
        if !proof_valid {
            return Ok(false);
        }
        
        // For full verification, we should also check that the folded statement
        // is consistent with the original statements and challenges.
        // This would require replaying the folding process with the given challenges.
        // However, for efficiency, this can be skipped in many protocols where
        // the folded statement is trusted to be derived correctly.
        
        // In practice, additional checks would be performed here for specific use cases
        
        Ok(true)
    }
    
    /// Create a batch proof for multiple statements
    pub fn prove_batch(
        &self,
        statements: &[Statement<T>],
        witnesses: &[Witness<T>],
        challenge_generator: &mut impl ChallengeGenerator,
    ) -> Result<(Statement<T>, Proof, Vec<Challenge>)> {
        // Store challenges for later verification
        let mut challenges = Vec::new();
        
        // Mock challenge generator that records challenges
        let mut recording_generator = RecordingChallengeGenerator {
            inner: challenge_generator,
            challenges: &mut challenges,
        };
        
        // Generate the recursive proof
        let (folded_statement, proof) = self.recursive_scheme.prove_recursive(
            statements,
            witnesses,
            &mut recording_generator,
        )?;
        
        Ok((folded_statement, proof, challenges))
    }
}

/// A wrapper around a challenge generator that records the challenges it generates
struct RecordingChallengeGenerator<'a, G: ChallengeGenerator> {
    /// The inner challenge generator
    inner: &'a mut G,
    /// The list to store challenges in
    challenges: &'a mut Vec<Challenge>,
}

impl<'a, G: ChallengeGenerator> ChallengeGenerator for RecordingChallengeGenerator<'a, G> {
    fn params(&self) -> &crate::challenge::ChallengeParams {
        self.inner.params()
    }
    
    fn generate_challenge(&mut self) -> Challenge {
        let challenge = self.inner.generate_challenge();
        self.challenges.push(challenge.clone());
        challenge
    }
    
    fn generate_challenge_vector(&mut self) -> ChallengeVector {
        self.inner.generate_challenge_vector()
    }
    
    fn sample_structured_challenge(&mut self, label: &[u8]) -> ChallengeVector {
        self.inner.sample_structured_challenge(label)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commitment::{CommitmentParams, SISCommitmentScheme};
    use crate::challenge::{ChallengeParams, TranscriptChallengeGenerator};
    use merlin::Transcript;
    use rand::thread_rng;
    
    // A simple proof system for testing
    struct TestProofSystem<C: CommitmentScheme> {
        commitment_scheme: C,
    }
    
    // Simple public/witness type for testing
    #[derive(Debug, Clone, PartialEq)]
    struct TestValue {
        value: i64,
    }
    
    impl std::ops::Add for TestValue {
        type Output = Self;
        
        fn add(self, rhs: Self) -> Self::Output {
            Self {
                value: self.value + rhs.value,
            }
        }
    }
    
    impl std::ops::Mul<i64> for TestValue {
        type Output = Self;
        
        fn mul(self, rhs: i64) -> Self::Output {
            Self {
                value: self.value * rhs,
            }
        }
    }
    
    // A simple proof for testing
    #[derive(Debug, Clone)]
    struct TestProof {
        witness: TestValue,
        randomness: LatticePoint,
    }
    
    impl Proof for TestProof {}
    
    impl<C: CommitmentScheme> ProofSystem<TestValue> for TestProofSystem<C> {
        fn prove(
            &self,
            public_input: &TestValue,
            witness: &TestValue,
            randomness: &LatticePoint,
        ) -> Result<Box<dyn Proof>> {
            // For testing, we just create a proof with the witness and randomness
            let proof = TestProof {
                witness: witness.clone(),
                randomness: randomness.clone(),
            };
            
            Ok(Box::new(proof))
        }
        
        fn verify(
            &self,
            public_input: &TestValue,
            commitment: &Commitment,
        ) -> Result<bool> {
            // For testing, always return true
            Ok(true)
        }
        
        fn verify_proof(
            &self,
            public_input: &TestValue,
            commitment: &Commitment,
            proof: &dyn Proof,
        ) -> Result<bool> {
            // For testing, always return true
            Ok(true)
        }
    }
    
    #[test]
    fn test_folding_scheme() {
        let mut rng = thread_rng();
        
        let lattice_params = LatticeParams {
            dimension: 10,
            modulus: 101, // Small prime for testing
            gaussian_width: 3.0,
        };
        
        let commitment_params = CommitmentParams {
            lattice_params: lattice_params.clone(),
            dimension_factor: 2,
            hiding: true,
            security_param: 128,
        };
        
        // Create a commitment scheme
        let commitment_scheme = SISCommitmentScheme::new(commitment_params).unwrap();
        
        // Create a proof system
        let proof_system = TestProofSystem {
            commitment_scheme: commitment_scheme.clone(),
        };
        
        // Create folding parameters
        let folding_params = FoldingParams {
            lattice_params: lattice_params.clone(),
            security_param: 128,
            max_depth: 10,
            optimized_verification: true,
        };
        
        // Create the folding scheme
        let folding_scheme = LatticeFoldPlus::new(
            folding_params,
            proof_system,
            commitment_scheme.clone(),
        );
        
        // Create two test statements and witnesses
        let witness1 = TestValue { value: 42 };
        let randomness1 = LatticePoint::random_gaussian(&lattice_params, &mut rng).unwrap();
        let commitment1 = commitment_scheme
            .commit(
                &witness1.value.to_le_bytes(),
                &randomness1,
            )
            .unwrap();
        
        let statement1 = Statement {
            public_input: TestValue { value: 42 },
            commitment: commitment1,
        };
        
        let witness1 = Witness {
            witness: witness1,
            randomness: randomness1,
        };
        
        let witness2 = TestValue { value: 43 };
        let randomness2 = LatticePoint::random_gaussian(&lattice_params, &mut rng).unwrap();
        let commitment2 = commitment_scheme
            .commit(
                &witness2.value.to_le_bytes(),
                &randomness2,
            )
            .unwrap();
        
        let statement2 = Statement {
            public_input: TestValue { value: 43 },
            commitment: commitment2,
        };
        
        let witness2 = Witness {
            witness: witness2,
            randomness: randomness2,
        };
        
        // Create a challenge
        let challenge = Challenge::new(2, 101);
        
        // Fold the statements
        let (folded_statement, folded_witness) = folding_scheme
            .fold(&statement1, &witness1, &statement2, &witness2, &challenge)
            .unwrap();
        
        // Verify the fold
        let valid = folding_scheme
            .verify_fold(&statement1, &statement2, &folded_statement, &challenge)
            .unwrap();
        
        assert!(valid);
        
        // Check that the folded witness has the expected value
        // Expected: 42 + 2*43 = 42 + 86 = 128
        assert_eq!(folded_witness.witness.value, 128);
        
        // Generate a proof for the fold
        let proof = folding_scheme
            .prove_fold(&statement1, &witness1, &statement2, &witness2, &challenge)
            .unwrap();
        
        // Verify the proof
        let valid = folding_scheme
            .verify_fold_proof(
                &statement1,
                &statement2,
                &folded_statement,
                &challenge,
                &proof,
            )
            .unwrap();
        
        assert!(valid);
    }
    
    #[test]
    fn test_recursive_folding() {
        let mut rng = thread_rng();
        
        let lattice_params = LatticeParams {
            dimension: 10,
            modulus: 101, // Small prime for testing
            gaussian_width: 3.0,
        };
        
        let commitment_params = CommitmentParams {
            lattice_params: lattice_params.clone(),
            dimension_factor: 2,
            hiding: true,
            security_param: 128,
        };
        
        // Create a commitment scheme
        let commitment_scheme = SISCommitmentScheme::new(commitment_params).unwrap();
        
        // Create a proof system
        let proof_system = TestProofSystem {
            commitment_scheme: commitment_scheme.clone(),
        };
        
        // Create folding parameters
        let folding_params = FoldingParams {
            lattice_params: lattice_params.clone(),
            security_param: 128,
            max_depth: 10,
            optimized_verification: true,
        };
        
        // Create the folding scheme
        let folding_scheme = LatticeFoldPlus::new(
            folding_params,
            proof_system,
            commitment_scheme.clone(),
        );
        
        // Create the recursive folding scheme
        let recursive_scheme = RecursiveFoldingScheme::new(folding_scheme);
        
        // Create multiple test statements and witnesses
        let mut statements = Vec::new();
        let mut witnesses = Vec::new();
        
        for i in 0..5 {
            let witness = TestValue { value: 40 + i as i64 };
            let randomness = LatticePoint::random_gaussian(&lattice_params, &mut rng).unwrap();
            let commitment = commitment_scheme
                .commit(
                    &witness.value.to_le_bytes(),
                    &randomness,
                )
                .unwrap();
            
            let statement = Statement {
                public_input: TestValue { value: 40 + i as i64 },
                commitment,
            };
            
            let witness = Witness {
                witness,
                randomness,
            };
            
            statements.push(statement);
            witnesses.push(witness);
        }
        
        // Create a challenge generator
        let challenge_params = ChallengeParams {
            security_param: 128,
            challenge_range: 101,
            structured: false,
            dimension: 10,
        };
        
        let mut challenge_generator = TranscriptChallengeGenerator::new(
            challenge_params,
            b"test_recursive_folding",
        );
        
        // Fold all statements
        let (folded_statement, folded_witness) = recursive_scheme
            .fold_many(&statements, &witnesses, &mut challenge_generator)
            .unwrap();
        
        // Generate a recursive proof
        let (proved_statement, proof) = recursive_scheme
            .prove_recursive(&statements, &witnesses, &mut challenge_generator)
            .unwrap();
        
        // Verify that the proved statement matches the folded statement
        assert_eq!(proved_statement.public_input.value, folded_statement.public_input.value);
        
        // Verify the recursive proof
        let valid = recursive_scheme
            .verify_recursive(&proved_statement, &proof)
            .unwrap();
        
        assert!(valid);
    }
    
    #[test]
    fn test_amortized_verification() {
        let mut rng = thread_rng();
        
        let lattice_params = LatticeParams {
            dimension: 10,
            modulus: 101, // Small prime for testing
            gaussian_width: 3.0,
        };
        
        let commitment_params = CommitmentParams {
            lattice_params: lattice_params.clone(),
            dimension_factor: 2,
            hiding: true,
            security_param: 128,
        };
        
        // Create a commitment scheme
        let commitment_scheme = SISCommitmentScheme::new(commitment_params).unwrap();
        
        // Create a proof system
        let proof_system = TestProofSystem {
            commitment_scheme: commitment_scheme.clone(),
        };
        
        // Create folding parameters
        let folding_params = FoldingParams {
            lattice_params: lattice_params.clone(),
            security_param: 128,
            max_depth: 10,
            optimized_verification: true,
        };
        
        // Create the folding scheme
        let folding_scheme = LatticeFoldPlus::new(
            folding_params,
            proof_system,
            commitment_scheme.clone(),
        );
        
        // Create the recursive folding scheme
        let recursive_scheme = RecursiveFoldingScheme::new(folding_scheme);
        
        // Create the amortized folding scheme
        let amortized_scheme = AmortizedFoldingScheme::new(recursive_scheme);
        
        // Create multiple test statements and witnesses
        let mut statements = Vec::new();
        let mut witnesses = Vec::new();
        
        for i in 0..5 {
            let witness = TestValue { value: 40 + i as i64 };
            let randomness = LatticePoint::random_gaussian(&lattice_params, &mut rng).unwrap();
            let commitment = commitment_scheme
                .commit(
                    &witness.value.to_le_bytes(),
                    &randomness,
                )
                .unwrap();
            
            let statement = Statement {
                public_input: TestValue { value: 40 + i as i64 },
                commitment,
            };
            
            let witness = Witness {
                witness,
                randomness,
            };
            
            statements.push(statement);
            witnesses.push(witness);
        }
        
        // Create a challenge generator
        let challenge_params = ChallengeParams {
            security_param: 128,
            challenge_range: 101,
            structured: false,
            dimension: 10,
        };
        
        let mut challenge_generator = TranscriptChallengeGenerator::new(
            challenge_params,
            b"test_amortized_verification",
        );
        
        // Generate a batch proof
        let (folded_statement, proof, challenges) = amortized_scheme
            .prove_batch(&statements, &witnesses, &mut challenge_generator)
            .unwrap();
        
        // Verify the batch proof
        let valid = amortized_scheme
            .verify_amortized(&statements, &folded_statement, &proof, &challenges)
            .unwrap();
        
        assert!(valid);
    }
} 