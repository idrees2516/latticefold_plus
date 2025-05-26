use crate::challenge_generation::{Challenge, ChallengeGenerator};
use crate::error::{LatticeFoldError, Result};
use crate::lattice::{LatticeMatrix, LatticeParams, LatticePoint};
use crate::proof::LatticeProof;
use crate::recursive_folding::{RecursiveFoldProof, RecursiveFoldingParams};
use merlin::Transcript;
use rand::Rng;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

/// Parameters for amortized verification
#[derive(Clone, Debug)]
pub struct AmortizedVerificationParams {
    /// The base lattice parameters
    pub lattice_params: LatticeParams,
    /// The batch size for verification
    pub batch_size: usize,
    /// Whether to use parallel computation
    pub parallel: bool,
    /// The security parameter (bits)
    pub security_param: usize,
}

impl Default for AmortizedVerificationParams {
    fn default() -> Self {
        Self {
            lattice_params: LatticeParams::default(),
            batch_size: 8,
            parallel: true,
            security_param: 128,
        }
    }
}

/// A batch of proofs to be verified together
#[derive(Clone, Debug)]
pub struct ProofBatch<F> {
    /// The proofs in the batch
    pub proofs: Vec<LatticeProof>,
    /// The batch verification challenges
    pub challenges: Vec<Challenge>,
    /// Phantom data for the field type
    pub _phantom: PhantomData<F>,
}

impl<F> ProofBatch<F> {
    /// Create a new proof batch
    pub fn new(proofs: Vec<LatticeProof>) -> Self {
        Self {
            proofs,
            challenges: Vec::new(),
            _phantom: PhantomData,
        }
    }
    
    /// Add a proof to the batch
    pub fn add_proof(&mut self, proof: LatticeProof) {
        self.proofs.push(proof);
    }
    
    /// Get the number of proofs in the batch
    pub fn len(&self) -> usize {
        self.proofs.len()
    }
    
    /// Check if the batch is empty
    pub fn is_empty(&self) -> bool {
        self.proofs.is_empty()
    }
    
    /// Clear the batch
    pub fn clear(&mut self) {
        self.proofs.clear();
        self.challenges.clear();
    }
    
    /// Get the aggregate size of all proofs in the batch in bytes
    pub fn size_in_bytes(&self) -> usize {
        self.proofs.iter().map(|p| p.size_in_bytes()).sum::<usize>()
    }
}

/// A verifier that can amortize proof verification costs
pub struct AmortizedVerifier<F> {
    /// The parameters for amortized verification
    pub params: AmortizedVerificationParams,
    /// The challenge generator
    pub challenge_generator: Arc<Mutex<ChallengeGenerator<F>>>,
    /// The current batch of proofs being processed
    current_batch: ProofBatch<F>,
    /// Counters for verification statistics
    stats: VerificationStats,
}

/// Statistics for verification performance
#[derive(Default, Debug, Clone)]
pub struct VerificationStats {
    /// The number of proofs verified
    pub proofs_verified: usize,
    /// The number of batches processed
    pub batches_processed: usize,
    /// The number of verification operations performed
    pub verification_ops: usize,
}

impl<F> AmortizedVerifier<F> {
    /// Create a new amortized verifier
    pub fn new(
        params: AmortizedVerificationParams,
        challenge_generator: ChallengeGenerator<F>,
    ) -> Self {
        Self {
            params,
            challenge_generator: Arc::new(Mutex::new(challenge_generator)),
            current_batch: ProofBatch::new(Vec::new()),
            stats: VerificationStats::default(),
        }
    }
    
    /// Reset the verification statistics
    pub fn reset_stats(&mut self) {
        self.stats = VerificationStats::default();
    }
    
    /// Get the current verification statistics
    pub fn get_stats(&self) -> VerificationStats {
        self.stats.clone()
    }
    
    /// Add a proof to the batch for verification
    pub fn queue_proof(&mut self, proof: LatticeProof) -> Result<()> {
        // Validate the proof dimensions match the expected parameters
        let expected_dim = self.params.lattice_params.n;
        if proof.commitment.dimension() != expected_dim || proof.response.dimension() != expected_dim {
            return Err(LatticeFoldError::InvalidInput(
                "Proof dimensions do not match verifier parameters".to_string(),
            ));
        }
        
        // Add the proof to the current batch
        self.current_batch.add_proof(proof);
        
        // If the batch is full, verify it automatically
        if self.current_batch.len() >= self.params.batch_size {
            self.verify_current_batch::<rand::rngs::ThreadRng>(None)?;
        }
        
        Ok(())
    }
    
    /// Verify the current batch of proofs
    pub fn verify_current_batch<R: Rng>(&mut self, rng: Option<&mut R>) -> Result<bool> {
        if self.current_batch.is_empty() {
            return Ok(true); // Nothing to verify
        }
        
        // Generate batch verification challenges
        self.generate_batch_challenges(rng)?;
        
        // Perform batch verification
        let result = self.perform_batch_verification()?;
        
        // Update statistics
        self.stats.proofs_verified += self.current_batch.len();
        self.stats.batches_processed += 1;
        self.stats.verification_ops += 1; // One combined operation for the whole batch
        
        // Clear the batch after verification
        self.current_batch.clear();
        
        Ok(result)
    }
    
    /// Generate random challenges for the batch verification
    fn generate_batch_challenges<R: Rng>(&mut self, provided_rng: Option<&mut R>) -> Result<()> {
        let mut challenges = Vec::with_capacity(self.current_batch.len());
        let mut challenge_gen = self.challenge_generator.lock().unwrap();
        
        // Add all proofs to the transcript
        for proof in &self.current_batch.proofs {
            challenge_gen.add_point(&proof.commitment, b"commitment");
            challenge_gen.add_point(&proof.response, b"response");
        }
        
        // Generate a challenge for each proof
        for _ in 0..self.current_batch.len() {
            let challenge = if let Some(rng) = provided_rng {
                challenge_gen.generate_challenge(Some(rng))
            } else {
                challenge_gen.generate_challenge::<rand::rngs::ThreadRng>(None)
            };
            
            challenges.push(challenge);
        }
        
        self.current_batch.challenges = challenges;
        
        Ok(())
    }
    
    /// Perform the actual batch verification
    fn perform_batch_verification(&self) -> Result<bool> {
        if self.current_batch.proofs.is_empty() || self.current_batch.challenges.is_empty() {
            return Err(LatticeFoldError::InvalidState(
                "Cannot perform batch verification with empty proofs or challenges".to_string(),
            ));
        }
        
        // Combine all proofs using their respective challenges
        let mut combined_commitment = LatticePoint::zero(self.params.lattice_params.n);
        let mut combined_response = LatticePoint::zero(self.params.lattice_params.n);
        
        let num_proofs = self.current_batch.len();
        
        for i in 0..num_proofs {
            let proof = &self.current_batch.proofs[i];
            let challenge = &self.current_batch.challenges[i];
            let challenge_int = challenge.as_integer();
            
            // Scale the commitment and response by the challenge
            let mut scaled_commitment = proof.commitment.clone();
            scaled_commitment.scale(challenge_int);
            
            let mut scaled_response = proof.response.clone();
            scaled_response.scale(challenge_int);
            
            // Add to the combined values
            combined_commitment.add(&scaled_commitment);
            combined_response.add(&scaled_response);
        }
        
        // Verify the combined proof
        // In a real implementation, this would depend on the specific verification equations
        // For a general SIS-based verification, we'd check if the matrix multiplication
        // A * combined_response = combined_commitment (mod q)
        // But since we don't have the specific verification equation, we'll just use
        // a simplified version here for illustration
        
        // For now, we'll just check if the combined values are "valid" according to
        // some simple criterion (e.g., norm bound)
        let response_norm = combined_response.norm();
        let commitment_norm = combined_commitment.norm();
        
        // A simplified verification check - in a real implementation this would
        // be according to the specific proof system's verification equation
        let max_allowed_norm = self.params.lattice_params.beta * (num_proofs as f64).sqrt();
        
        Ok(response_norm <= max_allowed_norm && commitment_norm <= max_allowed_norm)
    }
    
    /// Force verification of any pending proofs in the batch
    pub fn flush<R: Rng>(&mut self, rng: Option<&mut R>) -> Result<bool> {
        self.verify_current_batch(rng)
    }
}

/// A structure to handle recursive amortized verification
pub struct RecursiveAmortizedVerifier<F> {
    /// The underlying amortized verifier
    pub amortized_verifier: AmortizedVerifier<F>,
    /// The recursive folding parameters
    pub folding_params: RecursiveFoldingParams,
}

impl<F> RecursiveAmortizedVerifier<F> {
    /// Create a new recursive amortized verifier
    pub fn new(
        amortized_params: AmortizedVerificationParams,
        folding_params: RecursiveFoldingParams,
        challenge_generator: ChallengeGenerator<F>,
    ) -> Self {
        Self {
            amortized_verifier: AmortizedVerifier::new(amortized_params, challenge_generator),
            folding_params,
        }
    }
    
    /// Verify a recursive fold proof
    pub fn verify_recursive<R: Rng>(
        &mut self,
        proof: &RecursiveFoldProof<F>,
        rng: Option<&mut R>,
    ) -> Result<bool> {
        // For recursive proofs, we want to skip verifying all the base proofs
        // and directly verify just the folded proof
        
        // Add the final folded proof to the amortized verifier
        self.amortized_verifier.queue_proof(proof.folded_proof.clone())?;
        
        // Force verification of the current batch
        self.amortized_verifier.flush(rng)
    }
    
    /// Verify a batch of recursive fold proofs
    pub fn verify_recursive_batch<R: Rng>(
        &mut self,
        proofs: &[RecursiveFoldProof<F>],
        rng: Option<&mut R>,
    ) -> Result<bool> {
        if proofs.is_empty() {
            return Ok(true);
        }
        
        // Extract all the folded proofs
        for proof in proofs {
            self.amortized_verifier.queue_proof(proof.folded_proof.clone())?;
        }
        
        // Verify the batch
        self.amortized_verifier.flush(rng)
    }
    
    /// Get the verification statistics
    pub fn get_stats(&self) -> VerificationStats {
        self.amortized_verifier.get_stats()
    }
    
    /// Reset the verification statistics
    pub fn reset_stats(&mut self) {
        self.amortized_verifier.reset_stats();
    }
}

/// A verification aggregator for combining multiple verification results
pub struct VerificationAggregator<F> {
    /// The amortized verification parameters
    pub params: AmortizedVerificationParams,
    /// The current batch of verification results
    results: Vec<(LatticeProof, bool)>,
    /// Phantom data for the field type
    _phantom: PhantomData<F>,
}

impl<F> VerificationAggregator<F> {
    /// Create a new verification aggregator
    pub fn new(params: AmortizedVerificationParams) -> Self {
        Self {
            params,
            results: Vec::new(),
            _phantom: PhantomData,
        }
    }
    
    /// Add a verification result
    pub fn add_result(&mut self, proof: LatticeProof, result: bool) {
        self.results.push((proof, result));
    }
    
    /// Get the aggregated verification result
    pub fn aggregate(&self) -> bool {
        // Simple aggregation: all proofs must be valid
        self.results.iter().all(|(_, result)| *result)
    }
    
    /// Get the detailed verification results
    pub fn get_detailed_results(&self) -> &[(LatticeProof, bool)] {
        &self.results
    }
    
    /// Clear all verification results
    pub fn clear(&mut self) {
        self.results.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::challenge_generation::ChallengeParams;
    use rand::thread_rng;
    
    /// Helper to create a test proof
    fn create_test_proof(dim: usize, seed: usize) -> LatticeProof {
        let mut commitment = LatticePoint::zero(dim);
        let mut response = LatticePoint::zero(dim);
        
        // Create some simple test values
        for j in 0..dim {
            commitment.coordinates[j] = ((seed + 1) * (j + 1)) as i64;
            response.coordinates[j] = ((seed + 2) * (j + 2)) as i64;
        }
        
        LatticeProof {
            commitment,
            response,
            aux_data: None,
        }
    }
    
    #[test]
    fn test_amortized_verification() {
        // Create test parameters
        let lattice_params = LatticeParams {
            q: 97, // Small prime for testing
            n: 4,
            sigma: 3.0,
            beta: 2.0,
        };
        
        let amortized_params = AmortizedVerificationParams {
            lattice_params: lattice_params.clone(),
            batch_size: 3,
            parallel: false,
            security_param: 128,
        };
        
        let challenge_params = ChallengeParams::default();
        let challenge_generator = ChallengeGenerator::<i64>::new(challenge_params);
        
        // Create a verifier
        let mut verifier = AmortizedVerifier::new(amortized_params, challenge_generator);
        
        // Create some test proofs
        let proof1 = create_test_proof(lattice_params.n, 1);
        let proof2 = create_test_proof(lattice_params.n, 2);
        let proof3 = create_test_proof(lattice_params.n, 3);
        
        // Queue the proofs
        verifier.queue_proof(proof1).unwrap();
        verifier.queue_proof(proof2).unwrap();
        
        // Should not trigger automatic verification yet
        assert_eq!(verifier.current_batch.len(), 2);
        assert_eq!(verifier.stats.batches_processed, 0);
        
        // Add one more proof to trigger automatic batch verification
        verifier.queue_proof(proof3).unwrap();
        
        // Should have triggered verification
        assert_eq!(verifier.current_batch.len(), 0);
        assert_eq!(verifier.stats.batches_processed, 1);
        assert_eq!(verifier.stats.proofs_verified, 3);
    }
    
    #[test]
    fn test_manual_batch_verification() {
        // Create test parameters
        let lattice_params = LatticeParams {
            q: 97, // Small prime for testing
            n: 4,
            sigma: 3.0,
            beta: 2.0,
        };
        
        let amortized_params = AmortizedVerificationParams {
            lattice_params: lattice_params.clone(),
            batch_size: 5, // Larger than we'll use
            parallel: false,
            security_param: 128,
        };
        
        let challenge_params = ChallengeParams::default();
        let challenge_generator = ChallengeGenerator::<i64>::new(challenge_params);
        
        // Create a verifier
        let mut verifier = AmortizedVerifier::new(amortized_params, challenge_generator);
        
        // Create some test proofs
        let proof1 = create_test_proof(lattice_params.n, 1);
        let proof2 = create_test_proof(lattice_params.n, 2);
        
        // Queue the proofs
        verifier.queue_proof(proof1).unwrap();
        verifier.queue_proof(proof2).unwrap();
        
        // Manually flush the batch
        let result = verifier.flush::<rand::rngs::ThreadRng>(None).unwrap();
        
        // Check the verification statistics
        assert!(result); // Our test proofs should pass our simplified verification
        assert_eq!(verifier.stats.batches_processed, 1);
        assert_eq!(verifier.stats.proofs_verified, 2);
    }
    
    #[test]
    fn test_verification_aggregator() {
        // Create test parameters
        let lattice_params = LatticeParams {
            q: 97, // Small prime for testing
            n: 4,
            sigma: 3.0,
            beta: 2.0,
        };
        
        let amortized_params = AmortizedVerificationParams {
            lattice_params: lattice_params.clone(),
            batch_size: 3,
            parallel: false,
            security_param: 128,
        };
        
        // Create an aggregator
        let mut aggregator = VerificationAggregator::<i64>::new(amortized_params);
        
        // Create some test proofs
        let proof1 = create_test_proof(lattice_params.n, 1);
        let proof2 = create_test_proof(lattice_params.n, 2);
        let proof3 = create_test_proof(lattice_params.n, 3);
        
        // Add verification results
        aggregator.add_result(proof1, true);
        aggregator.add_result(proof2, true);
        aggregator.add_result(proof3, false);
        
        // Get aggregated result - should be false since one proof failed
        assert!(!aggregator.aggregate());
        
        // Get detailed results
        let detailed = aggregator.get_detailed_results();
        assert_eq!(detailed.len(), 3);
        assert!(detailed[0].1);
        assert!(detailed[1].1);
        assert!(!detailed[2].1);
        
        // Clear and add only valid results
        aggregator.clear();
        aggregator.add_result(proof1, true);
        aggregator.add_result(proof2, true);
        
        // Now aggregated result should be true
        assert!(aggregator.aggregate());
    }
}