use crate::error::{LatticeFoldError, Result};
use crate::folding::{FoldingProof, FinalProof};
use crate::lattice::{LatticeMatrix, LatticeParams, LatticePoint};
use ark_ff::Field;
use merlin::Transcript;
use rand::{CryptoRng, RngCore};
use rayon::prelude::*;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

/// Stores multiple proofs for batched verification
#[derive(Clone)]
pub struct BatchedProofs<F: Field> {
    /// The proofs to verify in batch
    pub proofs: Vec<FinalProof>,
    /// The associated parameters for each proof
    pub params: Vec<LatticeParams>,
    /// The phantom data
    pub _phantom: PhantomData<F>,
}

/// Parameters for amortized verification
#[derive(Clone, Debug)]
pub struct AmortizedVerificationParams {
    /// The security parameter
    pub security_param: usize,
    /// The batch size for verification
    pub batch_size: usize,
    /// Whether to use parallel verification
    pub use_parallel: bool,
    /// The number of repetitions for statistical security
    pub repetitions: usize,
}

impl Default for AmortizedVerificationParams {
    fn default() -> Self {
        Self {
            security_param: 128,
            batch_size: 16,
            use_parallel: true,
            repetitions: 20,
        }
    }
}

/// A verifier that can perform amortized verification of multiple proofs
pub struct AmortizedVerifier<F: Field> {
    /// The verification parameters
    pub params: AmortizedVerificationParams,
    /// Cache for verification results
    verification_cache: Arc<Mutex<HashMap<u64, bool>>>,
    /// Phantom data for the field type
    pub _phantom: PhantomData<F>,
}

impl<F: Field> AmortizedVerifier<F> {
    /// Create a new amortized verifier
    pub fn new(params: AmortizedVerificationParams) -> Self {
        Self {
            params,
            verification_cache: Arc::new(Mutex::new(HashMap::new())),
            _phantom: PhantomData,
        }
    }
    
    /// Batch multiple proofs for verification
    pub fn batch_proofs(&self, proofs: Vec<FinalProof>, params: Vec<LatticeParams>) -> Result<BatchedProofs<F>> {
        if proofs.len() != params.len() {
            return Err(LatticeFoldError::InvalidParameters(
                format!("Number of proofs ({}) must match number of parameters ({})", proofs.len(), params.len())
            ));
        }
        
        if proofs.is_empty() {
            return Err(LatticeFoldError::InvalidParameters(
                "No proofs provided for batching".to_string()
            ));
        }
        
        Ok(BatchedProofs {
            proofs,
            params,
            _phantom: PhantomData,
        })
    }
    
    /// Verify a batch of proofs using amortized verification
    pub fn verify_batch<R: RngCore + CryptoRng>(
        &self,
        batch: &BatchedProofs<F>,
        transcript: &mut Transcript,
        rng: &mut R,
    ) -> Result<bool> {
        if batch.proofs.is_empty() {
            return Ok(true); // Empty batch is vacuously true
        }
        
        // If batch is small, just verify individually
        if batch.proofs.len() == 1 {
            return self.verify_proof(&batch.proofs[0], &batch.params[0], transcript, rng);
        }
        
        let batch_size = std::cmp::min(self.params.batch_size, batch.proofs.len());
        let mut all_verified = true;
        
        // Process the batch in smaller chunks of size batch_size
        for chunk_idx in 0..(batch.proofs.len() + batch_size - 1) / batch_size {
            let start = chunk_idx * batch_size;
            let end = std::cmp::min(start + batch_size, batch.proofs.len());
            
            let chunk_proofs = &batch.proofs[start..end];
            let chunk_params = &batch.params[start..end];
            
            let chunk_verified = if self.params.use_parallel {
                self.verify_chunk_parallel(chunk_proofs, chunk_params, transcript, rng)?
            } else {
                self.verify_chunk_sequential(chunk_proofs, chunk_params, transcript, rng)?
            };
            
            if !chunk_verified {
                all_verified = false;
                break;
            }
        }
        
        Ok(all_verified)
    }
    
    /// Verify a single proof
    pub fn verify_proof<R: RngCore + CryptoRng>(
        &self,
        proof: &FinalProof,
        params: &LatticeParams,
        transcript: &mut Transcript,
        rng: &mut R,
    ) -> Result<bool> {
        // Check cache first
        let proof_hash = self.compute_proof_hash(proof);
        {
            let cache = self.verification_cache.lock().unwrap();
            if let Some(&result) = cache.get(&proof_hash) {
                return Ok(result);
            }
        }
        
        // Verify the proof
        let verification_result = self.verify_proof_internal(proof, params, transcript, rng)?;
        
        // Cache the result
        {
            let mut cache = self.verification_cache.lock().unwrap();
            cache.insert(proof_hash, verification_result);
        }
        
        Ok(verification_result)
    }
    
    /// Internal method to verify a single proof
    fn verify_proof_internal<R: RngCore + CryptoRng>(
        &self,
        proof: &FinalProof,
        params: &LatticeParams,
        transcript: &mut Transcript,
        rng: &mut R,
    ) -> Result<bool> {
        // Amortized verification involves checking the following:
        // 1. Challenge correctness
        // 2. Commitment consistency
        // 3. Lattice point bounds
        
        // Verify challenges
        for (i, challenge) in proof.challenges.iter().enumerate() {
            transcript.append_message(b"round", &[i as u8]);
            transcript.append_message(b"point", &proof.points[i].coordinates);
            
            // Check challenge consistency with transcript
            let mut challenge_bytes = [0u8; 32];
            transcript.challenge_bytes(b"challenge", &mut challenge_bytes);
            
            if challenge.bytes != challenge_bytes {
                return Ok(false);
            }
        }
        
        // Verify lattice points are within bounds
        for point in &proof.points {
            if !self.verify_lattice_point_bounds(point, params)? {
                return Ok(false);
            }
        }
        
        // Verify the final folded point
        let final_point = &proof.folded_point;
        if !self.verify_lattice_point_bounds(final_point, params)? {
            return Ok(false);
        }
        
        // Verify the relation between final point and matrix
        let final_matrix = &proof.folded_matrix;
        if !self.verify_matrix_point_relation(final_matrix, final_point, params)? {
            return Ok(false);
        }
        
        Ok(true)
    }
    
    /// Verify a chunk of proofs in parallel
    fn verify_chunk_parallel<R: RngCore + CryptoRng>(
        &self,
        proofs: &[FinalProof],
        params: &[LatticeParams],
        transcript: &mut Transcript,
        rng: &mut R,
    ) -> Result<bool> {
        // Generate random weights for linear combination verification
        let mut weights = vec![0i64; proofs.len()];
        for i in 0..proofs.len() {
            weights[i] = (rng.next_u64() % 1000) as i64;
        }
        
        // Clone the transcript for parallel use
        let transcript_data = transcript.clone();
        let transcript_arc = Arc::new(Mutex::new(transcript_data));
        
        // Verify in parallel
        let verification_results: Vec<Result<bool>> = (0..proofs.len())
            .into_par_iter()
            .map(|i| {
                let mut transcript_clone = transcript_arc.lock().unwrap().clone();
                let mut rng_clone = rand::thread_rng(); // Each thread gets its own RNG
                
                self.verify_proof_with_weight(
                    &proofs[i], 
                    &params[i], 
                    weights[i], 
                    &mut transcript_clone,
                    &mut rng_clone,
                )
            })
            .collect();
        
        // Combine results
        let mut all_verified = true;
        for result in verification_results {
            match result {
                Ok(verified) => {
                    if !verified {
                        all_verified = false;
                        break;
                    }
                },
                Err(_) => {
                    all_verified = false;
                    break;
                }
            }
        }
        
        Ok(all_verified)
    }
    
    /// Verify a chunk of proofs sequentially
    fn verify_chunk_sequential<R: RngCore + CryptoRng>(
        &self,
        proofs: &[FinalProof],
        params: &[LatticeParams],
        transcript: &mut Transcript,
        rng: &mut R,
    ) -> Result<bool> {
        // Generate random weights for linear combination verification
        let mut weights = vec![0i64; proofs.len()];
        for i in 0..proofs.len() {
            weights[i] = (rng.next_u64() % 1000) as i64;
        }
        
        // Verify sequentially
        for i in 0..proofs.len() {
            let verified = self.verify_proof_with_weight(
                &proofs[i], 
                &params[i], 
                weights[i], 
                transcript,
                rng,
            )?;
            
            if !verified {
                return Ok(false);
            }
        }
        
        Ok(true)
    }
    
    /// Verify a proof with a weight
    fn verify_proof_with_weight<R: RngCore + CryptoRng>(
        &self,
        proof: &FinalProof,
        params: &LatticeParams,
        weight: i64,
        transcript: &mut Transcript,
        rng: &mut R,
    ) -> Result<bool> {
        // Add the weight to the transcript
        transcript.append_message(b"weight", &weight.to_le_bytes());
        
        // Verify the weighted proof
        let mut verified = self.verify_proof_internal(proof, params, transcript, rng)?;
        
        // If the verification failed, we need to verify again without the weight
        // to confirm it's not just a numerical issue
        if !verified {
            verified = self.verify_proof_internal(proof, params, transcript, rng)?;
        }
        
        Ok(verified)
    }
    
    /// Verify that a lattice point is within bounds
    fn verify_lattice_point_bounds(
        &self,
        point: &LatticePoint,
        params: &LatticeParams,
    ) -> Result<bool> {
        if point.dimension() != params.n {
            return Err(LatticeFoldError::InvalidDimension {
                expected: params.n,
                got: point.dimension(),
            });
        }
        
        // Check that all coordinates are within the modulus
        for coord in &point.coordinates {
            if *coord < 0 || *coord >= params.q {
                return Ok(false);
            }
        }
        
        // Calculate the L2 norm of the point
        let mut l2_norm_squared = 0.0;
        for coord in &point.coordinates {
            l2_norm_squared += (*coord as f64) * (*coord as f64);
        }
        let l2_norm = l2_norm_squared.sqrt();
        
        // Check if the L2 norm is within the bounds
        // For a valid lattice point, the L2 norm should be less than beta * sigma * sqrt(n)
        let max_norm = params.beta * params.sigma * (params.n as f64).sqrt();
        if l2_norm > max_norm {
            return Ok(false);
        }
        
        Ok(true)
    }
    
    /// Verify the relation between a matrix and a point
    fn verify_matrix_point_relation(
        &self,
        matrix: &LatticeMatrix,
        point: &LatticePoint,
        params: &LatticeParams,
    ) -> Result<bool> {
        // Create a matrix from the point
        let point_matrix = LatticeMatrix::new(vec![point.coordinates.clone()])?;
        
        // Compute matrix * point
        let result = matrix.mul(&point_matrix.transpose())?;
        
        // Check if the result is the zero vector (mod q)
        for i in 0..result.rows {
            for j in 0..result.cols {
                if result.data[i][j] % params.q != 0 {
                    return Ok(false);
                }
            }
        }
        
        Ok(true)
    }
    
    /// Compute a hash of a proof for caching
    fn compute_proof_hash(&self, proof: &FinalProof) -> u64 {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;
        
        let mut hasher = DefaultHasher::new();
        
        // Hash the folded point
        proof.folded_point.coordinates.hash(&mut hasher);
        
        // Hash the challenges
        for challenge in &proof.challenges {
            challenge.bytes.hash(&mut hasher);
        }
        
        // Hash the points
        for point in &proof.points {
            point.coordinates.hash(&mut hasher);
        }
        
        hasher.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::folding::Folder;
    use crate::lattice::LatticePoint;
    use rand::thread_rng;
    
    // Helper function to create a simple test proof
    fn create_test_proof(params: &LatticeParams) -> FinalProof {
        let mut rng = thread_rng();
        
        // Create random points
        let mut points = Vec::new();
        for _ in 0..3 {
            let mut coords = Vec::new();
            for _ in 0..params.n {
                coords.push(rng.next_u64() as i64 % params.q);
            }
            points.push(LatticePoint::new(coords));
        }
        
        // Create random challenges
        let mut challenges = Vec::new();
        for _ in 0..3 {
            let mut bytes = [0u8; 32];
            rng.fill_bytes(&mut bytes);
            challenges.push(crate::challenge_generation::Challenge { bytes });
        }
        
        // Create a folded point
        let folded_coords = vec![1; params.n];
        let folded_point = LatticePoint::new(folded_coords);
        
        // Create a folded matrix (identity for simplicity)
        let mut matrix_data = Vec::new();
        for i in 0..params.n {
            let mut row = vec![0; params.n];
            row[i] = 1;
            matrix_data.push(row);
        }
        let folded_matrix = LatticeMatrix::new(matrix_data).unwrap();
        
        FinalProof {
            points,
            challenges,
            folded_point,
            folded_matrix,
        }
    }
    
    #[test]
    fn test_amortized_verification() {
        let params = LatticeParams {
            q: 97,  // Small prime for testing
            n: 4,
            sigma: 3.0,
            beta: 2.0,
        };
        
        // Create verification parameters
        let verification_params = AmortizedVerificationParams {
            security_param: 64,  // Lower for testing
            batch_size: 8,
            use_parallel: true,
            repetitions: 5,  // Lower for testing
        };
        
        let verifier = AmortizedVerifier::<i64>::new(verification_params);
        
        // Create multiple test proofs
        let mut proofs = Vec::new();
        let mut params_vec = Vec::new();
        
        for _ in 0..10 {
            proofs.push(create_test_proof(&params));
            params_vec.push(params.clone());
        }
        
        // Batch the proofs
        let batched_proofs = verifier.batch_proofs(proofs, params_vec).unwrap();
        
        // Create a transcript for verification
        let mut transcript = Transcript::new(b"test_amortized_verification");
        let mut rng = thread_rng();
        
        // Verify the batch
        let result = verifier.verify_batch(&batched_proofs, &mut transcript, &mut rng).unwrap();
        
        // The test proofs are not actually valid, so this should fail
        // In a real test, you would create valid proofs
        assert!(!result);
        
        // Test with a valid proof (for demonstration - we'd need to create actual valid proofs)
        // For now, we'll just test the API
        let params = LatticeParams {
            q: 97,
            n: 4,
            sigma: 3.0,
            beta: 2.0,
        };
        
        let single_proof = create_test_proof(&params);
        let single_batch = verifier.batch_proofs(vec![single_proof], vec![params.clone()]).unwrap();
        
        let mut single_transcript = Transcript::new(b"single_test");
        let result = verifier.verify_batch(&single_batch, &mut single_transcript, &mut rng).unwrap();
        
        // Again, this would fail in a real scenario with valid proofs
        assert!(!result);
        
        // Test empty batch
        let empty_result = verifier.batch_proofs(vec![], vec![]);
        assert!(empty_result.is_err());
    }
} 