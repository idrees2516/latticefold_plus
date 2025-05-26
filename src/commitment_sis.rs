use crate::error::{LatticeFoldError, Result};
use crate::lattice::{LatticeMatrix, LatticeParams, LatticePoint};
use ark_ff::{Field, PrimeField};
use ark_std::{rand::Rng, UniformRand};
use merlin::Transcript;
use rand::{CryptoRng, RngCore};
use std::marker::PhantomData;
use zeroize::Zeroize;

/// SIS-based commitment scheme based on the Short Integer Solution (SIS) problem.
/// As described in Section 3.2 of the LatticeFold+ paper.
#[derive(Clone, Debug)]
pub struct SISCommitment<F: Field> {
    /// The commitment matrix A
    pub matrix_a: LatticeMatrix,
    /// The blinding matrix B
    pub matrix_b: LatticeMatrix,
    /// The modulus q
    pub q: i64,
    /// The dimension of the lattice
    pub n: usize,
    /// The security parameter
    pub beta: f64,
    /// The norm bound for the SIS problem
    pub norm_bound: i64,
    /// Phantom data for the field type
    pub _phantom: PhantomData<F>,
}

/// A commitment with opening information
#[derive(Clone, Debug, Zeroize)]
pub struct SISCommitmentWithOpening<F: Field> {
    /// The commitment value
    pub commitment: Vec<i64>,
    /// The message that was committed
    #[zeroize(skip)]
    pub message: Vec<i64>,
    /// The randomness used
    #[zeroize(skip)]
    pub randomness: Vec<i64>,
    /// Phantom data for the field type
    pub _phantom: PhantomData<F>,
}

impl<F: Field> SISCommitment<F> {
    /// Create a new SIS-based commitment scheme with the given parameters
    pub fn new<R: Rng>(params: &LatticeParams, rng: &mut R) -> Self {
        // Generate the commitment matrix A
        let matrix_a = LatticeMatrix::random(params.n, 2 * params.n, params, rng);
        
        // Generate the blinding matrix B
        let matrix_b = LatticeMatrix::random(params.n, params.n, params, rng);
        
        // Calculate the norm bound based on the security parameter
        let norm_bound = (params.q as f64).powf(1.0 / params.beta) as i64;
        
        Self {
            matrix_a,
            matrix_b,
            q: params.q,
            n: params.n,
            beta: params.beta,
            norm_bound,
            _phantom: PhantomData,
        }
    }
    
    /// Commit to a message using the SIS-based commitment scheme
    pub fn commit<R: RngCore + CryptoRng>(
        &self,
        message: &[i64],
        rng: &mut R,
    ) -> Result<SISCommitmentWithOpening<F>> {
        if message.len() != self.n {
            return Err(LatticeFoldError::InvalidDimension {
                expected: self.n,
                got: message.len(),
            });
        }
        
        // Generate random values for the commitment
        let randomness: Vec<i64> = (0..self.n)
            .map(|_| {
                let val = rng.next_u64() as i64;
                val % self.q
            })
            .collect();
        
        // Compute the commitment: C = A * m + B * r (mod q)
        let message_vec = SISCommitment::<F>::pad_vector(message, 2 * self.n);
        let message_matrix = LatticeMatrix::new(vec![message_vec])?;
        
        let randomness_matrix = LatticeMatrix::new(vec![randomness.clone()])?;
        
        let am = self.matrix_a.mul(&message_matrix.transpose())?;
        let br = self.matrix_b.mul(&randomness_matrix.transpose())?;
        
        let mut commitment = Vec::with_capacity(self.n);
        for i in 0..self.n {
            let val = (am.data[i][0] + br.data[i][0]) % self.q;
            commitment.push((val + self.q) % self.q); // Ensure positive value
        }
        
        Ok(SISCommitmentWithOpening {
            commitment,
            message: message.to_vec(),
            randomness,
            _phantom: PhantomData,
        })
    }
    
    /// Verify a commitment opening
    pub fn verify(
        &self,
        commitment_with_opening: &SISCommitmentWithOpening<F>,
    ) -> Result<bool> {
        let message = &commitment_with_opening.message;
        let randomness = &commitment_with_opening.randomness;
        let commitment = &commitment_with_opening.commitment;
        
        if message.len() != self.n {
            return Err(LatticeFoldError::InvalidDimension {
                expected: self.n,
                got: message.len(),
            });
        }
        
        if randomness.len() != self.n {
            return Err(LatticeFoldError::InvalidDimension {
                expected: self.n,
                got: randomness.len(),
            });
        }
        
        if commitment.len() != self.n {
            return Err(LatticeFoldError::InvalidDimension {
                expected: self.n,
                got: commitment.len(),
            });
        }
        
        // Compute the expected commitment: C' = A * m + B * r (mod q)
        let message_vec = SISCommitment::<F>::pad_vector(message, 2 * self.n);
        let message_matrix = LatticeMatrix::new(vec![message_vec])?;
        
        let randomness_matrix = LatticeMatrix::new(vec![randomness.clone()])?;
        
        let am = self.matrix_a.mul(&message_matrix.transpose())?;
        let br = self.matrix_b.mul(&randomness_matrix.transpose())?;
        
        // Check if C == C'
        for i in 0..self.n {
            let expected = (am.data[i][0] + br.data[i][0]) % self.q;
            let expected = (expected + self.q) % self.q; // Ensure positive value
            if expected != commitment[i] {
                return Ok(false);
            }
        }
        
        Ok(true)
    }
    
    /// Batch verify multiple commitments with a single verification equation
    /// Implements the amortized verification technique from Section 4.3
    pub fn batch_verify<R: RngCore + CryptoRng>(
        &self,
        commitments_with_openings: &[SISCommitmentWithOpening<F>],
        rng: &mut R,
    ) -> Result<bool> {
        if commitments_with_openings.is_empty() {
            return Ok(true);
        }
        
        // Generate random weights for the linear combination
        let weights: Vec<i64> = (0..commitments_with_openings.len())
            .map(|_| {
                let val = rng.next_u64() as i64;
                val % self.q
            })
            .collect();
        
        // Compute the linear combination of messages and randomness
        let mut combined_message = vec![0i64; self.n];
        let mut combined_randomness = vec![0i64; self.n];
        let mut combined_commitment = vec![0i64; self.n];
        
        for (i, c) in commitments_with_openings.iter().enumerate() {
            for j in 0..self.n {
                combined_message[j] = (combined_message[j] + (weights[i] * c.message[j]) % self.q) % self.q;
                combined_randomness[j] = (combined_randomness[j] + (weights[i] * c.randomness[j]) % self.q) % self.q;
                combined_commitment[j] = (combined_commitment[j] + (weights[i] * c.commitment[j]) % self.q) % self.q;
            }
        }
        
        // Verify the combined commitment
        let combined = SISCommitmentWithOpening {
            commitment: combined_commitment,
            message: combined_message,
            randomness: combined_randomness,
            _phantom: PhantomData,
        };
        
        self.verify(&combined)
    }
    
    /// Create a commitment to zero that can be used for homomorphic operations
    pub fn commit_to_zero<R: RngCore + CryptoRng>(&self, rng: &mut R) -> Result<SISCommitmentWithOpening<F>> {
        let zeros = vec![0i64; self.n];
        self.commit(&zeros, rng)
    }
    
    /// Add two commitments homomorphically
    pub fn add(
        &self,
        a: &SISCommitmentWithOpening<F>,
        b: &SISCommitmentWithOpening<F>,
    ) -> Result<SISCommitmentWithOpening<F>> {
        if a.message.len() != self.n || b.message.len() != self.n {
            return Err(LatticeFoldError::InvalidDimension {
                expected: self.n,
                got: if a.message.len() != self.n { a.message.len() } else { b.message.len() },
            });
        }
        
        let mut commitment = Vec::with_capacity(self.n);
        let mut message = Vec::with_capacity(self.n);
        let mut randomness = Vec::with_capacity(self.n);
        
        for i in 0..self.n {
            commitment.push((a.commitment[i] + b.commitment[i]) % self.q);
            message.push((a.message[i] + b.message[i]) % self.q);
            randomness.push((a.randomness[i] + b.randomness[i]) % self.q);
        }
        
        Ok(SISCommitmentWithOpening {
            commitment,
            message,
            randomness,
            _phantom: PhantomData,
        })
    }
    
    /// Multiply a commitment by a scalar
    pub fn mul_scalar(
        &self,
        a: &SISCommitmentWithOpening<F>,
        scalar: i64,
    ) -> Result<SISCommitmentWithOpening<F>> {
        if a.message.len() != self.n {
            return Err(LatticeFoldError::InvalidDimension {
                expected: self.n,
                got: a.message.len(),
            });
        }
        
        let mut commitment = Vec::with_capacity(self.n);
        let mut message = Vec::with_capacity(self.n);
        let mut randomness = Vec::with_capacity(self.n);
        
        for i in 0..self.n {
            commitment.push((a.commitment[i] * scalar) % self.q);
            message.push((a.message[i] * scalar) % self.q);
            randomness.push((a.randomness[i] * scalar) % self.q);
        }
        
        Ok(SISCommitmentWithOpening {
            commitment,
            message,
            randomness,
            _phantom: PhantomData,
        })
    }
    
    /// Helper function to pad a vector to a specific length
    fn pad_vector(vec: &[i64], length: usize) -> Vec<i64> {
        let mut result = vec.to_vec();
        result.resize(length, 0);
        result
    }
}

/// Tests for the SIS-based commitment scheme
#[cfg(test)]
mod tests {
    use super::*;
    use rand::thread_rng;

    #[test]
    fn test_sis_commitment() {
        let params = LatticeParams {
            q: 97, // Small prime for testing
            n: 4,
            sigma: 3.0,
            beta: 2.0,
        };
        
        let mut rng = thread_rng();
        let sis = SISCommitment::<i64>::new(&params, &mut rng);
        
        // Test commitment and verification
        let message = vec![1, 2, 3, 4];
        let commitment = sis.commit(&message, &mut rng).unwrap();
        
        assert!(sis.verify(&commitment).unwrap());
        
        // Test commitment to different message
        let different_message = vec![5, 6, 7, 8];
        let mut invalid_commitment = commitment.clone();
        invalid_commitment.message = different_message;
        
        assert!(!sis.verify(&invalid_commitment).unwrap());
        
        // Test batch verification
        let message2 = vec![5, 6, 7, 8];
        let commitment2 = sis.commit(&message2, &mut rng).unwrap();
        
        let commitments = vec![commitment, commitment2];
        assert!(sis.batch_verify(&commitments, &mut rng).unwrap());
        
        // Test homomorphic properties
        let a = sis.commit(&vec![1, 2, 3, 4], &mut rng).unwrap();
        let b = sis.commit(&vec![5, 6, 7, 8], &mut rng).unwrap();
        
        // Test addition
        let c = sis.add(&a, &b).unwrap();
        assert!(sis.verify(&c).unwrap());
        
        // Expected message after addition
        let expected_message = vec![(1 + 5) % params.q, (2 + 6) % params.q, (3 + 7) % params.q, (4 + 8) % params.q];
        assert_eq!(c.message, expected_message);
        
        // Test scalar multiplication
        let scalar = 3;
        let d = sis.mul_scalar(&a, scalar).unwrap();
        assert!(sis.verify(&d).unwrap());
        
        // Expected message after scalar multiplication
        let expected_scaled = vec![(1 * scalar) % params.q, (2 * scalar) % params.q, (3 * scalar) % params.q, (4 * scalar) % params.q];
        assert_eq!(d.message, expected_scaled);
    }
} 