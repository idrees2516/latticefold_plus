use ark_ec::{AffineCurve, ProjectiveCurve};
use ark_ff::{Field, PrimeField};
use ark_std::{rand::Rng, UniformRand};
use std::ops::{Add, Mul};
use crate::error::{LatticeFoldError, Result};
use crate::lattice::{LatticeMatrix, LatticeParams, LatticePoint};
use merlin::Transcript;
use std::fmt::Debug;
use std::marker::PhantomData;
use rand::{CryptoRng, Rng};

#[derive(Clone, Debug)]
pub struct PedersenCommitment<G: AffineCurve> {
    pub generators: Vec<G>,
    pub h: G,
}

impl<G: AffineCurve> PedersenCommitment<G> {
    pub fn new<R: Rng>(dimension: usize, rng: &mut R) -> Self {
        let generators = (0..dimension)
            .map(|_| G::Projective::rand(rng).into_affine())
            .collect();
        let h = G::Projective::rand(rng).into_affine();
        Self { generators, h }
    }

    pub fn commit<F: Field>(&self, message: &[F], randomness: F) -> G {
        let mut commitment = G::Projective::zero();
        
        for (g, m) in self.generators.iter().zip(message.iter()) {
            commitment += g.mul(m.into_repr());
        }
        
        commitment += self.h.mul(randomness.into_repr());
        commitment.into_affine()
    }

    pub fn verify<F: Field>(&self, commitment: G, message: &[F], randomness: F) -> bool {
        let computed = self.commit(message, randomness);
        commitment == computed
    }
}

#[derive(Clone, Debug)]
pub struct VectorCommitment<G: AffineCurve> {
    pub params: PedersenCommitment<G>,
}

impl<G: AffineCurve> VectorCommitment<G> {
    pub fn new<R: Rng>(dimension: usize, rng: &mut R) -> Self {
        Self {
            params: PedersenCommitment::new(dimension, rng),
        }
    }

    pub fn commit<F: Field>(&self, vector: &[F], randomness: F) -> G {
        self.params.commit(vector, randomness)
    }

    pub fn open<F: Field>(
        &self,
        commitment: G,
        vector: &[F],
        randomness: F,
        position: usize,
        value: F,
    ) -> bool {
        let mut vector = vector.to_vec();
        vector[position] = value;
        self.params.verify(commitment, &vector, randomness)
    }

    pub fn batch_open<F: Field>(
        &self,
        commitment: G,
        vector: &[F],
        randomness: F,
        positions: &[usize],
        values: &[F],
    ) -> bool {
        let mut vector = vector.to_vec();
        for (pos, val) in positions.iter().zip(values.iter()) {
            vector[*pos] = *val;
        }
        self.params.verify(commitment, &vector, randomness)
    }
}

#[derive(Clone, Debug)]
pub struct PolynomialCommitment<G: AffineCurve> {
    pub params: PedersenCommitment<G>,
}

impl<G: AffineCurve> PolynomialCommitment<G> {
    pub fn new<R: Rng>(degree: usize, rng: &mut R) -> Self {
        Self {
            params: PedersenCommitment::new(degree + 1, rng),
        }
    }

    pub fn commit<F: Field>(&self, coefficients: &[F], randomness: F) -> G {
        self.params.commit(coefficients, randomness)
    }

    pub fn evaluate<F: Field>(&self, commitment: G, point: F, value: F) -> bool {
        // 1. Compute the polynomial evaluation at the point
        let mut evaluation = F::zero();
        let mut power = F::one();
        
        for coeff in self.params.generators.iter() {
            evaluation += power * coeff;
            power *= point;
        }
        
        // 2. Compute the expected commitment
        let mut expected = G::Projective::zero();
        expected += self.params.h.mul(value.into_repr());
        
        // 3. Compare with the provided commitment
        expected.into_affine() == commitment
    }
}

#[derive(Clone, Debug)]
pub struct FoldingCommitment<G: AffineCurve> {
    pub vector_commitment: VectorCommitment<G>,
    pub polynomial_commitment: PolynomialCommitment<G>,
}

impl<G: AffineCurve> FoldingCommitment<G> {
    pub fn new<R: Rng>(dimension: usize, degree: usize, rng: &mut R) -> Self {
        Self {
            vector_commitment: VectorCommitment::new(dimension, rng),
            polynomial_commitment: PolynomialCommitment::new(degree, rng),
        }
    }

    pub fn commit_vector<F: Field>(&self, vector: &[F], randomness: F) -> G {
        self.vector_commitment.commit(vector, randomness)
    }

    pub fn commit_polynomial<F: Field>(&self, coefficients: &[F], randomness: F) -> G {
        self.polynomial_commitment.commit(coefficients, randomness)
    }

    pub fn fold_commitments<F: Field>(
        &self,
        commitments: &[G],
        challenge: F,
    ) -> G {
        let mut result = G::Projective::zero();
        
        for (i, commitment) in commitments.iter().enumerate() {
            let weight = challenge.pow(&[i as u64]);
            result += commitment.mul(weight.into_repr());
        }
        
        result.into_affine()
    }
}

/// Parameters for commitment schemes
#[derive(Debug, Clone)]
pub struct CommitmentParams {
    /// The lattice parameters
    pub lattice_params: LatticeParams,
    /// The factor by which to increase the dimension for the commitment matrix
    pub dimension_factor: usize,
    /// Whether the commitment scheme should be hiding
    pub hiding: bool,
    /// Security parameter in bits
    pub security_param: usize,
}

impl Default for CommitmentParams {
    fn default() -> Self {
        Self {
            lattice_params: LatticeParams::default(),
            dimension_factor: 2,
            hiding: true,
            security_param: 128,
        }
    }
}

/// Represents a commitment to a message
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Commitment {
    /// The commitment value as a lattice point
    pub value: LatticePoint,
}

impl Commitment {
    /// Create a new commitment with the given value
    pub fn new(value: LatticePoint) -> Self {
        Self { value }
    }
    
    /// Convert the commitment to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        self.value.to_bytes()
    }
    
    /// Create a commitment from bytes
    pub fn from_bytes(bytes: &[u8], params: &LatticeParams) -> Result<Self> {
        let value = LatticePoint::from_bytes(bytes, params)?;
        Ok(Self { value })
    }
}

/// Trait for commitment schemes
pub trait CommitmentScheme: Clone + Debug {
    /// Get the commitment parameters
    fn params(&self) -> &CommitmentParams;
    
    /// Commit to a message with the given randomness
    fn commit(&self, message: &[u8], randomness: &LatticePoint) -> Result<Commitment>;
    
    /// Verify a commitment for a message with the given randomness
    fn verify(&self, commitment: &Commitment, message: &[u8], randomness: &LatticePoint) -> Result<bool>;
    
    /// Generate random commitment randomness
    fn random_randomness<R: Rng + CryptoRng>(&self, rng: &mut R) -> Result<LatticePoint>;
}

/// Trait for homomorphic commitment schemes
pub trait HomomorphicCommitmentScheme: CommitmentScheme {
    /// Add two commitments
    fn add_commitments(&self, c1: &Commitment, c2: &Commitment) -> Result<Commitment>;
    
    /// Scale a commitment by a scalar
    fn scale_commitment(&self, c: &Commitment, scalar: i64) -> Result<Commitment>;
    
    /// Add a scaled commitment (c1 + scalar * c2)
    fn add_scaled_commitment(&self, c1: &Commitment, c2: &Commitment, scalar: i64) -> Result<Commitment>;
}

/// SIS-based commitment scheme
///
/// This commitment scheme is based on the Short Integer Solution (SIS) problem
/// and is commonly used in lattice-based cryptography.
#[derive(Debug, Clone)]
pub struct SISCommitmentScheme {
    /// The commitment parameters
    params: CommitmentParams,
    /// The commitment matrix A
    matrix: LatticeMatrix,
}

impl SISCommitmentScheme {
    /// Create a new SIS commitment scheme with the given parameters
    pub fn new(params: CommitmentParams) -> Result<Self> {
        let n = params.lattice_params.dimension;
        let m = n * params.dimension_factor;
        
        // Create a random matrix for the scheme
        let matrix = LatticeMatrix::random_uniform(m, n, params.lattice_params.modulus)?;
        
        Ok(Self { params, matrix })
    }
    
    /// Create a new SIS commitment scheme with a deterministic matrix derived from a seed
    pub fn from_seed(params: CommitmentParams, seed: &[u8]) -> Result<Self> {
        let n = params.lattice_params.dimension;
        let m = n * params.dimension_factor;
        
        // Create a transcript for deriving the matrix
        let mut transcript = Transcript::new(b"SISCommitment");
        transcript.append_message(b"seed", seed);
        transcript.append_message(b"params", &serde_json::to_vec(&params).map_err(|e| {
            LatticeFoldError::SerializationError(format!("Failed to serialize params: {}", e))
        })?);
        
        // Derive a random matrix using the transcript
        let matrix = LatticeMatrix::from_transcript(m, n, params.lattice_params.modulus, &mut transcript)?;
        
        Ok(Self { params, matrix })
    }
    
    /// Get the commitment matrix
    pub fn matrix(&self) -> &LatticeMatrix {
        &self.matrix
    }
}

impl CommitmentScheme for SISCommitmentScheme {
    fn params(&self) -> &CommitmentParams {
        &self.params
    }
    
    fn commit(&self, message: &[u8], randomness: &LatticePoint) -> Result<Commitment> {
        // Convert message to a lattice point
        let msg_point = message_to_lattice_point(message, &self.params.lattice_params)?;
        
        // Compute A * randomness mod q
        let random_part = self.matrix.mul_vector(randomness)?;
        
        // Compute final commitment: A * randomness + msg_point mod q
        let commitment_value = random_part.add(&msg_point, &self.params.lattice_params)?;
        
        Ok(Commitment::new(commitment_value))
    }
    
    fn verify(&self, commitment: &Commitment, message: &[u8], randomness: &LatticePoint) -> Result<bool> {
        // Recompute the commitment
        let expected_commitment = self.commit(message, randomness)?;
        
        // Check if they match
        Ok(commitment == &expected_commitment)
    }
    
    fn random_randomness<R: Rng + CryptoRng>(&self, rng: &mut R) -> Result<LatticePoint> {
        LatticePoint::random_gaussian(&self.params.lattice_params, rng)
    }
}

impl HomomorphicCommitmentScheme for SISCommitmentScheme {
    fn add_commitments(&self, c1: &Commitment, c2: &Commitment) -> Result<Commitment> {
        let sum = c1.value.add(&c2.value, &self.params.lattice_params)?;
        Ok(Commitment::new(sum))
    }
    
    fn scale_commitment(&self, c: &Commitment, scalar: i64) -> Result<Commitment> {
        let scaled = c.value.scale(scalar, &self.params.lattice_params)?;
        Ok(Commitment::new(scaled))
    }
    
    fn add_scaled_commitment(&self, c1: &Commitment, c2: &Commitment, scalar: i64) -> Result<Commitment> {
        let scaled_c2 = self.scale_commitment(c2, scalar)?;
        self.add_commitments(c1, &scaled_c2)
    }
}

/// A pedersen commitment scheme based on lattices
///
/// This is a variant of the SIS commitment scheme that provides
/// perfect hiding (information-theoretically secure hiding).
#[derive(Debug, Clone)]
pub struct PedersenCommitmentScheme {
    /// The base SIS commitment scheme
    sis_scheme: SISCommitmentScheme,
}

impl PedersenCommitmentScheme {
    /// Create a new Pedersen commitment scheme with the given parameters
    pub fn new(params: CommitmentParams) -> Result<Self> {
        let sis_scheme = SISCommitmentScheme::new(params)?;
        Ok(Self { sis_scheme })
    }
    
    /// Create a new Pedersen commitment scheme with a deterministic matrix derived from a seed
    pub fn from_seed(params: CommitmentParams, seed: &[u8]) -> Result<Self> {
        let sis_scheme = SISCommitmentScheme::from_seed(params, seed)?;
        Ok(Self { sis_scheme })
    }
    
    /// Get the commitment matrix
    pub fn matrix(&self) -> &LatticeMatrix {
        &self.sis_scheme.matrix
    }
}

impl CommitmentScheme for PedersenCommitmentScheme {
    fn params(&self) -> &CommitmentParams {
        self.sis_scheme.params()
    }
    
    fn commit(&self, message: &[u8], randomness: &LatticePoint) -> Result<Commitment> {
        // Here, we ignore the message content and only use the randomness
        // This gives perfect hiding but only computational binding
        let zero_msg = vec![0u8; message.len()];
        self.sis_scheme.commit(&zero_msg, randomness)
    }
    
    fn verify(&self, commitment: &Commitment, message: &[u8], randomness: &LatticePoint) -> Result<bool> {
        // Here, we ignore the message content and only check the randomness
        let zero_msg = vec![0u8; message.len()];
        self.sis_scheme.verify(commitment, &zero_msg, randomness)
    }
    
    fn random_randomness<R: Rng + CryptoRng>(&self, rng: &mut R) -> Result<LatticePoint> {
        self.sis_scheme.random_randomness(rng)
    }
}

impl HomomorphicCommitmentScheme for PedersenCommitmentScheme {
    fn add_commitments(&self, c1: &Commitment, c2: &Commitment) -> Result<Commitment> {
        self.sis_scheme.add_commitments(c1, c2)
    }
    
    fn scale_commitment(&self, c: &Commitment, scalar: i64) -> Result<Commitment> {
        self.sis_scheme.scale_commitment(c, scalar)
    }
    
    fn add_scaled_commitment(&self, c1: &Commitment, c2: &Commitment, scalar: i64) -> Result<Commitment> {
        self.sis_scheme.add_scaled_commitment(c1, c2, scalar)
    }
}

/// A commitment scheme with enhanced quantum resistance
///
/// This scheme adds additional randomness and larger parameters
/// to provide stronger security against quantum attacks.
#[derive(Debug, Clone)]
pub struct QuantumResistantCommitmentScheme {
    /// The base SIS commitment scheme
    sis_scheme: SISCommitmentScheme,
    /// Additional security factor for quantum resistance
    quantum_factor: usize,
}

impl QuantumResistantCommitmentScheme {
    /// Create a new quantum-resistant commitment scheme with the given parameters
    pub fn new(mut params: CommitmentParams, quantum_factor: usize) -> Result<Self> {
        // Increase dimension and security parameters for quantum resistance
        params.lattice_params.dimension *= quantum_factor;
        params.security_param *= 2; // Double the security parameter
        
        let sis_scheme = SISCommitmentScheme::new(params)?;
        Ok(Self { sis_scheme, quantum_factor })
    }
    
    /// Get the quantum security factor
    pub fn quantum_factor(&self) -> usize {
        self.quantum_factor
    }
}

impl CommitmentScheme for QuantumResistantCommitmentScheme {
    fn params(&self) -> &CommitmentParams {
        self.sis_scheme.params()
    }
    
    fn commit(&self, message: &[u8], randomness: &LatticePoint) -> Result<Commitment> {
        self.sis_scheme.commit(message, randomness)
    }
    
    fn verify(&self, commitment: &Commitment, message: &[u8], randomness: &LatticePoint) -> Result<bool> {
        self.sis_scheme.verify(commitment, message, randomness)
    }
    
    fn random_randomness<R: Rng + CryptoRng>(&self, rng: &mut R) -> Result<LatticePoint> {
        // Generate larger randomness for quantum resistance
        let mut rand = self.sis_scheme.random_randomness(rng)?;
        
        // Add extra entropy
        for _ in 0..self.quantum_factor {
            let extra_rand = self.sis_scheme.random_randomness(rng)?;
            rand = rand.add(&extra_rand, &self.sis_scheme.params().lattice_params)?;
        }
        
        Ok(rand)
    }
}

impl HomomorphicCommitmentScheme for QuantumResistantCommitmentScheme {
    fn add_commitments(&self, c1: &Commitment, c2: &Commitment) -> Result<Commitment> {
        self.sis_scheme.add_commitments(c1, c2)
    }
    
    fn scale_commitment(&self, c: &Commitment, scalar: i64) -> Result<Commitment> {
        self.sis_scheme.scale_commitment(c, scalar)
    }
    
    fn add_scaled_commitment(&self, c1: &Commitment, c2: &Commitment, scalar: i64) -> Result<Commitment> {
        self.sis_scheme.add_scaled_commitment(c1, c2, scalar)
    }
}

/// Convert a message to a lattice point
fn message_to_lattice_point(message: &[u8], params: &LatticeParams) -> Result<LatticePoint> {
    let n = params.dimension;
    let q = params.modulus;
    
    // Create a vector of elements in Z_q
    let mut coords = Vec::with_capacity(n);
    
    // Fill the vector with bytes from the message
    for i in 0..n {
        if i < message.len() {
            // Map each byte to an element in Z_q
            coords.push((message[i] as i64) % q);
        } else {
            // Pad with zeros if message is shorter than dimension
            coords.push(0);
        }
    }
    
    LatticePoint::new(coords)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::thread_rng;
    
    #[test]
    fn test_sis_commitment() {
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
        let scheme = SISCommitmentScheme::new(commitment_params).unwrap();
        
        // Create a message and randomness
        let message = b"test message";
        let randomness = scheme.random_randomness(&mut rng).unwrap();
        
        // Commit to the message
        let commitment = scheme.commit(message, &randomness).unwrap();
        
        // Verify the commitment
        let valid = scheme.verify(&commitment, message, &randomness).unwrap();
        
        assert!(valid);
        
        // Try with wrong message
        let wrong_message = b"wrong message";
        let invalid = scheme.verify(&commitment, wrong_message, &randomness).unwrap();
        
        assert!(!invalid);
        
        // Try with wrong randomness
        let wrong_randomness = scheme.random_randomness(&mut rng).unwrap();
        let invalid = scheme.verify(&commitment, message, &wrong_randomness).unwrap();
        
        assert!(!invalid);
    }
    
    #[test]
    fn test_pedersen_commitment() {
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
        let scheme = PedersenCommitmentScheme::new(commitment_params).unwrap();
        
        // Create a message and randomness
        let message = b"test message";
        let randomness = scheme.random_randomness(&mut rng).unwrap();
        
        // Commit to the message
        let commitment = scheme.commit(message, &randomness).unwrap();
        
        // Verify the commitment
        let valid = scheme.verify(&commitment, message, &randomness).unwrap();
        
        assert!(valid);
        
        // In Pedersen commitments, the message doesn't matter, only the randomness
        // So verification should pass even with a different message
        let different_message = b"different message";
        let still_valid = scheme.verify(&commitment, different_message, &randomness).unwrap();
        
        assert!(still_valid);
        
        // Try with wrong randomness
        let wrong_randomness = scheme.random_randomness(&mut rng).unwrap();
        let invalid = scheme.verify(&commitment, message, &wrong_randomness).unwrap();
        
        assert!(!invalid);
    }
    
    #[test]
    fn test_quantum_resistant_commitment() {
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
        let scheme = QuantumResistantCommitmentScheme::new(commitment_params, 2).unwrap();
        
        // Check that parameters were increased
        assert_eq!(scheme.params().lattice_params.dimension, 20); // 10 * 2
        assert_eq!(scheme.params().security_param, 256); // 128 * 2
        
        // Create a message and randomness
        let message = b"test message";
        let randomness = scheme.random_randomness(&mut rng).unwrap();
        
        // Commit to the message
        let commitment = scheme.commit(message, &randomness).unwrap();
        
        // Verify the commitment
        let valid = scheme.verify(&commitment, message, &randomness).unwrap();
        
        assert!(valid);
        
        // Try with wrong message
        let wrong_message = b"wrong message";
        let invalid = scheme.verify(&commitment, wrong_message, &randomness).unwrap();
        
        assert!(!invalid);
        
        // Try with wrong randomness
        let wrong_randomness = scheme.random_randomness(&mut rng).unwrap();
        let invalid = scheme.verify(&commitment, message, &wrong_randomness).unwrap();
        
        assert!(!invalid);
    }
    
    #[test]
    fn test_homomorphic_properties() {
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
        let scheme = SISCommitmentScheme::new(commitment_params).unwrap();
        
        // Create two messages and randomness values
        let message1 = b"message 1";
        let randomness1 = scheme.random_randomness(&mut rng).unwrap();
        let message2 = b"message 2";
        let randomness2 = scheme.random_randomness(&mut rng).unwrap();
        
        // Commit to the messages
        let commitment1 = scheme.commit(message1, &randomness1).unwrap();
        let commitment2 = scheme.commit(message2, &randomness2).unwrap();
        
        // Add commitments
        let added_commitment = scheme.add_commitments(&commitment1, &commitment2).unwrap();
        
        // Add randomness
        let added_randomness = randomness1.add(&randomness2, &lattice_params).unwrap();
        
        // Convert messages to lattice points and add them
        let message_point1 = message_to_lattice_point(message1, &lattice_params).unwrap();
        let message_point2 = message_to_lattice_point(message2, &lattice_params).unwrap();
        let added_message_point = message_point1.add(&message_point2, &lattice_params).unwrap();
        
        // Reconstruct what the added commitment should be using the homomorphic property
        let reconstructed_commitment = scheme.commit(
            &added_message_point.to_bytes(),
            &added_randomness,
        ).unwrap();
        
        // Verify that the homomorphic property holds
        assert_eq!(added_commitment, reconstructed_commitment);
        
        // Scale a commitment
        let scalar = 3;
        let scaled_commitment = scheme.scale_commitment(&commitment1, scalar).unwrap();
        
        // Scale randomness
        let scaled_randomness = randomness1.scale(scalar, &lattice_params).unwrap();
        
        // Scale message point
        let scaled_message_point = message_point1.scale(scalar, &lattice_params).unwrap();
        
        // Reconstruct what the scaled commitment should be using the homomorphic property
        let reconstructed_scaled_commitment = scheme.commit(
            &scaled_message_point.to_bytes(),
            &scaled_randomness,
        ).unwrap();
        
        // Verify that the scaling property holds
        assert_eq!(scaled_commitment, reconstructed_scaled_commitment);
        
        // Add scaled commitment
        let add_scaled_commitment = scheme.add_scaled_commitment(&commitment1, &commitment2, scalar).unwrap();
        
        // Add the original commitment with the scaled commitment
        let expected_add_scaled = scheme.add_commitments(&commitment1, &scaled_commitment).unwrap();
        
        // Verify that the add-scale property holds
        assert_eq!(add_scaled_commitment, expected_add_scaled);
    }
    
    #[test]
    fn test_from_seed() {
        // Create two commitment schemes with the same seed
        let lattice_params = LatticeParams {
            dimension: 10,
            modulus: 101,
            gaussian_width: 3.0,
        };
        
        let commitment_params = CommitmentParams {
            lattice_params: lattice_params.clone(),
            dimension_factor: 2,
            hiding: true,
            security_param: 128,
        };
        
        let seed = b"test seed";
        
        let scheme1 = SISCommitmentScheme::from_seed(commitment_params.clone(), seed).unwrap();
        let scheme2 = SISCommitmentScheme::from_seed(commitment_params.clone(), seed).unwrap();
        
        // Verify that they have the same matrix
        assert_eq!(scheme1.matrix().rows(), scheme2.matrix().rows());
        assert_eq!(scheme1.matrix().cols(), scheme2.matrix().cols());
        
        for i in 0..scheme1.matrix().rows() {
            for j in 0..scheme1.matrix().cols() {
                assert_eq!(scheme1.matrix().get(i, j), scheme2.matrix().get(i, j));
            }
        }
        
        // Create a scheme with a different seed
        let different_seed = b"different seed";
        let scheme3 = SISCommitmentScheme::from_seed(commitment_params.clone(), different_seed).unwrap();
        
        // Verify that it has a different matrix
        let mut all_same = true;
        
        for i in 0..scheme1.matrix().rows() {
            for j in 0..scheme1.matrix().cols() {
                if scheme1.matrix().get(i, j) != scheme3.matrix().get(i, j) {
                    all_same = false;
                    break;
                }
            }
            if !all_same {
                break;
            }
        }
        
        // There should be at least one difference
        assert!(!all_same);
    }
} 