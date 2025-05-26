use crate::error::{LatticeFoldError, Result};
use merlin::Transcript;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use std::fmt::Debug;

/// Parameters for challenge generation
#[derive(Debug, Clone)]
pub struct ChallengeParams {
    /// The security parameter in bits
    pub security_param: usize,
    /// The challenge range (typically the lattice modulus q)
    pub challenge_range: i64,
    /// Whether to use structured challenges
    pub structured: bool,
    /// The dimension of challenges for structured mode
    pub dimension: usize,
}

impl Default for ChallengeParams {
    fn default() -> Self {
        Self {
            security_param: 128,
            challenge_range: 2053, // A prime often used as modulus
            structured: false,
            dimension: 32,
        }
    }
}

/// A type representing a single challenge value
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Challenge {
    /// The value of the challenge
    pub value: i64,
}

impl Challenge {
    /// Create a new challenge with a given value
    pub fn new(value: i64, modulus: i64) -> Self {
        Self {
            value: value.rem_euclid(modulus),
        }
    }
    
    /// Convert the challenge to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        self.value.to_le_bytes().to_vec()
    }
    
    /// Create a challenge from bytes
    pub fn from_bytes(bytes: &[u8], modulus: i64) -> Result<Self> {
        if bytes.len() < 8 {
            return Err(LatticeFoldError::InvalidInput(
                "Not enough bytes for challenge".to_string(),
            ));
        }
        
        let mut value_bytes = [0u8; 8];
        value_bytes.copy_from_slice(&bytes[..8]);
        let value = i64::from_le_bytes(value_bytes);
        
        Ok(Self::new(value, modulus))
    }
}

/// A vector of challenge values, used in structured challenge generation
#[derive(Debug, Clone)]
pub struct ChallengeVector {
    /// The individual challenge values
    pub values: Vec<i64>,
    /// The modulus for the challenges
    pub modulus: i64,
}

impl ChallengeVector {
    /// Create a new challenge vector with given values
    pub fn new(values: Vec<i64>, modulus: i64) -> Self {
        let normalized_values = values
            .into_iter()
            .map(|v| v.rem_euclid(modulus))
            .collect();
        
        Self {
            values: normalized_values,
            modulus,
        }
    }
    
    /// Get the dimension of the challenge vector
    pub fn dimension(&self) -> usize {
        self.values.len()
    }
    
    /// Convert the challenge vector to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(8 * self.values.len());
        for value in &self.values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }
    
    /// Create a challenge vector from bytes
    pub fn from_bytes(bytes: &[u8], dimension: usize, modulus: i64) -> Result<Self> {
        if bytes.len() < 8 * dimension {
            return Err(LatticeFoldError::InvalidInput(
                format!("Not enough bytes for challenge vector of dimension {}", dimension)
            ));
        }
        
        let mut values = Vec::with_capacity(dimension);
        
        for i in 0..dimension {
            let start = i * 8;
            let mut value_bytes = [0u8; 8];
            value_bytes.copy_from_slice(&bytes[start..start + 8]);
            let value = i64::from_le_bytes(value_bytes);
            values.push(value.rem_euclid(modulus));
        }
        
        Ok(Self {
            values,
            modulus,
        })
    }
    
    /// Generate a dot product with another vector of values
    pub fn dot_product(&self, other: &[i64]) -> Result<i64> {
        if self.values.len() != other.len() {
            return Err(LatticeFoldError::InvalidInput(
                format!(
                    "Dimension mismatch in dot product: {} vs {}",
                    self.values.len(),
                    other.len()
                )
            ));
        }
        
        let mut result = 0;
        for i in 0..self.values.len() {
            result = (result + self.values[i] * other[i]) % self.modulus;
        }
        
        Ok(result)
    }
}

/// An interface for generating cryptographic challenges
pub trait ChallengeGenerator {
    /// Get the challenge parameters
    fn params(&self) -> &ChallengeParams;
    
    /// Generate a single challenge
    fn generate_challenge(&mut self) -> Challenge;
    
    /// Generate a vector of challenges
    fn generate_challenge_vector(&mut self) -> ChallengeVector;
    
    /// Sample a structured challenge
    fn sample_structured_challenge(&mut self, label: &[u8]) -> ChallengeVector;
}

/// A challenge generator using the Merlin transcript system
pub struct TranscriptChallengeGenerator {
    /// Parameters for challenge generation
    params: ChallengeParams,
    /// The underlying transcript
    transcript: Transcript,
}

impl TranscriptChallengeGenerator {
    /// Create a new transcript-based challenge generator
    pub fn new(params: ChallengeParams, protocol_name: &[u8]) -> Self {
        Self {
            params,
            transcript: Transcript::new(protocol_name),
        }
    }
    
    /// Add a message to the transcript
    pub fn append_message(&mut self, label: &[u8], message: &[u8]) {
        self.transcript.append_message(label, message);
    }
    
    /// Generate a deterministic PRNG from the transcript
    fn get_transcript_rng(&mut self) -> ChaCha20Rng {
        let mut seed = [0u8; 32];
        self.transcript.challenge_bytes(b"rng_seed", &mut seed);
        ChaCha20Rng::from_seed(seed)
    }
}

impl ChallengeGenerator for TranscriptChallengeGenerator {
    fn params(&self) -> &ChallengeParams {
        &self.params
    }
    
    fn generate_challenge(&mut self) -> Challenge {
        let mut value_bytes = [0u8; 8];
        self.transcript.challenge_bytes(b"challenge", &mut value_bytes);
        
        let value = i64::from_le_bytes(value_bytes);
        let modulus = self.params.challenge_range;
        
        Challenge::new(value, modulus)
    }
    
    fn generate_challenge_vector(&mut self) -> ChallengeVector {
        let dimension = self.params.dimension;
        let modulus = self.params.challenge_range;
        
        let mut rng = self.get_transcript_rng();
        let values: Vec<i64> = (0..dimension)
            .map(|_| rng.gen_range(0..modulus))
            .collect();
        
        ChallengeVector::new(values, modulus)
    }
    
    fn sample_structured_challenge(&mut self, label: &[u8]) -> ChallengeVector {
        let dimension = self.params.dimension;
        let modulus = self.params.challenge_range;
        
        // Add the label to the transcript for domain separation
        self.transcript.append_message(b"structured_challenge_label", label);
        
        // For structured challenges, we generate a seed and then deterministically
        // expand it to the required number of challenge values
        let mut seed = [0u8; 32];
        self.transcript.challenge_bytes(b"structured_seed", &mut seed);
        
        // Use the seed to generate the challenges
        let mut rng = ChaCha20Rng::from_seed(seed);
        
        // In a structured challenge, values are not fully random but follow some pattern
        // For example, we could use a low-degree polynomial to define the sequence
        let mut values = Vec::with_capacity(dimension);
        
        // Generate first few values randomly
        let num_random = 3.min(dimension);
        for _ in 0..num_random {
            values.push(rng.gen_range(0..modulus));
        }
        
        // Fill in the rest using a simple recurrence relation (structured pattern)
        for i in num_random..dimension {
            // Simple linear recurrence: next = (3*prev - 2*prev2 + prev3) mod q
            let next = (3 * values[i - 1] - 2 * values[i - 2] + values[i - 3]) % modulus;
            values.push(next.rem_euclid(modulus));
        }
        
        ChallengeVector::new(values, modulus)
    }
}

/// A challenge generator optimized for fast verification
pub struct OptimizedChallengeGenerator {
    /// Parameters for challenge generation
    params: ChallengeParams,
    /// Seed for the pseudorandom number generator
    seed: [u8; 32],
    /// The current state/counter
    counter: u64,
}

impl OptimizedChallengeGenerator {
    /// Create a new optimized challenge generator with a random seed
    pub fn new<R: Rng>(params: ChallengeParams, rng: &mut R) -> Self {
        let mut seed = [0u8; 32];
        rng.fill(&mut seed);
        
        Self {
            params,
            seed,
            counter: 0,
        }
    }
    
    /// Create a new optimized challenge generator with a specific seed
    pub fn from_seed(params: ChallengeParams, seed: [u8; 32]) -> Self {
        Self {
            params,
            seed,
            counter: 0,
        }
    }
    
    /// Create a PRG from the current state
    fn get_prg(&self) -> ChaCha20Rng {
        // Combine seed and counter to get a unique PRNG for this challenge
        let mut hasher = blake3::Hasher::new();
        hasher.update(&self.seed);
        hasher.update(&self.counter.to_le_bytes());
        
        let hash = hasher.finalize();
        let mut seed = [0u8; 32];
        seed.copy_from_slice(hash.as_bytes());
        
        ChaCha20Rng::from_seed(seed)
    }
    
    /// Reset the counter to zero
    pub fn reset(&mut self) {
        self.counter = 0;
    }
}

impl ChallengeGenerator for OptimizedChallengeGenerator {
    fn params(&self) -> &ChallengeParams {
        &self.params
    }
    
    fn generate_challenge(&mut self) -> Challenge {
        let mut rng = self.get_prg();
        let value = rng.gen_range(0..self.params.challenge_range);
        
        // Increment counter for the next challenge
        self.counter += 1;
        
        Challenge::new(value, self.params.challenge_range)
    }
    
    fn generate_challenge_vector(&mut self) -> ChallengeVector {
        let dimension = self.params.dimension;
        let modulus = self.params.challenge_range;
        
        let mut rng = self.get_prg();
        let values: Vec<i64> = (0..dimension)
            .map(|_| rng.gen_range(0..modulus))
            .collect();
        
        // Increment counter for the next challenge vector
        self.counter += 1;
        
        ChallengeVector::new(values, modulus)
    }
    
    fn sample_structured_challenge(&mut self, label: &[u8]) -> ChallengeVector {
        let dimension = self.params.dimension;
        let modulus = self.params.challenge_range;
        
        // Create a domain-separated RNG based on the seed, counter, and label
        let mut hasher = blake3::Hasher::new();
        hasher.update(&self.seed);
        hasher.update(&self.counter.to_le_bytes());
        hasher.update(label);
        
        let hash = hasher.finalize();
        let mut seed = [0u8; 32];
        seed.copy_from_slice(hash.as_bytes());
        
        let mut rng = ChaCha20Rng::from_seed(seed);
        
        // In a structured challenge, values are generated deterministically
        let mut values = Vec::with_capacity(dimension);
        
        // Generate first few values randomly
        let num_random = 3.min(dimension);
        for _ in 0..num_random {
            values.push(rng.gen_range(0..modulus));
        }
        
        // Fill in the rest using a simple recurrence relation (structured pattern)
        for i in num_random..dimension {
            // Simple linear recurrence: next = (3*prev - 2*prev2 + prev3) mod q
            let next = (3 * values[i - 1] - 2 * values[i - 2] + values[i - 3]) % modulus;
            values.push(next.rem_euclid(modulus));
        }
        
        // Increment counter for the next challenge
        self.counter += 1;
        
        ChallengeVector::new(values, modulus)
    }
}

/// A strategy for combining multiple challenges
pub struct ChallengeCombiner {
    /// The modulus for all challenges
    modulus: i64,
}

impl ChallengeCombiner {
    /// Create a new challenge combiner
    pub fn new(modulus: i64) -> Self {
        Self { modulus }
    }
    
    /// Combine multiple challenges into a single challenge
    pub fn combine(&self, challenges: &[Challenge]) -> Challenge {
        if challenges.is_empty() {
            return Challenge::new(0, self.modulus);
        }
        
        // Simple polynomial combination: result = ∑ c_i * 2^(20*i)
        let mut result = 0;
        let base = 1 << 20; // 2^20, a large power to avoid collisions
        
        for (i, challenge) in challenges.iter().enumerate() {
            // For each challenge, multiply by a different power of the base
            let weight = base.pow(i as u32) % self.modulus;
            let term = (challenge.value * weight) % self.modulus;
            result = (result + term) % self.modulus;
        }
        
        Challenge::new(result, self.modulus)
    }
    
    /// Combine multiple challenge vectors using a weighted sum
    pub fn combine_vectors(&self, vectors: &[ChallengeVector]) -> Result<ChallengeVector> {
        if vectors.is_empty() {
            return Err(LatticeFoldError::InvalidInput(
                "Cannot combine empty list of challenge vectors".to_string(),
            ));
        }
        
        let dimension = vectors[0].dimension();
        
        // Ensure all vectors have the same dimension
        for vector in vectors {
            if vector.dimension() != dimension {
                return Err(LatticeFoldError::InvalidInput(
                    format!(
                        "Dimension mismatch in combining vectors: {} vs {}",
                        dimension,
                        vector.dimension()
                    )
                ));
            }
        }
        
        // Initialize result vector with zeros
        let mut result_values = vec![0; dimension];
        
        // Base for polynomial combination
        let base = 1 << 10; // 2^10, smaller than for single challenges to avoid overflow
        
        for (i, vector) in vectors.iter().enumerate() {
            // Weight for this vector
            let weight = base.pow(i as u32) % self.modulus;
            
            // Add weighted values from this vector
            for j in 0..dimension {
                let term = (vector.values[j] * weight) % self.modulus;
                result_values[j] = (result_values[j] + term) % self.modulus;
            }
        }
        
        // Normalize all values to be in [0, modulus)
        for value in &mut result_values {
            *value = value.rem_euclid(self.modulus);
        }
        
        Ok(ChallengeVector::new(result_values, self.modulus))
    }
    
    /// Hash a list of challenge vectors into a single challenge
    pub fn hash_vectors(&self, vectors: &[ChallengeVector]) -> Challenge {
        if vectors.is_empty() {
            return Challenge::new(0, self.modulus);
        }
        
        // Convert all vector bytes to a single byte array
        let mut all_bytes = Vec::new();
        for vector in vectors {
            all_bytes.extend_from_slice(&vector.to_bytes());
        }
        
        // Use BLAKE3 to hash the bytes
        let hash = blake3::hash(&all_bytes);
        
        // Convert first 8 bytes of hash to a challenge value
        let mut value_bytes = [0u8; 8];
        value_bytes.copy_from_slice(&hash.as_bytes()[..8]);
        let value = i64::from_le_bytes(value_bytes);
        
        Challenge::new(value, self.modulus)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::thread_rng;
    
    #[test]
    fn test_transcript_challenge_generator() {
        let params = ChallengeParams {
            security_param: 128,
            challenge_range: 101, // Small prime for testing
            structured: false,
            dimension: 4,
        };
        
        let mut generator = TranscriptChallengeGenerator::new(params, b"test_protocol");
        
        // Add some messages to the transcript
        generator.append_message(b"label1", b"message1");
        generator.append_message(b"label2", b"message2");
        
        // Generate a challenge
        let challenge = generator.generate_challenge();
        assert!(challenge.value >= 0 && challenge.value < 101);
        
        // Generate another challenge and ensure it's different
        generator.append_message(b"label3", b"message3");
        let challenge2 = generator.generate_challenge();
        assert!(challenge.value != challenge2.value);
        
        // Generate a challenge vector
        let vector = generator.generate_challenge_vector();
        assert_eq!(vector.dimension(), 4);
        for value in &vector.values {
            assert!(*value >= 0 && *value < 101);
        }
        
        // Generate a structured challenge
        let structured = generator.sample_structured_challenge(b"struct_test");
        assert_eq!(structured.dimension(), 4);
        
        // Test reproducibility - same transcript should give same challenges
        let mut generator2 = TranscriptChallengeGenerator::new(params, b"test_protocol");
        generator2.append_message(b"label1", b"message1");
        generator2.append_message(b"label2", b"message2");
        
        let challenge_repro = generator2.generate_challenge();
        assert_eq!(challenge.value, challenge_repro.value);
    }
    
    #[test]
    fn test_optimized_challenge_generator() {
        let mut rng = thread_rng();
        
        let params = ChallengeParams {
            security_param: 128,
            challenge_range: 101, // Small prime for testing
            structured: false,
            dimension: 4,
        };
        
        let mut generator = OptimizedChallengeGenerator::new(params, &mut rng);
        
        // Generate a challenge
        let challenge = generator.generate_challenge();
        assert!(challenge.value >= 0 && challenge.value < 101);
        
        // Generate another challenge and ensure it's different
        let challenge2 = generator.generate_challenge();
        assert!(challenge.value != challenge2.value);
        
        // Generate a challenge vector
        let vector = generator.generate_challenge_vector();
        assert_eq!(vector.dimension(), 4);
        for value in &vector.values {
            assert!(*value >= 0 && *value < 101);
        }
        
        // Test reset functionality
        let challenge3 = generator.generate_challenge();
        generator.reset();
        let challenge_after_reset = generator.generate_challenge();
        assert!(challenge3.value != challenge_after_reset.value);
        
        // Test reproducibility with fixed seed
        let seed = [42u8; 32];
        let mut generator1 = OptimizedChallengeGenerator::from_seed(params.clone(), seed);
        let mut generator2 = OptimizedChallengeGenerator::from_seed(params, seed);
        
        let c1 = generator1.generate_challenge();
        let c2 = generator2.generate_challenge();
        assert_eq!(c1.value, c2.value);
    }
    
    #[test]
    fn test_challenge_combiner() {
        let modulus = 101; // Small prime for testing
        let combiner = ChallengeCombiner::new(modulus);
        
        // Test combining single challenges
        let challenge1 = Challenge::new(10, modulus);
        let challenge2 = Challenge::new(20, modulus);
        let challenge3 = Challenge::new(30, modulus);
        
        let challenges = vec![challenge1, challenge2, challenge3];
        let combined = combiner.combine(&challenges);
        
        // Combined value should depend on all input challenges
        assert!(combined.value != 10 && combined.value != 20 && combined.value != 30);
        assert!(combined.value >= 0 && combined.value < modulus);
        
        // Test combining vectors
        let vector1 = ChallengeVector::new(vec![1, 2, 3, 4], modulus);
        let vector2 = ChallengeVector::new(vec![5, 6, 7, 8], modulus);
        
        let combined_vector = combiner.combine_vectors(&[vector1.clone(), vector2.clone()]).unwrap();
        assert_eq!(combined_vector.dimension(), 4);
        
        // Hash vectors to challenge
        let challenge = combiner.hash_vectors(&[vector1, vector2]);
        assert!(challenge.value >= 0 && challenge.value < modulus);
    }
    
    #[test]
    fn test_dot_product() {
        let modulus = 101;
        let challenge_vector = ChallengeVector::new(vec![1, 2, 3, 4], modulus);
        let other_vector = vec![5, 6, 7, 8];
        
        let dot_product = challenge_vector.dot_product(&other_vector).unwrap();
        // 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
        assert_eq!(dot_product, 70);
        
        // Test with overflow
        let challenge_vector = ChallengeVector::new(vec![10, 20, 30, 40], modulus);
        let other_vector = vec![10, 20, 30, 40];
        
        let dot_product = challenge_vector.dot_product(&other_vector).unwrap();
        // (10*10 + 20*20 + 30*30 + 40*40) mod 101 = (100 + 400 + 900 + 1600) mod 101 = 3000 mod 101 = 86
        assert_eq!(dot_product, 86);
    }
} 