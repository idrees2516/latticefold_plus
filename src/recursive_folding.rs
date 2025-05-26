use crate::error::{LatticeFoldError, Result};
use crate::folding::{FoldingCommitment, FoldingProof, FoldingScheme};
use crate::lattice::{LatticeMatrix, LatticeParams, LatticePoint};
use crate::commitment_sis::SISCommitment;
use ark_ff::Field;
use merlin::Transcript;
use rand::{CryptoRng, RngCore};
use std::collections::HashMap;
use std::marker::PhantomData;
use crate::challenge_generation::{Challenge, ChallengeGenerator};
use crate::proof::{LatticeProof, ProverState};
use std::sync::{Arc, Mutex};

/// A relation that can be proven using recursive folding
#[derive(Clone, Debug)]
pub struct RecursiveRelation<F: Field> {
    /// The relation identifier
    pub id: u64,
    /// The relation parameters
    pub params: LatticeParams,
    /// The relation matrix defining the relation R(x, w) where Ax = Bw
    pub matrix_a: LatticeMatrix,
    /// The relation matrix defining the relation R(x, w) where Ax = Bw
    pub matrix_b: LatticeMatrix,
    /// The phantom type parameter
    pub _phantom: PhantomData<F>,
}

/// A witness for a recursive relation
#[derive(Clone, Debug)]
pub struct RecursiveWitness<F: Field> {
    /// The public input x
    pub public_input: Vec<i64>,
    /// The witness w
    pub witness: Vec<i64>,
    /// The relation this witness satisfies
    pub relation_id: u64,
    /// The phantom type parameter
    pub _phantom: PhantomData<F>,
}

/// A proof for a recursive relation
#[derive(Clone, Debug)]
pub struct RecursiveProof<F: Field> {
    /// The relation this proof is for
    pub relation_id: u64,
    /// The folding proof
    pub folding_proof: FoldingProof,
    /// The commitment to the public input
    pub public_input_commitment: Vec<i64>,
    /// The commitment to the witness
    pub witness_commitment: Vec<i64>,
    /// The phantom type parameter
    pub _phantom: PhantomData<F>,
}

/// A recursive folding scheme that can fold proofs of different relations
#[derive(Clone, Debug)]
pub struct RecursiveFoldingScheme<F: Field> {
    /// The relations that can be folded
    pub relations: HashMap<u64, RecursiveRelation<F>>,
    /// The SIS commitment scheme
    pub commitment_scheme: SISCommitment<F>,
    /// The folding scheme for each relation
    pub folding_schemes: HashMap<u64, FoldingScheme>,
    /// The phantom type parameter
    pub _phantom: PhantomData<F>,
}

impl<F: Field> RecursiveFoldingScheme<F> {
    /// Create a new recursive folding scheme
    pub fn new<R: RngCore + CryptoRng>(params: &LatticeParams, rng: &mut R) -> Self {
        let commitment_scheme = SISCommitment::<F>::new(params, rng);
        
        Self {
            relations: HashMap::new(),
            commitment_scheme,
            folding_schemes: HashMap::new(),
            _phantom: PhantomData,
        }
    }
    
    /// Register a new relation
    pub fn register_relation<R: RngCore + CryptoRng>(
        &mut self,
        relation: RecursiveRelation<F>,
        rng: &mut R,
    ) -> Result<()> {
        let relation_id = relation.id;
        
        // Check if the relation already exists
        if self.relations.contains_key(&relation_id) {
            return Err(LatticeFoldError::InvalidParameters(
                format!("Relation with ID {} already exists", relation_id),
            ));
        }
        
        // Create a folding scheme for this relation
        let folding_scheme = FoldingScheme::new(relation.params.clone(), relation.matrix_a.clone());
        
        // Register the relation and folding scheme
        self.relations.insert(relation_id, relation);
        self.folding_schemes.insert(relation_id, folding_scheme);
        
        Ok(())
    }
    
    /// Prove satisfaction of a relation
    pub fn prove<R: RngCore + CryptoRng>(
        &mut self,
        witness: &RecursiveWitness<F>,
        transcript: &mut Transcript,
        rng: &mut R,
    ) -> Result<RecursiveProof<F>> {
        let relation_id = witness.relation_id;
        
        // Get the relation
        let relation = self.relations.get(&relation_id).ok_or_else(|| {
            LatticeFoldError::InvalidParameters(
                format!("Relation with ID {} does not exist", relation_id),
            )
        })?;
        
        // Verify that the witness satisfies the relation
        self.verify_relation(relation, &witness.public_input, &witness.witness)?;
        
        // Get the folding scheme
        let folding_scheme = self.folding_schemes.get_mut(&relation_id).ok_or_else(|| {
            LatticeFoldError::InvalidParameters(
                format!("Folding scheme for relation with ID {} does not exist", relation_id),
            )
        })?;
        
        // Create a lattice point from the witness
        let witness_point = LatticePoint::new(witness.witness.clone());
        
        // Create a proof using the folding scheme
        let folding_proof = folding_scheme.prove(&[witness_point], transcript, rng)?;
        
        // Commit to the public input and witness
        let public_input_commitment = self.commit_vector(&witness.public_input, rng)?;
        let witness_commitment = self.commit_vector(&witness.witness, rng)?;
        
        Ok(RecursiveProof {
            relation_id,
            folding_proof,
            public_input_commitment,
            witness_commitment,
            _phantom: PhantomData,
        })
    }
    
    /// Verify a recursive proof
    pub fn verify<R: RngCore + CryptoRng>(
        &mut self,
        proof: &RecursiveProof<F>,
        transcript: &mut Transcript,
        rng: &mut R,
    ) -> Result<bool> {
        let relation_id = proof.relation_id;
        
        // Get the relation
        let relation = self.relations.get(&relation_id).ok_or_else(|| {
            LatticeFoldError::InvalidParameters(
                format!("Relation with ID {} does not exist", relation_id),
            )
        })?;
        
        // Get the folding scheme
        let folding_scheme = self.folding_schemes.get_mut(&relation_id).ok_or_else(|| {
            LatticeFoldError::InvalidParameters(
                format!("Folding scheme for relation with ID {} does not exist", relation_id),
            )
        })?;
        
        // Verify the folding proof
        if !folding_scheme.verify(&proof.folding_proof, transcript)? {
            return Ok(false);
        }
        
        Ok(true)
    }
    
    /// Fold proofs of different relations into a single proof
    pub fn fold_proofs<R: RngCore + CryptoRng>(
        &mut self,
        proofs: &[RecursiveProof<F>],
        transcript: &mut Transcript,
        rng: &mut R,
    ) -> Result<RecursiveProof<F>> {
        if proofs.is_empty() {
            return Err(LatticeFoldError::InvalidParameters(
                "Cannot fold empty set of proofs".to_string(),
            ));
        }
        
        // Get the first proof as the base
        let base_proof = &proofs[0];
        let base_relation_id = base_proof.relation_id;
        
        // Get the base relation
        let base_relation = self.relations.get(&base_relation_id).ok_or_else(|| {
            LatticeFoldError::InvalidParameters(
                format!("Relation with ID {} does not exist", base_relation_id),
            )
        })?;
        
        // Create a new composite relation that combines all relations
        let composite_relation = self.create_composite_relation(proofs, rng)?;
        
        // Register the composite relation
        let composite_id = composite_relation.id;
        self.register_relation(composite_relation, rng)?;
        
        // Combine the folding proofs
        let mut folded_public_input = Vec::new();
        let mut folded_witness = Vec::new();
        
        for proof in proofs {
            folded_public_input.extend(proof.public_input_commitment.clone());
            folded_witness.extend(proof.witness_commitment.clone());
        }
        
        // Create a lattice point from the folded witness
        let folded_witness_point = LatticePoint::new(folded_witness.clone());
        
        // Get the folding scheme for the composite relation
        let folding_scheme = self.folding_schemes.get_mut(&composite_id).ok_or_else(|| {
            LatticeFoldError::InvalidParameters(
                format!("Folding scheme for relation with ID {} does not exist", composite_id),
            )
        })?;
        
        // Create a proof using the folding scheme
        let folding_proof = folding_scheme.prove(&[folded_witness_point], transcript, rng)?;
        
        Ok(RecursiveProof {
            relation_id: composite_id,
            folding_proof,
            public_input_commitment: folded_public_input,
            witness_commitment: folded_witness,
            _phantom: PhantomData,
        })
    }
    
    /// Create a composite relation that combines multiple relations
    fn create_composite_relation<R: RngCore + CryptoRng>(
        &self,
        proofs: &[RecursiveProof<F>],
        rng: &mut R,
    ) -> Result<RecursiveRelation<F>> {
        // Use cryptographic hash of all relation IDs as the new ID
        let mut hasher = blake3::Hasher::new();
        for proof in proofs {
            hasher.update(&proof.relation_id.to_le_bytes());
        }
        let hash = hasher.finalize();
        let composite_id = u64::from_le_bytes(hash.as_bytes()[0..8].try_into().unwrap());
        
        // Get the first relation as the base
        let base_relation_id = proofs[0].relation_id;
        let base_relation = self.relations.get(&base_relation_id).ok_or_else(|| {
            LatticeFoldError::InvalidParameters(
                format!("Relation with ID {} does not exist", base_relation_id),
            )
        })?;
        
        // Calculate the dimensions of the composite relation
        let mut total_rows = 0;
        let mut total_cols_a = 0;
        let mut total_cols_b = 0;
        
        for proof in proofs {
            let relation = self.relations.get(&proof.relation_id).ok_or_else(|| {
                LatticeFoldError::InvalidParameters(
                    format!("Relation with ID {} does not exist", proof.relation_id),
                )
            })?;
            
            total_rows += relation.matrix_a.rows;
            total_cols_a += relation.matrix_a.cols;
            total_cols_b += relation.matrix_b.cols;
        }
        
        // Create the composite matrices
        let mut matrix_a_data = vec![vec![0i64; total_cols_a]; total_rows];
        let mut matrix_b_data = vec![vec![0i64; total_cols_b]; total_rows];
        
        let mut row_offset = 0;
        let mut col_offset_a = 0;
        let mut col_offset_b = 0;
        
        for proof in proofs {
            let relation = self.relations.get(&proof.relation_id).ok_or_else(|| {
                LatticeFoldError::InvalidParameters(
                    format!("Relation with ID {} does not exist", proof.relation_id),
                )
            })?;
            
            // Copy the matrices into the composite matrices
            for i in 0..relation.matrix_a.rows {
                for j in 0..relation.matrix_a.cols {
                    matrix_a_data[row_offset + i][col_offset_a + j] = relation.matrix_a.data[i][j];
                }
                
                for j in 0..relation.matrix_b.cols {
                    matrix_b_data[row_offset + i][col_offset_b + j] = relation.matrix_b.data[i][j];
                }
            }
            
            row_offset += relation.matrix_a.rows;
            col_offset_a += relation.matrix_a.cols;
            col_offset_b += relation.matrix_b.cols;
        }
        
        // Create the composite matrices
        let matrix_a = LatticeMatrix::new(matrix_a_data)?;
        let matrix_b = LatticeMatrix::new(matrix_b_data)?;
        
        // Create the composite relation
        Ok(RecursiveRelation {
            id: composite_id,
            params: base_relation.params.clone(),
            matrix_a,
            matrix_b,
            _phantom: PhantomData,
        })
    }
    
    /// Verify that a witness satisfies a relation
    fn verify_relation(
        &self,
        relation: &RecursiveRelation<F>,
        public_input: &[i64],
        witness: &[i64],
    ) -> Result<bool> {
        if public_input.len() != relation.matrix_a.cols {
            return Err(LatticeFoldError::InvalidDimension {
                expected: relation.matrix_a.cols,
                got: public_input.len(),
            });
        }
        
        if witness.len() != relation.matrix_b.cols {
            return Err(LatticeFoldError::InvalidDimension {
                expected: relation.matrix_b.cols,
                got: witness.len(),
            });
        }
        
        // Compute Ax
        let public_input_matrix = LatticeMatrix::new(vec![public_input.to_vec()])?;
        let ax = relation.matrix_a.mul(&public_input_matrix.transpose())?;
        
        // Compute Bw
        let witness_matrix = LatticeMatrix::new(vec![witness.to_vec()])?;
        let bw = relation.matrix_b.mul(&witness_matrix.transpose())?;
        
        // Check Ax = Bw
        for i in 0..relation.matrix_a.rows {
            if ax.data[i][0] != bw.data[i][0] {
                return Ok(false);
            }
        }
        
        Ok(true)
    }
    
    /// Commit to a vector using the SIS commitment scheme
    fn commit_vector<R: RngCore + CryptoRng>(
        &self,
        vector: &[i64],
        rng: &mut R,
    ) -> Result<Vec<i64>> {
        let commitment = self.commitment_scheme.commit(vector, rng)?;
        Ok(commitment.commitment)
    }
}

/// Parameters for the recursive folding protocol
#[derive(Clone, Debug)]
pub struct RecursiveFoldingParams {
    /// The lattice parameters
    pub lattice_params: LatticeParams,
    /// The number of folding operations to perform
    pub num_folds: usize,
    /// The depth of the recursion tree
    pub recursion_depth: usize,
    /// Whether to use parallel computation
    pub parallel: bool,
    /// Whether to use memoization
    pub memoize: bool,
    /// The security parameter (bits)
    pub security_param: usize,
}

impl Default for RecursiveFoldingParams {
    fn default() -> Self {
        Self {
            lattice_params: LatticeParams::default(),
            num_folds: 4,
            recursion_depth: 2,
            parallel: true,
            memoize: true,
            security_param: 128,
        }
    }
}

/// A fold operation that maps multiple proofs to a single proof
#[derive(Clone, Debug)]
pub struct FoldOperation<F> {
    /// The number of proofs to fold
    pub arity: usize,
    /// The matrices used in the fold
    pub matrices: Vec<LatticeMatrix>,
    /// The challenges used in the fold
    pub challenges: Vec<Challenge>,
    /// Phantom data for the field type
    pub _phantom: PhantomData<F>,
}

impl<F> FoldOperation<F> {
    /// Create a new fold operation with the given arity
    pub fn new(
        arity: usize,
        matrices: Vec<LatticeMatrix>,
        challenges: Vec<Challenge>,
    ) -> Self {
        Self {
            arity,
            matrices,
            challenges,
            _phantom: PhantomData,
        }
    }
    
    /// Apply the fold operation to a set of proofs
    pub fn apply(&self, proofs: &[LatticeProof]) -> Result<LatticeProof> {
        if proofs.len() != self.arity {
            return Err(LatticeFoldError::InvalidInput(format!(
                "Expected {} proofs, but got {}",
                self.arity,
                proofs.len()
            )));
        }
        
        if self.matrices.len() != self.arity || self.challenges.len() != self.arity {
            return Err(LatticeFoldError::InvalidInput(format!(
                "Inconsistent fold operation: {} proofs, {} matrices, {} challenges",
                self.arity,
                self.matrices.len(),
                self.challenges.len()
            )));
        }
        
        // Extract the commitments from the proofs
        let mut folded_commitment = LatticePoint::zero(proofs[0].commitment.dimension());
        
        // Compute the linear combination of commitments
        for i in 0..self.arity {
            let challenge_scalar = self.challenges[i].as_integer();
            let mut weighted_commitment = proofs[i].commitment.clone();
            weighted_commitment.scale(challenge_scalar);
            folded_commitment.add(&weighted_commitment);
        }
        
        // Compute the folded response
        let mut folded_response = LatticePoint::zero(proofs[0].response.dimension());
        
        for i in 0..self.arity {
            let challenge_scalar = self.challenges[i].as_integer();
            
            // Apply matrix transformation to response
            let transformed_response = self.matrices[i].multiply_point(&proofs[i].response)?;
            
            // Scale by challenge
            let mut weighted_response = transformed_response.clone();
            weighted_response.scale(challenge_scalar);
            
            // Add to folded response
            folded_response.add(&weighted_response);
        }
        
        // Create the folded proof
        let folded_proof = LatticeProof {
            commitment: folded_commitment,
            response: folded_response,
            aux_data: None,
        };
        
        Ok(folded_proof)
    }
}

/// A recursive proof that consists of multiple folding operations
#[derive(Clone, Debug)]
pub struct RecursiveFoldProof<F> {
    /// The base proofs used in the recursion
    pub base_proofs: Vec<LatticeProof>,
    /// The folding operations applied
    pub fold_operations: Vec<FoldOperation<F>>,
    /// The final folded proof
    pub folded_proof: LatticeProof,
    /// A transcript of the folding process
    pub transcript: Option<Transcript>,
}

impl<F> RecursiveFoldProof<F> {
    /// Create a new recursive fold proof
    pub fn new(
        base_proofs: Vec<LatticeProof>,
        fold_operations: Vec<FoldOperation<F>>,
        folded_proof: LatticeProof,
    ) -> Self {
        Self {
            base_proofs,
            fold_operations,
            folded_proof,
            transcript: None,
        }
    }
    
    /// Get the verification cost (number of operations)
    pub fn verification_cost(&self) -> usize {
        let base_cost = self.base_proofs.len();
        let fold_cost = self.fold_operations.iter().map(|f| f.arity).sum::<usize>();
        base_cost + fold_cost
    }
    
    /// Get the proof size in bytes
    pub fn proof_size(&self) -> usize {
        // Size of folded proof
        let folded_size = self.folded_proof.size_in_bytes();
        
        // Size of fold operations
        let ops_size = self.fold_operations.iter().map(|f| {
            // Challenges + matrices
            let challenge_size = f.challenges.len() * 32; // 32 bytes per challenge
            let matrix_size = f.matrices.iter().map(|m| m.size_in_bytes()).sum::<usize>();
            challenge_size + matrix_size
        }).sum::<usize>();
        
        folded_size + ops_size
    }
}

/// A builder to create recursive fold proofs
pub struct RecursiveFoldBuilder<F> {
    /// The parameters for the recursive folding
    pub params: RecursiveFoldingParams,
    /// The challenge generator
    pub challenge_generator: ChallengeGenerator<F>,
    /// Cache for memoization
    cache: Arc<Mutex<HashMap<String, LatticeProof>>>,
}

impl<F> RecursiveFoldBuilder<F> {
    /// Create a new recursive fold builder
    pub fn new(
        params: RecursiveFoldingParams,
        challenge_generator: ChallengeGenerator<F>,
    ) -> Self {
        Self {
            params,
            challenge_generator,
            cache: Arc::new(Mutex::new(HashMap::new())),
        }
    }
    
    /// Build a recursive fold proof from a list of base proofs
    pub fn build(&mut self, base_proofs: Vec<LatticeProof>) -> Result<RecursiveFoldProof<F>> {
        // Clear the cache
        if self.params.memoize {
            self.cache.lock().unwrap().clear();
        }
        
        // Initialize fold operations
        let mut fold_operations = Vec::new();
        
        // Apply recursive folding
        let folded_proof = self.fold_recursive(
            &base_proofs,
            0,
            base_proofs.len(),
            &mut fold_operations,
        )?;
        
        // Create the recursive fold proof
        let proof = RecursiveFoldProof::new(base_proofs, fold_operations, folded_proof);
        
        Ok(proof)
    }
    
    /// Recursively fold proofs
    fn fold_recursive(
        &mut self,
        proofs: &[LatticeProof],
        start: usize,
        end: usize,
        fold_operations: &mut Vec<FoldOperation<F>>,
    ) -> Result<LatticeProof> {
        let num_proofs = end - start;
        
        // Base case: single proof
        if num_proofs == 1 {
            return Ok(proofs[start].clone());
        }
        
        // Try to get from cache if memoization is enabled
        if self.params.memoize {
            let cache_key = format!("{}:{}", start, end);
            let cache = self.cache.lock().unwrap();
            if let Some(cached_proof) = cache.get(&cache_key) {
                return Ok(cached_proof.clone());
            }
        }
        
        // Default case: fold in pairs
        let fold_arity = self.params.num_folds.min(num_proofs);
        
        // Divide the proofs into roughly equal parts
        let chunk_size = num_proofs / fold_arity;
        let remainder = num_proofs % fold_arity;
        
        let mut sub_proofs = Vec::with_capacity(fold_arity);
        let mut current = start;
        
        for i in 0..fold_arity {
            let sub_size = chunk_size + if i < remainder { 1 } else { 0 };
            let sub_end = current + sub_size;
            
            let sub_proof = self.fold_recursive(
                proofs,
                current,
                sub_end,
                fold_operations,
            )?;
            
            sub_proofs.push(sub_proof);
            current = sub_end;
        }
        
        // Generate challenges and matrices for the fold
        let mut challenges = Vec::with_capacity(fold_arity);
        let mut matrices = Vec::with_capacity(fold_arity);
        
        for i in 0..fold_arity {
            // Add the proof to the transcript
            self.challenge_generator.add_point(&sub_proofs[i].commitment, b"commitment");
            self.challenge_generator.add_point(&sub_proofs[i].response, b"response");
            
            // Generate a challenge
            let challenge = self.challenge_generator.generate_challenge::<rand::ThreadRng>(None);
            challenges.push(challenge.clone());
            
            // Derive a matrix from the challenge
            let derived_matrices = self.challenge_generator.derive_matrices(
                &challenge,
                &self.params.lattice_params,
                1,
            )?;
            
            matrices.push(derived_matrices[0].clone());
        }
        
        // Create a fold operation
        let fold_op = FoldOperation::new(fold_arity, matrices.clone(), challenges.clone());
        fold_operations.push(fold_op.clone());
        
        // Apply the fold
        let folded_proof = fold_op.apply(&sub_proofs)?;
        
        // Store in cache if memoization is enabled
        if self.params.memoize {
            let cache_key = format!("{}:{}", start, end);
            let mut cache = self.cache.lock().unwrap();
            cache.insert(cache_key, folded_proof.clone());
        }
        
        Ok(folded_proof)
    }
    
    /// Clear the memoization cache
    pub fn clear_cache(&mut self) {
        if self.params.memoize {
            self.cache.lock().unwrap().clear();
        }
    }
}

/// A verifier for recursive fold proofs
pub struct RecursiveFoldVerifier<F> {
    /// The parameters for the recursive folding
    pub params: RecursiveFoldingParams,
    /// The challenge generator
    pub challenge_generator: ChallengeGenerator<F>,
}

impl<F> RecursiveFoldVerifier<F> {
    /// Create a new recursive fold verifier
    pub fn new(
        params: RecursiveFoldingParams,
        challenge_generator: ChallengeGenerator<F>,
    ) -> Self {
        Self {
            params,
            challenge_generator,
        }
    }
    
    /// Verify a recursive fold proof
    pub fn verify(&mut self, proof: &RecursiveFoldProof<F>) -> Result<bool> {
        // Verify that the fold operations are consistent
        self.verify_fold_consistency(proof)?;
        
        // Recompute the folded proof
        let recomputed_proof = self.recompute_folded_proof(proof)?;
        
        // Check that the recomputed proof matches the claimed folded proof
        let commitments_equal = recomputed_proof.commitment == proof.folded_proof.commitment;
        let responses_equal = recomputed_proof.response == proof.folded_proof.response;
        
        Ok(commitments_equal && responses_equal)
    }
    
    /// Verify that the fold operations are consistent
    fn verify_fold_consistency(&self, proof: &RecursiveFoldProof<F>) -> Result<()> {
        // Check that base proofs and fold operations are not empty
        if proof.base_proofs.is_empty() {
            return Err(LatticeFoldError::InvalidInput(
                "No base proofs provided".to_string(),
            ));
        }
        
        if proof.fold_operations.is_empty() {
            return Err(LatticeFoldError::InvalidInput(
                "No fold operations provided".to_string(),
            ));
        }
        
        // Check dimensions consistency
        let expected_dim = proof.base_proofs[0].commitment.dimension();
        for (i, base_proof) in proof.base_proofs.iter().enumerate() {
            if base_proof.commitment.dimension() != expected_dim {
                return Err(LatticeFoldError::InvalidInput(format!(
                    "Base proof {} has inconsistent commitment dimension",
                    i
                )));
            }
            
            if base_proof.response.dimension() != expected_dim {
                return Err(LatticeFoldError::InvalidInput(format!(
                    "Base proof {} has inconsistent response dimension",
                    i
                )));
            }
        }
        
        // Check fold operations
        for (i, fold_op) in proof.fold_operations.iter().enumerate() {
            if fold_op.arity == 0 {
                return Err(LatticeFoldError::InvalidInput(format!(
                    "Fold operation {} has zero arity",
                    i
                )));
            }
            
            if fold_op.matrices.len() != fold_op.arity {
                return Err(LatticeFoldError::InvalidInput(format!(
                    "Fold operation {} has inconsistent number of matrices",
                    i
                )));
            }
            
            if fold_op.challenges.len() != fold_op.arity {
                return Err(LatticeFoldError::InvalidInput(format!(
                    "Fold operation {} has inconsistent number of challenges",
                    i
                )));
            }
            
            // Check matrix dimensions
            for (j, matrix) in fold_op.matrices.iter().enumerate() {
                if matrix.rows != expected_dim || matrix.cols != expected_dim {
                    return Err(LatticeFoldError::InvalidInput(format!(
                        "Matrix {} in fold operation {} has inconsistent dimensions",
                        j, i
                    )));
                }
            }
        }
        
        // Check final proof dimensions
        if proof.folded_proof.commitment.dimension() != expected_dim {
            return Err(LatticeFoldError::InvalidInput(
                "Folded proof has inconsistent commitment dimension".to_string(),
            ));
        }
        
        if proof.folded_proof.response.dimension() != expected_dim {
            return Err(LatticeFoldError::InvalidInput(
                "Folded proof has inconsistent response dimension".to_string(),
            ));
        }
        
        Ok(())
    }
    
    /// Recompute the folded proof from the base proofs and fold operations
    fn recompute_folded_proof(&mut self, proof: &RecursiveFoldProof<F>) -> Result<LatticeProof> {
        // Create a forest of proof trees
        let mut forest: Vec<LatticeProof> = proof.base_proofs.clone();
        
        // Apply each fold operation
        for fold_op in &proof.fold_operations {
            // Extract the proofs to fold
            let to_fold = forest.drain(0..fold_op.arity).collect::<Vec<_>>();
            
            // Apply the fold operation
            let folded = fold_op.apply(&to_fold)?;
            
            // Add the folded proof back to the forest
            forest.push(folded);
        }
        
        // The forest should now contain exactly one proof
        if forest.len() != 1 {
            return Err(LatticeFoldError::InvalidInput(format!(
                "Expected 1 final proof, but got {}",
                forest.len()
            )));
        }
        
        Ok(forest.pop().unwrap())
    }
    
    /// Verify a recursive fold proof with optimizations
    pub fn verify_optimized(&mut self, proof: &RecursiveFoldProof<F>) -> Result<bool> {
        // Check if the proof is small enough for direct verification
        if proof.base_proofs.len() <= 2 {
            return self.verify(proof);
        }
        
        // Otherwise, use optimized verification
        let folded_proof = self.recompute_folded_proof(proof)?;
        
        // Verify the folded proof against reference points
        let is_valid = folded_proof.commitment == proof.folded_proof.commitment &&
                       folded_proof.response == proof.folded_proof.response;
        
        Ok(is_valid)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::challenge_generation::ChallengeParams;
    use crate::lattice::LatticePoint;
    use rand::thread_rng;
    
    fn create_test_proofs(n: usize, dim: usize) -> Vec<LatticeProof> {
        let mut proofs = Vec::with_capacity(n);
        
        for i in 0..n {
            let mut commitment = LatticePoint::zero(dim);
            let mut response = LatticePoint::zero(dim);
            
            // Create some simple test values
            for j in 0..dim {
                commitment.coordinates[j] = ((i + 1) * (j + 1)) as i64;
                response.coordinates[j] = ((i + 2) * (j + 2)) as i64;
            }
            
            proofs.push(LatticeProof {
                commitment,
                response,
                aux_data: None,
            });
        }
        
        proofs
    }
    
    #[test]
    fn test_fold_operation() {
        // Create test parameters
        let params = LatticeParams {
            q: 97,  // Small prime for testing
            n: 4,
            sigma: 3.0,
            beta: 2.0,
        };
        
        // Create test proofs
        let proofs = create_test_proofs(2, params.n);
        
        // Create test matrices and challenges
        let mut matrices = Vec::new();
        let mut challenges = Vec::new();
        
        // Identity matrices for testing
        let mut matrix1_data = Vec::new();
        let mut matrix2_data = Vec::new();
        
        for i in 0..params.n {
            let mut row1 = vec![0; params.n];
            let mut row2 = vec![0; params.n];
            row1[i] = 1;
            row2[i] = 1;
            matrix1_data.push(row1);
            matrix2_data.push(row2);
        }
        
        matrices.push(LatticeMatrix::new(matrix1_data).unwrap());
        matrices.push(LatticeMatrix::new(matrix2_data).unwrap());
        
        // Simple challenges
        let challenge1 = Challenge::new([1; 32]);
        let challenge2 = Challenge::new([2; 32]);
        challenges.push(challenge1);
        challenges.push(challenge2);
        
        // Create the fold operation
        let fold_op = FoldOperation::<i64>::new(2, matrices, challenges);
        
        // Apply the fold
        let folded_proof = fold_op.apply(&proofs).unwrap();
        
        // Verify the results
        for i in 0..params.n {
            // With identity matrices and challenges 1 and 2, we expect:
            // folded_commitment[i] = 1 * proofs[0].commitment[i] + 2 * proofs[1].commitment[i]
            let expected_commitment = 1 * proofs[0].commitment.coordinates[i] + 
                                     2 * proofs[1].commitment.coordinates[i];
            
            // folded_response[i] = 1 * proofs[0].response[i] + 2 * proofs[1].response[i]
            let expected_response = 1 * proofs[0].response.coordinates[i] + 
                                   2 * proofs[1].response.coordinates[i];
            
            assert_eq!(folded_proof.commitment.coordinates[i], expected_commitment);
            assert_eq!(folded_proof.response.coordinates[i], expected_response);
        }
    }
    
    #[test]
    fn test_recursive_fold_builder() {
        // Create test parameters
        let lattice_params = LatticeParams {
            q: 97,  // Small prime for testing
            n: 4,
            sigma: 3.0,
            beta: 2.0,
        };
        
        let fold_params = RecursiveFoldingParams {
            lattice_params: lattice_params.clone(),
            num_folds: 2,
            recursion_depth: 2,
            parallel: false,
            memoize: true,
            security_param: 128,
        };
        
        let challenge_params = ChallengeParams::default();
        let challenge_generator = ChallengeGenerator::<i64>::new(challenge_params);
        
        // Create a builder
        let mut builder = RecursiveFoldBuilder::new(fold_params, challenge_generator);
        
        // Create test proofs
        let proofs = create_test_proofs(4, lattice_params.n);
        
        // Build a recursive fold proof
        let fold_proof = builder.build(proofs.clone()).unwrap();
        
        // Verify the structure
        assert_eq!(fold_proof.base_proofs.len(), 4);
        assert_eq!(fold_proof.fold_operations.len(), 3);  // 2 to fold pairs, 1 to fold the results
        
        // Create a verifier
        let challenge_params = ChallengeParams::default();
        let challenge_generator = ChallengeGenerator::<i64>::new(challenge_params);
        let mut verifier = RecursiveFoldVerifier::new(fold_params, challenge_generator);
        
        // Verify the proof
        let verification_result = verifier.verify(&fold_proof).unwrap();
        assert!(verification_result);
    }
    
    #[test]
    fn test_recursive_fold_with_invalid_proof() {
        // Create test parameters
        let lattice_params = LatticeParams {
            q: 97,  // Small prime for testing
            n: 4,
            sigma: 3.0,
            beta: 2.0,
        };
        
        let fold_params = RecursiveFoldingParams {
            lattice_params: lattice_params.clone(),
            num_folds: 2,
            recursion_depth: 2,
            parallel: false,
            memoize: true,
            security_param: 128,
        };
        
        let challenge_params = ChallengeParams::default();
        let challenge_generator = ChallengeGenerator::<i64>::new(challenge_params);
        
        // Create a builder
        let mut builder = RecursiveFoldBuilder::new(fold_params.clone(), challenge_generator);
        
        // Create test proofs
        let mut proofs = create_test_proofs(4, lattice_params.n);
        
        // Build a recursive fold proof
        let fold_proof = builder.build(proofs.clone()).unwrap();
        
        // Create a verifier
        let challenge_params = ChallengeParams::default();
        let challenge_generator = ChallengeGenerator::<i64>::new(challenge_params);
        let mut verifier = RecursiveFoldVerifier::new(fold_params, challenge_generator);
        
        // Verify the proof - should be valid
        let verification_result = verifier.verify(&fold_proof).unwrap();
        assert!(verification_result);
        
        // Now tamper with the folded proof
        let mut tampered_proof = fold_proof.clone();
        tampered_proof.folded_proof.commitment.coordinates[0] += 1;
        
        // Verify the tampered proof - should be invalid
        let verification_result = verifier.verify(&tampered_proof).unwrap();
        assert!(!verification_result);
    }
} 