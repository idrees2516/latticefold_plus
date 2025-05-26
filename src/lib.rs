pub mod challenge_generation;
pub mod commitment;
pub mod commitment_sis;
pub mod error;
pub mod final_proof;
pub mod folding;
pub mod lattice;
pub mod protocol;
pub mod quantum_resistance;
pub mod recursive_folding;
pub mod side_channel;
pub mod types;
pub mod zkp;

pub use challenge_generation::{Challenge, ChallengeGenerator};
pub use commitment::{
    FoldingCommitment, PedersenCommitment, PolynomialCommitment, VectorCommitment,
};
pub use commitment_sis::{SISCommitment, SISCommitmentWithOpening};
pub use error::{LatticeFoldError, Result};
pub use final_proof::{FinalProof, FinalProver, FinalVerifier};
pub use folding::{FoldingProof, FoldingScheme};
pub use lattice::{LatticeBasis, LatticeMatrix, LatticeParams, LatticePoint, LatticeRelation};
pub use protocol::{
    LatticeFoldInstance, ProtocolParams, ProtocolProof, ProtocolProver, ProtocolVerifier,
};
pub use quantum_resistance::{
    QuantumResistanceAnalyzer, QuantumResistanceParams, QuantumResistantSampler, SecurityLevel,
};
pub use recursive_folding::{
    RecursiveFoldingScheme, RecursiveProof, RecursiveRelation, RecursiveWitness,
};
pub use side_channel::{
    ConstantTimeGaussianSampler, ConstantTimeLatticePoint, ConstantTimeVerifier,
};
pub use types::{CommitmentParams, FinalProof, LatticeFoldInstance, LatticeFoldParams};
pub use zkp::{FoldingProof as ZKFoldingProof, ZKProof, ZKProver, ZKVerifier};

use ark_ec::{AffineCurve, ProjectiveCurve};
use ark_ff::{Field, PrimeField};
use ark_std::{rand::Rng, UniformRand};
use std::marker::PhantomData;

/// Create a new LatticeFold+ instance with the specified security level.
/// This initializes all the required components for the LatticeFold+ protocol.
pub fn setup_lattice_fold<G: AffineCurve, F: Field, R: Rng>(
    security_level: SecurityLevel,
    rng: &mut R,
) -> Result<ProtocolParams<G, F>> {
    // Get the optimal parameters for the security level
    let quantum_analyzer = QuantumResistanceAnalyzer::new();
    let qr_params = quantum_analyzer.get_params(security_level)?;
    let lattice_params = quantum_analyzer.create_lattice_params(&qr_params);
    
    // Setup commitment scheme
    let commitment_scheme = FoldingCommitment::new(
        lattice_params.n,
        lattice_params.n,
        rng,
    );
    
    // Setup SIS commitment scheme
    let sis_commitment = SISCommitment::<F>::new(&lattice_params, rng);
    
    // Setup lattice relation
    let basis = LatticeBasis::new(lattice_params.n);
    let target = LatticePoint::random(&basis, rng);
    let bound = F::from(2u32).pow(&[lattice_params.n as u64]);
    let lattice_relation = LatticeRelation::new(basis, target, bound);
    
    // Setup challenge generator
    let challenge_generator = ChallengeGenerator::new(qr_params.security_level);
    
    // Create protocol parameters
    Ok(ProtocolParams {
        dimension: lattice_params.n,
        modulus: F::from(lattice_params.q),
        security_param: qr_params.security_level,
        commitment_scheme,
        lattice_relation,
        _phantom: PhantomData,
    })
}

/// Prove a statement using the LatticeFold+ protocol.
pub fn prove_lattice_fold<G: AffineCurve, F: Field, R: Rng>(
    params: &ProtocolParams<G, F>,
    instances: &[LatticeFoldInstance<F>],
    rng: &mut R,
) -> Result<ProtocolProof<G, F>> {
    // Validate instances
    if instances.is_empty() {
        return Err(LatticeFoldError::InvalidParameters(
            "Must provide at least one instance".to_string(),
        ));
    }
    
    for instance in instances {
        if instance.witness.len() != params.dimension {
            return Err(LatticeFoldError::InvalidDimension {
                expected: params.dimension,
                got: instance.witness.len(),
            });
        }
        if instance.public_input.len() != params.dimension {
            return Err(LatticeFoldError::InvalidDimension {
                expected: params.dimension,
                got: instance.public_input.len(),
            });
        }
    }
    
    // Create side-channel resistant prover
    let mut prover = ProtocolProver::new(params.clone());
    
    // Generate the proof
    Ok(prover.prove(instances, rng))
}

/// Verify a proof using the LatticeFold+ protocol.
pub fn verify_lattice_fold<G: AffineCurve, F: Field>(
    params: &ProtocolParams<G, F>,
    proof: &ProtocolProof<G, F>,
) -> Result<bool> {
    // Create side-channel resistant verifier
    let mut verifier = ProtocolVerifier::new(params.clone());
    
    // Verify the proof
    Ok(verifier.verify(proof))
}

/// Create a recursive proof from multiple proofs.
pub fn create_recursive_proof<G: AffineCurve, F: Field, R: Rng>(
    params: &ProtocolParams<G, F>,
    proofs: &[ProtocolProof<G, F>],
    rng: &mut R,
) -> Result<ProtocolProof<G, F>> {
    if proofs.is_empty() {
        return Err(LatticeFoldError::InvalidParameters(
            "Must provide at least one proof".to_string(),
        ));
    }
    
    // Create recursive prover
    let mut prover = ProtocolProver::new(params.clone());
    
    // Generate the recursive proof
    let mut instances = Vec::new();
    for proof in proofs {
        instances.extend_from_slice(&proof.folding_proof.folded_witness);
    }
    
    // Create a combined instance
    let combined_instance = LatticeFoldInstance {
        witness: instances,
        public_input: vec![F::one()],
    };
    
    // Generate the proof
    Ok(prover.prove(&[combined_instance], rng))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bls12_381::{Bls12_381, Fr};
    use ark_std::test_rng;
    
    #[test]
    fn test_lattice_fold() {
        let mut rng = test_rng();
        
        // Setup with medium security level
        let params = setup_lattice_fold::<Bls12_381, Fr, _>(SecurityLevel::Medium, &mut rng).unwrap();
        
        // Generate test instances
        let mut instances = Vec::new();
        for _ in 0..3 {
            let witness = vec![Fr::rand(&mut rng); params.dimension];
            let public_input = vec![Fr::rand(&mut rng); params.dimension];
            instances.push(LatticeFoldInstance {
                witness,
                public_input,
            });
        }
        
        // Prove
        let proof = prove_lattice_fold(&params, &instances, &mut rng).unwrap();
        
        // Verify
        assert!(verify_lattice_fold(&params, &proof).unwrap());
        
        // Create recursive proof
        let recursive_proof = create_recursive_proof(&params, &[proof], &mut rng).unwrap();
        
        // Verify recursive proof
        assert!(verify_lattice_fold(&params, &recursive_proof).unwrap());
    }
} 