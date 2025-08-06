pub mod challenge_generation;
pub mod commitment;
pub mod commitment_sis;
pub mod commitment_transformation;
pub mod cyclotomic_ring;
pub mod double_commitment;
pub mod end_to_end_folding;
pub mod error;
pub mod final_proof;
pub mod folding_challenge_generation;
pub mod folding;
pub mod gadget;
pub mod gpu;
pub mod integration_tests;
pub mod security_integration_tests;
pub mod lattice;
pub mod memory;
pub mod parameter_generation;
pub mod modular_arithmetic;
pub mod monomial;
pub mod monomial_commitment;
pub mod monomial_set_checking;
pub mod multi_instance_folding;
pub mod range_check_protocol;
pub mod simd;
pub mod sumcheck_batching;
pub mod msis;
pub mod norm_computation;
pub mod ntt;
pub mod polynomial_multiplication;
pub mod protocol;
pub mod quantum_resistance;
pub mod r1cs;
pub mod r1cs_tests;
pub mod recursive_folding;
pub mod ring_sumcheck;
pub mod security_analysis;
pub mod security;
pub mod side_channel;
pub mod types;
pub mod zkp;
pub mod integration_tests;

#[cfg(test)]
pub mod tests;

pub use challenge_generation::{Challenge, ChallengeGenerator};
pub use commitment::{
    FoldingCommitment, PedersenCommitment, PolynomialCommitment, VectorCommitment,
};
pub use commitment_sis::{SISCommitment, SISCommitmentWithOpening};
pub use commitment_transformation::{
    CommitmentTransformationProtocol, CommitmentTransformationParams, 
    CommitmentTransformationProof, DecompositionProof, CommitmentTransformationStats
};
pub use folding_challenge_generation::{
    FoldingChallengeGenerator, FoldingChallengeParams, FoldingChallenges,
    StrongSamplingSet, FoldingChallengeStats
};
pub use cyclotomic_ring::{BalancedCoefficients, RingElement, MAX_RING_DIMENSION, MIN_RING_DIMENSION};
pub use double_commitment::{DoubleCommitmentScheme, DoubleCommitmentParams};
pub use modular_arithmetic::{BarrettParams, MontgomeryParams, ModularArithmetic};
pub use monomial::{Monomial, MonomialSet, MonomialMembershipTester};
pub use monomial_commitment::{MonomialVector, MonomialCommitmentScheme, CommitmentStats};
pub use monomial_set_checking::{
    MultilinearExtension, SumcheckProtocol, UnivariatePolynomial, SumcheckProof,
    MonomialSetCheckingProtocol, MonomialSetCheckingProof
};
pub use range_check_protocol::{
    RangeCheckProtocol, RangeCheckProof, RangeCheckStats, ConsistencyProof
};
pub use sumcheck_batching::{
    BatchedSumcheckProtocol, BatchedSumcheckProof, BatchedSumcheckStats
};
pub use security_analysis::{
    SecurityAnalyzer, SecurityAnalysisResults, LinearCommitmentSecurity, KnowledgeErrorAnalysis,
    ExtractorAnalysis, BindingVerification, NormBoundAnalysis, ParameterAdequacy,
    MaliciousProverResistance, OverallSecurityAssessment, AnalysisStatistics
};
pub use multi_instance_folding::{
    LinearRelation, MultiInstanceLinearRelation, LinearFoldingParams, LinearFoldingProof,
    LinearFoldingStats, LinearFoldingProtocol
};
pub use end_to_end_folding::{
    EndToEndFoldingSystem, EndToEndFoldingParams, EndToEndFoldingProof, EndToEndFoldingStats,
    R1CSInstance, LinearizationProof, TimingBreakdown, CommunicationBreakdown
};
pub use gadget::{GadgetVector, GadgetMatrix, GadgetParams, DecompositionCache, LookupTables, StreamingDecomposer, GadgetVerificationReport, VerificationSummary, TimingStats};
pub use msis::{MSISParams, MSISMatrix, OptimizationTarget, SecurityEstimate, MemoryRequirements, MatrixValidation};
pub use norm_computation::{InfinityNorm, EuclideanNorm, OperatorNorm};
pub use ntt::{NTTParams, NTTEngine, NTTMultiplier, get_ntt_params, clear_ntt_params_cache};
pub use polynomial_multiplication::{
    schoolbook_multiply_optimized, karatsuba_multiply_optimized, multiply_with_algorithm_selection
};
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
pub use parameter_generation::{
    AutomatedParameterGenerator, GeneratedParameters, TargetSecurityLevel, PerformanceMetrics,
    ProofSizeEstimates, ValidationStatus, AttackComplexityEstimator, PerformanceProfiler,
    ParameterOptimizer, GeneratorConfig
};
pub use recursive_folding::{
    RecursiveFoldingScheme, RecursiveProof, RecursiveRelation, RecursiveWitness,
};
pub use side_channel::{
    ConstantTimeGaussianSampler, ConstantTimeLatticePoint, ConstantTimeVerifier,
};
pub use ring_sumcheck::{
    MultilinearExtension, MultilinearExtensionStats, TensorProduct, TensorProductStats
};
pub use r1cs::{
    R1CSMatrices, CommittedR1CS, AuxiliaryMatrices, CommittedR1CSStats, CCSMatrices, 
    CommittedCCS, CCSStats, ConstraintSystem, ConstraintDegree
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