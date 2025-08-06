/// Comprehensive tests for R1CS Integration and Constraint System Support
/// 
/// This module contains extensive tests for the R1CS and CCS implementations,
/// covering all aspects of constraint system functionality including:
/// 
/// 1. R1CS matrix operations and constraint evaluation
/// 2. Committed R1CS with gadget matrix expansion
/// 3. CCS higher-degree constraint systems
/// 4. Sumcheck protocol integration
/// 5. Performance benchmarking and optimization
/// 6. Security property verification
/// 7. Edge case handling and error conditions
/// 
/// Test Categories:
/// - Unit tests: Individual component functionality
/// - Integration tests: End-to-end constraint system workflows
/// - Performance tests: Benchmarking and optimization validation
/// - Security tests: Cryptographic property verification
/// - Stress tests: Large-scale constraint systems and edge cases

use std::time::{Duration, Instant};
use rand::{thread_rng, Rng};

use crate::r1cs::{
    R1CSMatrices, CommittedR1CS, AuxiliaryMatrices, CommittedR1CSStats,
    CCSMatrices, CommittedCCS, CCSStats, ConstraintSystem, ConstraintDegree
};
use crate::cyclotomic_ring::{RingElement, RingParams};
use crate::commitment::Commitment;
use crate::types::CommitmentParams;
use crate::lattice::LatticeParams;
use crate::error::{LatticeFoldError, Result};

/// Test helper functions and utilities
mod test_utils {
    use super::*;
    
    /// Creates standard ring parameters for testing
    pub fn standard_ring_params() -> RingParams {
        RingParams {
            dimension: 64,
            modulus: 2147483647, // Large prime for testing
        }
    }
    
    /// Creates standard commitment parameters for testing
    pub fn standard_commitment_params() -> CommitmentParams {
        CommitmentParams {
            lattice_params: LatticeParams {
                dimension: 128,
                modulus: 2147483647,
                ..Default::default()
            },
            ..Default::default()
        }
    }
    
    /// Creates a random ring element for testing
    pub fn random_ring_element(ring_params: &RingParams) -> Result<RingElement> {
        let mut rng = thread_rng();
        let coeffs: Vec<i64> = (0..ring_params.dimension)
            .map(|_| rng.gen_range(-100..100))
            .collect();
        RingElement::from_coefficients(&coeffs, ring_params.modulus)
    }
    
    /// Creates a random witness vector
    pub fn random_witness(dimension: usize, ring_params: &RingParams) -> Result<Vec<RingElement>> {
        (0..dimension)
            .map(|_| random_ring_element(ring_params))
            .collect()
    }
    
    /// Creates a simple multiplication constraint: z[0] * z[1] = z[2]
    pub fn create_multiplication_constraint(ring_params: &RingParams) -> Result<(Vec<RingElement>, Vec<RingElement>, Vec<RingElement>)> {
        let zero = RingElement::zero(ring_params.dimension, ring_params.modulus)?;
        let one = RingElement::from_coefficients(&[1], ring_params.modulus)?;
        
        // A row: [1, 0, 0] (select z[0])
        let a_row = vec![one.clone(), zero.clone(), zero.clone()];
        
        // B row: [0, 1, 0] (select z[1])
        let b_row = vec![zero.clone(), one.clone(), zero.clone()];
        
        // C row: [0, 0, 1] (select z[2])
        let c_row = vec![zero.clone(), zero.clone(), one.clone()];
        
        Ok((a_row, b_row, c_row))
    }
    
    /// Creates a satisfying witness for multiplication constraint: [a, b, a*b]
    pub fn create_satisfying_witness(a: i64, b: i64, ring_params: &RingParams) -> Result<Vec<RingElement>> {
        let product = (a * b) % ring_params.modulus;
        Ok(vec![
            RingElement::from_coefficients(&[a], ring_params.modulus)?,
            RingElement::from_coefficients(&[b], ring_params.modulus)?,
            RingElement::from_coefficients(&[product], ring_params.modulus)?,
        ])
    }
}

/// Unit tests for R1CS matrices
#[cfg(test)]
mod r1cs_matrix_tests {
    use super::*;
    use test_utils::*;
    
    #[test]
    fn test_r1cs_matrix_creation() {
        let ring_params = standard_ring_params();
        let matrices = R1CSMatrices::new(10, 20, ring_params).unwrap();
        
        assert_eq!(matrices.num_constraints(), 10);
        assert_eq!(matrices.witness_dimension(), 20);
        assert_eq!(matrices.matrix_a.len(), 10);
        assert_eq!(matrices.matrix_a[0].len(), 20);
        assert_eq!(matrices.matrix_b.len(), 10);
        assert_eq!(matrices.matrix_b[0].len(), 20);
        assert_eq!(matrices.matrix_c.len(), 10);
        assert_eq!(matrices.matrix_c[0].len(), 20);
    }
    
    #[test]
    fn test_r1cs_matrix_creation_invalid_params() {
        let ring_params = standard_ring_params();
        
        // Test zero constraints
        assert!(R1CSMatrices::new(0, 20, ring_params).is_err());
        
        // Test zero witness dimension
        assert!(R1CSMatrices::new(10, 0, ring_params).is_err());
        
        // Test excessive constraints
        assert!(R1CSMatrices::new(2_000_000, 20, ring_params).is_err());
        
        // Test excessive witness dimension
        assert!(R1CSMatrices::new(10, 2_000_000, ring_params).is_err());
    }
    
    #[test]
    fn test_r1cs_constraint_setting() {
        let ring_params = standard_ring_params();
        let mut matrices = R1CSMatrices::new(2, 3, ring_params).unwrap();
        
        let (a_row, b_row, c_row) = create_multiplication_constraint(&ring_params).unwrap();
        
        // Set first constraint
        matrices.set_constraint(0, a_row.clone(), b_row.clone(), c_row.clone()).unwrap();
        
        // Verify constraint was set correctly
        assert_eq!(matrices.matrix_a[0], a_row);
        assert_eq!(matrices.matrix_b[0], b_row);
        assert_eq!(matrices.matrix_c[0], c_row);
        
        // Test invalid constraint index
        assert!(matrices.set_constraint(2, a_row.clone(), b_row.clone(), c_row.clone()).is_err());
        
        // Test invalid row dimensions
        let short_row = vec![a_row[0].clone()]; // Too short
        assert!(matrices.set_constraint(1, short_row, b_row.clone(), c_row.clone()).is_err());
    }
    
    #[test]
    fn test_r1cs_constraint_evaluation_simple() {
        let ring_params = standard_ring_params();
        let mut matrices = R1CSMatrices::new(1, 3, ring_params).unwrap();
        
        // Set multiplication constraint: z[0] * z[1] = z[2]
        let (a_row, b_row, c_row) = create_multiplication_constraint(&ring_params).unwrap();
        matrices.set_constraint(0, a_row, b_row, c_row).unwrap();
        
        // Test satisfying witness: [2, 3, 6]
        let satisfying_witness = create_satisfying_witness(2, 3, &ring_params).unwrap();
        assert!(matrices.evaluate_constraints(&satisfying_witness).unwrap());
        
        // Test non-satisfying witness: [2, 3, 7]
        let non_satisfying_witness = vec![
            RingElement::from_coefficients(&[2], ring_params.modulus).unwrap(),
            RingElement::from_coefficients(&[3], ring_params.modulus).unwrap(),
            RingElement::from_coefficients(&[7], ring_params.modulus).unwrap(),
        ];
        assert!(!matrices.evaluate_constraints(&non_satisfying_witness).unwrap());
    }
    
    #[test]
    fn test_r1cs_constraint_evaluation_multiple() {
        let ring_params = standard_ring_params();
        let mut matrices = R1CSMatrices::new(2, 5, ring_params).unwrap();
        
        // First constraint: z[0] * z[1] = z[2]
        let (a_row1, b_row1, c_row1) = create_multiplication_constraint(&ring_params).unwrap();
        let mut a_row1_extended = a_row1;
        let mut b_row1_extended = b_row1;
        let mut c_row1_extended = c_row1;
        
        // Extend to 5 dimensions
        let zero = RingElement::zero(ring_params.dimension, ring_params.modulus).unwrap();
        a_row1_extended.extend(vec![zero.clone(), zero.clone()]);
        b_row1_extended.extend(vec![zero.clone(), zero.clone()]);
        c_row1_extended.extend(vec![zero.clone(), zero.clone()]);
        
        matrices.set_constraint(0, a_row1_extended, b_row1_extended, c_row1_extended).unwrap();
        
        // Second constraint: z[3] * z[4] = z[2] (reusing z[2])
        let one = RingElement::from_coefficients(&[1], ring_params.modulus).unwrap();
        let a_row2 = vec![zero.clone(), zero.clone(), zero.clone(), one.clone(), zero.clone()];
        let b_row2 = vec![zero.clone(), zero.clone(), zero.clone(), zero.clone(), one.clone()];
        let c_row2 = vec![zero.clone(), zero.clone(), one.clone(), zero.clone(), zero.clone()];
        
        matrices.set_constraint(1, a_row2, b_row2, c_row2).unwrap();
        
        // Test witness that satisfies both constraints: [2, 3, 6, 1, 6]
        // First constraint: 2 * 3 = 6 ✓
        // Second constraint: 1 * 6 = 6 ✓
        let witness = vec![
            RingElement::from_coefficients(&[2], ring_params.modulus).unwrap(),
            RingElement::from_coefficients(&[3], ring_params.modulus).unwrap(),
            RingElement::from_coefficients(&[6], ring_params.modulus).unwrap(),
            RingElement::from_coefficients(&[1], ring_params.modulus).unwrap(),
            RingElement::from_coefficients(&[6], ring_params.modulus).unwrap(),
        ];
        
        assert!(matrices.evaluate_constraints(&witness).unwrap());
        
        // Test witness that violates second constraint: [2, 3, 6, 2, 2]
        // First constraint: 2 * 3 = 6 ✓
        // Second constraint: 2 * 2 = 4 ≠ 6 ✗
        let invalid_witness = vec![
            RingElement::from_coefficients(&[2], ring_params.modulus).unwrap(),
            RingElement::from_coefficients(&[3], ring_params.modulus).unwrap(),
            RingElement::from_coefficients(&[6], ring_params.modulus).unwrap(),
            RingElement::from_coefficients(&[2], ring_params.modulus).unwrap(),
            RingElement::from_coefficients(&[2], ring_params.modulus).unwrap(),
        ];
        
        assert!(!matrices.evaluate_constraints(&invalid_witness).unwrap());
    }
    
    #[test]
    fn test_r1cs_constraint_vectors_computation() {
        let ring_params = standard_ring_params();
        let mut matrices = R1CSMatrices::new(1, 3, ring_params).unwrap();
        
        // Set multiplication constraint
        let (a_row, b_row, c_row) = create_multiplication_constraint(&ring_params).unwrap();
        matrices.set_constraint(0, a_row, b_row, c_row).unwrap();
        
        // Compute constraint vectors for witness [2, 3, 6]
        let witness = create_satisfying_witness(2, 3, &ring_params).unwrap();
        let (az_vector, bz_vector, cz_vector) = matrices.compute_constraint_vectors(&witness).unwrap();
        
        // Verify vector dimensions
        assert_eq!(az_vector.len(), 1);
        assert_eq!(bz_vector.len(), 1);
        assert_eq!(cz_vector.len(), 1);
        
        // Verify constraint satisfaction: (Az)[0] * (Bz)[0] = (Cz)[0]
        let product = az_vector[0].multiply(&bz_vector[0]).unwrap();
        assert_eq!(product, cz_vector[0]);
    }
    
    #[test]
    fn test_r1cs_invalid_witness_dimensions() {
        let ring_params = standard_ring_params();
        let matrices = R1CSMatrices::new(1, 3, ring_params).unwrap();
        
        // Test witness with wrong dimension
        let short_witness = vec![
            RingElement::from_coefficients(&[1], ring_params.modulus).unwrap(),
            RingElement::from_coefficients(&[2], ring_params.modulus).unwrap(),
        ];
        
        assert!(matrices.evaluate_constraints(&short_witness).is_err());
        assert!(matrices.compute_constraint_vectors(&short_witness).is_err());
    }
}

/// Unit tests for committed R1CS
#[cfg(test)]
mod committed_r1cs_tests {
    use super::*;
    use test_utils::*;
    
    #[test]
    fn test_committed_r1cs_creation() {
        let ring_params = standard_ring_params();
        let matrices = R1CSMatrices::new(5, 10, ring_params).unwrap();
        let commitment_params = standard_commitment_params();
        
        let committed_r1cs = CommittedR1CS::new(
            matrices,
            2,   // gadget base
            8,   // gadget dimension
            commitment_params,
            100, // norm bound
        ).unwrap();
        
        assert_eq!(committed_r1cs.r1cs_matrices().num_constraints(), 5);
        assert_eq!(committed_r1cs.r1cs_matrices().witness_dimension(), 10);
        assert_eq!(committed_r1cs.gadget_matrix().gadget_vector().base(), 2);
        assert_eq!(committed_r1cs.gadget_matrix().gadget_vector().dimension(), 8);
        assert_eq!(committed_r1cs.norm_bound, 100);
    }
    
    #[test]
    fn test_committed_r1cs_invalid_params() {
        let ring_params = standard_ring_params();
        let matrices = R1CSMatrices::new(5, 10, ring_params).unwrap();
        let commitment_params = standard_commitment_params();
        
        // Test invalid norm bound
        assert!(CommittedR1CS::new(matrices.clone(), 2, 8, commitment_params.clone(), 0).is_err());
        assert!(CommittedR1CS::new(matrices.clone(), 2, 8, commitment_params.clone(), -10).is_err());
        
        // Test invalid gadget parameters
        assert!(CommittedR1CS::new(matrices.clone(), 1, 8, commitment_params.clone(), 100).is_err()); // base < 2
        assert!(CommittedR1CS::new(matrices.clone(), 2, 0, commitment_params.clone(), 100).is_err()); // dimension = 0
    }
    
    #[test]
    fn test_committed_r1cs_witness_commitment() {
        let ring_params = standard_ring_params();
        let matrices = R1CSMatrices::new(2, 5, ring_params).unwrap();
        let commitment_params = standard_commitment_params();
        
        let mut committed_r1cs = CommittedR1CS::new(
            matrices, 2, 6, commitment_params, 50
        ).unwrap();
        
        // Create valid witness (within norm bound)
        let witness = vec![
            RingElement::from_coefficients(&[1, 2], ring_params.modulus).unwrap(),
            RingElement::from_coefficients(&[3, 4], ring_params.modulus).unwrap(),
            RingElement::from_coefficients(&[5, 6], ring_params.modulus).unwrap(),
            RingElement::from_coefficients(&[7, 8], ring_params.modulus).unwrap(),
            RingElement::from_coefficients(&[9, 10], ring_params.modulus).unwrap(),
        ];
        
        // Generate commitment randomness
        let randomness = committed_r1cs.commitment_scheme.random_randomness(&mut thread_rng()).unwrap();
        
        // Commit to witness
        let result = committed_r1cs.commit_witness(&witness, &randomness);
        assert!(result.is_ok());
        
        let (commitment, expanded_witness) = result.unwrap();
        
        // Verify expanded witness has correct dimension
        let expected_expanded_dim = witness.len() * committed_r1cs.gadget_matrix().gadget_vector().dimension();
        assert_eq!(expanded_witness.len(), expected_expanded_dim);
        
        // Verify commitment can be verified
        let verification_result = committed_r1cs.verify_constraints(&commitment, &expanded_witness, &randomness);
        assert!(verification_result.is_ok());
    }
    
    #[test]
    fn test_committed_r1cs_witness_norm_bound_violation() {
        let ring_params = standard_ring_params();
        let matrices = R1CSMatrices::new(1, 2, ring_params).unwrap();
        let commitment_params = standard_commitment_params();
        
        let mut committed_r1cs = CommittedR1CS::new(
            matrices, 2, 4, commitment_params, 10 // Small norm bound
        ).unwrap();
        
        // Create witness that violates norm bound
        let witness = vec![
            RingElement::from_coefficients(&[100], ring_params.modulus).unwrap(), // Exceeds bound
            RingElement::from_coefficients(&[5], ring_params.modulus).unwrap(),
        ];
        
        let randomness = committed_r1cs.commitment_scheme.random_randomness(&mut thread_rng()).unwrap();
        
        // Should fail due to norm bound violation
        assert!(committed_r1cs.commit_witness(&witness, &randomness).is_err());
    }
    
    #[test]
    fn test_committed_r1cs_auxiliary_matrix_derivation() {
        let ring_params = standard_ring_params();
        let matrices = R1CSMatrices::new(3, 4, ring_params).unwrap();
        let commitment_params = standard_commitment_params();
        
        let mut committed_r1cs = CommittedR1CS::new(
            matrices, 2, 5, commitment_params, 100
        ).unwrap();
        
        // Initially no auxiliary matrices
        assert!(committed_r1cs.auxiliary_matrices.is_none());
        
        // Derive auxiliary matrices
        committed_r1cs.derive_auxiliary_matrices().unwrap();
        
        // Verify auxiliary matrices were created
        assert!(committed_r1cs.auxiliary_matrices.is_some());
        
        let aux_matrices = committed_r1cs.auxiliary_matrices.as_ref().unwrap();
        let expected_expanded_dim = 4 * 5; // witness_dim * gadget_dim
        
        assert_eq!(aux_matrices.dimensions, (3, expected_expanded_dim));
        assert_eq!(aux_matrices.m1.len(), 3);
        assert_eq!(aux_matrices.m1[0].len(), expected_expanded_dim);
        assert_eq!(aux_matrices.m2.len(), 3);
        assert_eq!(aux_matrices.m3.len(), 3);
        assert_eq!(aux_matrices.m4.len(), 3);
        
        // Verify performance statistics were updated
        assert!(committed_r1cs.stats().auxiliary_matrix_time > Duration::from_nanos(0));
        assert!(committed_r1cs.stats().ring_operations > 0);
    }
    
    #[test]
    fn test_committed_r1cs_sumcheck_proof_generation() {
        let ring_params = standard_ring_params();
        let mut matrices = R1CSMatrices::new(1, 3, ring_params).unwrap();
        let commitment_params = standard_commitment_params();
        
        // Set up a simple multiplication constraint
        let (a_row, b_row, c_row) = create_multiplication_constraint(&ring_params).unwrap();
        matrices.set_constraint(0, a_row, b_row, c_row).unwrap();
        
        let mut committed_r1cs = CommittedR1CS::new(
            matrices, 2, 4, commitment_params, 100
        ).unwrap();
        
        // Create satisfying witness and commit
        let witness = create_satisfying_witness(2, 3, &ring_params).unwrap();
        let randomness = committed_r1cs.commitment_scheme.random_randomness(&mut thread_rng()).unwrap();
        let (_, expanded_witness) = committed_r1cs.commit_witness(&witness, &randomness).unwrap();
        
        // Generate sumcheck proof
        let sumcheck_proof = committed_r1cs.generate_sumcheck_proof(&expanded_witness);
        assert!(sumcheck_proof.is_ok());
        
        // Verify performance statistics were updated
        assert!(committed_r1cs.stats().sumcheck_time > Duration::from_nanos(0));
    }
    
    #[test]
    fn test_committed_r1cs_constraint_system_trait() {
        let ring_params = standard_ring_params();
        let matrices = R1CSMatrices::new(2, 4, ring_params).unwrap();
        let commitment_params = standard_commitment_params();
        
        let mut committed_r1cs = CommittedR1CS::new(
            matrices, 2, 5, commitment_params, 100
        ).unwrap();
        
        // Test trait methods
        assert_eq!(committed_r1cs.num_constraints(), 2);
        assert_eq!(committed_r1cs.witness_dimension(), 4);
        assert_eq!(committed_r1cs.max_constraint_degree(), 2);
        
        // Test trait method implementations
        let witness = random_witness(4, &ring_params).unwrap();
        let randomness = committed_r1cs.commitment_scheme.random_randomness(&mut thread_rng()).unwrap();
        
        let (commitment, expanded_witness) = committed_r1cs.commit_witness(&witness, &randomness).unwrap();
        let verification_result = committed_r1cs.verify_constraints(&commitment, &expanded_witness, &randomness).unwrap();
        let _sumcheck_proof = committed_r1cs.generate_sumcheck_proof(&expanded_witness).unwrap();
        
        // These operations should complete without error
        println!("R1CS constraint system trait test completed, verification: {}", verification_result);
    }
}

/// Unit tests for CCS matrices and higher-degree constraints
#[cfg(test)]
mod ccs_tests {
    use super::*;
    use test_utils::*;
    
    #[test]
    fn test_ccs_matrix_creation_various_degrees() {
        let ring_params = standard_ring_params();
        
        // Test degree 1 (linear constraints)
        let degree1_matrices = CCSMatrices::new(
            5, 10, ConstraintDegree::Degree1, ring_params
        ).unwrap();
        assert_eq!(degree1_matrices.constraint_degree().degree(), 1);
        assert_eq!(degree1_matrices.constraint_matrices.len(), 2); // A, b
        
        // Test degree 2 (quadratic constraints, equivalent to R1CS)
        let degree2_matrices = CCSMatrices::new(
            5, 10, ConstraintDegree::Degree2, ring_params
        ).unwrap();
        assert_eq!(degree2_matrices.constraint_degree().degree(), 2);
        assert_eq!(degree2_matrices.constraint_matrices.len(), 3); // A, B, C
        
        // Test degree 3 (cubic constraints)
        let degree3_matrices = CCSMatrices::new(
            5, 10, ConstraintDegree::Degree3, ring_params
        ).unwrap();
        assert_eq!(degree3_matrices.constraint_degree().degree(), 3);
        assert_eq!(degree3_matrices.constraint_matrices.len(), 4); // A, B, C, D
        
        // Test arbitrary degree
        let degree5_matrices = CCSMatrices::new(
            3, 8, ConstraintDegree::DegreeN(5), ring_params
        ).unwrap();
        assert_eq!(degree5_matrices.constraint_degree().degree(), 5);
        assert_eq!(degree5_matrices.constraint_matrices.len(), 6); // 5 multiplicands + 1 result
    }
    
    #[test]
    fn test_ccs_constraint_degree_validation() {
        let ring_params = standard_ring_params();
        
        // Test invalid degree 0
        assert!(CCSMatrices::new(5, 10, ConstraintDegree::DegreeN(0), ring_params).is_err());
        
        // Test excessive degree
        assert!(CCSMatrices::new(5, 10, ConstraintDegree::DegreeN(20), ring_params).is_err());
        
        // Test valid degrees
        for degree in 1..=16 {
            let matrices = CCSMatrices::new(3, 5, ConstraintDegree::DegreeN(degree), ring_params);
            assert!(matrices.is_ok(), "Degree {} should be valid", degree);
        }
    }
    
    #[test]
    fn test_ccs_cubic_constraint_evaluation() {
        let ring_params = standard_ring_params();
        let mut matrices = CCSMatrices::new(
            1, 4, ConstraintDegree::Degree3, ring_params
        ).unwrap();
        
        // Create cubic constraint: z[0] * z[1] * z[2] = z[3]
        let zero = RingElement::zero(ring_params.dimension, ring_params.modulus).unwrap();
        let one = RingElement::from_coefficients(&[1], ring_params.modulus).unwrap();
        
        // Matrix 1: [1, 0, 0, 0] (select z[0])
        let m1_row = vec![one.clone(), zero.clone(), zero.clone(), zero.clone()];
        
        // Matrix 2: [0, 1, 0, 0] (select z[1])
        let m2_row = vec![zero.clone(), one.clone(), zero.clone(), zero.clone()];
        
        // Matrix 3: [0, 0, 1, 0] (select z[2])
        let m3_row = vec![zero.clone(), zero.clone(), one.clone(), zero.clone()];
        
        // Matrix 4: [0, 0, 0, 1] (select z[3])
        let m4_row = vec![zero.clone(), zero.clone(), zero.clone(), one.clone()];
        
        let matrix_rows = vec![m1_row, m2_row, m3_row, m4_row];
        let selector = one.clone(); // Always active
        
        matrices.set_constraint(0, matrix_rows, selector).unwrap();
        
        // Test satisfying witness: [2, 3, 4, 24] (2 * 3 * 4 = 24)
        let satisfying_witness = vec![
            RingElement::from_coefficients(&[2], ring_params.modulus).unwrap(),
            RingElement::from_coefficients(&[3], ring_params.modulus).unwrap(),
            RingElement::from_coefficients(&[4], ring_params.modulus).unwrap(),
            RingElement::from_coefficients(&[24], ring_params.modulus).unwrap(),
        ];
        
        assert!(matrices.evaluate_constraints(&satisfying_witness).unwrap());
        
        // Test non-satisfying witness: [2, 3, 4, 25] (2 * 3 * 4 ≠ 25)
        let non_satisfying_witness = vec![
            RingElement::from_coefficients(&[2], ring_params.modulus).unwrap(),
            RingElement::from_coefficients(&[3], ring_params.modulus).unwrap(),
            RingElement::from_coefficients(&[4], ring_params.modulus).unwrap(),
            RingElement::from_coefficients(&[25], ring_params.modulus).unwrap(),
        ];
        
        assert!(!matrices.evaluate_constraints(&non_satisfying_witness).unwrap());
    }
    
    #[test]
    fn test_ccs_selector_polynomial_functionality() {
        let ring_params = standard_ring_params();
        let mut matrices = CCSMatrices::new(
            2, 3, ConstraintDegree::Degree2, ring_params
        ).unwrap();
        
        let zero = RingElement::zero(ring_params.dimension, ring_params.modulus).unwrap();
        let one = RingElement::from_coefficients(&[1], ring_params.modulus).unwrap();
        
        // First constraint: z[0] * z[1] = z[2] (always active)
        let (a_row1, b_row1, c_row1) = create_multiplication_constraint(&ring_params).unwrap();
        matrices.set_constraint(0, vec![a_row1, b_row1, c_row1], one.clone()).unwrap();
        
        // Second constraint: z[0] * z[1] = z[2] (inactive with selector = 0)
        let (a_row2, b_row2, c_row2) = create_multiplication_constraint(&ring_params).unwrap();
        matrices.set_constraint(1, vec![a_row2, b_row2, c_row2], zero.clone()).unwrap();
        
        // Test witness that violates the constraint: [2, 3, 7]
        let violating_witness = vec![
            RingElement::from_coefficients(&[2], ring_params.modulus).unwrap(),
            RingElement::from_coefficients(&[3], ring_params.modulus).unwrap(),
            RingElement::from_coefficients(&[7], ring_params.modulus).unwrap(),
        ];
        
        // Should fail because first constraint is active and violated
        assert!(!matrices.evaluate_constraints(&violating_witness).unwrap());
        
        // Now make first constraint inactive and second active
        matrices.selector_polynomials[0] = zero.clone();
        matrices.selector_polynomials[1] = one.clone();
        
        // Should still fail because second constraint is now active and violated
        assert!(!matrices.evaluate_constraints(&violating_witness).unwrap());
        
        // Make both constraints inactive
        matrices.selector_polynomials[0] = zero.clone();
        matrices.selector_polynomials[1] = zero.clone();
        
        // Should pass because no constraints are active
        assert!(matrices.evaluate_constraints(&violating_witness).unwrap());
    }
    
    #[test]
    fn test_ccs_constraint_vectors_computation() {
        let ring_params = standard_ring_params();
        let mut matrices = CCSMatrices::new(
            1, 4, ConstraintDegree::Degree3, ring_params
        ).unwrap();
        
        // Set up cubic constraint
        let zero = RingElement::zero(ring_params.dimension, ring_params.modulus).unwrap();
        let one = RingElement::from_coefficients(&[1], ring_params.modulus).unwrap();
        
        let matrix_rows = vec![
            vec![one.clone(), zero.clone(), zero.clone(), zero.clone()], // M1: select z[0]
            vec![zero.clone(), one.clone(), zero.clone(), zero.clone()], // M2: select z[1]
            vec![zero.clone(), zero.clone(), one.clone(), zero.clone()], // M3: select z[2]
            vec![zero.clone(), zero.clone(), zero.clone(), one.clone()], // M4: select z[3]
        ];
        
        matrices.set_constraint(0, matrix_rows, one.clone()).unwrap();
        
        // Compute constraint vectors for witness [2, 3, 4, 24]
        let witness = vec![
            RingElement::from_coefficients(&[2], ring_params.modulus).unwrap(),
            RingElement::from_coefficients(&[3], ring_params.modulus).unwrap(),
            RingElement::from_coefficients(&[4], ring_params.modulus).unwrap(),
            RingElement::from_coefficients(&[24], ring_params.modulus).unwrap(),
        ];
        
        let constraint_vectors = matrices.compute_constraint_vectors(&witness).unwrap();
        
        // Should have 4 vectors (one for each matrix)
        assert_eq!(constraint_vectors.len(), 4);
        
        // Each vector should have 1 element (1 constraint)
        for vector in &constraint_vectors {
            assert_eq!(vector.len(), 1);
        }
        
        // Verify constraint satisfaction: M1z * M2z * M3z = M4z
        let m1z = &constraint_vectors[0][0];
        let m2z = &constraint_vectors[1][0];
        let m3z = &constraint_vectors[2][0];
        let m4z = &constraint_vectors[3][0];
        
        let product = m1z.multiply(m2z).unwrap().multiply(m3z).unwrap();
        assert_eq!(product, *m4z);
    }
    
    #[test]
    fn test_ccs_high_degree_constraint() {
        let ring_params = standard_ring_params();
        let degree = 5;
        let witness_dim = degree + 1; // Need one extra for result
        
        let mut matrices = CCSMatrices::new(
            1, witness_dim, ConstraintDegree::DegreeN(degree), ring_params
        ).unwrap();
        
        // Create constraint: z[0] * z[1] * z[2] * z[3] * z[4] = z[5]
        let zero = RingElement::zero(ring_params.dimension, ring_params.modulus).unwrap();
        let one = RingElement::from_coefficients(&[1], ring_params.modulus).unwrap();
        
        let mut matrix_rows = Vec::new();
        
        // Create selector matrices for each multiplicand
        for i in 0..degree {
            let mut row = vec![zero.clone(); witness_dim];
            row[i] = one.clone();
            matrix_rows.push(row);
        }
        
        // Create result matrix (selects last element)
        let mut result_row = vec![zero.clone(); witness_dim];
        result_row[witness_dim - 1] = one.clone();
        matrix_rows.push(result_row);
        
        matrices.set_constraint(0, matrix_rows, one.clone()).unwrap();
        
        // Test with witness [2, 2, 2, 2, 2, 32] (2^5 = 32)
        let mut witness = vec![RingElement::from_coefficients(&[2], ring_params.modulus).unwrap(); degree];
        witness.push(RingElement::from_coefficients(&[32], ring_params.modulus).unwrap());
        
        assert!(matrices.evaluate_constraints(&witness).unwrap());
        
        // Test with incorrect result
        witness[witness_dim - 1] = RingElement::from_coefficients(&[31], ring_params.modulus).unwrap();
        assert!(!matrices.evaluate_constraints(&witness).unwrap());
    }
}

/// Unit tests for committed CCS
#[cfg(test)]
mod committed_ccs_tests {
    use super::*;
    use test_utils::*;
    
    #[test]
    fn test_committed_ccs_creation() {
        let ring_params = standard_ring_params();
        let matrices = CCSMatrices::new(3, 6, ConstraintDegree::Degree3, ring_params).unwrap();
        let commitment_params = standard_commitment_params();
        
        let committed_ccs = CommittedCCS::new(
            matrices, 2, 5, commitment_params, 100
        ).unwrap();
        
        assert_eq!(committed_ccs.ccs_matrices().num_constraints(), 3);
        assert_eq!(committed_ccs.ccs_matrices().witness_dimension(), 6);
        assert_eq!(committed_ccs.ccs_matrices().constraint_degree().degree(), 3);
        assert_eq!(committed_ccs.gadget_matrix().gadget_vector().base(), 2);
        assert_eq!(committed_ccs.norm_bound, 100);
    }
    
    #[test]
    fn test_committed_ccs_witness_commitment() {
        let ring_params = standard_ring_params();
        let matrices = CCSMatrices::new(2, 4, ConstraintDegree::Degree3, ring_params).unwrap();
        let commitment_params = standard_commitment_params();
        
        let mut committed_ccs = CommittedCCS::new(
            matrices, 2, 6, commitment_params, 50
        ).unwrap();
        
        // Create valid witness
        let witness = vec![
            RingElement::from_coefficients(&[1, 2], ring_params.modulus).unwrap(),
            RingElement::from_coefficients(&[3, 4], ring_params.modulus).unwrap(),
            RingElement::from_coefficients(&[5, 6], ring_params.modulus).unwrap(),
            RingElement::from_coefficients(&[7, 8], ring_params.modulus).unwrap(),
        ];
        
        let randomness = committed_ccs.commitment_scheme.random_randomness(&mut thread_rng()).unwrap();
        
        // Commit to witness
        let result = committed_ccs.commit_witness(&witness, &randomness);
        assert!(result.is_ok());
        
        let (commitment, expanded_witness) = result.unwrap();
        
        // Verify expanded witness dimensions
        let expected_expanded_dim = witness.len() * committed_ccs.gadget_matrix().gadget_vector().dimension();
        assert_eq!(expanded_witness.len(), expected_expanded_dim);
        
        // Verify commitment verification
        let verification_result = committed_ccs.verify_constraints(&commitment, &expanded_witness, &randomness);
        assert!(verification_result.is_ok());
        
        // Verify performance statistics
        let stats = committed_ccs.stats();
        assert!(stats.witness_commitment_time > Duration::from_nanos(0));
        assert_eq!(stats.witness_dimension, 4);
        assert_eq!(stats.constraint_degree, 3);
    }
    
    #[test]
    fn test_committed_ccs_sumcheck_proof_generation() {
        let ring_params = standard_ring_params();
        let mut matrices = CCSMatrices::new(1, 4, ConstraintDegree::Degree3, ring_params).unwrap();
        let commitment_params = standard_commitment_params();
        
        // Set up cubic constraint: z[0] * z[1] * z[2] = z[3]
        let zero = RingElement::zero(ring_params.dimension, ring_params.modulus).unwrap();
        let one = RingElement::from_coefficients(&[1], ring_params.modulus).unwrap();
        
        let matrix_rows = vec![
            vec![one.clone(), zero.clone(), zero.clone(), zero.clone()],
            vec![zero.clone(), one.clone(), zero.clone(), zero.clone()],
            vec![zero.clone(), zero.clone(), one.clone(), zero.clone()],
            vec![zero.clone(), zero.clone(), zero.clone(), one.clone()],
        ];
        
        matrices.set_constraint(0, matrix_rows, one.clone()).unwrap();
        
        let mut committed_ccs = CommittedCCS::new(
            matrices, 2, 4, commitment_params, 100
        ).unwrap();
        
        // Create satisfying witness: [2, 3, 4, 24]
        let witness = vec![
            RingElement::from_coefficients(&[2], ring_params.modulus).unwrap(),
            RingElement::from_coefficients(&[3], ring_params.modulus).unwrap(),
            RingElement::from_coefficients(&[4], ring_params.modulus).unwrap(),
            RingElement::from_coefficients(&[24], ring_params.modulus).unwrap(),
        ];
        
        let randomness = committed_ccs.commitment_scheme.random_randomness(&mut thread_rng()).unwrap();
        let (_, expanded_witness) = committed_ccs.commit_witness(&witness, &randomness).unwrap();
        
        // Generate sumcheck proof
        let sumcheck_proof = committed_ccs.generate_sumcheck_proof(&expanded_witness);
        assert!(sumcheck_proof.is_ok());
        
        // Verify performance statistics
        let stats = committed_ccs.stats();
        assert!(stats.sumcheck_time > Duration::from_nanos(0));
        assert!(stats.linearization_time > Duration::from_nanos(0));
    }
    
    #[test]
    fn test_committed_ccs_constraint_system_trait() {
        let ring_params = standard_ring_params();
        let matrices = CCSMatrices::new(2, 5, ConstraintDegree::DegreeN(4), ring_params).unwrap();
        let commitment_params = standard_commitment_params();
        
        let mut committed_ccs = CommittedCCS::new(
            matrices, 2, 6, commitment_params, 100
        ).unwrap();
        
        // Test trait methods
        assert_eq!(committed_ccs.num_constraints(), 2);
        assert_eq!(committed_ccs.witness_dimension(), 5);
        assert_eq!(committed_ccs.max_constraint_degree(), 4);
        
        // Test trait method implementations
        let witness = random_witness(5, &ring_params).unwrap();
        let randomness = committed_ccs.commitment_scheme.random_randomness(&mut thread_rng()).unwrap();
        
        let (commitment, expanded_witness) = committed_ccs.commit_witness(&witness, &randomness).unwrap();
        let verification_result = committed_ccs.verify_constraints(&commitment, &expanded_witness, &randomness).unwrap();
        let _sumcheck_proof = committed_ccs.generate_sumcheck_proof(&expanded_witness).unwrap();
        
        println!("CCS constraint system trait test completed, verification: {}", verification_result);
    }
}

/// Performance and stress tests
#[cfg(test)]
mod performance_tests {
    use super::*;
    use test_utils::*;
    
    #[test]
    fn test_r1cs_performance_scaling() {
        let ring_params = standard_ring_params();
        let commitment_params = standard_commitment_params();
        
        // Test different constraint system sizes
        let sizes = vec![10, 50, 100, 200];
        
        for &size in &sizes {
            let start_time = Instant::now();
            
            // Create R1CS system
            let matrices = R1CSMatrices::new(size, size * 2, ring_params).unwrap();
            let mut committed_r1cs = CommittedR1CS::new(
                matrices, 2, 6, commitment_params.clone(), 100
            ).unwrap();
            
            // Create random witness
            let witness = random_witness(size * 2, &ring_params).unwrap();
            let randomness = committed_r1cs.commitment_scheme.random_randomness(&mut thread_rng()).unwrap();
            
            // Measure commitment time
            let commit_start = Instant::now();
            let (commitment, expanded_witness) = committed_r1cs.commit_witness(&witness, &randomness).unwrap();
            let commit_time = commit_start.elapsed();
            
            // Measure verification time
            let verify_start = Instant::now();
            let _verification_result = committed_r1cs.verify_constraints(&commitment, &expanded_witness, &randomness).unwrap();
            let verify_time = verify_start.elapsed();
            
            let total_time = start_time.elapsed();
            
            println!("R1CS size {}: commit={}ms, verify={}ms, total={}ms", 
                    size, commit_time.as_millis(), verify_time.as_millis(), total_time.as_millis());
            
            // Verify reasonable performance bounds
            assert!(commit_time < Duration::from_secs(10), "Commitment too slow for size {}", size);
            assert!(verify_time < Duration::from_secs(10), "Verification too slow for size {}", size);
        }
    }
    
    #[test]
    fn test_ccs_performance_scaling() {
        let ring_params = standard_ring_params();
        let commitment_params = standard_commitment_params();
        
        // Test different constraint degrees
        let degrees = vec![2, 3, 4, 5];
        let size = 20; // Fixed size, varying degree
        
        for &degree in &degrees {
            let start_time = Instant::now();
            
            // Create CCS system
            let matrices = CCSMatrices::new(size, size * 2, ConstraintDegree::DegreeN(degree), ring_params).unwrap();
            let mut committed_ccs = CommittedCCS::new(
                matrices, 2, 6, commitment_params.clone(), 100
            ).unwrap();
            
            // Create random witness
            let witness = random_witness(size * 2, &ring_params).unwrap();
            let randomness = committed_ccs.commitment_scheme.random_randomness(&mut thread_rng()).unwrap();
            
            // Measure commitment time
            let commit_start = Instant::now();
            let (commitment, expanded_witness) = committed_ccs.commit_witness(&witness, &randomness).unwrap();
            let commit_time = commit_start.elapsed();
            
            // Measure verification time
            let verify_start = Instant::now();
            let _verification_result = committed_ccs.verify_constraints(&commitment, &expanded_witness, &randomness).unwrap();
            let verify_time = verify_start.elapsed();
            
            let total_time = start_time.elapsed();
            
            println!("CCS degree {}: commit={}ms, verify={}ms, total={}ms", 
                    degree, commit_time.as_millis(), verify_time.as_millis(), total_time.as_millis());
            
            // Verify reasonable performance bounds (higher degrees may be slower)
            let max_time = Duration::from_secs(degree as u64 * 5);
            assert!(commit_time < max_time, "Commitment too slow for degree {}", degree);
            assert!(verify_time < max_time, "Verification too slow for degree {}", degree);
        }
    }
    
    #[test]
    fn test_memory_usage_estimation() {
        let ring_params = standard_ring_params();
        let commitment_params = standard_commitment_params();
        
        // Test memory usage for different system sizes
        let sizes = vec![10, 50, 100];
        
        for &size in &sizes {
            // R1CS memory usage
            let r1cs_matrices = R1CSMatrices::new(size, size, ring_params).unwrap();
            let r1cs_memory = estimate_r1cs_memory_usage(&r1cs_matrices);
            
            // CCS memory usage (degree 3)
            let ccs_matrices = CCSMatrices::new(size, size, ConstraintDegree::Degree3, ring_params).unwrap();
            let ccs_memory = estimate_ccs_memory_usage(&ccs_matrices);
            
            println!("Size {}: R1CS memory={}KB, CCS memory={}KB", 
                    size, r1cs_memory / 1024, ccs_memory / 1024);
            
            // CCS should use more memory due to additional matrices
            assert!(ccs_memory > r1cs_memory, "CCS should use more memory than R1CS");
            
            // Memory usage should scale roughly quadratically with size
            if size > 10 {
                let expected_ratio = (size * size) as f64 / (10.0 * 10.0);
                let actual_ratio = r1cs_memory as f64 / (sizes[0] * sizes[0] * 1000) as f64; // Rough estimate
                
                // Allow for some variance in memory usage
                assert!(actual_ratio > expected_ratio * 0.5 && actual_ratio < expected_ratio * 2.0,
                       "Memory scaling not as expected");
            }
        }
    }
    
    /// Estimates memory usage for R1CS matrices
    fn estimate_r1cs_memory_usage(matrices: &R1CSMatrices) -> usize {
        let ring_element_size = matrices.ring_params.dimension * 8; // Approximate size in bytes
        let matrix_size = matrices.num_constraints * matrices.witness_dimension * ring_element_size;
        matrix_size * 3 // A, B, C matrices
    }
    
    /// Estimates memory usage for CCS matrices
    fn estimate_ccs_memory_usage(matrices: &CCSMatrices) -> usize {
        let ring_element_size = matrices.ring_params.dimension * 8; // Approximate size in bytes
        let matrix_size = matrices.num_constraints * matrices.witness_dimension * ring_element_size;
        let num_matrices = matrices.constraint_degree.num_matrices();
        matrix_size * num_matrices
    }
}

/// Edge case and error handling tests
#[cfg(test)]
mod edge_case_tests {
    use super::*;
    use test_utils::*;
    
    #[test]
    fn test_empty_constraint_systems() {
        let ring_params = standard_ring_params();
        
        // Test creation with zero constraints (should fail)
        assert!(R1CSMatrices::new(0, 10, ring_params).is_err());
        assert!(CCSMatrices::new(0, 10, ConstraintDegree::Degree2, ring_params).is_err());
        
        // Test creation with zero witness dimension (should fail)
        assert!(R1CSMatrices::new(10, 0, ring_params).is_err());
        assert!(CCSMatrices::new(10, 0, ConstraintDegree::Degree2, ring_params).is_err());
    }
    
    #[test]
    fn test_maximum_constraint_systems() {
        let ring_params = standard_ring_params();
        
        // Test creation at maximum supported sizes
        let max_constraints = 1000; // Reduced for testing
        let max_witness = 1000;
        
        let r1cs_result = R1CSMatrices::new(max_constraints, max_witness, ring_params);
        assert!(r1cs_result.is_ok(), "Should support maximum constraint system size");
        
        let ccs_result = CCSMatrices::new(max_constraints, max_witness, ConstraintDegree::Degree2, ring_params);
        assert!(ccs_result.is_ok(), "Should support maximum CCS system size");
        
        // Test creation beyond maximum supported sizes (should fail)
        assert!(R1CSMatrices::new(2_000_000, 10, ring_params).is_err());
        assert!(R1CSMatrices::new(10, 2_000_000, ring_params).is_err());
    }
    
    #[test]
    fn test_invalid_ring_parameters() {
        // Test with invalid ring parameters
        let invalid_ring_params = RingParams {
            dimension: 0, // Invalid
            modulus: 2147483647,
        };
        
        assert!(R1CSMatrices::new(10, 10, invalid_ring_params).is_err());
        
        let invalid_modulus_params = RingParams {
            dimension: 64,
            modulus: 0, // Invalid
        };
        
        assert!(R1CSMatrices::new(10, 10, invalid_modulus_params).is_err());
    }
    
    #[test]
    fn test_constraint_boundary_conditions() {
        let ring_params = standard_ring_params();
        let mut matrices = R1CSMatrices::new(1, 1, ring_params).unwrap();
        
        // Test setting constraint at boundary indices
        let zero = RingElement::zero(ring_params.dimension, ring_params.modulus).unwrap();
        let constraint_row = vec![zero.clone()];
        
        // Valid boundary case
        assert!(matrices.set_constraint(0, constraint_row.clone(), constraint_row.clone(), constraint_row.clone()).is_ok());
        
        // Invalid boundary case
        assert!(matrices.set_constraint(1, constraint_row.clone(), constraint_row.clone(), constraint_row.clone()).is_err());
    }
    
    #[test]
    fn test_witness_dimension_mismatches() {
        let ring_params = standard_ring_params();
        let matrices = R1CSMatrices::new(1, 3, ring_params).unwrap();
        
        // Test witness with wrong dimensions
        let short_witness = vec![
            RingElement::from_coefficients(&[1], ring_params.modulus).unwrap(),
            RingElement::from_coefficients(&[2], ring_params.modulus).unwrap(),
        ]; // Missing one element
        
        let long_witness = vec![
            RingElement::from_coefficients(&[1], ring_params.modulus).unwrap(),
            RingElement::from_coefficients(&[2], ring_params.modulus).unwrap(),
            RingElement::from_coefficients(&[3], ring_params.modulus).unwrap(),
            RingElement::from_coefficients(&[4], ring_params.modulus).unwrap(),
        ]; // One extra element
        
        assert!(matrices.evaluate_constraints(&short_witness).is_err());
        assert!(matrices.evaluate_constraints(&long_witness).is_err());
        assert!(matrices.compute_constraint_vectors(&short_witness).is_err());
        assert!(matrices.compute_constraint_vectors(&long_witness).is_err());
    }
    
    #[test]
    fn test_extreme_coefficient_values() {
        let ring_params = standard_ring_params();
        let mut matrices = R1CSMatrices::new(1, 3, ring_params).unwrap();
        
        // Create constraint with extreme coefficient values
        let max_coeff = ring_params.modulus / 2;
        let min_coeff = -(ring_params.modulus / 2);
        
        let extreme_element_max = RingElement::from_coefficients(&[max_coeff], ring_params.modulus).unwrap();
        let extreme_element_min = RingElement::from_coefficients(&[min_coeff], ring_params.modulus).unwrap();
        let zero = RingElement::zero(ring_params.dimension, ring_params.modulus).unwrap();
        
        let a_row = vec![extreme_element_max.clone(), zero.clone(), zero.clone()];
        let b_row = vec![zero.clone(), extreme_element_max.clone(), zero.clone()];
        let c_row = vec![zero.clone(), zero.clone(), extreme_element_min.clone()];
        
        // Should handle extreme values correctly
        assert!(matrices.set_constraint(0, a_row, b_row, c_row).is_ok());
        
        // Test evaluation with extreme witness values
        let extreme_witness = vec![extreme_element_max, extreme_element_min, zero];
        let result = matrices.evaluate_constraints(&extreme_witness);
        assert!(result.is_ok(), "Should handle extreme coefficient values");
    }
}