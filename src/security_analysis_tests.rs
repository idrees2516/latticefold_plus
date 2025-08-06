/// Tests for security analysis implementation
/// 
/// This module contains comprehensive tests for the security analysis framework,
/// including unit tests for individual components and integration tests for
/// the complete security analysis workflow.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::security_analysis::*;
    use crate::commitment_transformation::*;
    use crate::cyclotomic_ring::RingElement;
    use crate::error::Result;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    /// Creates test parameters for security analysis
    fn create_test_params() -> Result<CommitmentTransformationParams> {
        CommitmentTransformationParams::new(
            128,    // kappa
            1024,   // ring_dimension
            2048,   // witness_dimension
            1000,   // norm_bound
            2_i64.pow(61) - 1, // modulus (NTT-friendly prime)
        )
    }

    /// Creates a test proof for security analysis
    fn create_test_proof(params: &CommitmentTransformationParams) -> Result<CommitmentTransformationProof> {
        // Create dummy witness
        let mut folding_witness = Vec::new();
        for _ in 0..params.witness_dimension {
            let element = RingElement::zero(params.ring_dimension, Some(params.modulus))?;
            folding_witness.push(element);
        }

        // Create dummy challenges
        let mut folding_challenges = Vec::new();
        for _ in 0..3 {
            let mut element = RingElement::zero(params.ring_dimension, Some(params.modulus))?;
            let coeffs = element.coefficients_mut();
            coeffs[0] = 1; // Set to non-zero value
            folding_challenges.push(element);
        }

        // Create dummy compressed data
        let mut compressed_data = Vec::new();
        for _ in 0..params.kappa {
            let element = RingElement::zero(params.ring_dimension, Some(params.modulus))?;
            compressed_data.push(element);
        }

        // Create dummy range check proof
        let range_check_proof = crate::range_check_protocol::RangeCheckProof::new(
            Vec::new(), // commitments
            Vec::new(), // responses
            Vec::new(), // challenges
        );

        // Create dummy decomposition proof
        let decomposition_proof = DecompositionProof::new(
            Vec::new(), // decomposition_commitment
            Vec::new(), // norm_bound_proof
            Vec::new(), // reconstruction_proof
            Vec::new(), // randomness
        );

        // Create dummy consistency proof
        let consistency_proof = crate::sumcheck_batching::BatchedSumcheckProof::new(
            Vec::new(), // polynomial_commitments
            Vec::new(), // evaluation_proofs
            Vec::new(), // batch_challenges
            0,          // num_variables
        );

        Ok(CommitmentTransformationProof {
            range_check_proof,
            decomposition_proof,
            consistency_proof,
            folding_witness,
            folding_challenges,
            compressed_data,
            params: params.clone(),
        })
    }

    #[test]
    fn test_security_analyzer_creation() -> Result<()> {
        let params = create_test_params()?;
        let analyzer = SecurityAnalyzer::new(params)?;
        
        // Test should pass if analyzer is created successfully
        Ok(())
    }

    #[test]
    fn test_norm_bound_verification() -> Result<()> {
        let params = create_test_params()?;
        let analyzer = SecurityAnalyzer::new(params)?;
        
        // Create test witness with small coefficients
        let mut witness = Vec::new();
        for _ in 0..params.witness_dimension {
            let mut element = RingElement::zero(params.ring_dimension, Some(params.modulus))?;
            let coeffs = element.coefficients_mut();
            coeffs[0] = 10; // Small coefficient within bounds
            witness.push(element);
        }
        
        // Verify norm bound
        let result = analyzer.verify_norm_bound_with_overflow_protection(&witness)?;
        assert!(result, "Small coefficients should satisfy norm bounds");
        
        Ok(())
    }

    #[test]
    fn test_norm_bound_violation() -> Result<()> {
        let params = create_test_params()?;
        let analyzer = SecurityAnalyzer::new(params)?;
        
        // Create test witness with large coefficients
        let mut witness = Vec::new();
        for _ in 0..params.witness_dimension {
            let mut element = RingElement::zero(params.ring_dimension, Some(params.modulus))?;
            let coeffs = element.coefficients_mut();
            coeffs[0] = params.norm_bound; // Coefficient at the bound (should fail)
            witness.push(element);
        }
        
        // Verify norm bound
        let result = analyzer.verify_norm_bound_with_overflow_protection(&witness)?;
        assert!(!result, "Large coefficients should violate norm bounds");
        
        Ok(())
    }

    #[test]
    fn test_knowledge_error_computation() -> Result<()> {
        let params = create_test_params()?;
        let mut analyzer = SecurityAnalyzer::new(params)?;
        let proof = create_test_proof(&params)?;
        
        // Compute knowledge error
        let knowledge_error = analyzer.compute_knowledge_error(&proof, None)?;
        
        // Verify error components are computed
        assert!(knowledge_error.binding_error >= 0.0);
        assert!(knowledge_error.consistency_error >= 0.0);
        assert!(knowledge_error.extractor_error >= 0.0);
        assert!(knowledge_error.range_proof_error >= 0.0);
        assert!(knowledge_error.sumcheck_error >= 0.0);
        
        // Verify total error is sum of components
        let expected_total = knowledge_error.binding_error 
            + knowledge_error.consistency_error
            + knowledge_error.extractor_error
            + knowledge_error.range_proof_error
            + knowledge_error.sumcheck_error;
        
        assert!((knowledge_error.total_knowledge_error - expected_total).abs() < 1e-10);
        
        Ok(())
    }

    #[test]
    fn test_extractor_properties() -> Result<()> {
        let params = create_test_params()?;
        let mut analyzer = SecurityAnalyzer::new(params)?;
        let proof = create_test_proof(&params)?;
        
        // Analyze extractor properties
        let extractor_analysis = analyzer.analyze_extractor_properties(&proof, None)?;
        
        // Verify extraction probability is reasonable
        assert!(extractor_analysis.extraction_probability >= 0.0);
        assert!(extractor_analysis.extraction_probability <= 1.0);
        
        // Verify expected attempts is positive
        assert!(extractor_analysis.expected_extraction_attempts > 0.0);
        
        // Verify runtime estimates are reasonable
        assert!(extractor_analysis.expected_runtime_ms > 0);
        assert!(extractor_analysis.worst_case_runtime_ms >= extractor_analysis.expected_runtime_ms);
        
        Ok(())
    }

    #[test]
    fn test_parameter_adequacy_assessment() -> Result<()> {
        let params = create_test_params()?;
        let analyzer = SecurityAnalyzer::new(params)?;
        
        // Assess parameter adequacy
        let adequacy = analyzer.assess_parameter_adequacy()?;
        
        // Verify security levels are computed
        assert!(adequacy.effective_security_bits > 0.0);
        assert!(adequacy.target_security_bits > 0.0);
        
        // Verify adequacy assessment is consistent
        let expected_adequate = adequacy.security_margin_bits >= 0.0;
        assert_eq!(adequacy.adequate_for_target, expected_adequate);
        
        Ok(())
    }

    #[test]
    fn test_constant_time_comparison() -> Result<()> {
        let params = create_test_params()?;
        let analyzer = SecurityAnalyzer::new(params)?;
        
        // Test constant-time comparison with various values
        assert!(analyzer.constant_time_less_than(5, 10));
        assert!(!analyzer.constant_time_less_than(10, 5));
        assert!(!analyzer.constant_time_less_than(10, 10));
        
        // Test with edge cases
        assert!(analyzer.constant_time_less_than(i64::MIN, 0));
        assert!(!analyzer.constant_time_less_than(i64::MAX, 0));
        assert!(analyzer.constant_time_less_than(-1, 1));
        
        Ok(())
    }

    #[test]
    fn test_witness_statistics() -> Result<()> {
        let params = create_test_params()?;
        let analyzer = SecurityAnalyzer::new(params)?;
        
        // Create test witness with known statistics
        let mut witness = Vec::new();
        for i in 0..10 {
            let mut element = RingElement::zero(params.ring_dimension, Some(params.modulus))?;
            let coeffs = element.coefficients_mut();
            coeffs[0] = i as i64; // Coefficients 0, 1, 2, ..., 9
            witness.push(element);
        }
        
        // Compute statistics
        let stats = analyzer.compute_witness_statistics(&witness)?;
        
        // Verify mean is approximately 4.5 (average of 0..9)
        assert!((stats.mean_coefficient - 4.5).abs() < 0.1);
        
        // Verify standard deviation is reasonable
        assert!(stats.coefficient_std_dev > 0.0);
        
        // Verify entropy is positive
        assert!(stats.entropy_bits > 0.0);
        
        Ok(())
    }

    #[test]
    fn test_edge_case_handling() -> Result<()> {
        let params = create_test_params()?;
        let analyzer = SecurityAnalyzer::new(params)?;
        
        // Test zero coefficient handling
        let zero_test = analyzer.test_zero_coefficient_handling()?;
        assert!(zero_test.test_passed, "Zero coefficient handling should pass");
        
        // Test maximum coefficient handling
        let max_test = analyzer.test_max_coefficient_handling()?;
        assert!(max_test.test_passed, "Maximum coefficient handling should pass");
        
        // Test minimum coefficient handling
        let min_test = analyzer.test_min_coefficient_handling()?;
        assert!(min_test.test_passed, "Minimum coefficient handling should pass");
        
        Ok(())
    }

    #[test]
    fn test_security_analysis_caching() -> Result<()> {
        let params = create_test_params()?;
        let mut analyzer = SecurityAnalyzer::new(params)?;
        let proof = create_test_proof(&params)?;
        
        // Perform analysis twice
        let result1 = analyzer.analyze_complete_security(&proof, None)?;
        let result2 = analyzer.analyze_complete_security(&proof, None)?;
        
        // Results should be identical (from cache)
        assert_eq!(result1.analysis_timestamp, result2.analysis_timestamp);
        
        // Check cache statistics
        let stats = analyzer.analysis_stats.lock().unwrap();
        assert!(stats.cache_hits > 0);
        
        Ok(())
    }

    #[test]
    fn test_malicious_prover_resistance_basic() -> Result<()> {
        let params = create_test_params()?;
        let mut analyzer = SecurityAnalyzer::new(params)?;
        let proof = create_test_proof(&params)?;
        
        // Test a single adversarial strategy
        let strategy_result = analyzer.test_adversarial_strategy(
            "Invalid Witness Attack",
            "Test attack with invalid witness",
            &proof,
        )?;
        
        // Verify test was executed
        assert!(strategy_result.test_cases > 0);
        assert!(strategy_result.average_attack_time_ms >= 0);
        
        // Protocol should be resistant to basic attacks
        assert!(strategy_result.protocol_resistant);
        
        Ok(())
    }
}