#[cfg(test)]
mod tests {
    use super::*;
    use ark_bls12_381::{Bls12_381, Fr};
    use ark_std::test_rng;
    use proptest::prelude::*;
    use rand::thread_rng;
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    #[test]
    fn test_lattice_operations() {
        let mut rng = test_rng();
        let params = LatticeParams {
            q: 97,  // Small prime for testing
            n: 4,
            sigma: 3.0,
            beta: 2.0,
        };

        let point1 = LatticePoint::random(&params, &mut rng);
        let point2 = LatticePoint::random(&params, &mut rng);

        // Test addition
        let sum = point1.clone().add_mod(&point2, params.q);
        assert!(sum.coordinates.iter().all(|&x| x >= 0 && x < params.q));

        // Test subtraction
        let diff = point1.clone().sub_mod(&point2, params.q);
        assert!(diff.coordinates.iter().all(|&x| x >= 0 && x < params.q));

        // Test scalar multiplication
        let scalar = 5;
        let scaled = point1.clone().scalar_mul_mod(scalar, params.q);
        assert!(scaled.coordinates.iter().all(|&x| x >= 0 && x < params.q));
    }

    #[test]
    fn test_lattice_matrix_operations() {
        let params = LatticeParams {
            q: 97,
            n: 4,
            sigma: 3.0,
            beta: 2.0,
        };

        let mut rng = thread_rng();
        let matrix1 = LatticeMatrix::random(4, 4, &params, &mut rng);
        let matrix2 = LatticeMatrix::random(4, 4, &params, &mut rng);

        // Test matrix multiplication
        let product = matrix1.mul(&matrix2).unwrap();
        assert_eq!(product.rows, 4);
        assert_eq!(product.cols, 4);

        // Test transpose
        let transposed = matrix1.transpose();
        assert_eq!(transposed.rows, matrix1.cols);
        assert_eq!(transposed.cols, matrix1.rows);

        // Test Gram-Schmidt orthogonalization
        let orthogonal = matrix1.gram_schmidt();
        for i in 0..orthogonal.rows {
            for j in (i + 1)..orthogonal.rows {
                let dot_product: i64 = (0..orthogonal.cols)
                    .map(|k| orthogonal.data[i][k] * orthogonal.data[j][k])
                    .sum();
                assert!(dot_product.abs() < params.q); // Should be close to 0
            }
        }
    }

    #[test]
    fn test_folding_scheme() {
        let params = LatticeParams {
            q: 97,
            n: 4,
            sigma: 3.0,
            beta: 2.0,
        };

        let basis_matrix = LatticeMatrix::random(
            params.n,
            params.n,
            &params,
            &mut thread_rng(),
        );

        let mut scheme = FoldingScheme::new(params.clone(), basis_matrix);
        let mut transcript = Transcript::new(b"test_folding");
        let mut rng = thread_rng();

        // Generate test points
        let points: Vec<LatticePoint> = (0..3)
            .map(|_| LatticePoint::random(&params, &mut rng))
            .collect();

        // Test proof generation
        let proof = scheme.prove(&points, &mut transcript, &mut rng).unwrap();

        // Test proof verification
        let mut verify_transcript = Transcript::new(b"test_folding");
        assert!(scheme.verify(&proof, &mut verify_transcript).unwrap());
    }

    #[test]
    fn test_commitment_scheme() {
        let mut rng = test_rng();
        let dimension = 10;
        let security_param = 128;

        // Setup
        let params = setup_lattice_fold::<Bls12_381, Fr, _>(dimension, security_param, &mut rng).unwrap();

        // Test vector commitment
        let vector = vec![Fr::rand(&mut rng); dimension];
        let randomness = Fr::rand(&mut rng);
        let commitment = params.commitment_scheme.commit_vector(&vector, randomness);

        assert!(params.commitment_scheme.vector_commitment.verify(
            commitment,
            &vector,
            randomness,
        ));

        // Test polynomial commitment
        let coefficients = vec![Fr::rand(&mut rng); dimension];
        let poly_commitment = params.commitment_scheme.commit_polynomial(&coefficients, randomness);

        assert!(params.commitment_scheme.polynomial_commitment.verify(
            poly_commitment,
            &coefficients,
            randomness,
        ));
    }

    proptest! {
        #[test]
        fn test_random_lattice_points(
            dimension in 4..16usize,
            num_points in 1..10usize,
        ) {
            let params = LatticeParams {
                q: 97,
                n: dimension,
                sigma: 3.0,
                beta: 2.0,
            };

            let basis_matrix = LatticeMatrix::random(
                params.n,
                params.n,
                &params,
                &mut thread_rng(),
            );

            let mut scheme = FoldingScheme::new(params.clone(), basis_matrix);
            let mut transcript = Transcript::new(b"test_folding");
            let mut rng = ChaCha20Rng::from_entropy();

            let points: Vec<LatticePoint> = (0..num_points)
                .map(|_| LatticePoint::random(&params, &mut rng))
                .collect();

            let proof = scheme.prove(&points, &mut transcript, &mut rng).unwrap();
            let mut verify_transcript = Transcript::new(b"test_folding");
            
            prop_assert!(scheme.verify(&proof, &mut verify_transcript).unwrap());
        }
    }
} 