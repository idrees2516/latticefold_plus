pub mod error;
pub mod lattice;
pub mod folding;

pub use error::{LatticeFoldError, Result};
pub use lattice::{LatticePoint, LatticeParams, LatticeMatrix};
pub use folding::{FoldingScheme, FoldingCommitment, FoldingProof};

#[cfg(test)]
mod tests {
    use super::*;
    use rand::thread_rng;
    use merlin::Transcript;

    #[test]
    fn test_lattice_operations() {
        let params = LatticeParams {
            q: 97,  // Small prime for testing
            n: 4,
            sigma: 3.0,
            beta: 2.0,
        };

        let mut rng = thread_rng();
        let point1 = LatticePoint::random(&params, &mut rng);
        let point2 = LatticePoint::random(&params, &mut rng);

        let sum = point1.clone().add_mod(&point2, params.q);
        assert!(sum.coordinates.iter().all(|&x| x >= 0 && x < params.q));

        let diff = point1.clone().sub_mod(&point2, params.q);
        assert!(diff.coordinates.iter().all(|&x| x >= 0 && x < params.q));

        let scalar = 5;
        let scaled = point1.clone().scalar_mul_mod(scalar, params.q);
        assert!(scaled.coordinates.iter().all(|&x| x >= 0 && x < params.q));
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

        // Generate some test points
        let points: Vec<LatticePoint> = (0..3)
            .map(|_| LatticePoint::random(&params, &mut rng))
            .collect();

        // Generate proof
        let proof = scheme.prove(&points, &mut transcript, &mut rng);

        // Verify proof
        let mut verify_transcript = Transcript::new(b"test_folding");
        assert!(scheme.verify(&proof, &mut verify_transcript));
    }

    #[test]
    fn test_gram_schmidt() {
        let params = LatticeParams {
            q: 97,
            n: 4,
            sigma: 3.0,
            beta: 2.0,
        };

        let matrix = LatticeMatrix::random(
            params.n,
            params.n,
            &params,
            &mut thread_rng(),
        );

        let orthogonal = matrix.gram_schmidt();
        
        // Verify orthogonality (approximately)
        for i in 0..params.n {
            for j in (i + 1)..params.n {
                let dot_product: i64 = orthogonal.data[i].iter()
                    .zip(orthogonal.data[j].iter())
                    .map(|(&x, &y)| x * y)
                    .sum();
                assert!(dot_product.abs() < params.q); // Should be close to 0
            }
        }
    }
}
