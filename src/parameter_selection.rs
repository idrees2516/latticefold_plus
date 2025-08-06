//! Parameter selection utility (Appendix C of the paper)
//! Provides a heuristic mapping from security-level + performance constraints
//! to concrete `LatticeParams` usable by the rest of the library.

use crate::error::{LatticeFoldError, Result};
use crate::lattice::LatticeParams;
use crate::quantum_resistance::{QuantumResistanceAnalyzer, SecurityLevel};

/// Constraints when selecting parameters.
#[derive(Clone, Debug)]
pub struct SelectionConstraints {
    /// Requested classical security bits (≥ 64)
    pub classical_security: usize,
    /// Maximum lattice dimension allowed (for performance)
    pub max_dimension: usize,
    /// Whether to favour power-of-two moduli q = 2^k + 1 (NTT-friendly)
    pub ntt_friendly: bool,
}

impl Default for SelectionConstraints {
    fn default() -> Self {
        Self {
            classical_security: 128,
            max_dimension: 2048,
            ntt_friendly: true,
        }
    }
}

/// Main entry-point – compute `LatticeParams` given constraints.
pub fn select_params(constraints: &SelectionConstraints) -> Result<LatticeParams> {
    // Map classical bits to predefined SecurityLevel buckets.
    let level = match constraints.classical_security {
        0..=127 => SecurityLevel::Medium,
        128..=191 => SecurityLevel::Medium,
        192..=255 => SecurityLevel::High,
        _ => SecurityLevel::VeryHigh,
    };

    let analyzer = QuantumResistanceAnalyzer::new();
    let qr = analyzer.get_params(level)?;
    let mut params = analyzer.create_lattice_params(&qr);

    // If dimension exceeds user cap, iteratively halve q and n until within bound.
    while params.n > constraints.max_dimension {
        if params.n <= 64 {
            return Err(LatticeFoldError::InvalidParameters(
                "Cannot satisfy max_dimension without breaking security".into(),
            ));
        }
        params.n /= 2;
        params.q = (params.q as f64 / 2.0).round() as i64 | 1; // keep odd prime power-of-twoish
    }

    // If NTT friendly requested, coerce q := 2^k + 1 >= current q.
    if constraints.ntt_friendly {
        let mut k = 1;
        while (1i64 << k) + 1 < params.q {
            k += 1;
        }
        params.q = (1i64 << k) + 1;
    }

    Ok(params)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_param_selection() {
        let constraints = SelectionConstraints::default();
        let params = select_params(&constraints).unwrap();
        assert!(params.n <= constraints.max_dimension);
    }
}
