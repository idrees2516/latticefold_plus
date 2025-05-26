use ark_ec::AffineCurve;
use ark_ff::Field;

/// A lattice folding instance containing a witness and public input
#[derive(Clone, Debug)]
pub struct LatticeFoldInstance<F: Field> {
    /// The witness vector
    pub witness: Vec<F>,
    /// The public input vector
    pub public_input: Vec<F>,
}

/// A final proof containing the commitment, response, and challenge
#[derive(Clone, Debug)]
pub struct FinalProof<G: AffineCurve, F: Field> {
    /// The commitment to the proof
    pub commitment: G,
    /// The response vector
    pub response: Vec<F>,
    /// The challenge value
    pub challenge: F,
}

/// Parameters for the lattice folding protocol
#[derive(Clone, Debug)]
pub struct LatticeFoldParams<G: AffineCurve, F: Field> {
    /// The dimension of the lattice
    pub dimension: usize,
    /// The modulus for the field
    pub modulus: F,
    /// The security parameter in bits
    pub security_param: usize,
    /// The commitment scheme parameters
    pub commitment_params: CommitmentParams,
    /// Phantom data for the curve type
    pub _phantom: std::marker::PhantomData<G>,
}

/// Parameters for the commitment scheme
#[derive(Clone, Debug)]
pub struct CommitmentParams {
    /// The number of generators
    pub num_generators: usize,
    /// The hiding parameter
    pub hiding_param: usize,
    /// The binding parameter
    pub binding_param: usize,
}

impl Default for CommitmentParams {
    fn default() -> Self {
        Self {
            num_generators: 128,
            hiding_param: 128,
            binding_param: 128,
        }
    }
} 