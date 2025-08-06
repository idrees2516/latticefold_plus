// Simple compilation test for monomial module
use latticefold_plus::monomial::{Monomial, MonomialSet, MonomialMembershipTester};
use latticefold_plus::cyclotomic_ring::RingElement;

fn main() {
    println!("Testing monomial compilation...");
    
    // Test basic monomial creation
    let m1 = Monomial::new(3);
    println!("Created monomial: {}", m1);
    
    // Test monomial set creation
    let ring_dim = 8;
    let modulus = Some(17);
    let monomial_set = MonomialSet::new(ring_dim, modulus).unwrap();
    println!("Created monomial set with cardinality: {}", monomial_set.cardinality());
    
    // Test membership tester creation
    let mut tester = MonomialMembershipTester::new(ring_dim, 17).unwrap();
    println!("Created membership tester");
    
    // Test zero polynomial
    let zero = RingElement::zero(ring_dim, modulus).unwrap();
    let is_zero_member = tester.test_membership(&zero).unwrap();
    println!("Zero polynomial membership: {}", is_zero_member);
    
    println!("All tests passed!");
}