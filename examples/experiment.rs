
use rand;

use hll::HLL;
use rand::Rng;

use std::cmp::Eq;
use std::collections::HashSet;
use std::hash::Hash;

fn main() {
    let samples = 10 * 1_000_000;
    let measure_every = 100_000;

    let mut rng = rand::thread_rng();
    let mut explicit = HashSet::new();
    let mut approx = HLL::new(16, 5);

    // print a header before starting
    println!("iterations,exact_cardinality,hll_cardinality,measurement_error");

    for i in 1..(samples + 1) {
        let e: u64 = rng.gen();
        explicit.insert(e);
        approx.add_raw(e);

        if i % measure_every == 0 {
            let Observation(n, e, a, err) = observe(i, &explicit, &approx);
            println!("{},{},{},{:0.6}", n, e, a, err);
        }
    }
}

struct Observation(usize, usize, f64, f64);

fn observe<T: Hash + Eq>(n: usize, explicit: &HashSet<T>, hll: &HLL) -> Observation {
    let exact = explicit.len();
    let approximate = hll.cardinality_estimate();
    Observation(
        n,
        exact,
        approximate,
        measurement_error(exact as f64, approximate),
    )
}

fn measurement_error(actual: f64, estimate: f64) -> f64 {
    (estimate - actual) / actual
}
