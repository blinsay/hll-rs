use hll::HLL;
use rand::Rng;

use std::collections::BTreeSet;

fn main() {
    let samples = 10 * 1_000_000;
    let measure_every = 500_000;

    let mut rng = rand::thread_rng();
    let mut explicit = BTreeSet::new();
    let mut approx: HLL<5, 16> = HLL::new();

    println!("iterations,exact_cardinality,exact_size,hll_cardinality,hll_size,measurement_error");

    for i in 1..(samples + 1) {
        let e: u64 = rng.gen();
        explicit.insert(e);
        approx.add_raw(e);

        if i % measure_every == 0 {
            let exact = explicit.len();
            let approximate = approx.cardinality().round();
            let measurement_error = measurement_error(exact as f64, approximate);
            let exact_size = explicit.len() * 8;
            let approx_size = approx.register_width() * approx.register_count();
            println!(
                "{},{},{},{},{},{:0.6}",
                i,
                exact,
                bytes_to_str(exact_size),
                approximate,
                bytes_to_str(approx_size),
                measurement_error
            );
        }
    }
}

fn bytes_to_str(bytes: usize) -> String {
    match bytes {
        b if b >= 1024 * 1024 * 1024 => format!("{}GiB", b / 1024 / 1024 / 1024),
        b if b >= 1024 * 1024 => format!("{}MiB", b / 1024 / 1024),
        b if b >= 1024 => format!("{}KiB", b / 1024),
        b => format!("{}B", b),
    }
}

fn measurement_error(actual: f64, estimate: f64) -> f64 {
    (estimate - actual) / actual
}
