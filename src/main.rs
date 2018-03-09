extern crate hll;
extern crate rand;

use hll::HLL;
use rand::Rng;
use std::collections::HashSet;

fn main() {
    let samples = 10 * 1_000_000;

    let mut rng = rand::thread_rng();
    let mut explicit = HashSet::new();
    let mut approx = HLL::new(16, 5);

    println!("n,explicit,hll,error");

    for i in 0..samples {
        let e: u64 = rng.gen();
        explicit.insert(e);
        approx.add_raw(e);

        if i % (5 * 10_000) == 0 {
            let stat = SetStat::from(i, &explicit, &approx);
            println!(
                "{},{},{},{:.06}",
                stat.n,
                stat.explicit,
                stat.approximate,
                stat.pct_error(),
            )
        }
    }

    let stat = SetStat::from(samples, &explicit, &approx);
    println!(
        "{},{},{},{:.06}",
        stat.n,
        stat.explicit,
        stat.approximate,
        stat.pct_error(),
    );
}

struct SetStat {
    n: usize,
    explicit: usize,
    approximate: f64,
}

impl SetStat {
    fn from<T: std::hash::Hash + Eq>(n: usize, explicit: &HashSet<T>, hll: &HLL) -> Self {
        Self {
            n: n,
            explicit: explicit.len(),
            approximate: hll.cardinality_estimate(),
        }
    }

    fn pct_error(&self) -> f64 {
        let explicit = self.explicit as f64;
        (explicit - self.approximate) / explicit
    }
}
