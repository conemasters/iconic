use iconic_bench::mip::gen_knapsack;
use iconic_mip::{heuristics, MipSettings};
fn main() {
    let (prob, _) = gen_knapsack(500, 4444 + 500);
    let settings = MipSettings::<f64>::default();
    let t0 = std::time::Instant::now();
    let res = heuristics::feasibility_pump(&prob, &settings, 5);
    println!(
        "pump total: {:.3}s result={:?}",
        t0.elapsed().as_secs_f64(),
        res.is_some()
    );
}
