pub mod corpuses;
pub mod brute_force;

use corpuses::CorpusData;
// use brute_force::{ BruteForceSpec, BruteForceOutput, FingerType, Thresholds, brute_force_all, };

#[allow(dead_code)]
fn good_heatmap_index(mut index: Vec<char>, corpus_data: &CorpusData) -> bool {
    index.sort_by(|a, b| corpus_data.get_symbol(a).partial_cmp(&corpus_data.get_symbol(b)).unwrap());
    corpus_data.get_symbol(&index[1]) < 1.3
}

fn main() -> Result<(), main_error::MainError> {
    let specs = brute_force::SpecConfig::from_toml("spec.toml");
    println!("{}", specs.brute_force_all());

    Ok(())
}
