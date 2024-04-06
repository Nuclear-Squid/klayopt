use std::io::Write;
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
    let output_file = std::fs::File::create("out.opt")?;
    let mut writer = std::io::BufWriter::new(output_file);

    let specs = brute_force::SpecConfig::from_toml("spec.toml");

    write!(writer, "{}", specs.brute_force_all())?;

    Ok(())
}
