pub mod corpuses;
pub mod brute_force;

use corpuses::CorpusData;
use brute_force::{ BruteForceSpec, FingerType, Thresholds, brute_force_all, };

fn main() -> Result<(), main_error::MainError> {
    let corpus_fr = CorpusData::from_json("fr.json")?;
    let corpus_en = CorpusData::from_json("en.json")?;
    let all_corpuses = vec![&corpus_fr, &corpus_en];

    let index = BruteForceSpec {
        corpuses: all_corpuses.clone(),
        finger: FingerType::IndexFinger,
        must_have_char: String::from("td"),
        must_not_have_char: String::from("j"),
        thresholds: Thresholds {
            min_load: 14.,
            max_load: 19.,
            max_sfu: 0.1,
        },
    };

    let index2 = BruteForceSpec {
        corpuses: all_corpuses.clone(),
        finger: FingerType::IndexFinger,
        must_have_char: String::from("nh"),
        must_not_have_char: String::from(""),
        thresholds: Thresholds {
            min_load: 14.,
            max_load: 19.,
            max_sfu: 0.1,
        },
    };

    let finger = BruteForceSpec {
        corpuses: all_corpuses.clone(),
        finger: FingerType::RegularFinger,
        must_have_char: String::from("o"),
        must_not_have_char: String::from(""),
        thresholds: Thresholds {
            min_load: 12.,
            max_load: 19.,
            max_sfu: 0.1,
        },
    };

    println!("{}", brute_force_all([index, index2, finger]));

    Ok(())
}
