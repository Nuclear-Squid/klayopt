use std::path::Path;
use serde::Deserialize;
use itertools::Itertools;
use crate::corpuses::CorpusData;

#[derive(Debug, Deserialize)]
struct TmpSpecConfig {
    corpuses: Vec<String>,
    spec: Vec<BruteForceSpec>,
}

#[derive(Debug)]
pub struct SpecConfig {
    corpuses: Vec<CorpusData>,
    specs: Vec<BruteForceSpec>,
}

#[derive(Debug, PartialEq, Deserialize, Default)]
#[serde(default)]
pub struct BruteForceSpec {
    pub finger: FingerType,
    pub must_have_char: String,
    pub must_not_have_char: String,
    pub thresholds: Thresholds,
}

#[derive(Debug, PartialEq, Copy, Clone, Deserialize, Default)]
pub enum FingerType {
    #[default]
    IndexFinger = 6,
    RegularFinger = 3,
}

#[derive(Debug, PartialEq, Deserialize)]
#[serde(default)]
pub struct Thresholds {
    pub min_load: f32,
    pub max_load: f32,
    pub max_sfu:  f32,
}

#[derive(Debug, Clone)]
pub struct BruteForceOutputElement {
    pub chars: Vec<char>,
    pub load_per_corpus: Vec<f32>,
    pub sfu_per_corpus: Vec<f32>,
}

#[derive(Clone)]
pub struct BruteForceOutput(pub Vec<Vec<BruteForceOutputElement>>);

impl Default for Thresholds {
    fn default() -> Self {
        Self {
            min_load: 0.,
            max_load: f32::INFINITY,
            max_sfu:  f32::INFINITY,
        }
    }
}

impl SpecConfig {
    pub fn from_toml<P: AsRef<Path>>(path: P) -> Self {
        let toml_str = std::fs::read_to_string(path).unwrap();
        let tmp: TmpSpecConfig = toml::from_str(&toml_str).unwrap();

        let deserialized_corpuses: Vec<CorpusData> =
            tmp.corpuses
                .iter()
                .map(|path| CorpusData::from_json(path).unwrap())
                .collect();

        Self {
            specs: tmp.spec,
            corpuses: deserialized_corpuses,
        }
    }
}

impl BruteForceOutput {
    pub fn new() -> Self { BruteForceOutput(Vec::new()) }
    pub fn with_capacity(size: usize) -> Self { BruteForceOutput(Vec::with_capacity(size)) }
    pub fn iter(&self) -> std::slice::Iter<'_, Vec<BruteForceOutputElement>> { self.0.iter() }
}

impl FromIterator<Vec<BruteForceOutputElement>> for BruteForceOutput {
    fn from_iter<I: IntoIterator<Item=Vec<BruteForceOutputElement>>>(iter: I) -> Self {
        let mut rv = Self::new();

        for x in iter {
            rv.0.push(x.to_vec());
        }

        rv
    }
}

impl<'a> BruteForceSpec {
    pub fn brute_force(&self, corpuses: &[CorpusData]) -> BruteForceOutput {
        let mut char_buffer = vec!['\0'; self.finger as usize];
        let mut load_matrix = vec![vec![0.; corpuses.len()]; self.finger as usize];
        let mut sfu_matrix  = vec![vec![0.; corpuses.len()]; self.finger as usize];

        for (i, c) in self.must_have_char.chars().enumerate() {
            char_buffer[i] = c;
            calculate_next_load_row(&corpuses, &mut load_matrix, i, c, self.thresholds.max_load);
            calculate_next_sfu_row(&char_buffer, &corpuses, &mut sfu_matrix, i, c, self.thresholds.max_sfu);
        }

        let mut output = BruteForceOutput::with_capacity(500);

        brute_force_inner(
            &mut output,
            &corpuses,
            &self.thresholds,
            &mut char_buffer,
            self.must_have_char.chars().count(),
            0,
            &mut load_matrix,
            &mut sfu_matrix,
            &self.must_not_have_char,
        );

        output
    }
}

impl std::fmt::Display for BruteForceOutputElement {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        write!(fmt, "[")?;
        for c in self.chars.iter() {
            write!(fmt, "{c}")?;
        }
        write!(fmt, "]  ")?;

        if self.chars.len() == 3 {
            write!(fmt, "   ")?;
        }

        let mut first_iter = true;

        for (load, sfu) in self.load_per_corpus.iter().zip(self.sfu_per_corpus.iter()) {
            if !first_iter {
                write!(fmt, "  |  ")?;
            }
            first_iter = false;
            if *load < 10. { write!(fmt, " ")?; }
            write!(fmt, "{load:.3}, {sfu:.3}")?;
        }

        Ok(())
    }
}

impl std::fmt::Display for BruteForceOutput {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        for output in self.0.iter() {
            if output.len() > 1 { write!(fmt, "\n")?; }

            for finger in output.iter() {
                println!("{finger}");
            }
        }

        Ok(())
    }
}

impl SpecConfig {
    pub fn brute_force_all(&self) -> BruteForceOutput {
        use std::rc::Rc;
        use std::ops::Deref;

        let mut buffer = Vec::<Rc<BruteForceOutput>>::new();

        'main_loop: for (i, spec) in self.specs.iter().enumerate() {
            for (j, previous_spec) in self.specs[..i].iter().enumerate() {
                if spec == previous_spec {
                    buffer.push(buffer[j].clone());
                    continue 'main_loop
                }
            }

            let res = spec.brute_force(&self.corpuses);
            buffer.push(Rc::new(res));
        }

        let no_char_in_common = |vec: &Vec<&Vec<BruteForceOutputElement>>| -> bool {
            vec.iter()
                .map(|out| out[0].chars.iter())
                .multi_cartesian_product()
                .all(|v| v[0] != v[1])
        };

        BruteForceOutput(
            buffer
            .iter()
            .map(|x| x.deref().iter())
            .multi_cartesian_product()
            .filter(no_char_in_common)
            // This is so fucking stupid
            .map(|v| v.into_iter().map(|v2| v2.clone()).flatten().collect())
            .collect()
        )
    }
}

fn brute_force_inner(
    output: &mut BruteForceOutput,
    corpuses: &[CorpusData],
    thresholds:  &Thresholds,
    chars: &mut Vec<char>,
    current_index: usize,
    start_pos: usize,
    load: &mut Vec<Vec<f32>>,
    sfu: &mut Vec<Vec<f32>>,
    unused_chars: &str,
) {
    static LETTERS: [char; 29] = ['a', 'b', 'c', 'd', 'é', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ',', '.', '’'];

    let end_pos = LETTERS.len() - 5 + current_index;
    for (mut current_char_pos, &current_char) in LETTERS[start_pos..end_pos].iter().enumerate() {
        if unused_chars.contains(current_char) || chars.contains(&current_char) { continue }
        current_char_pos += start_pos;

        if calculate_next_load_row(corpuses, load, current_index, current_char, thresholds.max_load) { continue }
        if calculate_next_sfu_row(chars, corpuses, sfu, current_index, current_char, thresholds.max_sfu) { continue }

        chars[current_index] = current_char;

        if current_index == chars.len() - 1 && load[chars.len() - 1].iter().all(|l| *l >= thresholds.min_load) {
            output.0.push(vec![BruteForceOutputElement {
                chars: chars.clone(),
                load_per_corpus: load[chars.len() - 1].clone(),
                sfu_per_corpus: sfu[chars.len() - 1].clone(),
            }]);
        }
        else if current_index < chars.len() - 1 {
            brute_force_inner(
                output,
                corpuses,
                thresholds,
                chars,
                current_index + 1,
                current_char_pos + 1,
                load,
                sfu,
                unused_chars,
            );
        }
    }
}

/// Returns `true` if one of the load values is greater than `max_load`
fn calculate_next_load_row(
    corpuses: &[CorpusData],
    load: &mut Vec<Vec<f32>>,
    current_index: usize,
    current_char: char,
    max_load: f32,
) -> bool {
    for (corpus_index, corpus_data) in corpuses.iter().enumerate() {
        let previous_value =
            if current_index > 0 {
                load[current_index - 1][corpus_index]
            } else { 0. };

        load[current_index][corpus_index] =
            corpus_data.get_symbol(&current_char) + previous_value;

        if load[current_index][corpus_index] >= max_load { return true }
    }

    false
}

/// Returns `true` if one of the sfu values is greater than `max_sfu`
fn calculate_next_sfu_row(
    chars: &[char],
    corpuses: &[CorpusData],
    sfu: &mut Vec<Vec<f32>>,
    current_index: usize,
    current_char: char,
    max_sfu: f32,
) -> bool {
    for (corpus_index, corpus_data) in corpuses.iter().enumerate() {
        let previous_value =
            if current_index > 0 {
                sfu[current_index - 1][corpus_index]
            } else { 0. };

        sfu[current_index][corpus_index] =
            previous_value +
            chars[..current_index]
                .iter()
                .map(|c| corpus_data.get_digram_both_ways(current_char, *c))
                .sum::<f32>();

        if sfu[current_index][corpus_index] >= max_sfu { return true }
    }

    false
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn validate_next_load_row() {
        let test_corpus = CorpusData::from_json("test_corpus.json").unwrap();
        let mut load_matrix = vec![vec![0.], vec![0.], vec![0.], vec![0.], vec![1.], vec![0.]];

        calculate_next_load_row(&[test_corpus], &mut load_matrix, 5, 'c', 25.);

        assert_eq!(load_matrix[5][0], 4.);
    }

    #[test]
    fn validate_next_sfu_row() {
        let test_corpus = CorpusData::from_json("test_corpus.json").unwrap();
        let chars = vec!['a', 'b', 'c', 'd', 'e', '\0'];
        let mut sfu_matrix = vec![vec![0.], vec![0.], vec![0.], vec![0.], vec![1.], vec![0.]];

        calculate_next_sfu_row(&chars, &[test_corpus], &mut sfu_matrix, 5, 'f', 12.);

        assert_eq!(sfu_matrix[5][0], 4.75);
    }
}
