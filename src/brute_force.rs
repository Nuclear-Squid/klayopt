use itertools::Itertools;
use crate::corpuses::CorpusData;

#[derive(PartialEq)]
pub struct BruteForceSpec<'a> {
    pub corpuses: Vec<&'a CorpusData>,
    pub finger: FingerType,
    pub must_have_char: String,
    pub must_not_have_char: String,
    pub thresholds: Thresholds,
}

#[derive(PartialEq, Copy, Clone)]
pub enum FingerType { RegularFinger = 3, IndexFinger = 6 }

#[derive(PartialEq)]
pub struct Thresholds {
    pub min_load: f32,
    pub max_load: f32,
    pub max_sfu:  f32,
}

#[derive(Debug, Clone)]
pub struct BruteForceOutputElement {
    chars: Vec<char>,
    load_per_corpus: Vec<f32>,
    sfu_per_corpus: Vec<f32>,
}

#[derive(Clone)]
pub struct BruteForceOutput<const N: usize>(pub Vec<[BruteForceOutputElement; N]>);

impl<const N: usize> BruteForceOutput<N> {
    pub fn new() -> Self { BruteForceOutput(Vec::new()) }
    pub fn with_capacity(size: usize) -> Self { BruteForceOutput(Vec::with_capacity(size)) }
    pub fn iter(&self) -> std::slice::Iter<'_, [BruteForceOutputElement; N]> { self.0.iter() }
}

impl<'a> BruteForceSpec<'a> {
    pub fn brute_force(&self) -> BruteForceOutput<1> {
        let mut char_buffer = vec!['\0'; self.finger as usize];
        let mut load_matrix = vec![vec![0.; self.corpuses.len()]; self.finger as usize];
        let mut sfu_matrix  = vec![vec![0.; self.corpuses.len()]; self.finger as usize];

        for (i, c) in self.must_have_char.chars().enumerate() {
            char_buffer[i] = c;
            calculate_next_load_row(&self.corpuses, &mut load_matrix, i, c, self.thresholds.max_load);
            calculate_next_sfu_row(&char_buffer, &self.corpuses, &mut sfu_matrix, i, c, self.thresholds.max_sfu);
        }

        let mut output = BruteForceOutput::<1>::with_capacity(500);

        brute_force_inner(
            &mut output,
            &self.corpuses,
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
            write!(fmt, "{load:.3}, {sfu:.3}")?;
        }

        Ok(())
    }
}

impl<const N: usize> std::fmt::Display for BruteForceOutput<N> {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        for output in self.0.iter() {
            if N > 1 { write!(fmt, "\n")?; }

            for finger in output.iter() {
                println!("{finger}");
            }
        }

        Ok(())
    }
}

pub fn brute_force_all<const N: usize>(specs: [BruteForceSpec; N]) -> BruteForceOutput<N> {
    use std::rc::Rc;
    use std::ops::Deref;

    let mut buffer = Vec::<Rc<BruteForceOutput<1>>>::new();

    'main_loop: for (i, spec) in specs.iter().enumerate() {
        for (j, previous_spec) in specs[..i].iter().enumerate() {
            if spec == previous_spec {
                buffer.push(buffer[j].clone());
                continue 'main_loop
            }
        }

        buffer.push(Rc::new(spec.brute_force()));
    }

    let no_char_in_common = |vec: &Vec<&[BruteForceOutputElement; 1]>| -> bool {
        vec.iter()
            .map(|out| out[0].chars.iter())
            .multi_cartesian_product()
            .all(|v| v[0] != v[1])
    };

    let vec_to_arr = |vec: Vec<&[BruteForceOutputElement; 1]>| -> [BruteForceOutputElement; N] {
        vec.iter()
            .map(|arr| arr[0].clone())
            .collect::<Vec<BruteForceOutputElement>>()
            .try_into()
            .unwrap()
    };

    BruteForceOutput(
        buffer
            .iter()
            .map(|x| x.deref().iter())
            .multi_cartesian_product()
            .filter(no_char_in_common)
            .map(vec_to_arr)
            .collect::<Vec<[BruteForceOutputElement; N]>>()
    )
}

fn brute_force_inner(
    output: &mut BruteForceOutput<1>,
    corpuses: &[&CorpusData],
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
            output.0.push([BruteForceOutputElement {
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
    corpuses: &[&CorpusData],
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
    corpuses: &[&CorpusData],
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
