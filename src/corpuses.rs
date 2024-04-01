use std::collections::HashMap;

// ╭─────────────────────────────────────────────────────────╮
// │                         Corpus                          │
// ╰─────────────────────────────────────────────────────────╯

#[derive(serde::Deserialize, Debug)]
struct TmpCorpusData {
    corpus:   String,
    symbols:  HashMap<char, f32>,
    digrams:  HashMap<String, f32>,
    trigrams: HashMap<String, f32>,
}

#[derive(PartialEq, Clone)]
pub struct CorpusData {
    name:     String,
    symbols:  HashMap<char, f32>,
    digrams:  HashMap<[char; 2], f32>,
    trigrams: HashMap<[char; 3], f32>,
}

#[derive(thiserror::Error, Debug)]
pub enum CorpusDeserializeError {
    #[error("An io error occured when desializing a corpus : {0}")]
    IO(#[from] std::io::Error),

    #[error("Serde couldn’t deserialise a corpus : {0}")]
    SerdeJson(#[from] serde_json::Error),
}

impl CorpusData {
    pub fn from_json<S>(path: S) -> Result<Self, CorpusDeserializeError>
    where S: AsRef<std::path::Path> {
        let input_file = std::fs::File::open(path)?;
        let reader = std::io::BufReader::new(input_file);
        let json: TmpCorpusData = serde_json::from_reader(reader)?;

        Ok(Self {
            name:     json.corpus,
            symbols:  json.symbols,
            digrams:  transpose_hashmap(&json.digrams),
            trigrams: transpose_hashmap(&json.trigrams),
        })
    }

    #[inline]
    pub fn get_name<'a>(&'a self) -> &'a str { &self.name }

    #[inline]
    pub fn get_symbol(&self, symbol: &char) -> f32 {
        self.symbols.get(symbol).map_or(0., |f| *f)
    }

    #[inline]
    pub fn get_digram(&self, digram: &[char; 2]) -> f32 {
        self.digrams.get(digram).map_or(0., |f| *f)
    }

    #[inline]
    pub fn get_trigram(&self, trigram: &[char; 3]) -> f32 {
        self.trigrams.get(trigram).map_or(0., |f| *f)
    }

    #[inline]
    pub fn get_digram_both_ways(&self, c1: char, c2: char) -> f32 {
        self.get_digram(&[c1, c2]) + self.get_digram(&[c2, c1])
    }
}

fn str_to_char_array<const N: usize>(input: &str) -> [char; N] {
    let mut rv = ['\0'; N];
    for (i, c) in input.chars().enumerate() { rv[i] = c; }
    return rv
}

fn transpose_hashmap<const N: usize>(input: &HashMap<String, f32>) -> HashMap<[char; N], f32> {
    input.iter().map(|(key, value)| (str_to_char_array(key), *value)).collect()
}
