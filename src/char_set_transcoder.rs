use std::collections::{BTreeSet, HashMap};

#[derive(Debug)]
pub struct CharSetTranscoder {
    pub char_set: BTreeSet<char>,
    char_to_index: HashMap<char, usize>,
    index_to_char: HashMap<usize, char>,
}


impl CharSetTranscoder {
    pub fn new(s: String) -> Self {
        let mut char_set = BTreeSet::new();

        for c in s.chars() {
            char_set.insert(c);
        }

        let mut char_to_index: HashMap<char, usize> = HashMap::new();
        let mut index_to_char: HashMap<usize, char> = HashMap::new();
        for (i, c) in char_set.iter().enumerate() {
            char_to_index.insert(*c, i);
            index_to_char.insert(i, *c);
        }

        CharSetTranscoder { char_set, char_to_index, index_to_char }
    }

    pub fn encode(&self, s: String) -> Vec<usize> {
        let mut index_vec: Vec<usize> = Vec::new();
        for c in s.chars() {
            index_vec.push(self.char_to_index.get(&c).cloned().unwrap())
        }
        index_vec
    }
    pub fn decode(&self, index_vec: Vec<usize>) -> String {
        let mut char_vec: Vec<char> = Vec::new();
        for index in index_vec.iter() {
            char_vec.push(self.index_to_char.get(&index).cloned().unwrap())
        }
        char_vec.iter().collect()
    }
}