use std::collections::{BTreeSet, HashMap};

#[derive(Debug)]
pub struct CharSetTranscoder {
    pub char_set: BTreeSet<char>,
    char_to_index: HashMap<char, u32>,
    index_to_char: HashMap<u32, char>,
}


impl CharSetTranscoder {
    pub fn new(s: String) -> Self {
        let mut char_set = BTreeSet::new();

        for c in s.chars() {
            char_set.insert(c);
        }

        let mut char_to_index: HashMap<char, u32> = HashMap::new();
        let mut index_to_char: HashMap<u32, char> = HashMap::new();
        for (i, c) in char_set.iter().enumerate() {
            char_to_index.insert(*c, i as u32);
            index_to_char.insert(i as u32, *c);
        }

        CharSetTranscoder { char_set, char_to_index, index_to_char }
    }

    pub fn encode(&self, s: String) -> Vec<u32> {
        let mut index_vec: Vec<u32> = Vec::new();
        for c in s.chars() {
            index_vec.push(self.char_to_index.get(&c).cloned().unwrap())
        }
        index_vec
    }
    pub fn decode(&self, index_vec: Vec<u32>) -> String {
        let mut char_vec: Vec<char> = Vec::new();
        for index in index_vec.iter() {
            char_vec.push(self.index_to_char.get(&index).cloned().unwrap())
        }
        char_vec.iter().collect()
    }
}