use std::collections::{BTreeSet, HashMap};

#[derive(Debug)]
pub struct CharSetTranscoder {
    pub char_set: BTreeSet<char>,
    char_to_index: HashMap<char, u16>,
    index_to_char: HashMap<u16, char>,
}


impl CharSetTranscoder {
    pub fn new(s: String) -> CharSetTranscoder {
        let mut char_set = BTreeSet::new();

        for c in s.chars() {
            char_set.insert(c);
        }

        let mut char_to_index: HashMap<char, u16> = HashMap::new();
        let mut index_to_char: HashMap<u16, char> = HashMap::new();
        for (i, c) in char_set.iter().enumerate() {
            char_to_index.insert(*c, i as u16);
            index_to_char.insert(i as u16, *c);
        }

        CharSetTranscoder { char_set, char_to_index, index_to_char }
    }
}