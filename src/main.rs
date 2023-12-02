use std::collections::BTreeSet;
use std::fs::File;
use std::io;
use std::io::Read;
use std::iter::FromIterator;

use clap::Parser;

use args::Args;

mod args;

fn load_file(path: String) -> Result<String, io::Error> {
    let mut file = File::open(path)?; // ? operator used for error propagation

    let mut contents = String::new();
    file.read_to_string(&mut contents)?;

    Ok(contents)
}

fn unique_chars(s: String) -> BTreeSet<char> {
    let mut char_set = BTreeSet::new();

    for c in s.chars() {
        char_set.insert(c);
    }
    char_set
}

fn main() {
    let args = Args::parse();

    let raw_contents = match load_file(args.input_path) {
        Ok(contents) => {
            println!("Loaded {} characters", contents.len());
            contents
        },
        Err(error) => {
            eprintln!("Error: {}", error);
            String::new()
        }
    };
    println!("First 1000 characters: {}", &raw_contents[0..1000]);

    let char_set = unique_chars(raw_contents);
    let char_string: String = char_set.clone().into_iter().collect();
    println!("Char set: {}", char_string);
    println!("Char set size: {}", char_set.len());


}


