use std::fs::File;
use std::io;
use std::io::Read;
use std::iter::FromIterator;

use clap::Parser;

use args::Args;

use crate::char_set_transcoder::CharSetTranscoder;

mod args;
mod char_set_transcoder;

fn load_file(path: String) -> Result<String, io::Error> {
    let mut file = File::open(path)?; // ? operator used for error propagation

    let mut contents = String::new();
    file.read_to_string(&mut contents)?;

    Ok(contents)
}


fn main() {
    let args = Args::parse();

    let raw_contents = match load_file(args.input_path) {
        Ok(contents) => {
            println!("Loaded {} characters", contents.len());
            contents
        }
        Err(error) => {
            eprintln!("Error: {}", error);
            String::new()
        }
    };
    println!("First 1000 characters: {}", &raw_contents[0..1000]);

    let char_set_transcoder = CharSetTranscoder::new(raw_contents);
    let char_string: String = char_set_transcoder.char_set.clone().into_iter().collect();
    println!("Char set: {}", char_string);
    println!("Char set size: {}", char_set_transcoder.char_set.len());
}


