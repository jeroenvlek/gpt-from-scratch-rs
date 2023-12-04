use std::fs::File;
use std::io;
use std::io::Read;

use candle_core::{Device, Shape, Tensor};
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

    let char_set_transcoder = CharSetTranscoder::new(raw_contents.clone());
    let char_string: String = char_set_transcoder.char_set.clone().into_iter().collect();
    println!("Char set: {}", char_string);
    println!("Char set size: {}", char_set_transcoder.char_set.len());
    println!("{:?}", char_set_transcoder.encode(String::from("hii there")));
    println!("{}", char_set_transcoder.decode(char_set_transcoder.encode(String::from("hii there"))));

    let encoded = char_set_transcoder.encode(raw_contents);
    let device = &Device::Cpu;
    let data = Tensor::from_vec(encoded.clone(), Shape::from(encoded.len()), device).unwrap();
    println!("Data shape: {:?}, dtype: {:?}", data.shape(), data.dtype());
    println!("First 1000 indices from tensor: {:?}", &data.to_vec1::<u32>().unwrap()[0..1000]);

    let training_size = (encoded.len() as f64 * 0.9) as usize;
    let training_data = Tensor::from_vec(encoded.iter().take(training_size).cloned().collect(), Shape::from(training_size), device).unwrap();
    println!("Training data shape: {:?}, dtype: {:?}", training_data.shape(), training_data.dtype());

    let validation_size = encoded.len() - training_size;
    let validation_data = Tensor::from_vec(encoded.iter().rev().take(validation_size).cloned().collect(), Shape::from(validation_size), device).unwrap();
    println!("Validation data shape: {:?}, dtype: {:?}", validation_data.shape(), validation_data.dtype());
}


