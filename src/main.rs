use std::fs::File;
use std::io;
use std::io::Read;

use candle_core::{Device, Shape, Tensor, IndexOp};
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

// fn random_batch(data: Tensor, batch_size: u32, block_size: u32) -> Tensor {
//
// }

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
    println!("First 1000 indices from tensor: {:?}", &data.i(0..1000));

    let data_size = *data.shape().dims().first().unwrap();
    let training_size = (data_size as f64 * 0.9) as usize;
    let training_data = data.i(0..training_size).unwrap();
    println!("Training data shape: {:?}, dtype: {:?}", training_data.shape(), training_data.dtype());

    let validation_size = data_size - training_size;
    let validation_data = data.i(0..validation_size).unwrap();
    println!("Validation data shape: {:?}, dtype: {:?}", validation_data.shape(), validation_data.dtype());

    let block_size = 8;
    println!("First block of training data: {:?}", &training_data.i(0..block_size).unwrap());

    for target_index in 1..block_size {
        let context = &training_data.i(0..target_index).unwrap();
        let target = training_data.i(target_index).unwrap();
        println!("when input is {:?} the target: {:?}", context, target)
    }

    // let batch_size = 4;
}


