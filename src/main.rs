use std::fs::File;
use std::io;
use std::io::Read;

use candle_core::{Device, IndexOp, Shape, Tensor};
use clap::Parser;

use args::Args;

use crate::char_set_transcoder::CharSetTranscoder;
use crate::dataset::Dataset;

mod args;
mod char_set_transcoder;
mod dataset;

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
    println!("First 1000 indices from tensor: {:?}", &data.i(0..1000));

    let mut dataset = Dataset::new(data, 0.9);
    println!("Training data shape: {:?}, dtype: {:?}", dataset.training_data.shape(), dataset.training_data.dtype());
    println!("Validation data shape: {:?}, dtype: {:?}", dataset.validation_data.shape(), dataset.validation_data.dtype());

    let block_size = 8usize;
    println!("First block of training data: {:?}", &dataset.training_data.i(0..block_size).unwrap());

    for target_index in 1..block_size {
        let context = &dataset.training_data.i(0..target_index).unwrap();
        let target = dataset.training_data.i(target_index).unwrap();
        println!("when input is {:?} the target: {:?}", context, target)
    }

    let batch_size = 4usize;
    let (stacked_contexts, stacked_targets) = dataset.random_batch(block_size, batch_size);
    println!("inputs:");
    println!("Contexts (xb) shape: {:?}", stacked_contexts.shape());
    println!("targets:");
    println!("Contexts (yb) shape: {:?}", stacked_targets.shape());

    for b in 0..batch_size {
        for t in 0..block_size {
            let context = stacked_contexts.i(b).unwrap().i(0..t + 1).unwrap();
            let target = stacked_targets.i(b).unwrap().i(t).unwrap();
            println!("when input is {:?} the target: {:?}", context, target);
        }
    }
}


