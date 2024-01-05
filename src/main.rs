use std::fs::File;
use std::io;
use std::io::Read;
use std::ops::Div;

use candle_core::{Device, IndexOp, Module, Shape, Tensor};
use clap::Parser;

use crate::bigram_language_model::BigramLanguageModel;
use args::Args;

use crate::char_set_transcoder::CharSetTranscoder;
use crate::dataset::Dataset;
use crate::self_attention_examples::self_attention_examples;
use crate::simple_bigram_language_model::SimpleBigramLanguageModel;

mod args;
mod bigram_language_model;
mod char_set_transcoder;
mod dataset;
mod sampling;
mod self_attention_examples;
mod simple_bigram_language_model;

fn load_file(path: &String) -> std::result::Result<String, io::Error> {
    let mut file = File::open(path)?; // ? operator used for error propagation

    let mut contents = String::new();
    file.read_to_string(&mut contents)?;

    Ok(contents)
}

fn main() {
    let args = Args::parse();

    let raw_contents = match load_file(&args.input_path) {
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
    println!(
        "{:?}",
        char_set_transcoder.encode(String::from("hii there"))
    );
    println!(
        "{}",
        char_set_transcoder.decode(char_set_transcoder.encode(String::from("hii there")))
    );

    let encoded = char_set_transcoder.encode(raw_contents);
    let device = &Device::cuda_if_available(0).expect("Cuda or CPU should be available");
    println!("Using device: {:?}", device);
    let data = Tensor::from_vec(encoded.clone(), Shape::from(encoded.len()), device).unwrap();
    println!("Data shape: {:?}, dtype: {:?}", data.shape(), data.dtype());
    println!("First 1000 indices from tensor: {:?}", &data.i(0..1000));

    let mut dataset = Dataset::new(data, 0.9);
    println!(
        "Training data shape: {:?}, dtype: {:?}",
        dataset.training_data.shape(),
        dataset.training_data.dtype()
    );
    println!(
        "Validation data shape: {:?}, dtype: {:?}",
        dataset.validation_data.shape(),
        dataset.validation_data.dtype()
    );

    let block_size = 8usize;
    println!(
        "First block of training data: {:?}",
        &dataset.training_data.i(0..block_size).unwrap()
    );

    for target_index in 1..block_size {
        let context = &dataset.training_data.i(0..target_index).unwrap();
        let target = dataset.training_data.i(target_index).unwrap();
        println!("when input is {:?} the target: {:?}", context, target)
    }

    let batch_size = 4usize;
    let (stacked_contexts, stacked_targets) = dataset
        .random_training_batch(block_size, batch_size)
        .unwrap();
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

    // run_simple_model(&args, &char_set_transcoder, device, &mut dataset);

    // The mathematical trick in self-attention
    self_attention_examples(device)
        .map_err(|error| eprintln!("Error executing the self attention examples: {}", error))
        .expect("Self attention example should not fail");

    // Full finished code, for reference
    run_complete_model(&args, &char_set_transcoder, device, &mut dataset)
        .map_err(|error| eprintln!("Error running the complete model: {}", error))
        .expect("Running the complete model should not fail should not fail");
}

fn run_complete_model(
    args: &Args,
    char_set_transcoder: &CharSetTranscoder,
    device: &Device,
    dataset: &mut Dataset,
) -> candle_core::Result<()> {
    // hyperparameters
    let batch_size = 16usize; // how many independent sequences will we process in parallel?
    let block_size = 32usize; // what is the maximum context length for predictions?
    let num_embdeddings = 64usize;
    let num_heads = 4usize;
    let num_blocks = 4usize; // JV: n_layer
    let dropout_rate = 0.0f32;
    // ------------

    let mut bigram_model = BigramLanguageModel::new(
        char_set_transcoder.char_set.len(),
        num_embdeddings,
        num_blocks,
        num_heads,
        block_size,
        dropout_rate,
        device,
    )?;

    bigram_model.train(dataset, args.num_epochs_complete, batch_size)?;

    let token_ids = bigram_model.generate(args.max_new_tokens, block_size, device)?;
    let decoded_tokens = char_set_transcoder.decode(token_ids);

    println!("Generated string: {}", decoded_tokens);

    Ok(())
}

fn run_simple_model(
    args: &Args,
    char_set_transcoder: &CharSetTranscoder,
    device: &Device,
    dataset: &mut Dataset,
) -> candle_core::Result<()> {
    let mut simple_bigram_model = SimpleBigramLanguageModel::new(
        char_set_transcoder.char_set.len(),
        char_set_transcoder.char_set.len(),
        device,
    );

    let batch_size = 32;
    simple_bigram_model.train(dataset, args.num_epochs_simple, batch_size)?;

    simple_bigram_model.generate(args.max_new_tokens, device)?;

    Ok(())
}
