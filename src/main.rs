use std::fs::File;
use std::io;
use std::io::Read;
use std::ops::Div;

use candle_core::{DType, Device, IndexOp, Module, Shape, Tensor, D};
use candle_nn::{linear_no_bias, ops, Linear, VarBuilder, VarMap};
use clap::Parser;

use args::Args;

use crate::char_set_transcoder::CharSetTranscoder;
use crate::dataset::Dataset;

mod args;
mod char_set_transcoder;
mod dataset;
mod simple_bigram_language_model;

fn load_file(path: String) -> std::result::Result<String, io::Error> {
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
    println!(
        "{:?}",
        char_set_transcoder.encode(String::from("hii there"))
    );
    println!(
        "{}",
        char_set_transcoder.decode(char_set_transcoder.encode(String::from("hii there")))
    );

    let encoded = char_set_transcoder.encode(raw_contents);
    let device = &Device::Cpu;
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
    // let (stacked_contexts, stacked_targets) = dataset.random_training_batch(block_size, batch_size).unwrap();
    // println!("inputs:");
    // println!("Contexts (xb) shape: {:?}", stacked_contexts.shape());
    // println!("targets:");
    // println!("Contexts (yb) shape: {:?}", stacked_targets.shape());
    //
    // for b in 0..batch_size {
    //     for t in 0..block_size {
    //         let context = stacked_contexts.i(b).unwrap().i(0..t + 1).unwrap();
    //         let target = stacked_targets.i(b).unwrap().i(t).unwrap();
    //         println!("when input is {:?} the target: {:?}", context, target);
    //     }
    // }
    //
    // let mut simple_bigram_model = SimpleBigramLanguageModel::new(
    //     char_set_transcoder.char_set.len(),
    //     char_set_transcoder.char_set.len(),
    //     device,
    // );
    // match simple_bigram_model.train(dataset, args.num_epochs, 32) {
    //     Ok(_) => println!("Finished training the model"),
    //     Err(error) => eprintln!("Error training the model: {}", error)
    // }
    //
    // match simple_bigram_model.generate(500, device) {
    //     Ok(generated_ids) => {
    //         let decoded = char_set_transcoder.decode(generated_ids);
    //         println!("Bigram model generated: {}", decoded);
    //     }
    //     Err(error) => eprintln!("Error generating characters with bigram model: {}", error)
    // }

    // The mathematical trick in self-attention
    self_attention_examples(device).expect("Self attention example failed!");
}

fn self_attention_examples(device: &Device) -> candle_core::Result<()> {
    // toy example illustrating how matrix multiplication can be used for a "weighted aggregation"
    let mut a = Tensor::tril2(3, DType::F32, device)?;
    let sum_a = a.sum_keepdim(1)?;
    a = a.broadcast_div(&sum_a)?;
    println!("A: {:?}", a.to_vec2::<f32>());

    let b = Tensor::rand(0.0f32, 10.0f32, Shape::from((3, 2)), device)?;
    println!("B: {:?}", b.to_vec2::<f32>());

    let c = a.matmul(&b)?;
    println!("C: {:?}", c.to_vec2::<f32>());

    // consider the following toy example:
    let dims = (4, 8, 2); // batch, time, channels
    let x = Tensor::rand(0f32, 10f32, Shape::from(dims), device)?;
    println!("x.shape: {:?}", x.shape());

    // We want x[b,t] = mean_{i<=t} x[b,i]
    let mut means: Vec<Tensor> = Vec::with_capacity(dims.0 * dims.1 * dims.2);
    for b_idx in 0..dims.0 {
        for t_idx in 0..dims.1 {
            let x_prev = x.i((b_idx, 0..t_idx + 1, ..))?;
            let mean = x_prev.mean_keepdim(0)?;
            means.push(mean);
        }
    }
    let x_bag_of_words = Tensor::stack(means.as_slice(), 1)?.reshape(Shape::from(dims))?;
    println!("xbow: {:?}", x_bag_of_words.to_vec3::<f32>());
    println!("xbow shape: {:?}", x_bag_of_words.shape());

    // version 2: using matrix multiply for a weighted aggregation
    let mut wei = Tensor::tril2(dims.1, DType::F32, device)?;
    let sum_wei = wei.sum_keepdim(1)?;
    wei = wei.broadcast_div(&sum_wei)?;
    let x_bag_of_words2 = wei.broadcast_matmul(&x)?;
    println!("xbow2: {:?}", x_bag_of_words2.to_vec3::<f32>());
    println!("xbow2 shape: {:?}", x_bag_of_words2.shape());
    println!(
        "allclose: {}",
        all_close(&x_bag_of_words, &x_bag_of_words2)?
    );

    // version 3: use Softmax
    let mut neg_inf = Tensor::from_vec(
        vec![f32::NEG_INFINITY; dims.1 * dims.1],
        Shape::from((dims.1, dims.1)),
        device,
    )?;
    wei = Tensor::tril2(dims.1, DType::U32, device)?
        .where_cond(&Tensor::tril2(dims.1, DType::F32, device)?, &neg_inf)?;
    println!("wei: {:?}", wei.to_vec2::<f32>());
    wei = ops::softmax(&wei, 1)?;
    println!("wei: {:?}", wei.to_vec2::<f32>());
    let x_bag_of_words3 = wei.broadcast_matmul(&x)?;
    println!("xbow3: {:?}", x_bag_of_words3.to_vec3::<f32>());
    println!("xbow3 shape: {:?}", x_bag_of_words3.shape());
    println!(
        "allclose: {}",
        all_close(&x_bag_of_words, &x_bag_of_words3)?
    );

    // version 4: self-attention!
    let (B, T, C) = (4usize, 8usize, 32usize); // batch, time, channels
    let x2 = Tensor::randn(0f32, 1f32, Shape::from((B, T, C)), device)?;

    // let's see a single Head perform self-attention
    const HEAD_SIZE: usize = 16;
    let var_map = VarMap::new();
    let key = linear_no_bias(
        C,
        HEAD_SIZE,
        VarBuilder::from_varmap(&var_map, DType::F32, device),
    )?;
    let query = linear_no_bias(
        C,
        HEAD_SIZE,
        VarBuilder::from_varmap(&var_map, DType::F32, device),
    )?;
    let value = linear_no_bias(
        C,
        HEAD_SIZE,
        VarBuilder::from_varmap(&var_map, DType::F32, device),
    )?;
    let k = key.forward(&x2)?;
    let q = query.forward(&x2)?;
    wei = q.matmul(&k.transpose(D::Minus2, D::Minus1)?)?; // (B, T, 16) @ (B, 16, T) ---> (B, T, T)
    println!("wei.shape: {:?}", wei.shape());

    neg_inf = Tensor::from_vec(
        vec![f32::NEG_INFINITY; B * T * T],
        Shape::from((B, T, T)),
        device,
    )?;

    let masked_fill = Tensor::stack(
        &((0..B)
            .map(|b| -> Tensor {
                Tensor::tril2(T, DType::U32, device)
                    .unwrap()
                    .where_cond(
                        &wei.i((b, .., ..)).unwrap(),
                        &neg_inf.i((b, .., ..)).unwrap(),
                    )
                    .unwrap()
            })
            .collect::<Vec<Tensor>>()),
        0,
    )?;
    println!("masked_fill.shape: {:?}", masked_fill.shape());
    println!("masked_fill: {:?}", masked_fill.to_vec3::<f32>());
    wei = ops::softmax(&masked_fill, D::Minus1)?;

    let v = value.forward(&x2) ?;
    let out = wei.matmul(&v)?;
    println!("out.shape: {:?}", out.shape());

    println!("wei[0]: {:?}", wei.i(0)?.to_vec2::<f32>());

    Ok(())
}

fn all_close(lhs: &Tensor, rhs: &Tensor) -> candle_core::Result<bool> {
    // JV: interesting to see different precisions between the different xbows, rounding fixes it, but should investigate further
    let element_compare = lhs.round_to(4)?.eq(&rhs.round_to(4)?)?.sum_all()?;
    Ok(element_compare.to_vec0::<u8>()? == lhs.shape().elem_count() as u8)
}
