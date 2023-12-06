use candle_core::{DType, Device, Result, Error, Shape, Tensor, D, IndexOp};
use candle_nn::{Module, ops};
use candle_nn::{loss, AdamW, Embedding, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use rand::distributions::Distribution;
use rand::prelude::ThreadRng;

use crate::dataset::Dataset;

pub struct BigramLanguageModel {
    vocab_size: usize,
    token_embedding_table: Embedding,
    var_map: VarMap,
    rng: ThreadRng,
}

impl BigramLanguageModel {
    pub fn new(vocab_size: usize, hidden_size: usize, device: &Device) -> Self {
        let var_map = VarMap::new();
        let var_builder = VarBuilder::from_varmap(&var_map, DType::F32, device);
        let embeddings = var_builder
            .get((vocab_size, hidden_size), "embeddings")
            .unwrap();
        let token_embedding_table = Embedding::new(embeddings, hidden_size);
        let rng =rand::thread_rng();
        Self {
            vocab_size,
            token_embedding_table,
            var_map,
            rng
        }
    }

    /// Inspired by:
    /// https://medium.com/@igumnovnsk/simplified-rust-example-of-training-a-neural-network-based-on-the-candle-framework-by-hugging-face-cf1ccd85a936
    pub fn train(&self, mut dataset: Dataset, num_epochs: usize, batch_size: usize) -> Result<()> {
        let mut optimizer = AdamW::new(self.var_map.all_vars(), ParamsAdamW::default())?;

        for epoch in 0..num_epochs {
            let (training_inputs, training_targets) =
                dataset.random_training_batch(self.vocab_size, batch_size)?;
            let logits = self.forward(&training_inputs)?;
            let (batch_size, time_size, channel_size) = logits.shape().dims3()?;
            let loss = loss::cross_entropy(
                &logits.reshape(Shape::from((batch_size * time_size, channel_size)))?,
                &training_targets.reshape(Shape::from((batch_size * time_size,)))?,
            )?;
            optimizer.backward_step(&loss)?;

            println!(
                "Epoch: {epoch:3} Train loss: {:8.5}",
                loss.to_scalar::<f32>()?
            );
        }

        Ok(())
    }

    fn sample_multinomial(&mut self, prs: &Vec<f32>) -> Result<u32> {
        let distr = rand::distributions::WeightedIndex::new(prs).map_err(Error::wrap)?;
        let next_token = distr.sample(&mut self.rng) as u32;
        Ok(next_token)
    }

    pub fn generate(&mut self, inputs: Tensor, max_new_tokens: usize) -> Result<Tensor> {
        let mut generated_ids = inputs;
        for _ in 0..max_new_tokens {
            let logits = self.forward(&generated_ids)?;
            let (batch_size, time_size, channel_size) = logits.shape().dims3()?;
            let most_recent_logits = logits.squeeze(1)?;
            let probabilities = ops::softmax(&most_recent_logits, D::Minus1)?;
            let next_token = self.sample_multinomial(&probabilities.to_vec2().unwrap().first().unwrap())?;
            let to_stack = [&generated_ids, &Tensor::try_from(next_token)?];
            generated_ids = Tensor::stack(&to_stack[..], 1)?;
        }

        Ok(generated_ids)
    }
}

impl Module for BigramLanguageModel {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.token_embedding_table.forward(xs)
    }
}
