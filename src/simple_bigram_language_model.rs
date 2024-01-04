use candle_core::{DType, Device, IndexOp, Result, Shape, Tensor};
use candle_nn::{loss, AdamW, Embedding, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use candle_nn::{ops, Module};
use rand::prelude::ThreadRng;

use crate::dataset::Dataset;
use crate::sampling::sample_multinomial;

pub struct SimpleBigramLanguageModel {
    /// Even though Karpathy calls the final bigram model simple, this is the intermediate version in the notebook.
    vocab_size: usize,
    token_embedding_table: Embedding,
    var_map: VarMap,
    rng: ThreadRng,
}

impl SimpleBigramLanguageModel {
    pub fn new(vocab_size: usize, hidden_size: usize, device: &Device) -> Self {
        let var_map = VarMap::new();
        let var_builder = VarBuilder::from_varmap(&var_map, DType::F32, device);
        let embeddings = var_builder
            .get((vocab_size, hidden_size), "embeddings")
            .unwrap();
        let token_embedding_table = Embedding::new(embeddings, hidden_size);
        let rng = rand::thread_rng();
        Self {
            vocab_size,
            token_embedding_table,
            var_map,
            rng,
        }
    }

    /// Inspired by:
    /// https://medium.com/@igumnovnsk/simplified-rust-example-of-training-a-neural-network-based-on-the-candle-framework-by-hugging-face-cf1ccd85a936
    pub fn train(&self, dataset: &mut Dataset, num_epochs: usize, batch_size: usize) -> Result<()> {
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

    pub fn generate(&mut self, max_new_tokens: usize, device: &Device) -> Result<Vec<u32>> {
        let mut generated_ids: Vec<u32> = Vec::with_capacity(max_new_tokens);
        generated_ids.push(0); // Karpathy uses idx = torch.zeros((1, 1)
        for i in 1..max_new_tokens {
            let logits = self.forward(&Tensor::from_vec(
                generated_ids.clone(),
                Shape::from(i),
                device,
            )?)?;
            let most_recent_logits = logits.i((i - 1, ..))?;
            let probabilities = ops::softmax(&most_recent_logits, 0)?;
            let vec = probabilities.to_vec1()?;
            let next_token = sample_multinomial(&mut self.rng, &vec)?;
            generated_ids.push(next_token);
        }

        Ok(generated_ids)
    }
}

impl Module for SimpleBigramLanguageModel {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.token_embedding_table.forward(xs)
    }
}
