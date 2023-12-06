use candle_core::{DType, Device, Result, Shape, Tensor};
use candle_nn::Module;
use candle_nn::{loss, AdamW, Embedding, Optimizer, ParamsAdamW, VarBuilder, VarMap};

use crate::dataset::Dataset;

pub struct BigramLanguageModel {
    vocab_size: usize,
    token_embedding_table: Embedding,
    var_map: VarMap,
}

impl BigramLanguageModel {
    pub fn new(vocab_size: usize, hidden_size: usize, device: &Device) -> Self {
        let var_map = VarMap::new();
        let var_builder = VarBuilder::from_varmap(&var_map, DType::F32, device);
        let embeddings = var_builder
            .get((vocab_size, hidden_size), "embeddings")
            .unwrap();
        let token_embedding_table = Embedding::new(embeddings, hidden_size);
        Self {
            vocab_size,
            token_embedding_table,
            var_map,
        }
    }

    /// Inspired by:
    /// https://medium.com/@igumnovnsk/simplified-rust-example-of-training-a-neural-network-based-on-the-candle-framework-by-hugging-face-cf1ccd85a936
    pub fn train(&self, mut dataset: Dataset, num_epochs: usize, batch_size: usize) -> Result<()> {
        let mut optimizer = AdamW::new(self.var_map.all_vars(), ParamsAdamW::default())?;

        for epoch in 0..num_epochs {
            let (training_inputs, training_targets) =
                dataset.random_training_batch(self.vocab_size, batch_size);
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
}

impl Module for BigramLanguageModel {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        self.token_embedding_table.forward(xs)
    }
}
