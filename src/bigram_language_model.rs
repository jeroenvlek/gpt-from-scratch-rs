use candle_core::{D, Device, DType, Error, IndexOp, Result, Shape, Tensor};
use candle_nn::{
    AdamW, Embedding, linear, Linear, linear_no_bias, loss, Optimizer, ParamsAdamW,
    VarBuilder, VarMap,
};
use candle_nn::{Module, ops};
use rand::distributions::Distribution;
use rand::prelude::ThreadRng;

use crate::dataset::Dataset;

#[derive(Clone)]
pub struct Head {
    /// one head of self-attention
    key: Linear,
    query: Linear,
    value: Linear,
    tril: Tensor,
    negative_infinity: Tensor,
    dropout_rate: f32,
}

impl Head {
    pub fn new(
        num_embeddings: usize,
        head_size: usize,
        block_size: usize,
        dropout_rate: f32,
        var_map: &VarMap,
        device: &Device,
    ) -> Result<Self> {
        let key = linear_no_bias(
            num_embeddings,
            head_size,
            VarBuilder::from_varmap(var_map, DType::F32, device),
        )?;
        let query = linear_no_bias(
            num_embeddings,
            head_size,
            VarBuilder::from_varmap(var_map, DType::F32, device),
        )?;
        let value = linear_no_bias(
            num_embeddings,
            head_size,
            VarBuilder::from_varmap(var_map, DType::F32, device),
        )?;
        let tril = Tensor::tril2(block_size, DType::F32, device)?;
        let negative_infinity = Tensor::try_from(f32::NEG_INFINITY)?;

        Ok(Self {
            key,
            query,
            value,
            tril,
            negative_infinity,
            dropout_rate,
        })
    }
}

impl Module for Head {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let k = self.key.forward(xs)?;
        let q = self.query.forward(xs)?;

        // compute attention scores ("affinities")
        let c = xs.shape().dims()[2]; // JV: batch, time, channel
        let mut weights = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? * (c as f64).powf(-0.5))?; // (B, T, C) @ (B, C, T) -> (B, T, T)
        let masked_fill = self
            .tril
            .broadcast_as(Shape::from(weights.shape()))?
            .where_cond(
                &weights,
                &self.negative_infinity.broadcast_as(weights.shape())?,
            )?; // (B, T, T)
        weights = ops::softmax(&masked_fill, D::Minus1)?; // (B, T, T)
        weights = ops::dropout(&weights, self.dropout_rate)?;
        // perform the weighted aggregation of the values
        let v = self.value.forward(&xs)?; // (B,T,C)
        let out = weights.matmul(&v)?; // (B, T, T) @ (B, T, C) -> (B, T, C)

        Ok(out)
    }
}

pub struct MultiHeadAttention {
    heads: Vec<Head>,
    proj: Linear,
    dropout_rate: f32,
}

impl MultiHeadAttention {
    pub fn new(
        num_embeddings: usize,
        num_heads: usize,
        head_size: usize,
        block_size: usize,
        dropout_rate: f32,
        var_map: &VarMap,
        device: &Device,
    ) -> Result<Self> {
        let heads = vec![
            Head::new(
                num_embeddings,
                head_size,
                block_size,
                dropout_rate,
                var_map,
                device
            )?;
            num_heads
        ];
        let proj = linear(
            num_embeddings,
            num_embeddings,
            VarBuilder::from_varmap(var_map, DType::F32, device),
        )?;

        Ok(Self {
            heads,
            proj,
            dropout_rate,
        })
    }
}

impl Module for MultiHeadAttention {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let concatenated = Tensor::cat(
            &self.heads
                .iter()
                .map(|h| h.forward(xs).expect("Could not apply head. Diggity"))
                .collect::<Vec<Tensor>>(),
            D::Minus1,
        )?;
        let projected = self.proj.forward(&concatenated)?;
        let out = ops::dropout(&projected, self.dropout_rate)?;

        Ok(out)
    }
}

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
        let distribution = rand::distributions::WeightedIndex::new(prs).map_err(Error::wrap)?;
        let next_token = distribution.sample(&mut self.rng) as u32;
        Ok(next_token)
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
            let next_token = self.sample_multinomial(&vec)?;
            generated_ids.push(next_token);
        }

        Ok(generated_ids)
    }
}

impl Module for BigramLanguageModel {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.token_embedding_table.forward(xs)
    }
}
