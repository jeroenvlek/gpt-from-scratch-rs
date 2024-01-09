use candle_core::{DType, Device, IndexOp, Result, Shape, Tensor, D};
use candle_nn::{
    embedding, layer_norm, linear, linear_no_bias, loss, sequential, Activation, AdamW, Embedding,
    LayerNorm, LayerNormConfig, Linear, Optimizer, ParamsAdamW, Sequential, VarBuilder, VarMap,
};
use candle_nn::{ops, Module};
use rand::prelude::ThreadRng;
use std::cmp::max;

use crate::dataset::Dataset;
use crate::sampling::sample_multinomial;

const FEED_FORWARD_OUT_SCALE: usize = 4;
const EPS: f64 = 1e-5;

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
        var_builder: VarBuilder,
        device: &Device,
    ) -> Result<Self> {
        let key = linear_no_bias(num_embeddings, head_size, var_builder.push_prefix("key"))?;
        let query = linear_no_bias(num_embeddings, head_size, var_builder.push_prefix("query"))?;
        let value = linear_no_bias(num_embeddings, head_size, var_builder.push_prefix("value"))?;
        let tril = Tensor::tril2(block_size, DType::U32, device)?;
        let negative_infinity = Tensor::try_from(f32::NEG_INFINITY)?.to_device(device)?;

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
        let (_, time_size, channel_size) = xs.shape().dims3()?; // JV: batch, time, channel
        let mut weights =
            (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? * (channel_size as f64).powf(-0.5))?; // (B, T, C) @ (B, C, T) -> (B, T, T)
        let masked_fill = self
            .tril
            .i((..time_size, ..time_size))?
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
    /// multiple heads of self-attention in parallel
    heads: Vec<Box<Head>>,
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
        var_builder: VarBuilder,
        device: &Device,
    ) -> Result<Self> {
        let mut heads = Vec::with_capacity(num_heads);
        for head_index in 0..num_heads {
            heads.push(Box::new(Head::new(
                num_embeddings,
                head_size,
                block_size,
                dropout_rate,
                var_builder.push_prefix(format!("head_{}", head_index)),
                device,
            )?))
        }

        let proj = linear(
            num_embeddings,
            num_embeddings,
            var_builder.push_prefix("proj"),
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
            &self
                .heads
                .iter()
                .map(|h| {
                    h.forward(xs)
                        .map_err(|error| eprintln!("Error creating the model: {}", error))
                        .expect("Could not apply head. Diggity")
                })
                .collect::<Vec<Tensor>>(),
            D::Minus1,
        )?.contiguous()?; // JV: This was necessary for Cuda to fix the stride/contiguous error. It doesn't occur on the CPU
        let projected = self.proj.forward(&concatenated)?;
        let out = ops::dropout(&projected, self.dropout_rate)?;

        Ok(out)
    }
}

pub struct FeedForward {
    /// a simple linear layer followed by a non-linearity
    net: Sequential,
}

impl FeedForward {
    pub fn new(num_embeddings: usize, dropout_rate: f32, var_builder: VarBuilder) -> Result<Self> {
        let mut net = sequential::seq();
        net = net.add(linear(
            num_embeddings,
            FEED_FORWARD_OUT_SCALE * num_embeddings,
            var_builder.push_prefix("linear1"),
        )?);
        net = net.add(Activation::Relu);
        net = net.add(linear(
            FEED_FORWARD_OUT_SCALE * num_embeddings,
            num_embeddings,
            var_builder.push_prefix("linear2"),
        )?);
        net = net.add(move |xs: &Tensor| ops::dropout(xs, dropout_rate));

        Ok(Self { net })
    }
}

impl Module for FeedForward {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.net.forward(xs)
    }
}

pub struct Block {
    /// Transformer block: communication followed by computation
    multi_head_attention: MultiHeadAttention,
    feed_forward: FeedForward,
    layer_normalization1: LayerNorm,
    layer_normalization2: LayerNorm,
}

impl Block {
    pub fn new(
        num_embeddings: usize,
        num_heads: usize,
        head_size: usize,
        block_size: usize,
        dropout_rate: f32,
        var_builder: VarBuilder,
        device: &Device,
    ) -> Result<Self> {
        // n_embd: embedding dimension, n_head: the number of heads we'd like
        let multi_head_attention = MultiHeadAttention::new(
            num_embeddings,
            num_heads,
            head_size,
            block_size,
            dropout_rate,
            var_builder.push_prefix("multi_head_attention"),
            device,
        )?;
        let feed_forward = FeedForward::new(
            num_embeddings,
            dropout_rate,
            var_builder.push_prefix("feed_forward"),
        )?;
        let layer_normalization1 = layer_norm(
            num_embeddings,
            LayerNormConfig::from(EPS),
            var_builder.push_prefix("layer_normalization1"),
        )?;
        let layer_normalization2 = layer_norm(
            num_embeddings,
            LayerNormConfig::from(EPS),
            var_builder.push_prefix("layer_normalization2"),
        )?;

        Ok(Self {
            multi_head_attention,
            feed_forward,
            layer_normalization1,
            layer_normalization2,
        })
    }
}

impl Module for Block {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let ln1 = self.layer_normalization1.forward(xs)?;
        let sa = xs.add(&self.multi_head_attention.forward(&ln1)?)?;
        let ln2 = self.layer_normalization2.forward(&sa)?;
        let ffwd_result = sa.add(&self.feed_forward.forward(&ln2)?);

        ffwd_result
    }
}

pub struct BigramLanguageModel {
    block_size: usize,
    token_embedding_table: Embedding,
    position_embedding_table: Embedding,
    blocks: Sequential,
    layer_normalization_final: LayerNorm,
    linear_head_final: Linear,
    var_map: VarMap,
    rng: ThreadRng,
}

impl BigramLanguageModel {
    pub fn new(
        vocab_size: usize,
        num_embeddings: usize,
        num_blocks: usize,
        num_heads: usize,
        block_size: usize,
        dropout_rate: f32,
        device: &Device,
    ) -> Result<Self> {
        // each token directly reads off the logits for the next token from a lookup table
        let var_map = VarMap::new();
        let var_builder = VarBuilder::from_varmap(&var_map, DType::F32, device);
        let token_embedding_table = embedding(
            vocab_size,
            num_embeddings,
            var_builder.push_prefix("token_embedding"),
        )?;
        let position_embedding_table = embedding(
            vocab_size,
            num_embeddings,
            var_builder.push_prefix("position_embedding"),
        )?;
        let mut blocks = sequential::seq();
        let head_size = num_embeddings / num_heads;
        for block_index in 0..num_blocks {
            blocks = blocks.add(Block::new(
                num_embeddings,
                num_heads,
                head_size,
                block_size,
                dropout_rate,
                var_builder.push_prefix(format!("block_{}", block_index)),
                device,
            )?);
        }
        let layer_normalization_final = layer_norm(
            num_embeddings,
            LayerNormConfig::from(EPS),
            var_builder.push_prefix("layer_normalization_final"),
        )?; // final layer norm
        let linear_head_final = linear(
            num_embeddings,
            vocab_size,
            var_builder.push_prefix("linear_head_final"),
        )?;
        let rng = rand::thread_rng();

        Ok(Self {
            block_size,
            token_embedding_table,
            position_embedding_table,
            blocks,
            layer_normalization_final,
            linear_head_final,
            var_map,
            rng,
        })
    }

    /// Inspired by:
    /// https://medium.com/@igumnovnsk/simplified-rust-example-of-training-a-neural-network-based-on-the-candle-framework-by-hugging-face-cf1ccd85a936
    pub fn train(&self, dataset: &mut Dataset, num_epochs: usize, batch_size: usize) -> Result<()> {
        let mut optimizer = AdamW::new(self.var_map.all_vars(), ParamsAdamW::default())?;

        for epoch in 0..num_epochs {
            let (training_inputs, training_targets) =
                dataset.random_training_batch(self.block_size, batch_size)?;
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

    pub fn generate(
        &mut self,
        max_new_tokens: usize,
        block_size: usize,
        device: &Device,
    ) -> Result<Vec<u32>> {
        // idx is (B, T) array of indices in the current context
        let mut generated_ids: Vec<u32> = Vec::with_capacity(max_new_tokens);
        generated_ids.push(0); // Karpathy uses idx = torch.zeros((1, 1), but candle doesn't have on-device multinomial sampling (yet?), so we use a Vec
        for i in 1..max_new_tokens {
            // crop idx to the last block_size tokens
            let generated_ids_cond: Vec<u32> = generated_ids
                .iter()
                .skip(max(generated_ids.len().saturating_sub(block_size), 0))
                .cloned()
                .collect();
            let generated_ids_cond_length = generated_ids_cond.len();
            let logits = self.forward(&Tensor::from_vec(
                generated_ids_cond,
                Shape::from((1, generated_ids_cond_length)),
                device,
            )?)?;
            // focus only on the last time step
            let most_recent_logits = logits.i((0, generated_ids_cond_length - 1, ..))?; // becomes (B, C)
            // apply softmax to get probabilities
            let probabilities = ops::softmax(&most_recent_logits, 0)?;
            // sample from the distribution
            let next_token =
                sample_multinomial(&mut self.rng, &probabilities.flatten_all()?.to_vec1()?)?;
            // append sampled index to the running sequence
            generated_ids.push(next_token);
        }

        Ok(generated_ids)
    }
}

impl Module for BigramLanguageModel {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (_, time_size) = xs.shape().dims2()?;

        // xs and targets are both (B,T) tensor of integers
        let token_embedding = self.token_embedding_table.forward(xs)?; // (B,T,C)

        let position_embedding = self.position_embedding_table.forward(&Tensor::arange(
            0,
            time_size as u32,
            xs.device(),
        )?)?; // (T,C)
        let x_embed_sum = token_embedding.broadcast_add(&position_embedding)?; // (B,T,C)
        let x_blocks = self.blocks.forward(&x_embed_sum)?; // (B,T,C)
        let x_norm = self.layer_normalization_final.forward(&x_blocks)?; // (B,T,C)
        let logits = self.linear_head_final.forward(&x_norm); // (B,T,vocab_size)

        logits
    }
}
