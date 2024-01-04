use candle_core::{DType, Device, Error, IndexOp, Result, Shape, Tensor, D};
use candle_nn::{
    embedding, layer_norm, linear, linear_no_bias, loss, sequential, Activation, AdamW, Embedding,
    LayerNorm, LayerNormConfig, Linear, Optimizer, ParamsAdamW, Sequential, VarBuilder, VarMap,
};
use candle_nn::{ops, Module};
use rand::distributions::Distribution;
use rand::prelude::ThreadRng;

use crate::dataset::Dataset;

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
        var_map: &VarMap,
        device: &Device,
    ) -> Result<Self> {
        let heads = vec![
            Box::new(Head::new(
                num_embeddings,
                head_size,
                block_size,
                dropout_rate,
                var_map,
                device
            )?);
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
            &self
                .heads
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

pub struct FeedForward {
    /// a simple linear layer followed by a non-linearity
    net: Sequential,
}

impl FeedForward {
    pub fn new(
        num_embeddings: usize,
        dropout_rate: f32,
        var_map: &VarMap,
        device: &Device,
    ) -> Result<Self> {
        let mut net = sequential::seq();
        net = net.add(linear(
            num_embeddings,
            FEED_FORWARD_OUT_SCALE * num_embeddings,
            VarBuilder::from_varmap(var_map, DType::F32, device),
        )?);
        net = net.add(Activation::Relu);
        net = net.add(linear(
            FEED_FORWARD_OUT_SCALE * num_embeddings,
            num_embeddings,
            VarBuilder::from_varmap(var_map, DType::F32, device),
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
        var_map: &VarMap,
        device: &Device,
    ) -> Result<Self> {
        // n_embd: embedding dimension, n_head: the number of heads we'd like
        let multi_head_attention = MultiHeadAttention::new(
            num_embeddings,
            num_heads,
            head_size,
            block_size,
            dropout_rate,
            var_map,
            device,
        )?;
        let feed_forward = FeedForward::new(num_embeddings, dropout_rate, var_map, device)?;
        let layer_normalization1 = layer_norm(
            num_embeddings,
            LayerNormConfig::from(EPS),
            VarBuilder::from_varmap(var_map, DType::F32, device),
        )?;
        let layer_normalization2 = layer_norm(
            num_embeddings,
            LayerNormConfig::from(EPS),
            VarBuilder::from_varmap(var_map, DType::F32, device),
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
    vocab_size: usize,
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
        let token_embedding_table = embedding(
            vocab_size,
            num_embeddings,
            VarBuilder::from_varmap(&var_map, DType::F32, device),
        )?;
        let position_embedding_table = embedding(
            vocab_size,
            num_embeddings,
            VarBuilder::from_varmap(&var_map, DType::F32, device),
        )?;
        let mut blocks = sequential::seq();
        let head_size = num_embeddings / num_heads;
        for _ in 0..num_blocks {
            blocks.add(Block::new(
                num_embeddings,
                num_heads,
                head_size,
                block_size,
                dropout_rate,
                &var_map,
                device,
            ));
        }
        let layer_normalization_final = layer_norm(
            num_embeddings,
            LayerNormConfig::from(EPS),
            VarBuilder::from_varmap(&var_map, DType::F32, device),
        )?; // final layer norm
        let linear_head_final = linear(
            num_embeddings,
            vocab_size,
            VarBuilder::from_varmap(&var_map, DType::F32, device),
        )?;
        let rng = rand::thread_rng();

        Ok(Self {
            vocab_size,
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
            let generated_ids_cond = generated_ids
                .iter()
                .skip(generated_ids.len() - block_size)
                .cloned()
                .collect();
            let logits = self.forward(&Tensor::from_vec(
                generated_ids_cond,
                Shape::from(i),
                device,
            )?)?;
            // focus only on the last time step
            let most_recent_logits = logits.i((i - 1, ..))?; // becomes (B, C)
                                                             // apply softmax to get probabilities
            let probabilities = ops::softmax(&most_recent_logits, 0)?;
            // sample from the distribution
            let next_token = self.sample_multinomial(&probabilities.to_vec1())?;
            // append sampled index to the running sequence
            generated_ids.push(next_token);
        }

        Ok(generated_ids)
    }
}

impl Module for BigramLanguageModel {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (_, time_size, _) = xs.shape().dims3()?;

        // xs and targets are both (B,T) tensor of integers
        let token_embedding = self.token_embedding_table.forward(xs)?; // (B,T,C)
        let position_embedding =
            self.position_embedding_table
                .forward(&Tensor::arange(0, time_size, xs.device())?)?; // (T,C)
        let x_embed_sum = token_embedding.broadcast_add(&position_embedding)?; // (B,T,C)
        let x_blocks = self.blocks.forward(&x_embed_sum)?; // (B,T,C)
        let x_norm = self.layer_normalization_final(&x_blocks)?; // (B,T,C)
        let logits = self.linear_head_final(&x_norm); // (B,T,vocab_size)

        logits
    }
}
