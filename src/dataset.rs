use candle_core::{IndexOp, Result, Tensor};
use rand::rngs::ThreadRng;
use rand::Rng;

#[derive(Debug)]
pub struct Dataset {
    pub training_data: Tensor,
    pub training_size: usize,
    pub validation_data: Tensor,
    pub validation_size: usize,
    rng: ThreadRng,
}

impl Dataset {
    pub fn new(data: Tensor, training_ratio: f64) -> Self {
        let data_size = *data.shape().dims().first().unwrap();
        let training_size = (data_size as f64 * training_ratio) as usize;
        let training_data = data.i(0..training_size).unwrap();

        let validation_size = data_size - training_size;
        let validation_data = data.i(0..validation_size).unwrap();
        let rng: ThreadRng = rand::thread_rng();

        Self {
            training_data,
            training_size,
            validation_data,
            validation_size,
            rng,
        }
    }

    pub fn random_training_batch(
        &mut self,
        block_size: usize,
        batch_size: usize,
    ) -> Result<(Tensor, Tensor)> {
        let max_block_indices: Vec<usize> = (0..batch_size)
            .map(|_| self.rng.gen_range(0..self.training_size - block_size))
            .collect();

        let context_rows = max_block_indices.iter().map(|&max_index| {
            self.training_data
                .i(max_index..max_index + block_size)
                .unwrap()
        });
        let stacked_contexts = Tensor::stack(&context_rows.collect::<Vec<_>>(), 0)?;

        let target_rows = max_block_indices.iter().map(|&max_index| {
            self.training_data
                .i(max_index + 1..max_index + block_size + 1)
                .unwrap()
        });
        let stacked_targets = Tensor::stack(&target_rows.collect::<Vec<_>>(), 0)?;

        Ok((stacked_contexts, stacked_targets))
    }
}
