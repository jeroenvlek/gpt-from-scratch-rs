use candle_core::Error;
use rand::distributions::Distribution;
use rand::prelude::ThreadRng;

pub fn sample_multinomial(rng: &mut ThreadRng, prs: &Vec<f32>) -> candle_core::Result<u32> {
    let distribution = rand::distributions::WeightedIndex::new(prs).map_err(Error::wrap)?;
    let next_token = distribution.sample(rng) as u32;

    Ok(next_token)
}
