use candle_core::{D, Device, DType, IndexOp, Module, Shape, Tensor};
use candle_nn::{linear_no_bias, ops, VarBuilder, VarMap};

const HEAD_SIZE: usize = 16;

pub fn self_attention_examples(device: &Device) -> candle_core::Result<()> {
    toy_example_weighted_aggregation(device)?;

    // consider the following toy example:
    let dims = (4, 8, 2); // batch, time, channels
    let x = Tensor::rand(0f32, 10f32, Shape::from(dims), device)?;
    println!("x.shape: {:?}", x.shape());

    let x_bag_of_words = example_1_bag_of_words(&x)?;

    let x_bag_of_words2 = example_2_bow_mat_mul(&x)?;
    println!(
        "allclose: {}",
        all_close(&x_bag_of_words, &x_bag_of_words2)?
    );

    let x_bag_of_words3 = example_3_softmax(&x)?;
    println!(
        "allclose: {}",
        all_close(&x_bag_of_words, &x_bag_of_words3)?
    );

    example_4_self_attention(device)?;

    scaled_attention_example(device)?;

    Ok(())
}

fn toy_example_weighted_aggregation(device: &Device) -> candle_core::Result<()> {
    // toy example illustrating how matrix multiplication can be used for a "weighted aggregation"
    let mut a = Tensor::tril2(3, DType::F32, device)?;
    let sum_a = a.sum_keepdim(1)?;
    a = a.broadcast_div(&sum_a)?;
    println!("A: {:?}", a.to_vec2::<f32>());

    let b = Tensor::rand(0.0f32, 10.0f32, Shape::from((3, 2)), device)?;
    println!("B: {:?}", b.to_vec2::<f32>());

    let c = a.matmul(&b)?;
    println!("C: {:?}", c.to_vec2::<f32>());

    Ok(())
}

fn example_1_bag_of_words(x: &Tensor) -> candle_core::Result<Tensor> {
    // We want x[b,t] = mean_{i<=t} x[b,i]
    let dims = x.shape().dims();
    let mut means: Vec<Tensor> = Vec::with_capacity(dims[0] * dims[1] * dims[2]);
    for b_idx in 0..dims[0] {
        for t_idx in 0..dims[1] {
            let x_prev = x.i((b_idx, 0..t_idx + 1, ..))?;
            let mean = x_prev.mean_keepdim(0)?;
            means.push(mean);
        }
    }
    let x_bag_of_words = Tensor::stack(means.as_slice(), 1)?.reshape(Shape::from(dims))?;
    println!("xbow: {:?}", x_bag_of_words.to_vec3::<f32>());
    println!("xbow shape: {:?}", x_bag_of_words.shape());

    Ok(x_bag_of_words)
}

fn example_2_bow_mat_mul(x: &Tensor) -> candle_core::Result<Tensor> {
    // version 2: using matrix multiply for a weighted aggregation
    let dims = x.shape().dims();
    let mut weights = Tensor::tril2(dims[1], DType::F32, x.device())?;
    let sum_weights = weights.sum_keepdim(1)?;
    weights = weights.broadcast_div(&sum_weights)?;
    let x_bag_of_words2 = weights.broadcast_matmul(&x)?;
    println!("xbow2: {:?}", x_bag_of_words2.to_vec3::<f32>());
    println!("xbow2 shape: {:?}", x_bag_of_words2.shape());

    Ok(x_bag_of_words2)
}
fn example_3_softmax(x: &Tensor) -> candle_core::Result<Tensor> {
    // version 3: use Softmax
    let T = x.shape().dims()[1];
    let mut neg_inf = Tensor::from_vec(
        vec![f32::NEG_INFINITY; T * T],
        Shape::from((T, T)),
        x.device(),
    )?;
    let mut weights = Tensor::tril2(T, DType::U32, x.device())?
        .where_cond(&Tensor::tril2(T, DType::F32, x.device())?, &neg_inf)?;
    println!("wei: {:?}", weights.to_vec2::<f32>());
    weights = ops::softmax(&weights, 1)?;
    println!("wei: {:?}", weights.to_vec2::<f32>());
    let x_bag_of_words3 = weights.broadcast_matmul(&x)?;
    println!("xbow3: {:?}", x_bag_of_words3.to_vec3::<f32>());
    println!("xbow3 shape: {:?}", x_bag_of_words3.shape());

    Ok(x_bag_of_words3)
}

fn example_4_self_attention(device: &Device) -> candle_core::Result<Tensor> {
    // version 4: self-attention!
    let (b, t, c) = (4usize, 8usize, 32usize); // batch, time, channels
    let x = Tensor::randn(0f32, 1f32, Shape::from((b, t, c)), device)?;

    // let's see a single Head perform self-attention
    let var_map = VarMap::new();
    let key = linear_no_bias(
        c,
        HEAD_SIZE,
        VarBuilder::from_varmap(&var_map, DType::F32, device),
    )?;
    let query = linear_no_bias(
        c,
        HEAD_SIZE,
        VarBuilder::from_varmap(&var_map, DType::F32, device),
    )?;
    let value = linear_no_bias(
        c,
        HEAD_SIZE,
        VarBuilder::from_varmap(&var_map, DType::F32, device),
    )?;
    let k = key.forward(&x)?;
    let q = query.forward(&x)?;
    let mut weights = q.matmul(&k.transpose(D::Minus2, D::Minus1)?)?; // (B, T, 16) @ (B, 16, T) ---> (B, T, T)
    println!("wei.shape: {:?}", weights.shape());

    let neg_inf = Tensor::try_from(f32::NEG_INFINITY)?.broadcast_as(weights.shape())?;

    let masked_fill = Tensor::tril2(t, DType::U32, device)?.broadcast_as(weights.shape())?.where_cond(&weights, &neg_inf)?;
    println!("masked_fill.shape: {:?}", masked_fill.shape());
    println!("masked_fill: {:?}", masked_fill.to_vec3::<f32>());
    weights = ops::softmax(&masked_fill, D::Minus1)?;

    let v = value.forward(&x)?;
    let out = weights.matmul(&v)?;
    println!("out.shape: {:?}", out.shape());

    println!("wei[0]: {:?}", weights.i(0)?.to_vec2::<f32>());

    Ok(out)
}

fn scaled_attention_example(device: &Device) -> candle_core::Result<()> {
    let (B, T) = (4usize, 8usize); // batch, time
    let k = Tensor::randn(0f32, 1f32, Shape::from((B, T, HEAD_SIZE)), device)?;
    let q = Tensor::randn(0f32, 1f32, Shape::from((B, T, HEAD_SIZE)), device)?;
    let weights = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? * (HEAD_SIZE as f64).powf(-0.5))?;

    println!("k.var(): {}", k.flatten_all()?.var(0)?);
    println!("q.var(): {}", q.flatten_all()?.var(0)?);
    println!("wei.var(): {}", weights.flatten_all()?.var(0)?);

    let tensor = Tensor::from_vec(vec![0.1, -0.2, 0.3, -0.2, 0.5], Shape::from(5), device)?;
    println!("softmax(tensor([0.1, -0.2, 0.3, -0.2, 0.5])): {}", ops::softmax(&tensor, D::Minus1)?);
    println!("softmax(tensor([0.1, -0.2, 0.3, -0.2, 0.5]) * 8): {}", ops::softmax(&(tensor * 8f64)?, D::Minus1)?);

    Ok(())
}

fn all_close(lhs: &Tensor, rhs: &Tensor) -> candle_core::Result<bool> {
    // JV: interesting to see different precisions between the different xbows, rounding fixes it, but should investigate further
    // seems like Candle does f64 by default given the * operator being overloaded only for f64
    let element_compare = lhs.round_to(4)?.eq(&rhs.round_to(4)?)?.sum_all()?;
    Ok(element_compare.to_vec0::<u8>()? == lhs.shape().elem_count() as u8)
}
