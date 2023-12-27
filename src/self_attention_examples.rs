use candle_core::{DType, Device, IndexOp, Module, Shape, Tensor, D};
use candle_nn::{linear_no_bias, ops, VarBuilder, VarMap};

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
    let mut wei = Tensor::tril2(dims[1], DType::F32, x.device())?;
    let sum_wei = wei.sum_keepdim(1)?;
    wei = wei.broadcast_div(&sum_wei)?;
    let x_bag_of_words2 = wei.broadcast_matmul(&x)?;
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
    let mut wei = Tensor::tril2(T, DType::U32, x.device())?
        .where_cond(&Tensor::tril2(T, DType::F32, x.device())?, &neg_inf)?;
    println!("wei: {:?}", wei.to_vec2::<f32>());
    wei = ops::softmax(&wei, 1)?;
    println!("wei: {:?}", wei.to_vec2::<f32>());
    let x_bag_of_words3 = wei.broadcast_matmul(&x)?;
    println!("xbow3: {:?}", x_bag_of_words3.to_vec3::<f32>());
    println!("xbow3 shape: {:?}", x_bag_of_words3.shape());

    Ok(x_bag_of_words3)
}

fn example_4_self_attention(device: &Device) -> candle_core::Result<Tensor> {
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
    let mut wei = q.matmul(&k.transpose(D::Minus2, D::Minus1)?)?; // (B, T, 16) @ (B, 16, T) ---> (B, T, T)
    println!("wei.shape: {:?}", wei.shape());

    let neg_inf = Tensor::from_vec(
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

    let v = value.forward(&x2)?;
    let out = wei.matmul(&v)?;
    println!("out.shape: {:?}", out.shape());

    println!("wei[0]: {:?}", wei.i(0)?.to_vec2::<f32>());

    Ok(out)
}

fn all_close(lhs: &Tensor, rhs: &Tensor) -> candle_core::Result<bool> {
    // JV: interesting to see different precisions between the different xbows, rounding fixes it, but should investigate further
    let element_compare = lhs.round_to(4)?.eq(&rhs.round_to(4)?)?.sum_all()?;
    Ok(element_compare.to_vec0::<u8>()? == lhs.shape().elem_count() as u8)
}
