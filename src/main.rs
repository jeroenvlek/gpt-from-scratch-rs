use candle_core::{Device, Tensor};

fn main() {
    let data: [u32; 3] = [1u32, 2, 3];
    let tensor = Tensor::new(&data, &Device::Cpu).unwrap();
    println!("tensor: {:?}", tensor.to_vec1::<u32>().unwrap());

    let nested_data: [[u32; 3]; 3] = [[1u32, 2, 3], [4, 5, 6], [7, 8, 9]];
    let nested_tensor = Tensor::new(&nested_data, &Device::Cpu).unwrap();
    println!("nested_tensor: {:?}", nested_tensor.to_vec2::<u32>().unwrap());
}


