use candle_core::{Device, Tensor, Error};

fn tensor_example() -> Result<String, Error> {
    let data: [u32; 3] = [1u32, 2, 3];
    let tensor = Tensor::new(&data, &Device::Cpu)?;
    println!("tensor: {:?}", tensor.to_vec1::<u32>()?);

    let nested_data: [[u32; 3]; 3] = [[1u32, 2, 3], [4, 5, 6], [7, 8, 9]];
    let nested_tensor = Tensor::new(&nested_data, &Device::Cpu)?;
    println!("nested_tensor: {:?}", nested_tensor.to_vec2::<u32>()?);

    Ok("Ok".to_string())
}

fn main() {
    match tensor_example() {
        Ok(message) => println!("Terminated with {}", message),
        Err(error) => eprintln!("Error: {}", error)
    }
}


