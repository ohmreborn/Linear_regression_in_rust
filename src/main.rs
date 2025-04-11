use candle_core::{Tensor, Device, DType, Shape};
use candle_core::utils::{cuda_is_available, metal_is_available};
use candle_nn::{linear, Linear, VarBuilder, VarMap, SGD, Module, loss, Optimizer};

use std::path::PathBuf;

fn get_batches(x: Tensor, y: Tensor, batch_size: usize) -> Result<Vec<(Tensor, Tensor)>, Box<dyn std::error::Error>> {
    let n_samples = x.dims()[0];
    let mut batches: Vec<(Tensor, Tensor)> = Vec::new();

    for i in (0..n_samples).step_by(batch_size) {
        let end = (i + batch_size).min(n_samples);
        let x_batch = x.narrow(0, i, end - i)?;
        let y_batch = y.narrow(0, i, end - i)?;
        batches.push((x_batch, y_batch));
    }

    Ok(batches)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // choose device
    let device = if cuda_is_available() {
        Device::new_cuda(0)?
    } else if metal_is_available() {
        Device::new_metal(0)?
    } else {
        Device::Cpu
    };
    let batch_size:usize = 3;

    // create a data
    let n_sample:f32 = 100.0;
    let div: Tensor = Tensor::new(&[n_sample],&device)?;
    let mut x: Tensor = Tensor::arange(0f32, n_sample, &device)?
        .reshape((100,1))?
        .broadcast_div(&div)?;
    let m: Tensor = Tensor::new(&[[3f32]], &device)?;
    let c: Tensor = Tensor::new(&[[3f32]], &device)?;
    let noise: Tensor = Tensor::rand(0f32, 1., Shape::from((100, 1)), &device)?;
    let y: Tensor = x.broadcast_mul(&m)?
        .broadcast_add(&c)?
        .broadcast_add(&noise)?;
    let data_loader: Vec<(Tensor,Tensor)> = get_batches(x, y, batch_size)?;
    // // initialize a model optimizer
    let varmap: VarMap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model: Linear = linear(1, 1, vb)?;
    let mut optimizer: SGD = SGD::new(varmap.all_vars(), 0.01)?;

    // training a model
    for _epoch in 0..10 {
        for (x_train,y_train) in &data_loader{
            let pred: Tensor = model.forward(&x_train)?;
            let loss_res: Tensor = loss::mse(&pred,&y_train)?;
            optimizer.backward_step(&loss_res)?;
            println!("{:?}",loss_res);
        }
    }

    // println!("\n--------------------------------------------\n");
    // for p in varmap.all_vars(){
    //     println!("{}",p);
    // }
    
    // // save a model
    // let path: PathBuf = PathBuf::from("model.safetensors");
    // match varmap.save(path.clone()){
    //     Ok(()) => println!("save file {} sucess",path.display()),
    //     Err(e) => {
    //         println!("cannot save file");
    //         panic!("{}",e);
    //     }
    // };

    // println!("\n--------------------------------------------\n");

    // // load a model
    // let mut new_varmap: VarMap = VarMap::new();
    // let new_vb = VarBuilder::from_varmap(&new_varmap, DType::F32, &device);
    // let _new_model:Linear = linear(1,1,new_vb)?;
    // match new_varmap.load("model.safetensors"){
    //     Ok(()) => println!("load model sucess"),
    //     Err(e) => panic!("{}",e)
    // };
    // for p in new_varmap.all_vars(){
    //     println!("{}",p);
    // }

    Ok(())
}
