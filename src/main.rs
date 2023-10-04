use std::vec;

use nekonet::{layer::{self, criterion::{self, Criterion}, Layer}, tensor::Tensor, optimizer::{self, Optimizer}};

fn main() {
    let mut input = vec![];
    let mut target = vec![];

    for _ in 0..20 {
        let x = rand::random::<f32>() * 10.;
        input.push(x);
        target.append(&mut vec![1.0, 0.0]);
    }

    for _ in 0..20 {
        let x = rand::random::<f32>() * 10. + 10.;
        input.push(x);
        target.append(&mut vec![0.0, 1.0]);
    }

    let input = Tensor::new(input, vec![40, 1]);
    let target = Tensor::new(target, vec![40, 2]);

    let fc1 = layer::Linear::new(1, 4);
    let relu1 = layer::activation::ReLU();
    let fc2 = layer::Linear::new(4, 2);
    // let relu2 = layer::activation::ReLU();
    // let fc3 = layer::Linear::new(2, 2);
    let softmax = layer::activation::Softmax();

    let out = fc1.output(input.clone());
    let out = relu1.output(out.clone());
    let out = fc2.output(out.clone());
    // let out = relu2.output(out.clone());
    // let out = fc3.output(out.clone());
    let pred = softmax.output(out.clone());

    let loss = criterion::CrossEntropyLoss::new();
    let loss = loss.output(pred.clone(), target.clone());

    let mut optimizer = optimizer::SGD::new(0.05);
    optimizer.add_params(fc1.params());
    optimizer.add_params(fc2.params());

    loss.all_require_grad(true);
    input.require_grad(false);
    loss.all_init_grad();

    for _ in 0..50 {
        loss.forward();
        loss.all_zero_grad();
        loss.one_grad();
        loss.backward().unwrap();

        dbg!(&pred);
        // dbg!(&m3);
        // dbg!(&m3.input_tensors());
        // dbg!(&fc2);
        // dbg!(&m2);
        // dbg!(&m1);
        // dbg!(&fc1);
        // dbg!(&input);
        

        optimizer.step();
    }

    // fc1.weight().dbg();
    // fc2.weight().dbg();
    // m2.dbg();
    // pred.dbg();

    // for tensor in pred.input_tensors() {
    //     tensor.dbg();
    // }

}
