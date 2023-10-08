use std::vec;

use nekonet::{
    data_util::{DataLoader, Dataset},
    graph::Graph,
    layer::{
        self, activation,
        criterion::{self, Criterion},
        Layer,
    },
    optimizer::{self, Optimizer},
    prelude::RawData,
    tensor::Tensor,
};

fn main() {
    let batch_size = 20;

    let input_placeholder = Tensor::zeros(vec![batch_size, 1]).no_grad();
    let target_placeholder = Tensor::zeros(vec![batch_size, 2]).no_grad();

    let bp = Bp::new();
    let pred = bp.pred(input_placeholder.clone());
    let loss = bp.loss(pred.clone(), target_placeholder.clone());
    let graph = Graph::from_output(loss.clone());
    graph.init_grad();

    let mut optimizer = optimizer::SGD::new(0.01);
    optimizer.add_params(bp.params());

    let now = std::time::Instant::now();

    let dataset = MyDataset::new();
    let dataloader = DataLoader::new(&dataset, 20, true);

    for _ in 0..10 {
        for (input_batch, target_batch) in dataloader.iter() {
            input_placeholder.set_data(input_batch.clone());
            target_placeholder.set_data(target_batch.clone());

            graph.forward();
            graph.zero_grad();
            graph.backward();

            dbg!(loss.data().borrow()[0]);

            optimizer.step();
        }
    }

    dbg!(now.elapsed().as_millis());

    let dataset_eval = MyDataset::new();
    let dataloader = DataLoader::new(&dataset_eval, 2000, false);
    let (input_data, target_data) = dataloader.iter().next().unwrap();

    let input = Tensor::new(input_data.clone(), vec![2000, 1]);
    let target = Tensor::new(target_data.clone(), vec![2000, 2]);
    let pred = bp.pred(input.clone());
    let loss = bp.loss(pred.clone(), target.clone());

    let gragh = Graph::from_output(loss.clone());
    gragh.forward();

    dbg!(loss.data().borrow()[0]);
}

struct Bp {
    fc1: layer::Linear,
    fc2: layer::Linear,
    fc3: layer::Linear,

    relu: activation::ReLU,
    softmax: activation::Softmax,

    criterion: criterion::CrossEntropyLoss,
}

impl Bp {
    fn new() -> Self {
        let fc1 = layer::Linear::new(1, 10);
        let fc2 = layer::Linear::new(10, 6);
        let fc3 = layer::Linear::new(6, 2);

        let relu = layer::activation::ReLU();
        let softmax = layer::activation::Softmax();

        let criterion = criterion::CrossEntropyLoss::new();

        Self {
            fc1,
            fc2,
            fc3,
            relu,
            softmax,
            criterion,
        }
    }

    fn pred(&self, input: Tensor) -> Tensor {
        let out = self.fc1.output(input.clone());
        let out = self.relu.output(out.clone());
        let out = self.fc2.output(out.clone());
        let out = self.relu.output(out.clone());
        let out = self.fc3.output(out.clone());
        let out = self.softmax.output(out.clone());
        out
    }

    fn loss(&self, pred: Tensor, target: Tensor) -> Tensor {
        self.criterion.output(pred, target)
    }

    fn params(&self) -> Vec<Tensor> {
        let mut params = vec![];
        params.append(&mut self.fc1.params());
        params.append(&mut self.fc2.params());
        params.append(&mut self.fc3.params());
        params
    }
}

struct MyDataset {
    input_data: Vec<RawData>,
    target_data: Vec<RawData>,
}

impl MyDataset {
    fn new() -> Self {
        let mut input = vec![];
        let mut target = vec![];
        for _ in 0..1000 {
            let x = rand::random::<f32>() * 10.;
            input.push(vec![x]);
            target.push(vec![1.0, 0.0]);

            let x = -rand::random::<f32>() * 10.;
            input.push(vec![x]);
            target.push(vec![0.0, 1.0]);
        }

        Self {
            input_data: input,
            target_data: target,
        }
    }
}

impl Dataset for MyDataset {
    fn len(&self) -> usize {
        self.input_data.len()
    }

    fn get(&self, index: usize) -> (RawData, RawData) {
        (
            self.input_data[index].clone(),
            self.target_data[index].clone(),
        )
    }
}
