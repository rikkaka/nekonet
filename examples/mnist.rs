use mnist::*;

use ndarray::{Array1, ArrayD};
use nekonet::prelude::*;

static BATCH_SIZE: usize = 20;
static EPOCH: usize = 100;

fn main() {
    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(50_000)
        .validation_set_length(10_000)
        .finalize();

    let dataset_train = MnistDataset::new(trn_img, trn_lbl);
    let dataset_test = MnistDataset::new(tst_img, tst_lbl.clone());

    let dataloader_train = DataLoader::new(&dataset_train, BATCH_SIZE, true);
    let dataloader_test = DataLoader::new(&dataset_test, 10_000, false);

    let input_placeholder = Tensor::zeros(vec![BATCH_SIZE, 784]).no_grad();
    let target_placeholder = Tensor::zeros(vec![BATCH_SIZE, 10]).no_grad();

    let bp = Bp::new();
    let pred = bp.pred(input_placeholder.clone());
    let loss = bp.loss(pred.clone(), target_placeholder.clone());
    let graph_train = Graph::from_output(loss.clone());

    let mut optimizer = optimizer::SGD::new(0.001);
    optimizer.add_params(bp.params());

    let (input_data, target_data) = dataloader_test.iter().next().unwrap();

    let input_test = Tensor::new(input_data.clone(), vec![10_000, 784]).no_grad();
    let target_test = Tensor::new(target_data.clone(), vec![10_000, 10]).no_grad();
    let pred_test = bp.pred(input_test.clone());
    let loss_test = bp.loss(pred_test.clone(), target_test.clone());

    let gragh_test = Graph::from_output(loss_test.clone());

    for i in 0..EPOCH {
        for (input_batch, target_batch) in dataloader_train.iter() {
            input_placeholder.set_data(input_batch.clone());
            target_placeholder.set_data(target_batch.clone());

            graph_train.forward();
            graph_train.zero_grad();
            graph_train.backward();

            // assert!(!&loss.data().borrow()[0].is_nan());

            optimizer.step();
        }

        dbg!(&loss);

        gragh_test.forward();
        let loss_test_data = loss_test.data().borrow().clone();
        let pred_class = predict(&pred_test.data().borrow());
        let acc = accuracy(&pred_class.into_raw_vec(), &tst_lbl);
        println!("epoch: {}, loss: {}, acc: {}", i, loss_test_data[0], acc);
    }
}

struct MnistDataset {
    img: Vec<f32>,
    lbl: Vec<f32>,
}

impl MnistDataset {
    fn new(img: Vec<u8>, lbl: Vec<u8>) -> Self {
        let img = img.into_iter().map(|x| x as f32 / 255.).collect();
        let lbl = lbl
            .into_iter()
            .flat_map(|x| {
                let mut v = vec![0.; 10];
                v[x as usize] = 1.;
                v
            })
            .collect();

        Self { img, lbl }
    }
}

impl Dataset for MnistDataset {
    fn len(&self) -> usize {
        self.lbl.len() / 10
    }

    fn get(&self, index: usize) -> (RawData, RawData) {
        let img = self.img[index * 784..(index + 1) * 784].to_vec();
        let lbl = self.lbl[index * 10..(index + 1) * 10].to_vec();
        (img, lbl)
    }
}
struct Bp {
    fc1: Linear,
    fc2: Linear,
    fc3: Linear,

    relu: ReLU,
    softmax: Softmax,

    criterion: CrossEntropyLoss,
}

impl Bp {
    fn new() -> Self {
        let fc1 = Linear::new(784, 100);
        let fc2 = Linear::new(100, 10);
        let fc3 = Linear::new(10, 10);

        let relu = ReLU();
        let softmax = Softmax();

        let criterion = CrossEntropyLoss::new();

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
        let pred = self.softmax.output(out.clone());
        pred
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

fn predict(pred_onehot: &ArrayD<f32>) -> Array1<u8> {
    let mut pred_class = Array1::zeros(pred_onehot.shape()[0]);
    for (i, row) in pred_onehot.rows().into_iter().enumerate() {
        let mut max = 0.;
        let mut max_index = 0;
        for (j, &x) in row.iter().enumerate() {
            if x > max {
                max = x;
                max_index = j;
            }
        }
        pred_class[i] = max_index as u8;
    }
    pred_class
}

fn accuracy(pred: &Vec<u8>, target: &Vec<u8>) -> f32 {
    let mut correct = 0;
    for i in 0..pred.len() {
        if pred[i] == target[i] {
            correct += 1;
        }
    }
    correct as f32 / pred.len() as f32
}
