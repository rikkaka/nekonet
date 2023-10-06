pub mod activation;
pub mod criterion;

use crate::tensor::{
    operation::{add, concat, matmul},
    Tensor,
};

pub trait Layer {
    fn params(&self) -> Vec<Tensor> {
        Vec::new()
    }
    fn leaf_tensors(&self) -> Vec<Tensor> {
        Vec::new()
    }
    // fn outputs(&self, input: Vec<Tensor>) -> Vec<Tensor>;
}

#[derive(Clone, Debug)]
pub struct Linear {
    weight: Tensor,
    bias: Tensor,
}

impl Layer for Linear {
    fn params(&self) -> Vec<Tensor> {
        vec![self.weight.clone(), self.bias.clone()]
    }

    // fn outputs(&self, input: Vec<Tensor>) -> Vec<Tensor> {
    //     assert_eq!(input.len(), 1, "Linear layer only accept one input");
    //     vec![self.output(input[0].clone())]

    // }
}

impl Linear {
    pub fn new(input_dim: usize, output_dim: usize) -> Linear {
        Linear {
            weight: Tensor::random(vec![input_dim, output_dim]),
            bias: Tensor::random(vec![1, output_dim]),
        }
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    pub fn bias(&self) -> &Tensor {
        &self.bias
    }

    pub fn output(&self, input: Tensor) -> Tensor {
        let batch_size = input.shape()[0];
        let output = matmul(input, self.weight.clone());
        let biases = concat((0..batch_size).map(|_| self.bias.clone()).collect(), 0);
        add(output, biases)
    }
}

// for shape of [batch_size, dims]
#[derive(Clone, Debug)]
pub struct BatchNormPlain {
    gamma: Tensor,
    beta: Tensor,

    running_mean: Option<Tensor>,
    running_std: Option<Tensor>,

    momentum: f32,
}

impl Layer for BatchNormPlain {
    fn params(&self) -> Vec<Tensor> {
        vec![self.gamma.clone(), self.beta.clone()]
    }

    fn leaf_tensors(&self) -> Vec<Tensor> {
        vec![
            self.running_mean.clone().unwrap(),
            self.running_std.clone().unwrap(),
        ]
    }
}

impl BatchNormPlain {
    pub fn new(num_features: usize, momentum: f32) -> Self {
        assert!(momentum >= 0.0 && momentum <= 1.0);
        BatchNormPlain {
            gamma: Tensor::ones(vec![num_features]),
            beta: Tensor::zeros(vec![num_features]),
            running_mean: None,
            running_std: None,
            momentum,
        }
    }

    // pub fn output(&self, input: Tensor, is_eval: bool) -> Vec<Tensor> {
    //     assert!(input.shape().len() == 2, "input shape must be [batch_size, dims]");

    //     let batch_size = input.shape()[0];
    //     let num_features = input.shape()[1];

    //     let outputs = split(input, 1).iter().map(|tensor| {
    //         let mean = mean(tensor.clone());
    //         let std = std(tensor.clone(), mean);
    //         let output = add(tensor.clone(), opposite(mean));
    //         let output = mul_scalar(output, reciprocal(std));
    //     });
    // }
}