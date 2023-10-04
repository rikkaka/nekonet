pub mod activation;
pub mod criterion;

use crate::tensor::{
    operation::{add, concat, matmul, split_rows},
    Tensor,
};

pub trait Layer {
    fn params(&self) -> Vec<Tensor> {
        Vec::new()
    }
}

#[derive(Clone, Debug)]
pub struct Linear {
    weight: Tensor,
    bias: Tensor,
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
        let output = matmul(input, self.weight.clone());
        let mut output = split_rows(output);
        for i in 0..output.len() {
            output[i] = add(output[i].clone(), self.bias.clone());
        }
        concat(output)
    }
}

impl Layer for Linear {
    fn params(&self) -> Vec<Tensor> {
        vec![self.weight.clone(), self.bias.clone()]
    }
}

