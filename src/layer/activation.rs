use crate::tensor::{operation, tensor_func::compound};

use super::{Layer, Tensor};

#[derive(Default)]
pub struct ReLU();

impl ReLU {
    pub fn output(&self, input: Tensor) -> Tensor {
        Tensor::from_input(compound::ReLU::new(input))
    }
}

impl Layer for ReLU {
    // fn outputs(&self, input: Vec<Tensor>) -> Vec<Tensor> {
    //     input.into_iter().map(|x| self.output(x)).collect()
    // }
}

#[derive(Default)]
pub struct Softmax();

impl Softmax {
    pub fn output(&self, input: Tensor) -> Tensor {
        let output = operation::split(input.clone(), 0);
        let output = output
            .into_iter()
            .map(|x| Tensor::from_input(compound::Softmax::new(x)))
            .collect::<Vec<_>>();
        operation::concat(output, 0)
    }
}

impl Layer for Softmax {
    // fn outputs(&self, input: Vec<Tensor>) -> Vec<Tensor> {
    //     input.into_iter().map(|x| self.output(x)).collect()
    // }
}
