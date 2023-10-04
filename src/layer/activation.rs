use crate::tensor::{operation, tensor_func::compound};

use super::Tensor;

#[derive(Default)]
pub struct ReLU();

impl ReLU {
    pub fn output(&self, input: Tensor) -> Tensor {
        Tensor::from_input(compound::ReLU::new(input))
    }
}

#[derive(Default)]
pub struct Softmax();

impl Softmax {
    pub fn output(&self, input: Tensor) -> Tensor {
        let output = operation::split_rows(input.clone());
        let output = output
            .into_iter()
            .map(|x| Tensor::from_input(compound::Softmax::new(x)))
            .collect::<Vec<_>>();
        operation::concat(output)
    }
}
