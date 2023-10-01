use crate::tensor::Tensor;

pub struct Linear {
    weight: Tensor,
    bias: Tensor,

    previous: Tensor
}

pub struct ReLU {
    previous: Tensor
}

pub struct Sigmoid {
    previous: Tensor
}

pub struct Softmax {
    previous: Tensor
}

impl Linear {
    pub fn new(input_dim: usize, output_dim: usize) -> Linear {
        Linear {
            weight: Tensor::random(vec![input_dim, output_dim]),
            bias: Tensor::random(vec![output_dim]),
            previous: Tensor::empty(vec![0])
        }
    }
}