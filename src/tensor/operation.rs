use super::{
    tensor_func::{Add, Ln, MatMul, Opposite, Pow, Reciprocal, ScalarMul},
    Tensor,
};

pub fn add(left: Tensor, right: Tensor) -> Tensor {
    Tensor::from_input(Add::new(left, right))
}

pub fn opposite(tensor: Tensor) -> Tensor {
    Tensor::from_input(Opposite::new(tensor))
}

pub fn reciprocal(tensor: Tensor) -> Tensor {
    Tensor::from_input(Reciprocal::new(tensor))
}

pub fn scalar_mul(left: Tensor, right: Tensor) -> Tensor {
    Tensor::from_input(ScalarMul::new(left, right))
}

pub fn pow(base: Tensor, exponent: Tensor) -> Tensor {
    Tensor::from_input(Pow::new(base, exponent))
}

pub fn ln(tensor: Tensor) -> Tensor {
    Tensor::from_input(Ln::new(tensor))
}

pub fn matmul(left: Tensor, right: Tensor) -> Tensor {
    Tensor::from_input(MatMul::new(left, right))
}
