use super::{
    tensor_func::basic::{
        Add, Concat, Ln, MatMul, Mean, Opposite, Pow, Reciprocal, ScalarMul, Slice, Sum,
    },
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

pub fn mul_scalar(tensor: Tensor, scalar: Tensor) -> Tensor {
    Tensor::from_input(ScalarMul::new(tensor, scalar))
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

pub fn slice(tensor: Tensor, axis: usize, index: usize) -> Tensor {
    Tensor::from_input(Slice::new(tensor, axis, index))
}

pub fn split(tensor: Tensor, axis: usize) -> Vec<Tensor> {
    let shape = tensor.shape();
    let mut result = Vec::new();
    for i in 0..shape[axis] {
        result.push(slice(tensor.clone(), axis, i));
    }
    result
}

pub fn concat(tensors: Vec<Tensor>, axis: usize) -> Tensor {
    Tensor::from_input(Concat::new(tensors, axis))
}

pub fn sum(input: Tensor) -> Tensor {
    Tensor::from_input(Sum::new(input))
}

pub fn mean(input: Tensor) -> Tensor {
    Tensor::from_input(Mean::new(input))
}

pub fn var(input: Tensor, input_mean: Tensor) -> Tensor {
    let input = add(input, opposite(input_mean));
    let exp = Tensor::scalar(2.0).no_grad();
    let input = pow(input, exp);
    mean(input)
}

pub fn std(input: Tensor, input_mean: Tensor) -> Tensor {
    let input_var = var(input.clone(), input_mean);
    let exp = Tensor::scalar(0.5).no_grad();
    pow(input_var, exp)
}
