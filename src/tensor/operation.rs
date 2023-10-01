use super::{
    tensor_func::{Add, Concat, Ln, MatMul, Opposite, Pow, Reciprocal, ScalarMul, Slice},
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

pub fn slice(tensor: Tensor, start: usize, end: usize) -> Tensor {
    Tensor::from_input(Slice::new(tensor, start, end))
}

pub fn split_row(tensor: Tensor, row: usize) -> Tensor {
    let shape = tensor.shape();
    let start = row * shape[1..].iter().product::<usize>();
    let end = (row + 1) * shape[1..].iter().product::<usize>();
    let tensor = slice(tensor, start, end);
    tensor
        .reshape(
            vec![1]
                .into_iter()
                .chain(shape[1..].iter().cloned())
                .collect(),
        )
        .unwrap();
    tensor
}

pub fn split_rows(tensor: Tensor) -> Vec<Tensor> {
    let shape = tensor.shape();
    let mut result = Vec::new();
    for i in 0..shape[0] {
        result.push(split_row(tensor.clone(), i));
    }
    result
}

pub fn concat(tensors: Vec<Tensor>) -> Tensor {
    Tensor::from_input(Concat::new(tensors))
}
