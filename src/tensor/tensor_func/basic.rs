use crate::tensor::{types::*, Tensor};

use super::TensorFunc;

pub struct Add(Tensor, Tensor);
pub struct Opposite(Tensor);
pub struct Reciprocal(Tensor);
pub struct ScalarMul(Tensor, Tensor);
pub struct Pow {
    base: Tensor,
    exponent: Tensor,
}
pub struct Ln(Tensor);
pub struct MatMul(Tensor, Tensor);

pub struct Slice {
    input: Tensor,

    start: usize,
    end: usize,
}

pub struct Concat {
    inputs: Vec<Tensor>,
}

pub struct Sum {
    input: Tensor,
}

pub struct Mean {
    input: Tensor,
}

impl Add {
    pub fn new(left: Tensor, right: Tensor) -> Add {
        assert_eq!(
            left.shape(),
            right.shape(),
            "shape mismatch, left: {:?}, right: {:?}",
            left.shape(),
            right.shape()
        );
        Add(left, right)
    }
}

impl Opposite {
    pub fn new(tensor: Tensor) -> Opposite {
        Opposite(tensor)
    }
}

impl Reciprocal {
    pub fn new(tensor: Tensor) -> Reciprocal {
        Reciprocal(tensor)
    }
}

impl ScalarMul {
    pub fn new(left: Tensor, right: Tensor) -> ScalarMul {
        assert_eq!(
            left.shape(),
            right.shape(),
            "shape mismatch, left: {:?}, right: {:?}",
            left.shape(),
            right.shape()
        );
        ScalarMul(left, right)
    }
}

impl Pow {
    pub fn new(base: Tensor, exponent: Tensor) -> Pow {
        assert_eq!(base.shape(), exponent.shape(), "shape mismatch");
        Pow { base, exponent }
    }
}

impl Ln {
    pub fn new(tensor: Tensor) -> Ln {
        Ln(tensor)
    }
}

impl MatMul {
    pub fn new(left: Tensor, right: Tensor) -> MatMul {
        assert_eq!(left.shape().len(), 2, "left shape should be 2-d");
        assert_eq!(right.shape().len(), 2, "right shape should be 2-d");
        assert_eq!(
            left.shape()[1],
            right.shape()[0],
            "shape mismatch, left: {:?}, right: {:?}",
            left.shape(),
            right.shape()
        );
        MatMul(left, right)
    }
}

impl TensorFunc for Add {
    fn output_shape(&self) -> Shape {
        self.0.shape().clone()
    }

    fn cal_output_data(&self) -> Data {
        let left = &self.0.data().borrow();
        let right = &self.1.data().borrow();
        left.iter().zip(right.iter()).map(|(l, r)| l + r).collect()
    }

    fn renew_grad(&self, _output_data: &Data, output_grad: &Grad) {
        if self.0.is_require_grad() {
            let mut left_grad = self.0.grad().unwrap().borrow_mut();
            for i in 0..output_grad.len() {
                left_grad[i] += output_grad[i];
            }
            drop(left_grad);
        }

        if self.1.is_require_grad() {
            let mut right_grad = self.1.grad().unwrap().borrow_mut();
            for i in 0..output_grad.len() {
                right_grad[i] += output_grad[i];
            }
            drop(right_grad);
        }
    }

    fn tensors(&self) -> Vec<Tensor> {
        vec![self.0.clone(), self.1.clone()]
    }
}

impl TensorFunc for Opposite {
    fn output_shape(&self) -> Shape {
        self.0.shape().clone()
    }

    fn cal_output_data(&self) -> Data {
        let input = &self.0.data().borrow();
        input.iter().map(|x| -x).collect()
    }

    fn renew_grad(&self, _output_data: &Data, output_grad: &Grad) {
        let mut input_grad = self.0.grad().unwrap().borrow_mut();
        for i in 0..output_grad.len() {
            input_grad[i] += -output_grad[i];
        }
        drop(input_grad);
    }

    fn tensors(&self) -> Vec<Tensor> {
        vec![self.0.clone()]
    }
}

impl TensorFunc for Reciprocal {
    fn output_shape(&self) -> Shape {
        self.0.shape().clone()
    }

    fn cal_output_data(&self) -> Data {
        let input = &self.0.data().borrow();
        input.iter().map(|x| 1.0 / x).collect()
    }

    fn renew_grad(&self, _output_data: &Data, output_grad: &Grad) {
        let mut input_grad = self.0.grad().unwrap().borrow_mut();

        for i in 0..output_grad.len() {
            input_grad[i] +=
                -output_grad[i] / (self.0.data().borrow()[i] * self.0.data().borrow()[i]);
        }
        drop(input_grad);
    }

    fn tensors(&self) -> Vec<Tensor> {
        vec![self.0.clone()]
    }
}

impl TensorFunc for ScalarMul {
    fn output_shape(&self) -> Shape {
        self.0.shape().clone()
    }

    fn cal_output_data(&self) -> Data {
        let left = &self.0.data().borrow();
        let right = &self.1.data().borrow();
        left.iter().zip(right.iter()).map(|(l, r)| l * r).collect()
    }

    fn renew_grad(&self, _output_data: &Data, output_grad: &Grad) {
        if self.0.is_require_grad() {
            let mut left_grad = self.0.grad().unwrap().borrow_mut();
            let right = &self.1.data().borrow();

            for i in 0..output_grad.len() {
                left_grad[i] += output_grad[i] * right[i];
            }
            drop(left_grad);
        }

        if self.1.is_require_grad() {
            let mut right_grad = self.1.grad().unwrap().borrow_mut();
            let left = &self.0.data().borrow();

            for i in 0..output_grad.len() {
                right_grad[i] += output_grad[i] * left[i];
            }
            drop(right_grad);
        }
    }

    fn tensors(&self) -> Vec<Tensor> {
        vec![self.0.clone(), self.1.clone()]
    }
}

impl TensorFunc for Pow {
    fn output_shape(&self) -> Shape {
        self.base.shape().clone()
    }

    fn cal_output_data(&self) -> Data {
        let base = &self.base.data().borrow();
        let exponent = &self.exponent.data().borrow();
        base.iter()
            .zip(exponent.iter())
            .map(|(b, e)| b.powf(*e))
            .collect()
    }

    fn renew_grad(&self, _output_data: &Data, output_grad: &Grad) {
        if self.base.is_require_grad() {
            let mut base_grad = self.base.grad().unwrap().borrow_mut();
            let base = &self.base.data().borrow();
            let exponent = &self.exponent.data().borrow();

            for i in 0..output_grad.len() {
                base_grad[i] += output_grad[i] * exponent[i] * base[i].powf(exponent[i] - 1.0);
            }
            drop(base_grad);
        }

        if self.exponent.is_require_grad() {
            let mut exponent_grad = self.exponent.grad().unwrap().borrow_mut();
            let base = &self.base.data().borrow();
            let exponent = &self.exponent.data().borrow();

            for i in 0..output_grad.len() {
                exponent_grad[i] += output_grad[i] * base[i].ln() * base[i].powf(exponent[i]);
            }
            drop(exponent_grad);
        }
    }

    fn tensors(&self) -> Vec<Tensor> {
        vec![self.base.clone(), self.exponent.clone()]
    }
}

impl TensorFunc for Ln {
    fn output_shape(&self) -> Shape {
        self.0.shape().clone()
    }

    fn cal_output_data(&self) -> Data {
        let input = &self.0.data().borrow();
        input.iter().map(|x| x.ln()).collect()
    }

    fn renew_grad(&self, _output_data: &Data, output_grad: &Grad) {
        let mut input_grad = self.0.grad().unwrap().borrow_mut();
        for i in 0..output_grad.len() {
            input_grad[i] += output_grad[i] / self.0.data().borrow()[i];
        }
        drop(input_grad);
    }

    fn tensors(&self) -> Vec<Tensor> {
        vec![self.0.clone()]
    }
}

impl TensorFunc for MatMul {
    fn output_shape(&self) -> Shape {
        vec![self.0.shape()[0], self.1.shape()[1]]
    }

    fn cal_output_data(&self) -> Data {
        let left = &self.0.data().borrow();
        let right = &self.1.data().borrow();
        let mut output = Vec::new();
        output.resize(self.output_shape().iter().product(), 0.0);

        let left_shape = self.0.shape();
        let right_shape = self.1.shape();

        matmul(left, &left_shape, right, &right_shape)
    }

    fn renew_grad(&self, _output_data: &Data, output_grad: &Grad) {
        if self.0.is_require_grad() {
            let mut left_grad = self.0.grad().unwrap().borrow_mut();
            let right = &self.1.data().borrow();

            let right_shape = self.1.shape();

            let (right_t, right_t_shape) = transpose(right, &right_shape);
            *left_grad = matmul(output_grad, &self.output_shape(), &right_t, &right_t_shape);
            drop(left_grad);
        }

        if self.1.is_require_grad() {
            let mut right_grad = self.1.grad().unwrap().borrow_mut();
            let left = &self.0.data().borrow();

            let left_shape = self.0.shape();

            let (left_t, left_t_shape) = transpose(left, &left_shape);
            *right_grad = matmul(&left_t, &left_t_shape, output_grad, &self.output_shape());
            drop(right_grad);
        }
    }

    fn tensors(&self) -> Vec<Tensor> {
        vec![self.0.clone(), self.1.clone()]
    }
}

fn matmul(left: &Data, left_shape: &Shape, right: &Data, right_shape: &Shape) -> Data {
    let mut output = Vec::new();
    output.resize(left_shape[0] * right_shape[1], 0.0);

    for i in 0..left_shape[0] {
        for j in 0..right_shape[1] {
            for k in 0..left_shape[1] {
                output[i * right_shape[1] + j] +=
                    left[i * left_shape[1] + k] * right[k * right_shape[1] + j];
            }
        }
    }

    output
}

fn transpose(data: &Data, shape: &Shape) -> (Data, Shape) {
    let mut output = Vec::new();
    output.resize(shape.iter().product(), 0.0);
    let output_shape = vec![shape[1], shape[0]];

    for i in 0..shape[0] {
        for j in 0..shape[1] {
            output[j * shape[0] + i] = data[i * shape[1] + j];
        }
    }

    (output, output_shape)
}

impl Slice {
    pub fn new(tensor: Tensor, start: usize, end: usize) -> Slice {
        assert!(start < end, "start should be less than end");
        assert!(
            end <= tensor.shape().iter().product(),
            "end should be less than or equal to multiplication of shape"
        );
        Slice {
            input: tensor,
            start,
            end,
        }
    }
}

impl TensorFunc for Slice {
    fn output_shape(&self) -> Shape {
        vec![self.end - self.start]
    }

    fn cal_output_data(&self) -> Data {
        let input = &self.input.data().borrow();
        input[self.start..self.end].to_vec()
    }

    fn renew_grad(&self, _output_data: &Data, output_grad: &Grad) {
        let mut input_grad = self.input.grad().unwrap().borrow_mut();
        for i in 0..output_grad.len() {
            input_grad[self.start + i] += output_grad[i];
        }
        drop(input_grad);
    }

    fn tensors(&self) -> Vec<Tensor> {
        vec![self.input.clone()]
    }
}

impl Concat {
    pub fn new(inputs: Vec<Tensor>) -> Concat {
        assert!(inputs.len() > 0, "inputs should not be empty");
        let shape = inputs[0].shape();
        for i in 1..inputs.len() {
            assert_eq!(
                inputs[i].shape(),
                shape,
                "shape mismatch, inputs[0]: {:?}, inputs[{}]: {:?}",
                shape,
                i,
                inputs[i].shape()
            );
        }
        Concat { inputs }
    }
}

impl TensorFunc for Concat {
    fn output_shape(&self) -> Shape {
        let mut output_shape = self.inputs[0].shape().clone();
        output_shape[0] = 0;
        for tensor in &self.inputs {
            output_shape[0] += tensor.shape()[0];
        }
        output_shape
    }

    fn cal_output_data(&self) -> Data {
        let mut output = Vec::new();
        for tensor in &self.inputs {
            output.extend_from_slice(&tensor.data().borrow());
        }
        output
    }

    fn renew_grad(&self, _output_data: &Data, output_grad: &Grad) {
        let mut start = 0;
        for tensor in &self.inputs {
            if tensor.is_require_grad() {
                let mut input_grad = tensor.grad().unwrap().borrow_mut();
                for i in 0..tensor.shape().iter().product() {
                    input_grad[i] += output_grad[start + i];
                }
                drop(input_grad);
            }
            start += tensor.shape().iter().product::<usize>();
        }
    }

    fn tensors(&self) -> Vec<Tensor> {
        self.inputs.clone()
    }
}

impl Sum {
    pub fn new(input: Tensor) -> Sum {
        Sum { input }
    }
}

impl TensorFunc for Sum {
    fn output_shape(&self) -> Shape {
        vec![1]
    }

    fn cal_output_data(&self) -> Data {
        let input = &self.input.data().borrow();
        let mut sum = 0.0;
        for i in 0..input.len() {
            sum += input[i];
        }
        vec![sum]
    }

    fn renew_grad(&self, _output_data: &Data, output_grad: &Grad) {
        let mut input_grad = self.input.grad().unwrap().borrow_mut();
        for i in 0..input_grad.len() {
            input_grad[i] += output_grad[0];
        }
        drop(input_grad);
    }

    fn tensors(&self) -> Vec<Tensor> {
        vec![self.input.clone()]
    }
}

impl Mean {
    pub fn new(input: Tensor) -> Mean {
        Mean { input }
    }
}

impl TensorFunc for Mean {
    fn output_shape(&self) -> Shape {
        vec![1]
    }

    fn cal_output_data(&self) -> Data {
        let input = &self.input.data().borrow();
        let mut sum = 0.0;
        for i in 0..input.len() {
            sum += input[i];
        }
        vec![sum / input.len() as f32]
    }

    fn renew_grad(&self, _output_data: &Data, output_grad: &Grad) {
        let mut input_grad = self.input.grad().unwrap().borrow_mut();
        for i in 0..input_grad.len() {
            input_grad[i] += output_grad[0] / input_grad.len() as f32;
        }
        drop(input_grad);
    }

    fn tensors(&self) -> Vec<Tensor> {
        vec![self.input.clone()]
    }
}
