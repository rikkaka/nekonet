use ndarray::{Ix2, concatenate, Axis, ArrayD};

use crate::tensor::{types::*, Tensor};

use super::TensorFunc;

pub struct Add(Tensor, Tensor);
pub struct Opposite(Tensor);
pub struct Reciprocal(Tensor);
pub struct MulScalar {
    tensor: Tensor,
    scalar: Tensor,
}
pub struct Pow {
    base: Tensor,
    exponent: Tensor,
}
pub struct Ln(Tensor);
pub struct MatMul(Tensor, Tensor);

pub struct Slice {
    input: Tensor,

    axis: usize,

    index: usize
}

pub struct Concat {
    inputs: Vec<Tensor>,

    axis: usize,
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

impl MulScalar {
    pub fn new(tensor: Tensor, scalar: Tensor) -> MulScalar {
        assert_eq!(
            scalar.shape(),
            vec![1],
            "scalar should be a scalar, but got shape: {:?}",
            scalar.shape()
        );
        MulScalar { scalar, tensor }
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
        let left = &self.0.data().borrow().to_owned();
        let right = &self.1.data().borrow().to_owned();
        left + right
    }

    fn renew_grad(&self, _output_data: &Data, output_grad: &Grad) {
        if self.0.is_require_grad() {
            let 
        }

        if self.1.is_require_grad() {
            
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
        let input = &self.0.data().borrow().to_owned();
        -input
    }

    fn renew_grad(&self, _output_data: &Data, output_grad: &Grad) {
        
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
        input.mapv(|x| x.recip())
    }

    fn renew_grad(&self, _output_data: &Data, output_grad: &Grad) {
        
    }

    fn tensors(&self) -> Vec<Tensor> {
        vec![self.0.clone()]
    }
}

impl TensorFunc for MulScalar {
    fn output_shape(&self) -> Shape {
        self.tensor.shape().clone()
    }

    fn cal_output_data(&self) -> Data {
        let scalar = &self.scalar.data().borrow()[0];
        let tensor = &self.tensor.data().borrow();
        tensor.mapv(|x| x * scalar)
    }

    fn renew_grad(&self, _output_data: &Data, output_grad: &Grad) {
        if self.scalar.is_require_grad() {
            
        }

        if self.tensor.is_require_grad() {
            
        }
    }

    fn tensors(&self) -> Vec<Tensor> {
        vec![self.scalar.clone(), self.tensor.clone()]
    }
}

impl TensorFunc for Pow {
    fn output_shape(&self) -> Shape {
        self.base.shape().clone()
    }

    fn cal_output_data(&self) -> Data {
        let base = &self.base.data().borrow().to_owned();
        let exponent = &self.exponent.data().borrow().to_owned();
        base * *exponent
    }

    fn renew_grad(&self, _output_data: &Data, output_grad: &Grad) {
        if self.base.is_require_grad() {
            
        }

        if self.exponent.is_require_grad() {
            
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
        input.mapv(|x| x.ln())
    }

    fn renew_grad(&self, _output_data: &Data, output_grad: &Grad) {
        
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
        
    }

    fn renew_grad(&self, _output_data: &Data, output_grad: &Grad) {
        if self.0.is_require_grad() {
            
        }

        if self.1.is_require_grad() {
            
        }
    }

    fn tensors(&self) -> Vec<Tensor> {
        vec![self.0.clone(), self.1.clone()]
    }
}

impl Slice {
    pub fn new(input: Tensor, axis: usize, index: usize) -> Slice {
        assert!(axis < input.shape().len(), "axis out of range");
        assert!(
            index < input.shape()[axis],
            "index out of range, index: {}, shape: {:?}",
            index,
            input.shape()
        );

        Slice {
            input,
            axis,
            index,
        }
    }
}

impl TensorFunc for Slice {
    fn output_shape(&self) -> Shape {
        let mut output_shape = self.input.shape().clone();
        output_shape[self.axis] = 1;
        output_shape
    }

    fn cal_output_data(&self) -> Data {
        self.input
            .data()
            .borrow()
            .slice_each_axis(|ax| {
                if ax.axis.0 == self.axis {
                    ndarray::Slice::from(self.index..self.index+1)
                } else {
                    ndarray::Slice::from(..)
                }
            })
            .to_owned()
    }

    fn renew_grad(&self, _output_data: &Data, output_grad: &Grad) {
        
    }

    fn tensors(&self) -> Vec<Tensor> {
        vec![self.input.clone()]
    }
}

impl Concat {
    pub fn new(inputs: Vec<Tensor>, axis: usize) -> Concat {
        assert!(inputs.len() > 0, "inputs should not be empty");
        assert!(axis < inputs[0].shape().len(), "axis out of range");

        Concat {
            inputs,
            axis,
        }
    }
}

impl TensorFunc for Concat {
    fn output_shape(&self) -> Shape {
        let mut output_shape = self.inputs[0].shape().clone();
        output_shape[self.axis] = 0;
        for tensor in &self.inputs {
            output_shape[self.axis] += tensor.shape()[self.axis];
        }
        output_shape
    }

    fn cal_output_data(&self) -> Data {
        let inputs_data = self
            .inputs
            .iter()
            .map(|x| x.data().borrow().to_owned().view())
            .collect::<Vec<_>>();
        concatenate(Axis(self.axis), &inputs_data).unwrap()
    }

    fn renew_grad(&self, _output_data: &Data, output_grad: &Grad) {
        
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
        let output_data = self.input.data().borrow().sum();
        ArrayD::from_elem(vec![1], output_data)
    }

    fn renew_grad(&self, _output_data: &Data, output_grad: &Grad) {

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
        let output_data = self.input.data().borrow().mean().unwrap();
        ArrayD::from_elem(vec![1], output_data)
    }

    fn renew_grad(&self, _output_data: &Data, output_grad: &Grad) {

    }

    fn tensors(&self) -> Vec<Tensor> {
        vec![self.input.clone()]
    }
}
