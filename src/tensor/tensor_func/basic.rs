use std::{ops::Range, borrow::BorrowMut};

use ndarray::{concatenate, ArrayD, ArrayViewD, Axis, Ix2, Zip};

use crate::{
    into_2d,
    tensor::{types::*, Tensor},
};

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

pub struct SliceAxis {
    input: Tensor,

    axis: usize,

    indices: Range<usize>,
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
            let mut input_grad = self.0.grad().unwrap().borrow_mut();
            input_grad.zip_mut_with(output_grad, |x, y| *x += y);
        }

        if self.1.is_require_grad() {
            let mut input_grad = self.1.grad().unwrap().borrow_mut();
            input_grad.zip_mut_with(output_grad, |x, y| *x += y);
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
        let mut input_grad = self.0.grad().unwrap().borrow_mut();
        input_grad.zip_mut_with(output_grad, |x, y| *x += -y);
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
        let mut input_grad = self.0.grad().unwrap().borrow_mut();
        let input = &self.0.data().borrow();
        ndarray::Zip::from(input_grad.view_mut())
            .and(input.view())
            .and(output_grad)
            .for_each(|x, y, z| *x += -z * y.recip().powf(2.));
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
            let mut scalar_grad = self.scalar.grad().unwrap().borrow_mut();
            let tensor = &self.tensor.data().borrow();
            ndarray::Zip::from(tensor.view())
                .and(output_grad)
                .for_each(|y, z| scalar_grad[0] += z * y);
        }

        if self.tensor.is_require_grad() {
            let mut tensor_grad = self.tensor.grad().unwrap().borrow_mut();
            let scalar = &self.scalar.data().borrow()[0];
            ndarray::Zip::from(tensor_grad.view_mut())
                .and(output_grad)
                .for_each(|x, y| *x += y * scalar);
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
        Zip::from(base.view())
            .and(exponent.view())
            .map_collect(|x, y| x.powf(*y))
    }

    fn renew_grad(&self, _output_data: &Data, output_grad: &Grad) {
        let base = &self.base.data().borrow();
        let exponent = &self.exponent.data().borrow();

        if self.base.is_require_grad() {
            let mut base_grad = self.base.grad().unwrap().borrow_mut();
            ndarray::Zip::from(base_grad.view_mut())
                .and(base.view())
                .and(exponent.view())
                .and(output_grad)
                .for_each(|x, y, z, w| *x += w * z * y.powf(z - 1.));
        }

        if self.exponent.is_require_grad() {
            let mut exponent_grad = self.exponent.grad().unwrap().borrow_mut();
            ndarray::Zip::from(exponent_grad.view_mut())
                .and(base.view())
                .and(exponent.view())
                .and(output_grad)
                .for_each(|x, y, z, w| *x += w * y.powf(*z) * y.ln());
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
        let mut input_grad = self.0.grad().unwrap().borrow_mut();
        let input = &self.0.data().borrow();
        ndarray::Zip::from(input_grad.view_mut())
            .and(input.view())
            .and(output_grad)
            .for_each(|x, y, z| *x += z / y);
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
        let left_data = self.0.data().borrow();
        let right_data = self.1.data().borrow();
        let (left, right) = into_2d!(left_data.view(), right_data.view());
        left.dot(&right).into_dyn()
    }

    fn renew_grad(&self, _output_data: &Data, output_grad: &Grad) {
        let (output_grad, left, right) = into_2d!(
            output_grad.view(),
            self.0.data().borrow().clone(),
            self.1.data().borrow().clone()
        );

        if self.0.is_require_grad() {
            let mut left_grad = self.0.grad().unwrap().borrow_mut();
            let adder = output_grad.dot(&right.t());
            *left_grad = left_grad.to_owned() + adder;
        }

        if self.1.is_require_grad() {
            let mut right_grad = self.1.grad().unwrap().borrow_mut();
            let adder = left.t().dot(&output_grad);
            *right_grad = right_grad.to_owned() + adder;
        }
    }

    fn tensors(&self) -> Vec<Tensor> {
        vec![self.0.clone(), self.1.clone()]
    }
}

impl SliceAxis {
    pub fn new(input: Tensor, axis: usize, indices: Range<usize>) -> SliceAxis {
        assert!(axis < input.shape().len(), "axis out of range");
        assert!(
            indices.end <= input.shape()[axis],
            "index out of range, index: {}, shape: {:?}",
            indices.end,
            input.shape()
        );

        SliceAxis {
            input,
            axis,
            indices,
        }
    }
}

impl TensorFunc for SliceAxis {
    fn output_shape(&self) -> Shape {
        let mut output_shape = self.input.shape().clone();
        output_shape[self.axis] = 1;
        output_shape
    }

    fn cal_output_data(&self) -> Data {
        self.input
            .data()
            .borrow()
            .slice_axis(Axis(self.axis), self.indices.clone().into())
            .to_owned()
    }

    fn renew_grad(&self, _output_data: &Data, output_grad: &Grad) {
        let mut input_grad = self.input.grad().unwrap().borrow_mut();
        input_grad
            .slice_axis_mut(Axis(self.axis), self.indices.clone().into())
            .zip_mut_with(output_grad, |x, y| *x += y);
    }

    fn tensors(&self) -> Vec<Tensor> {
        vec![self.input.clone()]
    }
}

impl Concat {
    pub fn new(inputs: Vec<Tensor>, axis: usize) -> Concat {
        assert!(inputs.len() > 0, "inputs should not be empty");
        assert!(axis < inputs[0].shape().len(), "axis out of range");

        Concat { inputs, axis }
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
        let mut input_iter = self.inputs.iter().map(|x| x.data().borrow());
        let input0 = input_iter.next().unwrap();
        let mut output = input0.clone();

        for input in input_iter {
            output.append(Axis(self.axis), input.view()).unwrap();
        }

        output
    }

    fn renew_grad(&self, _output_data: &Data, output_grad: &Grad) {
        let mut start = 0;
        for input in self.inputs.iter() {
            let input_shape = input.shape();
            let end = start + input_shape[self.axis];

            if input.is_require_grad() {
                let mut input_grad = input.grad().unwrap().borrow_mut();
                let adder = output_grad.slice_axis(Axis(self.axis), (start..end).into());
                *input_grad = input_grad.to_owned() + adder
            }

            start = end;
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
        let output_data = self.input.data().borrow().sum();
        ArrayD::from_elem(vec![1], output_data)
    }

    fn renew_grad(&self, _output_data: &Data, output_grad: &Grad) {
        let mut input_grad = self.input.grad().unwrap().borrow_mut();
        input_grad.mapv_inplace(|x| x + output_grad[0]);
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
        let mut input_grad = self.input.grad().unwrap().borrow_mut();
        let numbers = input_grad.len() as f32;
        input_grad.mapv_inplace(|x| x + output_grad[0] / numbers);
    }

    fn tensors(&self) -> Vec<Tensor> {
        vec![self.input.clone()]
    }
}

pub struct Debugger {
    input: Tensor, 
    marker: usize
}

impl Debugger {
    pub fn new(tensor: Tensor, marker: usize) -> Debugger {
        Debugger {
            input: tensor,
            marker
        }
    }
}

impl TensorFunc for Debugger {
    fn output_shape(&self) -> Shape {
        self.input.shape().clone()
    }

    fn cal_output_data(&self) -> Data {
        let data = self.input.data().borrow();
        if data.iter().any(|x| x.is_nan()) {
            println!("{:?}", self.marker);
            panic!("NaN detected");
        } else {
            data.clone()
        }
    }

    fn renew_grad(&self, _output_data: &Data, output_grad: &Grad) {
        let mut input_grad = self.input.grad().unwrap().borrow_mut();
        input_grad.zip_mut_with(output_grad, |x, y| *x += y);
    }
    
    fn tensors(&self) -> Vec<Tensor> {
        vec![self.input.clone()]
    }
}