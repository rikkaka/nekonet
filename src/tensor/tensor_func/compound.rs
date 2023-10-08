use ndarray::{Array2, ArrayD, Zip};

use super::{Tensor, TensorFunc};
use crate::{into_2d, tensor::types::*};
pub struct ReLU(Tensor);
pub struct Softmax(Tensor);

pub struct CrossEntropy {
    input: Tensor,
    target: Tensor,
}

impl ReLU {
    pub fn new(tensor: Tensor) -> ReLU {
        ReLU(tensor)
    }
}

impl Softmax {
    pub fn new(tensor: Tensor) -> Softmax {
        Softmax(tensor)
    }
}

impl TensorFunc for ReLU {
    fn output_shape(&self) -> Shape {
        self.0.shape().clone()
    }

    fn cal_output_data(&self) -> Data {
        let input = &self.0.data().borrow();
        input.mapv(|x| if x > 0.0 { x } else { 0.0 })
    }

    fn renew_grad(&self, _output_data: &Data, output_grad: &Grad) {
        let mut input_grad = self.0.grad().unwrap().borrow_mut();
        let input = &self.0.data().borrow();

        Zip::from(&mut input_grad.view_mut())
            .and(input.view())
            .and(output_grad.view())
            .for_each(|x, y, z| *x += if *y > 0.0 { *z } else { 0.0 })
    }

    fn tensors(&self) -> Vec<Tensor> {
        vec![self.0.clone()]
    }
}

impl TensorFunc for Softmax {
    fn output_shape(&self) -> Shape {
        self.0.shape().clone()
    }

    fn cal_output_data(&self) -> Data {
        let input = &self.0.data().borrow();
        // let input = input.to_owned();
        let max_val = input.iter().fold(f32::MIN, |x, &y| x.max(y));
        let input = input.mapv(|x| x - max_val);
        let output = input.mapv(|x| {let x = x.exp();
        if x.is_infinite() {panic!("123")} else {x}});
        let sum = output.sum();
        output.mapv(|x| x / sum)
    }

    fn renew_grad(&self, output_data: &Data, output_grad: &Grad) {
        let n = output_data.len();
        let output_data_slice = output_data.as_slice().unwrap();
        let jacobian: Array2<f32> = Array2::from_shape_fn((n, n), |(i, j)| {
            if i == j {
                output_data_slice[i] * (1.0 - output_data_slice[i])
            } else {
                -output_data_slice[i] * output_data_slice[j]
            }
        });

        let mut input_grad = self.0.grad().unwrap().borrow_mut();
        let output_grad = into_2d!(output_grad.view()).0;

        let adder = output_grad.dot(&jacobian);
        input_grad.zip_mut_with(&adder, |x, y| *x += *y);
    }

    fn tensors(&self) -> Vec<Tensor> {
        vec![self.0.clone()]
    }
}

impl CrossEntropy {
    pub fn new(input: Tensor, target: Tensor) -> CrossEntropy {
        assert_eq!(input.shape(), target.shape(), "shape mismatch");
        CrossEntropy { input, target }
    }
}

impl TensorFunc for CrossEntropy {
    fn output_shape(&self) -> Shape {
        vec![1]
    }

    fn cal_output_data(&self) -> Data {
        let input = &self.input.data().borrow();
        let target = &self.target.data().borrow();

        let output = Zip::from(input.view())
            .and(target.view())
            .map_collect(|x, y| -y * (x + 1e-20).ln())
            .sum();
        ArrayD::from_elem(vec![1], output)
    }

    fn renew_grad(&self, _output_data: &Data, _output_grad: &Grad) {
        let input = &self.input.data().borrow();
        let target = &self.target.data().borrow();

        if self.input.is_require_grad() {
            let mut input_grad = self.input.grad().unwrap().borrow_mut();

            Zip::from(input_grad.view_mut())
                .and(input.view())
                .and(target.view())
                .for_each(|g, p, y| *g += -y / (p + 1e-20))
        }

        if self.target.is_require_grad() {
            unimplemented!()
        }
    }

    fn tensors(&self) -> Vec<Tensor> {
        vec![self.input.clone(), self.target.clone()]
    }
}

pub fn cross_entropy(input: Tensor, target: Tensor) -> Tensor {
    Tensor::from_input(CrossEntropy::new(input, target))
}
