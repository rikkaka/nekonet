use super::{Tensor, TensorFunc};
use crate::tensor::types::*;
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
        input.iter().map(|x| x.max(0.0)).collect()
    }

    fn renew_grad(&self, _output_data: &Data, output_grad: &Grad) {
        let mut input_grad = self.0.grad().unwrap().borrow_mut();
        let input = &self.0.data().borrow();

        for i in 0..output_grad.len() {
            input_grad[i] += output_grad[i] * if input[i] > 0.0 { 1.0 } else { 0.0 };
        }
        drop(input_grad);
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
        let mut output = Vec::new();
        let mut sum = 0.0;
        for i in 0..input.len() {
            sum += input[i].exp();
        }
        for i in 0..input.len() {
            output.push(input[i].exp() / sum);
        }
        output
    }

    fn renew_grad(&self, output_data: &Data, output_grad: &Grad) {
        let mut input_grad = self.0.grad().unwrap().borrow_mut();

        for i in 0..output_grad.len() {
            let mut sum = 0.0;
            for j in 0..output_grad.len() {
                if i == j {
                    sum += output_grad[j] * output_data[i] * (1.0 - output_data[i]);
                } else {
                    sum -= output_grad[j] * output_data[i] * output_data[j];
                }
            }
            input_grad[i] += sum;
        }
        drop(input_grad);
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

    fn cal_output_data(&self) -> super::Data {
        let input = &self.input.data().borrow();
        let target = &self.target.data().borrow();

        let mut output = 0.0;
        for i in 0..input.len() {
            output += -target[i] * (input[i] + 1e-10).ln();
        }
        vec![output]
    }

    fn renew_grad(&self, _output_data: &Data, output_grad: &Grad) {
        if self.input.is_require_grad() {
            let mut input_grad = self.input.grad().unwrap().borrow_mut();
            let input = &self.input.data().borrow();
            let target = &self.target.data().borrow();

            for i in 0..input.len() {
                input_grad[i] += -target[i] / (input[i] + 1e-10) * output_grad[0];
            }
            drop(input_grad);
        }

        if self.target.is_require_grad() {
            let mut target_grad = self.target.grad().unwrap().borrow_mut();
            let input = &self.input.data().borrow();

            for i in 0..input.len() {
                target_grad[i] += -input[i].ln() * output_grad[0];
            }
            drop(target_grad);
        }
    }

    fn tensors(&self) -> Vec<Tensor> {
        vec![self.input.clone(), self.target.clone()]
    }
}

pub fn cross_entropy(input: Tensor, target: Tensor) -> Tensor {
    Tensor::from_input(CrossEntropy::new(input, target))
}
