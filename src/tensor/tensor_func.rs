pub mod basic;
pub mod compound;

use super::{types::*, Tensor};

pub(crate) trait TensorFunc {
    fn output_shape(&self) -> Shape;
    fn cal_output_data(&self) -> Data;
    fn renew_grad(&self, output_data: &Data, output_grad: &Grad);
    fn tensors(&self) -> Vec<Tensor>;

    fn forward_prev(&self) {
        for tensor in self.tensors() {
            tensor.forward();
        }
    }

    fn forward(&self, output: Tensor) {
        let mut output_data = output.data().borrow_mut();
        *output_data = self.cal_output_data();
    }

    fn backward(&self, output: Tensor) {
        if self.inputs_require_grad().iter().all(|x| !x) || !output.is_require_grad() {
            return;
        }

        let output_data = output.data().borrow();
        let output_grad = output.grad().unwrap().borrow();
        self.renew_grad(&output_data, &output_grad);
    }

    fn inputs_require_grad(&self) -> Vec<bool> {
        let mut inputs_require_grad = Vec::new();
        for tensor in self.tensors() {
            inputs_require_grad.push(tensor.is_require_grad());
        }
        inputs_require_grad
    }
}

#[derive(Default)]
pub struct Head {}

impl TensorFunc for Head {
    fn output_shape(&self) -> Shape {
        vec![]
    }
    fn cal_output_data(&self) -> Data {
        panic!("Head should not be called")
    }
    fn renew_grad(&self, _: &Data, _: &Grad) {}
    fn forward(&self, _: Tensor) {}
    fn tensors(&self) -> Vec<Tensor> {
        vec![]
    }
}
