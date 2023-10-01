use crate::tensor::{
    operation::{concat, split_rows},
    tensor_func::TensorFunc,
    Data, Grad, Shape, Tensor,
};

struct ReLU(Tensor);
struct Softmax(Tensor);

impl ReLU {
    fn new(tensor: Tensor) -> ReLU {
        ReLU(tensor)
    }
}

impl Softmax {
    fn new(tensor: Tensor) -> Softmax {
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

    fn cal_grad(&self, _output_data: &Data, output_grad: &Grad) {
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

    fn cal_grad(&self, output_data: &Data, output_grad: &Grad) {
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

pub fn relu(input: Tensor) -> Tensor {
    Tensor::from_input(ReLU::new(input))
}

pub fn softmax(input: Tensor) -> Tensor {
    let mut output = split_rows(input);
    for i in 0..output.len() {
        output[i] = Tensor::from_input(Softmax::new(output[i].clone()));
    }
    concat(output)
}
