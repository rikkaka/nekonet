use crate::tensor::Tensor;

pub trait Optimizer {
    fn add_param(&mut self, param: Tensor);
    fn add_params(&mut self, params: Vec<Tensor>) {
        for param in params {
            self.add_param(param);
        }
    }

    fn step(&mut self);
}

pub struct SGD {
    lr: f32,

    params: Vec<Tensor>,
}

impl SGD {
    pub fn new(lr: f32) -> SGD {
        SGD {
            lr,
            params: Vec::new(),
        }
    }

    pub fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
}

impl Optimizer for SGD {
    fn add_param(&mut self, param: Tensor) {
        self.params.push(param);
    }

    fn step(&mut self) {
        for param in &self.params {
            let mut data = param.data().borrow_mut();
            let grad = param.grad().unwrap().borrow();
            *data = data.clone() - self.lr * grad.clone();
        }
    }
}
