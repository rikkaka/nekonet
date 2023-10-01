use std::{rc::Rc, cell::{RefCell, Cell}, vec};
use anyhow::Result;

pub struct Value {
    value: Cell<f32>,
    grad: Cell<f32>,
    input: Box<dyn ValueFunc>,
}

impl Default for Value {
    fn default() -> Self {
        Self {
            value: Cell::new(0.),
            grad: Cell::new(0.),
            input: Box::new(Head {})
        }
    }
}

trait ValueFunc {
    fn cal_output_value(&self) -> f32;
    fn renew_grad(&self, output_grad: f32);
    fn values(&self) -> Data;
}

pub struct Head {}

impl ValueFunc for Head {
    fn values(&self) -> Data {
        vec![]
    }
    fn cal_output_value(&self) -> f32 {
        panic!()
    }
    fn renew_grad(&self, _: f32) {}
}

pub type Data = Vec<Value>;
pub type Shape = Vec<usize>;
pub type Grad = Vec<f32>;

#[derive(Clone)]
pub struct Tensor {
    inner: Rc<TensorInner>,
}

pub(crate) struct TensorInner {
    data: RefCell<Data>,
    shape: Shape,
    is_require_grad: Cell<bool>,
    forwarded: Cell<bool>,
}