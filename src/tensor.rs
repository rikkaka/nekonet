pub mod operation;
pub mod tensor_func;
pub mod types;

mod util;

use ndarray::ArrayD;
use types::*;

use std::{
    cell::{Cell, RefCell},
    collections::VecDeque,
    fmt::Debug,
    hash::Hash,
    rc::Rc,
};

use anyhow::{anyhow, Result};

use hashbrown::HashSet;

use self::tensor_func::{Head, TensorFunc};

#[derive(Clone, Default)]
pub struct Tensor {
    inner: Rc<TensorInner>,
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.inner, &other.inner)
    }
}

impl Eq for Tensor {}

impl Hash for Tensor {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        Rc::as_ptr(&self.inner).hash(state);
    }
}

impl Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let addr = Rc::as_ptr(&self.inner);
        write!(
            f,
            "Tensor(inner_addr={:p}, \ninner={:?}, \nis_require_grad={:?})",
            addr,
            self.inner,
            self.is_require_grad()
        )
    }
}

pub(crate) struct TensorInner {
    data: RefCell<Data>,
    grad: RefCell<Data>,
    require_grad: Cell<bool>,
    input: RefCell<Box<dyn TensorFunc>>,
}

impl Tensor {
    pub fn new(data: RawData, shape: Shape) -> Tensor {
        Tensor {
            inner: Rc::new(TensorInner::new(data, shape)),
        }
    }

    pub fn scalar(val: f32) -> Tensor {
        Tensor::new(vec![val], vec![1])
    }

    pub fn empty(shape: Shape) -> Tensor {
        let mut data = Vec::new();
        data.resize(shape.iter().product(), 0.0);
        Tensor::new(data, shape)
    }

    pub fn random(shape: Shape) -> Tensor {
        let mut data = Vec::new();
        data.resize(shape.iter().product(), 0.0);
        for item in &mut data {
            *item = rand::random::<f32>() * 2.0 - 1.0;
        }
        Tensor::new(data, shape)
    }

    pub fn zeros(shape: Shape) -> Tensor {
        let mut data = Vec::new();
        data.resize(shape.iter().product(), 0.0);
        Tensor::new(data, shape)
    }

    pub fn ones(shape: Shape) -> Tensor {
        let mut data = Vec::new();
        data.resize(shape.iter().product(), 1.0);
        Tensor::new(data, shape)
    }

    pub(crate) fn from_input<TF: TensorFunc + 'static>(input: TF) -> Tensor {
        Tensor {
            inner: Rc::new(TensorInner::from_input(input)),
        }
    }

    pub fn set_data(&self, data: RawData) {
        assert_eq!(
            data.len(),
            self.inner.data.borrow().len(),
            "data length mismatch"
        );
        let shape = self.shape();
        *self.inner.data.borrow_mut() = ArrayD::from_shape_vec(shape, data).unwrap();
    }

    // pub(crate) unsafe fn set_input(&self, input: Box<dyn TensorFunc>) {
    //     *self.inner.input.borrow_mut() = input;
    // }

    pub fn shape(&self) -> Shape {
        self.inner.data.borrow().shape().to_vec()
    }

    pub fn data(&self) -> &RefCell<Data> {
        &self.inner.data
    }

    pub fn raw_data(&self) -> RawData {
        self.inner.data.borrow().clone().into_raw_vec()
    }

    pub fn grad(&self) -> Result<&RefCell<Grad>> {
        self.inner.grad()
    }

    pub fn raw_grad(&self) -> Result<RawData> {
        Ok(self.inner.grad()?.borrow().clone().into_raw_vec())
    }

    pub fn require_grad(self, val: bool) -> Tensor {
        self.inner.require_grad(val);
        self
    }

    pub fn no_grad(self) -> Tensor {
        self.inner.require_grad(false);
        self
    }

    pub fn as_require_grad(&self, tensor: &Tensor) {
        tensor.inner.require_grad(self.is_require_grad());
    }

    pub fn is_require_grad(&self) -> bool {
        self.inner.is_require_grad()
    }

    pub fn init_grad(&self) {
        if self.is_require_grad() {
            self.inner.init_grad();
        }
    }

    pub fn one_grad(&self) {
        self.inner.set_grad(1.);
    }

    pub fn zero_grad(&self) {
        self.inner.set_grad(0.);
    }

    pub fn dbg(&self) {
        self.inner.dbg();
    }

    pub fn forward(&self) {
        self.inner.input.borrow_mut().forward(self.clone());
    }

    pub fn backward(&self) {
        self.inner.input.borrow_mut().backward(self.clone());
    }

    pub fn input_tensors(&self) -> Vec<Tensor> {
        self.inner.input.borrow().tensors()
    }

    pub fn reshape(&self, shape: Shape) -> Result<()> {
        self.inner.reshape(shape)
    }
}

// return all tensors in the graph, with current tensor as the root, in topological order
pub(crate) fn all_tensors_topological(outputs: Vec<Tensor>) -> Vec<Tensor> {
    let mut visited = HashSet::new();
    let mut tensors = Vec::new();
    let mut q = VecDeque::from(outputs);

    while !q.is_empty() {
        let tensor = q.pop_front().unwrap();
        if visited.contains(&tensor) {
            continue;
        }
        visited.insert(tensor.clone());

        tensors.push(tensor.clone());
        for tensor in tensor.input_tensors() {
            q.push_back(tensor);
        }
    }
    tensors
}

impl Default for TensorInner {
    fn default() -> TensorInner {
        TensorInner {
            data: RefCell::new(ArrayD::zeros(vec![0])),
            grad: RefCell::new(ArrayD::zeros(vec![0])),
            require_grad: Cell::new(true),
            input: RefCell::new(Box::<Head>::default()),
        }
    }
}

impl std::fmt::Debug for TensorInner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Tensor(data={:?}, \ngrad={:?})",
            self.data.borrow(),
            self.grad.borrow(),
        )
    }
}

impl TensorInner {
    fn new(data: RawData, shape: Shape) -> TensorInner {
        assert_eq!(data.len(), shape.iter().product());

        let data = ArrayD::from_shape_vec(shape.clone(), data).unwrap();
        let data = RefCell::new(data);
        TensorInner {
            data,
            ..Default::default()
        }
    }

    fn from_input<TF: TensorFunc + 'static>(input: TF) -> TensorInner {
        let shape = input.output_shape();

        let data = ArrayD::zeros(shape);
        let data = RefCell::new(data);

        let input = RefCell::new(Box::new(input) as Box<dyn TensorFunc>);

        TensorInner {
            data,
            input,

            ..Default::default()
        }
    }

    fn require_grad(&self, val: bool) {
        self.require_grad.set(val);
    }

    fn is_require_grad(&self) -> bool {
        self.require_grad.get()
    }

    fn init_grad(&self) {
        let mut grad = self.grad.borrow_mut();
        *grad = ArrayD::zeros(self.data.borrow().shape().to_vec());
    }

    fn set_grad(&self, val: f32) {
        self.grad.borrow_mut().fill(val);
    }

    fn grad(&self) -> Result<&RefCell<Grad>> {
        if self.grad.borrow().len() == 0 {
            return Err(anyhow!("grad is not initialized"));
        }
        Ok(&self.grad)
    }

    fn reshape(&self, new_shape: Shape) -> Result<()> {
        self.data.borrow_mut().to_shape(new_shape)?;
        Ok(())
    }

    fn dbg(&self) {
        dbg!(self);
    }
}

#[cfg(test)]
mod test {
    use super::all_tensors_topological;

    use super::operation::add;
    use super::Tensor;
    #[test]
    fn test_all_tensors() {
        let x1 = Tensor::new(vec![1., 2., 3., 4., 5., 6.], vec![2, 3]);
        let x2 = Tensor::new(vec![1., 2., 3., 4., 5., 6.], vec![2, 3]);
        let y1 = add(x1.clone(), x2.clone());
        let y2 = add(x1.clone(), x2.clone());
        let z = add(y1.clone(), y2.clone());
        let out = add(z.clone(), z.clone());

        assert_eq!(all_tensors_topological(vec![out]).len(), 6);
    }
}
