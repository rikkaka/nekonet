pub mod operation;
pub mod tensor_func;
pub mod types;

mod util;

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
        write!(f, "Tensor(inner_addr={:p}, \ninner={:?})", addr, self.inner)
    }
}

pub(crate) struct TensorInner {
    data: RefCell<Data>,
    shape: RefCell<Shape>,
    grad: RefCell<Grad>,
    require_grad: Cell<bool>,
    input: RefCell<Box<dyn TensorFunc>>,
}

impl Tensor {
    pub fn new(data: Data, shape: Shape) -> Tensor {
        Tensor {
            inner: Rc::new(TensorInner::new(data, shape)),
        }
    }

    pub fn empty(shape: Shape) -> Tensor {
        let mut data = Vec::new();
        data.resize(shape.iter().product(), 0.0);
        Tensor::new(data, shape)
    }

    pub fn random(shape: Shape) -> Tensor {
        let mut data = Vec::new();
        data.resize(shape.iter().product(), 0.0);
        for i in 0..data.len() {
            data[i] = rand::random::<f32>() * 2.0 - 1.0;
        }
        Tensor::new(data, shape)
    }

    pub(crate) fn from_input<TF: TensorFunc + 'static>(input: TF) -> Tensor {
        Tensor {
            inner: Rc::new(TensorInner::from_input(input)),
        }
    }

    pub fn shape(&self) -> Shape {
        self.inner.shape.borrow().clone()
    }

    pub fn data(&self) -> &RefCell<Data> {
        &self.inner.data
    }

    pub fn grad(&self) -> Result<&RefCell<Grad>> {
        self.inner.grad()
    }

    pub fn require_grad(&self, val: bool) {
        self.inner.require_grad(val);
    }

    pub fn is_require_grad(&self) -> bool {
        self.inner.is_require_grad()
    }

    pub fn all_require_grad(&self, val: bool) {
        self.require_grad(val);
        for tensor in self.input_tensors() {
            tensor.all_require_grad(val);
        }
    }

    pub fn init_grad(&self) {
        self.inner.init_grad();
    }

    pub fn all_init_grad(&self) {
        if self.is_require_grad() {
            self.init_grad();
        }
        for tensor in self.input_tensors() {
            tensor.all_init_grad();
        }
    }

    pub fn one_grad(&self) {
        self.inner.set_grad(1.);
    }

    pub fn zero_grad(&self) {
        self.inner.set_grad(0.);
    }

    pub fn all_zero_grad(&self) {
        if self.is_require_grad() {
            self.zero_grad();
            for tensor in self.input_tensors() {
                tensor.all_zero_grad();
            }
        }
    }

    pub fn dbg(&self) {
        self.inner.dbg();
        // println!("===== input tensors: ======");
        // for tensor in self.input_tensors() {
        //     tensor.inner.dbg();
        // }
        // println!("==== end input tensors ====");
    }

    pub fn forward(&self) {
        let mut visited = HashSet::new();
        self.forward_inner(&mut visited);
    }

    pub fn forward_inner(&self, visited: &mut HashSet<Tensor>) {
        if visited.contains(self) {
            return;
        }
        visited.insert(self.clone());

        for tensor in self.input_tensors() {
            tensor.forward_inner(visited);
        }

        self.inner
            .input
            .borrow_mut()
            .forward(self.clone());
    }

    pub fn backward(&self) -> Result<()> {
        let mut q = VecDeque::from(vec![self.clone()]);
        let mut visited = HashSet::new();

        while !q.is_empty() {
            let tensor = q.pop_front().unwrap();
            if visited.contains(&tensor) {
                continue;
            }
            visited.insert(tensor.clone());

            tensor.backward_inner();
            for tensor in tensor.input_tensors() {
                q.push_back(tensor);
            }
        }
        Ok(())
    }
    
    // renew the grads of all input tensors
    fn backward_inner(&self) {
        self.inner.input
            .borrow_mut()
            .backward(self.clone());
    }

    pub fn input_tensors(&self) -> Vec<Tensor> {
        self.inner.input.borrow().tensors()
    }

    pub fn reshape(&self, shape: Shape) -> Result<()> {
        self.inner.reshape(shape)
    }

    pub fn all_tensors(&self) -> Vec<Tensor> {
        let mut visited = HashSet::new();
        let mut tensors = Vec::new();
        let mut q = VecDeque::from(vec![self.clone()]);

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
}

impl Default for TensorInner {
    fn default() -> TensorInner {
        TensorInner {
            data: RefCell::new(Vec::new()),
            shape: RefCell::new(Vec::new()),
            grad: RefCell::new(Vec::new()),
            require_grad: Cell::new(false),
            input: RefCell::new(Box::new(Head::default())),
        }
    }
}

impl std::fmt::Debug for TensorInner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let data = self.data.borrow();
        let shape = self
            .shape
            .borrow()
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<String>>()
            .join("x");
        write!(
            f,
            "Tensor(shape={}, \ndata={:?}, \ngrad={:?}, \nrequire_grad={})",
            shape,
            data,
            self.grad.borrow(),
            self.is_require_grad()
        )
    }
}

impl TensorInner {
    fn new(data: Data, shape: Shape) -> TensorInner {
        assert_eq!(data.len(), shape.iter().product());

        let data = RefCell::new(data);
        let shape = RefCell::new(shape);
        TensorInner {
            data,
            shape,
            ..Default::default()
        }
    }

    fn from_input<TF: TensorFunc + 'static>(input: TF) -> TensorInner {
        let shape = input.output_shape();

        let mut data = Vec::new();
        data.resize(shape.iter().product(), 0.0);
        let data = RefCell::new(data);
        let shape = RefCell::new(shape);

        let input = RefCell::new(Box::new(input) as Box<dyn TensorFunc>);

        TensorInner {
            data,
            shape,
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
        grad.resize(self.data.borrow().len(), 0.0);
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

    fn reshape(&self, shape: Shape) -> Result<()> {
        if self.shape.borrow().iter().product::<usize>() != shape.iter().product::<usize>() {
            return Err(anyhow!(
                "cannot reshape tensor of shape {:?} to {:?}",
                self.shape,
                shape
            ));
        }

        *self.shape.borrow_mut() = shape;
        Ok(())
    }

    fn dbg(&self) {
        dbg!(self);
    }
}

#[cfg(test)]
mod test {
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

        assert_eq!(out.all_tensors().len(), 6);
    }
}
