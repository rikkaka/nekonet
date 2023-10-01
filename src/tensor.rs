pub mod activation;
pub mod layer;
pub mod operation;
pub mod tensor_func;

use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};

use anyhow::{anyhow, Result};

use self::tensor_func::{Head, TensorFunc};

pub type Data = Vec<f32>;
pub type Shape = Vec<usize>;
pub type Grad = Vec<f32>;

#[derive(Clone, Debug, Default)]
pub struct Tensor {
    inner: Rc<TensorInner>,
}

pub(crate) struct TensorInner {
    data: RefCell<Data>,
    shape: RefCell<Shape>,
    grad: RefCell<Grad>,
    require_grad: Cell<bool>,
    input: RefCell<Box<dyn TensorFunc>>,
    forwarded: Cell<bool>,
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

    pub fn is_forwarded(&self) -> &Cell<bool> {
        &self.inner.forwarded
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

    pub fn set_grad_1(&self) {
        self.inner.set_grad(1.);
    }

    pub fn dbg(&self) {
        self.inner.dbg();
        println!("===== input tensors: ======");
        for tensor in self.input_tensors() {
            tensor.inner.dbg();
        }
        println!("==== end input tensors ====");
    }

    pub fn forward(&self) {
        self.inner
            .input
            .borrow()
            .forward(self.data().borrow_mut(), &self.is_forwarded());
        self.inner.forwarded.set(true);
    }

    pub fn backward(&self) -> Result<()> {
        if !self.is_require_grad() {
            return Ok(());
        }

        self.inner
            .input
            .borrow_mut()
            .backward(&self.data().borrow(), &self.grad()?.borrow())?;
        Ok(())
    }

    pub fn input_tensors(&self) -> Vec<Tensor> {
        self.inner.input.borrow().tensors()
    }

    pub fn unforwarded(&self) {
        self.inner.unforwarded();
    }

    pub fn all_unforwarded(&self) {
        self.inner.unforwarded();
        for tensor in self.input_tensors() {
            tensor.inner.unforwarded();
        }
    }

    pub fn reshape(&self, shape: Shape) -> Result<()> {
        self.inner.reshape(shape)
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
            forwarded: Cell::new(false),
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
            return Err(anyhow!("grad is not initialized, {:?}", self));
        }
        Ok(&self.grad)
    }

    fn unforwarded(&self) {
        self.forwarded.set(false);
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
