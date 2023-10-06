use crate::{
    layer::Layer,
    tensor::{self, Tensor},
};

pub struct Builder {
    outputs: Vec<Tensor>,
}

impl Builder {
    pub fn new() -> Self {
        Builder {
            outputs: Vec::new(),
        }
    }

    pub fn add_output(mut self, output: Tensor) -> Self {
        self.outputs.push(output);
        self
    }

    pub fn add_outputs(mut self, outputs: Vec<Tensor>) -> Self {
        self.outputs.extend(outputs);
        self
    }

    pub fn build(self) -> Graph {
        Graph::from_outputs(self.outputs)
    }
}

pub struct Graph {
    // topological order, from output to input
    tensors: Vec<Tensor>,

    outputs: Vec<Tensor>,
}

impl Graph {
    pub fn from_output(output: Tensor) -> Self {
        Graph::from_outputs(vec![output])
    }

    pub fn from_outputs(outputs: Vec<Tensor>) -> Self {
        let graph = Graph {
            tensors: tensor::all_tensors_topological(outputs.clone()),
            outputs,
        };
        graph.init_grad();
        graph
    }

    pub fn zero_grad(&self) {
        for tensor in &self.tensors {
            tensor.zero_grad();
        }
    }

    pub fn init_grad(&self) {
        for tensor in &self.tensors {
            tensor.init_grad();
        }
    }

    pub fn forward(&self) {
        // in reversed order
        for tensor in self.tensors.iter().rev() {
            tensor.forward();
        }
    }

    pub fn backward(&self) {
        // in original order
        for tensor in &self.outputs {
            tensor.one_grad();
        }

        for tensor in self.tensors.iter() {
            tensor.backward();
        }
    }
}
