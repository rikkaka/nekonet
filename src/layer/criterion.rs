use crate::tensor::{operation, tensor_func::compound, Tensor};

pub trait Criterion {
    fn output(&self, predict: Tensor, target: Tensor) -> Tensor;
}

pub struct CrossEntropyLoss {
    reduction: Reduction,
}

impl CrossEntropyLoss {
    pub fn new() -> CrossEntropyLoss {
        CrossEntropyLoss {
            reduction: Reduction::Mean,
        }
    }

    pub fn reduction(mut self, reduction: Reduction) -> CrossEntropyLoss {
        self.reduction = reduction;
        self
    }
}

pub enum Reduction {
    Sum,
    Mean,
    None,
}

impl Default for Reduction {
    fn default() -> Self {
        Reduction::Mean
    }
}

impl Criterion for CrossEntropyLoss {
    fn output(&self, predict: Tensor, target: Tensor) -> Tensor {
        assert_eq!(predict.shape(), target.shape(), "shape mismatch");
        let output = operation::split(predict.clone(), 0)
            .into_iter()
            .zip(operation::split(target.clone(), 0).into_iter())
            .map(|(ip, tg)| {
                let tg = tg.no_grad();
                let output = compound::cross_entropy(ip, tg);
                output.reshape(vec![1, 1]).unwrap();
                output
            })
            .collect::<Vec<_>>();
        let output = operation::concat(output, 0);

        match self.reduction {
            Reduction::None => output,
            Reduction::Sum => operation::sum(output),
            Reduction::Mean => operation::mean(output),
        }
    }
}
