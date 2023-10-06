pub mod data_util;
pub mod graph;
pub mod layer;
pub mod optimizer;
pub mod tensor;

pub mod prelude {
    pub use crate::{
        data_util::{DataLoader, Dataset},
        graph::Graph,
        layer::{
            activation::{ReLU, Softmax},
            criterion::{Criterion, CrossEntropyLoss},
            Layer, Linear,
        },
        optimizer::{self, Optimizer},
        tensor::{types::*, Tensor},
    };
}
