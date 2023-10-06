use ndarray::ArrayD;

pub type RawData = Vec<f32>;
pub type Data = ArrayD<f32>;
pub type Shape = Vec<usize>;
pub type Grad = Data;
pub type Coords = Vec<usize>;
