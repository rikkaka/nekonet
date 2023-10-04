use super::types::*;

fn cal_index(shape: &Shape, coordinate: &Coordinate) -> usize {
    let mut index = 0;
    for i in 0..shape.len() {
        index *= shape[i];
        index += coordinate[i];
    }
    index
}
