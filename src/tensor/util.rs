use itertools::Itertools;

use super::types::*;

pub(crate) struct IndexCalculator {
    shape: Shape,
}

impl IndexCalculator {
    pub fn new(shape: Shape) -> IndexCalculator {
        IndexCalculator { shape }
    }

    pub fn cal_index(&self, coords: &Coords) -> usize {
        let mut index = 0;
        let mut multiplier = 1;

        for i in (0..self.shape.len()).rev() {
            index += coords[i] * multiplier;
            multiplier *= self.shape[i];
        }

        index
    }

    pub fn cal_slice_indexs(&self, axis: usize, index_at_axis: usize) -> Vec<usize> {
        if axis == 0 {
            let len: usize = self.shape[1..].iter().product();
            return (index_at_axis * len..(index_at_axis + 1) * len).collect();
        }

        self.shape
            .iter()
            .enumerate()
            .map(|(i, &dim)| {
                if i == axis {
                    index_at_axis..index_at_axis + 1
                } else {
                    0..dim
                }
            })
            .multi_cartesian_product()
            .map(|coords| self.cal_index(&coords))
            .collect()
    }
}

#[cfg(test)]
mod test_index_calculator {
    use super::*;

    #[test]
    fn test_cal_index() {
        let index_calculator = IndexCalculator::new(vec![2, 3, 4]);
        assert_eq!(index_calculator.cal_index(&vec![0, 0, 0]), 0);
        assert_eq!(index_calculator.cal_index(&vec![0, 0, 1]), 1);
        assert_eq!(index_calculator.cal_index(&vec![0, 1, 0]), 4);
        assert_eq!(index_calculator.cal_index(&vec![0, 1, 2]), 6);
        assert_eq!(index_calculator.cal_index(&vec![0, 2, 3]), 11);
        assert_eq!(index_calculator.cal_index(&vec![1, 1, 3]), 19);
        assert_eq!(index_calculator.cal_index(&vec![1, 2, 3]), 23);
    }

    #[test]
    fn test_cal_slice_indexs() {
        let index_calculator = IndexCalculator::new(vec![2, 3, 4]);
        assert_eq!(
            index_calculator.cal_slice_indexs(1, 2),
            vec![8, 9, 10, 11, 20, 21, 22, 23]
        );
    }
}
