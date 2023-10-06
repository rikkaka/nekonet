use rand::seq::SliceRandom;

use crate::tensor::types::Data;

pub trait Dataset {
    fn len(&self) -> usize;
    fn get(&self, index: usize) -> (Data, Data);
}

pub struct DataLoader<'a, D: Dataset> {
    dataset: &'a D,
    batch_size: usize,
    shuffle: bool,
}

impl<'a, D: Dataset> DataLoader<'a, D> {
    pub fn new(dataset: &'a D, batch_size: usize, shuffle: bool) -> Self {
        DataLoader {
            dataset,
            batch_size,
            shuffle,
        }
    }

    pub fn iter(&self) -> DataLoaderIter<D> {
        let mut indexs: Vec<usize> = (0..self.dataset.len()).collect();
        if self.shuffle {
            indexs.shuffle(&mut rand::thread_rng());
        };

        DataLoaderIter {
            dataloader: self,
            current_index: 0,
            indexs,
        }
    }
}

pub struct DataLoaderIter<'a, 'b, D: Dataset> {
    dataloader: &'a DataLoader<'b, D>,
    current_index: usize,
    indexs: Vec<usize>,
}

impl<'a, 'b, D: Dataset> Iterator for DataLoaderIter<'a, 'b, D> {
    type Item = (Data, Data);

    fn next(&mut self) -> Option<Self::Item> {
        let mut batch_input = Vec::new();
        let mut batch_target = Vec::new();
        for i in 0..self.dataloader.batch_size {
            let index = self.current_index + i;
            if index >= self.dataloader.dataset.len() {
                return None;
            }
            let (input, target) = self.dataloader.dataset.get(self.indexs[index]);
            batch_input.extend(input);
            batch_target.extend(target);
        }
        self.current_index += self.dataloader.batch_size;
        Some((batch_input, batch_target))
    }
}
