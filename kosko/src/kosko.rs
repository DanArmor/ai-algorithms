use std::ops::Add;

use ndarray::{linalg::Dot, prelude::*, ShapeBuilder};

pub struct Network {
    pub w: ndarray::Array2<i32>,
}

impl Network {
    fn sign(x: i32) -> i32 {
        if x > 0 {
            1
        } else if x < 0 {
            -1
        } else {
            0
        }
    }
    pub fn new(n: usize, m: usize) -> Self {
        Self {
            w: Array::zeros((n, m).f()),
        }
    }
    pub fn train(mut self, x: &Vec<ndarray::Array1<i32>>, y: &Vec<ndarray::Array1<i32>>) -> Self {
        let x = x
            .iter()
            .map(|v| v.to_shape((1, v.len())).unwrap())
            .collect::<Vec<_>>();
        let y = y
            .iter()
            .map(|v| v.to_shape((1, v.len())).unwrap())
            .collect::<Vec<_>>();
        for i in 0..x.len() {
            self.w = self.w.add(y[i].t().dot(&x[i]));
        }
        self
    }
    fn calc(&self, v: &ndarray::Array2<i32>) -> ndarray::Array2<i32> {
        let y = self.w.dot(v);
        let x = self.w.t().dot(&y);
        let mut r = self.w.dot(&x);
        r.iter_mut().for_each(|x| *x = Network::sign(*x));
        r
    }
    pub fn predict(&self, v: &ndarray::Array1<i32>) -> Vec<i32> {
        let v = v.to_shape((v.len(), 1)).unwrap();
        let mut y = self.calc(&v.to_owned());
        for _ in 0..128 {
            let old_y = y.clone();
            y = self.calc(&v.to_owned());
            if old_y == y {
                break;
            }
        }
        y.into_raw_vec()
    }
}
