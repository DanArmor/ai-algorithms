use std::ops::Add;

use ndarray::{linalg::Dot, prelude::*, ShapeBuilder};

pub struct Network {
    pub w: ndarray::Array2<f32>,
}

impl Network {
    fn sign(x: f32) -> f32 {
        if x > 0.0 {
            1.0
        } else if x < 0.0 {
            -1.0
        } else {
            0.0
        }
    }
    fn sign_i(x: f32) -> i32 {
        if x > 0.0 {
            1
        } else if x < 0.0 {
            -1
        } else {
            0
        }
    }
    pub fn new(n: usize) -> Self {
        Self {
            w: Array::zeros((n, n).f()),
        }
    }
    pub fn train(mut self, x: &Vec<ndarray::Array1<i32>>) -> Self {
        let n = x.len();
        let z = x
            .iter()
            .map(|v| {
                ndarray::Array2::from_shape_vec((1, v.len()), v.iter().map(|x| *x as f32).collect())
                    .unwrap()
            })
            .collect::<Vec<_>>();
        for i in 0..z.len() {
            self.w = self.w.add(z[i].t().dot(&z[i]));
        }
        self.w.diag_mut().iter_mut().for_each(|x| *x = 0.0);
        self.w.iter_mut().for_each(|x| *x = *x / n as f32);
        self
    }
    pub fn predict(&self, v: &ndarray::Array1<i32>) -> Vec<i32> {
        let v = ndarray::Array1::from_vec(v.iter().map(|x| *x as f32).collect());
        let v = v.to_shape((v.len(), 1)).unwrap();
        let mut y = self.w.dot(&v);
        y.iter_mut().for_each(|x| *x = Network::sign(*x));
        for _ in 0..128 {
            let old_y = y.clone();
            y = self.w.dot(&y);
            y.iter_mut().for_each(|x| *x = Network::sign(*x));
            println!("y:{}", y);
            if old_y == y {
                break;
            }
        }
        y.into_raw_vec()
            .iter()
            .map(|x| Network::sign_i(*x))
            .collect()
    }
}
