use crate::activation::*;
use crate::error_func::*;
use rand::distributions::{Distribution, Uniform};
use std::str::FromStr;

#[derive(Debug, PartialEq)]
pub enum Solution {
    Car,
    Heli,
    Sheep,
    Airplane,
}

#[derive(Debug, Clone)]
pub struct Sample {
    pub data: Vec<f32>,
    pub solution: Vec<f32>,
}

#[derive(Debug)]
pub struct NeuroLayer {
    raw_input: Vec<f32>,
    input: Vec<f32>,
    output: Vec<f32>,
    basis: Vec<f32>,
    weights: Vec<Vec<f32>>,
    grad: Vec<f32>,
    old_grads: Vec<Vec<f32>>,
    old_outputs: Vec<Vec<f32>>,
    pub activation: Activation,
}

impl NeuroLayer {
    pub fn new(
        neurons_amount: usize,
        back_links_amount: usize,
        activation: ActivationFunc,
    ) -> Self {
        Self {
            raw_input: vec![0.0; neurons_amount],
            input: vec![0.0; neurons_amount],
            output: vec![0.0; neurons_amount],
            basis: vec![0.001; neurons_amount],
            weights: (0..neurons_amount)
                .map(|_| {
                    (0..back_links_amount)
                        .map(|_| rand::random::<f32>())
                        .collect::<Vec<f32>>()
                })
                .collect::<Vec<_>>(),
            grad: vec![0.0; neurons_amount],
            old_grads: vec![],
            old_outputs: vec![],
            activation: Activation::new(activation),
        }
    }
    pub fn forward(&mut self, input: Vec<f32>) -> Vec<f32> {
        self.raw_input = input;
        self.input = self
            .weights
            .iter()
            .zip(self.basis.iter())
            .map(|(v, b)| {
                v.iter()
                    .zip(self.raw_input.iter())
                    .map(|(w, x)| x * w)
                    .sum::<f32>()
                    + b
            })
            .collect::<Vec<f32>>();
        self.output = self.activation.f(self.input.clone());
        self.old_outputs.push(self.output.clone());
        self.output.clone()
    }
    pub fn backward(
        &mut self,
        grad: Vec<f32>,
        weights: Vec<Vec<f32>>,
    ) -> (Vec<f32>, Vec<Vec<f32>>) {
        self.grad.fill(0.0);
        for i in 0..weights[0].len() {
            for j in 0..weights.len() {
                self.grad[i] += grad[j] * weights[j][i];
            }
        }
        let df = self.activation.df(self.input.clone());
        for i in 0..self.grad.len() {
            self.grad[i] *= df[i];
        }
        self.old_grads.push(self.grad.clone());

        (self.grad.clone(), self.weights.clone())
    }
    pub fn correct(&mut self, prev_outputs: Vec<Vec<f32>>, learning_rate: f32) {
        if prev_outputs.len() == 0 {
            self.clear();
            return;
        }

        for k in 0..self.old_grads.len() {
            for i in 0..self.weights.len() {
                for j in 0..self.weights[i].len() {
                    self.weights[i][j] = self.weights[i][j]
                        - learning_rate * self.old_grads[k][i] * prev_outputs[k][j]
                            / self.old_grads.len() as f32;
                }
            }
            for i in 0..self.basis.len() {
                self.basis[i] -= learning_rate * self.old_grads[k][i] / self.old_grads.len() as f32;
            }
        }

        self.clear();
    }
    pub fn clear(&mut self) {
        self.old_grads.clear();
        self.old_outputs.clear();
    }
}

#[derive(Debug)]
pub struct NeuroNetwork {
    pub layers: Vec<NeuroLayer>,
    batch_size: usize,
    epoch_amount: usize,
    error_function: ErrorFunction,
}

#[derive(Debug, serde::Deserialize, serde::Serialize)]
pub struct NeuroLayerJson {
    pub basis: Vec<f32>,
    pub weights: Vec<Vec<f32>>,
    pub activation: String,
}

#[derive(Debug, serde::Deserialize, serde::Serialize)]
pub struct NeuroNetworkJson {
    pub layers: Vec<NeuroLayerJson>,
    pub batch_size: usize,
    pub epoch_amount: usize,
    pub error_func: String,
}

pub fn neural_layer_to_json(layer: &NeuroLayer) -> NeuroLayerJson {
    NeuroLayerJson {
        basis: layer.basis.clone(),
        weights: layer.weights.clone(),
        activation: layer.activation.name.clone(),
    }
}

pub fn neural_to_json(net: &NeuroNetwork) -> NeuroNetworkJson {
    NeuroNetworkJson {
        layers: net.layers.iter().map(|x| neural_layer_to_json(x)).collect(),
        batch_size: net.batch_size,
        epoch_amount: net.epoch_amount,
        error_func: net.error_function.name.clone(),
    }
}

fn json_to_layer(j: NeuroLayerJson) -> NeuroLayer {
    let mut layer = NeuroLayer::new(
        j.weights.len(),
        j.weights[0].len(),
        ActivationFunc::from_str(&j.activation).unwrap(),
    );
    layer.weights = j.weights;
    layer.basis = j.basis;
    layer
}

fn json_to_network(j: NeuroNetworkJson) -> NeuroNetwork {
    let layers = j.layers.into_iter().map(|x| json_to_layer(x)).collect();
    NeuroNetwork {
        layers: layers,
        batch_size: j.batch_size,
        epoch_amount: j.epoch_amount,
        error_function: ErrorFunction::new(ErrorFunc::from_str(&j.error_func).unwrap()),
    }
}

impl NeuroNetwork {
    pub fn new(layers: Vec<usize>) -> NeuroNetwork {
        let mut neuro_layers: Vec<NeuroLayer> = Vec::with_capacity(layers.len());
        neuro_layers.push(NeuroLayer::new(layers[0], 0, ActivationFunc::Sigmoid));
        for i in 1..layers.len() {
            neuro_layers.push(NeuroLayer::new(
                layers[i],
                layers[i - 1],
                ActivationFunc::Sigmoid,
            ));
        }
        for k in 1..neuro_layers.len() {
            let fan = (neuro_layers[k].weights.len() as f32
                + neuro_layers[k - 1].weights.len() as f32)
                .sqrt();
            let glorot = 6.0f32.sqrt() / fan;
            let between = Uniform::from(-glorot..glorot);
            let mut rng = rand::thread_rng();
            for i in 0..neuro_layers[k].weights.len() {
                for j in 0..neuro_layers[k].weights[i].len() {
                    neuro_layers[k].weights[i][j] = between.sample(&mut rng);
                }
                // neuro_layers[k].basis[i] = between.sample(&mut rng);
            }
        }
        // for i in 0..neuro_layers[0].basis.len() {
        // neuro_layers[0].basis[i] = rand::random::<f32>();
        // }
        Self {
            layers: neuro_layers,
            batch_size: 1,
            epoch_amount: 100,
            error_function: ErrorFunction::new(ErrorFunc::Simple),
        }
    }
    pub fn with_activation(self, activation: ActivationFunc) -> Self {
        Self {
            layers: self
                .layers
                .into_iter()
                .map(|x| NeuroLayer {
                    activation: Activation::new(activation.clone()),
                    ..x
                })
                .collect(),
            ..self
        }
    }
    pub fn with_last_activation(self, activation: ActivationFunc) -> Self {
        let mut new_net = Self {
            layers: self.layers,
            ..self
        };
        let layers_amount = new_net.layers.len();
        new_net.layers[layers_amount - 1].activation = Activation::new(activation);
        new_net
    }
    pub fn with_error(self, error_func: ErrorFunc) -> Self {
        Self {
            error_function: ErrorFunction::new(error_func),
            ..self
        }
    }
    pub fn with_epoch(self, epoch_amount: usize) -> Self {
        Self {
            epoch_amount: epoch_amount,
            ..self
        }
    }
    pub fn with_batch_size(self, batch_size: usize) -> Self {
        Self {
            batch_size: batch_size,
            ..self
        }
    }
    fn forward(&mut self, data: Vec<f32>) {
        self.layers[0].raw_input = data.clone();
        self.layers[0].input = data.clone();
        self.layers[0].output = data.clone();
        self.layers[0].old_outputs.push(data.clone());
        let mut input = self.layers[0].output.clone();
        for i in 1..self.layers.len() {
            input = self.layers[i].forward(input);
        }
    }
    fn backward(&mut self, desired_output: Vec<f32>) -> f32 {
        let last_activation = &self.layers[self.layers.len() - 1].activation;
        let mut grad = self.layers[self.layers.len() - 1]
            .output
            .iter()
            .zip(desired_output.iter())
            .zip(last_activation.df(self.layers[self.layers.len() - 1].input.clone()))
            .map(|((x, y), z)| self.error_function.df(*x, *y) as f32 * z)
            .collect::<Vec<_>>();
        let cost = grad.iter().sum::<f32>();
        let mut weights = self.layers[self.layers.len() - 1].weights.clone();

        let layers_amount = self.layers.len();
        self.layers[layers_amount - 1].old_grads.push(grad.clone());
        self.layers[layers_amount - 1].grad = grad.clone();

        for i in (1..layers_amount - 1).rev() {
            (grad, weights) = self.layers[i].backward(grad, weights);
        }
        cost
    }
    fn train_step(&mut self, batch: &Batch, learning_rate: f32) -> f32 {
        let mut batch_cost = 0.0;
        batch.data.iter().for_each(|sample| {
            self.forward(sample.data.clone());
            batch_cost += self.backward(sample.solution.clone());
        });
        self.correct(learning_rate);
        batch_cost / batch.data.len() as f32
    }
    fn correct(&mut self, learning_rate: f32) {
        for i in 1..self.layers.len() {
            let old_outputs = self.layers[i - 1].old_outputs.clone();
            self.layers[i].correct(old_outputs, learning_rate);
        }
        self.layers[0].correct(vec![], learning_rate);
    }
    pub fn train(&mut self, data: Vec<Sample>, learning_rate: f32) {
        let mut batches: Vec<Batch> = data
            .chunks(self.batch_size)
            .map(|samples| Batch::new(samples.into()))
            .collect();
        for i in 0..self.epoch_amount {
            let mut epoch_cost = 0.0;
            for j in 0..batches.len() {
                epoch_cost += self.train_step(&batches[j], learning_rate);
            }
            if i % 100 == 0 {
                println!("Cost: {} ({})", epoch_cost, i);
            }
        }
    }
    fn clear_layers(&mut self) {
        for i in 0..self.layers.len() {
            self.layers[i].clear();
        }
    }
    pub fn solve(&mut self, data: Vec<f32>) -> Vec<f32> {
        self.forward(data);
        let output = self.layers.last().unwrap().output.clone();
        self.clear_layers();
        output
    }
}

#[derive(Debug, PartialEq)]
pub enum NeuroLayers {
    Zero,
    One,
    Two,
}

struct Batch {
    data: Vec<Sample>,
}
impl Batch {
    fn new(data: Vec<Sample>) -> Self {
        Self { data: data }
    }
}
