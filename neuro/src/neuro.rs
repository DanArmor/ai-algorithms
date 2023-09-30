use nalgebra::*;

#[derive(Debug, PartialEq)]
pub enum Solution {
    Car,
    Heli,
    Sheep,
    Airplane,
}
struct Sample {
    data: DVector<f32>,
    solution: Solution,
}

struct NeuroLayer<F: Fn(f32) -> f32> {
    raw_input: Vec<f32>,
    input: Vec<f32>,
    output: Vec<f32>,
    basis: Vec<f32>,
    weights: Vec<Vec<f32>>,
    activation: Activation<F>,
    grad: Vec<f32>,
    old_grads: Vec<Vec<f32>>,
}

impl<F: Fn(f32) -> f32> NeuroLayer<F> {
    fn forward(&mut self, input: Vec<f32>) -> Vec<f32> {
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
        for i in 0..self.output.len() {
            self.output[i] = self.activation.f(self.input[i]);
        }
        self.output.clone()
    }
    fn backward(&mut self, grad: Vec<f32>, weights: Vec<Vec<f32>>) -> (Vec<f32>, Vec<Vec<f32>>) {
        self.grad.fill(0.0);
        for i in 0..weights[0].len() {
            for j in 0..weights.len() {
                self.grad[i] += grad[j] * weights[j][i];
            }
            for j in 0..weights.len() {
                self.grad[i] *= self.activation.df(self.input[i]);
            }
        }
        self.old_grads.push(self.grad.clone());

        (self.grad.clone(), self.weights.clone())
    }
    
}

struct NeuroNetwork<F: Fn(f32) -> f32, EF: Fn(f32, f32) -> f32> {
    layers: NeuroLayer<F>,
    batch_size: usize,
    epoch_amount: usize,
    activation: Activation<F>,
    error_function: ErrorFunction<EF>,
}

struct ErrorFunction<F: Fn(f32, f32) -> f32> {
    f: F,
    df: F,
}
impl<F: Fn(f32, f32) -> f32> ErrorFunction<F> {
    pub fn new(f: F, df: F) -> Self {
        Self { f: f, df: df }
    }
    pub fn f(&self, x: f32, y: f32) -> f32 {
        self.f(x, y)
    }
    pub fn df(&self, x: f32, y: f32) -> f32 {
        self.df(x, y)
    }
}

struct Activation<F: Fn(f32) -> f32> {
    f: F,
    df: F,
}
impl<F: Fn(f32) -> f32> Activation<F> {
    pub fn new(f: F, df: F) -> Self {
        Self { f: f, df: df }
    }
    pub fn f(&self, x: f32) -> f32 {
        self.f(x)
    }
    pub fn df(&self, x: f32) -> f32 {
        self.df(x)
    }
    pub fn v_f(&self, v_x: &DVector<f32>) -> DVector<f32> {
        DVector::from_vec(v_x.iter().map(|x| self.f(*x)).collect())
    }
    pub fn v_df(&self, v_x: &DVector<f32>) -> DVector<f32> {
        DVector::from_vec(v_x.iter().map(|x| self.df(*x)).collect())
    }
}

impl<F: Fn(f32) -> f32, EF: Fn(f32, f32) -> f32> NeuroNetwork<F, EF> {
    fn forward(&mut self, data: &DVector<f32>) {
        for i in 0..data.len() {
            self.inputs[0][i] = data[i];
            self.layers[0][i] = data[i];
        }
        for i in 1..self.layers.len() {
            self.inputs[i] = &self.weights[i - 1] * &self.layers[i - 1] + &self.basis[i - 1];
            self.layers[i] = self.activation.v_f(&self.inputs[i]);
        }
    }
    fn backward(&mut self, desired_output: DVector<f32>, learning_rate: f32) {
        let errors = self
            .layers
            .last()
            .unwrap()
            .iter()
            .zip(desired_output.iter())
            .zip(self.inputs.last().unwrap().iter())
            .map(|((x, y), z)| self.error_function.df(*x, *y) as f32 * self.activation.df(*z))
            .collect::<Vec<_>>();
        self.errors[self.errors.len() - 1] = DVector::from_vec(errors);
        for i in (0..self.weights.len()).rev() {}
    }
}

#[derive(Debug, PartialEq)]
pub enum NeuroLayers {
    Zero,
    One,
    Two,
}

#[derive(Debug, PartialEq)]
pub enum ActivationFunc {
    Sig,
    Gip,
    Arctn,
}

#[derive(Debug, PartialEq)]
pub enum LearningMode {
    OneByOne,
    Packet,
}

struct Batch<'a> {
    data: Vec<&'a Sample>,
}
impl<'a> Batch<'a> {
    fn new(data: Vec<&'a Sample>) -> Self {
        Self { data: data }
    }
}

fn train_step<F: Fn(f32) -> f32, EF: Fn(f32, f32) -> f32>(
    network: &mut NeuroNetwork<F, EF>,
    batch: &Batch,
) {
    batch.data.iter().for_each(|sample| {
        network.forward(&sample.data);
    })
}

fn train<F: Fn(f32) -> f32, EF: Fn(f32, f32) -> f32>(
    data: &Vec<Sample>,
    amount_epoch: usize,
    mut network: NeuroNetwork<F, EF>,
) -> NeuroNetwork<F, EF> {
    let mut batches: Vec<Batch> = vec![];
    for i in (0..data.len()).step_by(network.batch_size) {
        let mut batch: Vec<&Sample> = vec![];
        for j in 0..network.batch_size {
            batch.push(&data[i + j]);
        }
        batches.push(Batch::new(batch));
    }
    for _ in 0..amount_epoch {
        for j in 0..batches.len() {
            train_step(&mut network, &batches[j]);
        }
    }
    network
}

fn create_network<F: Fn(f32) -> f32, EF: Fn(f32, f32) -> f32>(
    data: &Vec<Sample>,
    epoch_amount: usize,
    batch_size: usize,
    activation: Activation<F>,
    error_func: ErrorFunction<EF>,
) -> NeuroNetwork<F, EF> {
    let network = NeuroNetwork::<F, EF> {
        layers: vec![],
        weights: vec![],
        basis: vec![],
        inputs: vec![],
        errors: vec![],
        batch_size: batch_size,
        epoch_amount: epoch_amount,
        activation: activation,
        error_function: error_func,
    };
    for i in 0..epoch_amount {}
    network
}
