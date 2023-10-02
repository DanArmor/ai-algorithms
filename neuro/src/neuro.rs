#[derive(Debug, PartialEq)]
pub enum Solution {
    Car,
    Heli,
    Sheep,
    Airplane,
}
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
}

impl NeuroLayer {
    pub fn new(neurons_amount: usize, back_links_amount: usize) -> Self {
        Self {
            raw_input: vec![0.0; neurons_amount],
            input: vec![0.0; neurons_amount],
            output: vec![0.0; neurons_amount],
            basis: vec![1.0; neurons_amount],
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
        }
    }
    pub fn forward(&mut self, input: Vec<f32>, activation: &Activation) -> Vec<f32> {
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
            self.output[i] = activation.f(self.input[i]);
        }
        self.old_outputs.push(self.output.clone());
        self.output.clone()
    }
    pub fn backward(
        &mut self,
        grad: Vec<f32>,
        weights: Vec<Vec<f32>>,
        activation: &Activation,
    ) -> (Vec<f32>, Vec<Vec<f32>>) {
        self.grad.fill(0.0);
        for i in 0..weights[0].len() {
            for j in 0..weights.len() {
                self.grad[i] += grad[j] * weights[j][i];
            }
            self.grad[i] *= activation.df(self.input[i]);
        }
        self.old_grads.push(self.grad.clone());

        (self.grad.clone(), self.weights.clone())
    }
    pub fn correct(&mut self, prev_outputs: Vec<Vec<f32>>, learning_rate: f32) {
        if prev_outputs.len() == 0 {
            self.clear();
            return;
        }
        let mut grad = self.old_grads[0].clone();
        for k in 1..self.old_grads.len() {
            for i in 0..grad.len() {
                grad[i] += self.old_grads[k][i] * prev_outputs[k][i];
            }
        }
        for i in 0..grad.len() {
            grad[i] = grad[i] / self.old_grads.len() as f32;
        }
        for i in 0..self.weights.len() {
            for j in 0..self.weights[i].len() {
                self.weights[i][j] = self.weights[i][j] - learning_rate * grad[i];
            }
        }
        for i in 0..self.basis.len() {
            self.basis[i] -= learning_rate * grad[i];
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
    activation: Activation,
    error_function: ErrorFunction,
}

struct ErrorFunction {
    f: Box<dyn Fn(f32, f32) -> f32>,
    df: Box<dyn Fn(f32, f32) -> f32>,
}
impl ErrorFunction {
    pub fn new(
        f: impl Fn(f32, f32) -> f32 + 'static,
        df: impl Fn(f32, f32) -> f32 + 'static,
    ) -> Self {
        Self {
            f: Box::new(f),
            df: Box::new(df),
        }
    }
    pub fn f(&self, x: f32, y: f32) -> f32 {
        (self.f)(x, y)
    }
    pub fn df(&self, x: f32, y: f32) -> f32 {
        (self.df)(x, y)
    }
}

impl std::fmt::Debug for ErrorFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Error func")
    }
}

pub struct Activation {
    f: Box<dyn Fn(f32) -> f32>,
    df: Box<dyn Fn(f32) -> f32>,
}
impl Activation {
    pub fn new(f: impl Fn(f32) -> f32 + 'static, df: impl Fn(f32) -> f32 + 'static) -> Self {
        Self {
            f: Box::new(f),
            df: Box::new(df),
        }
    }
    pub fn f(&self, x: f32) -> f32 {
        (self.f)(x)
    }
    pub fn df(&self, x: f32) -> f32 {
        (self.df)(x)
    }
}

impl std::fmt::Debug for Activation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Activation func")
    }
}

fn sigmoid(v: f32) -> f32 {
    1.0 / (1.0 + f32::exp(-v))
}

fn sigmoid_df(v: f32) -> f32 {
    let x = sigmoid(v);
    x * (1.0 - x)
}

fn simple_err_func(x: f32, y: f32) -> f32 {
    (x - y).powi(2) / 2.0
}

fn simple_err_func_df(x: f32, y: f32) -> f32 {
    x - y
}

impl NeuroNetwork {
    pub fn new(layers: Vec<usize>) -> NeuroNetwork {
        let mut neuro_layers: Vec<NeuroLayer> = Vec::with_capacity(layers.len());
        let activation = Activation::new(sigmoid, sigmoid_df);
        neuro_layers.push(NeuroLayer::new(layers[0], 0));
        for i in 1..layers.len() {
            neuro_layers.push(NeuroLayer::new(layers[i], layers[i - 1]));
        }
        Self {
            layers: neuro_layers,
            batch_size: 1,
            epoch_amount: 100,
            activation: Activation::new(sigmoid, sigmoid_df),
            error_function: ErrorFunction::new(simple_err_func, simple_err_func_df),
        }
    }
    pub fn with_activation(
        self,
        f: impl Fn(f32) -> f32 + 'static,
        df: impl Fn(f32) -> f32 + 'static,
    ) -> Self {
        Self {
            activation: Activation::new(f, df),
            ..self
        }
    }
    pub fn with_error(
        self,
        f: impl Fn(f32, f32) -> f32 + 'static,
        df: impl Fn(f32, f32) -> f32 + 'static,
    ) -> Self {
        Self {
            error_function: ErrorFunction::new(f, df),
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
            input = self.layers[i].forward(input, &self.activation);
        }
    }
    fn backward(&mut self, desired_output: Vec<f32>) {
        let mut grad = self.layers[self.layers.len() - 1]
            .output
            .iter()
            .zip(desired_output.iter())
            .zip(self.layers[self.layers.len() - 1].input.iter())
            .map(|((x, y), z)| self.error_function.df(*x, *y) as f32 * self.activation.df(*z))
            .collect::<Vec<_>>();
        let mut weights = self.layers[self.layers.len() - 1].weights.clone();

        let layers_amount = self.layers.len();
        self.layers[layers_amount - 1].old_grads.push(grad.clone());
        self.layers[layers_amount - 1].grad = grad.clone();

        for i in (1..layers_amount - 1).rev() {
            (grad, weights) = self.layers[i].backward(grad, weights, &self.activation);
        }
    }
    fn train_step(&mut self, batch: &Batch, learning_rate: f32) {
        batch.data.iter().for_each(|sample| {
            self.forward(sample.data.clone());
            self.backward(sample.solution.clone());
        });
        self.correct(learning_rate);
    }
    fn correct(&mut self, learning_rate: f32) {
        for i in 1..self.layers.len() {
            let old_outputs = self.layers[i - 1].old_outputs.clone();
            self.layers[i].correct(old_outputs, learning_rate);
        }
        self.layers[0].correct(vec![], learning_rate);
    }
    pub fn train(&mut self, data: Vec<Sample>, learning_rate: f32) {
        let mut batches: Vec<Batch> = vec![];
        for i in (0..data.len()).step_by(self.batch_size) {
            let mut batch: Vec<&Sample> = vec![];
            for j in 0..self.batch_size {
                batch.push(&data[i + j]);
            }
            batches.push(Batch::new(batch));
        }
        for _ in 0..self.epoch_amount {
            for j in 0..batches.len() {
                println!("D: {:#?}\n", self);
                self.train_step(&batches[j], learning_rate);
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
