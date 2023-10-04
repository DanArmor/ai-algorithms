use std::str::FromStr;

#[derive(Debug, PartialEq, Clone)]
pub enum ActivationFunc {
    Sigmoid,
    Tanh,
    Softmax,
    Relu,
}

impl FromStr for ActivationFunc {
    type Err = ();
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "sigmoid" => Ok(Self::Sigmoid),
            "tanh" => Ok(Self::Tanh),
            "softmax" => Ok(Self::Softmax),
            "relu" => Ok(Self::Relu),
            _ => Err(()),
        }
    }
}

impl ActivationFunc {
    fn get_name(&self) -> String {
        match self {
            Self::Sigmoid => "sigmoid".into(),
            Self::Relu => "relu".into(),
            Self::Softmax => "softmax".into(),
            Self::Tanh => "tanh".into(),
        }
    }
    fn f(&self) -> ActivationFn {
        match self {
            Self::Sigmoid => sigmoid,
            Self::Relu => relu,
            Self::Softmax => softmax,
            Self::Tanh => tanh,
        }
    }
    fn df(&self) -> ActivationFn {
        match self {
            Self::Sigmoid => sigmoid_df,
            Self::Relu => relu_df,
            Self::Softmax => softmax_df,
            Self::Tanh => tanh_df,
        }
    }
}

pub type ActivationFn = fn(Vec<f32>) -> Vec<f32>;

pub struct Activation {
    f: ActivationFn,
    df: ActivationFn,
    pub name: String,
}
impl Activation {
    pub fn new(activation: ActivationFunc) -> Self {
        Self {
            f: activation.f(),
            df: activation.df(),
            name: activation.get_name(),
        }
    }
    pub fn f(&self, v: Vec<f32>) -> Vec<f32> {
        (self.f)(v)
    }
    pub fn df(&self, v: Vec<f32>) -> Vec<f32> {
        (self.df)(v)
    }
}

impl std::fmt::Debug for Activation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name)
    }
}

pub fn sigmoid(v: Vec<f32>) -> Vec<f32> {
    v.into_iter().map(|x| 1.0 / (1.0 + f32::exp(-x))).collect()
}

pub fn sigmoid_df(v: Vec<f32>) -> Vec<f32> {
    let t = sigmoid(v);
    t.into_iter().map(|x| x * (1.0 - x)).collect()
}

pub fn tanh(v: Vec<f32>) -> Vec<f32> {
    v.into_iter().map(|x| x.tanh()).collect()
}

pub fn tanh_df(v: Vec<f32>) -> Vec<f32> {
    v.into_iter().map(|x| 1.0 - x.tanh().powi(2)).collect()
}

pub fn relu(v: Vec<f32>) -> Vec<f32> {
    v.into_iter()
        .map(|x| if x > 0.0 { x } else { 0.0 })
        .collect()
}
pub fn relu_df(v: Vec<f32>) -> Vec<f32> {
    v.into_iter()
        .map(|x| if x > 0.0 { 1.0 } else { 0.0 })
        .collect()
}

pub fn softmax(mut v: Vec<f32>) -> Vec<f32> {
    let max = v.iter().max_by(|x, y| x.total_cmp(y)).unwrap().clone();
    v = v.into_iter().map(|x| x - max).collect();
    let sum = v.iter().map(|x| x.exp()).sum::<f32>();
    if sum == 0.0 {
        return vec![0.0; v.len()];
    }
    v.into_iter().map(|x| x.exp() / sum).collect()
}

pub fn softmax_df(v: Vec<f32>) -> Vec<f32> {
    let t = softmax(v);
    t.into_iter().map(|x| x * (1.0 - x)).collect()
}
