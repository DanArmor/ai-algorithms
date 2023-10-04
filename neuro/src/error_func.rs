use std::str::FromStr;

pub struct ErrorFunction {
    f: ErrorFn,
    df: ErrorFn,
    pub name: String,
}
impl ErrorFunction {
    pub fn new(error_func: ErrorFunc) -> Self {
        Self {
            f: error_func.f(),
            df: error_func.df(),
            name: error_func.get_name(),
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
        write!(f, "{}", self.name)
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum ErrorFunc {
    Simple,
}

impl FromStr for ErrorFunc {
    type Err = ();
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "simple" => Ok(Self::Simple),
            _ => Err(()),
        }
    }
}

impl ErrorFunc {
    fn get_name(&self) -> String {
        match self {
            Self::Simple => "simple".into(),
        }
    }
    fn f(&self) -> ErrorFn {
        match self {
            Self::Simple => simple_err_func,
        }
    }
    fn df(&self) -> ErrorFn {
        match self {
            Self::Simple => simple_err_func_df,
        }
    }
}

fn simple_err_func(x: f32, y: f32) -> f32 {
    (x - y).powi(2) / 2.0
}

fn simple_err_func_df(x: f32, y: f32) -> f32 {
    x - y
}

pub type ErrorFn = fn(f32, f32) -> f32;
