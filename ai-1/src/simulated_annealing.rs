pub trait State {
    fn setup(&mut self);
    fn energy(&self) -> f64;
    fn changed_state(&self) -> Self;
}

#[derive(Debug)]
pub struct SolutionStepInfo<T: State + Clone> {
    pub index: usize,
    pub temperature: f64,
    pub final_energy: f64,
    pub bad_decisions: i64,
    pub good_decisions: i64,
    pub final_state: T,
}

#[derive(Debug)]
pub struct SolutionInfo<T: State + Clone> {
    pub min_temperature: f64,
    pub max_temperature: f64,
    pub n_steps: i64,
    pub steps: Vec<SolutionStepInfo<T>>,
}

pub fn sim_ang<T: State + Clone>(
    init_state: T,
    min_temperature: f64,
    max_temperature: f64,
    dec_temp: impl Fn(f64) -> f64,
    n_steps: i64,
) -> (T, SolutionInfo<T>) {
    let mut state = init_state;
    state.setup();
    let mut temperature = max_temperature;
    let mut solution_info = SolutionInfo {
        min_temperature: min_temperature,
        max_temperature: max_temperature,
        n_steps: n_steps,
        steps: vec![],
    };
    let mut step_index = 0;
    while temperature > min_temperature {
        let mut bad_decisions = 0i64;
        let mut good_decisions = 0i64;

        for i in 0..n_steps {
            let new_state = state.changed_state();
            let delta_energy = new_state.energy() - state.energy();
            if delta_energy > 0.0 {
                let p = f64::exp(-delta_energy / temperature);
                let bound_p = rand::random::<f64>();
                if p > bound_p {
                    bad_decisions += 1;
                    state = new_state;
                }
            } else {
                good_decisions += 1;
                state = new_state;
            }
        }
        solution_info.steps.push(SolutionStepInfo {
            index: step_index,
            temperature: temperature,
            final_energy: state.energy(),
            bad_decisions: bad_decisions,
            good_decisions: good_decisions,
            final_state: state.clone(),
        });
        step_index += 1;
        temperature = dec_temp(temperature);
    }
    (state, solution_info)
}
