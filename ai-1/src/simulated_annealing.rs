pub trait State {
    fn setup(&mut self) -> Self;
    fn energy(&self) -> f64;
    fn changed_state(&self) -> Self;
}

pub fn sim_ang(
    init_state: impl State,
    min_temperature: f64,
    max_temperature: f64,
    dec_temp: fn(f64) -> f64,
    n_steps: i64,
) -> impl State {
    let mut state = init_state;
    state.setup();
    let mut temperature = max_temperature;
    while temperature > min_temperature {
        for _ in 0..n_steps {
            let new_state = state.changed_state();
            let delta_energy = new_state.energy() - state.energy();
            if delta_energy > 0.0 {
                let p = f64::exp(-delta_energy / temperature);
                let bound_p = rand::random::<f64>();
                if p > bound_p {
                    state = new_state;
                }
            } else {
                state = new_state;
            }
        }
        temperature = dec_temp(temperature);
    }
    state
}
