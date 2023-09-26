// Состояние решения
pub trait State {
    // Первоначальное решение
    fn setup(&mut self);
    // Расчет энергии
    fn energy(&self) -> f64;
    // Шаг изменения решения
    fn changed_state(&self) -> Self;
}

// Информация об итерации решения
#[derive(Debug, Clone)]
pub struct SolutionStepInfo<T: State + Clone> {
    // Номер итерации
    pub index: usize,
    // Температура на итерации
    pub temperature: f64,
    // Финальная энергия
    pub final_energy: f64,
    // Кол-во плохих решений
    pub bad_decisions: i64,
    // Кол-во хороших решений
    pub good_decisions: i64,
    // Конечное состояние в итерации
    pub final_state: T,
}

// Информация о решении
#[derive(Debug, Clone)]
pub struct SolutionInfo<T: State + Clone> {
    // Мин температура
    pub min_temperature: f64,
    // Макс температура
    pub max_temperature: f64,
    // Кол-во шагов без изменения температуры
    pub n_steps: i64,
    // Данные о шагах
    pub steps: Vec<SolutionStepInfo<T>>,
}

// Имитация отжига
pub fn sim_ang<T: State + Clone>(
    init_state: T,
    min_temperature: f64,
    max_temperature: f64,
    dec_temp: impl Fn(f64) -> f64,
    n_steps: i64,
) -> (T, SolutionInfo<T>) {
    // Настроим первоначальное состояние решения
    let mut state = init_state;
    state.setup();

    // Параметры решения
    let mut temperature = max_temperature;
    let mut solution_info = SolutionInfo {
        min_temperature: min_temperature,
        max_temperature: max_temperature,
        n_steps: n_steps,
        steps: vec![],
    };
    // Номер итерации
    let mut step_index = 0;

    while temperature > min_temperature {
        // Подсчитываем количество плохих и хороших решений
        let mut bad_decisions = 0i64;
        let mut good_decisions = 0i64;

        // n шагов без изменения температуры
        for _ in 0..n_steps {
            // Меняем решение
            let new_state = state.changed_state();
            // Рассчитываем разницу
            let delta_energy = new_state.energy() - state.energy();
            // Новое решение хуже старого
            if delta_energy > 0.0 {
                // Оценим вероятность допуска
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
        // Сохраним данные об итерации
        solution_info.steps.push(SolutionStepInfo {
            index: step_index,
            temperature: temperature,
            final_energy: state.energy(),
            bad_decisions: bad_decisions,
            good_decisions: good_decisions,
            final_state: state.clone(),
        });
        step_index += 1;
        // Понизим температуру
        temperature = dec_temp(temperature);
    }
    (state, solution_info)
}
