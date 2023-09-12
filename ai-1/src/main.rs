#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release

mod simulated_annealing;

use eframe::egui;
use egui::plot::{Legend, Line, Plot, PlotPoints};
use rand::Rng;

fn main() -> Result<(), eframe::Error> {
    env_logger::init(); // Log to stderr (if you run with `RUST_LOG=debug`).

    let options = eframe::NativeOptions {
        initial_window_size: Some(egui::vec2(350.0, 400.0)),
        ..Default::default()
    };
    eframe::run_native(
        "My egui App with a plot",
        options,
        Box::new(|_cc| Box::<MyApp>::default()),
    )
}

#[derive(Debug, Clone)]
pub struct QueenState {
    positions: Vec<usize>,
    n: usize,
}

impl QueenState {
    fn new(n: usize) -> Self {
        Self {
            positions: vec![],
            n: n,
        }
    }
}

impl simulated_annealing::State for QueenState {
    fn changed_state(&self) -> Self {
        let mut new_positions = self.positions.clone();
        let first_index = rand::thread_rng().gen_range(0..new_positions.len());
        let second_index = rand::thread_rng().gen_range(0..new_positions.len());
        new_positions.swap(first_index, second_index);
        Self {
            positions: new_positions,
            n: self.n,
        }
    }
    fn energy(&self) -> f64 {
        let mut energy = 0.0f64;
        for i in 0..self.positions.len() {
            for j in i + 1..self.positions.len() {
                if i.abs_diff(j) == self.positions[i].abs_diff(self.positions[j]) {
                    energy += 1.0f64;
                }
            }
        }
        energy
    }
    fn setup(&mut self) {
        self.positions = (0..self.n).collect();
    }
}

pub struct CustomPlot {
    pub plot_id: String,
    pub lines: Vec<CustomLine>,
    pub width: f32,
    pub height: f32,
    pub title: String,
}

pub struct CustomLine {
    pub data: Vec<[f64; 2]>,
    pub name: String,
}

impl CustomLine {
    fn new(data: Vec<[f64; 2]>, name: impl Into<String>) -> Self {
        Self {
            data: data,
            name: name.into(),
        }
    }
}

impl CustomPlot {
    fn new(plot_id: impl Into<String>, width: f32, height: f32, title: impl Into<String>) -> Self {
        Self {
            plot_id: plot_id.into(),
            width: width,
            height: height,
            title: title.into(),
            lines: Default::default(),
        }
    }
    fn add_line(&mut self, data: Vec<[f64; 2]>, name: impl Into<String>) {
        self.lines.push(CustomLine::new(data, name));
    }
    fn clear_lines(&mut self) {
        self.lines.clear();
    }
}

impl egui::Widget for &mut CustomPlot {
    fn ui(self, ui: &mut egui::Ui) -> egui::Response {
        ui.vertical(|ui| {
            ui.label(self.title.clone());
            let my_plot = Plot::new(self.plot_id.clone())
                .clamp_grid(true)
                .auto_bounds_x()
                .auto_bounds_y()
                .legend(Legend::default());

            my_plot.show(ui, |plot_ui| {
                for line in &self.lines {
                    plot_ui.line(
                        Line::new(PlotPoints::from(line.data.clone())).name(line.name.clone()),
                    );
                }
            })
        })
        .response
    }
}

struct MyApp {
    max_temperature_str: String,
    min_temperature_str: String,
    temperature_alpha: String,
    queens_amount: String,
    steps_n: String,
    plot: CustomPlot,
    chess_white: egui_extras::RetainedImage,
    chess_black: egui_extras::RetainedImage,
    chess_queen_white: egui_extras::RetainedImage,
    chess_queen_black: egui_extras::RetainedImage,
    state: QueenState,
    solution: simulated_annealing::SolutionInfo<QueenState>,
    promise:
        Option<poll_promise::Promise<(QueenState, simulated_annealing::SolutionInfo<QueenState>)>>,
}

impl MyApp {
    fn show_chess_board(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        egui::ScrollArea::new([true, true])
            .min_scrolled_height(400.0)
            .auto_shrink([true, true])
            .show(ui, |ui| {
                egui::Grid::new("grid_chess")
                    .min_row_height(32.0)
                    .min_col_width(32.0)
                    .show(ui, |ui| {
                        for i in 0..self.state.n {
                            for j in 0..self.state.n {
                                if i % 2 == j % 2 {
                                    if self.state.positions[i] == j {
                                        ui.add(egui::Image::new(
                                            self.chess_queen_white.texture_id(ctx),
                                            self.chess_queen_white.size_vec2(),
                                        ));
                                    } else {
                                        ui.add(egui::Image::new(
                                            self.chess_white.texture_id(ctx),
                                            self.chess_white.size_vec2(),
                                        ));
                                    }
                                } else {
                                    if self.state.positions[i] == j {
                                        ui.add(egui::Image::new(
                                            self.chess_queen_black.texture_id(ctx),
                                            self.chess_queen_black.size_vec2(),
                                        ));
                                    } else {
                                        ui.add(egui::Image::new(
                                            self.chess_black.texture_id(ctx),
                                            self.chess_black.size_vec2(),
                                        ));
                                    }
                                }
                            }
                            ui.end_row();
                        }
                    });
            });
    }
    fn add_lines(&mut self) {
        self.plot.add_line(
            self.solution
                .steps
                .iter()
                .map(|x| [x.index as f64, x.temperature])
                .collect(),
            "Температура",
        );
        self.plot.add_line(
            self.solution
                .steps
                .iter()
                .map(|x| [x.index as f64, x.bad_decisions as f64])
                .collect(),
            "Количество принятых плохих решений",
        );
        self.plot.add_line(
            self.solution
                .steps
                .iter()
                .map(|x| [x.index as f64, x.final_energy])
                .collect(),
            "Энергия лучшего решения",
        );
    }
}

impl Default for MyApp {
    fn default() -> Self {
        Self {
            min_temperature_str: "0.1".into(),
            max_temperature_str: "25".into(),
            temperature_alpha: "0.98".into(),
            queens_amount: "5".into(),
            steps_n: "10".into(),
            plot: CustomPlot::new("plot_1", 800.0, 400.0, "Изменения параметров"),
            chess_white: egui_extras::RetainedImage::from_image_bytes(
                "chess_white.png",
                include_bytes!("chess_white.png"),
            )
            .unwrap(),
            chess_black: egui_extras::RetainedImage::from_image_bytes(
                "chess_black.png",
                include_bytes!("chess_black.png"),
            )
            .unwrap(),
            chess_queen_white: egui_extras::RetainedImage::from_image_bytes(
                "chess_queen_white.png",
                include_bytes!("chess_queen_white.png"),
            )
            .unwrap(),
            chess_queen_black: egui_extras::RetainedImage::from_image_bytes(
                "chess_queen_black.png",
                include_bytes!("chess_queen_black.png"),
            )
            .unwrap(),
            state: QueenState::new(0),
            solution: simulated_annealing::SolutionInfo {
                min_temperature: 0.0,
                max_temperature: 0.0,
                n_steps: 0,
                steps: vec![],
            },
            promise: Option::None,
        }
    }
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            let border_y = 18.0;

            ui.heading("Simulated annealing");
            // add some whitespace in y direction
            ui.add_space(border_y);
            egui::ScrollArea::new([true, true])
                .auto_shrink([true, true])
                .show(ui, |ui| {
                    ui.vertical(|ui| {
                        ui.with_layout(egui::Layout::left_to_right(egui::Align::TOP), |ui| {
                            ui.vertical(|ui| {
                                ui.label("Минимальная температура");
                                ui.text_edit_singleline(&mut self.min_temperature_str);

                                ui.label("Максимальная температура");
                                ui.text_edit_singleline(&mut self.max_temperature_str);

                                ui.label("Коэффициент температуры");
                                ui.text_edit_singleline(&mut self.temperature_alpha);

                                ui.label("Количество ферзей");
                                ui.text_edit_singleline(&mut self.queens_amount);

                                ui.label("Количество шагов при постоянном значении температуры");
                                ui.text_edit_singleline(&mut self.steps_n);

                                if ui.button("Посчитать").clicked() {
                                    if self.promise.is_none() {
                                        self.plot.clear_lines();
                                        // Calculate parameters
                                        let min_temperature_str =
                                            self.min_temperature_str.parse().unwrap();
                                        let max_temperature_str =
                                            self.max_temperature_str.parse().unwrap();
                                        let queens_amount = self.queens_amount.parse().unwrap();
                                        let temperature_alpha =
                                            self.temperature_alpha.parse::<f64>().unwrap();

                                        self.promise = Some(poll_promise::Promise::<(
                                            QueenState,
                                            simulated_annealing::SolutionInfo<QueenState>,
                                        )>::spawn_thread(
                                            "Simulated annealing calculation",
                                            move || {
                                                simulated_annealing::sim_ang(
                                                    QueenState::new(queens_amount),
                                                    min_temperature_str,
                                                    max_temperature_str,
                                                    |x| x * temperature_alpha,
                                                    queens_amount as i64,
                                                )
                                            },
                                        ));
                                    }
                                }
                            });

                            ui.add(&mut self.plot);
                        });
                        ui.label("Шахматная доска");
                        match &self.promise {
                            Some(p) => {
                                if let Some(value) = p.ready() {
                                    self.state = value.0.clone();
                                    self.solution = value.1.clone();
                                    self.add_lines();
                                    self.promise = Option::None;
                                } else {
                                    ui.spinner();
                                }
                            }
                            None => {
                                self.show_chess_board(ui, ctx);
                            }
                        };
                    });
                });
        });
    }
}
