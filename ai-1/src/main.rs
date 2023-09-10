#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release

mod simulated_annealing;

use eframe::egui;
use egui::plot::{Legend, Line, Plot, PlotPoints};

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
    plots: Vec<CustomPlot>,
    chess_white: egui_extras::RetainedImage,
    chess_black: egui_extras::RetainedImage,
    chess_queen_white: egui_extras::RetainedImage,
    chess_queen_black: egui_extras::RetainedImage,
}

impl Default for MyApp {
    fn default() -> Self {
        Self {
            max_temperature_str: "0".into(),
            min_temperature_str: "0".into(),
            temperature_alpha: "0".into(),
            queens_amount: "0".into(),
            steps_n: "0".into(),
            plots: vec![
                CustomPlot::new("plot_1", 400.0, 400.0, "Title_1"),
                CustomPlot::new("plot_2", 400.0, 400.0, "Title_2"),
                CustomPlot::new("plot_3", 400.0, 400.0, "Title_3"),
                CustomPlot::new("plot_4", 400.0, 400.0, "Title_4"),
            ],
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
        }
    }
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            // these are just some dummy variables for the example,
            // such that the plot is not at position (0,0)
            let height = 400.0;
            let border_x = 11.0;
            let border_y = 18.0;
            let width = 600.0;

            ui.heading("Simulated annealing");
            // add some whitespace in y direction
            ui.add_space(border_y);
            ui.vertical(|ui| {
                ui.with_layout(egui::Layout::left_to_right(egui::Align::TOP), |ui| {
                    ui.vertical(|ui| {
                        ui.label("Минимальная температура");
                        ui.text_edit_singleline(&mut self.max_temperature_str);

                        ui.label("Максимальная температура");
                        ui.text_edit_singleline(&mut self.min_temperature_str);

                        ui.label("Коэффициент температуры");
                        ui.text_edit_singleline(&mut self.temperature_alpha);

                        ui.label("Количество ферзей");
                        ui.text_edit_singleline(&mut self.queens_amount);

                        ui.label("Количество шагов при постоянном значении температуры");
                        ui.text_edit_singleline(&mut self.steps_n);

                        if ui.button("Посчитать").clicked() {
                            self.plots[0].add_line(vec![[0.0, 5.0], [7.0, 20.0]], "name1")
                            // simulated_annealing::sim_ang(init_state, min_temperature, max_temperature, dec_temp, n_steps)
                        }
                    });
                    egui::Grid::new("grid_plots")
                        .num_columns(2)
                        .min_row_height(200.0)
                        .max_col_width(400.0)
                        .show(ui, |ui| {
                            ui.add(&mut self.plots[0]);
                            ui.add(&mut self.plots[1]);
                            ui.end_row();
                            ui.add(&mut self.plots[2]);
                            ui.add(&mut self.plots[3]);
                        });
                });

                ui.label("Шахматная доска");
                egui::Grid::new("grid_chess")
                    .num_columns(15)
                    .show(ui, |ui| {
                        ui.add(egui::Image::new(
                            self.chess_white.texture_id(ctx),
                            self.chess_white.size_vec2(),
                        ));
                    });
            });
        });
    }
}
