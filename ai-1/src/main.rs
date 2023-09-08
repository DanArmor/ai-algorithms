#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release

mod simulated_annealing;

use eframe::egui;
use eframe::egui::ColorImage;
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

static ASCII_UPPER: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";

pub struct CustomPlot {
    pub plot_id: String,
    pub data: Vec<[f64; 2]>,
    pub width: f32,
    pub height: f32,
    pub title: String,
}

impl CustomPlot {
    fn new(plot_id: impl Into<String>) -> Self {
        Self {
            plot_id: plot_id.into(),
            ..Default::default()
        }
    }
    // TODO: remove default and implement "with" methods
}

impl Default for CustomPlot {
    fn default() -> Self {
        Self {
            plot_id: "test_plot_id".into(),
            data: vec![[0.0, 0.0]],
            width: 200.0,
            height: 200.0,
            title: "Test_title".into()
        }
    }
}

impl egui::Widget for CustomPlot {
    fn ui(self, ui: &mut egui::Ui) -> egui::Response {
    ui.vertical(|ui| {
            ui.label(self.title);
            let my_plot = Plot::new(self.plot_id)
            .height(self.height)
            .width(self.width)
            .legend(Legend::default());

            // let's create a dummy line in the plot
            let graph: Vec<[f64; 2]> = vec![[0.0, 1.0], [2.0, 3.0], [3.0, 2.0]];
            my_plot.show(ui, |plot_ui| {
                plot_ui.line(Line::new(PlotPoints::from(graph)).name("curve"));
            })
        }).response
    }
}

#[derive(Default)]
struct MyApp {
    screenshot: Option<ColorImage>,
    max_temperature_str: String,
    min_temperature_str: String,
    temperature_alpha: String,
    queens_amount: String,
    steps_n: String,
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
                        frame.request_screenshot();
                    }
                });
                egui::Grid::new("grid_plots").show(ui, |ui| {
                    ui.add(CustomPlot::default());

                    // CustomPlot::default();

                    // ui.end_row();

                    // CustomPlot::default();

                    // CustomPlot::default();
                });
            });
        });
    }

    fn post_rendering(&mut self, _screen_size_px: [u32; 2], frame: &eframe::Frame) {
        // this is inspired by the Egui screenshot example
        if let Some(screenshot) = frame.screenshot() {
            self.screenshot = Some(screenshot);
        }
    }
}
