#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // Скрывает консоль на Windows

mod art1;

use eframe::egui;
use egui::plot::{Legend, Line, Plot, PlotPoints};
use rand::Rng;

fn main() -> Result<(), eframe::Error> {
    env_logger::init(); // Лог в stderr (`RUST_LOG=debug`).

    let options = eframe::NativeOptions {
        initial_window_size: Some(egui::vec2(350.0, 400.0)),
        ..Default::default()
    };
    eframe::run_native(
        "Simulated annealing",
        options,
        Box::new(|_cc| Box::<MyApp>::default()),
    )
}
struct MyApp {
    // Количество векторов-прототипов
    amount_clasters: String,
    // Параметр внимательности
    p: String,
    // Бета параметр
    b: String,
}

impl Default for MyApp {
    fn default() -> Self {
        Self {
            amount_clasters: "0".into(),
            b: "0".into(),
            p: "0".into(),
        }
    }
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Simulated annealing");
            ui.add_space(18.0);

            egui::ScrollArea::new([true, true])
                .auto_shrink([true, true])
                .show(ui, |ui| {
                    ui.horizontal(|ui|{
                        ui.vertical(|ui| {
                            ui.label("Количество векторов-прототипов");
                            ui.text_edit_singleline(&mut self.amount_clasters);
    
                            ui.label("Параметр внимательности");
                            ui.text_edit_singleline(&mut self.p);
    
                            ui.label("Бета параметр");
                            ui.text_edit_singleline(&mut self.b);
                        });
                    });
                });
        });
    }
}
