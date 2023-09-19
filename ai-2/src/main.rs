#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // Скрывает консоль на Windows

mod art1;

use art1::Claster;
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
    // Кластеры
    clasters: Option<Vec<Claster>>,
    // Данные
    data: Vec<bit_vec::BitVec>,
    // Colors
    colors: Vec<egui::Color32>,
    // Color map
    colors_map: Option<std::collections::HashMap<String, egui::Color32>>,
}

impl Default for MyApp {
    fn default() -> Self {
        let mut data: Vec<bit_vec::BitVec> = vec![];
        for i in 0..10 {
            data.push(bit_vec::BitVec::from_bytes(&[rand::random::<u8>()]))
        }
        Self {
            amount_clasters: "5".into(),
            b: "1.0".into(),
            p: "0.1".into(),
            clasters: None,
            data: data,
            colors: vec![
                egui::Color32::from_rgb(128, 0, 0),
                egui::Color32::from_rgb(255, 99, 71),
                egui::Color32::from_rgb(240, 128, 128),
                egui::Color32::from_rgb(255, 165, 0),
                egui::Color32::from_rgb(240, 230, 140),
                egui::Color32::from_rgb(154, 205, 50),
                egui::Color32::from_rgb(124, 252, 0),
                egui::Color32::from_rgb(0, 250, 154),
                egui::Color32::from_rgb(32, 178, 170),
                egui::Color32::from_rgb(70, 130, 180),
                egui::Color32::from_rgb(100, 149, 237),
                egui::Color32::from_rgb(138, 43, 226),
                egui::Color32::from_rgb(123, 104, 238),
                egui::Color32::from_rgb(221, 160, 221),
                egui::Color32::from_rgb(255, 248, 220),
            ],
            colors_map: None,
        }
    }
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("ART1");
            ui.add_space(18.0);

            egui::ScrollArea::new([true, true])
                .auto_shrink([true, true])
                .show(ui, |ui| {
                    ui.with_layout(egui::Layout::left_to_right(egui::Align::TOP), |ui| {
                        ui.vertical(|ui| {
                            ui.label("Количество векторов-прототипов");
                            ui.text_edit_singleline(&mut self.amount_clasters);

                            ui.label("Параметр внимательности");
                            ui.text_edit_singleline(&mut self.p);

                            ui.label("Бета параметр");
                            ui.text_edit_singleline(&mut self.b);

                            if ui.button("Посчитать").clicked() {
                                let p = match self.p.parse() {
                                    Ok(v) => v,
                                    Err(_) => {
                                        self.p = "0.1".into();
                                        0f64
                                    }
                                };
                                let b = match self.b.parse() {
                                    Ok(v) => v,
                                    Err(_) => {
                                        self.b = "1.0".into();
                                        0f64
                                    }
                                };
                                let amount_clasters = match self.amount_clasters.parse() {
                                    Ok(v) => v,
                                    Err(_) => {
                                        self.amount_clasters = "5".into();
                                        0usize
                                    }
                                };
                                self.clasters = Some(art1::art1(&self.data, &amount_clasters, &p, &b));
                                if self.clasters.as_ref().unwrap().len() > self.colors.len() {
                                    self.colors_map = None;
                                } else {
                                    self.colors_map = Some(std::collections::HashMap::default());
                                    for i in 0..self.clasters.as_ref().unwrap().len() {
                                        self.colors_map.as_mut().unwrap().insert(
                                            self.clasters.as_ref().unwrap()[i].id.clone(),
                                            self.colors[i],
                                        );
                                    }
                                }
                            }
                        });
                        egui::Grid::new("start_grid")
                            .min_row_height(12.0)
                            .min_col_width(12.0)
                            .spacing([6.0, 0.0])
                            .show(ui, |ui| {
                                ui.label("Изначальные векторы-признаки");
                                ui.end_row();
                                for v in &self.data {
                                    ui.label(
                                        egui::RichText::new(format!("{:?}", v))
                                            .font(egui::FontId::proportional(20.0)),
                                    );
                                    ui.end_row();
                                }
                            });
                        egui::Grid::new("start_grid")
                            .min_row_height(12.0)
                            .min_col_width(12.0)
                            .spacing([6.0, 0.0])
                            .show(ui, |ui| {
                                ui.label("Конечные векторы-признаки");
                                ui.end_row();
                                match self.colors_map.as_ref() {
                                    Some(color_map) => {
                                        for claster in self.clasters.as_ref().unwrap() {
                                            ui.label(
                                                egui::RichText::new(format!(
                                                    "Прототип: {:?}",
                                                    claster.v
                                                ))
                                                .font(egui::FontId::proportional(25.0))
                                                .color(color_map[&claster.id])
                                                .strong(),
                                            );
                                            ui.end_row();
                                            for index in &claster.indexes {
                                                ui.label(
                                                    egui::RichText::new(format!(
                                                        "{:?}",
                                                        self.data[*index]
                                                    ))
                                                    .font(egui::FontId::proportional(20.0))
                                                    .color(color_map[&claster.id]),
                                                );
                                                ui.end_row();
                                            }
                                        }
                                    }
                                    None => {
                                        for v in &self.data {
                                            ui.label(
                                                egui::RichText::new(format!("{:?}", v))
                                                    .font(egui::FontId::proportional(20.0)),
                                            );
                                            ui.end_row();
                                        }
                                    }
                                }
                            });
                    });
                });
        });
    }
}
