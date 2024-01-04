use crossbeam::channel::{unbounded, Receiver};
use eframe::{
    egui::{
        Align, CentralPanel, CollapsingHeader, Color32, Context, Grid, Label, Layout, RichText,
        ScrollArea, SidePanel, Slider, Ui, Vec2,
    },
    run_native, App, CreationContext,
};
use image;
use image::GenericImageView;
use notify::{Error, Event, RecommendedWatcher, RecursiveMode, Watcher};

use std::time::Duration;

mod kosko;

#[derive(Debug, Clone)]
struct LayerOptions {
    neurons: usize,
}

pub struct NeuroApp {
    network: Option<kosko::Network>,
    solution: Option<Vec<i32>>,
    sample_size: usize,
    answer_size: usize,
    sample_amount: usize,
    input: Vec<i32>,
    samples: Vec<Vec<i32>>,
    answers: Vec<Vec<i32>>,
    toasts: egui_notify::Toasts,
}

impl NeuroApp {
    fn new(_: &CreationContext<'_>) -> Self {
        let (changes_sender, changes_receiver) = unbounded();
        let mut now = std::time::SystemTime::now();
        let duration = std::time::Duration::from_millis(500);
        let mut watcher: RecommendedWatcher = Watcher::new(
            move |result: Result<Event, Error>| match result {
                Ok(event) => {
                    if event.kind.is_modify() {
                        if now.elapsed().unwrap() > duration {
                            now = std::time::SystemTime::now();
                            changes_sender.send(event);
                        }
                    }
                }
                Err(e) => (),
            },
            notify::Config::default()
                .with_compare_contents(true)
                .with_poll_interval(Duration::from_millis(500)),
        )
        .unwrap();
        let mut layers_options = vec![LayerOptions { neurons: 1 }; 16];
        layers_options[0].neurons = 784;
        layers_options[1].neurons = 2;
        let app = Self {
            network: None,
            sample_amount: 1,
            answer_size: 1,
            sample_size: 1,
            input: vec![1],
            samples: vec![vec![1]],
            answers: vec![vec![1]],
            solution: None,
            toasts: egui_notify::Toasts::default(),
        };
        app
    }
}

impl App for NeuroApp {
    fn update(&mut self, ctx: &Context, _: &mut eframe::Frame) {
        SidePanel::right("right_panel")
            .min_width(250.)
            .show(ctx, |ui| {
                ScrollArea::vertical().show(ui, |ui| {
                    CollapsingHeader::new("Neuro Network")
                        .default_open(true)
                        .show(ui, |ui| {
                            ui.add_space(10.0);

                            ui.label("Samples amount");
                            ui.separator();
                            let samples_before = self.sample_amount;
                            ui.add(Slider::new(&mut self.sample_amount, 1..=32));
                            if self.sample_amount.abs_diff(samples_before) != 0 {
                                for _ in 0..self.sample_amount.abs_diff(samples_before) {
                                    if self.sample_amount > samples_before {
                                        self.samples.push(vec![1; self.sample_size]);
                                        self.answers.push(vec![1; self.answer_size]);
                                    } else {
                                        self.samples.truncate(self.sample_amount);
                                    }
                                }
                            }

                            ui.add_space(10.0);

                            ui.label("Samples size");
                            ui.separator();
                            let sample_size_before = self.sample_size;
                            ui.add(Slider::new(&mut self.sample_size, 1..=32));
                            if self.sample_size.abs_diff(sample_size_before) != 0 {
                                if self.sample_size > sample_size_before {
                                    for i in 0..self.samples.len() {
                                        self.samples[i].append(&mut vec![
                                            1;
                                            self.sample_size.abs_diff(
                                                sample_size_before
                                            )
                                        ]);
                                    }
                                    self.input.append(&mut vec![
                                        1;
                                        self.sample_size
                                            .abs_diff(sample_size_before)
                                    ])
                                } else {
                                    for i in 0..self.samples.len() {
                                        self.samples[i].truncate(self.sample_size);
                                    }
                                    self.input.truncate(self.sample_size);
                                }
                            }

                            ui.add_space(10.0);

                            ui.label("Answer size");
                            ui.separator();
                            let answer_size_before = self.answer_size;
                            ui.add(Slider::new(&mut self.answer_size, 1..=32));
                            if self.answer_size.abs_diff(answer_size_before) != 0 {
                                if self.answer_size > answer_size_before {
                                    for i in 0..self.answers.len() {
                                        self.answers[i].append(&mut vec![
                                            1;
                                            self.answer_size.abs_diff(
                                                answer_size_before
                                            )
                                        ]);
                                    }
                                } else {
                                    for i in 0..self.answers.len() {
                                        self.answers[i].truncate(self.answer_size);
                                    }
                                }
                            }

                            ui.add_space(10.0);

                            ui.label("Actions");
                            ui.separator();
                            ui.horizontal(|ui| {
                                if ui.button("Learn").clicked() {
                                    let samples = self
                                        .samples
                                        .iter()
                                        .map(|x| ndarray::Array1::from_vec(x.clone()))
                                        .collect::<Vec<_>>();
                                    let answers = self
                                        .answers
                                        .iter()
                                        .map(|x| ndarray::Array1::from_vec(x.clone()))
                                        .collect::<Vec<_>>();
                                    self.network = Some(
                                        kosko::Network::new(self.answer_size, self.sample_size)
                                            .train(&samples, &answers),
                                    );
                                }
                            });
                        });
                });
            });
        CentralPanel::default().show(ctx, |ui| {
            ui.with_layout(Layout::left_to_right(Align::TOP), |ui| {
                ui.vertical(|ui| {
                    ui.label("Samples / Asnwers");
                    for i in 0..self.samples.len() {
                        ui.horizontal(|ui| {
                            ui.label(format!("{}", i));
                            ui.add_space(6.0);
                            for j in 0..self.samples[i].len() {
                                if ui.button(format!("{}", self.samples[i][j])).clicked() {
                                    self.samples[i][j] =
                                        if self.samples[i][j] == 1 { -1 } else { 1 }
                                }
                            }
                            ui.add_space(12.0);
                            for j in 0..self.answers[i].len() {
                                if ui.button(format!("{}", self.answers[i][j])).clicked() {
                                    self.answers[i][j] = if self.answers[i][j] == 1 {
                                        -1
                                    } else {
                                        1
                                    }
                                }
                            }
                        });
                    }
                    match &self.network {
                        Some(n) => {
                            ui.label("Weights:");
                            let shape = n.w.shape();
                            for i in 0..shape[0] {
                                ui.horizontal(|ui| {
                                    for j in 0..shape[1] {
                                        ui.button(format!("{}", n.w.get((i, j)).unwrap()));
                                    }
                                });
                            }
                        }
                        None => (),
                    }
                });
                ui.add_space(12.0);
                ui.vertical(|ui| {
                    ui.label("Input");
                    ui.horizontal(|ui| {
                        for i in 0..self.input.len() {
                            if ui.button(format!("{}", self.input[i])).clicked() {
                                self.input[i] = if self.input[i] == 1 { -1 } else { 1 };
                                match &self.network {
                                    Some(n) => {
                                        self.solution = Some(n.predict(&ndarray::Array1::from_vec(
                                            self.input.clone(),
                                        )))
                                    }
                                    None => (),
                                }
                            }
                        }
                    });
                });
                match self.solution.clone() {
                    Some(output) => {
                        ui.vertical(|ui| {
                            ui.add_space(20.0);
                            ui.label("Solution:");
                            ui.horizontal(|ui| {
                                for i in 0..output.len() {
                                    ui.button(format!("{}", output[i]));
                                }
                            });
                        });
                    }
                    None => (),
                }
            });
        });
    }
}

fn main() {
    let native_options = eframe::NativeOptions::default();
    run_native(
        "Kosko",
        native_options,
        Box::new(|cc| Box::new(NeuroApp::new(cc))),
    )
    .unwrap();
}
