use crossbeam::channel::{unbounded, Receiver, Sender};
use eframe::{run_native, App, CreationContext};
use egui::{Align, CollapsingHeader, Color32, Context, Layout, ScrollArea, Slider, Ui, Vec2};
use egui_graphs::{Change, ChangeNode, Edge, Graph, GraphView, Node, SettingsInteraction};
use petgraph::{
    stable_graph::{NodeIndex, StableUnGraph},
    visit::EdgeRef,
    Undirected,
};
use rand::Rng;

mod neuro;
mod settings;

use neuro::*;

pub struct NeuroApp {
    g: Graph<(), (), Undirected>,
    settings_style: settings::SettingsStyle,
    settings_navigation: settings::SettingsNavigation,
    changes_receiver: Receiver<Change>,
    changes_sender: Sender<Change>,
    neuro_layers: NeuroLayers,
    activation_func: ActivationFunc,
    learning_mode: LearningMode,
    learning_norm: f32,
    amount_epoch: usize,
    network: Option<neuro::NeuroNetwork>,
}

fn distance(a: Vec2, b: Vec2) -> f32 {
    ((a.x - b.x).powi(2) + (a.y - b.y).powi(2)).sqrt()
}

impl NeuroApp {
    fn new(_: &CreationContext<'_>) -> Self {
        let (changes_sender, changes_receiver) = unbounded();
        let mut app = Self {
            g: Graph::from(&StableUnGraph::default()),
            settings_style: settings::SettingsStyle::default(),
            settings_navigation: settings::SettingsNavigation::default(),
            changes_receiver: changes_receiver,
            changes_sender: changes_sender,
            neuro_layers: NeuroLayers::Zero,
            activation_func: ActivationFunc::Sig,
            learning_mode: LearningMode::OneByOne,
            learning_norm: 0.5,
            amount_epoch: 1000,
            network: None,
        };
        app
    }
}

impl App for NeuroApp {
    fn update(&mut self, ctx: &Context, _: &mut eframe::Frame) {
        egui::SidePanel::right("right_panel")
            .min_width(250.)
            .show(ctx, |ui| {
                ScrollArea::vertical().show(ui, |ui| {
                    CollapsingHeader::new("Neuro Network")
                        .default_open(true)
                        .show(ui, |ui| {
                            ui.add_space(10.0);

                            ui.label("Amount of hidden layers");
                            ui.separator();
                            ui.horizontal(|ui| {
                                ui.radio_value(&mut self.neuro_layers, NeuroLayers::Zero, "0");
                                ui.radio_value(&mut self.neuro_layers, NeuroLayers::One, "1");
                                ui.radio_value(&mut self.neuro_layers, NeuroLayers::Two, "2");
                            });

                            ui.add_space(10.0);

                            ui.label("Activation function");
                            ui.separator();
                            ui.horizontal(|ui| {
                                ui.radio_value(
                                    &mut self.activation_func,
                                    ActivationFunc::Sig,
                                    "Sig",
                                );
                                ui.radio_value(
                                    &mut self.activation_func,
                                    ActivationFunc::Gip,
                                    "Gip",
                                );
                                ui.radio_value(
                                    &mut self.activation_func,
                                    ActivationFunc::Arctn,
                                    "ArcTg",
                                );
                            });

                            ui.add_space(10.0);

                            ui.label("Learning mode");
                            ui.separator();
                            ui.horizontal(|ui| {
                                ui.radio_value(
                                    &mut self.learning_mode,
                                    LearningMode::OneByOne,
                                    "One-by-one",
                                );
                                ui.radio_value(
                                    &mut self.learning_mode,
                                    LearningMode::Packet,
                                    "Packet",
                                );
                            });

                            ui.add_space(10.0);

                            ui.label("Learning norm");
                            ui.separator();
                            ui.add(Slider::new(&mut self.learning_norm, 0. ..=1.));

                            ui.add_space(10.0);

                            ui.label("Amount of epoch");
                            ui.separator();
                            ui.add(Slider::new(&mut self.amount_epoch, 1..=4000));

                            ui.add_space(10.0);

                            ui.label("Actions");
                            ui.separator();
                            ui.horizontal(|ui| {
                                ui.button("Recognize");
                                if ui.button("Learn").clicked() {
                                    let mut net = neuro::NeuroNetwork::new(vec![2, 6, 2])
                                        .with_epoch(10000)
                                        .with_batch_size(1);
                                    let examples = vec![
                                        Sample {
                                            data: vec![1.0, 1.0],
                                            solution: vec![1.0, 0.0],
                                        },
                                        Sample {
                                            data: vec![1.0, 0.0],
                                            solution: vec![0.0, 1.0],
                                        },
                                        Sample {
                                            data: vec![0.0, 1.0],
                                            solution: vec![0.0, 1.0],
                                        },
                                        Sample {
                                            data: vec![0.0, 0.0],
                                            solution: vec![1.0, 0.0],
                                        },
                                    ];
                                    net.train(examples, 0.1);
                                    let output = net.solve(vec![1.0, 1.0]);
                                    println!("{:?}", output);

                                    let output = net.solve(vec![1.0, 0.0]);
                                    println!("{:?}", output);

                                    let output = net.solve(vec![0.0, 1.0]);
                                    println!("{:?}", output);

                                    let output = net.solve(vec![0.0, 0.0]);
                                    println!("{:?}", output);
                                }
                            });
                        });
                    CollapsingHeader::new("Ui")
                        .default_open(true)
                        .show(ui, |ui| {
                            ui.add_space(10.0);

                            ui.label("Ui settings");
                            ui.separator();
                        });
                });
            });
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.with_layout(Layout::left_to_right(Align::TOP), |ui| {
                egui::Grid::new("some_unique_id")
                    .spacing(Vec2::new(1., 1.))
                    .show(ui, |ui| {
                        for i in 0..8 {
                            ui.button("  ");
                            ui.button("  ");
                            ui.button("  ");
                            ui.button("  ");
                            ui.button("  ");
                            ui.button("  ");
                            ui.button("  ");
                            ui.button("  ");
                            ui.end_row();
                        }
                    });
                ui.add_space(12.0);
                egui::Grid::new("unique_id_2")
                    .spacing(Vec2::new(1., 1.))
                    .show(ui, |ui| {
                        for i in 0..8 {
                            ui.button("  ");
                            ui.button("  ");
                            ui.end_row();
                        }
                    });
            });
        });
    }
}

fn main() {
    let native_options = eframe::NativeOptions::default();
    run_native(
        "Ant-algo",
        native_options,
        Box::new(|cc| Box::new(NeuroApp::new(cc))),
    )
    .unwrap();
}
