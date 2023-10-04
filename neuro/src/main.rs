use crossbeam::channel::{unbounded, Receiver};
use eframe::{
    egui::{
        Align, CentralPanel, CollapsingHeader, Color32, Context, Grid, Label, Layout, RichText,
        ScrollArea, SidePanel, Slider, Vec2,
    },
    run_native, App, CreationContext,
};
use image;
use image::GenericImageView;
use notify::{Error, Event, RecommendedWatcher, RecursiveMode, Watcher};

use std::time::Duration;

mod activation;
mod error_func;
mod neuro;

use activation::*;
use neuro::*;

struct LayerOptions {
    neurons: usize,
}

pub struct NeuroApp {
    changes_receiver: Receiver<notify::Event>,
    neuro_layers: NeuroLayers,
    activation_func: ActivationFunc,
    learning_norm: f32,
    amount_epoch: usize,
    batch_size: usize,
    input_file_path: String,
    train_data_folder_path: String,
    network: Option<neuro::NeuroNetwork>,
    watcher: RecommendedWatcher,
    watching: Option<String>,
    promise: Option<poll_promise::Promise<NeuroNetwork>>,
    toasts: egui_notify::Toasts,
}

impl NeuroApp {
    fn new(_: &CreationContext<'_>) -> Self {
        let (changes_sender, changes_receiver) = unbounded();
        let mut watcher: RecommendedWatcher = Watcher::new(
            move |result: Result<Event, Error>| match result {
                Ok(event) => {
                    if event.kind.is_modify() {
                        changes_sender.send(event);
                    }
                }
                Err(e) => (),
            },
            notify::Config::default()
                .with_compare_contents(true)
                .with_poll_interval(Duration::from_millis(500)),
        )
        .unwrap();
        let mut app = Self {
            changes_receiver: changes_receiver,
            neuro_layers: NeuroLayers::Zero,
            activation_func: ActivationFunc::Sigmoid,
            learning_norm: 0.5,
            batch_size: 1,
            amount_epoch: 1000,
            input_file_path: "".into(),
            train_data_folder_path: "".into(),
            network: None,
            watcher: watcher,
            watching: None,
            promise: None,
            toasts: egui_notify::Toasts::default(),
        };
        app
    }
    fn handle_changes(&mut self) {
        if self.network.is_none() {
            return;
        }
        let file_path = &self.input_file_path;
        self.changes_receiver.try_iter().for_each(|_| {
            let input = get_file_data(file_path.clone());
            let output = self.network.as_mut().unwrap().solve(input);
            println!("R: {:#?}", output);
        });
    }
}

fn get_file_data(file_path: String) -> Vec<f32> {
    let mut img = image::open(file_path).unwrap();
    let mut input: Vec<f32> = vec![0.0; 784];
    for pixel in img.pixels() {
        input[pixel.0 as usize * 28 + pixel.1 as usize] = 1.0 - pixel.2 .0[0] as f32 / 255.0;
    }
    input
}

fn train() -> NeuroNetwork {
    let train_len = 100;
    let mut train_labels: Vec<u8> = mnist_read::read_labels("train-labels-idx1-ubyte");
    let mut train_data = mnist_read::read_data("train-images-idx3-ubyte")
        .chunks(784)
        .map(|v| v.into())
        .collect::<Vec<Vec<u8>>>();
    train_labels.truncate(train_len);
    train_data.truncate(train_len);
    let examples = train_labels
        .iter()
        .zip(train_data.iter())
        .map(|(label, data)| {
            let mut v: Vec<f32> = vec![0.0; 10];
            v[*label as usize] = 1.0;
            Sample {
                data: data.iter().map(|x| *x as f32 / 255.0).collect::<Vec<_>>(),
                solution: v,
            }
        })
        .collect::<Vec<_>>();
    let mut net = neuro::NeuroNetwork::new(vec![784, 32, 32, 16, 10])
        .with_epoch(200)
        .with_batch_size(10)
        .with_activation(ActivationFunc::Relu)
        .with_last_activation(ActivationFunc::Softmax);

    net.train(examples.clone(), 0.03);
    net
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
                                    ActivationFunc::Sigmoid,
                                    "Sig",
                                );
                                ui.radio_value(
                                    &mut self.activation_func,
                                    ActivationFunc::Relu,
                                    "Relu",
                                );
                                ui.radio_value(
                                    &mut self.activation_func,
                                    ActivationFunc::Softmax,
                                    "Softmax",
                                );
                            });

                            ui.add_space(10.0);

                            ui.label("Batch size");
                            ui.separator();
                            ui.add(Slider::new(&mut self.batch_size, 1..=128));

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
                                if ui.button("Learn").clicked() {
                                    if self.promise.is_none() {
                                        let path =
                                            std::path::Path::new(&self.train_data_folder_path);
                                        if path.exists() {
                                            if path.is_dir() {
                                                self.network = None;
                                                self.promise = Some(poll_promise::Promise::<
                                                    NeuroNetwork,
                                                >::spawn_thread(
                                                    "Neural network training",
                                                    move || train(),
                                                ));
                                            } else if path.is_file() {
                                                let file = std::fs::File::open(path).unwrap();
                                                let reader = std::io::BufReader::new(file);
                                                let net: NeuroNetworkJson =
                                                    serde_json::from_reader(reader).unwrap();
                                                self.network = Some(json_to_network(net));
                                            }
                                        }
                                    }
                                }
                            });
                            if self.promise.is_some() {
                                ui.spinner();
                            }
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
        CentralPanel::default().show(ctx, |ui| {
            ui.with_layout(Layout::left_to_right(Align::TOP), |ui| {
                ui.vertical(|ui| {
                    ui.horizontal(|ui| {
                        ui.label("Input file");
                        ui.add_visible(
                            self.watching.is_none(),
                            Label::new(RichText::new("Wrong path").color(Color32::RED)),
                        )
                    });
                    if ui.text_edit_singleline(&mut self.input_file_path).changed() {
                        match self.watcher.watch(
                            std::path::Path::new(&self.input_file_path),
                            RecursiveMode::NonRecursive,
                        ) {
                            Ok(_) => {
                                match &self.watching {
                                    Some(path) => {
                                        println!("Unwatch: {}", path);
                                        self.watcher.unwatch(std::path::Path::new(&path))
                                    }
                                    None => Ok(()),
                                }
                                .unwrap();
                                self.watching = Some(self.input_file_path.clone());
                            }
                            Err(_) => {
                                match self.watching.clone() {
                                    Some(path) => {
                                        println!("Unwatch: {}", path);
                                        self.watching = None;
                                        self.watcher.unwatch(std::path::Path::new(&path))
                                    }
                                    None => Ok(()),
                                }
                                .unwrap();
                            }
                        }
                    }
                    ui.add_space(6.0);
                    ui.label("Folder with training data");
                    ui.text_edit_singleline(&mut self.train_data_folder_path);
                });
                ui.add_space(12.0);
                Grid::new("unique_id_2")
                    .spacing(Vec2::new(1., 1.))
                    .show(ui, |ui| {
                        for i in 0..8 {
                            ui.button("  ");
                            ui.button("  ");
                            ui.end_row();
                        }
                    });
            });
            self.toasts.show(ctx);
        });
        self.handle_changes();
        self.promise = match self.promise.take() {
            Some(p) => match p.try_take() {
                Ok(value) => {
                    self.toasts
                        .success("Training completed")
                        .set_duration(Some(std::time::Duration::from_secs(2)));
                    self.network = Some(value);
                    None
                }
                Err(p) => Some(p),
            },
            None => None,
        };
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
