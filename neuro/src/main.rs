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

mod activation;
mod error_func;
mod neuro;

use activation::*;
use neuro::*;

#[derive(Debug, Clone)]
struct LayerOptions {
    neurons: usize,
}

pub struct NeuroApp {
    // Changes
    changes_receiver: Receiver<notify::Event>,
    layers_activation: ActivationFunc,
    final_activation: ActivationFunc,
    learning_norm: f32,
    amount_epoch: usize,
    batch_size: usize,
    // Path to input file
    input_file_path: String,
    // Path to training data or json-network
    train_data_folder_path: String,
    // Neural network
    network: Option<neuro::NeuralNetwork>,
    solution: Option<Vec<f32>>,
    best_solution: Option<usize>,
    // Input file watch
    watcher: RecommendedWatcher,
    watching: Option<String>,
    // Promise for training function
    promise: Option<poll_promise::Promise<NeuralNetwork>>,
    // Notifications
    toasts: egui_notify::Toasts,
    // Layers Options
    picked_layers: usize,
    layers_options: Vec<LayerOptions>,
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
        layers_options[0] = 784;
        let mut app = Self {
            changes_receiver: changes_receiver,
            layers_activation: ActivationFunc::Sigmoid,
            final_activation: ActivationFunc::Sigmoid,
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
            layers_options: layers_options,
            picked_layers: 2,
            solution: None,
            best_solution: None,
        };
        app
    }
    fn handle_changes(&mut self) {
        if self.network.is_none() || self.watching.is_none() {
            return;
        }
        let file_path = self.watching.as_ref().unwrap();
        self.changes_receiver.try_iter().for_each(|_| {
            let input = get_file_data(file_path.clone());
            let output = self.network.as_mut().unwrap().solve(input);
            let best_solution = output
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(index, _)| index)
                .unwrap();
            self.solution = Some(output);
            self.best_solution = Some(best_solution);
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

fn train(
    path: &std::path::Path,
    layers_activation: ActivationFunc,
    final_activation: ActivationFunc,
    layers_options: Vec<usize>,
    epoch: usize,
    batch_size: usize,
    learning_rate: f32,
) -> NeuralNetwork {
    let file = std::fs::File::open(path.join(std::path::Path::new("train.json"))).unwrap();
    let reader = std::io::BufReader::new(file);
    let labels: Vec<String> = serde_json::from_reader(reader).unwrap();
    let samples: Vec<Sample> = std::fs::read_dir(path)
        .unwrap()
        .map(|x| x.unwrap())
        .filter(|x| {
            let name = String::from(x.file_name().to_str().unwrap());
            !name.ends_with(".json")
        })
        .map(|x| {
            let name = String::from(x.file_name().to_str().unwrap());
            let label = name.chars().take_while(|&x| x != '_').collect::<String>();
            let inputs = get_file_data(String::from(x.path().to_str().unwrap()));
            let mut output: Vec<f32> = vec![0.0; labels.len()];
            output[labels.iter().position(|x| *x == label).unwrap()] = 1.0;
            Sample {
                data: inputs,
                solution: output,
            }
        })
        .collect();
    let mut net = neuro::NeuralNetwork::new(layers_options, labels)
        .with_activation(layers_activation)
        .with_last_activation(final_activation)
        .with_epoch(epoch)
        .with_batch_size(batch_size);

    net.train(samples, learning_rate);
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
                            ui.add_enabled(self.network.is_none(), |ui: &mut Ui| {
                                ui.add_space(10.0);

                                ui.label("Amount of layers");
                                ui.separator();
                                ui.add(Slider::new(&mut self.picked_layers, 2..=16));

                                ui.add_space(10.0);

                                ui.label("Activation function");
                                ui.separator();
                                ui.horizontal(|ui| {
                                    ui.radio_value(
                                        &mut self.layers_activation,
                                        ActivationFunc::Sigmoid,
                                        "Sig",
                                    );
                                    ui.radio_value(
                                        &mut self.layers_activation,
                                        ActivationFunc::Relu,
                                        "Relu",
                                    );
                                    ui.radio_value(
                                        &mut self.layers_activation,
                                        ActivationFunc::Softmax,
                                        "Softmax",
                                    );
                                    ui.radio_value(
                                        &mut self.layers_activation,
                                        ActivationFunc::Arctan,
                                        "Atan",
                                    );
                                    ui.radio_value(
                                        &mut self.layers_activation,
                                        ActivationFunc::Tanh,
                                        "Tanh",
                                    );
                                });

                                ui.add_space(10.0);

                                ui.label("Final activation function");
                                ui.separator();
                                ui.horizontal(|ui| {
                                    ui.radio_value(
                                        &mut self.final_activation,
                                        ActivationFunc::Sigmoid,
                                        "Sig",
                                    );
                                    ui.radio_value(
                                        &mut self.final_activation,
                                        ActivationFunc::Relu,
                                        "Relu",
                                    );
                                    ui.radio_value(
                                        &mut self.final_activation,
                                        ActivationFunc::Softmax,
                                        "Softmax",
                                    );
                                    ui.radio_value(
                                        &mut self.final_activation,
                                        ActivationFunc::Arctan,
                                        "Atan",
                                    );
                                    ui.radio_value(
                                        &mut self.final_activation,
                                        ActivationFunc::Tanh,
                                        "Tanh",
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
                                let r = ui.add(Slider::new(&mut self.amount_epoch, 1..=50000));

                                ui.add_space(10.0);
                                r
                            });

                            ui.label("Actions");
                            ui.separator();
                            ui.horizontal(|ui| {
                                if ui.button("Learn").clicked() {
                                    if self.promise.is_none() {
                                        let path_str = self.train_data_folder_path.clone();
                                        if std::path::Path::new(&path_str).exists() {
                                            if std::path::Path::new(&path_str).is_dir() {
                                                self.network = None;
                                                self.best_solution = None;
                                                self.solution = None;
                                                let layers_activation =
                                                    self.layers_activation.clone();
                                                let final_activation =
                                                    self.final_activation.clone();
                                                let layers_options = self
                                                    .layers_options
                                                    .clone()
                                                    .into_iter()
                                                    .take(self.picked_layers)
                                                    .map(|x| x.neurons)
                                                    .collect();
                                                let amount_epoch = self.amount_epoch.clone();
                                                let batch_size = self.batch_size.clone();
                                                let learning_norm = self.learning_norm.clone();
                                                self.promise = Some(poll_promise::Promise::<
                                                    NeuralNetwork,
                                                >::spawn_thread(
                                                    "Neural network training",
                                                    move || {
                                                        train(
                                                            std::path::Path::new(&path_str),
                                                            layers_activation,
                                                            final_activation,
                                                            layers_options,
                                                            amount_epoch,
                                                            batch_size,
                                                            learning_norm,
                                                        )
                                                    },
                                                ));
                                            } else if std::path::Path::new(&path_str).is_file() {
                                                self.network = None;
                                                self.best_solution = None;
                                                self.solution = None;
                                                let file = std::fs::File::open(
                                                    std::path::Path::new(&path_str),
                                                )
                                                .unwrap();
                                                let reader = std::io::BufReader::new(file);
                                                let net: NeuralNetworkJson =
                                                    serde_json::from_reader(reader).unwrap();
                                                let net = json_to_network(net);
                                                self.picked_layers = net.layers.len();
                                                self.layers_options
                                                    .iter_mut()
                                                    .take(self.picked_layers)
                                                    .zip(net.layers.iter().take(self.picked_layers))
                                                    .for_each(|(opt, layer)| {
                                                        opt.neurons = layer.neurons();
                                                    });
                                                self.amount_epoch = net.epoch();
                                                self.batch_size = net.batch_size();
                                                self.learning_norm = net.learning_rate();
                                                self.layers_activation = net.activation();
                                                self.final_activation = net.final_activation();
                                                self.network = Some(net);
                                            }
                                        }
                                    }
                                }
                                if ui.button("Drop").clicked() {
                                    if self.promise.is_none() {
                                        self.train_data_folder_path = "".into();
                                        self.network = None;
                                        self.best_solution = None;
                                        self.solution = None;
                                    }
                                }
                                if ui.button("Save").clicked() {
                                    if self.promise.is_none() {
                                        let path =
                                            std::path::Path::new(&self.train_data_folder_path);
                                        if !path.exists() || !path.is_dir() {
                                            let j = neural_to_json(self.network.as_ref().unwrap());
                                            std::fs::write(
                                                path,
                                                serde_json::to_string(&j).unwrap(),
                                            )
                                            .unwrap();
                                            self.train_data_folder_path = "".into();
                                            self.network = None;
                                            self.best_solution = None;
                                            self.solution = None;
                                        }
                                    }
                                }
                            });
                            if self.promise.is_some() {
                                ui.spinner();
                            }
                        });
                    CollapsingHeader::new("Layers settings")
                        .default_open(true)
                        .show(ui, |ui| {
                            ui.add_enabled(self.network.is_none(), |ui: &mut Ui| {
                                ui.add_space(10.0);

                                ui.label(format!("Layer {}", 1));
                                ui.separator();
                                let r = ui.add(Slider::new(
                                    &mut self.layers_options[0].neurons,
                                    1..=3000,
                                ));
                                for i in 1..self.picked_layers {
                                    ui.add_space(10.0);

                                    let r = ui.label(format!("Layer {}", i + 1));
                                    ui.separator();
                                    ui.add(Slider::new(
                                        &mut self.layers_options[i].neurons,
                                        1..=3000,
                                    ));
                                }
                                r
                            });
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
                                    Some(path) => self.watcher.unwatch(std::path::Path::new(&path)),
                                    None => Ok(()),
                                }
                                .unwrap();
                                self.watching = Some(self.input_file_path.clone());
                            }
                            Err(_) => {
                                match self.watching.clone() {
                                    Some(path) => {
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
                    ui.label("Training data load/save path");
                    ui.text_edit_singleline(&mut self.train_data_folder_path);
                });
                ui.add_space(12.0);
                match self.solution.clone() {
                    Some(output) => {
                        let labels = &self.network.as_ref().unwrap().labels();
                        Grid::new("unique_id_2")
                            .spacing(Vec2::new(1., 1.))
                            .show(ui, |ui| {
                                for i in 0..output.len() {
                                    if i == self.best_solution.unwrap() {
                                        ui.label(
                                            RichText::new(labels[i].clone()).color(Color32::GREEN),
                                        );
                                    } else {
                                        ui.label(RichText::new(labels[i].clone()));
                                    }
                                    ui.label(format!("{}", output[i]));
                                    ui.end_row();
                                }
                            });
                    }
                    None => (),
                }
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
