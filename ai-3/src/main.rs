use ant_algo::IterationInfo;
use eframe::{run_native, App, CreationContext};
use egui::{CollapsingHeader, Color32, Context, ScrollArea, Slider, Ui, Vec2};
use egui_graphs::{Edge, Graph, GraphView, Node};
use petgraph::{
    stable_graph::{NodeIndex, StableUnGraph},
    visit::EdgeRef,
    Undirected,
};
use rand::Rng;

mod ant_algo;
mod settings;

pub struct AntOptions {
    alpha: f32,
    beta: f32,
    q: f32,
    p: f32,
    nodes: i64,
    ant_amount: i64,
    iterations_amount: i64,
}

impl Default for AntOptions {
    fn default() -> Self {
        AntOptions {
            alpha: 0.5,
            beta: 0.5,
            q: 64.0,
            p: 0.5,
            nodes: 3,
            ant_amount: 1,
            iterations_amount: 1,
        }
    }
}

struct SolutionInfo {
    solution: Vec<IterationInfo>,
    best_iteration: usize,
    best_ant: i64,
}

pub struct AntApp {
    g: Graph<(), ant_algo::EdgeInfo, Undirected>,
    ant_options: AntOptions,
    settings_style: settings::SettingsStyle,
    settings_navigation: settings::SettingsNavigation,
    solution: Option<SolutionInfo>,
    iteration_i: i64,
    ant_i: i64,
    edge_i: i64,
    show_pheromones: bool,
    pheromones_k: f32,
}

fn distance(a: Vec2, b: Vec2) -> f32 {
    ((a.x - b.x).powi(2) + (a.y - b.y).powi(2)).sqrt()
}

impl AntApp {
    fn new(_: &CreationContext<'_>) -> Self {
        let mut app = Self {
            g: Graph::from(&StableUnGraph::default()),
            ant_options: AntOptions::default(),
            settings_style: settings::SettingsStyle::default(),
            settings_navigation: settings::SettingsNavigation::default(),
            solution: None,
            iteration_i: 0,
            ant_i: 0,
            edge_i: 0,
            show_pheromones: true,
            pheromones_k: 1.0,
        };
        for _ in 0..app.ant_options.nodes {
            app.add_random_node();
        }
        app
    }

    fn remove_edges(&mut self, start: NodeIndex, end: NodeIndex) {
        let g_idxs = self
            .g
            .g
            .edges_connecting(start, end)
            .map(|e| e.id())
            .collect::<Vec<_>>();
        if g_idxs.is_empty() {
            return;
        }

        g_idxs.iter().for_each(|e| {
            self.g.g.remove_edge(*e).unwrap();
        });
    }
    fn connect_node(&mut self, node: NodeIndex) {
        let indexes: Vec<_> = self.g.g.node_indices().collect();
        indexes.into_iter().for_each(|x| {
            if x != node {
                let mut edge_data = ant_algo::EdgeInfo {
                    distance: distance(
                        self.g.node(x).unwrap().location(),
                        self.g.node(node).unwrap().location(),
                    ),
                    pheromones: 0.0,
                    probability_parameters: 0.0,
                };
                edge_data.recalculate(self.ant_options.alpha, self.ant_options.beta);
                self.g.g.add_edge(
                    x,
                    node,
                    Edge::new(edge_data)
                        .with_color(Color32::from_rgba_unmultiplied(128, 128, 128, 0)),
                );
            }
        });
    }
    fn remove_node(&mut self, idx: NodeIndex) {
        let neighbors = self.g.g.neighbors_undirected(idx).collect::<Vec<_>>();
        neighbors.iter().for_each(|n| {
            self.remove_edges(idx, *n);
        });

        self.g.g.remove_node(idx).unwrap();
    }
    fn random_node_idx(&self) -> Option<NodeIndex> {
        let nodes_cnt = self.g.g.node_count();
        if nodes_cnt == 0 {
            return None;
        }

        let random_n_idx = rand::thread_rng().gen_range(0..nodes_cnt);
        self.g.g.node_indices().nth(random_n_idx)
    }
    fn remove_random_node(&mut self) {
        let idx = self.random_node_idx().unwrap();
        self.remove_node(idx);
    }
    fn add_random_node(&mut self) {
        let random_n_idx = self.random_node_idx();
        if random_n_idx.is_none() {
            self.g.g.add_node(Node::<()>::default());
            return;
        }

        let random_n = self.g.g.node_weight(random_n_idx.unwrap()).unwrap();
        // location of new node is in surrounging of random existing node
        let mut rng = rand::thread_rng();
        let x_sign = (rng.gen_range(0..1) * -2 + 1) as f32;
        let y_sign = (rng.gen_range(0..1) * -2 + 1) as f32;
        let location = Vec2::new(
            random_n.location().x + 30. * x_sign + rng.gen_range(0. ..200.) * x_sign,
            random_n.location().y + 30. * y_sign + rng.gen_range(0. ..200.) * y_sign,
        );

        let idx = self.g.g.add_node(Node::new(location, ()));
        self.connect_node(idx);
    }
    fn ant_options_sliders(&mut self, ui: &mut Ui) {
        let nodes_before = self.ant_options.nodes;
        ui.add(Slider::new(&mut self.ant_options.nodes, 3..=200).text("Nodes"));
        let delta_nodes = self.ant_options.nodes - nodes_before;
        if delta_nodes != 0 {
            self.reset_graph();
        }
        (0..delta_nodes.abs()).for_each(|_| {
            if delta_nodes > 0 {
                self.add_random_node();
            } else {
                self.remove_random_node();
            }
        });

        ui.add(Slider::new(&mut self.ant_options.alpha, 0. ..=400.).text("alpha"));
        ui.add(Slider::new(&mut self.ant_options.beta, 0. ..=400.).text("beta"));
        ui.add(Slider::new(&mut self.ant_options.p, 0. ..=1.).text("p"));
        ui.add(Slider::new(&mut self.ant_options.q, 1. ..=10000.).text("Q"));
        ui.add(Slider::new(&mut self.ant_options.ant_amount, 1..=128).text("Ant amount"));
        ui.add(
            Slider::new(&mut self.ant_options.iterations_amount, 1..=128).text("Iterations amount"),
        );
    }
    fn ui_settings(&mut self, ui: &mut Ui) {
        if ui
            .checkbox(
                &mut self.settings_navigation.fit_to_screen_enabled,
                "Fit to screen",
            )
            .changed()
        {
            self.settings_navigation.zoom_and_pan_enabled =
                !self.settings_navigation.zoom_and_pan_enabled;
        }
    }
    fn reset_graph_color(&mut self) {
        self.g.g.node_weights_mut().for_each(|x| {
            x.clone_from(&x.clone().with_color(Color32::from_rgb(200, 200, 200)));
        });
        self.g.g.edge_weights_mut().for_each(|x| {
            x.clone_from(
                &Edge::new(x.data().unwrap().clone())
                    .with_color(Color32::from_rgba_unmultiplied(128, 128, 128, 0)),
            );
        });
    }
    fn reset_graph(&mut self) {
        self.solution = None;
        self.g.g.node_weights_mut().for_each(|x| {
            x.clone_from(&x.clone().with_color(Color32::from_rgb(200, 200, 200)));
        });
        self.g.g.edge_weights_mut().for_each(|x| {
            let mut new_edge_data = x.data().unwrap().clone();
            new_edge_data.pheromones = 0.0;
            new_edge_data.probability_parameters = 0.0;
            x.clone_from(
                &Edge::new(new_edge_data)
                    .with_color(Color32::from_rgba_unmultiplied(128, 128, 128, 0)),
            );
        });
    }
    fn update_graph(&mut self) {
        match &self.solution {
            Some(v) => {
                let v = &v.solution;
                let iteration = &v[self.iteration_i as usize];
                let ant = &iteration.ants[self.ant_i as usize];
                if self.show_pheromones {
                    iteration
                        .old_edges
                        .iter()
                        .zip(self.g.g.edge_weights_mut())
                        .for_each(|(color_edge, x)| {
                            x.clone_from(
                                &x.clone().with_color(Color32::from_rgba_unmultiplied(
                                    0,
                                    (((0.0 as f32 + 255 as f32 * color_edge.pheromones)
                                        * self.pheromones_k)
                                        as u8)
                                        .clamp(0, 255),
                                    0,
                                    (((0.0 as f32 + 255 as f32 * color_edge.pheromones)
                                        * self.pheromones_k)
                                        as u8)
                                        .clamp(0, 255),
                                )),
                            );
                        });
                }
                for i in 0..=self.edge_i {
                    let node = self.g.g.node_weight_mut(ant.tabu[i as usize]).unwrap();
                    node.clone_from(&node.clone().with_color(Color32::from_rgb(85, 24, 93)));
                }
                for i in 0..=self.edge_i {
                    let e = self
                        .g
                        .g
                        .edge_weight_mut(ant.edges[i as usize].index)
                        .unwrap();
                    e.clone_from(
                        &e.clone()
                            .with_color(Color32::from_rgba_unmultiplied(255, 213, 36, 128)),
                    );
                }
            }
            None => (),
        }
    }
}

impl App for AntApp {
    fn update(&mut self, ctx: &Context, _: &mut eframe::Frame) {
        egui::SidePanel::right("right_panel")
            .min_width(250.)
            .show(ctx, |ui| {
                ScrollArea::vertical().show(ui, |ui| {
                    CollapsingHeader::new("ant_options")
                        .default_open(true)
                        .show(ui, |ui| {
                            ui.add_space(10.0);

                            ui.label("Ant algorithm");
                            ui.separator();

                            self.ant_options_sliders(ui);

                            if ui.button("Calculate").clicked() {
                                self.reset_graph();
                                self.iteration_i = 0;
                                self.ant_i = 0;
                                self.edge_i = 0;
                                let res = ant_algo::ant_algo(
                                    &mut self.g,
                                    self.ant_options.iterations_amount,
                                    self.ant_options.ant_amount,
                                    self.ant_options.alpha,
                                    self.ant_options.beta,
                                    self.ant_options.q,
                                    self.ant_options.p,
                                );
                                let best_iteration = res
                                    .iter()
                                    .min_by(|x, y| {
                                        x.ants[x.best_ant_i as usize]
                                            .edges
                                            .iter()
                                            .map(|x| x.edge_info.distance)
                                            .sum::<f32>()
                                            .total_cmp(
                                                &y.ants[y.best_ant_i as usize]
                                                    .edges
                                                    .iter()
                                                    .map(|x| x.edge_info.distance)
                                                    .sum::<f32>(),
                                            )
                                    })
                                    .unwrap()
                                    .clone();
                                let solution = SolutionInfo {
                                    solution: res,
                                    best_iteration: best_iteration.index,
                                    best_ant: best_iteration.best_ant_i,
                                };
                                self.solution = Some(solution);
                                self.update_graph();
                            }
                            match &self.solution {
                                Some(solution) => {
                                    let v = &solution.solution;
                                    let iteration_before = self.iteration_i;
                                    ui.add(
                                        Slider::new(
                                            &mut self.iteration_i,
                                            0..=(v.len() - 1) as i64,
                                        )
                                        .text("Iteration"),
                                    );
                                    let ant_before = self.ant_i;
                                    ui.add(
                                        Slider::new(
                                            &mut self.ant_i,
                                            0..=(v.first().unwrap().ants.len() - 1) as i64,
                                        )
                                        .text("Ant"),
                                    );
                                    let edge_before = self.edge_i;
                                    ui.add(
                                        Slider::new(
                                            &mut self.edge_i,
                                            0..=(v
                                                .first()
                                                .unwrap()
                                                .ants
                                                .first()
                                                .unwrap()
                                                .edges
                                                .len()
                                                - 1)
                                                as i64,
                                        )
                                        .text("Edge"),
                                    );
                                    ui.label(format!(
                                        "Best path: Iteration#{} Ant#{} / {}",
                                        solution.best_iteration,
                                        v[solution.best_iteration].best_ant_i,
                                        v[solution.best_iteration].best_path_len
                                    ));
                                    ui.label(format!(
                                        "Best path for iteration: Ant#{} / {}",
                                        v[self.iteration_i as usize].best_ant_i,
                                        v[self.iteration_i as usize].best_path_len
                                    ));
                                    ui.label(format!(
                                        "Current Ant path: {}",
                                        v[self.iteration_i as usize].ants[self.ant_i as usize]
                                            .distance
                                    ));

                                    if ui.button("Show best path").clicked() {
                                        self.iteration_i = solution.best_iteration as i64;
                                        self.ant_i = solution.best_ant;
                                        self.edge_i =
                                            (v.first().unwrap().ants.first().unwrap().edges.len()
                                                - 1)
                                                as i64;
                                        self.reset_graph_color();
                                        self.update_graph();
                                    } else {
                                        if self.iteration_i != iteration_before
                                            || self.ant_i != ant_before
                                            || self.edge_i != edge_before
                                        {
                                            self.reset_graph_color();
                                            if self.iteration_i - iteration_before != 0 {
                                                self.ant_i = 0;
                                                self.edge_i = 0;
                                            } else if self.ant_i - ant_before != 0 {
                                                self.edge_i = 0;
                                            }
                                            self.update_graph();
                                        }
                                    }
                                    if ui
                                        .checkbox(&mut self.show_pheromones, "Show pheromones")
                                        .changed()
                                    {
                                        self.reset_graph_color();
                                        self.update_graph();
                                    }
                                    let old_pheromones_k = self.pheromones_k;
                                    ui.add_enabled(
                                        self.show_pheromones,
                                        Slider::new(&mut self.pheromones_k, 0. ..=1.)
                                            .text("Pheromones visiability"),
                                    );
                                    if old_pheromones_k != self.pheromones_k {
                                        if self.show_pheromones {
                                            self.reset_graph_color();
                                            self.update_graph();
                                        }
                                    }
                                }
                                None => (),
                            }
                        });
                    CollapsingHeader::new("Ui")
                        .default_open(true)
                        .show(ui, |ui| {
                            ui.add_space(10.0);

                            ui.label("Ui settings");
                            ui.separator();

                            self.ui_settings(ui);
                        });
                });
            });
        egui::CentralPanel::default().show(ctx, |ui| {
            let settings_navigation = &egui_graphs::SettingsNavigation::new()
                .with_fit_to_screen_enabled(self.settings_navigation.fit_to_screen_enabled)
                .with_screen_padding(self.settings_navigation.screen_padding)
                .with_zoom_and_pan_enabled(self.settings_navigation.zoom_and_pan_enabled)
                .with_zoom_speed(self.settings_navigation.zoom_speed);
            let settings_style = &egui_graphs::SettingsStyle::new()
                .with_labels_always(self.settings_style.labels_always)
                .with_edge_radius_weight(self.settings_style.edge_radius_weight)
                .with_folded_radius_weight(self.settings_style.folded_node_radius_weight);
            ui.add(
                &mut GraphView::new(&mut self.g)
                    .with_styles(settings_style)
                    .with_navigations(settings_navigation),
            );
        });
    }
}

fn main() {
    let native_options = eframe::NativeOptions::default();
    run_native(
        "Ant-algo",
        native_options,
        Box::new(|cc| Box::new(AntApp::new(cc))),
    )
    .unwrap();
}
