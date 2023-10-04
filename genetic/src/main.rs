use crossbeam::channel::{unbounded, Receiver, Sender};
use eframe::{run_native, App, CreationContext};
use egui::{CollapsingHeader, Color32, Context, ScrollArea, Slider, Ui, Vec2};
use egui_graphs::{Change, ChangeNode, Edge, Graph, GraphView, Node, SettingsInteraction};
use petgraph::{
    stable_graph::{NodeIndex, StableUnGraph},
    visit::EdgeRef,
    Undirected,
};
use rand::Rng;
use std::collections::HashMap;
use std::sync::Arc;

mod genetic;
mod settings;

use genetic::*;

pub struct GeneticOptions {
    nodes_amount: usize,
    crossover_p: f32,
    mutation_p: f32,
    population_amount: usize,
    population_size: usize,
}

impl Default for GeneticOptions {
    fn default() -> Self {
        GeneticOptions {
            nodes_amount: 3,
            crossover_p: 0.5,
            mutation_p: 0.5,
            population_amount: 3,
            population_size: 3,
        }
    }
}

pub struct GeneticApp {
    g: Graph<(), EdgeInfo, Undirected>,
    genetic_options: GeneticOptions,
    settings_style: settings::SettingsStyle,
    settings_navigation: settings::SettingsNavigation,
    solution: Option<TSPSolution>,
    population_i: usize,
    chromosome_i: usize,
    pheromones_k: f32,
    drag_enabled: bool,
    changes_receiver: Receiver<Change>,
    changes_sender: Sender<Change>,
    show_parents: bool,
}

fn distance(a: Vec2, b: Vec2) -> f32 {
    ((a.x - b.x).powi(2) + (a.y - b.y).powi(2)).sqrt()
}

impl GeneticApp {
    fn new(_: &CreationContext<'_>) -> Self {
        let (changes_sender, changes_receiver) = unbounded();
        let mut app = Self {
            g: Graph::from(&StableUnGraph::default()),
            genetic_options: GeneticOptions::default(),
            settings_style: settings::SettingsStyle::default(),
            settings_navigation: settings::SettingsNavigation::default(),
            solution: None,
            population_i: 0,
            chromosome_i: 0,
            pheromones_k: 1.0,
            drag_enabled: false,
            changes_receiver: changes_receiver,
            changes_sender: changes_sender,
            show_parents: false,
        };
        for _ in 0..app.genetic_options.nodes_amount {
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
                let mut edge_data = EdgeInfo {
                    distance: distance(
                        self.g.node(x).unwrap().location(),
                        self.g.node(node).unwrap().location(),
                    ),
                };
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
    fn genetic_options_sliders(&mut self, ui: &mut Ui) {
        let nodes_before = self.genetic_options.nodes_amount;
        ui.add(Slider::new(&mut self.genetic_options.nodes_amount, 3..=200).text("Nodes"));
        let delta_nodes = self.genetic_options.nodes_amount.abs_diff(nodes_before);
        if delta_nodes != 0 {
            self.reset_graph();
        }
        (0..delta_nodes).for_each(|_| {
            if self.genetic_options.nodes_amount > nodes_before {
                self.add_random_node();
            } else {
                self.remove_random_node();
            }
        });
        ui.add(Slider::new(&mut self.genetic_options.crossover_p, 0. ..=1.).text("Crossover%"));
        ui.add(
            Slider::new(&mut self.genetic_options.population_amount, 1..=10000)
                .text("Population amount"),
        );
        ui.add(
            Slider::new(&mut self.genetic_options.population_size, 2..=512).text("Population size"),
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
        ui.checkbox(&mut self.drag_enabled, "Drag enabled");
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
        self.reset_graph_color();
    }
    fn update_graph(&mut self) {
        match &self.solution {
            Some(v) => {
                let v = &v.iterations;
                let iteration = &v[self.population_i];
                let chromosome = &index_to_chromosome(iteration, self.chromosome_i);

                for i in 0..self.g.g.node_count() {
                    let node = self.g.g.node_weight_mut(NodeIndex::from(i as u32)).unwrap();
                    node.clone_from(&node.clone().with_color(Color32::from_rgb(85, 24, 93)));
                }
                for i in 0..self.g.g.node_count() - 1 {
                    let e = self
                        .g
                        .g
                        .edges_connecting(chromosome.travel_list[i], chromosome.travel_list[i + 1])
                        .map(|x| x.id())
                        .collect::<Vec<_>>()[0];
                    let e = self.g.g.edge_weight_mut(e).unwrap();
                    e.clone_from(
                        &e.clone()
                            .with_color(Color32::from_rgba_unmultiplied(255, 213, 36, 128)),
                    );
                }
                let e = self
                    .g
                    .g
                    .edges_connecting(
                        chromosome.travel_list[chromosome.travel_list.len() - 1],
                        chromosome.travel_list[0],
                    )
                    .map(|x| x.id())
                    .collect::<Vec<_>>()[0];
                let e = self.g.g.edge_weight_mut(e).unwrap();
                e.clone_from(
                    &e.clone()
                        .with_color(Color32::from_rgba_unmultiplied(255, 213, 36, 128)),
                );
            }
            None => (),
        }
    }
    fn handle_changes(&mut self) {
        let mut node_id: Option<NodeIndex> = None;
        self.changes_receiver.try_iter().for_each(|ch| {
            if let Change::Node(ChangeNode::Location { id, old, new }) = ch.clone() {
                node_id = Some(id);
            }
        });
        match node_id {
            Some(id) => {
                self.reset_graph();
                let neighbors = self.g.g.neighbors_undirected(id).collect::<Vec<_>>();
                neighbors.iter().for_each(|n| {
                    self.remove_edges(id, *n);
                });
                self.connect_node(id)
            }
            None => (),
        }
    }
    fn build_edges_dist(&self) -> Arc<HashMap<NodeIndex, HashMap<NodeIndex, f32>>> {
        let mut dict: HashMap<NodeIndex, HashMap<NodeIndex, f32>> = HashMap::default();
        for i in 0..self.g.g.node_count() {
            dict.insert(NodeIndex::from(i as u32), HashMap::default());
        }
        for i in 0..self.g.g.node_count() {
            for j in i + 1..self.g.g.node_count() {
                let e = self
                    .g
                    .g
                    .edges_connecting(NodeIndex::from(i as u32), NodeIndex::from(j as u32))
                    .map(|x| x.id())
                    .collect::<Vec<_>>()[0];
                let e = self.g.g.edge_weight(e).unwrap();
                let distance = e.data().unwrap().distance;
                dict.get_mut(&NodeIndex::from(i as u32))
                    .unwrap()
                    .insert(NodeIndex::from(j as u32), distance);
                dict.get_mut(&NodeIndex::from(j as u32))
                    .unwrap()
                    .insert(NodeIndex::from(i as u32), distance);
            }
        }
        Arc::new(dict)
    }
}

fn index_to_chromosome(iteration: &TSPIteration, index: usize) -> TSPChromosome {
    if index < iteration.old.len() {
        iteration.old[index].clone()
    } else {
        iteration.new[index % iteration.old.len()].clone()
    }
}

impl App for GeneticApp {
    fn update(&mut self, ctx: &Context, _: &mut eframe::Frame) {
        egui::SidePanel::right("right_panel")
            .min_width(250.)
            .show(ctx, |ui| {
                ScrollArea::vertical().show(ui, |ui| {
                    CollapsingHeader::new("Genetic options")
                        .default_open(true)
                        .show(ui, |ui| {
                            ui.add_space(10.0);

                            ui.label("Genetic algorithm");
                            ui.separator();

                            self.genetic_options_sliders(ui);

                            if ui.button("Calculate").clicked() {
                                self.reset_graph();
                                self.population_i = 0;
                                self.chromosome_i = 0;
                                self.genetic_options.mutation_p =
                                    1.0 - self.genetic_options.crossover_p;
                                let res: TSPSolution = solve(
                                    self.genetic_options.population_amount,
                                    TSPChromosome::generate_random_population(
                                        self.genetic_options.nodes_amount,
                                        self.build_edges_dist(),
                                    ),
                                    self.genetic_options.crossover_p,
                                    self.genetic_options.mutation_p,
                                );
                                self.solution = Some(res);
                                self.update_graph();
                            }
                            match &self.solution {
                                Some(solution) => {
                                    let v = &solution.iterations;
                                    let iteration_before = self.population_i;
                                    ui.add(
                                        Slider::new(&mut self.population_i, 0..=(v.len() - 1))
                                            .text("Population(Iteration)"),
                                    );
                                    let chromosome_before = self.chromosome_i;
                                    ui.add(
                                        Slider::new(
                                            &mut self.chromosome_i,
                                            0..=(v.first().unwrap().old.len() * 2 - 1),
                                        )
                                        .text("Chromosome"),
                                    );

                                    ui.label(format!(
                                        "Best path: Population#{} Chromosome#{} / {}",
                                        solution.best_population_i,
                                        solution.best_chromosome_i,
                                        v[solution.best_population_i].old
                                            [solution.best_chromosome_i]
                                            .path_length,
                                    ));
                                    ui.label(format!(
                                        "Best path for Population: Chromosome#{} / {}",
                                        v[self.population_i].best_chromosome_i,
                                        index_to_chromosome(
                                            &v[self.population_i],
                                            v[self.population_i].best_chromosome_i
                                        )
                                        .path_length
                                    ));
                                    ui.label(format!(
                                        "Current Chromosome path: {}",
                                        index_to_chromosome(
                                            &v[self.population_i],
                                            self.chromosome_i
                                        )
                                        .path_length
                                    ));

                                    if ui.button("Show best path").clicked() {
                                        self.population_i = solution.best_population_i;
                                        self.chromosome_i = solution.best_chromosome_i;
                                        self.reset_graph_color();
                                        self.update_graph();
                                    } else {
                                        if self.population_i != iteration_before
                                            || self.chromosome_i != chromosome_before
                                        {
                                            self.reset_graph_color();
                                            if self.population_i.abs_diff(iteration_before) != 0 {
                                                self.chromosome_i = 0;
                                            }
                                            self.update_graph();
                                        }
                                    }
                                    if ui
                                        .checkbox(&mut self.show_parents, "Show parents")
                                        .changed()
                                    {
                                        self.reset_graph_color();
                                        self.update_graph();
                                    }
                                    let old_pheromones_k = self.pheromones_k;
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
                    .with_navigations(settings_navigation)
                    .with_interactions(
                        &SettingsInteraction::new().with_dragging_enabled(self.drag_enabled),
                    )
                    .with_changes(&self.changes_sender),
            );
        });
        self.handle_changes();
    }
}

fn main() {
    let native_options = eframe::NativeOptions::default();
    run_native(
        "Ant-algo",
        native_options,
        Box::new(|cc| Box::new(GeneticApp::new(cc))),
    )
    .unwrap();
}
