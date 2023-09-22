use eframe::{run_native, App, CreationContext};
use egui::{CollapsingHeader, Color32, Context, ScrollArea, Slider, Ui, Vec2};
use egui_graphs::{Edge, Graph, GraphView, Node};
use petgraph::{
    stable_graph::{NodeIndex, StableGraph, StableUnGraph},
    visit::EdgeRef,
    Undirected,
};
use rand::Rng;

mod settings;

pub struct AntOptions {
    alpha: f64,
    nodes: i64,
}

impl Default for AntOptions {
    fn default() -> Self {
        AntOptions {
            alpha: 0.5,
            nodes: 3,
        }
    }
}

pub struct AntApp {
    g: Graph<(), (), Undirected>,
    ant_options: AntOptions,
    settings_style: settings::SettingsStyle,
    settings_navigation: settings::SettingsNavigation,
}

impl AntApp {
    fn new(_: &CreationContext<'_>) -> Self {
        let g = generate_graph();
        Self {
            g: Graph::from(&g),
            ant_options: AntOptions::default(),
            settings_style: settings::SettingsStyle::default(),
            settings_navigation: settings::SettingsNavigation::default()
        }
    }
}

impl AntApp {
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
                self.g.g.add_edge(
                    x,
                    node,
                    Edge::new(()).with_color(Color32::from_rgba_unmultiplied(128, 128, 128, 32)),
                );
            }
        });
    }
    fn remove_node(&mut self, idx: NodeIndex) {
        let neighbors = self.g.g.neighbors_undirected(idx).collect::<Vec<_>>();
        neighbors.iter().for_each(|n| {
            self.remove_edges(idx, *n);
            self.remove_edges(*n, idx);
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
        let n = self.g.g.node_weight_mut(idx).unwrap();
        *n = n.with_label(format!("{:?}", idx));
    }
    fn ant_options_sliders(&mut self, ui: &mut Ui) {
        let nodes_before = self.ant_options.nodes;
        ui.add(Slider::new(&mut self.ant_options.nodes, 3..=200).text("Nodes"));
        let delta_nodes = self.ant_options.nodes - nodes_before;
        (0..delta_nodes.abs()).for_each(|_| {
            if delta_nodes > 0 {
                self.add_random_node();
            } else {
                self.remove_random_node();
            }
        });

        ui.add(Slider::new(&mut self.ant_options.alpha, 0. ..=1.).text("alpha"));
    }
    fn ui_settings(&mut self, ui: &mut Ui) {
        if ui.checkbox(&mut self.settings_navigation.fit_to_screen_enabled, "Fit to screen")
        .changed() {
            self.settings_navigation.zoom_and_pan_enabled = !self.settings_navigation.zoom_and_pan_enabled;
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

fn generate_graph() -> StableGraph<(), (), Undirected> {
    let mut g = StableUnGraph::default();

    let a = g.add_node(());
    let b = g.add_node(());
    let c = g.add_node(());

    g.add_edge(a, b, ());
    g.add_edge(b, c, ());
    g.add_edge(c, a, ());

    g
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
