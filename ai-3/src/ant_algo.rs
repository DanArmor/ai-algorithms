use eframe::{run_native, App, CreationContext};
use egui::{CollapsingHeader, Color32, Context, ScrollArea, Slider, Ui, Vec2};
use egui_graphs::{Edge, Graph, GraphView, Node};
use petgraph::{
    stable_graph::{EdgeIndex, EdgeReference, NodeIndex, StableGraph, StableUnGraph},
    visit::EdgeRef,
    Undirected,
};
use rand::Rng;
use weighted_rand::builder::NewBuilder;

#[derive(Debug, Clone)]
pub struct Ant {
    tabu: Vec<NodeIndex>,
    pub edges: Vec<EdgeAnt>,
    current_node: NodeIndex,
    pub ant_index: i64,
    pub iteration_index: i64,
    pub distance: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct EdgeInfo {
    pub distance: f32,
    pub pheromones: f32,
    pub probability_parameters: f32,
}

impl EdgeInfo {
    pub fn recalculate(&mut self, alpha: f32, beta: f32) {
        self.probability_parameters = self.pheromones.powf(alpha) * self.distance.powf(beta);
    }
}

#[derive(Debug, Clone, Copy)]
pub struct EdgeAnt {
    pub index: EdgeIndex,
    pub source: NodeIndex,
    pub target: NodeIndex,
    pub probability: f32,
    pub edge_info: EdgeInfo,
}

fn edge_to_edge_ant(edge: &EdgeReference<'_, Edge<EdgeInfo>>) -> EdgeAnt {
    EdgeAnt {
        index: edge.id(),
        edge_info: edge.weight().data().unwrap().clone(),
        source: edge.source(),
        target: edge.target(),
        probability: 1.0,
    }
}

impl Ant {
    fn new(start_index: NodeIndex, ant_index: i64, iteration_index: i64) -> Self {
        Self {
            tabu: vec![],
            current_node: start_index,
            edges: vec![],
            ant_index: ant_index,
            iteration_index: iteration_index,
            distance: 0.0,
        }
    }
    fn add_tabu_node(&mut self, node_index: NodeIndex) {
        self.tabu.push(node_index);
    }
    fn add_edge(&mut self, edge: EdgeAnt) {
        self.edges.push(edge)
    }
    fn get_edge_ant(&mut self, g: &Graph<(), EdgeInfo, petgraph::Undirected>) -> Vec<EdgeAnt> {
        let mut edges =
            g.g.edges_directed(self.current_node, petgraph::Outgoing)
                .filter(|x| !self.tabu.contains(&x.target()))
                .map(|x| edge_to_edge_ant(&x))
                .collect::<Vec<_>>();
        let sum: f32 = edges
            .iter()
            .map(|x| x.edge_info.probability_parameters)
            .sum();
        if sum == 0.0 {
            let edges_amount = edges.len() as f32;
            edges
                .iter_mut()
                .for_each(|x| x.probability = 1.0 / edges_amount);
        } else {
            edges
                .iter_mut()
                .for_each(|x| x.probability = x.edge_info.probability_parameters / sum);
        }
        edges
    }
    fn travel_graph(&mut self, g: &Graph<(), EdgeInfo, petgraph::Undirected>) {
        let graph_size = g.g.node_count();
        while self.tabu.len() != graph_size - 1 {
            let edges = self.get_edge_ant(g);
            let indexes_weights = edges
                .iter()
                .map(|x| {
                    if x.probability.is_nan() {
                        0.0
                    } else {
                        x.probability
                    }
                })
                .collect::<Vec<_>>();
            let wa_table =
                weighted_rand::builder::WalkerTableBuilder::new(&indexes_weights).build();
            let next_edge = edges[wa_table.next()];
            let next_node = next_edge.target;
            self.add_tabu_node(self.current_node.clone());
            self.add_edge(next_edge);
            self.current_node = next_node;
        }
        self.add_tabu_node(self.current_node.clone());
        let final_edge =
            g.g.edges_connecting(self.current_node, self.tabu[0])
                .collect::<Vec<_>>()[0];
        self.add_edge(edge_to_edge_ant(&final_edge));
    }
}

fn random_node_idx(g: &Graph<(), EdgeInfo, petgraph::Undirected>) -> Option<NodeIndex> {
    let nodes_cnt = g.g.node_count();
    if nodes_cnt == 0 {
        return None;
    }

    let random_n_idx = rand::thread_rng().gen_range(0..nodes_cnt);
    g.g.node_indices().nth(random_n_idx)
}

fn update_edges(
    ants: &Vec<Ant>,
    g: &mut Graph<(), EdgeInfo, petgraph::Undirected>,
    alpha: f32,
    beta: f32,
    q: f32,
    p: f32,
) {
    for ant in ants {
        let pheromones: f32 = q / ant.edges.iter().map(|x| x.edge_info.distance).sum::<f32>();
        for edge in &ant.edges {
            let mut new_edge_data = edge.edge_info.clone();
            new_edge_data.pheromones = new_edge_data.pheromones * p + pheromones;
            new_edge_data.recalculate(alpha, beta);

            let edge_to_change = g.g.edge_weight_mut(edge.index).unwrap();
            edge_to_change.clone_from(
                &Edge::new(new_edge_data)
                    .with_color(Color32::from_rgba_unmultiplied(128, 128, 128, 0)),
            );
        }
    }
}

#[derive(Debug, Clone)]
pub struct IterationInfo {
    pub index: usize,
    pub old_edges: Vec<EdgeInfo>,
    pub ants: Vec<Ant>,
    pub best_ant_i: i64,
    pub best_path_len: f32,
}

pub fn ant_algo(
    g: &mut Graph<(), EdgeInfo, petgraph::Undirected>,
    iterations_amount: i64,
    ant_amount: i64,
    alpha: f32,
    beta: f32,
    q: f32,
    p: f32,
) -> Vec<IterationInfo> {
    let mut iterations: Vec<IterationInfo> = vec![];
    for iteration_i in 0..iterations_amount {
        let old_edges = g
            .edges_iter()
            .map(|x| x.1.data().unwrap().clone())
            .collect::<Vec<_>>();
        let mut ants = vec![];
        for ant_i in 0..ant_amount {
            let mut ant = Ant::new(random_node_idx(g).unwrap(), ant_i, iteration_i);
            ant.travel_graph(g);
            ant.distance = ant.edges.iter().map(|x| x.edge_info.distance).sum::<f32>();
            ants.push(ant);
        }
        update_edges(&ants, g, alpha, beta, q, p);
        let best_ant_i = ants
            .iter()
            .enumerate()
            .min_by(|x, y| x.1.distance.total_cmp(&y.1.distance))
            .unwrap()
            .0;
        let best_path_len = ants[best_ant_i].distance;
        iterations.push(IterationInfo {
            index: iteration_i as usize,
            old_edges: old_edges,
            ants: ants,
            best_ant_i: best_ant_i as i64,
            best_path_len: best_path_len,
        })
    }
    iterations
}
