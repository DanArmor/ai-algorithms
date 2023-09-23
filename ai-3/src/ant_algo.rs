use eframe::{run_native, App, CreationContext};
use egui::{CollapsingHeader, Color32, Context, ScrollArea, Slider, Ui, Vec2};
use egui_graphs::{Edge, Graph, GraphView, Node};
use petgraph::{
    stable_graph::{NodeIndex, StableGraph, StableUnGraph},
    visit::EdgeRef,
    Undirected,
};
use weighted_rand::builder::NewBuilder;
#[derive(Debug)]
struct Ant {
    tabu: Vec<NodeIndex>,
    current_node: NodeIndex,
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

struct EdgeAnt {
    source: NodeIndex,
    target: NodeIndex,
    probability: f32,
    edge_info: EdgeInfo,
}

impl Ant {
    fn new(start_index: NodeIndex) -> Self {
        Self {
            tabu: vec![],
            current_node: start_index,
        }
    }
    fn add_tabu_node(&mut self, node_index: NodeIndex) {
        self.tabu.push(node_index);
    }
    fn get_edge_ant(&mut self, g: &Graph<(), EdgeInfo, petgraph::Undirected>) -> Vec<EdgeAnt> {
        let mut edges =
            g.g.edges_directed(self.current_node, petgraph::Outgoing)
                .filter(|x|{
                    !self.tabu.contains(&x.target())
                })
                .map(|x| EdgeAnt {
                    edge_info: x.weight().data().unwrap().clone(),
                    source: x.source(),
                    target: x.target(),
                    probability: 0.0,
                })
                .collect::<Vec<_>>();
        let sum: f32 = edges
            .iter()
            .map(|x| x.edge_info.probability_parameters)
            .sum();
        edges
            .iter_mut()
            .for_each(|x| x.probability = x.edge_info.probability_parameters / sum);
        edges
    }
    fn travel_graph(&mut self, g: &Graph<(), EdgeInfo, petgraph::Undirected>) {
        let graph_size = g.g.node_count();
        while self.tabu.len() != graph_size-1 {
            let edges = self.get_edge_ant(g);
            let indexes_weights = edges.iter().map(|x| {
                if x.probability.is_nan() {
                    0.0
                } else {
                    x.probability
                }
            }).collect::<Vec<_>>();
            print!("T: {:?}", indexes_weights);
            let wa_table =
                weighted_rand::builder::WalkerTableBuilder::new(&indexes_weights).build();
            let next_node = edges[wa_table.next()].target;
            self.add_tabu_node(self.current_node.clone());
            self.current_node = next_node;
        }
        self.add_tabu_node(self.current_node.clone());
    }
}

pub fn ant_algo(g: &Graph<(), EdgeInfo, petgraph::Undirected>, start_index: NodeIndex) {
    let mut ant = Ant::new(start_index);
    ant.travel_graph(g);
    print!("Data: {:?}", ant);
}
