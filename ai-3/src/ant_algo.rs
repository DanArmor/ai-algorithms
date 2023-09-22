use egui_graphs::Graph;
use petgraph::stable_graph::NodeIndex;

struct Ant {
    tabu: Vec<NodeIndex>,
}

struct EdgeAnt {
    distance: f64,
    source: NodeIndex,
    target: NodeIndex,
    probability: f64
}

impl Ant {
    fn new() -> Self {
        Self { tabu: vec![] }
    }
    fn add_tabu_node(&mut self, node_index: NodeIndex) {
        self.tabu.push(node_index);
    }
    fn travel_graph(&mut self, g: &Graph<(), (), petgraph::Undirected>, start_index: NodeIndex) {
        let graph_size = g.g.node_count();
        while self.tabu.len() != graph_size {
            let edges: Vec<_> = g.edges_directed(start_index, petgraph::Direction::Outgoing)
                .map(|x| {
                    1
                }).collect();
        }
    }
}

struct EdgeInfo {
    pheromones: f64,
}

fn ant_algo(g: &Graph<(), (), petgraph::Undirected>, start_index: NodeIndex) {}
