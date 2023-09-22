use egui_graphs::Graph;
use petgraph::stable_graph::NodeIndex;

struct Ant {
    tabu: Vec<usize>,
}

impl Ant {
    fn new() -> Self {
        Self {
            tabu: vec![]
        }
    }
}

fn ant_algo(g: Graph<(), (), petgraph::Undirected>, start_index: NodeIndex) {
    let start_node = g.node(start_index).unwrap();
    let graph_size = g.nodes_iter().count();
    let mut ant = Ant::new();
    while ant.tabu.len() != graph_size - 1 {
        
    }
    g.edges_directed(start_index, petgraph::Direction::Outgoing);
}