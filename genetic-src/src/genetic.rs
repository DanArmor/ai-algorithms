use std::collections::HashMap;
use std::sync::Arc;

use petgraph::stable_graph::NodeIndex;
use rand::{seq::SliceRandom, Rng};

use weighted_rand::{builder::NewBuilder, table::WalkerTable};

#[derive(Debug, Clone, PartialEq)]
pub struct TSPParentInfo {
    pub chromosome_index: usize,
    pub population_index: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TSPChromosomeType {
    Crossover(TSPParentInfo, TSPParentInfo),
    Mutation(TSPParentInfo),
    NoHistory,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TSPChromosome {
    pub index: usize,
    pub population: usize,
    pub travel_list: Vec<NodeIndex>,
    pub path_length: f32,
    pub edges_dist: Arc<HashMap<NodeIndex, HashMap<NodeIndex, f32>>>,
    pub chromosome_type: TSPChromosomeType,
}

fn distance(
    travel_list: &Vec<NodeIndex>,
    edges_dist: Arc<HashMap<NodeIndex, HashMap<NodeIndex, f32>>>,
) -> f32 {
    let mut sum = 0.0;
    for i in 0..travel_list.len() - 1 {
        sum += edges_dist[&travel_list[i]][&travel_list[i + 1]];
    }
    sum += edges_dist[&travel_list[travel_list.len() - 1]][&travel_list[0]];
    sum
}

impl TSPChromosome {
    pub fn new(
        index: usize,
        population: usize,
        travel_list: Vec<NodeIndex>,
        edges_dist: Arc<HashMap<NodeIndex, HashMap<NodeIndex, f32>>>,
    ) -> Self {
        let path_length = distance(&travel_list, edges_dist.clone());
        Self {
            index: index,
            population: population,
            travel_list: travel_list,
            path_length: path_length,
            edges_dist: edges_dist,
            chromosome_type: TSPChromosomeType::NoHistory,
        }
    }
    pub fn generate_random_population(
        indices: Vec<NodeIndex>,
        population_size: usize,
        edges_dist: Arc<HashMap<NodeIndex, HashMap<NodeIndex, f32>>>,
    ) -> Vec<Self> {
        let mut travel_list = indices;
        (0..population_size)
            .into_iter()
            .map(|index| {
                travel_list.shuffle(&mut rand::thread_rng());
                TSPChromosome::new(index, 0, travel_list.clone(), edges_dist.clone())
            })
            .collect::<Vec<_>>()
    }
}

#[derive(Debug, Clone)]
pub struct TSPIteration {
    pub old: Vec<TSPChromosome>,
    pub new: Vec<TSPChromosome>,
    pub best_chromosome_i: usize,
}

impl GeneticIteration<TSPChromosome> for TSPIteration {
    fn new_iter(old: Vec<TSPChromosome>, new: Vec<TSPChromosome>) -> Self {
        Self {
            old: old.clone(),
            new: new,
            best_chromosome_i: old
                .into_iter()
                .map(|x| x.path_length)
                .enumerate()
                .min_by(|x, y| x.1.total_cmp(&y.1))
                .unwrap()
                .0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TSPSolution {
    pub iterations: Vec<TSPIteration>,
    pub best_population_i: usize,
    pub best_chromosome_i: usize,
}

impl Solution<TSPChromosome, TSPIteration> for TSPSolution {
    fn new_solution() -> Self {
        Self {
            iterations: vec![],
            best_population_i: 0,
            best_chromosome_i: 0,
        }
    }
    fn add_iteration(&mut self, iteration: TSPIteration) {
        let dist = iteration.old[iteration.best_chromosome_i].path_length;
        let best = iteration.best_chromosome_i;
        self.iterations.push(iteration);
        if self.iterations[self.best_population_i].old[self.best_chromosome_i].path_length > dist {
            self.best_population_i = self.iterations.len() - 1;
            self.best_chromosome_i = best;
        }
    }
}

pub trait GeneticIteration<T: Chromosome> {
    fn new_iter(old: Vec<T>, new: Vec<T>) -> Self;
}

pub trait Solution<ChromosomeType: Chromosome, T: GeneticIteration<ChromosomeType>>: Clone {
    fn new_solution() -> Self;
    fn add_iteration(&mut self, iteration: T);
}

pub trait Chromosome: Clone + PartialEq {
    fn mutate(&self) -> Self;
    fn crossover(&self, other: &Self) -> (Self, Self);
    fn health(&self) -> f32;
    fn set_index(&mut self, index: usize);
    fn get_index(&self) -> usize;
    fn set_population(&mut self, population: usize);
    fn get_population(&mut self) -> usize;
}

impl Chromosome for TSPChromosome {
    fn mutate(&self) -> Self {
        let mut mutant = self.clone();
        let first = rand::thread_rng().gen_range(0..mutant.travel_list.len() - 1);
        let second = rand::thread_rng().gen_range(first..mutant.travel_list.len());
        mutant.travel_list[first..=second].shuffle(&mut rand::thread_rng());
        mutant.chromosome_type = TSPChromosomeType::Mutation(TSPParentInfo {
            chromosome_index: self.index,
            population_index: self.population,
        });
        mutant.path_length = mutant.health();
        mutant
    }
    fn crossover(&self, other: &Self) -> (Self, Self) {
        let first = rand::thread_rng().gen_range(0..self.travel_list.len() - 1);
        let second = rand::thread_rng().gen_range(first..self.travel_list.len());
        let mut offspring_1 = other.clone();
        for i in first..=second {
            let index_in_offspring_1 = offspring_1
                .travel_list
                .iter()
                .position(|x| *x == self.travel_list[i])
                .unwrap();
            offspring_1.travel_list.remove(index_in_offspring_1);
        }
        offspring_1.chromosome_type = TSPChromosomeType::Crossover(
            TSPParentInfo {
                chromosome_index: self.index,
                population_index: self.population,
            },
            TSPParentInfo {
                chromosome_index: other.index,
                population_index: other.population,
            },
        );
        let mut offspring_2 = offspring_1.clone();

        let mut shuffeled_gen = self.travel_list[first..=second].to_vec();
        shuffeled_gen.shuffle(&mut rand::thread_rng());
        offspring_1.travel_list.append(&mut shuffeled_gen);

        shuffeled_gen = self.travel_list[first..=second].to_vec();
        shuffeled_gen.shuffle(&mut rand::thread_rng());
        offspring_2.travel_list.append(&mut shuffeled_gen);

        offspring_1.path_length = offspring_1.health();
        offspring_2.path_length = offspring_2.health();

        (offspring_1, offspring_2)
    }
    fn health(&self) -> f32 {
        distance(&self.travel_list, self.edges_dist.clone())
    }
    fn set_index(&mut self, index: usize) {
        self.index = index;
    }
    fn get_index(&self) -> usize {
        self.index
    }
    fn set_population(&mut self, population: usize) {
        self.population = population;
    }
    fn get_population(&mut self) -> usize {
        self.population
    }
}

#[derive(Debug, Clone, Copy)]
pub struct EdgeInfo {
    pub distance: f32,
}

pub fn get_two_indexs(table: &WalkerTable) -> (usize, usize) {
    let first = table.next_rng(&mut rand::thread_rng());
    let mut second = table.next_rng(&mut rand::thread_rng());
    while first == second {
        second = table.next_rng(&mut rand::thread_rng());
    }
    (first, second)
}

pub fn pick_for_population<ChromosomeType: Chromosome>(
    population: &Vec<ChromosomeType>,
    mutants: &Vec<ChromosomeType>,
    table: &WalkerTable,
) -> usize {
    let mut index = table.next_rng(&mut rand::thread_rng());
    if population.contains(&mutants[index]) {
        index = 0;
        while population.contains(&mutants[index]) {
            index += 1;
        }
    }
    index
}

pub fn solve<
    ChromosomeType: Chromosome,
    IterationType: GeneticIteration<ChromosomeType>,
    SolutionType: Solution<ChromosomeType, IterationType>,
>(
    population_amount: usize,
    mut population: Vec<ChromosomeType>,
    crossover_p: f32,
    mutation_p: f32,
) -> SolutionType {
    let mut solution = SolutionType::new_solution();
    let wa_table =
        weighted_rand::builder::WalkerTableBuilder::new(&[crossover_p, mutation_p]).build();

    let population_size = population.len();
    for population_i in 0..population_amount {
        let mut old = population.clone();
        for i in 0..old.len() {
            old[i].set_population(population_i);
            old[i].set_index(i);
        }
        let old_wa_table = weighted_rand::builder::WalkerTableBuilder::new(
            &old.iter().map(|x| x.health() as u32).collect::<Vec<_>>(),
        )
        .inverse()
        .build();
        let mut new: Vec<ChromosomeType> = vec![];
        while new.len() != old.len() {
            let v = wa_table.next_rng(&mut rand::thread_rng());
            match v {
                0 if old.len().abs_diff(new.len()) >= 2 => {
                    let (first, second) = get_two_indexs(&old_wa_table);
                    let (offspring_1, offspring_2) = old[first].crossover(&old[second]);
                    new.push(offspring_1);
                    new.push(offspring_2);
                }
                _ => {
                    new.push(old[old_wa_table.next_rng(&mut rand::thread_rng())].mutate());
                }
            }
        }
        for i in 0..new.len() {
            new[i].set_index(i);
            new[i].set_population(population_i);
        }
        solution.add_iteration(IterationType::new_iter(old.clone(), new.clone()));
        new.append(&mut old);
        let new_wa_table = weighted_rand::builder::WalkerTableBuilder::new(
            &new.iter().map(|x| x.health() as u32).collect::<Vec<_>>(),
        )
        .inverse()
        .build();
        population.clear();
        for i in 0..population_size {
            population.push(new[pick_for_population(&population, &new, &new_wa_table)].clone());
        }
    }
    solution
}
