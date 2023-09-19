use rand::distributions::{Alphanumeric, DistString};

use bit_vec;

#[derive(Debug)]
pub struct Claster {
    pub v: bit_vec::BitVec,
    pub indexes: Vec<usize>,
    pub id: String,
}

impl Claster {
    pub fn new() -> Self {
        Self {
            v: bit_vec::BitVec::default(),
            indexes: Vec::default(),
            id: Alphanumeric.sample_string(&mut rand::thread_rng(), 16),
        }
    }
    fn add(&mut self, entity: &mut ClasterEntity) {
        if self.indexes.is_empty() {
            self.v = entity.v.clone();
        } else {
            self.v.and(&entity.v);
        }
        self.indexes.push(entity.data_index);
        entity.claster_id = self.id.clone();
    }
    pub fn recalculate(&mut self, data: &Vec<bit_vec::BitVec>) {
        self.v.clear();
        match self.indexes.first() {
            Some(i) => self.v = data[*i].clone(),
            None => ()
        }
        for i in &self.indexes {
            self.v.and(&data[*i]);
        }
    }
    pub fn contains(&self, index: &usize) -> bool {
        self.indexes.contains(index)
    }
    pub fn remove(&mut self, data: &Vec<bit_vec::BitVec>, entity: &mut ClasterEntity) {
        self.indexes.remove(
            self.indexes
                .iter()
                .position(|x| x == &entity.data_index)
                .unwrap(),
        );
        self.recalculate(data);
        entity.claster_id = "-".into();
    }
}

fn get_energy(v: &bit_vec::BitVec) -> u32 {
    v.blocks().map(|x| x.count_ones()).sum()
}

fn similar(P: &bit_vec::BitVec, E: &bit_vec::BitVec, b: f64) -> bool {
    let mut intersect = P.clone();
    intersect.and(E);
    get_energy(&intersect) as f64 / (b + get_energy(P) as f64)
        > get_energy(E) as f64 / (b + P.len() as f64)
}

fn pass(P: &bit_vec::BitVec, E: &bit_vec::BitVec, p: f64) -> bool {
    let mut intersect = P.clone();
    intersect.and(E);
    (get_energy(&intersect) as f64 / get_energy(E) as f64) >= p
}

#[derive(Debug)]
struct ClasterEntity {
    v: bit_vec::BitVec,
    data_index: usize,
    claster_id: String,
}

#[derive(Debug)]
struct ClastersManager {
    clasters: Vec<Claster>,
    b: f64,
    p: f64,
}

impl ClastersManager {
    fn add_claster(&mut self, claster: Claster) {
        self.clasters.push(claster);
    }
    fn remove_from_claster(&mut self, data: &Vec<bit_vec::BitVec>, entity: &mut ClasterEntity) {
        match self.clasters.iter().position(|x| x.id == entity.claster_id) {
            Some(index) => self.clasters[index].remove(data, entity),
            None => (),
        }
    }
    fn find_claster(&mut self, entity: &mut ClasterEntity) -> Option<usize> {
        for i in 0..self.clasters.len() {
            if entity.claster_id != self.clasters[i].id && similar(&self.clasters[i].v, &entity.v, self.b) {
                if pass(&self.clasters[i].v, &entity.v, self.p) {
                    return Some(i);
                }
            }
        }
        None
    }
    fn dispatch_entity(&mut self, data: &Vec<bit_vec::BitVec>, entity: &mut ClasterEntity) -> bool {
        match self.find_claster(entity) {
            Some(index) => {
                self.remove_from_claster(data, entity);
                self.clasters[index].add(entity);
                true
            },
            None => false
        }
    }
    fn recalculate(&mut self) {
        self.clasters = self
            .clasters
            .drain(..)
            .filter(|x| x.indexes.len() != 0)
            .collect();
    }
}

pub fn art1(
    data: &Vec<bit_vec::BitVec>,
    amount_clasters: &usize,
    p: &f64,
    b: &f64,
) -> Vec<Claster> {
    let mut entities: Vec<_> = data
        .iter()
        .enumerate()
        .map(|(i, x)| ClasterEntity {
            v: x.clone(),
            data_index: i,
            claster_id: "-".into(),
        })
        .collect();
    let mut first_claster = Claster::new();
    first_claster.add(&mut entities[0]);
    let mut clasters = ClastersManager {
        clasters: vec![first_claster],
        b: *b,
        p: *p,
    };

    let mut changed = true;
    let mut max_iters = 500;
    while changed && max_iters > 0 {
        max_iters -= 1;
        changed = false;
        for entity in &mut entities {
            changed = clasters.dispatch_entity(data, entity);
            if entity.claster_id == "-" {
                let mut claster = Claster::new();
                claster.add(entity);
                clasters.add_claster(claster);
                changed = true;
            }
            clasters.recalculate();
        }
    }
    clasters.clasters
}
