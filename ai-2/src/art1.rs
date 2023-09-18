use bit_vec;

struct Claster {
    v: bit_vec::BitVec,
    indexes: Vec<usize>,
}

impl Claster {
    pub fn new() -> Self {
        Self {
            v: bit_vec::BitVec::default(),
            indexes: Vec::default(),
        }
    }
    pub fn add(&mut self, data: &Vec<bit_vec::BitVec>, index: &usize) {
        if self.indexes.is_empty() {
            self.v = data[*index].clone();
        } else {
            self.v.and(&data[*index]);
        }
        self.indexes.push(*index);
    }
    pub fn recalculate(&mut self, data: &Vec<bit_vec::BitVec>) {
        self.v.clear();
        for i in &self.indexes {
            self.v.and(&data[*i]);
        }
    }
    pub fn contains(&self, index: &usize) -> bool {
        self.indexes.contains(index)
    }
    pub fn remove(&mut self, data: &Vec<bit_vec::BitVec>, index: &usize) {
        self.indexes
            .remove(self.indexes.iter().position(|x| x == index).unwrap());
        self.recalculate(data);
    }
}

fn get_energy(v: &bit_vec::BitVec) -> u32 {
    v.blocks().map(|x| x.count_ones()).sum()
}

fn similar(P: &bit_vec::BitVec, E: &bit_vec::BitVec, b: f64) -> bool {
    let mut intersect = P.clone();
    intersect.and(E);
    get_energy(&intersect) as f64 / b + get_energy(P) as f64
        > get_energy(E) as f64 / (b + P.len() as f64)
}

pub fn art1(data: &Vec<bit_vec::BitVec>, amount_clasters: &usize, p: &f64, b: &f64) -> i64 {
    let mut clasters: Vec<Claster> = vec![];
    clasters.push(Claster::new());
    clasters[0].add(data, &0);
    while true {
        for v in data {}
    }
    1
}
