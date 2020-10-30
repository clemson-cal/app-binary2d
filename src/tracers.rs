use rand::Rng;
use std::ops::{Add, Mul};
use ndarray::{Array, Ix2};
use num::rational::Rational64;
use num::ToPrimitive;
use crate::Direction;
use crate::scheme::*;




// ============================================================================
#[repr(C)]
#[derive(hdf5::H5Type, Clone)]
pub struct Tracer
{
    pub id: usize,
    pub x : f64,
    pub y : f64,
}




// ============================================================================
impl Add for Tracer
{
    type Output = Self;

    fn add(self, other: Self) -> Self
    {
        Self{
            id: self.id,
            x : self.x + other.x,
            y : self.y + other.y,
        }
    }
}

impl Mul<Rational64> for Tracer
{
    type Output = Self;

    fn mul(self, b: Rational64) -> Self
    {
        Self{
            id: self.id,
            x : self.x * b.to_f64().unwrap(),
            y : self.y * b.to_f64().unwrap(),
        }
    }
}




// ============================================================================
impl Tracer
{
    pub fn _default() -> Tracer
    {
        return Tracer{x: 0.0, y: 0.0, id: 0};
    }

    pub fn new(xy: (f64, f64), id: usize) -> Tracer
    {
        return Tracer{x: xy.0, y: xy.1, id: id}
    }

    pub fn randomize(start: (f64, f64), length: f64, id: usize) -> Tracer
    {
        let mut rng = rand::thread_rng();
        let rand_x = rng.gen_range(0.0, length) + start.0;
        let rand_y = rng.gen_range(0.0, length) + start.1;
        return Tracer{x: rand_x, y: rand_y, id: id};
    }

    pub fn update(&self, mesh: &Mesh, index: crate::BlockIndex, vfield_x: &Array<f64, Ix2>, vfield_y: &Array<f64, Ix2>, dt: f64) -> Tracer
    {
        let (ix, iy) = verify_indexes(mesh.get_cell_index(index, self.x, self.y), mesh.block_size);
        let dx = mesh.cell_spacing_x();
        let dy = mesh.cell_spacing_y();
        let wx = (mesh.face_center_at(index, ix + 1, iy, Direction::X).0 - self.x) / dx; // Triple check sign...
        let wy = (mesh.face_center_at(index, ix, iy + 1, Direction::Y).1 - self.y) / dy;
        let vx = (1.0 - wx) * vfield_x[[ix, iy]] + wx * vfield_x[[ix + 1, iy]];
        let vy = (1.0 - wy) * vfield_y[[ix, iy]] + wy * vfield_y[[ix, iy + 1]];

        return Tracer{
            x : self.x + vx * dt,
            y : self.y + vy * dt,
            id: self.id,
        };
    }
}




// ============================================================================
fn verify_indexes(ij: (usize, usize), block_size: usize) -> (usize, usize)
{
    let (ix, iy) = ij;
    if ix > block_size
    {
        println!("tracers::verify_cell_index : tracer moved beyond ghost zones (X). Crashing....");
    }
    if iy > block_size
    {
        println!("tracers::verify_cell_index : tracer moved beyond ghost zones (Y). Crashing....");
    }
    (ix, iy)
}




// ============================================================================
pub fn push_new_tracers(init_tracers: Vec<Tracer>, neigh_tracers: NeighborTracerVecs, mesh: &Mesh, index: BlockIndex) -> Vec<Tracer>
{
    let r = mesh.block_length();
    let (x0, y0) = mesh.block_start(index);
    let mut tracers = Vec::new();

    for (i, block_tracers) in neigh_tracers.iter().flat_map(|r| r.iter()).enumerate()
    {
        // algorithmically unneccesary?
        if i == 5 // This is my block 
        {
            continue;
        }
        for t in block_tracers.iter()
        {
            if (t.x >= x0) & (t.x < x0 + r) & (t.y >= y0) & (t.y < y0 + r) 
            {
                tracers.push(t.clone());
            }
        }
    }
    tracers.extend(init_tracers); // consumes 'init_tracers'
    return tracers;
}

pub fn filter_block_tracers(tracers: Vec<Tracer>, mesh: &Mesh, index: BlockIndex) -> (Vec<Tracer>, Vec<Tracer>)
{
    let r = mesh.block_length();
    let (x0, y0) = mesh.block_start(index);
    let mut mine   = Vec::new();    
    let mut others = Vec::new();    

    for t in tracers.into_iter()
    {   
        if (t.x < x0) | (t.x >= x0 + r) | (t.y < y0) | (t.y >= y0 + r) // if left my block
        {
            others.push(t);
        }
        else // never left
        {
            mine.push(t);
        }
    }
    return (mine, others);
}

pub fn apply_tracer_target(tracers: Vec<Tracer>, mesh: &Mesh, index: BlockIndex) -> Vec<Tracer>
{
    if tracers.len() < mesh.tracers_per_block
    {    
        let mut rng = rand::thread_rng();
        let id0     = mesh.tracers_per_block * mesh.num_blocks * mesh.num_blocks;
        let init    = |_| Tracer::randomize(mesh.block_start(index), mesh.block_length(), rng.gen::<usize>() + id0);
        
        let tracer_deficit = mesh.tracers_per_block - tracers.len();
        let mut new = (0..tracer_deficit).map(init).collect::<Vec<Tracer>>();
        
        new.extend(tracers);
        return new;
    }
    return tracers;
}



