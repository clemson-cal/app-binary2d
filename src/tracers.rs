use rand::Rng;
use std::ops::{Add, Mul};
use ndarray::{Array, Ix2};
use num::rational::Rational64;
use num::ToPrimitive;
use crate::Direction;




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

    pub fn randomize(start: (f64, f64), length: f64, id: usize) -> Tracer
    {
        let mut rng = rand::thread_rng();
        let rand_x = rng.gen_range(0.0, length) + start.0;
        let rand_y = rng.gen_range(0.0, length) + start.1;
        return Tracer{x: rand_x, y: rand_y, id: id};
    }


    pub fn update(&self, mesh: &crate::scheme::Mesh, index: crate::BlockIndex, vfield_x: &Array<f64, Ix2>, vfield_y: &Array<f64, Ix2>, dt: f64) -> Tracer
    {
        // let (ix, iy) = mesh.get_cell_index(index, self.x, self.y);
        let (ix, iy) = calc_my_cell_index(self.x, self.y, index, &mesh);
        let dx = mesh.cell_spacing_x();
        let dy = mesh.cell_spacing_y();

        // Need to triple check sign because I've tied myself in a knot lol
        let wx = (mesh.face_center(index, ix + 1, iy, Direction::X).0 - self.x) / dx;
        let wy = (mesh.face_center(index, ix, iy + 1, Direction::Y).1 - self.y) / dy;
        let vx = (1.0 - wx) * vfield_x[[ix, iy]] + wx * vfield_x[[ix + 1, iy]];
        let vy = (1.0 - wy) * vfield_y[[ix, iy]] + wy * vfield_y[[ix, iy + 1]];

        return Tracer{
            x : self.x + vx * dt,
            y : self.y + vy * dt,
            id: self.id,
        };
    }
}



pub fn calc_my_cell_index(x: f64, y: f64, bindex: crate::BlockIndex, mesh: &crate::scheme::Mesh) -> (usize, usize)
{
    let (mut ix, mut iy) = mesh.get_cell_index(bindex, x, y);

    if ix >= mesh.block_size
    {
        ix -= 1;
    }

    if iy >= mesh.block_size
    {
        iy -= 1;
    }

    return (ix, iy);
}




// ============================================================================
// pub fn rebin_tracers(
//     state   : BlockState, 
//     mesh    : &Mesh, 
//     sender  : &crossbeam::Sender<Vec<Tracer>>, 
//     receiver: &crossbeam::Receiver<NeighborTracerVecs>,
//     block_index: BlockIndex) -> BlockState
// {
//     // Only send tracers I'm not responsible for to the hashmap
//     let (my_tracers_0, their_tracers) = filter_block_tracers(state.tracers, &mesh, block_index);
//     sender.send(their_tracers).unwrap();   

//     let my_tracers    = apply_tracer_target(my_tracers_0, &mesh, block_index);
//     let neigh_tracers = receiver.recv().unwrap();

//     return BlockState{
//         solution: state.solution,
//         tracers : push_new_tracers(my_tracers, neigh_tracers, &mesh, block_index),
//     };
// }

// pub fn push_new_tracers(init_tracers: Vec<Tracer>, neigh_tracers: NeighborTracerVecs, mesh: &Mesh, index: BlockIndex) -> Vec<Tracer>
// {
//     let r = mesh.block_length();
//     let (x0, y0) = mesh.block_start(index);
//     let mut tracers = Vec::new();

//     for (i, block_tracers) in neigh_tracers.iter().flat_map(|r| r.iter()).enumerate()
//     {
//         // algorithmically unneccesary?
//         if i == 5 // This is my block 
//         {
//             continue;
//         }
//         for t in block_tracers.iter()
//         {
//             if (t.x >= x0) & (t.x < x0 + r) & (t.y >= y0) & (t.y < y0 + r) 
//             {
//                 tracers.push(t.clone());
//             }
//         }
//     }
//     tracers.extend(init_tracers); // consumes 'init_tracers'
//     return tracers;
// }

// pub fn filter_block_tracers(tracers: Vec<Tracer>, mesh: &Mesh, index: BlockIndex) -> (Vec<Tracer>, Vec<Tracer>)
// {
//     let r = mesh.block_length();
//     let (x0, y0) = mesh.block_start(index);
//     let mut mine   = Vec::new();    
//     let mut others = Vec::new();    

//     for t in tracers.into_iter()
//     {   
//         if (t.x < x0) | (t.x >= x0 + r) | (t.y < y0) | (t.y >= y0 + r) // if left my block
//         {
//             others.push(t);
//         }
//         else // never left
//         {
//             mine.push(t);
//         }
//     }
//     return (mine, others);
// }

// pub fn apply_tracer_target(tracers: Vec<Tracer>, mesh: &Mesh, index: BlockIndex) -> Vec<Tracer>
// {
//     if tracers.len() < mesh.tracers_per_block
//     {    
//         let mut rng = rand::thread_rng();
//         let id0     = mesh.tracers_per_block * mesh.num_blocks * mesh.num_blocks;
//         let init    = |_| Tracer::randomize(mesh.block_start(index), mesh.block_length(), rng.gen::<usize>() + id0);
        
//         let tracer_deficit = mesh.tracers_per_block - tracers.len();
//         let mut new = (0..tracer_deficit).map(init).collect::<Vec<Tracer>>();
        
//         new.extend(tracers);
//         return new;
//     }
//     return tracers;
// }




// ============================================================================
// pub fn apply_boundary_condition(tracer: &Tracer, domain_radius: f64) -> Tracer
// {
//     let has_left_x: bool = (tracer.x >= domain_radius | tracer.x < -domain_radius);
//     let has_left_y: bool = (tracer.y >= domain_radius | tracer.y < -domain_radius);
    
//     if has_left_x | has_left_y
//     {
//         return tracer.randomize((-domain_radius, -domain_radius), 2.0 * domain_radius, 2.0 * tracer.id);
//     }

//     return tracer;
// }




