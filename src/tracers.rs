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
    pub fn new(xy: (f64, f64), id: usize) -> Tracer
    {
        return Tracer{x: xy.0, y: xy.1, id: id}
    }

    pub fn update(&self, v: (f64, f64), dt: f64) ->Tracer
    {
        Tracer{
            x : self.x + v.0 * dt,
            y : self.y + v.1 * dt,
            id: self.id,
        }
    }
}




// ============================================================================
pub fn update_tracers(
    tracer: Tracer, 
    mesh: &Mesh,
    index: crate::BlockIndex,
    vfield_x: &Array<f64, Ix2>,
    vfield_y: &Array<f64, Ix2>,
    pad_size: usize,
    dt: f64) -> Tracer
{   
    let (i, j) = mesh.get_cell_index(index, tracer.x, tracer.y); // --> telling nominal i,j's
    let dx = mesh.cell_spacing_x();
    let dy = mesh.cell_spacing_y();
    let wx = (tracer.x - mesh.face_center_at(index, i, j, Direction::X).0) / dx; 
    let wy = (tracer.y - mesh.face_center_at(index, i, j, Direction::Y).1) / dy;

    // Need to get logical ix, iy, jx, jy appropriate for the padded array size
    // vx.shape = [n + 1, n]
    // vy.shape = [n, n + 1]
    let m = (i + pad_size as i64) as usize;
    let n = (j + pad_size as i64) as usize;
    let vx = (1.0 - wx) * vfield_x[[m, n]] + wx * vfield_x[[m + 1, n]];
    let vy = (1.0 - wy) * vfield_y[[m, n]] + wy * vfield_y[[m, n + 1]];

    tracer.update((vx, vy), dt)
}

pub fn push_new_tracers(init_tracers: Vec<Tracer>, neigh_tracers: NeighborTracerVecs, mesh: &Mesh, index: BlockIndex) -> Vec<Tracer>
{
    let r = mesh.block_length();
    let (x0, y0) = mesh.block_start(index);
    let mut tracers = Vec::new();

    for block_tracers in neigh_tracers.iter().flat_map(|r| r.iter())
    {
        for t in block_tracers.iter()
        {
            if (t.x >= x0) && (t.x < x0 + r) && (t.y >= y0) && (t.y < y0 + r) 
            {
                tracers.push(t.clone());
            }
        }
    }
    tracers.extend(init_tracers);
    return tracers;
}

pub fn filter_block_tracers(tracers: Vec<Tracer>, mesh: &Mesh, index: BlockIndex) -> (Vec<Tracer>, Vec<Tracer>)
{
    let r = mesh.block_length();
    let (x0, y0) = mesh.block_start(index);
    return tracers.into_iter().partition(|t| t.x >= x0 && t.x < x0 + r && t.y >= y0 && t.y < y0 + r);
}



