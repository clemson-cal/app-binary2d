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
        let (ix, iy) = mesh.get_cell_index(index, self.x, self.y);
        let dx = mesh.cell_spacing_x();
        let dy = mesh.cell_spacing_y();
        let wx = (self.x - mesh.face_center(index, ix + 1, ix, Direction::X).0) / dx;
        let wy = (self.y - mesh.face_center(index, ix, iy + 1, Direction::Y).1) / dy;
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
// pub fn apply_boundary_condition(tracer: &Tracer, domain_radius: f64) -> Tracer
// {
//     let mut x = tracer.x;
//     let mut y = tracer.y;

//     if x >= domain_radius {
//         x -= 2.0 * domain_radius;
//     }
//     if x < -domain_radius {
//         x += 2.0 * domain_radius;
//     }
//     if y >= domain_radius {
//         y -= 2.0 * domain_radius;
//     }
//     if y < -domain_radius {
//         y += 2.0 * domain_radius;
//     }

//     Tracer{
//         x: x,
//         y: y,
//         id: tracer.id,
//     }
// }




