use std::ops::{Add, Sub, Mul, Div};
use kepler_two_body::OrbitalState;

use crate::physics::{
    CellData,
    Direction,
    HydroError,
    ItemizedChange,
    Solver,
};




// ============================================================================
pub trait Arithmetic: Add<Output=Self> + Sub<Output=Self> + Mul<f64, Output=Self> + Div<f64, Output=Self> + Sized {}




// ============================================================================
pub trait Zeros {
    fn zeros() -> Self;
}




// ============================================================================
pub trait ItemizeData: Zeros + Arithmetic + Copy + Clone + hdf5::H5Type {
}




// ============================================================================
pub trait Conserved: Clone + Copy + Send + Sync + hdf5::H5Type + ItemizeData {
    fn mass_and_momentum(&self) -> (f64, f64, f64);
}




// ============================================================================
pub trait Primitive: Clone + Copy + Send + Sync + hdf5::H5Type {
    fn velocity_x(self) -> f64;
    fn velocity_y(self) -> f64;
    fn mass_density(self) -> f64;
}




// ============================================================================
pub trait Hydrodynamics: Copy + Send
{
    type Conserved: Conserved;
    type Primitive: Primitive;

    fn gamma_law_index(&self) -> f64;
    fn plm_gradient(&self, theta: f64, a: &Self::Primitive, b: &Self::Primitive, c: &Self::Primitive) -> Self::Primitive;
    fn try_to_primitive(&self, u: Self::Conserved) -> Result<Self::Primitive, HydroError>;
    fn to_primitive(&self, u: Self::Conserved) -> Self::Primitive;
    fn to_conserved(&self, p: Self::Primitive) -> Self::Conserved;

    fn source_terms(
        &self,
        solver: &Solver,
        conserved: Self::Conserved,
        background_conserved: Self::Conserved,
        x: f64,
        y: f64,
        dt: f64,
        two_body_state: &OrbitalState) -> ItemizedChange<Self::Conserved>;

    fn intercell_flux<'a>(
        &self,
        solver: &Solver,
        l: &CellData<'a, Self::Primitive>,
        r: &CellData<'a, Self::Primitive>,
        f: &(f64, f64),
        x: f64,
        y: f64,
        two_body_state: &kepler_two_body::OrbitalState,
        axis: Direction) -> Self::Conserved;
}
