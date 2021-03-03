use std::ops::{Add, Sub, Mul, Div};
use kepler_two_body::OrbitalState;
use crate::app::AnyPrimitive;
use crate::mesh::Mesh;
use crate::state::ItemizedChange;
use crate::physics::{
    CellData,
    Direction,
    HydroErrorType,
    Physics,
};




// ============================================================================
pub trait Arithmetic:
    Add<Output=Self> +
    Sub<Output=Self> +
    Mul<f64, Output=Self> +
    Div<f64, Output=Self> + Sized
{
}




// ============================================================================
pub trait Zeros {
    fn zeros() -> Self;
}




// ============================================================================
pub trait Conserved: Clone + Copy + Send + Sync + Zeros + Arithmetic {
    fn mass_and_momentum(&self) -> (f64, f64, f64);
}




// ============================================================================
pub trait Primitive: Clone + Copy + Send + Sync {
    fn velocity_x(self) -> f64;
    fn velocity_y(self) -> f64;
    fn mass_density(self) -> f64;
}




/**
 * Interface for a hydrodynamics system
 */
pub trait Hydrodynamics: Clone + Copy + Send {

    type Conserved: Conserved;
    type Primitive: Primitive;

    fn gamma_law_index(&self) -> f64;
    fn global_mach_number(&self) -> Option<f64>;
    fn plm_gradient(&self, theta: f64, a: &Self::Primitive, b: &Self::Primitive, c: &Self::Primitive) -> Self::Primitive;
    fn try_to_primitive(&self, u: Self::Conserved) -> Result<Self::Primitive, HydroErrorType>;
    fn to_primitive(&self, u: Self::Conserved) -> Self::Primitive;
    fn to_conserved(&self, p: Self::Primitive) -> Self::Conserved;
    fn from_any(&self, p: AnyPrimitive) -> Self::Primitive;
    fn max_signal_speed(&self, u: Self::Conserved) -> f64;

    fn source_terms(
        &self,
        physics: &Physics,
        mesh: &Mesh,
        conserved: Self::Conserved,
        background_conserved: Self::Conserved,
        x: f64,
        y: f64,
        t: f64,
        dt: f64,
        two_body_state: &OrbitalState) -> ItemizedChange<Self::Conserved>;

    fn intercell_flux<'a>(
        &self,
        physics: &Physics,
        l: &CellData<'a, Self::Primitive>,
        r: &CellData<'a, Self::Primitive>,
        x: f64,
        y: f64,
        gravitational_potiential: f64,
        axis: Direction) -> Self::Conserved;

}




/**
 * Interface for a struct that can act as an initial or background hydrodynamics
 * model
 */
pub trait InitialModel: Clone {

    /**
     * Return the hydrodynamics state at a given cylindrical radius
     */
    fn primitive_at<H: Hydrodynamics>(&self, hydro: &H, xy: (f64, f64)) -> AnyPrimitive;

    /**
     * Validate the model
     */
    fn validate<H: Hydrodynamics>(&self, hydro: &H) -> anyhow::Result<()>;
}
