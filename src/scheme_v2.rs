#![allow(unused)]
#![allow(dead_code)]

// ============================================================================
use ndarray::Ix2;
use num::rational::Rational64;
use ndarray::Array;
use kepler_two_body::{OrbitalElements, OrbitalState};




pub type BlockIndex = (usize, usize);




// ============================================================================
pub struct BlockData<Conserved>
{
    pub initial_conserved: Array<Conserved, Ix2>,
    pub cell_centers:   Array<(f64, f64), Ix2>,
    pub face_centers_x: Array<(f64, f64), Ix2>,
    pub face_centers_y: Array<(f64, f64), Ix2>,
    pub index:          BlockIndex,
}




// ============================================================================
pub struct State<Conserved>
{
    pub time: f64,
    pub iteration: Rational64,
    pub conserved: Vec<Array<Conserved, Ix2>>,
}




// ============================================================================
trait Hydrodynamics
{
    type Conserved;
    type Primitive;

    fn source_terms(
        &self,
        solver: &Solver,
        conserved: Conserved,
        background_conserved: Conserved,
        x: f64,
        y: f64,
        dt: f64,
        two_body_state: &OrbitalState) -> [Conserved; 5];
}




struct TwoDimensionalIsothermalHydrodynamics;




use hydro_iso2d::Conserved;
use hydro_iso2d::Primitive;

impl Hydrodynamics for TwoDimensionalIsothermalHydrodynamics
{
    type Conserved = hydro_iso2d::Conserved;
    type Primitive = hydro_iso2d::Primitive;

    fn source_terms(
        &self,
        solver: &Solver,
        conserved: Conserved,
        background_conserved: Conserved,
        x: f64,
        y: f64,
        dt: f64,
        two_body_state: &kepler_two_body::OrbitalState) -> [Conserved; 5]
    {
        let p1 = two_body_state.0;
        let p2 = two_body_state.1;

        let [ax1, ay1] = p1.gravitational_acceleration(x, y, solver.softening_length);
        let [ax2, ay2] = p2.gravitational_acceleration(x, y, solver.softening_length);

        let rho = conserved.density();
        let fx1 = rho * ax1;
        let fy1 = rho * ay1;
        let fx2 = rho * ax2;
        let fy2 = rho * ay2;

        let x1 = p1.position_x();
        let y1 = p1.position_y();
        let x2 = p2.position_x();
        let y2 = p2.position_y();

        let sink_rate1 = solver.sink_kernel(x - x1, y - y1);
        let sink_rate2 = solver.sink_kernel(x - x2, y - y2);

        let r = (x * x + y * y).sqrt();
        let y = (r - solver.domain_radius) / solver.buffer_scale;
        let omega_outer = (two_body_state.total_mass() / solver.domain_radius.powi(3)).sqrt();
        let buffer_rate = 0.5 * solver.buffer_rate * (1.0 + f64::tanh(y)) * omega_outer;

        return [
            Conserved(0.0, fx1, fy1) * dt,
            Conserved(0.0, fx2, fy2) * dt,
            conserved * (-sink_rate1 * dt),
            conserved * (-sink_rate2 * dt),
            (conserved - background_conserved) * (-dt * buffer_rate),
        ];
    }
}




// ============================================================================
pub struct Solver
{
    pub buffer_rate: f64,
    pub buffer_scale: f64,
    pub cfl: f64,
    pub domain_radius: f64,
    pub mach_number: f64,
    pub nu: f64,
    pub plm: f64,
    pub rk_order: i64,
    pub sink_radius: f64,
    pub sink_rate: f64,
    pub softening_length: f64,
    pub orbital_elements: OrbitalElements,
}

impl Solver
{
    fn source_terms<H: Hydrodynamics>(
        &self,
        hydrodynamics: H,
        conserved: Conserved,
        background_conserved: Conserved,
        x: f64,
        y: f64,
        dt: f64,
        two_body_state: &OrbitalState) -> [Conserved; 5]
    {
        hydrodynamics.source_terms(self, conserved, background_conserved, x, y, dt, two_body_state)
    }
    
    fn sink_kernel(&self, dx: f64, dy: f64) -> f64
    {
        let r2 = dx * dx + dy * dy;
        let s2 = self.sink_radius * self.sink_radius;

        if r2 < s2 * 9.0 {
            self.sink_rate * f64::exp(-(r2 / s2).powi(3))
        } else {
            0.0
        }
    }
}
