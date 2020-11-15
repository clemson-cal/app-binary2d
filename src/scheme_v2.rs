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
#[derive(Copy, Clone)]
struct CellData<'a, Primitive>
{
    pc: &'a Primitive,
    gx: &'a Primitive,
    gy: &'a Primitive,
}

impl<'a, Primitive> CellData<'_, Primitive>
{
    fn new(pc: &'a Primitive, gx: &'a Primitive, gy: &'a Primitive) -> CellData<'a, Primitive>
    {
        CellData{
            pc: pc,
            gx: gx,
            gy: gy,
        }
    }
}




// ============================================================================
trait Hydrodynamics
{
    type Conserved;
    type Primitive;
    type Direction;

    fn velocity_x        (&self, primitie: &Self::Primitive) -> f64;
    fn velocity_y        (&self, primitie: &Self::Primitive) -> f64;
    fn density           (&self, primitie: &Self::Primitive) -> f64;
    fn gradient_field<'a>(&self, cell_data: &CellData<'a, Self::Primitive>, axis: Self::Direction) -> &'a Self::Primitive;
    fn strain_field  <'a>(&self, cell_data: &CellData<'a, Self::Primitive>, row: Self::Direction, col: Self::Direction) -> f64;
    fn stress_field  <'a>(&self, cell_data: &CellData<'a, Self::Primitive>, kinematic_viscosity: f64, row: Self::Direction, col: Self::Direction) -> f64;

    fn source_terms(
        &self,
        solver: &Solver,
        conserved: Self::Conserved,
        background_conserved: Self::Conserved,
        x: f64,
        y: f64,
        dt: f64,
        two_body_state: &OrbitalState) -> [Self::Conserved; 5];
}




// ============================================================================
struct TwoDimensionalIsothermalHydrodynamics
{
}

impl Hydrodynamics for TwoDimensionalIsothermalHydrodynamics
{
    type Direction = hydro_iso2d::Direction;
    type Conserved = hydro_iso2d::Conserved;
    type Primitive = hydro_iso2d::Primitive;

    fn velocity_x(&self, primitive: &Self::Primitive) -> f64 { primitive.velocity_x() }
    fn velocity_y(&self, primitive: &Self::Primitive) -> f64 { primitive.velocity_y() }
    fn density   (&self, primitive: &Self::Primitive) -> f64 { primitive.density() }

    fn gradient_field<'a>(&self, cell_data: &CellData<'a, Self::Primitive>, axis: Self::Direction) -> &'a Self::Primitive
    {
        use hydro_iso2d::Direction::{X, Y};

        match axis
        {
            X => cell_data.gx,
            Y => cell_data.gy,
        }
    }

    fn strain_field<'a>(&self, cell_data: &CellData<'a, Self::Primitive>, row: Self::Direction, col: Self::Direction) -> f64
    {
        use hydro_iso2d::Direction::{X, Y};

        match (row, col)
        {
            (X, X) => cell_data.gx.velocity_x() - cell_data.gy.velocity_y(),
            (X, Y) => cell_data.gx.velocity_y() + cell_data.gy.velocity_x(),
            (Y, X) => cell_data.gx.velocity_y() + cell_data.gy.velocity_x(),
            (Y, Y) =>-cell_data.gx.velocity_x() + cell_data.gy.velocity_y(),
        }
    }

    fn stress_field<'a>(&self, cell_data: &CellData<'a, Self::Primitive>, kinematic_viscosity: f64, row: Self::Direction, col: Self::Direction) -> f64
    {
        kinematic_viscosity * cell_data.pc.density() * self.strain_field(cell_data, row, col)
    }

    fn source_terms(
        &self,
        solver: &Solver,
        conserved: Self::Conserved,
        background_conserved: Self::Conserved,
        x: f64,
        y: f64,
        dt: f64,
        two_body_state: &kepler_two_body::OrbitalState) -> [Self::Conserved; 5]
    {
        use hydro_iso2d::Conserved;

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
        conserved: H::Conserved,
        background_conserved: H::Conserved,
        x: f64,
        y: f64,
        dt: f64,
        two_body_state: &OrbitalState) -> [H::Conserved; 5]
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
