use godunov_core::runge_kutta;
use kepler_two_body::{OrbitalElements, OrbitalState};
use crate::mesh::Mesh;




// ============================================================================
pub struct SourceTerms
{
    pub fx1: f64,
    pub fy1: f64,
    pub fx2: f64,
    pub fy2: f64,
    pub sink_rate1: f64,
    pub sink_rate2: f64,
    pub buffer_rate: f64,
}



// ============================================================================
#[derive(Clone)]
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
    pub stress_dim: i64,
    pub force_flux_comm: bool,
    pub low_mem: bool,
    pub orbital_elements: OrbitalElements,
}

impl Solver
{
    pub fn runge_kutta(&self) -> runge_kutta::RungeKuttaOrder
    {
        use std::convert::TryFrom;
        runge_kutta::RungeKuttaOrder::try_from(self.rk_order).expect("illegal RK order")
    }

    pub fn need_flux_communication(&self) -> bool
    {
        self.force_flux_comm
    }

    pub fn effective_resolution(&self, mesh: &Mesh) -> f64
    {
        f64::min(mesh.cell_spacing_x(), mesh.cell_spacing_y())
    }

    pub fn min_time_step(&self, mesh: &Mesh) -> f64
    {
        self.cfl * self.effective_resolution(mesh) / self.maximum_orbital_velocity()
    }

    pub fn sink_kernel(&self, dx: f64, dy: f64) -> f64
    {
        let r2 = dx * dx + dy * dy;
        let s2 = self.sink_radius * self.sink_radius;

        if r2 < s2 * 9.0 {
            self.sink_rate * f64::exp(-(r2 / s2).powi(3))
        } else {
            0.0
        }
    }

    pub fn sound_speed_squared(&self, xy: &(f64, f64), state: &OrbitalState) -> f64
    {
        -state.gravitational_potential(xy.0, xy.1, self.softening_length) / self.mach_number.powi(2)
    }

    pub fn maximum_orbital_velocity(&self) -> f64
    {
        1.0 / self.softening_length.sqrt()
    }

    pub fn source_terms(&self, two_body_state: &kepler_two_body::OrbitalState, x: f64, y: f64, surface_density: f64) -> SourceTerms
    {
        let p1 = two_body_state.0;
        let p2 = two_body_state.1;

        let [ax1, ay1] = p1.gravitational_acceleration(x, y, self.softening_length);
        let [ax2, ay2] = p2.gravitational_acceleration(x, y, self.softening_length);

        let fx1 = surface_density * ax1;
        let fy1 = surface_density * ay1;
        let fx2 = surface_density * ax2;
        let fy2 = surface_density * ay2;

        let x1 = p1.position_x();
        let y1 = p1.position_y();
        let x2 = p2.position_x();
        let y2 = p2.position_y();

        let sink_rate1 = self.sink_kernel(x - x1, y - y1);
        let sink_rate2 = self.sink_kernel(x - x2, y - y2);

        let r = (x * x + y * y).sqrt();
        let y = (r - self.domain_radius) / self.buffer_scale;
        let omega_outer = (two_body_state.total_mass() / self.domain_radius.powi(3)).sqrt();
        let buffer_rate = 0.5 * self.buffer_rate * (1.0 + f64::tanh(y)) * omega_outer;

        SourceTerms{
            fx1: fx1,
            fy1: fy1,
            fx2: fx2,
            fy2: fy2,
            sink_rate1: sink_rate1,
            sink_rate2: sink_rate2,
            buffer_rate: buffer_rate,
        }
    }
}
