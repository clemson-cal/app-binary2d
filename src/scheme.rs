use kepler_two_body::OrbitalElements;
use ndarray::{Axis, Array, Ix2};
use hydro_iso2d::*;
use kepler_two_body::OrbitalState;
use crate::SolutionState;




// ============================================================================
pub struct BlockData
{
    pub initial_conserved: Array<Conserved, Ix2>,
    pub cell_centers:   Array<(f64, f64), Ix2>,
    pub face_centers_x: Array<(f64, f64), Ix2>,
    pub face_centers_y: Array<(f64, f64), Ix2>,
}




// ============================================================================
pub struct Solver
{
    pub sink_rate: f64,
    pub buffer_rate: f64,
    pub buffer_scale: f64,
    pub softening_length: f64,
    pub sink_radius: f64,
    pub domain_radius: f64,
    pub cfl: f64,
    pub plm: f64,
    pub nu: f64,
    pub mach_number: f64,
    pub orbital_elements: OrbitalElements,
}




// ============================================================================
impl Solver
{
    fn source_terms(&self,
        conserved: Conserved,
        background_conserved: Conserved,
        x: f64,
        y: f64,
        dt: f64,
        two_body_state: &OrbitalState) -> [Conserved; 5]
    {
        let [ax1, ay1] = two_body_state.0.gravitational_acceleration(x, y, self.softening_length);
        let [ax2, ay2] = two_body_state.1.gravitational_acceleration(x, y, self.softening_length);

        let rho = conserved.density();
        let fx1 = rho * ax1;
        let fy1 = rho * ay1;
        let fx2 = rho * ax2;
        let fy2 = rho * ay2;

        let x1 = two_body_state.0.position_x();
        let y1 = two_body_state.0.position_y();
        let x2 = two_body_state.1.position_x();
        let y2 = two_body_state.1.position_y();

        let sink_rate1 = self.sink_kernel(x - x1, y - y1);
        let sink_rate2 = self.sink_kernel(x - x2, y - y2);

        let r = (x * x + y * y).sqrt();
        let y = (r - self.domain_radius) / self.buffer_scale;
        let buffer_rate = 0.5 * self.buffer_rate * (1.0 + f64::tanh(y));

        return [
            Conserved(0.0, fx1, fy1) * dt,
            Conserved(0.0, fx2, fy2) * dt,
            conserved * (-sink_rate1 * dt),
            conserved * (-sink_rate2 * dt),
            (conserved - background_conserved) * (-dt * buffer_rate),
        ];
    }

    fn sound_speed_squared(&self, xy: &(f64, f64), state: &OrbitalState) -> f64
    {
        -state.gravitational_potential(xy.0, xy.1, self.softening_length) / self.mach_number
    }

    fn maximum_orbital_velocity(&self) -> f64
    {
        1.0 / self.softening_length.sqrt()
    }

    fn sink_kernel(&self, dx: f64, dy: f64) -> f64
    {
        let r2 = dx * dx + dy * dy;
        let s2 = self.sink_radius * self.sink_radius;

        if r2 < s2 * 9.0 {
            self.sink_rate * f64::exp(-r2 * r2 / s2 / s2)
        } else {
            0.0
        }
    }
}




// ============================================================================
#[derive(Copy, Clone)]
struct CellData<'a>
{
    pc: &'a Primitive,
    gx: &'a Primitive,
    gy: &'a Primitive,
}

impl<'a> CellData<'_>
{
    fn new(pc: &'a Primitive, gx: &'a Primitive, gy: &'a Primitive) -> CellData<'a>
    {
        CellData{
            pc: pc,
            gx: gx,
            gy: gy,
        }
    }

    fn strain_field(&self, row: Direction, col: Direction) -> f64
    {
        use Direction::{X, Y};
        match (row, col)
        {
            (X, X) => self.gx.velocity_x() - self.gy.velocity_y(),
            (X, Y) => self.gx.velocity_y() + self.gy.velocity_x(),
            (Y, X) => self.gx.velocity_y() + self.gy.velocity_x(),
            (Y, Y) =>-self.gx.velocity_x() + self.gy.velocity_y(),
        }
    }

    fn gradient_field(&self, axis: Direction) -> &Primitive
    {
        use Direction::{X, Y};
        match axis
        {
            X => self.gx,
            Y => self.gy,
        }
    }
}




// ============================================================================
pub fn cell_centers(domain_radius: f64, block_size: usize) -> Array<(f64, f64), Ix2>
{
    use ndarray_ops::*;
    let xv = Array::linspace(-domain_radius, domain_radius, block_size + 1);
    let yv = Array::linspace(-domain_radius, domain_radius, block_size + 1);
    let xc = adjacent_mean(&xv, Axis(0));
    let yc = adjacent_mean(&yv, Axis(0));
    return cartesian_product2(xc, yc);
}

pub fn face_centers_x(domain_radius: f64, block_size: usize) -> Array<(f64, f64), Ix2>
{
    use ndarray_ops::*;
    let xv = Array::linspace(-domain_radius, domain_radius, block_size + 1);
    let yv = Array::linspace(-domain_radius, domain_radius, block_size + 1);
    let yc = adjacent_mean(&yv, Axis(0));
    return cartesian_product2(xv, yc);
}

pub fn face_centers_y(domain_radius: f64, block_size: usize) -> Array<(f64, f64), Ix2>
{
    use ndarray_ops::*;
    let xv = Array::linspace(-domain_radius, domain_radius, block_size + 1);
    let yv = Array::linspace(-domain_radius, domain_radius, block_size + 1);
    let xc = adjacent_mean(&xv, Axis(0));
    return cartesian_product2(xc, yv);
}




// ============================================================================
pub fn advance(state: SolutionState, block_data: &BlockData, solver: &Solver) -> SolutionState
{
    use ndarray::{s, azip};
    use ndarray_ops::{extend_periodic, map_stencil3};
    use godunov_core::piecewise_linear::plm_gradient3;
    use Direction::{X, Y};

    // ============================================================================
    let a0 = Axis(0);
    let a1 = Axis(1);
    let dx = block_data.face_centers_x[[1,0]].0 - block_data.face_centers_x[[0,0]].0;
    let dy = block_data.face_centers_y[[0,1]].1 - block_data.face_centers_y[[0,0]].1;
    let dt = solver.cfl * f64::min(dx, dy) / solver.maximum_orbital_velocity();
    let two_body_state = solver.orbital_elements.orbital_state_from_time(state.time);

    let intercell_flux = |l: &CellData, r: &CellData, f: &(f64, f64), axis: Direction| -> Conserved
    {
        let cs2 = solver.sound_speed_squared(f, &two_body_state);
        let pl = *l.pc + *l.gradient_field(axis) * 0.5;
        let pr = *r.pc - *r.gradient_field(axis) * 0.5;
        let nu = solver.nu;
        let mu = nu * (l.pc.density() + r.pc.density());
        let tau_x = 0.5 * mu * (l.strain_field(axis, X) + r.strain_field(axis, X));
        let tau_y = 0.5 * mu * (l.strain_field(axis, Y) + r.strain_field(axis, Y));
        riemann_hlle(pl, pr, axis, cs2) + Conserved(0.0, -tau_x, -tau_y)
    };

    // ============================================================================
    let pe = extend_periodic(state.conserved.mapv(Conserved::to_primitive), 2);
    let gx = map_stencil3(&pe, a0, |a, b, c| plm_gradient3(solver.plm, a, b, c));
    let gy = map_stencil3(&pe, a1, |a, b, c| plm_gradient3(solver.plm, a, b, c));
    let xf = &block_data.face_centers_x;
    let yf = &block_data.face_centers_y;

    // ============================================================================
    let cell_data = azip![
        pe.slice(s![1..-1,1..-1]),
        gx.slice(s![ ..  ,1..-1]),
        gy.slice(s![1..-1, ..  ])]
    .apply_collect(CellData::new);

    // ============================================================================
    let fx = azip![
        cell_data.slice(s![..-1,1..-1]),
        cell_data.slice(s![ 1..,1..-1]),
        xf]
    .apply_collect(|l, r, f| intercell_flux(l, r, f, X));

    // ============================================================================
    let fy = azip![
        cell_data.slice(s![1..-1,..-1]),
        cell_data.slice(s![1..-1, 1..]),
        yf]
    .apply_collect(|l, r, f| intercell_flux(l, r, f, Y));

    // ============================================================================
    let du = azip![
        fx.slice(s![..-1,..]),
        fx.slice(s![ 1..,..]),
        fy.slice(s![..,..-1]),
        fy.slice(s![.., 1..])]
    .apply_collect(|&a, &b, &c, &d| ((b - a) / dx + (d - c) / dy) * -dt);

    // ============================================================================
    let sources = azip![
        &state.conserved,
        &block_data.initial_conserved,
        &block_data.cell_centers]
    .apply_collect(|&u, &u0, &(x, y)| solver.source_terms(u, u0, x, y, dt, &two_body_state));

    // ============================================================================
    SolutionState {
        time:      state.time + dt,
        iteration: state.iteration + 1,
        conserved: state.conserved + du + sources.map(|s| s[0] + s[1] + s[2] + s[3] + s[4]),
    }
}
