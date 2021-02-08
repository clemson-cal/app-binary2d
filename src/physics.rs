use godunov_core::runge_kutta;

use kepler_two_body::{
    OrbitalElements,
    OrbitalState,
};

use crate::mesh::{
    Mesh,
};

use crate::traits::{
    Arithmetic,
    Conserved,
    Hydrodynamics,
    ItemizeData,
    Primitive,
    Zeros,
};




// ============================================================================
#[derive(Copy, Clone)]
pub struct CellData<'a, P: Primitive>
{
    pub pc: &'a P,
    pub gx: &'a P,
    pub gy: &'a P,
}




// ============================================================================
#[derive(Copy, Clone)]
pub enum Direction {
    X,
    Y,
}




// ============================================================================
#[derive(Clone, Copy, hdf5::H5Type)]
#[repr(C)]
pub struct ItemizedChange<C: ItemizeData>
{
    pub sink1:   C,
    pub sink2:   C,
    pub grav1:   C,
    pub grav2:   C,
    pub buffer:  C,
    pub cooling: C,
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
    pub lambda: f64,
    pub plm: f64,
    pub rk_order: i64,
    pub sink_radius: f64,
    pub sink_rate: f64,
    pub softening_length: f64,
    pub force_flux_comm: bool,
    pub orbital_elements: OrbitalElements,
}




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
#[derive(Clone, Copy)]
pub struct Isothermal {
}




// ============================================================================
#[derive(Clone, Copy)]
pub struct Euler {
    pub gamma_law_index: f64,
}




// ============================================================================
impl<'a, P: Primitive> CellData<'_, P>
{
    pub fn new(pc: &'a P, gx: &'a P, gy: &'a P) -> CellData<'a, P>
    {
        CellData{
            pc: pc,
            gx: gx,
            gy: gy,
        }
    }

    pub fn stress_field(&self, nu: f64, lambda: f64, dx: f64, dy: f64, row: Direction, col: Direction) -> f64
    {
        use Direction::{X, Y};

        let shear_stress = match (row, col)
        {
            (X, X) => 4.0 / 3.0 * self.gx.velocity_x() / dx - 2.0 / 3.0 * self.gy.velocity_y() / dy,
            (X, Y) => 1.0 / 1.0 * self.gx.velocity_y() / dx + 1.0 / 1.0 * self.gy.velocity_x() / dy,
            (Y, X) => 1.0 / 1.0 * self.gx.velocity_y() / dx + 1.0 / 1.0 * self.gy.velocity_x() / dy,
            (Y, Y) =>-2.0 / 3.0 * self.gx.velocity_x() / dx + 4.0 / 3.0 * self.gy.velocity_y() / dy,
        };

        let bulk_stress = match (row, col)
        {
            (X, X) => self.gx.velocity_x() / dx + self.gy.velocity_y() / dy,
            (X, Y) => 0.0,
            (Y, X) => 0.0,
            (Y, Y) => self.gx.velocity_x() / dx + self.gy.velocity_y() / dy,
        };

        self.pc.mass_density() * (nu * shear_stress + lambda * bulk_stress)
    }

    pub fn gradient_field(&self, axis: Direction) -> &P
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
impl<C: ItemizeData> ItemizedChange<C>
{
    pub fn zeros() -> Self
    {
        Self{
            sink1:   C::zeros(),
            sink2:   C::zeros(),
            grav1:   C::zeros(),
            grav2:   C::zeros(),
            buffer:  C::zeros(),
            cooling: C::zeros(),
        }
    }

    pub fn total(&self) -> C
    {
        self.sink1 + self.sink2 + self.grav1 + self.grav2 + self.buffer + self.cooling
    }

    pub fn add_mut(&mut self, s0: &Self)
    {
        self.sink1   =  self.sink1   + s0.sink1;
        self.sink2   =  self.sink2   + s0.sink2;
        self.grav1   =  self.grav1   + s0.grav1;
        self.grav2   =  self.grav2   + s0.grav2;
        self.buffer  =  self.buffer  + s0.buffer;
        self.cooling =  self.cooling + s0.cooling;
    }

    pub fn mul_mut(&mut self, s: f64)
    {
        self.sink1   =  self.sink1   * s;
        self.sink2   =  self.sink2   * s;
        self.grav1   =  self.grav1   * s;
        self.grav2   =  self.grav2   * s;
        self.buffer  =  self.buffer  * s;
        self.cooling =  self.cooling * s;
    }

    pub fn add(&self, s0: &Self) -> Self
    {
        let mut result = self.clone();
        result.add_mut(s0);
        return result;
    }

    pub fn mul(&self, s: f64) -> Self
    {
        let mut result = self.clone();
        result.mul_mut(s);
        return result;
    }
}




// ============================================================================
impl<C: ItemizeData> ItemizedChange<C> where C: Conserved
{
    fn pert1(time: f64, delta: (f64, f64, f64), elements: OrbitalElements) -> OrbitalElements
    {
        let (dm, dpx, dpy) = delta;
        elements.perturb(time, -dm, 0.0, -dpx, 0.0, -dpy, 0.0).unwrap() - elements
    }
    fn pert2(time: f64, delta: (f64, f64, f64), elements: OrbitalElements) -> OrbitalElements
    {
        let (dm, dpx, dpy) = delta;
        elements.perturb(time, 0.0, -dm, 0.0, -dpx, 0.0, -dpy).unwrap() - elements
    }
    pub fn perturbation(&self, time: f64, elements: OrbitalElements) -> ItemizedChange<OrbitalElements>
    {
        ItemizedChange{
            sink1:    Self::pert1(time, self.sink1.mass_and_momentum(), elements),
            sink2:    Self::pert2(time, self.sink2.mass_and_momentum(), elements),
            grav1:    Self::pert1(time, self.grav1.mass_and_momentum(), elements),
            grav2:    Self::pert2(time, self.grav2.mass_and_momentum(), elements),
            buffer:   OrbitalElements::zeros(),
            cooling:  OrbitalElements::zeros(),
        }
    }
}




// ============================================================================
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




// ============================================================================
impl Isothermal
{
    pub fn new() -> Self
    {
        Self{}
    }
}




// ============================================================================
// TODO: Implement density floor feature here.
impl Hydrodynamics for Isothermal
{
    type Conserved = hydro_iso2d::Conserved;
    type Primitive = hydro_iso2d::Primitive;

    fn gamma_law_index(&self) -> f64 {
        1.0
    }

    fn plm_gradient(&self, theta: f64, a: &Self::Primitive, b: &Self::Primitive, c: &Self::Primitive) -> Self::Primitive
    {
        godunov_core::piecewise_linear::plm_gradient3(theta, a, b, c)
    }

    fn to_primitive(&self, u: Self::Conserved) -> Self::Primitive
    {
        u.to_primitive()
    }

    fn to_conserved(&self, p: Self::Primitive) -> Self::Conserved
    {
        p.to_conserved()
    }

    fn source_terms(
        &self,
        solver: &Solver,
        conserved: Self::Conserved,
        background_conserved: Self::Conserved,
        x: f64,
        y: f64,
        dt: f64,
        two_body_state: &kepler_two_body::OrbitalState) -> ItemizedChange<Self::Conserved>
    {
        if conserved.density() < 0.0 { panic!("Density is negative!") }
        let st = solver.source_terms(two_body_state, x, y, conserved.density());
        
        ItemizedChange{
            grav1:   hydro_iso2d::Conserved(0.0, st.fx1, st.fy1) * dt,
            grav2:   hydro_iso2d::Conserved(0.0, st.fx2, st.fy2) * dt,
            sink1:   conserved * (-st.sink_rate1 * dt),
            sink2:   conserved * (-st.sink_rate2 * dt),
            buffer: (conserved - background_conserved) * (-dt * st.buffer_rate),
            cooling: Self::Conserved::zeros(),
        }
    }

    fn intercell_flux<'a>(
        &self,
        solver: &Solver,
        l: &CellData<'a, hydro_iso2d::Primitive>,
        r: &CellData<'a, hydro_iso2d::Primitive>,
        f: &(f64, f64),
        dx: f64,
        dy: f64,
        two_body_state: &kepler_two_body::OrbitalState,
        axis: Direction) -> hydro_iso2d::Conserved
    {
        let cs2 = solver.sound_speed_squared(f, &two_body_state);
        let pl  = *l.pc + *l.gradient_field(axis) * 0.5;
        let pr  = *r.pc - *r.gradient_field(axis) * 0.5;
        let nu  = solver.nu;
        let lam = solver.lambda;
        let tau_x = 0.5 * (l.stress_field(nu, lam, dx, dy, axis, Direction::X) + r.stress_field(nu, lam, dx, dy, axis, Direction::X));
        let tau_y = 0.5 * (l.stress_field(nu, lam, dx, dy, axis, Direction::Y) + r.stress_field(nu, lam, dx, dy, axis, Direction::Y));
        let iso2d_axis = match axis {
            Direction::X => hydro_iso2d::Direction::X,
            Direction::Y => hydro_iso2d::Direction::Y,
        };
        hydro_iso2d::riemann_hlle(pl, pr, iso2d_axis, cs2) + hydro_iso2d::Conserved(0.0, -tau_x, -tau_y)
    }
}




// ============================================================================
impl Euler
{
    pub fn new() -> Self
    {
        Self{gamma_law_index: 5.0 / 3.0}
    }
}




// ============================================================================
impl Hydrodynamics for Euler
{
    type Conserved = hydro_euler::euler_2d::Conserved;
    type Primitive = hydro_euler::euler_2d::Primitive;

    fn gamma_law_index(&self) -> f64 {
        self.gamma_law_index
    }

    fn plm_gradient(&self, theta: f64, a: &Self::Primitive, b: &Self::Primitive, c: &Self::Primitive) -> Self::Primitive
    {
        godunov_core::piecewise_linear::plm_gradient4(theta, a, b, c)
    }

    fn to_primitive(&self, conserved: Self::Conserved) -> Self::Primitive
    {
        conserved.to_primitive(self.gamma_law_index)
    }

    fn to_conserved(&self, p: Self::Primitive) -> Self::Conserved
    {
        p.to_conserved(self.gamma_law_index)
    }

    fn source_terms(
        &self,
        solver: &Solver,
        conserved: Self::Conserved,
        background_conserved: Self::Conserved,
        x: f64,
        y: f64,
        dt: f64,
        two_body_state: &kepler_two_body::OrbitalState) -> ItemizedChange<Self::Conserved>
    {
        let st        = solver.source_terms(two_body_state, x, y, conserved.mass_density());
        let primitive = conserved.to_primitive(self.gamma_law_index);
        let vx        = primitive.velocity_1();
        let vy        = primitive.velocity_2();

        ItemizedChange{
            grav1:   hydro_euler::euler_2d::Conserved(0.0, st.fx1, st.fy1, st.fx1 * vx + st.fy1 * vy) * dt,
            grav2:   hydro_euler::euler_2d::Conserved(0.0, st.fx2, st.fy2, st.fx2 * vx + st.fy2 * vy) * dt,
            sink1:   conserved * (-st.sink_rate1 * dt),
            sink2:   conserved * (-st.sink_rate2 * dt),
            buffer: (conserved - background_conserved) * (-dt * st.buffer_rate),
            cooling: Self::Conserved::zeros(),
        }
    }

    fn intercell_flux<'a>(
        &self,
        solver: &Solver,
        l: &CellData<'a, Self::Primitive>,
        r: &CellData<'a, Self::Primitive>,
        _: &(f64, f64),
        dx: f64,
        dy: f64,
        _: &kepler_two_body::OrbitalState,
        axis: Direction) -> Self::Conserved
    {
        let pl = *l.pc + *l.gradient_field(axis) * 0.5;
        let pr = *r.pc - *r.gradient_field(axis) * 0.5;

        let nu    = solver.nu;
        let lam   = solver.lambda;
        let tau_x = 0.5 * (l.stress_field(nu, lam, dx, dy, axis, Direction::X) + r.stress_field(nu, lam, dx, dy, axis, Direction::X));
        let tau_y = 0.5 * (l.stress_field(nu, lam, dx, dy, axis, Direction::Y) + r.stress_field(nu, lam, dx, dy, axis, Direction::Y));
        let vx = 0.5 * (l.pc.velocity_x() + r.pc.velocity_x());
        let vy = 0.5 * (l.pc.velocity_y() + r.pc.velocity_y());
        let viscous_flux = hydro_euler::euler_2d::Conserved(0.0, -tau_x, -tau_y, -(tau_x * vx + tau_y * vy));

        let euler_axis = match axis {
            Direction::X => hydro_euler::geometry::Direction::X,
            Direction::Y => hydro_euler::geometry::Direction::Y,
        };
        hydro_euler::euler_2d::riemann_hlle(pl, pr, euler_axis, self.gamma_law_index) + viscous_flux
    }
}




// ============================================================================
impl Arithmetic for hydro_iso2d::Conserved {}
impl Arithmetic for hydro_euler::euler_2d::Conserved {}
impl Arithmetic for kepler_two_body::OrbitalElements {}




// ============================================================================
impl ItemizeData for hydro_iso2d::Conserved {}
impl ItemizeData for hydro_euler::euler_2d::Conserved {}
impl ItemizeData for kepler_two_body::OrbitalElements {}




// ============================================================================
impl Zeros for hydro_iso2d::Conserved
{
    fn zeros() -> Self
    {
        Self(0.0, 0.0, 0.0)
    }
}

impl Zeros for hydro_euler::euler_2d::Conserved
{
    fn zeros() -> Self
    {
        Self(0.0, 0.0, 0.0, 0.0)
    }
}

impl Zeros for kepler_two_body::OrbitalElements {
    fn zeros() -> Self
    {
        Self(0.0, 0.0, 0.0, 0.0)
    }
}




// ============================================================================
impl Conserved for hydro_iso2d::Conserved {
    fn mass_and_momentum(&self) -> (f64, f64, f64) {
        (self.0, self.1, self.2)
    }
}

impl Conserved for hydro_euler::euler_2d::Conserved {
    fn mass_and_momentum(&self) -> (f64, f64, f64) {
        (self.0, self.1, self.2)
    }
}




// ============================================================================
impl Primitive for hydro_iso2d::Primitive
{
    fn velocity_x(self) -> f64   { self.velocity_x() }
    fn velocity_y(self) -> f64   { self.velocity_y() }
    fn mass_density(self) -> f64 { self.density() }
}

impl Primitive for hydro_euler::euler_2d::Primitive
{
    fn velocity_x(self) -> f64 { self.velocity(hydro_euler::geometry::Direction::X) }
    fn velocity_y(self) -> f64 { self.velocity(hydro_euler::geometry::Direction::Y) }
    fn mass_density(self) -> f64 { self.mass_density() }
}
