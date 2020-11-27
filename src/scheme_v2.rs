#![allow(unused)]

// ============================================================================
use std::ops::{Add, Sub, Mul, Div};
use ndarray::{Axis, Array, ArcArray, Ix1, Ix2};
use ndarray_ops::MapArray3by3;
use num::rational::Rational64;
use kepler_two_body::{OrbitalElements, OrbitalState};
use godunov_core::{solution_states, runge_kutta};




// ============================================================================
type NeighborPrimitiveBlock<Primitive> = [[ArcArray<Primitive, Ix2>; 3]; 3];
type BlockState<Conserved> = solution_states::SolutionStateArcArray<Conserved, Ix2>;
pub type BlockIndex = (usize, usize);

#[derive(Copy, Clone)]
pub enum Direction { X, Y }




// ============================================================================
pub trait Conserved: Copy + Send + Sync + Add<Output=Self> + Sub<Output=Self> + Mul<f64, Output=Self> + Div<f64, Output=Self> {}
pub trait Primitive: Copy + Send + Sync {}

impl Conserved for hydro_iso2d::Conserved {}
impl Conserved for hydro_euler::euler_2d::Conserved {}

impl Primitive for hydro_iso2d::Primitive {}
impl Primitive for hydro_euler::euler_2d::Primitive {}




// ============================================================================
pub struct BlockData<C: Conserved>
{
    pub initial_conserved: ArcArray<C, Ix2>,
    pub cell_centers:      ArcArray<(f64, f64), Ix2>,
    pub face_centers_x:    ArcArray<(f64, f64), Ix2>,
    pub face_centers_y:    ArcArray<(f64, f64), Ix2>,
    pub index:             BlockIndex,
}




// ============================================================================
pub struct State<C: Conserved>
{
    pub time: f64,
    pub iteration: Rational64,
    pub conserved: Vec<ArcArray<C, Ix2>>,
}




// ============================================================================
#[derive(Copy, Clone)]
pub struct CellData<'a, P: Primitive>
{
    pc: &'a P,
    gx: &'a P,
    gy: &'a P,
}

impl<'a, P: Primitive> CellData<'_, P>
{
    fn new(pc: &'a P, gx: &'a P, gy: &'a P) -> CellData<'a, P>
    {
        CellData{
            pc: pc,
            gx: gx,
            gy: gy,
        }
    }
}




// ============================================================================
struct SourceTerms
{
    fx1: f64,
    fy1: f64,
    fx2: f64,
    fy2: f64,
    sink_rate1: f64,
    sink_rate2: f64,
    buffer_rate: f64,
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
    pub fn effective_resolution(&self, mesh: &Mesh) -> f64
    {
        f64::min(mesh.cell_spacing_x(), mesh.cell_spacing_y())
    }

    pub fn min_time_step(&self, mesh: &Mesh) -> f64
    {
        self.cfl * self.effective_resolution(mesh) / self.maximum_orbital_velocity()
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

    fn sound_speed_squared(&self, xy: &(f64, f64), state: &OrbitalState) -> f64
    {
        -state.gravitational_potential(xy.0, xy.1, self.softening_length) / self.mach_number.powi(2)
    }

    fn maximum_orbital_velocity(&self) -> f64
    {
        1.0 / self.softening_length.sqrt()
    }

    fn source_terms(&self, two_body_state: &kepler_two_body::OrbitalState, x: f64, y: f64, surface_density: f64) -> SourceTerms
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
pub struct Mesh
{
    pub num_blocks: usize,
    pub block_size: usize,
    pub domain_radius: f64,
}

impl Mesh
{
    pub fn block_length(&self) -> f64
    {
        2.0 * self.domain_radius / (self.num_blocks as f64)
    }

    pub fn block_start(&self, block_index: BlockIndex) -> (f64, f64)
    {
        (
            -self.domain_radius + (block_index.0 as f64) * self.block_length(),
            -self.domain_radius + (block_index.1 as f64) * self.block_length(),
        )
    }

    pub fn block_vertices(&self, block_index: BlockIndex) -> (Array<f64, Ix1>, Array<f64, Ix1>)
    {
        let start = self.block_start(block_index);
        let xv = Array::linspace(start.0, start.0 + self.block_length(), self.block_size + 1);
        let yv = Array::linspace(start.1, start.1 + self.block_length(), self.block_size + 1);
        (xv, yv)
    }

    pub fn cell_centers(&self, block_index: BlockIndex) -> Array<(f64, f64), Ix2>
    {
        use ndarray_ops::{adjacent_mean, cartesian_product2};
        let (xv, yv) = self.block_vertices(block_index);
        let xc = adjacent_mean(&xv, Axis(0));
        let yc = adjacent_mean(&yv, Axis(0));
        return cartesian_product2(xc, yc);
    }

    pub fn face_centers_x(&self, block_index: BlockIndex) -> Array<(f64, f64), Ix2>
    {
        use ndarray_ops::{adjacent_mean, cartesian_product2};
        let (xv, yv) = self.block_vertices(block_index);
        let yc = adjacent_mean(&yv, Axis(0));
        return cartesian_product2(xv, yc);
    }

    pub fn face_centers_y(&self, block_index: BlockIndex) -> Array<(f64, f64), Ix2>
    {
        use ndarray_ops::{adjacent_mean, cartesian_product2};
        let (xv, yv) = self.block_vertices(block_index);
        let xc = adjacent_mean(&xv, Axis(0));
        return cartesian_product2(xc, yv);
    }

    pub fn cell_spacing_x(&self) -> f64
    {
        self.block_length() / (self.block_size as f64)
    }

    pub fn cell_spacing_y(&self) -> f64
    {
        self.block_length() / (self.block_size as f64)
    }

    pub fn total_zones(&self) -> usize
    {
        self.num_blocks * self.num_blocks * self.block_size * self.block_size
    }

    pub fn zones_per_block(&self) -> usize
    {
        self.block_size * self.block_size
    }

    pub fn block_indexes(&self) -> Vec<BlockIndex>
    {
        (0..self.num_blocks)
        .map(|i| (0..self.num_blocks)
        .map(move |j| (i, j)))
        .flatten()
        .collect()
    }

    pub fn neighbor_block_indexes(&self, block_index: BlockIndex) -> [[BlockIndex; 3]; 3]
    {
        let b = self.num_blocks;
        let m = |i, j| (i % b, j % b);
        let (i, j) = block_index;
        [
            [m(i + b - 1, j + b - 1), m(i + b - 1, j + b + 0), m(i + b - 0, j + b + 1)],
            [m(i + b + 0, j + b - 1), m(i + b + 0, j + b + 0), m(i + b + 0, j + b + 1)],
            [m(i + b + 1, j + b - 1), m(i + b + 1, j + b + 0), m(i + b + 1, j + b + 1)],
        ]
    }
}




// ============================================================================
pub trait Hydrodynamics: Sync
{
    type Conserved: Conserved;
    type Primitive: Primitive;

    fn gradient_field<'a>(&self, cell_data: &CellData<'a, Self::Primitive>, axis: Direction) -> &'a Self::Primitive;
    fn strain_field  <'a>(&self, cell_data: &CellData<'a, Self::Primitive>, row: Direction, col: Direction) -> f64;
    fn stress_field  <'a>(&self, cell_data: &CellData<'a, Self::Primitive>, kinematic_viscosity: f64, row: Direction, col: Direction) -> f64;
    fn plm_gradient(&self, theta: f64, a: &Self::Primitive, b: &Self::Primitive, c: &Self::Primitive) -> Self::Primitive;
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
        two_body_state: &OrbitalState) -> [Self::Conserved; 5];

    fn intercell_flux<'a>(
        &self,
        solver: &Solver,
        l: &CellData<'a, Self::Primitive>, 
        r: &CellData<'a, Self::Primitive>, 
        f: &(f64, f64), 
        two_body_state: &kepler_two_body::OrbitalState,
        axis: Direction) -> Self::Conserved;
}

pub struct Isothermal {
}

pub struct Euler {
    gamma_law_index: f64,
}




// ============================================================================
impl Isothermal
{
    pub fn new() -> Self
    {
        Self{}
    }
}

impl Hydrodynamics for Isothermal
{
    type Conserved = hydro_iso2d::Conserved;
    type Primitive = hydro_iso2d::Primitive;

    fn gradient_field<'a>(&self, cell_data: &CellData<'a, Self::Primitive>, axis: Direction) -> &'a Self::Primitive
    {
        match axis
        {
            Direction::X => cell_data.gx,
            Direction::Y => cell_data.gy,
        }
    }

    fn strain_field<'a>(&self, cell_data: &CellData<'a, Self::Primitive>, row: Direction, col: Direction) -> f64
    {
        use Direction::{X, Y};

        match (row, col)
        {
            (X, X) => cell_data.gx.velocity_x() - cell_data.gy.velocity_y(),
            (X, Y) => cell_data.gx.velocity_y() + cell_data.gy.velocity_x(),
            (Y, X) => cell_data.gx.velocity_y() + cell_data.gy.velocity_x(),
            (Y, Y) =>-cell_data.gx.velocity_x() + cell_data.gy.velocity_y(),
        }
    }

    fn stress_field<'a>(&self, cell_data: &CellData<'a, Self::Primitive>, kinematic_viscosity: f64, row: Direction, col: Direction) -> f64
    {
        kinematic_viscosity * cell_data.pc.density() * self.strain_field(cell_data, row, col)
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
        two_body_state: &kepler_two_body::OrbitalState) -> [Self::Conserved; 5]
    {
        let st = solver.source_terms(two_body_state, x, y, conserved.density());
        return [
            hydro_iso2d::Conserved(0.0, st.fx1, st.fy1) * dt,
            hydro_iso2d::Conserved(0.0, st.fx2, st.fy2) * dt,
            conserved * (-st.sink_rate1 * dt),
            conserved * (-st.sink_rate2 * dt),
            (conserved - background_conserved) * (-dt * st.buffer_rate),
        ];
    }

    fn intercell_flux<'a>(
        &self,
        solver: &Solver,
        l: &CellData<'a, hydro_iso2d::Primitive>, 
        r: &CellData<'a, hydro_iso2d::Primitive>, 
        f: &(f64, f64), 
        two_body_state: &kepler_two_body::OrbitalState,
        axis: Direction) -> hydro_iso2d::Conserved
    {
        let cs2 = solver.sound_speed_squared(f, &two_body_state);
        let pl = *l.pc + *self.gradient_field(l, axis) * 0.5;
        let pr = *r.pc - *self.gradient_field(r, axis) * 0.5;
        let nu = solver.nu;
        let tau_x = 0.5 * (self.stress_field(l, nu, axis, Direction::X) + self.stress_field(r, nu, axis, Direction::X));
        let tau_y = 0.5 * (self.stress_field(l, nu, axis, Direction::Y) + self.stress_field(r, nu, axis, Direction::Y));
        let iso2d_axis = match axis {
            Direction::X => hydro_iso2d::Direction::X,
            Direction::Y => hydro_iso2d::Direction::Y,
        };
        hydro_iso2d::riemann_hlle(pl, pr, iso2d_axis, cs2) + hydro_iso2d::Conserved(0.0, -tau_x, -tau_y)
    }
}




// ============================================================================   
impl Hydrodynamics for Euler
{
    type Conserved = hydro_euler::euler_2d::Conserved;
    type Primitive = hydro_euler::euler_2d::Primitive;

    fn gradient_field<'a>(&self, cell_data: &CellData<'a, Self::Primitive>, axis: Direction) -> &'a Self::Primitive
    {
        match axis
        {
            Direction::X => cell_data.gx,
            Direction::Y => cell_data.gy,
        }
    }

    fn strain_field<'a>(&self, cell_data: &CellData<'a, Self::Primitive>, row: Direction, col: Direction) -> f64
    {
        use Direction::{X, Y};

        match (row, col)
        {
            (X, X) => cell_data.gx.velocity_1() - cell_data.gy.velocity_2(),
            (X, Y) => cell_data.gx.velocity_2() + cell_data.gy.velocity_1(),
            (Y, X) => cell_data.gx.velocity_2() + cell_data.gy.velocity_1(),
            (Y, Y) =>-cell_data.gx.velocity_1() + cell_data.gy.velocity_2(),
        }
    }

    fn stress_field<'a>(&self, cell_data: &CellData<'a, Self::Primitive>, kinematic_viscosity: f64, row: Direction, col: Direction) -> f64
    {
        kinematic_viscosity * cell_data.pc.mass_density() * self.strain_field(cell_data, row, col)
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
        two_body_state: &kepler_two_body::OrbitalState) -> [Self::Conserved; 5]
    {
        todo!("We need to add the energy source terms associated with gravitational force below");

        let st = solver.source_terms(two_body_state, x, y, conserved.mass_density());
        return [
            hydro_euler::euler_2d::Conserved(0.0, st.fx1, st.fy1, 0.0) * dt,
            hydro_euler::euler_2d::Conserved(0.0, st.fx2, st.fy2, 0.0) * dt,
            conserved * (-st.sink_rate1 * dt),
            conserved * (-st.sink_rate2 * dt),
            (conserved - background_conserved) * (-dt * st.buffer_rate),
        ];
    }

    fn intercell_flux<'a>(
        &self,
        solver: &Solver,
        l: &CellData<'a, Self::Primitive>, 
        r: &CellData<'a, Self::Primitive>, 
        f: &(f64, f64), 
        two_body_state: &kepler_two_body::OrbitalState,
        axis: Direction) -> Self::Conserved
    {
        todo!()
    }
}




// ============================================================================
fn advance_internal<H: Hydrodynamics>(
    state:      BlockState<H::Conserved>,
    hydro:      &H,
    block_data: &BlockData<H::Conserved>,
    solver:     &Solver,
    mesh:       &Mesh,
    sender:     &crossbeam::Sender<Array<H::Primitive, Ix2>>,
    receiver:   &crossbeam::Receiver<NeighborPrimitiveBlock<H::Primitive>>,
    dt:         f64) -> BlockState<H::Conserved>
{
    // ============================================================================
    use ndarray::{s, azip};
    use ndarray_ops::{map_stencil3};
    use godunov_core::piecewise_linear::plm_gradient3;

    // ============================================================================
    let dx = mesh.cell_spacing_x();
    let dy = mesh.cell_spacing_y();
    let two_body_state = solver.orbital_elements.orbital_state_from_time(state.time);

    let sum_sources = |s: [H::Conserved; 5]| s[0] + s[1] + s[2] + s[3] + s[4];

    // ============================================================================
    sender.send(state.conserved.mapv(|u| hydro.to_primitive(u))).unwrap();

    // ============================================================================
    let sources = azip![
        &state.conserved,
        &block_data.initial_conserved,
        &block_data.cell_centers]
    .apply_collect(|&u, &u0, &(x, y)| sum_sources(hydro.source_terms(&solver, u, u0, x, y, dt, &two_body_state)));

    let pe = ndarray_ops::extend_from_neighbor_arrays_2d(&receiver.recv().unwrap(), 2, 2, 2, 2);
    let gx = map_stencil3(&pe, Axis(0), |a, b, c| hydro.plm_gradient(solver.plm, a, b, c));
    let gy = map_stencil3(&pe, Axis(1), |a, b, c| hydro.plm_gradient(solver.plm, a, b, c));
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
    .apply_collect(|l, r, f| hydro.intercell_flux(&solver, l, r, f, &two_body_state, Direction::X));

    // ============================================================================
    let fy = azip![
        cell_data.slice(s![1..-1,..-1]),
        cell_data.slice(s![1..-1, 1..]),
        yf]
    .apply_collect(|l, r, f| hydro.intercell_flux(&solver, l, r, f, &two_body_state, Direction::Y));

    // ============================================================================
    let du = azip![
        fx.slice(s![..-1,..]),
        fx.slice(s![ 1..,..]),
        fy.slice(s![..,..-1]),
        fy.slice(s![.., 1..])]
    .apply_collect(|&a, &b, &c, &d| ((b - a) / dx + (d - c) / dy) * -dt);

    // ============================================================================
    BlockState::<H::Conserved>{
        time: state.time + dt,
        iteration: state.iteration + 1,
        conserved: state.conserved + du + sources,
    }
}




// ============================================================================
fn advance_internal_rk<H: Hydrodynamics>(
    conserved:  &mut ArcArray<H::Conserved, Ix2>,
    hydro:      &H,
    block_data: &BlockData<H::Conserved>,
    solver:     &Solver,
    mesh:       &Mesh,
    sender:     &crossbeam::Sender<Array<H::Primitive, Ix2>>,
    receiver:   &crossbeam::Receiver<NeighborPrimitiveBlock<H::Primitive>>,
    time:       f64,
    dt:         f64,
    fold:       usize)
{
    use std::convert::TryFrom;

    let update = |state| advance_internal(state, hydro, block_data, solver, mesh, sender, receiver, dt);
    let mut state = BlockState::<H::Conserved> {
        time: time,
        iteration: Rational64::new(0, 1),
        conserved: conserved.clone(),
    };
    let rk_order = runge_kutta::RungeKuttaOrder::try_from(solver.rk_order).unwrap();

    for _ in 0..fold
    {
        state = rk_order.advance(state, update);
    }
    *conserved = state.conserved;
}




// ============================================================================
pub fn advance<H: Hydrodynamics>(
    state: &mut State<H::Conserved>,
    hydro: &H,
    block_data: &Vec<BlockData<H::Conserved>>,
    mesh: &Mesh,
    solver: &Solver,
    dt: f64,
    fold: usize)
{
    crossbeam::scope(|scope|
    {
        use std::collections::HashMap;

        let time = state.time;
        let mut receivers       = Vec::new();
        let mut senders         = Vec::new();
        let mut block_primitive = HashMap::new();
        let hydro = hydro.clone();

        for (u, b) in state.conserved.iter_mut().zip(block_data)
        {
            let (their_s, my_r) = crossbeam::channel::unbounded();
            let (my_s, their_r) = crossbeam::channel::unbounded();

            senders.push(my_s);
            receivers.push(my_r);

            scope.spawn(move |_| advance_internal_rk(u, hydro, b, solver, mesh, &their_s, &their_r, time, dt, fold));
        }

        for _ in 0..fold
        {
            for _ in 0..solver.rk_order
            {
                for (block_data, r) in block_data.iter().zip(receivers.iter())
                {
                    block_primitive.insert(block_data.index, r.recv().unwrap().to_shared());
                }

                for (block_data, s) in block_data.iter().zip(senders.iter())
                {
                    s.send(mesh.neighbor_block_indexes(block_data.index).map(|i| block_primitive
                        .get(i)
                        .unwrap()
                        .clone()))
                    .unwrap();
                }
            }

            state.iteration += 1;
            state.time += dt;
        }
    }).unwrap();
}
