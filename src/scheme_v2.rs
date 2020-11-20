#![allow(unused)]

// ============================================================================
use ndarray::{Axis, Array, ArcArray, Ix1, Ix2};
use ndarray_ops::MapArray3by3;
use num::rational::Rational64;
use kepler_two_body::{OrbitalElements, OrbitalState};
use godunov_core::{solution_states, runge_kutta};
use std::ops::{Add, Sub, Mul, Div};




// ============================================================================
type NeighborPrimitiveBlock<Primitive> = [[ArcArray<Primitive, Ix2>; 3]; 3];
type BlockState<Conserved> = solution_states::SolutionStateArray<Conserved, Ix2>;
pub type BlockIndex = (usize, usize);




// ============================================================================
pub trait Conserved: Add<Output=Self> + Sub<Output=Self> + Mul<f64, Output=Self> + Div<f64, Output=Self> + Copy {}
pub trait Primitive: Copy {}

impl Conserved for hydro_iso2d::Conserved {}
impl Conserved for hydro_euler::euler_2d::Conserved {}

impl Primitive for hydro_iso2d::Primitive {}
impl Primitive for hydro_euler::euler_2d::Primitive {}




// ============================================================================
pub struct BlockData<C: Conserved>
{
    pub initial_conserved: Array<C, Ix2>,
    pub cell_centers:      Array<(f64, f64), Ix2>,
    pub face_centers_x:    Array<(f64, f64), Ix2>,
    pub face_centers_y:    Array<(f64, f64), Ix2>,
    pub index:             BlockIndex,
}




// ============================================================================
pub struct State<C: Conserved>
{
    pub time: f64,
    pub iteration: Rational64,
    pub conserved: Vec<Array<C, Ix2>>,
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

    pub fn effective_resolution(&self, mesh: &Mesh) -> f64
    {
        f64::min(mesh.cell_spacing_x(), mesh.cell_spacing_y())
    }

    pub fn min_time_step(&self, mesh: &Mesh) -> f64
    {
        self.cfl * self.effective_resolution(mesh) / self.maximum_orbital_velocity()
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
pub trait Hydrodynamics
{
    type Conserved: Conserved;
    type Primitive: Primitive;
    type Direction;

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

    fn intercell_flux_x<'a>(
        &self,
        solver: &Solver,
        l: &CellData<'a, Self::Primitive>, 
        r: &CellData<'a, Self::Primitive>, 
        f: &(f64, f64), 
        two_body_state: &kepler_two_body::OrbitalState) -> Self::Conserved;

    fn intercell_flux_y<'a>(
        &self,
        solver: &Solver,
        l: &CellData<'a, Self::Primitive>, 
        r: &CellData<'a, Self::Primitive>, 
        f: &(f64, f64), 
        two_body_state: &kepler_two_body::OrbitalState) -> Self::Conserved;

    fn plm_gradient(&self, theta: f64, a: &Self::Primitive, b: &Self::Primitive, c: &Self::Primitive) -> Self::Primitive;

    fn to_primitive(&self, u: Self::Conserved) -> Self::Primitive;
}




// ============================================================================
struct Isothermal
{
}

impl Hydrodynamics for Isothermal
{
    type Conserved = hydro_iso2d::Conserved;
    type Primitive = hydro_iso2d::Primitive;
    type Direction = hydro_iso2d::Direction;

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
            hydro_iso2d::Conserved(0.0, fx1, fy1) * dt,
            hydro_iso2d::Conserved(0.0, fx2, fy2) * dt,
            conserved * (-sink_rate1 * dt),
            conserved * (-sink_rate2 * dt),
            (conserved - background_conserved) * (-dt * buffer_rate),
        ];
    }

    fn intercell_flux_x<'a>(
        &self,
        solver: &Solver,
        l: &CellData<'a, Self::Primitive>, 
        r: &CellData<'a, Self::Primitive>, 
        f: &(f64, f64), 
        two_body_state: &kepler_two_body::OrbitalState) -> Self::Conserved 
    {
        self.intercell_flux(solver, l, r, f, two_body_state, Self::Direction::X)
    }
    
    fn intercell_flux_y<'a>(
        &self,
        solver: &Solver,
        l: &CellData<'a, Self::Primitive>, 
        r: &CellData<'a, Self::Primitive>, 
        f: &(f64, f64), 
        two_body_state: &kepler_two_body::OrbitalState) -> Self::Conserved
    {
        self.intercell_flux(solver, l, r, f, two_body_state, Self::Direction::Y)
    }

    fn plm_gradient(&self, theta: f64, a: &Self::Primitive, b: &Self::Primitive, c: &Self::Primitive) -> Self::Primitive
    {
        godunov_core::piecewise_linear::plm_gradient3(theta, a, b, c)
    }

    fn to_primitive(&self, u: Self::Conserved) -> Self::Primitive
    {
        u.to_primitive()
    }
}




// ============================================================================
impl Isothermal
{
    fn intercell_flux<'a>(
        &self,
        solver: &Solver,
        l: &CellData<'a, hydro_iso2d::Primitive>, 
        r: &CellData<'a, hydro_iso2d::Primitive>, 
        f: &(f64, f64), 
        two_body_state: &kepler_two_body::OrbitalState,
        axis: hydro_iso2d::Direction) -> hydro_iso2d::Conserved
    {
        let cs2 = solver.sound_speed_squared(f, &two_body_state);
        let pl = *l.pc + *self.gradient_field(l, axis) * 0.5;
        let pr = *r.pc - *self.gradient_field(r, axis) * 0.5;
        let nu = solver.nu;
        let tau_x = 0.5 * (self.stress_field(l, nu, axis, hydro_iso2d::Direction::X) + self.stress_field(r, nu, axis, hydro_iso2d::Direction::X));
        let tau_y = 0.5 * (self.stress_field(l, nu, axis, hydro_iso2d::Direction::Y) + self.stress_field(r, nu, axis, hydro_iso2d::Direction::Y));
        hydro_iso2d::riemann_hlle(pl, pr, axis, cs2) + hydro_iso2d::Conserved(0.0, -tau_x, -tau_y)
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
    .apply_collect(|l, r, f| hydro.intercell_flux_x(&solver, l, r, f, &two_body_state));

    // ============================================================================
    let fy = azip![
        cell_data.slice(s![1..-1,..-1]),
        cell_data.slice(s![1..-1, 1..]),
        yf]
    .apply_collect(|l, r, f| hydro.intercell_flux_y(&solver, l, r, f, &two_body_state));

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
    conserved:  &mut Array<H::Conserved, Ix2>,
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
// pub fn advance<H: Hydrodynamics>(
//     state: &mut State<H::Conserved>,
//     hydro: &H,
//     block_data: &Vec<BlockData<H::Conserved>>,
//     mesh: &Mesh,
//     solver: &Solver,
//     dt: f64,
//     fold: usize)
// {
    // crossbeam::scope(|scope|
    // {
    //     use std::collections::HashMap;

    //     let time = state.time;
    //     let mut receivers       = Vec::new();
    //     let mut senders         = Vec::new();
    //     let mut block_primitive = HashMap::new();

    //     for (u, b) in state.conserved.iter_mut().zip(block_data)
    //     {
    //         let (their_s, my_r) = crossbeam::channel::unbounded();
    //         let (my_s, their_r) = crossbeam::channel::unbounded();

    //         senders.push(my_s);
    //         receivers.push(my_r);

    //         scope.spawn(move |_| advance_internal_rk(u, b, solver, mesh, &their_s, &their_r, time, dt, fold));
    //     }

    //     for _ in 0..fold
    //     {
    //         for _ in 0..solver.rk_order
    //         {
    //             for (block_data, r) in block_data.iter().zip(receivers.iter())
    //             {
    //                 block_primitive.insert(block_data.index, r.recv().unwrap().to_shared());
    //             }

    //             for (block_data, s) in block_data.iter().zip(senders.iter())
    //             {
    //                 s.send(mesh.neighbor_block_indexes(block_data.index).map(|i| block_primitive
    //                     .get(i)
    //                     .unwrap()
    //                     .clone()))
    //                 .unwrap();
    //             }
    //         }

    //         state.iteration += 1;
    //         state.time += dt;
    //     }
    // }).unwrap();
// }




// ============================================================================
// use hydro_euler::euler_2d::Primitive as Primitive_Euler;
// use hydro_euler::euler_2d::Conserved as Conserved_Euler;
// use hydro_euler::euler_2d::riemann_hlle as riemann_euler;
// struct TwoDimensionalHydrodynamicsWithEnergy;
    
// impl Hydrodynamics for TwoDimensionalHydrodynamicsWithEnergy
// {
//     type Conserved = Conserved_Euler;
//     type Primitive = Primitive_Euler;
//     type Direction = hydro_euler::geometry::Direction;

//     fn gradient_field<'a>(&self, cell_data: &CellData<'a, Self::Primitive>, axis: Self::Direction) -> &'a Self::Primitive
//     {
//         use hydro_euler::geometry::Direction::{X, Y, Z};

//         match axis
//         {
//             X => cell_data.gx,
//             Y => cell_data.gy,
//             Z => panic!(),
//         }
//     }

//     fn strain_field<'a>(&self, cell_data: &CellData<'a, Self::Primitive>, row: Self::Direction, col: Self::Direction) -> f64
//     {
//         use hydro_euler::geometry::Direction::{X, Y};

//         match (row, col)
//         {
//             (X, X) => cell_data.gx.velocity_1() - cell_data.gy.velocity_2(),
//             (X, Y) => cell_data.gx.velocity_2() + cell_data.gy.velocity_1(),
//             (Y, X) => cell_data.gx.velocity_2() + cell_data.gy.velocity_1(),
//             (Y, Y) =>-cell_data.gx.velocity_1() + cell_data.gy.velocity_2(),
//             (_, _) => panic!(),     
//         }
//     }

//     fn stress_field<'a>(&self, cell_data: &CellData<'a, Self::Primitive>, kinematic_viscosity: f64, row: Self::Direction, col: Self::Direction) -> f64
//     {
//         kinematic_viscosity * cell_data.pc.mass_density() * self.strain_field(cell_data, row, col)
//     }

//     fn source_terms(
//         &self,
//         solver: &Solver,
//         conserved: Self::Conserved,
//         background_conserved: Self::Conserved,
//         x: f64,
//         y: f64,
//         dt: f64,
//         two_body_state: &kepler_two_body::OrbitalState) -> [Self::Conserved; 5]
//     {
//         let p1 = two_body_state.0;
//         let p2 = two_body_state.1;

//         let [ax1, ay1] = p1.gravitational_acceleration(x, y, solver.softening_length);
//         let [ax2, ay2] = p2.gravitational_acceleration(x, y, solver.softening_length);

//         let rho = conserved.mass_density();
//         let fx1 = rho * ax1;
//         let fy1 = rho * ay1;
//         let fx2 = rho * ax2;
//         let fy2 = rho * ay2;

//         let x1 = p1.position_x();
//         let y1 = p1.position_y();
//         let x2 = p2.position_x();
//         let y2 = p2.position_y();

//         let sink_rate1 = solver.sink_kernel(x - x1, y - y1);
//         let sink_rate2 = solver.sink_kernel(x - x2, y - y2);

//         let r = (x * x + y * y).sqrt();
//         let y = (r - solver.domain_radius) / solver.buffer_scale;
//         let omega_outer = (two_body_state.total_mass() / solver.domain_radius.powi(3)).sqrt();
//         let buffer_rate = 0.5 * solver.buffer_rate * (1.0 + f64::tanh(y)) * omega_outer;

//         return [
//             Conserved_Euler(0.0, fx1, fy1, 0.0) * dt,
//             Conserved_Euler(0.0, fx2, fy2, 0.0) * dt,
//             conserved * (-sink_rate1 * dt),
//             conserved * (-sink_rate2 * dt),
//             (conserved - background_conserved) * (-dt * buffer_rate),
//         ];
//     }

//     fn intercell_flux<'a>(
//         &self,
//         solver: &Solver,
//         l: &CellData<'a, Self::Primitive>, 
//         r: &CellData<'a, Self::Primitive>, 
//         f: &(f64, f64), 
//         two_body_state: &kepler_two_body::OrbitalState,
//         axis: Self::Direction) -> Self::Conserved
//     {
//         let cs2 = solver.sound_speed_squared(f, &two_body_state);
//         let pl = *l.pc + *self.gradient_field(l, axis) * 0.5;
//         let pr = *r.pc - *self.gradient_field(r, axis) * 0.5;
//         let nu = solver.nu;
//         let tau_x = 0.5 * (self.stress_field(l, nu, axis, Self::Direction::X) + self.stress_field(r, nu, axis, Self::Direction::X));
//         let tau_y = 0.5 * (self.stress_field(l, nu, axis, Self::Direction::Y) + self.stress_field(r, nu, axis, Self::Direction::Y));
//         hydro_euler::euler_2d::riemann_hlle(pl, pr, axis, cs2) + Conserved_Euler(0.0, -tau_x, -tau_y, 0.0)
//     }

//     fn plm_gradient(&self, theta: f64, a: &Self::Primitive, b: &Self::Primitive, c: &Self::Primitive) -> Self::Primitive
//     {
//         godunov_core::piecewise_linear::plm_gradient4(theta, a, b, c)
//     }
// }
