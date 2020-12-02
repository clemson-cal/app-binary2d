// ============================================================================
use std::ops::{Add, Sub, Mul, Div};
use ndarray::{Axis, Array, ArcArray, Ix1, Ix2};
use ndarray_ops::MapArray3by3;
use num::rational::Rational64;
use kepler_two_body::{OrbitalElements, OrbitalState};
use godunov_core::{solution_states, runge_kutta};
use futures::future::{Future};
use std::collections::HashMap;




// ============================================================================
type NeighborPrimitiveBlock<Primitive> = [[ArcArray<Primitive, Ix2>; 3]; 3];
type BlockState<Conserved> = solution_states::SolutionStateArcArray<Conserved, Ix2>;
pub type BlockIndex = (usize, usize);

#[derive(Copy, Clone)]
pub enum Direction { X, Y }




// ============================================================================
pub trait Arithmetic: Add<Output=Self> + Sub<Output=Self> + Mul<f64, Output=Self> + Div<f64, Output=Self> + Sized {}
pub trait Conserved: Clone + Copy + Send + Sync + Arithmetic {}
pub trait Primitive: Clone + Copy + Send + Sync
{
    fn velocity_x(self) -> f64;
    fn velocity_y(self) -> f64;
    fn mass_density(self) -> f64;
}

impl Arithmetic for hydro_iso2d::Conserved {}
impl Arithmetic for hydro_euler::euler_2d::Conserved {}

impl Conserved for hydro_iso2d::Conserved {}
impl Conserved for hydro_euler::euler_2d::Conserved {}

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




// ============================================================================
#[derive(Clone)]
pub struct BlockData<C: Conserved>
{
    pub initial_conserved: ArcArray<C, Ix2>,
    pub cell_centers:      ArcArray<(f64, f64), Ix2>,
    pub face_centers_x:    ArcArray<(f64, f64), Ix2>,
    pub face_centers_y:    ArcArray<(f64, f64), Ix2>,
    pub index:             BlockIndex,
}




// ============================================================================
#[derive(Clone)]
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

    fn stress_field(&self, kinematic_viscosity: f64, dimensionality: i64, row: Direction, col: Direction) -> f64
    {
        use Direction::{X, Y};

        let stress = if dimensionality == 2 {
            // This form of the stress tensor comes from Eqn. 7 in Farris+
            // (2014). Formally it corresponds a "true" dimensionality of 2.
            match (row, col)
            {
                (X, X) =>  self.gx.velocity_x() - self.gy.velocity_y(),
                (X, Y) =>  self.gx.velocity_y() + self.gy.velocity_x(),
                (Y, X) =>  self.gx.velocity_y() + self.gy.velocity_x(),
                (Y, Y) => -self.gx.velocity_x() + self.gy.velocity_y(),
            }
        } else if dimensionality == 3 {
            // This form of the stress tensor is the correct one for vertically
            // averaged hydrodynamics, when the bulk viscosity is equal to zero.
            match (row, col)
            {
                (X, X) => 4.0 / 3.0 * self.gx.velocity_x() - 2.0 / 3.0 * self.gy.velocity_y(),
                (X, Y) => 1.0 / 1.0 * self.gx.velocity_y() + 1.0 / 1.0 * self.gy.velocity_x(),
                (Y, X) => 1.0 / 1.0 * self.gx.velocity_y() + 1.0 / 1.0 * self.gy.velocity_x(),
                (Y, Y) =>-2.0 / 3.0 * self.gx.velocity_x() + 4.0 / 3.0 * self.gy.velocity_y(),
            }
        } else {
            panic!("The true dimension must be 2 or 3")
        };

        kinematic_viscosity * self.pc.mass_density() * stress
    }

    fn gradient_field(&self, axis: Direction) -> &P
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
    pub orbital_elements: OrbitalElements,
}

impl Solver
{
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
#[derive(Clone)]
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
pub trait Hydrodynamics: Copy + Send
{
    type Conserved: Conserved;
    type Primitive: Primitive;

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

#[derive(Clone, Copy)]
pub struct Isothermal {
}

#[derive(Clone, Copy)]
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
        let pl  = *l.pc + *l.gradient_field(axis) * 0.5;
        let pr  = *r.pc - *r.gradient_field(axis) * 0.5;
        let nu  = solver.nu;
        let dim = solver.stress_dim;
        let tau_x = 0.5 * (l.stress_field(nu, dim, axis, Direction::X) + r.stress_field(nu, dim, axis, Direction::X));
        let tau_y = 0.5 * (l.stress_field(nu, dim, axis, Direction::Y) + r.stress_field(nu, dim, axis, Direction::Y));
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

impl Hydrodynamics for Euler
{
    type Conserved = hydro_euler::euler_2d::Conserved;
    type Primitive = hydro_euler::euler_2d::Primitive;

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
        _solver: &Solver,
        _conserved: Self::Conserved,
        _background_conserved: Self::Conserved,
        _x: f64,
        _y: f64,
        _dt: f64,
        _two_body_state: &kepler_two_body::OrbitalState) -> [Self::Conserved; 5]
    {
        todo!("Energy source term due to gravity")
    }

    fn intercell_flux<'a>(
        &self,
        _solver: &Solver,
        _l: &CellData<'a, Self::Primitive>, 
        _r: &CellData<'a, Self::Primitive>, 
        _f: &(f64, f64), 
        _two_body_state: &kepler_two_body::OrbitalState,
        _axis: Direction) -> Self::Conserved
    {
        todo!("Intercell flux function for Euler; viscous energy flux")
    }
}




// ============================================================================
async fn join_3by3<T: Clone + Future>(a: [[&T; 3]; 3]) -> [[T::Output; 3]; 3]
{
    [
        [a[0][0].clone().await, a[0][1].clone().await, a[0][2].clone().await],
        [a[1][0].clone().await, a[1][1].clone().await, a[1][2].clone().await],
        [a[2][0].clone().await, a[2][1].clone().await, a[2][2].clone().await],
    ]
}




// ============================================================================
async fn advance_tokio_rk<H: 'static + Hydrodynamics>(
    state: State<H::Conserved>,
    hydro: H,
    block_data: &Vec<BlockData<H::Conserved>>,
    mesh: &Mesh,
    solver: &Solver,
    dt: f64,
    runtime: &tokio::runtime::Runtime) -> State<H::Conserved>
{
    use futures::future::{FutureExt, join_all};

    let scheme = UpdateScheme::new(hydro);
    let time = state.time;

    let pc_map: HashMap<_, _> = state.conserved.iter().zip(block_data).map(|(uc, block)|
    {
        let uc = uc.clone();
        let primitive = async move {
            scheme.compute_block_primitive(uc).to_shared()
        };
        (block.index, runtime.spawn(primitive).map(|p| p.unwrap()).shared())
    }).collect();

    let flux_map: HashMap<_, _> = block_data.iter().map(|block|
    {
        let solver      = solver.clone();
        let mesh        = mesh.clone();
        let pc_map      = pc_map.clone();
        let block       = block.clone();
        let block_index = block.index;

        let flux = async move {
            let pn = join_3by3(mesh.neighbor_block_indexes(block_index).map(|i| &pc_map[i])).await;
            let pe = ndarray_ops::extend_from_neighbor_arrays_2d(&pn, 2, 2, 2, 2);
            let (fx, fy) = scheme.compute_block_fluxes(&pe, &block, &solver, time);
            (fx.to_shared(), fy.to_shared())
        };
        (block_index, runtime.spawn(flux).map(|f| f.unwrap()).shared())
    }).collect();

    let u1_vec = state.conserved.iter().zip(block_data).map(|(uc, block)|
    {
        let solver   = solver.clone();
        let mesh     = mesh.clone();
        let flux_map = flux_map.clone();
        let block    = block.clone();
        let uc       = uc.clone();

        let u1 = async move {
            let (fx, fy) = flux_map[&block.index].clone().await;
            scheme.compute_block_updated_conserved(uc, fx.to_owned(), fy.to_owned(), &block, &solver, &mesh, time, dt).to_shared()
        };
        runtime.spawn(u1).map(|u| u.unwrap()).shared()
    });

    State {
        time: state.time + dt,
        iteration: state.iteration + 1,
        conserved: join_all(u1_vec).await
    }
}




// ============================================================================
async fn advance_rk1<C, F, U>(state: State<C>, update: U, _runtime: &tokio::runtime::Runtime) -> State<C>
    where
    C: Conserved,
    U: Fn(State<C>) -> F,
    F: std::future::Future<Output=State<C>>
{
    update(state).await
}




// ============================================================================
async fn advance_rk2<C, F, U>(state: State<C>, update: U, runtime: &tokio::runtime::Runtime) -> State<C>
    where
    C: 'static + Conserved,
    U: Fn(State<C>) -> F,
    F: std::future::Future<Output=State<C>>
{
    let b1 = Rational64::new(1, 2);

    let s1 = state.clone();
    let s1 = update(s1).await;
    let s1 = update(s1).await.weighted_average(b1, &state, runtime).await;
    s1
}




// ============================================================================
async fn advance_rk3<C, F, U>(state: State<C>, update: U, runtime: &tokio::runtime::Runtime) -> State<C>
    where
    C: 'static + Conserved,
    U: Fn(State<C>) -> F,
    F: std::future::Future<Output=State<C>>
{
    let b1 = Rational64::new(3, 4);
    let b2 = Rational64::new(1, 3);

    let s1 = state.clone();
    let s1 = update(s1).await;
    let s1 = update(s1).await.weighted_average(b1, &state, runtime).await;
    let s1 = update(s1).await.weighted_average(b2, &state, runtime).await;
    s1
}




// ============================================================================
pub fn advance_tokio<H: 'static + Hydrodynamics>(
    mut state:  State<H::Conserved>,
    hydro:      H,
    block_data: &Vec<BlockData<H::Conserved>>,
    mesh:       &Mesh,
    solver:     &Solver,
    dt:         f64,
    fold:       usize,
    runtime:    &tokio::runtime::Runtime) -> State<H::Conserved>
{
    let update = |state| advance_tokio_rk(state, hydro, block_data, mesh, solver, dt, runtime);

    for _ in 0..fold {
        state = match solver.rk_order {
            1 => runtime.block_on(advance_rk1(state, update, runtime)),
            2 => runtime.block_on(advance_rk2(state, update, runtime)),
            3 => runtime.block_on(advance_rk3(state, update, runtime)),
            _ => panic!("illegal RK order {}", solver.rk_order),
        }
    }
    return state;
}




// ============================================================================
impl<C: 'static + Conserved> State<C>
{
    async fn weighted_average(self, br: Rational64, s0: &State<C>, runtime: &tokio::runtime::Runtime) -> State<C>
    {
        use num::ToPrimitive;
        use futures::future::FutureExt;
        use futures::future::join_all;

        let bf = br.to_f64().unwrap();

        let u_avg = self.conserved
            .iter()
            .zip(&s0.conserved)
            .map(|(u1, u2)| {
                let u1 = u1.clone();
                let u2 = u2.clone();
                runtime.spawn(async move { u1 * (-bf + 1.) + u2 * bf }).map(|u| u.unwrap())
            });

        State{
            time:      self.time      * (-bf + 1.) + s0.time      * bf,
            iteration: self.iteration * (-br + 1 ) + s0.iteration * br,
            conserved: join_all(u_avg).await,
        }
    }
}




// ============================================================================
#[derive(Copy, Clone)]
struct UpdateScheme<H: Hydrodynamics>
{
    hydro: H,
}

impl<H: Hydrodynamics> UpdateScheme<H>
{
    fn new(hydro: H) -> Self
    {
        Self{hydro: hydro}
    }

    fn compute_block_primitive(&self, conserved: ArcArray<H::Conserved, Ix2>) -> Array<H::Primitive, Ix2>
    {
        conserved.mapv(|u| self.hydro.to_primitive(u))
    }

    fn compute_block_fluxes(
        &self,
        pe:     &Array<H::Primitive, Ix2>,
        block:  &BlockData<H::Conserved>,
        solver: &Solver,
        time:   f64) -> (Array<H::Conserved, Ix2>, Array<H::Conserved, Ix2>)
    {
        use ndarray::{s, azip};
        use ndarray_ops::{map_stencil3};

        let two_body_state = solver.orbital_elements.orbital_state_from_time(time);

        // ========================================================================
        let gx = map_stencil3(&pe, Axis(0), |a, b, c| self.hydro.plm_gradient(solver.plm, a, b, c));
        let gy = map_stencil3(&pe, Axis(1), |a, b, c| self.hydro.plm_gradient(solver.plm, a, b, c));
        let xf = &block.face_centers_x;
        let yf = &block.face_centers_y;

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
        .apply_collect(|l, r, f| self.hydro.intercell_flux(&solver, l, r, f, &two_body_state, Direction::X));

        // ============================================================================
        let fy = azip![
            cell_data.slice(s![1..-1,..-1]),
            cell_data.slice(s![1..-1, 1..]),
            yf]
        .apply_collect(|l, r, f| self.hydro.intercell_flux(&solver, l, r, f, &two_body_state, Direction::Y));

        (fx, fy)
    }

    fn compute_block_updated_conserved(
        &self,
        uc:       ArcArray<H::Conserved, Ix2>,
        fx:       Array<H::Conserved, Ix2>,
        fy:       Array<H::Conserved, Ix2>,
        block:    &BlockData<H::Conserved>,
        solver:   &Solver,
        mesh:     &Mesh,
        time:     f64,
        dt:       f64) -> Array<H::Conserved, Ix2>
    {
        use ndarray::{s, azip};

        // ============================================================================
        let dx = mesh.cell_spacing_x();
        let dy = mesh.cell_spacing_y();
        let two_body_state = solver.orbital_elements.orbital_state_from_time(time);

        let sum_sources = |s: [H::Conserved; 5]| s[0] + s[1] + s[2] + s[3] + s[4];

        // ============================================================================
        let sources = azip![
            &uc,
            &block.initial_conserved,
            &block.cell_centers]
        .apply_collect(|&u, &u0, &(x, y)| sum_sources(self.hydro.source_terms(&solver,u, u0, x, y, dt, &two_body_state)));

        // ============================================================================
        let du = if solver.need_flux_communication() {
            azip![
                fx.slice(s![1..-2, 1..-1]),
                fx.slice(s![2..-1, 1..-1]),
                fy.slice(s![1..-1, 1..-2]),
                fy.slice(s![1..-1, 2..-1])]
        } else {
            azip![
                fx.slice(s![..-1,..]),
                fx.slice(s![ 1..,..]),
                fy.slice(s![..,..-1]),
                fy.slice(s![.., 1..])]
        }.apply_collect(|&a, &b, &c, &d| ((b - a) / dx + (d - c) / dy) * -dt);

        (uc + du + sources).to_owned()
    }
}








/**
 * The code below advances the state using the old message-passing
 * parallelization strategy based on channels. I would prefer to either
 * deprecate it, since it duplicates msot of the update scheme, and will
 * thus need to be kept in sync manually as the scheme evolves. The only
 * reason to retain it is for benchmarking purposes.
 */




// ============================================================================
fn advance_channels_internal_block<H: Hydrodynamics>(
    state:      BlockState<H::Conserved>,
    hydro:      H,
    block_data: &BlockData<H::Conserved>,
    solver:     &Solver,
    mesh:       &Mesh,
    sender:     &crossbeam::Sender<Array<H::Primitive, Ix2>>,
    receiver:   &crossbeam::Receiver<NeighborPrimitiveBlock<H::Primitive>>,
    dt:         f64) -> BlockState<H::Conserved>
{
    let scheme = UpdateScheme::new(hydro);

    sender.send(scheme.compute_block_primitive(state.conserved.clone())).unwrap();

    let pe = ndarray_ops::extend_from_neighbor_arrays_2d(&receiver.recv().unwrap(), 2, 2, 2, 2);
    let (fx, fy) = scheme.compute_block_fluxes(&pe, block_data, solver, state.time);
    let u1 = scheme.compute_block_updated_conserved(state.conserved, fx, fy, block_data, solver, mesh, state.time, dt);

    BlockState::<H::Conserved>{
        time: state.time + dt,
        iteration: state.iteration + 1,
        conserved: u1.to_shared(),
    }
}




// ============================================================================
fn advance_channels_internal<H: Hydrodynamics>(
    conserved:  &mut ArcArray<H::Conserved, Ix2>,
    hydro:      H,
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

    let update = |state| advance_channels_internal_block(state, hydro, block_data, solver, mesh, sender, receiver, dt);
    let mut state = BlockState {
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
pub fn advance_channels<H: Hydrodynamics>(
    state: &mut State<H::Conserved>,
    hydro: H,
    block_data: &Vec<BlockData<H::Conserved>>,
    mesh: &Mesh,
    solver: &Solver,
    dt: f64,
    fold: usize)
{
    if solver.need_flux_communication() {
        panic!("the message-passing parallelization strategy does not support flux communication");
    }

    crossbeam::scope(|scope|
    {
        let time = state.time;
        let mut receivers       = Vec::new();
        let mut senders         = Vec::new();
        let mut block_primitive = HashMap::new();

        for (u, b) in state.conserved.iter_mut().zip(block_data)
        {
            let (their_s, my_r) = crossbeam::channel::unbounded();
            let (my_s, their_r) = crossbeam::channel::unbounded();

            senders.push(my_s);
            receivers.push(my_r);

            scope.spawn(move |_| advance_channels_internal(u, hydro, b, solver, mesh, &their_s, &their_r, time, dt, fold));
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
