use num::rational::Rational64;
use ndarray::{Axis, Array, ArcArray, Ix1, Ix2};
use ndarray_ops::MapArray3by3;
use kepler_two_body::{OrbitalElements, OrbitalState};
use godunov_core::solution_states;
use godunov_core::runge_kutta;
use hydro_iso2d::{Conserved, Primitive, Direction, riemann_hlle};




// ============================================================================
type NeighborPrimitiveBlock = [[ArcArray<Primitive, Ix2>; 3]; 3];
type BlockState = solution_states::SolutionStateArcArray<Conserved, Ix2>;
pub type BlockIndex = (usize, usize);




// ============================================================================
#[derive(Clone)]
pub struct BlockData
{
    pub initial_conserved: ArcArray<Conserved, Ix2>,
    pub cell_centers:      ArcArray<(f64, f64), Ix2>,
    pub face_centers_x:    ArcArray<(f64, f64), Ix2>,
    pub face_centers_y:    ArcArray<(f64, f64), Ix2>,
    pub index:             BlockIndex,
}




// ============================================================================
#[derive(Clone)]
pub struct State
{
    pub time: f64,
    pub iteration: Rational64,
    pub conserved: Vec<ArcArray<Conserved, Ix2>>,
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
    pub orbital_elements: OrbitalElements,
    pub dim: f64,
}

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
        let p1 = two_body_state.0;
        let p2 = two_body_state.1;

        let [ax1, ay1] = p1.gravitational_acceleration(x, y, self.softening_length);
        let [ax2, ay2] = p2.gravitational_acceleration(x, y, self.softening_length);

        let rho = conserved.density();
        let fx1 = rho * ax1;
        let fy1 = rho * ay1;
        let fx2 = rho * ax2;
        let fy2 = rho * ay2;

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

    fn strain_field(&self, true_dimension: f64, row: Direction, col: Direction) -> f64
    {
        use Direction::{X, Y};
        match (row, col)
        {
            (X, X) => 2.0 * ( (1.0 - 1.0 / true_dimension) * self.gx.velocity_x() - (1.0 / true_dimension) * self.gy.velocity_y() ),
            (X, Y) => self.gx.velocity_y() + self.gy.velocity_x(),
            (Y, X) => self.gx.velocity_y() + self.gy.velocity_x(),
            (Y, Y) => 2.0 * ( (1.0 - 1.0 / true_dimension) * self.gx.velocity_y() - (1.0 / true_dimension) * self.gy.velocity_x() ),
        }
    }

    fn stress_field(&self, kinematic_viscosity: f64, true_dimension: f64, row: Direction, col: Direction) -> f64
    {
        kinematic_viscosity * self.pc.density() * self.strain_field(true_dimension, row, col)
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
async fn advance_tokio_rk(state: State, block_data: &Vec<BlockData>, mesh: &Mesh, solver: &Solver, dt: f64, runtime: &tokio::runtime::Runtime) -> State
{
    use std::collections::HashMap;
    use ndarray::{s, azip};
    use ndarray_ops::{map_stencil3};
    use futures::future::{Future, FutureExt, join_all};
    use godunov_core::piecewise_linear::plm_gradient3;
    use Direction::{X, Y};


    async fn join_3by3<T: Clone + Future>(a: [[&T; 3]; 3]) -> [[T::Output; 3]; 3]
    {
        [
            [a[0][0].clone().await, a[0][1].clone().await, a[0][2].clone().await],
            [a[1][0].clone().await, a[1][1].clone().await, a[1][2].clone().await],
            [a[2][0].clone().await, a[2][1].clone().await, a[2][2].clone().await],
        ]
    }


    let pc_map: HashMap<_, _> = state.conserved.iter().zip(block_data).map(|(uc, block)|
    {
        let uc = uc.clone();
        let primitive = async move {
            uc.mapv(Conserved::to_primitive).to_shared()
        };
        (block.index, runtime.spawn(primitive).map(|p| p.unwrap()).shared())
    }).collect();


    let u1_vec = state.conserved.iter().zip(block_data).map(|(uc, block)|
    {
        let solver = solver.clone();
        let mesh   = mesh.clone();
        let block  = block.clone();
        let time   = state.time;
        let pc_map = pc_map.clone();
        let uc     = uc.clone();

        let u1 = async move {

            // ============================================================================
            let dx = mesh.cell_spacing_x();
            let dy = mesh.cell_spacing_y();
            let two_body_state = solver.orbital_elements.orbital_state_from_time(time);

            // ============================================================================
            let intercell_flux = |l: &CellData, r: &CellData, f: &(f64, f64), axis: Direction| -> Conserved
            {
                let cs2 = solver.sound_speed_squared(f, &two_body_state);
                let pl = *l.pc + *l.gradient_field(axis) * 0.5;
                let pr = *r.pc - *r.gradient_field(axis) * 0.5;
                let nu = solver.nu;
                let dim   = solver.dim;
                let tau_x = 0.5 * (l.stress_field(nu, dim, axis, X) + r.stress_field(nu, dim, axis, X));
                let tau_y = 0.5 * (l.stress_field(nu, dim, axis, Y) + r.stress_field(nu, dim, axis, Y));
                riemann_hlle(pl, pr, axis, cs2) + Conserved(0.0, -tau_x, -tau_y)
            };

            let sum_sources = |s: [Conserved; 5]| s[0] + s[1] + s[2] + s[3] + s[4];

            // ============================================================================
            let pn = join_3by3(mesh.neighbor_block_indexes(block.index).map(|i| &pc_map[i])).await;

            let pe = ndarray_ops::extend_from_neighbor_arrays_2d(&pn, 2, 2, 2, 2);
            let gx = map_stencil3(&pe, Axis(0), |a, b, c| plm_gradient3(solver.plm, a, b, c));
            let gy = map_stencil3(&pe, Axis(1), |a, b, c| plm_gradient3(solver.plm, a, b, c));
            let xf = &block.face_centers_x;
            let yf = &block.face_centers_y;

            // ============================================================================
            let sources = azip![
                &uc,
                &block.initial_conserved,
                &block.cell_centers]
            .apply_collect(|&u, &u0, &(x, y)| sum_sources(solver.source_terms(u, u0, x, y, dt, &two_body_state)));

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

            (uc + &du + &sources).to_shared()
        };
        runtime.spawn(u1).map(|u| u.unwrap())
    });


    State {
        time: state.time + dt,
        iteration: state.iteration + 1,
        conserved: join_all(u1_vec).await,
    }
}




// ============================================================================
impl State
{
    async fn weighted_average(self, br: Rational64, s0: &State, runtime: &tokio::runtime::Runtime) -> State
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
async fn advance_rk1<FutureState, F: Fn(State) -> FutureState>(state: State, update: F, _runtime: &tokio::runtime::Runtime) -> State
where
    FutureState: std::future::Future<Output=State>
{
    update(state).await
}




// ============================================================================
async fn advance_rk2<FutureState, F: Fn(State) -> FutureState>(state: State, update: F, runtime: &tokio::runtime::Runtime) -> State
where
    FutureState: std::future::Future<Output=State>
{
    let b1 = Rational64::new(1, 2);

    let s1 = state.clone();
    let s1 = update(s1).await;
    let s1 = update(s1).await.weighted_average(b1, &state, runtime).await;
    s1
}




// ============================================================================
async fn advance_rk3<FutureState, F: Fn(State) -> FutureState>(state: State, update: F, runtime: &tokio::runtime::Runtime) -> State
where
    FutureState: std::future::Future<Output=State>
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
pub fn advance_tokio(mut state: State, block_data: &Vec<BlockData>, mesh: &Mesh, solver: &Solver, dt: f64, fold: usize, runtime: &tokio::runtime::Runtime) -> State
{
    let update = |state| advance_tokio_rk(state, block_data, mesh, solver, dt, runtime);

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








/**
 * The code below advances the state using the old message-passing
 * parallelization strategy based on channels. I would prefer to either
 * deprecate it, since it duplicates msot of the update scheme, and will
 * thus need to be kept in sync manually as the scheme evolves. The only
 * reason to retain it is for benchmarking purposes.
 */




// ============================================================================
fn advance_channels_internal_block(
    state:      BlockState,
    block_data: &crate::BlockData,
    solver:     &Solver,
    mesh:       &Mesh,
    sender:     &crossbeam::Sender<Array<Primitive, Ix2>>,
    receiver:   &crossbeam::Receiver<NeighborPrimitiveBlock>,
    dt:         f64) -> BlockState
{
    // ============================================================================
    use ndarray::{s, azip};
    use ndarray_ops::{map_stencil3};
    use godunov_core::piecewise_linear::plm_gradient3;
    use Direction::{X, Y};

    // ============================================================================
    let dx = mesh.cell_spacing_x();
    let dy = mesh.cell_spacing_y();
    let two_body_state = solver.orbital_elements.orbital_state_from_time(state.time);

    // ============================================================================
    let intercell_flux = |l: &CellData, r: &CellData, f: &(f64, f64), axis: Direction| -> Conserved
    {
        let cs2   = solver.sound_speed_squared(f, &two_body_state);
        let pl    = *l.pc + *l.gradient_field(axis) * 0.5;
        let pr    = *r.pc - *r.gradient_field(axis) * 0.5;
        let nu    = solver.nu;
        let dim   = solver.dim;
        let tau_x = 0.5 * (l.stress_field(nu, dim, axis, X) + r.stress_field(nu, dim, axis, X));
        let tau_y = 0.5 * (l.stress_field(nu, dim, axis, Y) + r.stress_field(nu, dim, axis, Y));
        riemann_hlle(pl, pr, axis, cs2) + Conserved(0.0, -tau_x, -tau_y)
    };

    let sum_sources = |s: [Conserved; 5]| s[0] + s[1] + s[2] + s[3] + s[4];

    // ============================================================================
    sender.send(state.conserved.mapv(Conserved::to_primitive)).unwrap();

    // ============================================================================
    let sources = azip![
        &state.conserved,
        &block_data.initial_conserved,
        &block_data.cell_centers]
    .apply_collect(|&u, &u0, &(x, y)| sum_sources(solver.source_terms(u, u0, x, y, dt, &two_body_state)));

    let pe = ndarray_ops::extend_from_neighbor_arrays_2d(&receiver.recv().unwrap(), 2, 2, 2, 2);
    let gx = map_stencil3(&pe, Axis(0), |a, b, c| plm_gradient3(solver.plm, a, b, c));
    let gy = map_stencil3(&pe, Axis(1), |a, b, c| plm_gradient3(solver.plm, a, b, c));
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
    BlockState{
        time: state.time + dt,
        iteration: state.iteration + 1,
        conserved: (state.conserved + du + sources).to_shared(),
    }
}




// ============================================================================
fn advance_channels_internal(
    conserved:  &mut ArcArray<Conserved, Ix2>,
    block_data: &crate::BlockData,
    solver:     &Solver,
    mesh:       &Mesh,
    sender:     &crossbeam::Sender<Array<Primitive, Ix2>>,
    receiver:   &crossbeam::Receiver<NeighborPrimitiveBlock>,
    time:       f64,
    dt:         f64,
    fold:       usize)
{
    use std::convert::TryFrom;

    let update = |state| advance_channels_internal_block(state, block_data, solver, mesh, sender, receiver, dt);
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
pub fn advance_channels(state: &mut State, block_data: &Vec<BlockData>, mesh: &Mesh, solver: &Solver, dt: f64, fold: usize)
{
    crossbeam::scope(|scope|
    {
        use std::collections::HashMap;

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

            scope.spawn(move |_| advance_channels_internal(u, b, solver, mesh, &their_s, &their_r, time, dt, fold));
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
