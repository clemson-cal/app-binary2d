use std::future::Future;
use std::collections::HashMap;
use num::rational::Rational64;
use num::ToPrimitive;
use ndarray::{Axis, Array, ArcArray, Ix2};
use ndarray_ops::MapArray3by3;
use godunov_core::runge_kutta;
use kepler_two_body::OrbitalElements;

use crate::physics::{
    Direction,
    CellData,
    ItemizedChange,
    Solver,
};

use crate::mesh::{
    Mesh,
    BlockIndex,
};

use crate::traits::{
    Hydrodynamics,
    Conserved,
    ItemizeData,
};




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
pub struct BlockSolution<C: Conserved>
{
    pub conserved: ArcArray<C, Ix2>,
    pub integrated_source_terms: ItemizedChange<C>,
    pub orbital_elements_change: ItemizedChange<OrbitalElements>,
}




// ============================================================================
#[derive(Clone)]
struct BlockState<C: Conserved>
{
    pub time: f64,
    pub iteration: Rational64,
    pub solution: BlockSolution<C>,
}




// ============================================================================
#[derive(Clone)]
pub struct State<C: Conserved>
{
    pub time: f64,
    pub iteration: Rational64,
    pub solution: Vec<BlockSolution<C>>,
}




// ============================================================================
#[derive(Copy, Clone)]
struct UpdateScheme<H: Hydrodynamics>
{
    hydro: H,
}




// ============================================================================
impl<C: ItemizeData> runge_kutta::WeightedAverage for ItemizedChange<C>
{
    fn weighted_average(self, br: Rational64, s0: &Self) -> Self
    {
        let bf = br.to_f64().unwrap();
        Self{
            sink1:   self.sink1   * (-bf + 1.) + s0.sink1   * bf,
            sink2:   self.sink2   * (-bf + 1.) + s0.sink2   * bf,
            grav1:   self.grav1   * (-bf + 1.) + s0.grav1   * bf,
            grav2:   self.grav2   * (-bf + 1.) + s0.grav2   * bf,
            buffer:  self.buffer  * (-bf + 1.) + s0.buffer  * bf,
            cooling: self.cooling * (-bf + 1.) + s0.cooling * bf,
        }
    }
}




// ============================================================================
impl<C: Conserved> runge_kutta::WeightedAverage for BlockState<C>
{
    fn weighted_average(self, br: Rational64, s0: &Self) -> Self
    {
        let bf = br.to_f64().unwrap();
        Self{
            time:      self.time      * (-bf + 1.) + s0.time      * bf,
            iteration: self.iteration * (-br + 1 ) + s0.iteration * br,
            solution:  self.solution.weighted_average(br, &s0.solution),
        }
    }
}




// ============================================================================
impl<C: Conserved> runge_kutta::WeightedAverage for BlockSolution<C>
{
    fn weighted_average(self, br: Rational64, s0: &Self) -> Self
    {
        let s1 = self;
        let bf = br.to_f64().unwrap();
        let u0 = s0.conserved.clone();
        let u1 = s1.conserved.clone();
        let t0 = &s0.integrated_source_terms;
        let t1 = &s1.integrated_source_terms;
        let e0 = &s0.orbital_elements_change;
        let e1 = &s1.orbital_elements_change;

        BlockSolution{
            conserved: u1 * (-bf + 1.) + u0 * bf,
            integrated_source_terms: t1.weighted_average(br, t0),
            orbital_elements_change: e1.weighted_average(br, e0),
        }
    }
}




// ============================================================================
#[async_trait::async_trait]
impl<C: Conserved> runge_kutta::WeightedAverageAsync for State<C>
{
    type Runtime = tokio::runtime::Runtime;
    async fn weighted_average(self, br: Rational64, s0: &Self, runtime: &Self::Runtime) -> Self
    {
        use futures::future::join_all;
        use godunov_core::runge_kutta::WeightedAverage;

        let bf = br.to_f64().unwrap();
        let s_avg = self.solution
            .into_iter()
            .zip(&s0.solution)
            .map(|(s1, s0)| (s1, s0.clone()))
            .map(|(s1, s0)| runtime.spawn(async move { s1.weighted_average(br, &s0) }))
            .map(|f| async { f.await.unwrap() });

        State{
            time:      self.time      * (-bf + 1.) + s0.time      * bf,
            iteration: self.iteration * (-br + 1 ) + s0.iteration * br,
            solution: join_all(s_avg).await,
        }
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
pub fn advance_rayon<H: 'static + Hydrodynamics>(
    mut state: State<H::Conserved>,
    hydro: H,
    block_data: &Vec<BlockData<H::Conserved>>,
    mesh: &Mesh,
    solver: &Solver,
    dt: f64,
    fold: usize,
    pool: &rayon::ThreadPool) -> State<H::Conserved>
{
    use futures::future::{FutureExt, join_all};
    use rayon_future::FutureSpawn;
    use std::rc::Rc;

    if solver.rk_order != 1 {
        todo!("RK != 1 with rayon pool");
    }
    if solver.need_flux_communication() {
        todo!("flux communication with rayon pool");
    }

    for _ in 0..fold {
        let scheme = UpdateScheme::new(hydro);
        let time = state.time;
        let iter = state.iteration;

        let solution = pool.scope(move |scope| {

            let mut pc_map = HashMap::new();
            let mut s1_vec = Vec::new();

            for (block, solution) in block_data.iter().zip(state.solution.iter()) {
                let conserved = solution.conserved.clone();
                let primitive = move || {
                    scheme.compute_block_primitive(conserved)
                };
                pc_map.insert(block.index, scope.run(primitive).shared());
            }

            let pc_map = Rc::new(pc_map);

            for (block, solution) in block_data.iter().zip(state.solution) {
                let pc_map = Rc::clone(&pc_map);

                let s1 = async move {
                    let pn = join_3by3(mesh.neighbor_block_indexes(block.index).map_3by3(|i| &pc_map[i])).await;
                    let pe = ndarray_ops::extend_from_neighbor_arrays_2d(&pn, 2, 2, 2, 2);
                    let fg = scope.run(move || scheme.compute_block_fluxes(&pe, block, solver, mesh, time)).await;

                    let (fx, fy) = fg;
                    let (fx, fy) = (fx.to_shared(), fy.to_shared());

                    scope.run(move || scheme.compute_block_updated_solution(solution, fx, fy, block, solver, mesh, time, dt)).await
                };
                s1_vec.push(s1.shared());
            }
            futures::executor::block_on(join_all(s1_vec))
        });

        state = State{
            time: time + dt,
            iteration: iter + 1,
            solution: solution,
        };
    }
    return state;
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

    let pc_map: HashMap<_, _> = state.solution.iter().zip(block_data).map(|(solution, block)|
    {
        let uc = solution.conserved.clone();
        let primitive = async move {
            scheme.compute_block_primitive(uc).to_shared()
        };
        let primitive = runtime.spawn(primitive);
        let primitive = async {
            primitive.await.unwrap()
        };
        return (block.index, primitive.shared());

    }).collect();

    let flux_map: HashMap<_, _> = block_data.iter().map(|block|
    {
        let solver      = solver.clone();
        let mesh        = mesh.clone();
        let pc_map      = pc_map.clone();
        let block       = block.clone();
        let block_index = block.index;

        let flux = async move {
            let pn = join_3by3(mesh.neighbor_block_indexes(block_index).map_3by3(|i| &pc_map[i])).await;
            let pe = ndarray_ops::extend_from_neighbor_arrays_2d(&pn, 2, 2, 2, 2);
            let (fx, fy) = scheme.compute_block_fluxes(&pe, &block, &solver, &mesh, time);
            (fx.to_shared(), fy.to_shared())
        };
        let flux = runtime.spawn(flux);
        let flux = async {
            flux.await.unwrap()
        };
        return (block_index, flux.shared());

    }).collect();

    let s1_vec = state.solution.iter().zip(block_data).map(|(solution, block)|
    {
        let solver   = solver.clone();
        let mesh     = mesh.clone();
        let flux_map = flux_map.clone();
        let block    = block.clone();
        let solution = solution.clone();

        let s1 = async move {
            let (fx, fy) = if ! solver.need_flux_communication() {
                flux_map[&block.index].clone().await
            } else {
                let flux_n = join_3by3(mesh.neighbor_block_indexes(block.index).map_3by3(|i| &flux_map[i])).await;
                let fx_n = flux_n.map_3by3(|f| f.0.clone());
                let fy_n = flux_n.map_3by3(|f| f.1.clone());
                let fx_e = ndarray_ops::extend_from_neighbor_arrays_2d(&fx_n, 1, 1, 1, 1);
                let fy_e = ndarray_ops::extend_from_neighbor_arrays_2d(&fy_n, 1, 1, 1, 1);
                (fx_e.to_shared(), fy_e.to_shared())
            };
            scheme.compute_block_updated_solution(solution, fx, fy, &block, &solver, &mesh, time, dt)
        };
        let s1 = runtime.spawn(s1);
        let s1 = async {
            s1.await.unwrap()
        };
        return s1;
    });

    State {
        time: state.time + dt,
        iteration: state.iteration + 1,
        solution: join_all(s1_vec).await
    }
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
        state = runtime.block_on(solver.runge_kutta().advance_async(state, update, runtime));
    }
    return state;
}




// ============================================================================
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
        mesh:     &Mesh,
        time:   f64) -> (Array<H::Conserved, Ix2>, Array<H::Conserved, Ix2>)
    {
        use ndarray::{s, azip};
        use ndarray_ops::{map_stencil3};

        // ========================================================================
        let two_body_state = solver.orbital_elements.orbital_state_from_time(time);
        let gx = map_stencil3(&pe, Axis(0), |a, b, c| self.hydro.plm_gradient(solver.plm, a, b, c));
        let gy = map_stencil3(&pe, Axis(1), |a, b, c| self.hydro.plm_gradient(solver.plm, a, b, c));
        let dx = mesh.cell_spacing_x();
        let dy = mesh.cell_spacing_y();
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
        .apply_collect(|l, r, f| self.hydro.intercell_flux(&solver, l, r, f, dx, dy, &two_body_state, Direction::X));

        // ============================================================================
        let fy = azip![
            cell_data.slice(s![1..-1,..-1]),
            cell_data.slice(s![1..-1, 1..]),
            yf]
        .apply_collect(|l, r, f| self.hydro.intercell_flux(&solver, l, r, f, dx, dy, &two_body_state, Direction::Y));

        (fx, fy)
    }

    fn compute_block_updated_solution(
        &self,
        solution: BlockSolution<H::Conserved>,
        fx:       ArcArray<H::Conserved, Ix2>,
        fy:       ArcArray<H::Conserved, Ix2>,
        block:    &BlockData<H::Conserved>,
        solver:   &Solver,
        mesh:     &Mesh,
        time:     f64,
        dt:       f64) -> BlockSolution<H::Conserved>
    {
        let dx = mesh.cell_spacing_x();
        let dy = mesh.cell_spacing_y();
        let two_body_state = solver.orbital_elements.orbital_state_from_time(time);

        let mut ds = ItemizedChange::zeros();

        let u1 = ArcArray::from_shape_fn(solution.conserved.dim(), |i| {
            let m = if solver.need_flux_communication() {
                (i.0 + 1, i.1 + 1)
            } else {
                i
            };
            let du = ((fx[(m.0 + 1, m.1)] - fx[m]) / dx +
                      (fy[(m.0, m.1 + 1)] - fy[m]) / dy) * -dt;
            let uc = solution.conserved[i];
            let u0 = block.initial_conserved[i];
            let (x, y)  = block.cell_centers[i];
            let sources = self.hydro.source_terms(&solver, uc, u0, x, y, dt, &two_body_state);

            ds.add_mut(&sources);
            uc + du + sources.total()
        });

        let ds = ds.mul(dx * dy);
        let de = ds.perturbation(time, solver.orbital_elements);

        BlockSolution{
            conserved: u1,
            integrated_source_terms: solution.integrated_source_terms.add(&ds),
            orbital_elements_change: solution.orbital_elements_change.add(&de),
        }
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
fn advance_crossbeam_internal_block<H: Hydrodynamics>(
    state:      BlockState<H::Conserved>,
    hydro:      H,
    block_data: &BlockData<H::Conserved>,
    solver:     &Solver,
    mesh:       &Mesh,
    sender:     &crossbeam::Sender<Array<H::Primitive, Ix2>>,
    receiver:   &crossbeam::Receiver<[[ArcArray<H::Primitive, Ix2>; 3]; 3]>,
    dt:         f64) -> BlockState<H::Conserved>
{
    let scheme = UpdateScheme::new(hydro);

    sender.send(scheme.compute_block_primitive(state.solution.conserved.clone())).unwrap();

    let solution = state.solution;

    let pe = ndarray_ops::extend_from_neighbor_arrays_2d(&receiver.recv().unwrap(), 2, 2, 2, 2);
    let (fx, fy) = scheme.compute_block_fluxes(&pe, block_data, solver, mesh, state.time);
    let s1 = scheme.compute_block_updated_solution(solution, fx.to_shared(), fy.to_shared(), block_data, solver, mesh, state.time, dt);

    BlockState::<H::Conserved>{
        time: state.time + dt,
        iteration: state.iteration + 1,
        solution: s1,
    }
}




// ============================================================================
fn advance_crossbeam_internal<H: Hydrodynamics>(
    solution:   &mut BlockSolution<H::Conserved>,
    time:       f64,
    hydro:      H,
    block_data: &BlockData<H::Conserved>,
    solver:     &Solver,
    mesh:       &Mesh,
    sender:     &crossbeam::Sender<Array<H::Primitive, Ix2>>,
    receiver:   &crossbeam::Receiver<[[ArcArray<H::Primitive, Ix2>; 3]; 3]>,
    dt:         f64,
    fold:       usize)
{
    let update = |state| advance_crossbeam_internal_block(state, hydro, block_data, solver, mesh, sender, receiver, dt);
    let mut state = BlockState {
        time: time,
        iteration: Rational64::new(0, 1),
        solution: solution.clone(),
    };

    for _ in 0..fold
    {
        state = solver.runge_kutta().advance(state, update);
    }
    *solution = state.solution;
}




// ============================================================================
pub fn advance_crossbeam<H: Hydrodynamics>(
    state: &mut State<H::Conserved>,
    hydro: H,
    block_data: &Vec<BlockData<H::Conserved>>,
    mesh: &Mesh,
    solver: &Solver,
    dt: f64,
    fold: usize)
{
    if solver.need_flux_communication() {
        todo!("flux communication with crossbeam");
    }
    let time = state.time;

    crossbeam::scope(|scope|
    {
        let mut receivers       = Vec::new();
        let mut senders         = Vec::new();
        let mut block_primitive = HashMap::new();

        for (solution, block) in state.solution.iter_mut().zip(block_data)
        {
            let (their_s, my_r) = crossbeam::channel::unbounded();
            let (my_s, their_r) = crossbeam::channel::unbounded();

            senders.push(my_s);
            receivers.push(my_r);

            scope.spawn(move |_| advance_crossbeam_internal(solution, time, hydro, block, solver, mesh, &their_s, &their_r, dt, fold));
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
                    s.send(mesh.neighbor_block_indexes(block_data.index).map_3by3(|i| block_primitive
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
