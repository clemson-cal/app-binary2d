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

use crate::tracers::{
    Tracer,
    update_tracers,
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
    pub tracers: Vec<Tracer>,
}

impl<C: Conserved> BlockSolution<C>
{
    fn new_tracers(&self, new_tracers: Vec<Tracer>) -> Self
    {
        BlockSolution{
            conserved: self.conserved.clone(),
            integrated_source_terms: self.integrated_source_terms,
            orbital_elements_change: self.orbital_elements_change,
            tracers: new_tracers,
        }
    }
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
        let tr0 = &s0.tracers;
        let tr1 = &s1.tracers;

        BlockSolution{
            conserved: u1 * (-bf + 1.) + u0 * bf,
            integrated_source_terms: t1.weighted_average(br, t0),
            orbital_elements_change: e1.weighted_average(br, e0),
            tracers: tr0.iter().zip(tr1.iter()).map(|(tr0, tr1)| tr1.clone().weighted_average(br, &tr0.clone())).collect(),
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

    let fv_map: HashMap<_, _> = block_data.iter().map(|block|
    {
        let solver      = solver.clone();
        let mesh        = mesh.clone();
        let pc_map      = pc_map.clone();
        let block       = block.clone();
        let block_index = block.index;

        let fv = async move {
            let pn = join_3by3(mesh.neighbor_block_indexes(block_index).map_3by3(|i| &pc_map[i])).await;
            let pe = ndarray_ops::extend_from_neighbor_arrays_2d(&pn, 2, 2, 2, 2);
            let (fx, fy, vstar_x, vstar_y) = scheme.compute_block_fluxes_and_face_velocities(&pe, &block, &solver, time);
            (fx.to_shared(), fy.to_shared(), vstar_x.to_shared(), vstar_y.to_shared())
        };
        let fv = runtime.spawn(fv);
        let fv = async {
            fv.await.unwrap()
        };
        return (block_index, fv.shared());

    }).collect();

    let s1_vec = state.solution.iter().zip(block_data).map(|(solution, block)|
    {
        let solver   = solver.clone();
        let mesh     = mesh.clone();
        let fv_map = fv_map.clone();
        let block    = block.clone();
        let solution = solution.clone();

        let s1 = async move {
            let (fx, fy, vstar_x, vstar_y) = if ! solver.need_flux_communication() {
                fv_map[&block.index].clone().await
            } else {
                let fv_n = join_3by3(mesh.neighbor_block_indexes(block.index).map_3by3(|i| &fv_map[i])).await;
                let fx_n = fv_n.map_3by3(|fv| fv.0.clone());
                let fy_n = fv_n.map_3by3(|fv| fv.1.clone());
                let vx_n = fv_n.map_3by3(|fv| fv.2.clone());
                let vy_n = fv_n.map_3by3(|fv| fv.3.clone());
                let fx_e = ndarray_ops::extend_from_neighbor_arrays_2d(&fx_n, 1, 1, 1, 1);
                let fy_e = ndarray_ops::extend_from_neighbor_arrays_2d(&fy_n, 1, 1, 1, 1);
                let vx_e = ndarray_ops::extend_from_neighbor_arrays_2d(&vx_n, 1, 1, 1, 1);
                let vy_e = ndarray_ops::extend_from_neighbor_arrays_2d(&vy_n, 1, 1, 1, 1);
                (fx_e.to_shared(), fy_e.to_shared(), vx_e.to_shared(), vy_e.to_shared())
            };
            scheme.compute_block_updated_solution(solution, (fx, Some(vstar_x)), (fy, Some(vstar_y)), &block, &solver, &mesh, time, dt)
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
        if solver.using_tracers() {
            state = runtime.block_on(rebin_tracers(state, &mesh, block_data, runtime));
        }
        state = runtime.block_on(solver.runge_kutta().advance_async(state, update, runtime));
    }
    return state;
}




// ============================================================================
async fn rebin_tracers<C: Conserved>(
    state: State<C>,
    mesh : &Mesh,
    block_data: &Vec<BlockData<C>>,
    runtime: &tokio::runtime::Runtime) -> State<C>
{
    use futures::future::{FutureExt, join_all};
    use crate::{tracers_on_and_off_block, push_new_tracers};
    use std::sync::Arc;

    let tracer_map: HashMap<_, _> = state.solution.iter().zip(block_data).map(|(s, block)|
    {
        let mesh         = mesh.clone();
        let block        = block.clone();
        let tracers      = s.tracers.clone();
        let block_index  = block.index;

        let their_tracers = async move {
            let (_mine, theirs) = tracers_on_and_off_block(tracers, &mesh, block_index);
            Arc::new(theirs)
        };
        let theirs = runtime.spawn(their_tracers);
        let theirs = async {
            theirs.await.unwrap()
        };
        return (block_index, theirs.shared());
    }).collect();

    let s1_vec = state.solution.iter().zip(block_data).map(|(s, block)|
    {
        let mesh         = mesh.clone();
        let block        = block.clone();
        let solution     = s.clone();
        let tracers      = s.tracers.clone();
        let tracer_map   = tracer_map.clone();

        let s1 = async move {
            let tr_n = join_3by3(mesh.neighbor_block_indexes(block.index).map_3by3(|i| &tracer_map[i])).await;
            solution.new_tracers(push_new_tracers(tracers, tr_n, &mesh, block.index))
        };
        let s1 = runtime.spawn(s1);
        let s1 = async {
            s1.await.unwrap()
        };
        return s1;
    });

    State {
        time: state.time,
        iteration: state.iteration,
        solution: join_all(s1_vec).await
    }
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
        time:   f64) -> (Array<H::Conserved, Ix2>, Array<H::Conserved, Ix2>)
    {
        use ndarray::{s, azip};
        use ndarray_ops::{map_stencil3};

        // ========================================================================
        let two_body_state = solver.orbital_elements.orbital_state_from_time(time);
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

    fn compute_block_fluxes_and_face_velocities(
        &self,
        pe:     &Array<H::Primitive, Ix2>,
        block:  &BlockData<H::Conserved>,
        solver: &Solver,
        time:   f64) -> (Array<H::Conserved, Ix2>, Array<H::Conserved, Ix2>, Array<f64, Ix2>, Array<f64, Ix2>)
    {
        use ndarray::{s, azip};
        use ndarray_ops::{map_stencil3};
        use crate::traits::{Primitive};

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
        let flux_and_vx = |(f, u): (H::Conserved, H::Conserved)| (f, self.hydro.to_primitive(u).velocity_x());
        let flux_and_vy = |(f, u): (H::Conserved, H::Conserved)| (f, self.hydro.to_primitive(u).velocity_y());

        // ============================================================================
        let fv_x = azip![
            cell_data.slice(s![..-1,1..-1]),
            cell_data.slice(s![ 1..,1..-1]),
            xf]
        .apply_collect(|l, r, f| flux_and_vx(self.hydro.intercell_flux_plus_state(&solver, l, r, f, &two_body_state, Direction::X)));

        // ============================================================================
        let fv_y = azip![
            cell_data.slice(s![1..-1,..-1]),
            cell_data.slice(s![1..-1, 1..]),
            yf]
        .apply_collect(|l, r, f| flux_and_vy(self.hydro.intercell_flux_plus_state(&solver, l, r, f, &two_body_state, Direction::Y)));

        let fx      = fv_x.map(|fv| fv.0);
        let fy      = fv_y.map(|fv| fv.0);
        let vstar_x = fv_x.map(|fv| fv.1);
        let vstar_y = fv_y.map(|fv| fv.1);

        (fx, fy, vstar_x, vstar_y)
    }


    fn compute_block_tracer_update(
        &self,
        tracers:  Vec<Tracer>, 
        vstar_x:  ArcArray<f64, Ix2>,
        vstar_y:  ArcArray<f64, Ix2>,
        index:    BlockIndex,
        solver:   &Solver,
        mesh:     &Mesh,
        dt:       f64) -> Vec<Tracer>
    {
        if !solver.need_flux_communication() {
            panic!();
        }
        tracers.into_iter()
               .map(|t| update_tracers(t, &mesh, index, &vstar_x.to_owned(), &vstar_y.to_owned(), 1, dt))
               .collect()
    }

    fn compute_block_updated_solution(
        &self,
        solution: BlockSolution<H::Conserved>,
        fv_x:       (ArcArray<H::Conserved, Ix2>, Option<ArcArray<f64, Ix2>>),
        fv_y:       (ArcArray<H::Conserved, Ix2>, Option<ArcArray<f64, Ix2>>),
        block:    &BlockData<H::Conserved>,
        solver:   &Solver,
        mesh:     &Mesh,
        time:     f64,
        dt:       f64) -> BlockSolution<H::Conserved>
    {
        let dx = mesh.cell_spacing_x();
        let dy = mesh.cell_spacing_y();
        let two_body_state = solver.orbital_elements.orbital_state_from_time(time);

        let (fx, vstar_x) = fv_x;
        let (fy, vstar_y) = fv_y;

        if solver.using_tracers()  && (vstar_x.is_none() || vstar_y.is_none()) {
            panic!("Updating with tracers so need to send face velocities to block_udpate!");
        }

        let s1 = if ! solver.low_mem {
            use ndarray::{s, azip};

            let itemized_sources = azip![
                &solution.conserved,
                &block.initial_conserved,
                &block.cell_centers]
            .apply_collect(|&u, &u0, &(x, y)| self.hydro.source_terms(&solver, u, u0, x, y, dt, &two_body_state));

            let sources = itemized_sources.map(ItemizedChange::total);
            let ds = itemized_sources.fold(ItemizedChange::zeros(), |a, b| a.add(b)).mul(dx * dy);
            let de = ds.perturbation(time, solver.orbital_elements);

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

            let new_tracers = if solver.using_tracers()  {
                self.compute_block_tracer_update(solution.tracers, vstar_x.unwrap(), vstar_y.unwrap(), block.index, &solver, &mesh, dt)
            }
            else {
                solution.tracers
            };

            BlockSolution{
                conserved: solution.conserved + du + sources,
                integrated_source_terms: solution.integrated_source_terms.add(&ds),
                orbital_elements_change: solution.orbital_elements_change.add(&de),
                tracers: new_tracers,
            }
        } else {

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

            let new_tracers = if solver.using_tracers()  {
                self.compute_block_tracer_update(solution.tracers, vstar_x.unwrap(), vstar_y.unwrap(), block.index, &solver, &mesh, dt)
            }
            else {
                solution.tracers
            };

            BlockSolution{
                conserved: u1,
                integrated_source_terms: solution.integrated_source_terms.add(&ds),
                orbital_elements_change: solution.orbital_elements_change.add(&de),
                tracers: new_tracers,
            }
        };
        return s1;
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
    receiver:   &crossbeam::Receiver<[[ArcArray<H::Primitive, Ix2>; 3]; 3]>,
    dt:         f64) -> BlockState<H::Conserved>
{
    let scheme = UpdateScheme::new(hydro);

    sender.send(scheme.compute_block_primitive(state.solution.conserved.clone())).unwrap();

    let solution = state.solution;

    let pe = ndarray_ops::extend_from_neighbor_arrays_2d(&receiver.recv().unwrap(), 2, 2, 2, 2);
    let (fx, fy) = scheme.compute_block_fluxes(&pe, block_data, solver, state.time);
    let s1 = scheme.compute_block_updated_solution(solution, (fx.to_shared(), None), (fy.to_shared(), None), block_data, solver, mesh, state.time, dt);

    BlockState::<H::Conserved>{
        time: state.time + dt,
        iteration: state.iteration + 1,
        solution: s1,
    }
}




// ============================================================================
fn advance_channels_internal<H: Hydrodynamics>(
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
    let update = |state| advance_channels_internal_block(state, hydro, block_data, solver, mesh, sender, receiver, dt);
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
        todo!("flux communication with message-passing parallelization");
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

            scope.spawn(move |_| advance_channels_internal(solution, time, hydro, block, solver, mesh, &their_s, &their_r, dt, fold));
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
