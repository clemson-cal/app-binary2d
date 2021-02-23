use std::future::Future;
use std::collections::HashMap;
use ndarray::{Axis, Array, ArcArray, Ix2};
use ndarray_ops::MapArray3by3;
use crate::app::{
    AnyModel,
    AnyHydro
};
use crate::physics::{
    CellData,
    Direction,
    HydroError,
    Physics,
};
use crate::mesh::{
    Mesh,
    BlockIndex,
};
use crate::state::{
    BlockState,
    ItemizedChange,
    State,
};
use crate::traits::{
    Conserved,
    Hydrodynamics,
    Zeros,
};




// ============================================================================
#[derive(Clone)]
pub struct BlockData<C: Conserved> {
    pub initial_conserved: ArcArray<C, Ix2>,
    pub cell_centers:      ArcArray<(f64, f64), Ix2>,
    pub face_centers_x:    ArcArray<(f64, f64), Ix2>,
    pub face_centers_y:    ArcArray<(f64, f64), Ix2>,
    pub index:             BlockIndex,
}

impl<C: Conserved> BlockData<C> {
    pub fn from_model<H>(model: &AnyModel, hydro: &H, mesh: &Mesh, index: BlockIndex) -> anyhow::Result<Self>
    where
        H: Hydrodynamics<Conserved = C> + Into<AnyHydro>
    {
        Ok(Self {
            initial_conserved: BlockState::from_model(model, hydro, mesh, index)?.conserved,
            cell_centers: mesh.cell_centers(index).to_shared(),
            face_centers_x: mesh.face_centers_x(index).to_shared(),
            face_centers_y: mesh.face_centers_y(index).to_shared(),
            index,
        })
    }
}




// ============================================================================
#[derive(Copy, Clone)]
struct UpdateScheme<H: Hydrodynamics> {
    hydro: H,
}




// ============================================================================
async fn try_join_3by3<F, T, E>(a: [[&F; 3]; 3]) -> Result<[[T; 3]; 3], E>
where
    F: Clone + Future<Output = Result<T, E>>
{
    Ok([
        [a[0][0].clone().await?, a[0][1].clone().await?, a[0][2].clone().await?],
        [a[1][0].clone().await?, a[1][1].clone().await?, a[1][2].clone().await?],
        [a[2][0].clone().await?, a[2][1].clone().await?, a[2][2].clone().await?],
    ])
}




// ============================================================================
async fn try_advance_rk<H: 'static + Hydrodynamics>(
    state: State<H::Conserved>,
    hydro: H,
    block_data: &HashMap<BlockIndex, BlockData<H::Conserved>>,
    mesh: &Mesh,
    physics: &Physics,
    dt: f64,
    runtime: &tokio::runtime::Runtime) -> Result<State<H::Conserved>, HydroError>
{
    use futures::future::{FutureExt, join_all};
    use std::sync::Arc;


    let scheme = UpdateScheme::new(hydro);
    let time = state.time;
    let mut pc_map = HashMap::with_capacity(state.solution.len());
    let mut fg_map = HashMap::with_capacity(state.solution.len());
    let mut s1_vec = Vec::with_capacity(state.solution.len());


    for index in mesh.block_indexes() {
        let uc = state.solution[&index].conserved.clone();
        let block = block_data[&index].clone();
        let primitive = async move {
            scheme.try_block_primitive(uc, block).map(|p| p.to_shared())
        };
        pc_map.insert(index, runtime.spawn(primitive).map(|f| f.unwrap()).shared());
    }
    let pc_map = Arc::new(pc_map);


    for index in mesh.block_indexes() {
        let physics     = physics.clone();
        let mesh        = mesh.clone();
        let pc_map      = pc_map.clone();
        let block       = block_data[&index].clone();

        let flux = async move {
            let pn = try_join_3by3(mesh.neighbor_block_indexes(block.index).map_3by3(|i| &pc_map[i])).await?;
            let pe = ndarray_ops::extend_from_neighbor_arrays_2d(&pn, 2, 2, 2, 2);
            let (fx, fy) = scheme.compute_block_fluxes(&pe, &block, &physics, &mesh, time);
            Ok::<_, HydroError>((fx.to_shared(), fy.to_shared()))
        };
        fg_map.insert(index, runtime.spawn(flux).map(|f| f.unwrap()).shared());
    }
    let fg_map = Arc::new(fg_map);


    for index in mesh.block_indexes() {
        let physics  = physics.clone();
        let mesh     = mesh.clone();
        let fg_map   = fg_map.clone();
        let block    = block_data[&index].clone();
        let solution = state.solution[&index].clone();

        let s1 = async move {
            let (fx, fy) = if ! physics.need_flux_communication() {
                fg_map[&block.index].clone().await?
            } else {
                let flux_n = try_join_3by3(mesh.neighbor_block_indexes(block.index).map_3by3(|i| &fg_map[i])).await?;
                let fx_n = flux_n.map_3by3(|f| f.0.clone());
                let fy_n = flux_n.map_3by3(|f| f.1.clone());
                let fx_e = ndarray_ops::extend_from_neighbor_arrays_2d(&fx_n, 1, 1, 1, 1);
                let fy_e = ndarray_ops::extend_from_neighbor_arrays_2d(&fy_n, 1, 1, 1, 1);
                (fx_e.to_shared(), fy_e.to_shared())
            };
            Ok::<_, HydroError>((index, scheme.compute_block_updated_solution(solution, fx, fy, &block, &physics, &mesh, time, dt)))
        };
        s1_vec.push(runtime.spawn(s1).map(|f| f.unwrap()))
    }


    let solution = join_all(s1_vec)
        .await
        .into_iter()
        .collect::<Result<_, _>>()
        .map_err(|e| e.with_orbital_state(physics.orbital_state_from_time(time)))?;


    Ok(State {
        time: state.time + dt,
        iteration: state.iteration + 1,
        solution: solution,
    })
}




// ============================================================================
impl<H: Hydrodynamics> UpdateScheme<H>
{
    fn new(hydro: H) -> Self {
        Self{hydro}
    }

    fn try_block_primitive(
        &self,
        conserved: ArcArray<H::Conserved, Ix2>,
        block: BlockData<H::Conserved>) -> Result<Array<H::Primitive, Ix2>, HydroError>
    {
        let x: Result<Vec<_>, _> = conserved
            .iter()
            .zip(block.cell_centers.iter())
            .map(|(&u, &xy)| self
                .hydro
                .try_to_primitive(u)
                .map_err(|e| e.at_position(xy)))
            .collect();
        Ok(Array::from_shape_vec(conserved.dim(), x?).unwrap())
    }

    fn compute_block_fluxes(
        &self,
        pe:      &Array<H::Primitive, Ix2>,
        block:   &BlockData<H::Conserved>,
        physics: &Physics,
        mesh:    &Mesh,
        time:    f64) -> (Array<H::Conserved, Ix2>, Array<H::Conserved, Ix2>)
    {
        use ndarray::{s, azip};
        use ndarray_ops::{map_stencil3};

        // ========================================================================
        let two_body_state = physics.orbital_state_from_time(time);
        let phi = |&(x, y)| two_body_state.gravitational_potential(x, y, physics.softening_length());
        let gx = map_stencil3(&pe, Axis(0), |a, b, c| self.hydro.plm_gradient(physics.plm, a, b, c));
        let gy = map_stencil3(&pe, Axis(1), |a, b, c| self.hydro.plm_gradient(physics.plm, a, b, c));
        let dx = mesh.cell_spacing();
        let dy = mesh.cell_spacing();
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
        .apply_collect(|l, r, f| self.hydro.intercell_flux(&physics, l, r, dx, dy, phi(f), Direction::X));

        // ============================================================================
        let fy = azip![
            cell_data.slice(s![1..-1,..-1]),
            cell_data.slice(s![1..-1, 1..]),
            yf]
        .apply_collect(|l, r, f| self.hydro.intercell_flux(&physics, l, r, dx, dy, phi(f), Direction::Y));

        (fx, fy)
    }

    fn compute_block_updated_solution(
        &self,
        solution: BlockState<H::Conserved>,
        fx:       ArcArray<H::Conserved, Ix2>,
        fy:       ArcArray<H::Conserved, Ix2>,
        block:    &BlockData<H::Conserved>,
        physics:  &Physics,
        mesh:     &Mesh,
        time:     f64,
        dt:       f64) -> BlockState<H::Conserved>
    {
        let dx = mesh.cell_spacing();
        let dy = mesh.cell_spacing();
        let two_body_state = physics.orbital_state_from_time(time);

        let mut ds = ItemizedChange::zeros();

        let u1 = ArcArray::from_shape_fn(solution.conserved.dim(), |i| {
            let m = if physics.need_flux_communication() {
                (i.0 + 1, i.1 + 1)
            } else {
                i
            };
            let du = ((fx[(m.0 + 1, m.1)] - fx[m]) / dx +
                      (fy[(m.0, m.1 + 1)] - fy[m]) / dy) * -dt;
            let uc = solution.conserved[i];
            let u0 = block.initial_conserved[i];
            let (x, y)  = block.cell_centers[i];
            let sources = self.hydro.source_terms(physics, mesh, uc, u0, x, y, time, dt, &two_body_state);

            ds = ds + sources;
            uc + du + sources.total()
        });

        let ds = ds * (dx * dy);
        let de = ds.perturbation(time, physics.orbital_elements());

        BlockState {
            conserved: u1,
            integrated_source_terms: solution.integrated_source_terms + ds,
            orbital_elements_change: solution.orbital_elements_change + de,
        }
    }
}




// ============================================================================
pub fn advance<H>(
    mut state:  State<H::Conserved>,
    hydro:      H,
    mesh:       &Mesh,
    physics:    &Physics,
    fold:       usize,
    block_data: &HashMap<BlockIndex, BlockData<H::Conserved>>,
    runtime:    &tokio::runtime::Runtime) -> Result<(State<H::Conserved>, f64), HydroError>
where
    H: Hydrodynamics + Into<AnyHydro> + 'static
{
    let rk = physics.rk_order;
    let dt = physics.cfl * mesh.cell_spacing() / state.max_signal_speed(&hydro);

    for _ in 0..fold {
        let try_update = |state| try_advance_rk(state, hydro, &block_data, mesh, physics, dt, runtime);
        state = runtime.block_on(rk.try_advance_async(state, try_update, runtime))?;
    }
    Ok((state, dt))
}
