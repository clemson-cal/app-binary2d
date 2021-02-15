use std::collections::HashMap;
use num::ToPrimitive;
use num::rational::Rational64;
use serde::{Serialize, Deserialize};
use ndarray::{ArcArray, Ix2};
use kepler_two_body::OrbitalElements;
use godunov_core::runge_kutta;
use crate::traits::{
    Conserved,
    Hydrodynamics,
    InitialModel,
    Zeros,
};
use crate::mesh::{
    BlockIndex,
    Mesh,
};




/**
 * Changes to the conserved quantities or orbital elements induced by the
 * different types of simulation source terms
 */
#[derive(Clone, Copy, Serialize, Deserialize, derive_more::Add, derive_more::Mul)]
pub struct ItemizedChange<Data> {
    pub sink1:     Data,
    pub sink2:     Data,
    pub grav1:     Data,
    pub grav2:     Data,
    pub buffer:    Data,
    pub cooling:   Data,
    pub fake_mass: Data,
}




/**
 * The solution state for an individual grid block
 */
#[derive(Clone, Serialize, Deserialize)]
pub struct BlockState<C: Conserved> {
    pub conserved: ArcArray<C, Ix2>,
    pub integrated_source_terms: ItemizedChange<C>,
    pub orbital_elements_change: ItemizedChange<OrbitalElements>,
}




/**
 * The full solution state for the simulation
 */
#[derive(Clone, Serialize, Deserialize)]
pub struct State<C: Conserved> {
    pub time: f64,
    pub iteration: Rational64,
    pub solution: HashMap<BlockIndex, BlockState<C>>,
}




// ============================================================================
impl<C: Conserved> BlockState<C> {

    /**
     * Generate a block state from the given initial model, hydrodynamics
     * instance and grid geometry.
     */
    pub fn from_model<M, H>(model: &M, hydro: &H, mesh: &Mesh, index: BlockIndex) -> Self
    where
        M: InitialModel,
        H: Hydrodynamics<Conserved = C>
    {
        let norm = |(x, y)| f64::sqrt(x * x + y * y);
        let cons = |r| hydro.to_conserved(hydro.from_any(&model.primitive_at(hydro, r)));

        Self {
            conserved: mesh.cell_centers(index).mapv(norm).mapv(cons).to_shared(),
            integrated_source_terms: ItemizedChange::zeros(),
            orbital_elements_change: ItemizedChange::zeros(),
        }
    }
}




// ============================================================================
impl<C: Conserved> State<C> {

    /**
     * Generate a state from the given initial model, hydrodynamics instance,
     * and map of grid geometry.
     */
    pub fn from_model<M, H>(model: &M, hydro: &H, mesh: &Mesh) -> Self
    where
        M: InitialModel,
        H: Hydrodynamics<Conserved = C>
    {
        let time = 0.0;
        let iteration = Rational64::new(0, 1);
        let solution = mesh
            .block_indexes()
            .map(|index| (index, BlockState::from_model(model, hydro, mesh, index)))
            .collect::<HashMap<_, _>>();
        Self{time, iteration, solution}
    }

    /**
     * Return the total number of grid zones in this state.
     */
    pub fn total_zones(&self) -> usize {
        self.solution.values().map(|solution| solution.conserved.len()).sum()
    }
}




// ============================================================================
impl<Data> Zeros for ItemizedChange<Data> where Data: Zeros {
    fn zeros() -> Self {
        Self {
            sink1:     Data::zeros(),
            sink2:     Data::zeros(),
            grav1:     Data::zeros(),
            grav2:     Data::zeros(),
            buffer:    Data::zeros(),
            cooling:   Data::zeros(),
            fake_mass: Data::zeros(),
        }
    }
}




// ============================================================================
impl<C: Conserved> runge_kutta::WeightedAverage for BlockState<C> {
    fn weighted_average(self, br: Rational64, s0: &Self) -> Self {
        let s1 = self;
        let bf = br.to_f64().unwrap();
        let u0 = s0.conserved.clone();
        let u1 = s1.conserved.clone();
        let c0 = s0.integrated_source_terms;
        let c1 = s1.integrated_source_terms;
        let e0 = s0.orbital_elements_change;
        let e1 = s1.orbital_elements_change;

        Self {
            conserved:                u1 * (-bf + 1.) + u0 * bf,
            integrated_source_terms:  c1 * (-bf + 1.) + c0 * bf,
            orbital_elements_change:  e1 * (-bf + 1.) + e0 * bf, 
        }
    }
}




// ============================================================================
impl<C: Conserved> runge_kutta::WeightedAverage for State<C> {
    fn weighted_average(self, br: Rational64, s0: &Self) -> Self {
        let bf = br.to_f64().unwrap();
        let s_avg = self.solution
            .into_iter()
            .map(|(index, s1)| (index, s1.weighted_average(br, &s0.solution[&index])));

        Self{
            time:      self.time      * (-bf + 1.) + s0.time      * bf,
            iteration: self.iteration * (-br + 1 ) + s0.iteration * br,
            solution: s_avg.into_iter().collect(),
        }
    }
}




// ============================================================================
#[async_trait::async_trait]
impl<C: Conserved> runge_kutta::WeightedAverageAsync for State<C> {

    type Runtime = tokio::runtime::Runtime;

    async fn weighted_average(self, br: Rational64, s0: &Self, runtime: &Self::Runtime) -> Self {
        use futures::future::join_all;
        use godunov_core::runge_kutta::WeightedAverage;

        let bf = br.to_f64().unwrap();
        let s_avg = self.solution.into_iter().map(|(index, s1)| {
            let s0 = s0.clone();
            async move {
                runtime.spawn(
                    async move {
                        (index, s1.weighted_average(br, &s0.solution[&index]))
                    }
                ).await.unwrap()
            }
        });

        Self{
            time:      self.time      * (-bf + 1.) + s0.time      * bf,
            iteration: self.iteration * (-br + 1 ) + s0.iteration * br,
            solution: join_all(s_avg).await.into_iter().collect(),
        }
    }
}
