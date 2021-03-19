use std::collections::HashMap;
use num::ToPrimitive;
use num::rational::Rational64;
use serde::{Serialize, Deserialize};
use ndarray::{ArcArray, Ix2};
use kepler_two_body::OrbitalElements;
use godunov_core::runge_kutta;
use crate::app::{AnyModel, TimeSeriesSample}; 
use crate::mesh::{
    BlockIndex,
    Mesh,
};
use crate::traits::{
    Conserved,
    Hydrodynamics,
    InitialModel,
    Zeros,
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
    pub fn from_model<H>(model: &AnyModel, hydro: &H, mesh: &Mesh, index: BlockIndex) -> anyhow::Result<Self>
    where
        H: Hydrodynamics<Conserved = C>
    {
        model.validate(hydro)?;
        let cons = |r| hydro.to_conserved(hydro.from_any(model.primitive_at(hydro, r)));

        Ok(Self {
            conserved: mesh.cell_centers(index).mapv(cons).to_shared(),
            integrated_source_terms: ItemizedChange::zeros(),
            orbital_elements_change: ItemizedChange::zeros(),
        })
    }
}




// ============================================================================
impl<C: Conserved> State<C> {

    /**
     * Generate a state from the given initial model, hydrodynamics instance,
     * and map of grid geometry.
     */
    pub fn from_model<H>(model: &AnyModel, hydro: &H, mesh: &Mesh) -> anyhow::Result<Self>
    where
        H: Hydrodynamics<Conserved = C>
    {
        let time = 0.0;
        let iteration = Rational64::new(0, 1);
        let solution = mesh
            .block_indexes()
            .map(|index| Ok((index, BlockState::from_model(model, hydro, mesh, index)?)))
            .collect::<Result<HashMap<_, _>, anyhow::Error>>()?;
        Ok(Self{time, iteration, solution})
    }

    /**
     * Return the total number of grid zones in this state.
     */
    pub fn total_zones(&self) -> usize {
        self.solution.values().map(|solution| solution.conserved.len()).sum()
    }
}

impl<C: Conserved> State<C> {
    pub fn max_signal_speed<H>(&self, hydro: &H) -> f64
    where
        H: Hydrodynamics<Conserved = C>
    {
        let mut a = 0.0;

        for (_, block) in &self.solution {
            for u in block.conserved.iter() {
                a = hydro.max_signal_speed(*u).max(a);
            }
        }
        a
    }

    pub fn time_series_sample(&self) -> TimeSeriesSample<C> {
        let totals: (ItemizedChange<C>, ItemizedChange<kepler_two_body::OrbitalElements>) = self.solution
            .iter()
            .map(|(_, s)| (s.integrated_source_terms, s.orbital_elements_change))
            .fold((ItemizedChange::zeros(), ItemizedChange::zeros()), |a, b| (a.0 + b.0, a.1 + b.1));

        TimeSeriesSample {
            time: self.time,
            integrated_source_terms: totals.0,
            orbital_elements_change: totals.1,
        }
    }
}




// ============================================================================
impl<Data> ItemizedChange<Data>
where
    Data: std::ops::Add<Output = Data>
{
    pub fn total(self) -> Data {
        self.sink1 + self.sink2 + self.grav1 + self.grav2 + self.buffer + self.cooling + self.fake_mass
    }
}

impl<Data> Zeros for ItemizedChange<Data>
where
    Data: Zeros
{
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

        Self {
            time:      self.time      * (-bf + 1.) + s0.time      * bf,
            iteration: self.iteration * (-br + 1 ) + s0.iteration * br,
            solution: s_avg.into_iter().collect(),
        }
    }
}




// ============================================================================
#[async_trait::async_trait]
impl<C: Conserved + 'static> runge_kutta::WeightedAverageAsync for State<C> {

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

        Self {
            time:      self.time      * (-bf + 1.) + s0.time      * bf,
            iteration: self.iteration * (-br + 1 ) + s0.iteration * br,
            solution: join_all(s_avg).await.into_iter().collect(),
        }
    }
}




// ============================================================================
impl<C> ItemizedChange<C>
where C: Conserved {

    fn pert1(time: f64, delta: (f64, f64, f64), elements: OrbitalElements) -> Result<OrbitalElements, kepler_two_body::UnboundOrbitalState> {
        let (dm, dpx, dpy) = delta;
        Ok(elements.perturb(time, -dm, 0.0, -dpx, 0.0, -dpy, 0.0)? - elements)
    }

    fn pert2(time: f64, delta: (f64, f64, f64), elements: OrbitalElements) -> Result<OrbitalElements, kepler_two_body::UnboundOrbitalState> {
        let (dm, dpx, dpy) = delta;
        Ok(elements.perturb(time, 0.0, -dm, 0.0, -dpx, 0.0, -dpy)? - elements)
    }

    pub fn perturbation(&self, time: f64, elements: OrbitalElements) -> Result<ItemizedChange<OrbitalElements>, kepler_two_body::UnboundOrbitalState> {
        Ok(ItemizedChange {
            sink1:     Self::pert1(time, self.sink1.mass_and_momentum(), elements)?,
            sink2:     Self::pert2(time, self.sink2.mass_and_momentum(), elements)?,
            grav1:     Self::pert1(time, self.grav1.mass_and_momentum(), elements)?,
            grav2:     Self::pert2(time, self.grav2.mass_and_momentum(), elements)?,
            buffer:    OrbitalElements::zeros(),
            cooling:   OrbitalElements::zeros(),
            fake_mass: OrbitalElements::zeros(),
        })
    }
}
