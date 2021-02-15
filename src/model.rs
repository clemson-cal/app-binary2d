use serde::{Serialize, Deserialize};
use crate::app::{AnyPrimitive, AnyHydro};
use crate::traits::{Hydrodynamics, InitialModel};




#[derive(Clone, Serialize, Deserialize)]
pub struct FiniteDiskModel {

    width: f64,

    radius: f64,

    mach_number: Option<f64>,
}




#[derive(Clone, Serialize, Deserialize)]
pub struct InfiniteDiskModel {

    accretion_rate: f64,

    mach_number: Option<f64>,
}




// ============================================================================
impl InitialModel for FiniteDiskModel {

    fn primitive_at(&self, _hydro: &AnyHydro, _: f64) -> AnyPrimitive {
        todo!()
    }

    fn validate(&self, _hydro: &AnyHydro) -> anyhow::Result<()> {
        Ok(())
    }
}




// ============================================================================
impl InitialModel for InfiniteDiskModel {

    fn primitive_at(&self, _hydro: &AnyHydro, _: f64) -> AnyPrimitive {
        todo!()
    }

    fn validate(&self, _hydro: &AnyHydro) -> anyhow::Result<()> {
        Ok(())
    }
}
