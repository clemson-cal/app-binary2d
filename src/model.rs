use serde::{Serialize, Deserialize};
use crate::app::AnyPrimitive;
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



impl InitialModel for FiniteDiskModel {
    fn primitive_at<H: Hydrodynamics>(&self, _hydro: &H, _: f64) -> AnyPrimitive { todo!() }
    fn validate<H: Hydrodynamics>(&self, _hydro: &H) -> anyhow::Result<()> { Ok(()) }
}



impl InitialModel for InfiniteDiskModel {
    fn primitive_at<H: Hydrodynamics>(&self, _hydro: &H, _: f64) -> AnyPrimitive { todo!() }
    fn validate<H: Hydrodynamics>(&self, _hydro: &H) -> anyhow::Result<()> { Ok(()) }
}
