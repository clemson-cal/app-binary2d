use serde::{Serialize, Deserialize};
use crate::app::{AnyPrimitive, AnyHydro};
use crate::traits::InitialModel;




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

    fn primitive_at(&self, _hydro: &AnyHydro, xy: (f64, f64)) -> AnyPrimitive {
        todo!()
    }

    fn validate(&self, _hydro: &AnyHydro) -> anyhow::Result<()> {
        Ok(())
    }
}




// ============================================================================
impl InitialModel for InfiniteDiskModel {

    fn primitive_at(&self, _hydro: &AnyHydro, xy: (f64, f64)) -> AnyPrimitive {
        let (x, y) = xy;
        let r = (x * x + y * y).sqrt();
        let sd = 1.0;
        let vp = r.powf(-0.5);
        let vx = vp * (-y / r);
        let vy = vp * ( x / r);
        AnyPrimitive{
            velocity_x: vx,
            velocity_y: vy,
            surface_density: sd,
            surface_pressure: sd * 0.01,
        }
    }

    fn validate(&self, _hydro: &AnyHydro) -> anyhow::Result<()> {
        Ok(())
    }
}
