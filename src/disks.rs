use crate::app::AnyPrimitive;
use crate::mesh::Mesh;
use crate::traits::{Hydrodynamics, InitialModel};
use libm::{erf, exp, sqrt};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

static G: f64 = 1.0; // gravitational constant
static M: f64 = 1.0; // system mass

#[derive(Clone, Serialize, Deserialize)]
pub struct FiniteDiskModel {
    disk_mass: f64,
    disk_width: f64,
    disk_radius: f64,
    softening_length: f64,
    mach_number: Option<f64>,
}

impl FiniteDiskModel {
    pub fn mach_number<H: Hydrodynamics>(&self, hydro: &H) -> f64 {
        self.mach_number
            .unwrap_or(hydro.global_mach_number().unwrap())
    }

    pub fn failure_radius<H: Hydrodynamics>(&self, hydro: &H) -> f64 {
        let ma = self.mach_number(hydro);
        let r0 = self.disk_radius;
        let dr = self.disk_width;
        0.5 * (r0 + sqrt(r0 * r0 + 2.0 * dr * dr * (ma * ma * hydro.gamma_law_index() - 1.0)))
    }

    pub fn real_mach_number_squared<H: Hydrodynamics>(&self, r: f64, hydro: &H) -> f64 {
        let ma = self.mach_number(hydro);
        let rs = self.softening_length;
        ma * ma
            - ((r * r - 2.0 * rs * rs) / (r * r + rs * rs) - self.dlogrho_dlogr(r))
                / hydro.gamma_law_index()
    }

    pub fn phi_velocity_squared<H: Hydrodynamics>(&self, r: f64, hydro: &H) -> f64 {
        self.sound_speed_squared(r, hydro) * self.real_mach_number_squared(r, hydro)
    }

    pub fn vertically_integrated_pressure<H: Hydrodynamics>(&self, r: f64, hydro: &H) -> f64 {
        self.surface_density(r) * self.sound_speed_squared(r, hydro) / hydro.gamma_law_index()
    }

    pub fn kepler_speed_squared(&self, r: f64) -> f64 {
        let rs = self.softening_length;
        G * M * r * r / (r * r + rs * rs).powf(1.5)
    }

    pub fn sound_speed_squared<H: Hydrodynamics>(&self, r: f64, hydro: &H) -> f64 {
        self.kepler_speed_squared(r) / self.mach_number(hydro).powi(2)
    }

    pub fn surface_density(&self, r: f64) -> f64 {
        let r0 = self.disk_radius;
        let dr = self.disk_width;
        self.sigma0() * exp(-((r - r0) / dr).powi(2))
    }

    fn dlogrho_dlogr(&self, r: f64) -> f64 {
        let r0 = self.disk_radius;
        let dr = self.disk_width;
        -2.0 * r * (r - r0) / dr.powi(2)
    }

    fn sigma0(&self) -> f64 {
        let r0 = self.disk_radius;
        let dr = self.disk_width;
        let md = self.disk_mass;
        let total =
            PI * dr * dr * (exp(-(r0 / dr).powi(2)) + sqrt(PI) * r0 / dr * (1.0 + erf(r0 / dr)));
        md / total
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct InfiniteDiskModel {
    mach_number: Option<f64>,
    surface_density: Option<f64>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct InfiniteAlphaDiskModel {
    mach_number: Option<f64>,
    surface_density: Option<f64>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct ResidualTestModel {}

impl InitialModel for FiniteDiskModel {
    fn primitive_at<H: Hydrodynamics>(&self, hydro: &H, xy: (f64, f64)) -> AnyPrimitive {
        let r0 = self.disk_radius;
        let dr = self.disk_width;
        let gamma = hydro.gamma_law_index();

        let (x, y) = xy;
        let r = (x * x + y * y).sqrt();
        let vp = r.powf(-0.5);
        let vx = vp * (-y / r);
        let vy = vp * (x / r);
        let sd = self.sigma0() * exp(-((r - r0) / dr).powi(2));
        let pg = sd * self.sound_speed_squared(r, hydro) / gamma;

        AnyPrimitive {
            velocity_x: vx,
            velocity_y: vy,
            surface_density: sd,
            surface_pressure: pg,
        }
    }

    fn validate<H: Hydrodynamics>(&self, hydro: &H, mesh: &Mesh) -> anyhow::Result<()> {
        if self.failure_radius(hydro) >= mesh.domain_radius * sqrt(2.0) {
            match (self.mach_number, hydro.global_mach_number()) {
                (Some(_), Some(_)) => anyhow::bail! {
                    "A Mach number must be specified either in hydro or model (not both)"
                },
                (None, None) => anyhow::bail! {
                    "A Mach number must be specified either in hydro or model"
                },
                _ => Ok(()),
            }
        } else {
            anyhow::bail! {concat!{
                "equilibrium disk model fails inside the domain, ",
                "use a larger mach_number, larger disk_width, or a smaller domain_radius."}
            }
        }
    }
}

impl InitialModel for InfiniteDiskModel {
    fn primitive_at<H: Hydrodynamics>(&self, hydro: &H, xy: (f64, f64)) -> AnyPrimitive {
        let (x, y) = xy;
        let r = (x * x + y * y).sqrt();
        let sd = self.surface_density.or(Some(1.0)).unwrap();
        let vp = r.powf(-0.5);
        let vx = vp * (-y / r);
        let vy = vp * (x / r);

        let mach_number = hydro.global_mach_number().or(self.mach_number).unwrap();
        let cs = vp / mach_number;
        let gm = hydro.gamma_law_index();
        let pg = cs * cs * sd / gm;

        AnyPrimitive {
            velocity_x: vx,
            velocity_y: vy,
            surface_density: sd,
            surface_pressure: pg,
        }
    }

    fn validate<H: Hydrodynamics>(&self, hydro: &H, _mesh: &Mesh) -> anyhow::Result<()> {
        match (self.mach_number, hydro.global_mach_number()) {
            (Some(_), Some(_)) => anyhow::bail! {
                "A Mach number must be specified either in hydro or model (not both)"
            },
            (None, None) => anyhow::bail! {
                "A Mach number must be specified either in hydro or model"
            },
            _ => Ok(()),
        }
    }
}

impl InitialModel for InfiniteAlphaDiskModel {
    fn primitive_at<H: Hydrodynamics>(&self, hydro: &H, xy: (f64, f64)) -> AnyPrimitive {
        let (x, y) = xy;
        let r = (x * x + y * y).sqrt();
        let sd = self.surface_density.or(Some(1.0)).unwrap() * r.powf(-3.0 / 5.0);
        let vp = r.powf(-0.5);
        let vx = vp * (-y / r);
        let vy = vp * (x / r);

        let mach_number =
            hydro.global_mach_number().or(self.mach_number).unwrap() * r.powf(-1.0 / 20.0);
        let cs = vp / mach_number;
        let gm = hydro.gamma_law_index();
        let pg = cs * cs * sd / gm;

        AnyPrimitive {
            velocity_x: vx,
            velocity_y: vy,
            surface_density: sd,
            surface_pressure: pg,
        }
    }

    fn validate<H: Hydrodynamics>(&self, hydro: &H, _mesh: &Mesh) -> anyhow::Result<()> {
        match (self.mach_number, hydro.global_mach_number()) {
            (Some(_), Some(_)) => anyhow::bail! {
                "A Mach number must be specified either in hydro or model (not both)"
            },
            (None, None) => anyhow::bail! {
                "A Mach number must be specified either in hydro or model"
            },
            _ => Ok(()),
        }
    }
}

impl InitialModel for ResidualTestModel {
    fn primitive_at<H: Hydrodynamics>(&self, _hydro: &H, xy: (f64, f64)) -> AnyPrimitive {
        let (x, y) = xy;
        let r = (x * x + y * y).sqrt();
        let phi = f64::atan2(x, y);
        let r1 = ((x - 1.0).powi(2) + (y - 1.0).powi(2)).sqrt();
        let r2 = ((x + 1.0).powi(2) + (y + 1.0).powi(2)).sqrt();
        let sd = 1.0 + f64::exp(-r1.powi(2));
        let pg = 1.0 + f64::exp(-r2.powi(2));
        let vp = r.powf(-0.5) * f64::exp(-5.0 / r - r.powi(2) / 3.0);
        let vr = f64::sin(phi - 3.14159 / 4.0) * f64::exp(-5.0 / r - r.powi(2) / 3.0);
        let vx = vp * (-y / r) + vr * (x / r);
        let vy = vp * (x / r) + vr * (y / r);

        AnyPrimitive {
            velocity_x: vx,
            velocity_y: vy,
            surface_density: sd,
            surface_pressure: pg,
        }
    }

    fn validate<H: Hydrodynamics>(&self, _hydro: &H, _mesh: &Mesh) -> anyhow::Result<()> {
        Ok(())
    }
}
