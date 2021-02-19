use std::{
    ffi::OsStr,
    fs::{File, read_to_string},
    path::Path,
};
use serde::{Serialize, Deserialize};
use yaml_patch::Patch;
use crate::io;
use crate::mesh::Mesh;
use crate::model::{InfiniteDiskModel, FiniteDiskModel};
use crate::physics::{Euler, Isothermal, Physics};
use crate::state::State;
use crate::tasks::Tasks;
use crate::traits::{Conserved, Hydrodynamics, InitialModel};




pub static DESCRIPTION: &str = env!("CARGO_PKG_DESCRIPTION");
pub static VERSION_AND_BUILD: &str = git_version::git_version!(prefix=concat!("v", env!("CARGO_PKG_VERSION"), " "));




/**
 * Either of the supported hydro systems 
 */
#[derive(Clone, Serialize, Deserialize, derive_more::From)]
#[serde(deny_unknown_fields, rename_all = "snake_case")]
pub enum AnyHydro {
    Isothermal (Isothermal),
    Euler      (Euler),
}




/**
 * Any of the simulation initial model types
 */
#[derive(Clone, Serialize, Deserialize, derive_more::From)]
#[serde(deny_unknown_fields, rename_all = "snake_case")]
pub enum AnyModel {
    InfiniteDisk(InfiniteDiskModel),
    FiniteDisk(FiniteDiskModel),
}




/**
 * Either of the simulation state types corresponding to the different hydro
 * systems 
 */
#[derive(Clone, Serialize, Deserialize, derive_more::From)]
#[serde(deny_unknown_fields, rename_all = "snake_case")]
pub enum AnyState {
    Isothermal (State<<Isothermal as Hydrodynamics>::Conserved>),
    Euler      (State<<Euler      as Hydrodynamics>::Conserved>),
}




/**
 * Description of the hydrodynamics state that is compatible with any of the
 * supported hydro systems
 */
#[derive(Clone, Serialize, Deserialize)]
pub struct AnyPrimitive {

    /// X-component of gas velocity
    pub velocity_x: f64,

    /// Y-component of gas velocity
    pub velocity_y: f64,

    /// Vertically integrated mass density, Sigma
    pub surface_density: f64,

    /// Vertically integrated gas pressure, P
    pub surface_pressure: f64,
}




/**
 * Simulation control: how long to run for, how frequently to perform side
 * effects, etc
 */
#[derive(Clone, Serialize, Deserialize)]
pub struct Control {

    /// Number of orbits the simulation will be run to
    pub num_orbits: f64,

    /// Number of orbits between writing checkpoints
    pub checkpoint_interval: f64,

    /// Number of iterations between performing side effects. Larger values
    /// yield less terminal output and more accurate performance estimates.
    pub fold: usize,

    /// Number of worker threads on the Tokio runtime
    pub num_threads: usize,

    /// Output directory
    pub output_directory: String,
}




/**
 * Runtime configuration
 */
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Configuration {
    pub hydro: AnyHydro,
    pub model: AnyModel,
    pub mesh: Mesh,
    pub physics: Physics,
    pub control: Control,
}




/**
 * App state
 */
#[derive(Clone, Serialize, Deserialize)]
pub struct App {
    pub state: AnyState,
    pub tasks: Tasks,
    pub config: Configuration,
    pub version: String,
}




// ============================================================================
impl InitialModel for AnyModel {

    fn primitive_at<H: Hydrodynamics>(&self, hydro: &H, xy: (f64, f64)) -> AnyPrimitive {
        match self {
            Self::FiniteDisk  (model) => model.primitive_at(hydro, xy),
            Self::InfiniteDisk(model) => model.primitive_at(hydro, xy),
        }
    }

    fn validate<H: Hydrodynamics>(&self, hydro: &H) -> std::result::Result<(), anyhow::Error> {
        match self {
            Self::FiniteDisk  (model) => model.validate(hydro),
            Self::InfiniteDisk(model) => model.validate(hydro),
        }
    }
}




// ============================================================================
impl Configuration {

    pub fn package<H>(hydro: &H, model: &AnyModel, mesh: &Mesh, physics: &Physics, control: &Control) -> Self
    where
        H: Hydrodynamics + Into<AnyHydro>,
    {
        Configuration {
            hydro: hydro.clone().into(),
            model: model.clone().into(),
            mesh: mesh.clone(),
            physics: physics.clone(),
            control: control.clone(),
        }
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        Ok(())
    }

    /**
     * Patch this config struct with inputs from the command line. The inputs
     * can be names of YAML files or key=value pairs.
     */
    pub fn patch_from(&mut self, overrides: Vec<String>) -> anyhow::Result<()> {
        for extra_config_str in overrides {
            if extra_config_str.ends_with(".yaml") {
                self.patch_from_reader(File::open(extra_config_str)?)?
            } else {
                self.patch_from_key_val(&extra_config_str)?
            }
        }
        Ok(())
    }
}




// ============================================================================
impl App {

    /**
     * Return self as a result, which will be in an error state if any of the
     * configuration items did not pass validation.
     */
    pub fn validate(self) -> anyhow::Result<Self> {
        Ok(self)
    }

    /**
     * Patch the config struct with inputs from the command line.
     */
    pub fn with_patched_config(mut self, overrides: Vec<String>) -> anyhow::Result<Self> {
        self.config.patch_from(overrides)?;
        Ok(self)
    }

    /**
     * Construct a new App instance from a user configuration.
     */
    pub fn from_config(mut config: Configuration, overrides: Vec<String>) -> anyhow::Result<Self> {

        config.patch_from(overrides)?;

        let state = match &config.hydro {
            AnyHydro::Isothermal(hydro) => {
                State::from_model(&config.model, hydro, &config.mesh)?.into()
            },
            AnyHydro::Euler(hydro) => {
                State::from_model(&config.model, hydro, &config.mesh)?.into()
            },
        };
        let tasks = Tasks::new();

        Ok(Self{state, tasks, config, version: VERSION_AND_BUILD.to_string()})
    }

    /**
     * Construct a new App instance from a file: may be a config.yaml or a
     * chkpt.0000.cbor.
     */
    pub fn from_file(filename: &str, overrides: Vec<String>) -> anyhow::Result<Self> {
        match Path::new(&filename).extension().and_then(OsStr::to_str) {
            Some("yaml") => Self::from_config(serde_yaml::from_str(&read_to_string(filename)?)?, overrides),
            Some("cbor") => Ok(io::read_cbor::<Self>(filename)?.with_patched_config(overrides)?),
            _ => anyhow::bail!("unknown input file type {}", filename.to_string()),
        }
    }

    /**
     * Construct a new App instance from references to the member variables.
     */
    pub fn package<H, C>(state: &State<C>, tasks: &Tasks, hydro: &H, model: &AnyModel, mesh: &Mesh, physics: &Physics, control: &Control) -> Self
    where
        H: Hydrodynamics<Conserved = C> + Into<AnyHydro>,
        C: Conserved,
        AnyState: From<State<C>>,
    {
        Self {
            state: AnyState::from(state.clone()),
            tasks: tasks.clone(),
            config: Configuration::package(hydro, model, mesh, physics, control),
            version: VERSION_AND_BUILD.to_string(),
        }
    }
}
