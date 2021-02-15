use serde::{Serialize, Deserialize};
use crate::mesh::Mesh;
use crate::model::{InfiniteDiskModel, FiniteDiskModel};
use crate::physics::{Isothermal, Euler};
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
impl Configuration {

    // pub fn package<H, M>(hydro: &H, model: &M, mesh: &Mesh, control: &Control) -> Self
    // where
    //     H: Hydrodynamics,
    //     M: InitialModel,
    //     AnyHydro: From<H>,
    //     AnyModel: From<M>,
    // {
    //     Configuration {
    //         hydro: hydro.clone().into(),
    //         model: model.clone().into(),
    //         mesh: mesh.clone(),
    //         control: control.clone(),
    //     }
    // }

    pub fn validate(&self) -> anyhow::Result<()> {
        // self.hydro.validate()?;
        // self.model.validate()?;
        // self.mesh.validate(self.control.start_time)?;
        // self.control.validate()?;
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
        // self.config.validate()?;
        Ok(self)
    }

    // /**
    //  * Construct a new App instance from a user configuration.
    //  */
    // pub fn from_config(mut config: Configuration) -> Result<Self, Error> {

    //     for extra_config_str in std::env::args().skip_while(|s| !s.contains('=')) {
    //         if extra_config_str.ends_with(".yaml") {
    //             config.patch_from_reader(File::open(extra_config_str)?)?
    //         } else {
    //             config.patch_from_key_val(&extra_config_str)?
    //         }
    //     }

    //     let geometry = config.mesh.grid_blocks_geometry(config.control.start_time);
    //     let state = match &config.hydro {
    //         AnyHydro::Newtonian(hydro) => {
    //             AnyState::from(State::from_model(&config.model, hydro, &geometry, config.control.start_time))
    //         },
    //         AnyHydro::Relativistic(hydro) => {
    //             AnyState::from(State::from_model(&config.model, hydro, &geometry, config.control.start_time))
    //         },
    //     };
    //     let tasks = Tasks::new(config.control.start_time);
    //     Ok(Self{state, tasks, config, version: VERSION_AND_BUILD.to_string()})
    // }

    // /**
    //  * Construct a new App instance from a file: may be a config.yaml or a
    //  * chkpt.0000.cbor.
    //  */
    // pub fn from_file(filename: &str) -> Result<Self, Error> {
    //     match Path::new(&filename).extension().and_then(OsStr::to_str) {
    //         Some("yaml") => Self::from_config(serde_yaml::from_str(&read_to_string(filename)?)?),
    //         Some("cbor") => Ok(io::read_cbor(filename)?),
    //         _ => Err(Error::UnknownInputType(filename.to_string())),
    //     }
    // }

    // /**
    //  * Construct a new App instance from a preset (hard-coded) configuration
    //  * name, or otherwise an input file if no matching preset is found.
    //  */
    // pub fn from_preset_or_file(input: &str) -> Result<Self, Error> {
    //     match input {
    //         "jet_in_cloud" => Self::from_config(serde_yaml::from_str(std::include_str!("../setups/jet_in_cloud.yaml"))?),
    //         _ => Self::from_file(input),
    //     }
    // }

    // /**
    //  * Construct a new App instance from references to the member variables.
    //  */
    // pub fn package<H, M, C>(state: &State<C>, tasks: &mut Tasks, hydro: &H, model: &M, mesh: &Mesh, control: &Control) -> Self
    // where
    //     H: Hydrodynamics<Conserved = C>,
    //     M: InitialModel,
    //     C: Conserved,
    //     AnyHydro: From<H>,
    //     AnyModel: From<M>,
    //     AnyState: From<State<C>>,
    // {
    //     Self {
    //         state: AnyState::from(state.clone()),
    //         tasks: tasks.clone(),
    //         config: Configuration::package(hydro, model, mesh, control),
    //         version: VERSION_AND_BUILD.to_string(),
    //     }
    // }
}
