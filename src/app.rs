use std::{
    ffi::OsStr,
    fs::{File, read_to_string},
    path::Path,
};
use serde::{Serialize, Deserialize};
use yaml_patch::Patch;
use crate::io;
use crate::mesh::Mesh;
use crate::disks::{InfiniteDiskModel, InfiniteAlphaDiskModel, FiniteDiskModel, ResidualTestModel};
use crate::physics::{Euler, Isothermal, Physics};
use crate::state::{ItemizedChange, State};
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
    Isothermal(Isothermal),
    Euler(Euler),
}




/**
 * Any of the simulation initial model types
 */
#[derive(Clone, Serialize, Deserialize, derive_more::From)]
#[serde(deny_unknown_fields, rename_all = "snake_case")]
pub enum AnyModel {
    InfiniteDisk(InfiniteDiskModel),
    InfiniteAlphaDisk(InfiniteAlphaDiskModel),
    FiniteDisk(FiniteDiskModel),
    ResidualTest(ResidualTestModel),
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

    /// How frequently to add a TimeSeriesSample to the time series. If
    /// omitted or nil, no time series samples are taken.
    pub time_series_interval: Option<f64>,

    /// Number of iterations between performing side effects. Larger values
    /// yield less terminal output and more accurate performance estimates.
    pub fold: usize,

    /// Number of worker threads on the Tokio runtime. If omitted or nil,
    /// defaults to 2x the number of physical cores.
    pub num_threads: Option<usize>,

    /// Output directory
    pub output_directory: String,

    /// If this valus is set to something non-zero, then a stack of fallback
    /// states will be retained so the code can proceed in safety mode. If
    /// omitted or nil, no fallback states are saved.
    #[serde(default)]
    pub fallback_stack_size: usize,
}

impl Control {
    pub fn num_threads(&self) -> usize {
        match self.num_threads {
            Some(n) => n,
            None => num_cpus::get() * 2,
        }
    }
}




/**
 * A sample of globally derived data reductions to be accumulated as a time
 * series, parameterized around the type of the conserved quantities.
 */
#[derive(Clone, Serialize, Deserialize)]
pub struct TimeSeriesSample<C> {
    pub time: f64,
    pub integrated_source_terms: ItemizedChange<C>,
    pub orbital_elements_change: ItemizedChange<kepler_two_body::OrbitalElements>,
}

pub type TimeSeries<C> = Vec<TimeSeriesSample<C>>;




/**
 * Enum for any type of time series
 */
#[derive(Clone, Serialize, Deserialize, derive_more::From)]
#[serde(rename_all = "snake_case")]
pub enum AnyTimeSeries {
    Isothermal (TimeSeries<<Isothermal as Hydrodynamics>::Conserved>),
    Euler      (TimeSeries<<Euler      as Hydrodynamics>::Conserved>),
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
    pub time_series: AnyTimeSeries,
    pub config: Configuration,
    pub version: String,
}




// ============================================================================
impl InitialModel for AnyModel {

    fn primitive_at<H: Hydrodynamics>(&self, hydro: &H, xy: (f64, f64)) -> AnyPrimitive {
        match self {
            Self::FiniteDisk  (model)      => model.primitive_at(hydro, xy),
            Self::InfiniteDisk(model)      => model.primitive_at(hydro, xy),
            Self::InfiniteAlphaDisk(model) => model.primitive_at(hydro, xy),
            Self::ResidualTest(model)      => model.primitive_at(hydro, xy),
        }
    }

    fn validate<H: Hydrodynamics>(&self, hydro: &H) -> std::result::Result<(), anyhow::Error> {
        match self {
            Self::FiniteDisk  (model)      => model.validate(hydro),
            Self::InfiniteDisk(model)      => model.validate(hydro),
            Self::InfiniteAlphaDisk(model) => model.validate(hydro),
            Self::ResidualTest(model)      => model.validate(hydro),
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

        let (state, time_series) = match &config.hydro {
            AnyHydro::Isothermal(hydro) => {
                let time_series: Vec<TimeSeriesSample<<Isothermal as Hydrodynamics>::Conserved>> = Vec::new();
                (State::from_model(&config.model, hydro, &config.mesh)?.into(), time_series.into())
            },
            AnyHydro::Euler(hydro) => {
                let time_series: Vec<TimeSeriesSample<<Euler as Hydrodynamics>::Conserved>> = Vec::new();
                (State::from_model(&config.model, hydro, &config.mesh)?.into(), time_series.into())
            },
        };
        let tasks = Tasks::new();

        Ok(Self{state, tasks, time_series, config, version: VERSION_AND_BUILD.to_string()})
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
     * Construct a new App instance from a preset (hard-coded) configuration
     * name, or otherwise an input file if no matching preset is found.
     */
    pub fn from_preset_or_file(input: &str, overrides: Vec<String>) -> anyhow::Result<Self> {
        for (key, yaml) in Self::presets() {
            if input == key {
                return Ok(Self::from_config(serde_yaml::from_str(yaml)?, overrides)?)
            }
        }
        Self::from_file(input, overrides)
    }

    /**
     * Construct a new App instance from references to the member variables.
     */
    pub fn package<H, C>(
        state: &State<C>,
        tasks: &Tasks,
        time_series: &TimeSeries<C>,
        hydro: &H,
        model: &AnyModel,
        mesh: &Mesh,
        physics: &Physics,
        control: &Control) -> Self
    where
        H: Hydrodynamics<Conserved = C> + Into<AnyHydro>,
        C: Conserved,
        AnyState: From<State<C>>,
        AnyTimeSeries: From<TimeSeries<C>>,
    {
        Self {
            state: AnyState::from(state.clone()),
            tasks: tasks.clone(),
            time_series: time_series.clone().into(),
            config: Configuration::package(hydro, model, mesh, physics, control),
            version: VERSION_AND_BUILD.to_string(),
        }
    }

    pub fn presets() -> Vec<(&'static str, &'static str)> {
        vec![
            ("cooling-cbd", include_str!("../setups/cooling-cbd.yaml")),
            ("iso-circular", include_str!("../setups/iso-circular.yaml")),
            ("test-grid-visc", include_str!("../setups/test-grid-visc.yaml")),
        ]
    }
}
