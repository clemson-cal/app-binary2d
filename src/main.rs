/**
 * @brief      Code to solve gas-driven binary evolution
 *
 *
 * @copyright  Jonathan Zrake, Clemson University (2020)
 *
 */




// ============================================================================
mod io;
mod scheme;
static ORBITAL_PERIOD: f64 = 2.0 * std::f64::consts::PI;

use std::time::Instant;
use std::collections::HashMap;
use num::rational::Rational64;
use ndarray::{ArcArray, Ix2};
use clap::Clap;
use kind_config;
use io_logical::verified;
use scheme::{
    State,
    Conserved,
    BlockIndex,
    BlockData,
    Mesh,
    Solver,
    Hydrodynamics,
    Isothermal,
    Euler
};




// ============================================================================
fn main() -> anyhow::Result<()>
{
    let _silence_hdf5_errors = hdf5::silence_errors();
    let app = App::parse();

    let model = kind_config::Form::new()
        .item("block_size"      , 256    , "Number of grid cells (per direction, per block)")
        .item("buffer_rate"     , 1e3    , "Rate of damping in the buffer region [orbital frequency @ domain radius]")
        .item("buffer_scale"    , 1.0    , "Length scale of the buffer transition region [a]")
        .item("cfl"             , 0.5    , "CFL parameter [~0.4-0.7]")
        .item("cpi"             , 1.0    , "Checkpoint interval [Orbits]")
        .item("domain_radius"   , 6.0    , "Half-size of the domain [a]")
        .item("hydro"           , "iso"  , "Hydrodynamics mode: [iso|euler]")
        .item("mach_number"     , 10.0   , "Orbital Mach number of the disk")
        .item("nu"              , 0.1    , "Kinematic viscosity [Omega a^2]")
        .item("num_blocks"      , 1      , "Number of blocks per (per direction)")
        .item("one_body"        , false  , "Collapse the binary to a single body (validation of central potential)")
        .item("plm"             , 1.5    , "PLM parameter theta [1.0, 2.0] (0.0 reverts to PCM)")
        .item("rk_order"        , 2      , "Runge-Kutta time integration order [1|2|3]")
        .item("sink_radius"     , 0.05   , "Radius of the sink region [a]")
        .item("sink_rate"       , 10.0   , "Sink rate to model accretion [Omega]")
        .item("softening_length", 0.05   , "Gravitational softening length [a]")
        .item("stress_dim"      , 2      , "Viscous stress tensor dimensionality [2:Farris14|3:Corrected]")
        .item("tfinal"          , 0.0    , "Time at which to stop the simulation [Orbits]")
        .item("tsi"             , 0.1    , "Time series interval [Orbits]")
        .merge_value_map_freezing(&app.restart_model_parameters()?, &vec!["num_blocks", "block_size", "one_body", "domain_radius"])?
        .merge_string_args(&app.model_parameters)?;

    let hydro: String = model.get("hydro").into();

    match hydro.as_str() {
        "iso"   => run(Driver::new(Isothermal::new()), app, model),
        "euler" => run(Driver::new(Euler::new()), app, model),
        _       => Err(anyhow::anyhow!("no such hydrodynamics mode '{}'", hydro))
    }
}




// ============================================================================
#[derive(Clap)]
struct App
{
    #[clap(about="Model parameters")]
    model_parameters: Vec<String>,

    #[clap(short, long, about="Restart file or directory [use latest checkpoint if directory]")]
    restart: Option<String>,

    #[clap(short, long, about="Output directory [default: data/ or restart directory]")]
    outdir: Option<String>,

    #[clap(long, default_value="1", about="Number of iterations between side effects")]
    fold: usize,

    #[clap(long, default_value="1", about="Number of worker threads to use")]
    threads: usize,

    #[clap(long, about="Parallelize on the tokio runtime [default: message passing]")]
    tokio: bool,

    #[clap(long, about="Do flux communication even if it's not needed [benchmarking]")]
    flux_comm: bool,
}

impl App
{
    fn restart_file(&self) -> anyhow::Result<Option<verified::File>>
    {
        if let Some(restart) = self.restart.clone() {
            Ok(Some(verified::file_or_most_recent_matching_in_directory(restart, "chkpt.????.h5")?))
        } else {
            Ok(None)
        }
    }

    fn restart_rundir(&self) -> anyhow::Result<Option<verified::Directory>>
    {
        Ok(self.restart_file()?.map(|f| f.parent()))
    }

    fn restart_rundir_child(&self, filename: &str) -> anyhow::Result<Option<verified::File>>
    {
        if let Some(restart_rundir) = self.restart_rundir()? {
            Ok(Some(restart_rundir.existing_child(filename)?))
        } else {
            Ok(None)
        }
    }

    fn output_directory(&self) -> anyhow::Result<verified::Directory>
    {
        if let Some(outdir) = self.outdir.clone() {
            Ok(verified::Directory::require(outdir)?)
        } else if let Some(restart) = &self.restart_file()? {
            Ok(restart.parent())
        } else {
            Ok(verified::Directory::require("data".into())?)
        }
    }

    fn restart_model_parameters(&self) -> anyhow::Result<HashMap<String, kind_config::Value>>
    {
        if let Some(restart) = self.restart_file()? {
            Ok(io::read_model(restart)?)
        } else {
            Ok(HashMap::new())
        }
    }

    fn compute_units(&self, num_blocks: usize) -> usize
    {
        if self.tokio {
            num_cpus::get_physical().min(self.threads)
        } else {
            num_cpus::get_physical().min(num_blocks)
        }
    }
}




// ============================================================================
#[derive(hdf5::H5Type)]
#[repr(C)]
pub struct TimeSeriesSample
{
    pub time: f64,
}




// ============================================================================
#[derive(Clone, hdf5::H5Type)]
#[repr(C)]
pub struct RecurringTask
{
    count: usize,
    next_time: f64,
}

impl RecurringTask
{
    pub fn new() -> Self
    {
        Self{
            count: 0,
            next_time: 0.0,
        }
    }
    pub fn advance(&mut self, interval: f64)
    {
        self.count += 1;
        self.next_time += interval;
    }
}




// ============================================================================
#[derive(Clone)]
pub struct Tasks
{
    pub write_checkpoint: RecurringTask,
    pub record_time_sample: RecurringTask,

    pub call_count_this_run: usize,
    pub tasks_last_performed: Instant,
}




// ============================================================================
impl From<Tasks> for Vec<(String, RecurringTask)>
{
    fn from(tasks: Tasks) -> Self {
        vec![
            ("write_checkpoint".into(), tasks.write_checkpoint),
            ("record_time_sample".into(), tasks.record_time_sample),
        ]
    }
}

impl From<Vec<(String, RecurringTask)>> for Tasks
{
    fn from(a: Vec<(String, RecurringTask)>) -> Tasks {
        let task_map: HashMap<_, _> = a.into_iter().collect();
        Tasks {
            write_checkpoint:   task_map.get("write_checkpoint")  .cloned().unwrap_or_else(RecurringTask::new),
            record_time_sample: task_map.get("record_time_sample").cloned().unwrap_or_else(RecurringTask::new),
            call_count_this_run: 0,
            tasks_last_performed: Instant::now(),
        }
    }
}




// ============================================================================
impl Tasks
{
    fn new() -> Self
    {
        Self{
            write_checkpoint: RecurringTask::new(),
            record_time_sample: RecurringTask::new(),
            call_count_this_run: 0,
            tasks_last_performed: Instant::now(),
        }
    }

    fn record_time_sample<C: Conserved>(&mut self,
        state: &State<C>,
        time_series: &mut Vec<TimeSeriesSample>,
        model: &kind_config::Form)
    {
        self.record_time_sample.advance(model.get("tsi").into());
        time_series.push(TimeSeriesSample{time: state.time});
    }

    fn write_checkpoint<C: io::H5Conserved<T>, T: hdf5::H5Type + Clone>(&mut self,
        state: &State<C>,
        time_series: &Vec<TimeSeriesSample>,
        block_data: &Vec<BlockData<C>>,
        model: &kind_config::Form,
        app: &App) -> anyhow::Result<()>
    {
        let outdir = app.output_directory()?;
        let fname_chkpt       = outdir.child(&format!("chkpt.{:04}.h5", self.write_checkpoint.count));
        let fname_time_series = outdir.child("time_series.h5");

        self.write_checkpoint.advance(model.get("cpi").into());

        println!("write checkpoint {}", fname_chkpt);
        io::write_checkpoint(&fname_chkpt, &state, &block_data, &model.value_map(), &self)?;
        io::write_time_series(&fname_time_series, time_series)?;

        Ok(())
    }

    fn perform<C: io::H5Conserved<T>, T: hdf5::H5Type + Clone>(
        &mut self,
        state: &State<C>,
        time_series: &mut Vec<TimeSeriesSample>,
        block_data: &Vec<BlockData<C>>,
        mesh: &Mesh,
        model: &kind_config::Form,
        app: &App) -> anyhow::Result<()>
    {
        let elapsed     = self.tasks_last_performed.elapsed().as_secs_f64();
        let mzps        = (mesh.total_zones() as f64) * (app.fold as f64) * 1e-6 / elapsed;
        let mzps_per_cu = mzps / app.compute_units(block_data.len()) as f64 * i64::from(model.get("rk_order")) as f64;

        self.tasks_last_performed = Instant::now();

        if self.call_count_this_run > 0
        {
            println!("[{:05}] orbit={:.3} Mzps={:.2} (per cu-rk={:.2})", state.iteration, state.time / ORBITAL_PERIOD, mzps, mzps_per_cu);
        }

        if state.time / ORBITAL_PERIOD >= self.record_time_sample.next_time
        {
            self.record_time_sample(state, time_series, model);
        }

        if state.time / ORBITAL_PERIOD >= self.write_checkpoint.next_time
        {
            self.write_checkpoint(state, time_series, block_data, model, app)?;
        }

        self.call_count_this_run += 1;
        Ok(())
    }
}




// ============================================================================
trait InitialModel: Hydrodynamics
{
    fn primitive_at(&self, xy: (f64, f64)) -> Self::Primitive;
}




// ============================================================================
impl InitialModel for Isothermal
{
    fn primitive_at(&self, xy: (f64, f64)) -> Self::Primitive
    {
        let (x, y) = xy;
        let r0 = f64::sqrt(x * x + y * y);
        let ph = f64::sqrt(1.0 / (r0 * r0 + 0.01));
        let vp = f64::sqrt(ph);
        let vx = vp * (-y / r0);
        let vy = vp * ( x / r0);
        return hydro_iso2d::Primitive(1.0, vx, vy);
    }
}

impl InitialModel for Euler
{
    fn primitive_at(&self, _xy: (f64, f64)) -> Self::Primitive
    {
        todo!("InitialModel for Euler");
    }
}




/**
 * This struct provides functions to construct the core simulation data
 * structures. It is parameterized around System: Hydrodynamics and a Model:
 * InitialModel.
 */
struct Driver<System: Hydrodynamics + InitialModel>
{
    system: System,
}

impl<System: Hydrodynamics + InitialModel> Driver<System>
{
    fn new(system: System) -> Self
    {
        Self{system: system}
    }
    fn initial_conserved(&self, block_index: BlockIndex, mesh: &Mesh) -> ArcArray<System::Conserved, Ix2>
    {
        mesh.cell_centers(block_index)
            .mapv(|x| self.system.primitive_at(x))
            .mapv(|p| self.system.to_conserved(p))
            .to_shared()
    }
    fn initial_state(&self, mesh: &Mesh) -> State<System::Conserved>
    {
        State{
            time: 0.0,
            iteration: Rational64::new(0, 1),
            conserved: mesh.block_indexes().iter().map(|&i| self.initial_conserved(i, mesh)).collect()
        } 
    }
    fn initial_time_series(&self) -> Vec<TimeSeriesSample>
    {
        Vec::new()
    }
    fn block_data(&self, mesh: &Mesh) -> Vec<BlockData<System::Conserved>>
    {
        mesh.block_indexes().iter().map(|&block_index| {
            BlockData{
                cell_centers:      mesh.cell_centers(block_index).to_shared(),
                face_centers_x:    mesh.face_centers_x(block_index).to_shared(),
                face_centers_y:    mesh.face_centers_y(block_index).to_shared(),
                initial_conserved: self.initial_conserved(block_index, &mesh).to_shared(),
                index: block_index,
            }
        }).collect()
    }
}




// ============================================================================
fn create_solver(model: &kind_config::Form, app: &App) -> Solver
{
    let one_body: bool = model.get("one_body").into();

    Solver{
        buffer_rate:      model.get("buffer_rate").into(),
        buffer_scale:     model.get("buffer_scale").into(),
        cfl:              model.get("cfl").into(),
        domain_radius:    model.get("domain_radius").into(),
        mach_number:      model.get("mach_number").into(),
        nu:               model.get("nu").into(),
        plm:              model.get("plm").into(),
        rk_order:         model.get("rk_order").into(),
        sink_radius:      model.get("sink_radius").into(),
        sink_rate:        model.get("sink_rate").into(),
        softening_length: model.get("softening_length").into(),
        stress_dim:       model.get("stress_dim").into(),
        force_flux_comm:  app.flux_comm,
        orbital_elements: kepler_two_body::OrbitalElements(if one_body {1e-9} else {1.0}, 1.0, 1.0, 0.0),
    }
}

fn create_mesh(model: &kind_config::Form) -> Mesh
{
    Mesh{
        num_blocks: i64::from(model.get("num_blocks")) as usize,
        block_size: i64::from(model.get("block_size")) as usize,
        domain_radius: model.get("domain_radius").into(),
    }
}




// ============================================================================
fn run<S, C, T>(driver: Driver<S>, app: App, model: kind_config::Form) -> anyhow::Result<()>
    where
    S: 'static + Hydrodynamics<Conserved=C> + InitialModel,
    C: io::H5Conserved<T>,
    T: hdf5::H5Type + Clone
{
    let solver     = create_solver(&model, &app);
    let mesh       = create_mesh(&model);
    let block_data = driver.block_data(&mesh);
    let tfinal     = f64::from(model.get("tfinal"));
    let dt         = solver.min_time_step(&mesh);
    let mut state  = app.restart_file()?.map(io::read_state(&driver.system)).unwrap_or_else(|| Ok(driver.initial_state(&mesh)))?;
    let mut tasks  = app.restart_file()?.map(io::read_tasks).unwrap_or_else(|| Ok(Tasks::new()))?;
    let mut time_series = app.restart_rundir_child("time_series.h5")?.map(io::read_time_series).unwrap_or_else(|| Ok(driver.initial_time_series()))?;

    time_series.retain(|s| s.time < state.time);

    println!();
    for key in &model.sorted_keys() {
        println!("\t{:.<25} {: <8} {}", key, model.get(key), model.about(key));
    }
    println!();
    println!("\trestart file            = {}",      app.restart_file()?.map(|f|f.to_string()).unwrap_or("none".to_string()));
    println!("\tcompute units           = {:.04}",  app.compute_units(block_data.len()));
    println!("\teffective grid spacing  = {:.04}a", solver.effective_resolution(&mesh));
    println!("\tsink radius / grid cell = {:.04}",  solver.sink_radius / solver.effective_resolution(&mesh));
    println!();

    tasks.perform(&state, &mut time_series, &block_data, &mesh, &model, &app)?;

    use tokio::runtime::Builder;

    let runtime = if app.tokio {
        Some(Builder::new_multi_thread().worker_threads(app.threads).build()?)
    } else {
        None
    };

    while state.time < tfinal * ORBITAL_PERIOD
    {
        if app.tokio {
            state = scheme::advance_tokio(state, driver.system, &block_data, &mesh, &solver, dt, app.fold, runtime.as_ref().unwrap());
        } else {
            scheme::advance_channels(&mut state, driver.system, &block_data, &mesh, &solver, dt, app.fold);
        }
        tasks.perform(&state, &mut time_series, &block_data, &mesh, &model, &app)?;
    }
    Ok(())
}
