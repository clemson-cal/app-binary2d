/**
 * @brief      Code to solve gas-driven binary evolution
 *             
 *
 * @copyright  Jonathan Zrake, Clemson University (2020)
 *
 */




// ============================================================================
use std::time::Instant;
use std::collections::HashMap;
use num::rational::Rational64;
use ndarray::{ArcArray, Ix2};
use clap::Clap;
use kind_config;
use scheme::{State, BlockIndex, BlockData};
use hydro_iso2d::*;

mod io;
mod scheme;
static ORBITAL_PERIOD: f64 = 2.0 * std::f64::consts::PI;




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

    #[clap(long, about="Whether to parallelize on the tokio runtime [default: message passing]")]
    tokio: bool,
}

impl App
{

    /**
     * Return the last filename matching chkpt.*.h5 in the given directory, or
     * an error if the directory was empty.
     */
    fn last_checkpoint_in_directory(&self, path: std::path::PathBuf) -> anyhow::Result<String>
    {
        let path_dir = path.clone();
        let mut path_with_chkpt = path.clone();

        path_with_chkpt.push("chkpt.*.h5");

        let mut checkpoints: Vec<_> = glob::glob(path_with_chkpt.to_str().unwrap())
            .unwrap()
            .map(|x| x.unwrap().to_str().unwrap().to_string())
            .collect();

        checkpoints.sort();

        if let Some(checkpoint) = checkpoints.last() {
            Ok(checkpoint.to_string().into())
        } else {
            Err(anyhow::anyhow!("the restart directory '{}' has no checkpoints", path_dir.to_str().unwrap()))
        }
    }

    /**
     * Determine the name of a restart file, if the --restart option had been
     * given, failing if the checkpoint file cannot be found. If no restart
     * file was requested, then return None.
     */
    fn restart_file(&self) -> anyhow::Result<Option<String>>
    {
        if let Some(restart) = &self.restart {
            let path = std::path::PathBuf::from(&restart);
            if path.is_file() {
                Ok(Some(restart.into()))
            } else if path.is_dir() {
                Ok(Some(self.last_checkpoint_in_directory(path)?))
            } else {
                Err(anyhow::anyhow!("missing restart file '{}'", restart))
            }
        } else {
            Ok(None)
        }
    }

    /**
     * Return the requested data output directory, or an error if it was
     * supposed to be inferred from the restart file and the restart file could
     * not be found.
     */
    fn output_directory(&self) -> anyhow::Result<String>
    {
        if let Some(outdir) = &self.outdir {
            Ok(outdir.into())
        } else if let Some(restart) = &self.restart_file()? {
            Ok(std::path::Path::new(restart).parent().unwrap().to_str().unwrap().into())
        } else {
            Ok("data".into())
        }
    }

    fn restart_model_parameters(&self) -> anyhow::Result<HashMap<String, kind_config::Value>>
    {
        if let Some(restart) = &self.restart_file()? {
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
pub struct Tasks
{
    pub checkpoint_next_time: f64,
    pub checkpoint_count: usize,
    pub call_count_this_run: usize,
    pub tasks_last_performed: Instant,
}

impl Tasks
{
    fn write_checkpoint(&mut self, state: &State, block_data: &Vec<BlockData>, model: &kind_config::Form, app: &App) -> Result<(), Box<dyn std::error::Error>>
    {
        let checkpoint_interval: f64 = model.get("cpi").into();
        let outdir = app.output_directory()?;
        let fname = format!("{}/chkpt.{:04}.h5", outdir, self.checkpoint_count);

        std::fs::create_dir_all(outdir).unwrap();

        self.checkpoint_count += 1;
        self.checkpoint_next_time += checkpoint_interval;

        println!("write checkpoint {}", fname);
        io::write_checkpoint(&fname, &state, &block_data, &model.value_map(), &self).unwrap();
        Ok(())
    }

    fn perform(&mut self, state: &State, block_data: &Vec<BlockData>, mesh: &scheme::Mesh, model: &kind_config::Form, app: &App) -> Result<(), Box<dyn std::error::Error>>
    {
        let elapsed     = self.tasks_last_performed.elapsed().as_secs_f64();
        let mzps        = (mesh.total_zones() as f64) * (app.fold as f64) * 1e-6 / elapsed;
        let mzps_per_cu = mzps / app.compute_units(block_data.len()) as f64 * i64::from(model.get("rk_order")) as f64;

        self.tasks_last_performed = Instant::now();

        if self.call_count_this_run > 0
        {
            println!("[{:05}] orbit={:.3} Mzps={:.2} (per cu-rk={:.2})", state.iteration, state.time / ORBITAL_PERIOD, mzps, mzps_per_cu);
        }
        if state.time / ORBITAL_PERIOD >= self.checkpoint_next_time
        {
            self.write_checkpoint(state, block_data, model, app)?;
        }
        self.call_count_this_run += 1;
        Ok(())
    }
}




// ============================================================================
fn disk_model(xy: (f64, f64)) -> Primitive
{
    let (x, y) = xy;
    let r0 = f64::sqrt(x * x + y * y);
    let ph = f64::sqrt(1.0 / (r0 * r0 + 0.01));
    let vp = f64::sqrt(ph);
    let vx = vp * (-y / r0);
    let vy = vp * ( x / r0);
    return Primitive(1.0, vx, vy);
}

fn initial_conserved(block_index: BlockIndex, mesh: &scheme::Mesh) -> ArcArray<Conserved, Ix2>
{
    mesh.cell_centers(block_index)
        .mapv(disk_model)
        .mapv(Primitive::to_conserved)
        .to_shared()
}

fn initial_state(mesh: &scheme::Mesh) -> State
{
    State{
        time: 0.0,
        iteration: Rational64::new(0, 1),
        conserved: mesh.block_indexes().iter().map(|&i| initial_conserved(i, mesh)).collect()
    } 
}

fn initial_tasks() -> Tasks
{
    Tasks{
        checkpoint_next_time: 0.0,
        checkpoint_count: 0,
        call_count_this_run: 0,
        tasks_last_performed: Instant::now(),
    }
}

fn block_data(block_index: BlockIndex, mesh: &scheme::Mesh) -> BlockData
{
    BlockData{
        cell_centers:    mesh.cell_centers(block_index).to_shared(),
        face_centers_x:  mesh.face_centers_x(block_index).to_shared(),
        face_centers_y:  mesh.face_centers_y(block_index).to_shared(),
        initial_conserved: initial_conserved(block_index, &mesh).to_shared(),
        index: block_index,
    }
}

fn create_solver(model: &kind_config::Form) -> scheme::Solver
{
    let one_body: bool = model.get("one_body").into();

    scheme::Solver{
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
        orbital_elements: kepler_two_body::OrbitalElements(if one_body {1e-9} else {1.0}, 1.0, 1.0, 0.0),
    }
}

fn create_mesh(model: &kind_config::Form) -> scheme::Mesh
{
    scheme::Mesh{
        num_blocks: i64::from(model.get("num_blocks")) as usize,
        block_size: i64::from(model.get("block_size")) as usize,
        domain_radius: model.get("domain_radius").into(),
    }
}

fn create_block_data(mesh: &scheme::Mesh) -> Vec<BlockData>
{
    mesh.block_indexes().iter().map(|&i| block_data(i, &mesh)).collect()
}




// ============================================================================
fn main() -> Result<(), Box<dyn std::error::Error>>
{
    let _silence_hdf5_errors = hdf5::silence_errors();
    let app = App::parse();

    let model = kind_config::Form::new()
        .item("num_blocks"      , 1      , "Number of blocks per (per direction)")
        .item("block_size"      , 100    , "Number of grid cells (per direction, per block)")
        .item("buffer_rate"     , 1e3    , "Rate of damping in the buffer region [orbital frequency @ domain radius]")
        .item("buffer_scale"    , 1.0    , "Length scale of the buffer transition region")
        .item("one_body"        , false  , "Collapse the binary to a single body (validation of central potential)")
        .item("cfl"             , 0.4    , "CFL parameter")
        .item("cpi"             , 1.0    , "Checkpoint interval [Orbits]")
        .item("domain_radius"   , 24.0   , "Half-size of the domain")
        .item("mach_number"     , 10.0   , "Orbital Mach number of the disk")
        .item("nu"              , 0.1    , "Kinematic viscosity [Omega a^2]")
        .item("plm"             , 1.5    , "PLM parameter theta [1.0, 2.0] (0.0 reverts to PCM)")
        .item("rk_order"        , 1      , "Runge-Kutta time integration order")
        .item("sink_radius"     , 0.05   , "Radius of the sink region")
        .item("sink_rate"       , 10.0   , "Sink rate to model accretion")
        .item("softening_length", 0.05   , "Gravitational softening length")
        .item("tfinal"          , 0.0    , "Time at which to stop the simulation [Orbits]")
        .merge_value_map(&app.restart_model_parameters()?)?
        .merge_string_args(&app.model_parameters)?;

    let solver     = create_solver(&model);
    let mesh       = create_mesh(&model);
    let block_data = create_block_data(&mesh);
    let tfinal     = f64::from(model.get("tfinal"));
    let dt         = solver.min_time_step(&mesh);
    let mut state  = app.restart_file()?.map(|r| io::read_state(&r)).unwrap_or_else(|| Ok(initial_state(&mesh)))?;
    let mut tasks  = app.restart_file()?.map(|r| io::read_tasks(&r)).unwrap_or_else(|| Ok(initial_tasks()))?;

    println!();
    for key in &model.sorted_keys() {
        println!("\t{:.<25} {: <8} {}", key, model.get(key), model.about(key));
    }
    println!();
    println!("\trestart file            = {}",      app.restart_file()?.unwrap_or("none".to_string()));
    println!("\tcompute units           = {:.04}",  app.compute_units(block_data.len()));
    println!("\teffective grid spacing  = {:.04}a", solver.effective_resolution(&mesh));
    println!("\tsink radius / grid cell = {:.04}",  solver.sink_radius / solver.effective_resolution(&mesh));
    println!();

    tasks.perform(&state, &block_data, &mesh, &model, &app)?;

    use tokio::runtime::Builder;
    let runtime = Builder::new_multi_thread()
            .worker_threads(app.threads)
            .build()
            .unwrap();

    while state.time < tfinal * ORBITAL_PERIOD
    {
        if app.tokio {
            state = scheme::advance_tokio(state, &block_data, &mesh, &solver, dt, app.fold, &runtime);
        } else {
            scheme::advance_channels(&mut state, &block_data, &mesh, &solver, dt, app.fold);
        }
        tasks.perform(&state, &block_data, &mesh, &model, &app)?;
    }
    Ok(())
}
