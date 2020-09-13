/**
 * @brief      Code to solve gas-driven binary evolution
 *             
 *
 * @copyright  Jonathan Zrake, Clemson University (2020)
 *
 */




// ============================================================================
use std::time::Instant;
use num::rational::Rational64;
use ndarray::{Array, Ix2};
use kind_config;
use hydro_iso2d::*;
// use std::convert::TryInto;
// use godunov_core::runge_kutta as rk;
mod io;
mod scheme;


static ORBITAL_PERIOD: f64 = 2.0 * std::f64::consts::PI;




// ============================================================================
pub type BlockIndex = (usize, usize);

pub struct BlockData
{
    pub initial_conserved: Array<Conserved, Ix2>,
    pub cell_centers:   Array<(f64, f64), Ix2>,
    pub face_centers_x: Array<(f64, f64), Ix2>,
    pub face_centers_y: Array<(f64, f64), Ix2>,
    pub index:          BlockIndex,
}

pub struct State
{
    time: f64,
    iteration: Rational64,
    conserved: Vec<Array<Conserved, Ix2>>,
}




// ============================================================================
fn initial_primitive(xy: (f64, f64)) -> Primitive
{
    let (x, y) = xy;
    let r0 = f64::sqrt(x * x + y * y);
    let ph = f64::sqrt(1.0 / (r0 * r0 + 0.01));
    let vp = f64::sqrt(ph);
    let vx = vp * (-y / r0);
    let vy = vp * ( x / r0);
    return Primitive(1.0, vx, vy);
}

fn initial_conserved(mesh: &scheme::Mesh, block_index: (usize, usize)) -> Array<Conserved, Ix2>
{
    mesh.cell_centers(block_index)
        .mapv(initial_primitive)
        .mapv(Primitive::to_conserved)
}

fn block_data(block_index: BlockIndex, mesh: &scheme::Mesh) -> BlockData
{
    BlockData{
        cell_centers:    mesh.cell_centers(block_index),
        face_centers_x:  mesh.face_centers_x(block_index),
        face_centers_y:  mesh.face_centers_y(block_index),
        initial_conserved: initial_conserved(&mesh, block_index),
        index: block_index,
    }
}




// ============================================================================
struct TaskList
{
    checkpoint_next_time: f64,
    checkpoint_count: usize,
    tasks_last_performed: Instant,
}

impl TaskList
{
    fn new() -> TaskList
    {
        TaskList{
            checkpoint_next_time: 0.0,
            checkpoint_count: 0,
            tasks_last_performed: Instant::now(),
        }
    }
    fn perform(&mut self, state: &State, mesh: &scheme::Mesh, block_data: &Vec<BlockData>, run_config: &kind_config::Form)
    {
        let checkpoint_interval: f64 = run_config.get("cpi").into();
        let elapsed     = self.tasks_last_performed.elapsed().as_secs_f64();
        let kzps_per_cu = (mesh.zones_per_block() as f64) * 1e-3 / elapsed;
        let kzps        = (mesh.total_zones()     as f64) * 1e-3 / elapsed;
        self.tasks_last_performed = Instant::now();

        if state.time > 0.0
        {
            println!("[{:05}] orbit={:.3} kzps={:.0} (per cu={:.0})", state.iteration, state.time / ORBITAL_PERIOD, kzps, kzps_per_cu);
        }
        if state.time / ORBITAL_PERIOD >= self.checkpoint_next_time
        {
            let fname = format!("data/chkpt.{:04}.h5", self.checkpoint_count);

            self.checkpoint_count += 1;
            self.checkpoint_next_time += checkpoint_interval;

            println!("Write checkpoint {}", fname);
            io::write_checkpoint(&fname, &state, &block_data, &run_config).expect("HDF5 write failed");
        }
    }
}




// ============================================================================
fn run() -> Result<(), Box<dyn std::error::Error>>
{
    let opts = kind_config::Form::new()
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
        .item("rk_order"        , 2      , "Runge-Kutta time integration order")
        .item("sink_radius"     , 0.05   , "Radius of the sink region")
        .item("sink_rate"       , 10.0   , "Sink rate to model accretion")
        .item("softening_length", 0.05   , "Gravitational softening length")
        .item("tfinal"          , 1.0    , "Time at which to stop the simulation [Orbits]")
        .merge_string_args(std::env::args().skip(1))?;

    // let rk_order:            rk::RungeKuttaOrder = i64::from(opts.get("rk_order")).try_into()?;
    let block_size:          usize               = i64::from(opts.get("block_size")) as usize;
    let one_body:            bool                = opts.get("one_body")     .into();
    let tfinal:              f64                 = opts.get("tfinal")       .into();

    println!();
    for key in &opts.sorted_keys() {
        println!("\t{:.<24} {: <8} {}", key, opts.get(key), opts.about(key));
    }
    println!();

    // ============================================================================
    let solver = scheme::Solver{
        buffer_rate:      opts.get("buffer_rate").into(),
        buffer_scale:     opts.get("buffer_scale").into(),
        cfl:              opts.get("cfl").into(),
        domain_radius:    opts.get("domain_radius").into(),
        mach_number:      opts.get("mach_number").into(),
        nu:               opts.get("nu").into(),
        orbital_elements: kepler_two_body::OrbitalElements(if one_body {1e-9} else {1.0}, 1.0, 1.0, 0.0),
        plm:              opts.get("plm").into(),
        sink_radius:      opts.get("sink_radius").into(),
        sink_rate:        opts.get("sink_rate").into(),
        softening_length: opts.get("softening_length").into(),
    };

    let mesh = scheme::Mesh{
        num_blocks: i64::from(opts.get("num_blocks")) as usize,
        block_size: block_size,
        domain_radius: opts.get("domain_radius").into(),
    };

    let mut state = State{
        time: 0.0,
        iteration: Rational64::new(0, 1),
        conserved: mesh.block_indexes().into_iter().map(|i| initial_conserved(&mesh, i)).collect()
    };

    let block_data: Vec<BlockData> = mesh.block_indexes().into_iter().map(|i| block_data(i, &mesh)).collect();
    let dt = solver.min_time_step(&mesh);
    let mut tasks = TaskList::new();

    tasks.perform(&state, &mesh, &block_data, &opts);

    while state.time < tfinal * ORBITAL_PERIOD
    {
        scheme::advance_super(&mut state, &block_data, &mesh, &solver, dt);
        tasks.perform(&state, &mesh, &block_data, &opts);
    }
    Ok(())
}




// ============================================================================
fn main()
{
    run().unwrap_or_else(|error| println!("{}", error));
}
