/**
 * @brief      Code to solve gas-driven binary evolution
 *             
 *
 * @copyright  Jonathan Zrake, Clemson University (2020)
 *
 */




// ============================================================================
use crate::scheme::BlockData;
use std::convert::TryInto;
use std::time::Instant;
use num::rational::Rational64;
use ndarray::{Array, Ix2};
use kind_config;
use hydro_iso2d::*;
use godunov_core::runge_kutta as rk;
use godunov_core::solution_states::SolutionStateArray;
mod io;
mod scheme;

type SolutionState = SolutionStateArray<Conserved, Ix2>;




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

fn initial_conserved(domain_radius: f64, block_size: usize) -> Array<Conserved, Ix2>
{
    scheme::cell_centers(domain_radius, block_size)
        .mapv(initial_primitive)
        .mapv(Primitive::to_conserved)
}

fn initial_state(block_data: &BlockData) -> SolutionState
{
    SolutionState {
        time: 0.0,
        iteration: Rational64::new(0, 1),
        conserved: block_data.initial_conserved.clone(),
    }
}




// ============================================================================
struct TaskList
{
    checkpoint_next_time: f64,
    checkpoint_count: usize,
}

impl TaskList
{
    fn new() -> TaskList
    {
        TaskList{
            checkpoint_next_time:0.0,
            checkpoint_count:0
        }
    }
    fn perform(&mut self, state: &SolutionState, checkpoint_interval: f64, mzps: f64)
    {
        if state.time >= self.checkpoint_next_time
        {
            let fname = format!("chkpt.{:04}.h5", self.checkpoint_count);

            println!("Write checkpoint {}", fname);
            io::write_hdf5(&state, &fname).expect("HDF5 write failed");

            self.checkpoint_count += 1;
            self.checkpoint_next_time += checkpoint_interval;
        }
        println!("[{:05}] t={:.3} Mzps={:.2}", state.iteration, state.time, mzps);
    }
}




// ============================================================================
fn run() -> Result<(), Box<dyn std::error::Error>>
{
    let arg_key_vals = kind_config::to_string_map_from_key_val_pairs(std::env::args().skip(1))?;
    let opts = kind_config::Form::new()
        .item("rk_order"       , 2      , "Runge-Kutta time integration order")
        .item("block_size"     , 100    , "Number of grid cells to use")
        .item("tfinal"         , 0.2    , "Time at which to stop the simulation")
        .item("cpi"            , 0.5    , "Checkpoint interval [Orbits]")
        .item("cfl"            , 0.3    , "CFL parameter")
        .item("plm"            , 1.5    , "PLM parameter theta [1.0, 2.0] (0.0 reverts to PCM)")
        .item("nu"             , 0.1    , "Kinematic viscosity [Omega a^2]")
        .item("mach_number"    , 10.0   , "Orbital Mach number of the disk")
        .item("domain_radius"  , 24.0   , "Half-size of the domain")
        .merge_string_map(arg_key_vals)?;

    let rk_order:            rk::RungeKuttaOrder = opts.get("rk_order")     .as_int().try_into()?;
    let block_size:          usize               = opts.get("block_size")   .as_int() as usize;
    let tfinal:              f64                 = opts.get("tfinal")       .as_float();
    let cpi:                 f64                 = opts.get("cpi")          .as_float();
    let cfl:                 f64                 = opts.get("cfl")          .as_float();
    let plm:                 f64                 = opts.get("plm")          .as_float();
    let nu:                  f64                 = opts.get("nu")           .as_float();
    let mach_number:         f64                 = opts.get("mach_number")  .as_float();
    let domain_radius:       f64                 = opts.get("domain_radius").as_float();

    println!();
    for (key, parameter) in &opts
    {
        println!("\t{:.<24} {: <8} {}", key, parameter.value, parameter.about);
    }
    println!();

    // ============================================================================
    let solver = scheme::Solver{
        sink_rate:        1.0,
        buffer_rate:      1.0,
        buffer_scale:     1.0,
        softening_length: 0.05,
        sink_radius:      0.05,
        domain_radius:   10.0,
        cfl:              cfl,
        plm:              plm,
        nu:               nu,
        mach_number:      mach_number,
        orbital_elements: kepler_two_body::OrbitalElements(1.0, 1.0, 1.0, 0.0),
    };

    let block_data = scheme::BlockData{
        cell_centers:    scheme::cell_centers(domain_radius, block_size),
        face_centers_x:  scheme::face_centers_x(domain_radius, block_size),
        face_centers_y:  scheme::face_centers_y(domain_radius, block_size),
        initial_conserved: initial_conserved(domain_radius, block_size),
    };
    let mut state = initial_state(&block_data);
    let mut tasks = TaskList::new();
    let advance = |s| rk_order.advance(s, |s| scheme::advance(s, &block_data, &solver));

    tasks.perform(&state, cpi, 0.0);

    while state.time < tfinal
    {
        let start = Instant::now();
        state = advance(state);
        tasks.perform(&state, cpi, ((block_size * block_size) as f64) * 1e-6 / start.elapsed().as_secs_f64());
    }
    Ok(())
}




// ============================================================================
fn main()
{
    run().unwrap_or_else(|error| println!("{}", error));
}
