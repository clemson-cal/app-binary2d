use binary2d::app::{
    self,
    App,
    AnyHydro,
    AnyModel,
    AnyState,
    AnyTimeSeries,
    Configuration,
    Control,
    TimeSeries,
};
use binary2d::io;
use binary2d::mesh::Mesh;
use binary2d::physics::{ORBITAL_PERIOD, Physics};
use binary2d::scheme;
use binary2d::state::State;
use binary2d::tasks::Tasks;
use binary2d::traits::{
    Conserved,
    Hydrodynamics,
};




// ============================================================================
fn side_effects<C, H>(
    state: &State<C>,
    tasks: &mut Tasks,
    time_series: &mut TimeSeries<C>,
    hydro: &H,
    model: &AnyModel,
    mesh: &Mesh,
    physics: &Physics,
    control: &Control,
    dt: f64) -> anyhow::Result<()>
where
    H: Hydrodynamics<Conserved = C> + Into<AnyHydro>,
    C: Conserved,
    AnyState: From<State<C>>,
    AnyTimeSeries: From<TimeSeries<C>>,
{
    if tasks.iteration_message.next_time <= state.time {
        let time = tasks.iteration_message.advance(0.0);
        let mzps = 1e-6 * state.total_zones() as f64 / time * control.fold as f64;
        if tasks.iteration_message.count_this_run > 1 {
            println!("[{:05}] orbit={:.5} steps/orbit={:.2e} Mzps={:.2} Mzps-rk/cu={:.2}",
                state.iteration,
                state.time / ORBITAL_PERIOD,
                ORBITAL_PERIOD / dt,
                mzps,
                mzps * physics.rk_substeps() as f64 / num_cpus::get().min(control.num_threads()) as f64)
        }
    }
    if tasks.record_time_series.next_time <= state.time {
        if let Some(interval) = control.time_series_interval {
            tasks.record_time_series.advance(interval * ORBITAL_PERIOD);
            time_series.push(state.time_series_sample());

            println!("record time series sample #{}", time_series.len());
        }
    }
    if tasks.report_progress.next_time <= state.time {
        if tasks.report_progress.count > 0 {
            println!("");
            println!("\torbits / hour ........ {:0.2}", 1.0 / tasks.report_progress.elapsed_hours());
            println!("\truntime so far ....... {:0.3} hours", tasks.simulation_startup.elapsed_hours());
            println!("");
        }
        tasks.report_progress.advance(ORBITAL_PERIOD);
    }
    if tasks.write_checkpoint.next_time <= state.time {

        std::fs::create_dir_all(&control.output_directory)
            .map_err(|_| anyhow::anyhow!("unable to create output directory {}", control.output_directory))?;

        let filename = format!("{}/chkpt.{:04}.cbor", control.output_directory, tasks.write_checkpoint.count);
        let app = App::package(state, tasks, time_series, hydro, model, mesh, physics, control);

        if tasks.write_checkpoint.count_this_run > 0 || tasks.write_checkpoint.count == 0 {
            io::write_cbor(&app, &filename)?;
        }
        tasks.write_checkpoint.advance(control.checkpoint_interval * ORBITAL_PERIOD);
    }
    Ok(())
}




struct LastCrash {
    time: f64,
}




// ============================================================================
fn run<C, H>(
    mut state: State<C>,
    mut tasks: Tasks,
    mut time_series: TimeSeries<C>,
    hydro: H,
    model: AnyModel,
    mesh: Mesh,
    control: Control,
    physics: Physics) -> anyhow::Result<()>
where
    H: Hydrodynamics<Conserved = C> + Into<AnyHydro> + 'static,
    C: Conserved,
    AnyState: From<State<C>>,
    AnyTimeSeries: From<TimeSeries<C>>,
{
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(control.num_threads())
        .build()?;


    let block_data = mesh
        .block_indexes()
        .map(|index| Ok((index, scheme::BlockData::from_model(&model, &hydro, &mesh, index)?)))
        .collect::<Result<_, anyhow::Error>>()?;


    let mut fallback_stack = std::collections::VecDeque::new();
    let mut crash: Option<LastCrash> = None;
    let mut dt = 0.0;
    side_effects(&state, &mut tasks, &mut time_series, &hydro, &model, &mesh, &physics, &control, dt)?;


    let safety = |mut physics: Physics, crash: &Option<LastCrash>| {
        if crash.is_some() {
            if let Some(safe_cfl) = physics.safe_cfl {
                physics.cfl = safe_cfl
            }
            if let Some(safe_plm) = physics.safe_plm {
                physics.plm = safe_plm
            }
            if let Some(safe_mach_ceiling) = physics.safe_mach_ceiling {
                physics.mach_ceiling = Some(safe_mach_ceiling)
            }
            if let Some(safe_rk_order) = physics.safe_rk_order {
                physics.rk_order = safe_rk_order
            }
        }
        physics
    };


    while state.time < control.num_orbits * ORBITAL_PERIOD {

        if control.fallback_stack_size > 1 {
            if fallback_stack.len() > control.fallback_stack_size {
                fallback_stack.pop_front();
            }
            fallback_stack.push_back((state.clone(), time_series.clone(), tasks.clone()));
        }

        state = match scheme::advance(
            state.clone(),
            hydro,
            &mesh,
            &safety(physics.clone(), &crash),
            control.fold,
            &mut dt,
            &block_data,
            &runtime) {

            Ok(next_state) => {
                if let Some(last_crash) = &crash {
                    if last_crash.time + physics.safe_mode_duration * ORBITAL_PERIOD < next_state.time {
                        crash = None;
                        println!("surpassed previous crash time plus a margin of {} orbits, proceeding in normal mode", physics.safe_mode_duration);
                    }
                }
                next_state
            }
            Err(e) => {

                if fallback_stack.is_empty() || crash.is_some() {

                    std::fs::create_dir_all(&control.output_directory)
                        .map_err(|_| anyhow::anyhow!("unable to create output directory {}", control.output_directory))?;
            
                    let filename = format!("{}/chkpt.fail.cbor", control.output_directory);
                    let app = App::package(&state, &tasks, &time_series, &hydro, &model, &mesh, &physics, &control);
            
                    if tasks.write_checkpoint.count_this_run > 0 || tasks.write_checkpoint.count == 0 {
                        io::write_cbor(&app, &filename)?;
                    }
                    return Err(e.into())
                }
                let (former_state, former_time_series, former_tasks) = fallback_stack[0].clone();

                crash = Some(LastCrash { time: fallback_stack.back().unwrap().0.time });
                fallback_stack.clear();

                println!("{} {}", e.source, e);
                println!("rewind to orbit {:.5}, proceed in safety mode...", former_state.time / ORBITAL_PERIOD);

                time_series = former_time_series;
                tasks = former_tasks;
                former_state
            }
        };

        side_effects(&state, &mut tasks, &mut time_series, &hydro, &model, &mesh, &physics, &control, dt)?;
    }
    Ok(())
}




// ============================================================================
fn main() -> anyhow::Result<()> {

    println!();
    println!("{}", app::DESCRIPTION);
    println!("{}", app::VERSION_AND_BUILD);
    println!();

    match std::env::args().nth(1) {
        None => {
            println!("usage: binary2d <input.yaml|chkpt.cbor|preset> [opts.yaml|group.key=value] [...]");
            println!();
            println!("These are the preset model setups:");
            println!();
            for (key, _) in App::presets() {
                println!("  {}", key);
            }
            println!();
            println!("To run any of these presets, run e.g. `binary2d iso-circular`.");
            Ok(())
        }
        Some(input) => {
            let overrides = std::env::args().skip(2).collect();
            let app = App::from_preset_or_file(&input, overrides)?.validate()?;

            for line in serde_yaml::to_string(&app.config)?.split("\n").skip(1) {
                println!("{}", line);
            }

            let App{state, tasks, time_series, config, ..} = app;
            let Configuration{hydro, model, mesh, control, physics} = config;

            println!("worker threads ...... {}", control.num_threads());
            println!("compute cores ....... {}", num_cpus::get());
            println!("grid spacing ........ {:.3}a", mesh.cell_spacing());
            println!();

            match (state, time_series, hydro) {
                (AnyState::Isothermal(state), AnyTimeSeries::Isothermal(time_series), AnyHydro::Isothermal(hydro)) => {
                    run(state, tasks, time_series, hydro, model, mesh, control, physics)
                }
                (AnyState::Euler(state), AnyTimeSeries::Euler(time_series), AnyHydro::Euler(hydro)) => {
                    run(state, tasks, time_series, hydro, model, mesh, control, physics)
                }
                _ => unreachable!()
            }
        }
    }
}
