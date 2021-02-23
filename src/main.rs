use binary2d::app::{
    self,
    App,
    AnyHydro,
    AnyModel,
    AnyState,
    Configuration,
    Control,
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
{
    if tasks.iteration_message.next_time <= state.time {
        let time = tasks.iteration_message.advance(0.0);
        let mzps = 1e-6 * state.total_zones() as f64 / time * control.fold as f64;
        if tasks.iteration_message.count_this_run > 1 {
            println!("[{:05}] orbit={:.5} dt={:.2e} Mzps={:.2}", state.iteration, state.time / ORBITAL_PERIOD, dt, mzps);
        }
    }
    if tasks.write_checkpoint.next_time <= state.time {
        std::fs::create_dir_all(&control.output_directory)?;
        tasks.write_checkpoint.advance(control.checkpoint_interval * ORBITAL_PERIOD);
        let filename = format!("{}/chkpt.{:04}.cbor", control.output_directory, tasks.write_checkpoint.count - 1);
        let app = App::package(state, tasks, hydro, model, mesh, physics, control);
        io::write_cbor(&app, &filename)?;
    }
    Ok(())
}




// ============================================================================
fn run<C, H>(
    mut state: State<C>,
    mut tasks: Tasks,
    hydro: H,
    model: AnyModel,
    mesh: Mesh,
    control: Control,
    physics: Physics) -> anyhow::Result<()>
where
    H: Hydrodynamics<Conserved = C> + Into<AnyHydro> + 'static,
    C: Conserved,
    AnyState: From<State<C>>,
{
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(control.num_threads)
        .build()?;

    let block_data = mesh
        .block_indexes()
        .map(|index| Ok((index, scheme::BlockData::from_model(&model, &hydro, &mesh, index)?)))
        .collect::<Result<_, anyhow::Error>>()?;

    // let dt_max = physics.min_time_step(&mesh);
    // let dt_min = dt_max * 0.1;

    side_effects(&state, &mut tasks, &hydro, &model, &mesh, &physics, &control, 0.0)?;

    while state.time < control.num_orbits * ORBITAL_PERIOD {

        // let f = state.time / ORBITAL_PERIOD;
        // let dt = dt_max * f.min(1.0) + dt_min * (1.0 - f).max(0.0);

        let (s, dt) = scheme::advance(state, hydro, &mesh, &physics, control.fold, &block_data, &runtime)?;
        state = s;

        side_effects(&state, &mut tasks, &hydro, &model, &mesh, &physics, &control, dt)?;
    }

    Ok(())
}




// ============================================================================
fn main() -> anyhow::Result<()> {

    println!();
    println!("\t{}", app::DESCRIPTION);
    println!("\t{}", app::VERSION_AND_BUILD);
    println!();

    let input = match std::env::args().nth(1) {
        None => anyhow::bail!("no input file given"),
        Some(input) => input,
    };
    let overrides = std::env::args().skip(2).collect();
    let app = App::from_file(&input, overrides)?.validate()?;

    for line in serde_yaml::to_string(&app.config)?.split("\n").skip(1) {
        println!("\t{}", line);
    }

    let App{state, tasks, config, ..} = app;
    let Configuration{hydro, model, mesh, control, physics} = config;

    match (state, hydro) {
        (AnyState::Isothermal(state), AnyHydro::Isothermal(hydro)) => {
            run(state, tasks, hydro, model, mesh, control, physics)
        }
        (AnyState::Euler(state), AnyHydro::Euler(hydro)) => {
            run(state, tasks, hydro, model, mesh, control, physics)
        }
        _ => unreachable!()
    }
}
