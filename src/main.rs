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
use binary2d::physics::Physics;
use binary2d::scheme;
use binary2d::state::State;
use binary2d::tasks::Tasks;
use binary2d::traits::{
    Conserved,
    Hydrodynamics,
};




// ============================================================================
fn side_effects<C, H>(state: &State<C>, tasks: &mut Tasks, hydro: &H, model: &AnyModel, mesh: &Mesh, physics: &Physics, control: &Control, outdir: &str)
    -> anyhow::Result<()>
where
    H: Hydrodynamics<Conserved = C> + Into<AnyHydro>,
    C: Conserved,
    AnyState: From<State<C>>,
{

    if tasks.iteration_message.next_time <= state.time {
        let time = tasks.iteration_message.advance(0.0);
        let mzps = 1e-6 * state.total_zones() as f64 / time * control.fold as f64;
        if tasks.iteration_message.count_this_run > 1 {
            println!("[{:05}] t={:.5} blocks={} Mzps={:.2})", state.iteration, state.time, state.solution.len(), mzps);
        }
    }

    if tasks.write_checkpoint.next_time <= state.time {
        tasks.write_checkpoint.advance(control.checkpoint_interval);
        let filename = format!("{}/chkpt.{:04}.cbor", outdir, tasks.write_checkpoint.count - 1);
        let app = App::package(state, tasks, hydro, model, mesh, physics, control);
        io::write_cbor(&app, &filename)?;
    }

    Ok(())
}




// ============================================================================
#[allow(unused)]
fn run<C, H>(mut state: State<C>, mut tasks: Tasks, hydro: H, model: AnyModel, mesh: Mesh, control: Control, physics: Physics, outdir: String)
    -> anyhow::Result<()>
where
    H: Hydrodynamics<Conserved = C> + Into<AnyHydro> + 'static,
    C: Conserved,
    AnyState: From<State<C>>,
{
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(control.num_threads)
        .build()?;

    // BOGUS
    let dt = 0.001;

    while state.time < control.num_orbits * 2.0 * std::f64::consts::PI {
        side_effects(&state, &mut tasks, &hydro, &model, &mesh, &physics, &control, ".")?;
        state = scheme::advance(state, hydro, &model, &mesh, &physics, dt, control.fold, &runtime)?;
    }

    Ok(())
}




// ============================================================================
fn main() -> anyhow::Result<()> {

    let input = match std::env::args().nth(1) {
        None => anyhow::bail!("no input file given"),
        Some(input) => input,
    };
    let outdir = io::parent_directory(&input);

    println!();
    println!("\t{}", app::DESCRIPTION);
    println!("\t{}", app::VERSION_AND_BUILD);
    println!();
    println!("\tinput file ........ {}", input);
    println!("\toutput drectory ... {}", outdir);

    let app = App::from_file(&input)?.validate()?;

    println!();
    for line in serde_yaml::to_string(&app.config)?.split("\n").skip(1) {
        println!("\t{}", line);
    }

    let App{state, tasks, config, ..} = app;
    let Configuration{hydro, model, mesh, control, physics} = config;

    match (state, hydro) {
        (AnyState::Isothermal(state), AnyHydro::Isothermal(hydro)) => {
            run(state, tasks, hydro, model, mesh, control, physics, outdir)
        }
        (AnyState::Euler(state), AnyHydro::Euler(hydro)) => {
            run(state, tasks, hydro, model, mesh, control, physics, outdir)
        }
        _ => unreachable!()
    }
}
