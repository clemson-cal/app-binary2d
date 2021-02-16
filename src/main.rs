use binary2d::state::State;
use binary2d::tasks::Tasks;
use binary2d::mesh::Mesh;
use binary2d::app::{
    self,
    App,
    AnyHydro,
    AnyModel,
    AnyState,
    Configuration,
    Control,
};
use binary2d::traits::{
    Conserved,
    Hydrodynamics,
};
use binary2d::io;
use binary2d::scheme;




// ============================================================================
#[allow(unused)]
fn run<C, H>(mut state: State<C>, mut tasks: Tasks, hydro: H, model: AnyModel, mesh: Mesh, control: Control, outdir: String)
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

    // BOGUS
    let solver = binary2d::physics::Solver {
        buffer_rate: 0.0,
        buffer_scale: 0.0,
        cfl: 0.0,
        domain_radius: 0.0,
        mach_number: 0.0,
        nu: 0.0,
        lambda: 0.0,
        plm: 0.0,
        rk_order: 0,
        sink_radius: 0.0,
        sink_rate: 0.0,
        softening_length: 0.0,
        force_flux_comm: false,
        orbital_elements: kepler_two_body::OrbitalElements(0.0, 0.0, 0.0, 0.0),
        relative_density_floor: 0.0,
        relative_fake_mass_rate: 0.0,
    };

    while state.time < control.num_orbits * 2.0 * std::f64::consts::PI {
        state = scheme::advance(state, hydro, &model, &mesh, &solver, dt, control.fold, &runtime)?;
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
    let Configuration{hydro, model, mesh, control} = config;

    match (state, hydro) {
        (AnyState::Isothermal(state), AnyHydro::Isothermal(hydro)) => {
            run(state, tasks, hydro, model, mesh, control, outdir)
        }
        (AnyState::Euler(state), AnyHydro::Euler(hydro)) => {
            run(state, tasks, hydro, model, mesh, control, outdir)
        }
        _ => unreachable!()
    }
    // io::write_cbor(&app, "chkpt.0000.cbor")?;
}
