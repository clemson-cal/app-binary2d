#![allow(unused)]


mod app;
mod io;
mod mesh;
mod model;
mod physics;
mod state;
mod tasks;
mod traits;


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

    // let App{state, tasks, config, ..} = App::from_preset_or_file(&input)?.validate()?;

    // for line in serde_yaml::to_string(&config)?.split("\n").skip(1) {
    //     println!("\t{}", line);
    // }
    // println!();

    // let Configuration{hydro, model, mesh, control} = config;

    // match (state, hydro) {
    //     (AgnosticState::Newtonian(state), AgnosticHydro::Newtonian(hydro)) => {
    //         run(state, tasks, hydro, model, mesh, control, outdir)
    //     },
    //     (AgnosticState::Relativistic(state), AgnosticHydro::Relativistic(hydro)) => {
    //         run(state, tasks, hydro, model, mesh, control, outdir)
    //     },
    //     _ => unreachable!(),
    // }

    Ok(())
}
