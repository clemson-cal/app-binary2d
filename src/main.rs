use crate::app::App;

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
    // let App{state, tasks, config, ..} = app.clone();

    println!();
    for line in serde_yaml::to_string(&app.config)?.split("\n").skip(1) {
        println!("\t{}", line);
    }

    io::write_cbor(&app, "chkpt.0000.cbor")?;

    Ok(())
}
