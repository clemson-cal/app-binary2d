use anyhow::Result;
use binary2d::app::{AnyState, App};
use binary2d::io;
use binary2d::state;
use clap::{App as clapApp, Arg, ArgMatches, crate_authors};
use std::collections::HashMap;

fn argument_parse() -> ArgMatches<'static> {
    let matches = clapApp::new("Checkpoint up-sampling program")
        .author(crate_authors!())
        .arg(
            Arg::with_name("INPUT")
                .help("Input file name")
                .required(true)
                .index(1),
        )
        .arg(
            Arg::with_name("OUTPUT")
                .short("o")
                .help("Output file name")
                .default_value("chkpt.upsampled.cbor"),
        )
        .get_matches();
    matches
}

fn upsample(mut app: App) -> App {
    match app.state {
        AnyState::Isothermal(ref mut state) => {
            let mut new_solution = HashMap::new();
            let mut new_bs = app.config.mesh.block_size;

            new_bs *= 2;

            for (&index, block) in &state.solution {
                let new_conserved = ndarray::Array::from_shape_fn([new_bs, new_bs], |(i, j)| {
                    block.conserved[[i / 2, j / 2]]
                });
                let new_block = state::BlockState {
                    conserved: new_conserved.to_shared(),
                    integrated_source_terms: block.integrated_source_terms,
                    orbital_elements_change: block.orbital_elements_change,
                };
                new_solution.insert(index, new_block);
            }
            state.solution = new_solution;
            app.config.mesh.block_size = new_bs;
            app
        }
        AnyState::Euler(ref mut state) => {
            let mut new_solution = HashMap::new();
            let mut new_bs = app.config.mesh.block_size;

            new_bs *= 2;

            for (&index, block) in &state.solution {
                let new_conserved = ndarray::Array::from_shape_fn([new_bs, new_bs], |(i, j)| {
                    block.conserved[[i / 2, j / 2]]
                });
                let new_block = state::BlockState {
                    conserved: new_conserved.to_shared(),
                    integrated_source_terms: block.integrated_source_terms,
                    orbital_elements_change: block.orbital_elements_change,
                };
                new_solution.insert(index, new_block);
            }
            state.solution = new_solution;
            app.config.mesh.block_size = new_bs;
            app
        }
    }
}

fn main() -> Result<()> {
    let matches = argument_parse();

    let infile = matches.value_of("INPUT").unwrap();
    let outfile = matches.value_of("OUTPUT").unwrap();

    let app: App = binary2d::io::read_cbor(infile)?;
    let config = app.config.clone();
    let new_app = upsample(app);

    println!("checkpoint file .... {}", infile);
    println!("output file ........ {}", outfile);
    println!("number of blocks ... {}", config.mesh.num_blocks);
    println!("old block size ..... {}", config.mesh.block_size);
    println!("new block size ..... {}", new_app.config.mesh.block_size);

    io::write_cbor(&new_app, outfile)?;

    Ok(())
}
