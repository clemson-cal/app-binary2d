# The Circumbinary Disk Code (CDC)
_Simulates accretion from circumbinary gas disks onto stellar and black hole binaries._

__Overview__: This is a Rust port of the Mara3 binary setup, with considerable performance enhancements and more physics, including thermodynamics and massless tracer particles. Unlike the Mara3 version, this version does not yet support fixed mesh refinement (FMR), although it will soon. It is parallelized for multi-threaded execution. Hybrid distributed memory parallelism with MPI will also be added soon.

CDC is written and maintained by the [Computational Astrophysics Lab](https://jzrake.people.clemson.edu) at the [Clemson University Department of Physics and Astronomy](http://www.clemson.edu/science/departments/physics-astro). Core developers include:

- Jonathan Zrake (Clemson)
- Chris Tiede (NYU)
- Ryan Westernacher-Schneider (Clemson)
- Jack Hu (Clemson)


## Publications

- [Gas-driven inspiral of binaries in thin accretion disks (Tiede+ 2020)](https://ui.adsabs.harvard.edu/abs/2020ApJ...900...43T/abstract)
- [Equilibrium eccentricity of accreting binaries (Zrake+ 2020)](https://ui.adsabs.harvard.edu/abs/2020arXiv201009707Z/abstract)

<sub>Simulations for these publications were run with the [Mara3 implementation](https://github.com/jzrake/Mara3) of this setup. Simulations in forthcoming papers will use the Rust version. The hydrodynamic algorithms are identical.<sub>


## Quick start

__Requirements__: A Rust compiler

__Build and install__:

Make sure you have the Rust compiler installed. [rust-lang.org](https://www.rust-lang.org/learn/get-started) provides this one-liner to get started:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

This will print a list of _flags_ for things like the data output directory and the execution strategy. You also have a list of _model parameters_, which control things like the physical setup, the mesh parameters, and the output cadence. When run without anything on the command line, the code will print the model parameters default values and brief description:
=======
Then check out and build the code:

```bash
git clone https://github.com/clemson-cal/app-binary2d.git
cd app-binary2d
cargo install --path .
```

This will install an executable at `.cargo/bin/binary2d`, which should be in your system path. Running `binary2d` from a command prompt anywhere on your machine, you should see a usage message like this:

```bash

Clemson CAL Circumbinary Disk Code (CDC)
v0.2.0 f6e19be

usage: binary2d <input.yaml|chkpt.cbor> [opts.yaml|group.key=value] [...]
```

If you are hacking on the code, you will want to compile and run the code in the project directory. From the _app-binary2d_ root, you would do this:

```bash
cargo build --release
cargo run
```

In order to inspect and plot the data outputs, you'll need to build the _cdc_loader_ Python extension module:

```bash
source buid_loader.sh
```

The script will build the loader extension, and create a softlink in the project root under the _lib_ subdirectory. It also adds that directory to your `PYTHONPATH` environment variable so that Python scripts which load the output files can find it. You'll need to restore your Python path each time you restart your terminal, either by re-running `source build_loader.sh`, or modifying your `PYTHONPATH` by hand. You should rebuild the loader each time you update your local repository.


## Running a setup

The code comes with sample input files in the _setups_ directory. To run a simulation of an equal-mass binary on a circular orbit in the locally-isothermal mode (uniform Mach number), run the following command:

```bash
binary2d setups/iso-circular.yaml
```

The code will output checkpoint files to a subdirectory _data_ in the present working directory. To plot the code output after say 10 orbits, run

```bash
tools/plot data/chkpt.0010.cbor
```

This will show a relief of the surface density. You can explore more options for plotting with `tools/plot -h`. To lauch a custom run with different parameters, just copy one of the preset parameter files to your working directory, modify it, and then run the code pointing to the location of your custom parameter file. As an alternative, you can run the code against a "base" parameter file, and supply extra parameters on the command line to override the ones in the YAML file. For example, to run the _iso-circular_ setup to 1000 orbits, you would do `binary2d setups/iso-circular.yaml control.num_orbits=1000`. You can also supply the names of additional YAML files on the command line, and any properties listed in those files will replace those in the original parameter file. Input files and parameters given later in the command line sequence take priority.


## Quick workflow reference

- __Restart a run__: `binary2d chkpt.1234.cbor`
- __Query the configuration__: `tools/show-config chkpt.1234.cbor`
- __Extract time series data__: `tools/time-series chkpt.1234.cbor`
