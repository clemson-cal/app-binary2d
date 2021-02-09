# The Circumbinary Disk Code (CDC)
_Simulates accretion from circumbinary gas disks onto stellar and black hole binaries._

__Overview__: This is a Rust port of the Mara3 binary setup, with considerable performance enhancements and more physics, including thermodynamics and massless tracer particles. Unlike the Mara3 version, this version does not yet support fixed mesh refinement (FMR), although it will soon. It is parallelized for multi-threaded execution. Hybrid distributed memory parallelism with MPI will be added in the near future.

CDC is written and maintained by the [Computational Astrophysics Lab](https://jzrake.people.clemson.edu) at the [Clemson University Department of Physics and Astronomy](http://www.clemson.edu/science/departments/physics-astro). Core developers include:

- Jonathan Zrake (Clemson)
- Chris Tiede (NYU)
- Ryan Westernacher-Schneider (Clemson)
- Jack Hu (Clemson)


# Publications

- [Gas-driven inspiral of binaries in thin accretion disks (Tiede+ 2020)](https://ui.adsabs.harvard.edu/abs/2020ApJ...900...43T/abstract)
- [Equilibrium eccentricity of accreting binaries (Zrake+ 2020)](https://ui.adsabs.harvard.edu/abs/2020arXiv201009707Z/abstract)

<sub>Simulations for these publications were run with the [Mara3 implementation](https://github.com/jzrake/Mara3) of this setup. Simulations in forthcoming papers will use the Rust version. The hydrodynamic algorithms are identical.<sub>


# Quick start

__Requirements__: Rust and HDF5

__Build and install__:

```Bash
> git clone https://github.com/clemson-cal/app-binary2d.git
> cd app-binary2d
> cargo install --path .
```

This will install an executable at `.cargo/bin/binary2d`, which should be in your system path. You should now be able to run the code from anywhere on your machine:

```Bash
> cd
> binary2d --help
```

This will print a list of _flags_ for things like the data output directory and the execution strategy. You also have a list of _model parameters_, which control things like the physical setup, the mesh parameters, and the output cadence. When run without anything on the command line, the code will print the model parameters default values and brief description:

```
> binary2d

    block_size............... 1024     Number of grid cells (per direction, per block)
    buffer_rate.............. 1000     Rate of damping in the buffer region [orbital frequency @ domain radius]
    buffer_scale............. 1        Length scale of the buffer transition region
    cfl...................... 0.4      CFL parameter
    cpi...................... 1        Checkpoint interval [Orbits]
    domain_radius............ 6        Half-size of the domain
    mach_number.............. 10       Orbital Mach number of the disk
    nu....................... 0.1      Kinematic viscosity [Omega a^2]
    num_blocks............... 1        Number of blocks per (per direction)
    one_body................. false    Collapse the binary to a single body (validation of central potential)
    plm...................... 1.5      PLM parameter theta [1.0, 2.0] (0.0 reverts to PCM)
    rk_order................. 1        Runge-Kutta time integration order
    sink_radius.............. 0.05     Radius of the sink region
    sink_rate................ 10       Sink rate to model accretion
    softening_length......... 0.05     Gravitational softening length
    tfinal................... 0        Time at which to stop the simulation [Orbits]

    restart file            = none
    effective grid spacing  = 0.0117a
    sink radius / grid cell = 4.2667

write checkpoint data/chkpt.0000.h5
```

You can inspect the output file `data/chkpt.0000.h5`, which is simply the simulation initial condition, using the provided plotting script:

```Bash
> python3 plot.py data/chkpt.0000.h5
```

The model parameters are specified as `key=value` pairs on the command line (no dashes), possibly mixed in with the command line options (which control execution, and do have dashes). For example, to run a simulation for 500 binary orbits, and output the data to a directory called `my-run`, you would type

```Bash
> binary2d --outdir=my-disk tfinal=500
```


# Workflow

__Model parameters__

The _model parameters_ control the grid resolution, domain size, physical conditions, and solver parameters. They are a set of `key=value` pairs which are provided on the command line. All of the model parameters have default values, which define the fiducial simulation setup. You can inspect the model parameters of a particular run by looking in the HDF5 checkpoint files under the `model` group.

Some detail on the model parameters affecting the mesh:

- `domain_radius` controls the half-length of the domain, in units of the binary semi-major axis
- `block_size=64` means the blocks all have shape 64x64
- `num_blocks=8` means there are 8x8 blocks in the domain

When it starts up, the code will print the grid spacing in units of the semi-major axis. Low, medium, and high resolution should aim for `dx=0.04a`, `dx=0.02a` and `dx=0.01a` respectively.

__Checkpoint files__

Checkpoints are HDF5 files containing a snapshot of the simulation state. They are written at a cadence given by the `cpi` model parameter (for checkpoint interval), and will appear in the directory specified by the `--outdir` flag.

Checkpoint files may be inspected either with the `h5ls` utility or with the `h5py` Python module. Checkpoints have the following group structure:
```
model                    Group            # Model parameters
state                    Group            # Simulaton time, iteration, fields, orbital elements
tasks                    Group            # Schedule of side-effects (outputs, etc.)
version                  Dataset {SCALAR} # The code version and git SHA
```

The `/state` group of the checkpoint file has the structure
```
time                     Dataset {SCALAR}
iteration                Dataset {SCALAR}
solution                 Group
```

The `/state/solution` group is a list of blocks in the domain. A block with the name `0:022-010` means "block index (22, 10) on mesh level 0" (note that until the FMR feature is written, all blocks are at level 0). Each block is a group with the following datasets:
```
conserved                Dataset {64, 64} # 2D array of conserved densities
integrated_source_terms  Dataset {SCALAR} # Accumulated source terms on this block
orbital_elements_change  Dataset {SCALAR} # Change in the binary orbit from forcing on this block
```

_Conserved variable arrays_

In isothermal mode, the 2D arrays of conserved quantities have element type `(f64, f64, f64)`. The order of these floats is (surface density, x-momentum, y-momentum). When the energy equation is being solved, the type is `(f64, f64, f64, f64)` with the final element being the total energy density. When loaded into Python via `h5py`, the numpy `dtype` will read `[('0', '<f8'), ('1', '<f8'), ('2', '<f8')]`. This means you must access the conserved variable fields using the strings `'0'`, `'1'`, etc. For example, to load the 2D array of surface density, you would write

```Python
h5f = h5py.File('data/chkpt.0000.h5', 'r')
sigma = h5f['state']['solution']['0:000-000']['conserved']['0']
```

_Source terms_

The dataset `integrated_source_terms` is a nested struct, containing at the outermost level a list of 6 types of source terms:
```
sink1   # Accretion source terms associated with the primary
sink2   # Accretion source terms associated with the secondary
grav1   # Gravitational forces from the primary
grav2   # Gravitational forces from the secondary
buffer  # Source terms due to driving in the buffer region
cooling # Energy losses due to cooling (zeros in isothermal mode)
```
Each of these data members represents the time-integrated conserved quantities added to a given block by one of the source terms. The data type is the same as the data structure of conserved quantities: 3 components in isothermal mode and 4 components in energy-conserving mode. To load the amount of x and y momentum added to block (0, 0) by the gravitational force of the primary, you would write
```Python
source_t = h5f['state']['solution']['0:000-000']['integrated_source_terms']
fx_grav1 = source_t['grav1']['1']
fy_grav1 = source_t['grav1']['2']
```

_Orbital evolution_

The dataset `orbital_elements_change` tabulates the accumulated perturbation to the binary orbit, resulting from the transfer of mass and momentum between the disk and the binary. Its format parallels the source terms dataset: there is one orbital evolution measure per block, per source term category (note that the buffer and cooling terms will be zeros). Each measure of the orbital evolution is a `kepler_two_body::OrbitalElements` struct from the [Kepler two-body crate](https://github.com/clemson-cal/kepler-two-body). The order of the elements is `(a, M, q, e)` where `a` is the semi-major axis, `M` is the binary total mass, `q` is the mass ratio, and `e` is the eccentricity. These items are also accessed like the conserved variable structs, with the key indexes `'0'`, `'1'`, etc.


__Time series data__

The code generates HDF5 time series data at runtime, and stores it as `time_series.h5` in the `--outdir` location. The time series data is an immediate science product, useful mainly to track the binary orbital evolution, although the itemized source terms are also a part of the time series output. Each time series sample is global: it is obtained by totaling the values of `integrated_source_terms` and `orbital_elements_change` over all the blocks. The data structure for the time series sample type can be found in the source code at `main::TimeSeriesSample`.

The time series will continue where it left off in restarted runs (see the section below on restarts). However, if there is already a time series file in your output directory, and you try to start a fresh run (or a less-evolved one), the code will not destroy your existing time series data, unless you run it with the `--truncate` flag.


__Restarting runs__

Runs can be restarted from HDF5 checkpoint files. These files contain a complete snapshot of the simulation state, such that a run which was restarted will be identical to one that ran uninterrupted. A run can be restarted by supplying the name of the checkpoint file, e.g. `--restart=my-run/chkpt.0000.h5`. Alternatively, setting the flag `--restart=my-run` will find the most recent checkpoint file in that directory, and continue from it. In a restarted run, subsequent outputs are placed in the same directory as the checkpoint file, unless a different directory is given with the `--outdir` flag.

All the model parameters are stored in the checkpoint file. Model parameters you provide on the command line when restarting a run will supersede those in the checkpoint, although some, such as the block size and domain radius, can only be specified when a new initial condition is being generated. You'll see an error message if you try to supersede those on a restart. Superseding physical conditions (e.g. disk Mach number or viscosity) or solver parameters (CFL, etc) can be very useful, for example if you want to see how an already well-evolved run responds to these changes.


__Restarting at higher resolution__

It may be useful to evolve a simulation to a quasi-steady state at low resolution to save time, and then restart it at a higher resolution. You can do this with the [`upsample.py`](tools/upsample.py) script in the `tools` directory, for example

```Bash
tools/upsample.py low-res/chkpt.0012.h5 --output high-res/chkpt.0012.h5
```

This script will generate a valid checkpoint file, with the same number of grid blocks, but where the grid spacing on each block is cut in half. The script is thanks to [Jack Hu](https://github.com/Javk5pakfa). It uses piecewise-constant prolongation, which means your new checkpoint will have the same level of pixelization as the original, even though it has more zones. However once you restart from the new checkpoint file, the solution will develop a higher level of detail and accuracy.


# Performance and parallelization

CDC uses a simple block-based domain decomposition to facilitate parallel processing of the solution update. At present, only shared memory parallelization is supported (although a hybrid shared/distributed scheme based on MPI is planned). There are two shared memory parallelization strategies: _message-passing_ and _task-based_. The message-passing mode assigns each block of the domain to its own worker thread, and utilizes channels from the [Crossbeam](https://github.com/crossbeam-rs/crossbeam) crate to pass messages between the workers. The task-based mode assigns a computation on each block to its own _task_, and then spawns those tasks into a [Tokio](https://github.com/tokio-rs/tokio) runtime, which in turn delegates the tasks to a user-specified number of worker threads.


__Using the Tokio runtime__

To use task-based parallelism, pass the `--tokio` flag on the command line. The `--threads` flag will then specify the number of worker threads in the Tokio runtime. This is 1 by default, and setting it to the number of physical cores on your machine should be optimal. `--threads` is ignored in message-passing mode, because message passing requires a single thread per block.


__Performance characteristics__

The performance characteristics with `--tokio` are different from the message-passing mode in a few ways:

- Single-block / single-threaded execution is about 10% slower with `--tokio`
- Parallel execution with blocks ~ threads is as bad as 50% slower
- Parallel execution with blocks >> threads is as much as 2x faster

The last configuration with `--tokio` approaches ~70% scaling on a single node. On a 28-core Mac Pro workstation, it tops 90 million zone-updates per second, per RK step (Mzps), with the block size set to 64. For comparison, message passing with blocks comparable to physical cores tops out at ~100 Mzps with message passing.

Since the optimal mode depends on the mesh configuration, we are tentatively planning to maintain both parallelization strategies. That may not mean that all runtime configurations are supported in both parallelization modes. For example runs with tracer particles may require `--tokio`.


# Developer guidelines

The main branch may updated incrementally with improvements to the code performance, documentation, and non-breaking changes to the user interface or analysis scripts. New features which are expected to remain incomplete for days or weeks are developed on separate _feature branches_. Presently, the work-in-progess features are on the following branches:

- `energy-eqn`: Includes thermodynamics of viscous heating, radiative cooling, etc.
- `tracers`: Massless tracer particles which help to visualize the accretion flow
- `orbital-ev`: Evolution of the binary orbital parameters under gravitational and accretion forces
