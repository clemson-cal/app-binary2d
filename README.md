# The Circumbinary Disk Code (CDC)
_Simulates accretion from circumbinary gas disks onto stellar and black hole binaries._

__Overview__: This is a Rust port of the Mara3 binary setup, with considerable performance enhancements and more physics, including thermodynamics and massless tracer particles. Unlike the Mara3 version, this version does not yet support fixed mesh refinement (FMR), although it will soon. It is parallelized for multi-threaded execution. Hybrid distributed memory parallelism with MPI will be added in the near future.

CDC is written and maintained by the [Computational Astrophysics Lab](https://jzrake.people.clemson.edu) at the [Clemson University Department of Physics and Astronomy](http://www.clemson.edu/science/departments/physics-astro). Core developers include:

- Jonathan Zrake (Clemson)
- Chris Tiede (NYU)
- Ryan Westernacher-Schneider (Clemson)
- Jack Hu (Clemson)

# Recent publications
- [Gas-driven inspiral of binaries in thin accretion disks (Tiede+ 2020)](https://ui.adsabs.harvard.edu/abs/2020ApJ...900...43T/abstract)
- [Equilibrium eccentricity of accreting binaries (Zrake+ 2020)](https://ui.adsabs.harvard.edu/abs/2020arXiv201009707Z/abstract)

<sub>Simulations for these publications were run with the [Mara3 implementation](https://github.com/jzrake/Mara3) of this setup. Simulations in forthcoming papers will use the Rust version. The hydrodynamic algorithms are identical.<sub>

# Quick start

__Requirements__: Rust and HDF5


__Build and install__:

```Bash
> git cone https://github.com/clemson-cal/app-binary2d.git
> cd app-binary2d
> cargo install --path .
```

This will install an executable at `.cargo/bin/binary2d`, which should be in your system path. You should now be able to run the code from anywhere on your machine:

```Bash
> cd
> mkdir binary-code-project; cd binary-code-project
> binary2d --help
```

This will print a list of _command line options_ for things like the data output directory and the execution strategy. You also have a list of _model parameters_, which control things like the physical setup, the mesh parameters, and the output cadence. When run without anything on the command line, the code will print the model parameters default values and brief description:

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
> python3 plot.py data/chkpt.0000h5
```

The model parameters are specified as `key=value` pairs on the command line (no dashes), possibly mixed in with the command line options (which control execution, and do have dashes). For example, to run a simulation for 500 binary orbits, and output the data to a directory called `my-disk`, you would type

```Bash
> binary2d --outdir=my-disk tfinal=500
```


# Workflow


__Model parameters__

The _model parameters_ control the grid resolution, domain size, physical conditions, and solver parameters. They are a set of `key=value` pairs which are provided on the command line. All of the model parameters have default values, which define the fiducial simulation setup. You can inspect the model parameters of a particular run by looking in the HDF5 checkpoint files under the `model` group.


__Restarting runs__

Runs can be restarted from HDF5 checkpoint files (e.g. `chkpt.0000.h5`). These files contain a complete snapshot of the simulation state, such that a run which was restarted will be identical to one that had run uninterrupted. Checkpoints are written at a cadence given by the `cpi` model parameter (for checkpoint interval), and will appear in the directory specified by the `--outdir` flag.

A run can be restarted either by supplying the name of the checkpoint file, e.g. `--restart=my-run/chkpt.0000.h5`, or the run directory itself, in which case the most recent checkpoint file in that directory is used. In a restarted run, the default value of the output directory becomes the parent directory of the checkpoint file. However, you can still supply `--outdir` to override the default, for example if you want to "branch" a run.

All the model parameters are stored in the checkpoint file, so you don't need to provide them again. Any model parameters you do provide will supercede the ones in the checkpoint. Be careful how you use this feature -- superceding parameters which only specified the initial condition would not effect the run, and doing this could confuse you later on. Other model parameters, such as the block size, cannot be changed and would (hopefully!) trigger a runtime error. Changing other parameters such as physical conditions or solver parameters can be very useful, for example if you want to see how an already well-evolved run responds to specific parameters.


# Performance and parallelization

CDC uses a simple block-based domain decomposition to facilitate parallel processing of the solution update. At present, only shared memory parallelization is supported (although distributed memory parallelization using MPI is planned). There are two shared memory parallelization strategies: _message-passing_ and _task-based_. The message-passing mode assigns each block of the domain to its own worker thread, and utilizes channels from the [Crossbeam](https://github.com/crossbeam-rs/crossbeam) crate to pass messages between the workers. The task-based mode assigns a computation on each block to its own _task_, and then spawns those tasks into a [Tokio](https://github.com/tokio-rs/tokio) runtime, which in turn delegates those tasks to a user-specified number of worker threads.


__Using the Tokio runtime__

To use task-based parallism, pass `--tokio` flag on the command line. The `--threads` flag will then specify the number of worker threads in the Tokio runtime. This is 1 by default, and setting it to the number of physical cores on your machine should be optimal. `--threads` is ignored in message-passing mode, because message passing only works with one thread per block.


__Performance characteristics__

The performance characteristics with `--tokio` are different from the message-passing mode in a few ways:

- Single-block / single-threaded execution is about 10% slower with `--tokio`
- Parallel execution with blocks ~ threads is as bad as 50% slower
- Parallel execution with blocks >> threads is as much as 2x faster

The last configuration with `--tokio` approaches ~70% scaling on a single node. On a 28-core Mac Pro workstation, it should top 90 Mzps per RK step, with the block size set to 64. For comparison, message passing with blocks ~ physical cores tops out at ~100 Mzps with message passing.

Since the optimal mode depends on the mesh configuration, we are tentatively planning to maintain both parallelization strategies. That may not mean that all runtime configurations are supported in both parallelization modes. For example runs with tracer particles may require `--tokio`.
