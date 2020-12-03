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

This will install an executable at `.cargo/bin/binary2d`. Before using the executable, it is necessary to add the path of the excutable to the system PATH variable. To do this, first navigate to the .bashrc file located in your default directory. 

```
cd
vi .bashrc
```

Within the file, there should be a variable named PATH already ( if there isn't one, you can add one using the same format below ). On Palmetto, the PATH to the binary2d executable is: `$HOME/.cargo/bin`. Add this to the end of the line but within the quotation marks so it will look something like this:

```
PATH="$HOME/.local/bin:$HOME/bin:$PATH:$HOME/.cargo/bin"
```

After saving the .bashrc file, go back to the default directory and source it. Now the executable is ready to run. On any Unix machine in general, the `.cargo` directory path will be that of your default directory. To run the executable, use the commands here:

```Bash
> cd
> mkdir binary-code-project; cd binary-code-project
> binary2d --help
```

Note that the checkpoint files will be saved to `data/temp` in the directory that you run the executable. 

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

Upon startup the code will output the corresponding grid resolution in terms of cell size per semi-major axis.

__Data outputs__

The code's data products are a series of _checkpoint_ files in HDF5 format. Checkpoints are written at a cadence given by the `cpi` model parameter (for checkpoint interval), and will appear in the directory specified by the `--outdir` flag.

Checkpoint files may be inspected either with the `h5ls` utility or with the `h5py` Python module. They have the following root-level group structure:

```
iteration    Dataset {SCALAR}   # The iteration number
time         Dataset {SCALAR}   # The simulation time
conserved    Group              # 2D arrays of the conserved quantities, one for each block
model        Group              # The model parameters
tasks        Group              # Times when analysis and output tasks were last performed
```

In isothermal mode, the 2D arrays of conserved quantities have element type `[f64; 3]`. The order of these floats is (surface density, x-momentum, y-momentum). When the energy equation is being solved, the type is `[f64; 4]` and the final element is the total energy.

__Restarting runs__

Runs can be restarted from HDF5 checkpoint files. These files contain a complete snapshot of the simulation state, such that a run which was restarted will be identical to one that ran uninterrupted. A run can be restarted by supplying the name of the checkpoint file, e.g. `--restart=my-run/chkpt.0000.h5`. Alternatively, setting the flag `--restart=my-run` will find the most recent checkpoint file in that directory, and continue from it. In a restarted run, subsequent outputs are placed in the same directory as the checkpoint file, unless a different directory is given with the `--outdir` flag.

All the model parameters are stored in the checkpoint file. Any model parameters you do provide on the command line will supercede those in the checkpoint. Be careful how you use this feature -- superceding model parameters sometimes makes sense, but can other times be confusing. For example, if the parameter is only used once to generate the initial condition, then superceding it would have no effect, other than obscuring the parameters used to start the simulation. Other model parameters, such as the block size or domain radius, should never be superceded; doing so _should_ but is not guaranteed to cause a runtime error. Superceding certain physical conditions (e.g. disk Mach number or viscosity) or solver parameters (CFL, etc) can be very useful, for example if you want to see how an already well-evolved run responds to these changes.


# Performance and parallelization

CDC uses a simple block-based domain decomposition to facilitate parallel processing of the solution update. At present, only shared memory parallelization is supported (although distributed memory parallelization using MPI is planned). There are two shared memory parallelization strategies: _message-passing_ and _task-based_. The message-passing mode assigns each block of the domain to its own worker thread, and utilizes channels from the [Crossbeam](https://github.com/crossbeam-rs/crossbeam) crate to pass messages between the workers. The task-based mode assigns a computation on each block to its own _task_, and then spawns those tasks into a [Tokio](https://github.com/tokio-rs/tokio) runtime, which in turn delegates those tasks to a user-specified number of worker threads.


__Using the Tokio runtime__

To use task-based parallism, pass `--tokio` flag on the command line. The `--threads` flag will then specify the number of worker threads in the Tokio runtime. This is 1 by default, and setting it to the number of physical cores on your machine should be optimal. `--threads` is ignored in message-passing mode, because message passing only works with one thread per block.


__Performance characteristics__

The performance characteristics with `--tokio` are different from the message-passing mode in a few ways:

- Single-block / single-threaded execution is about 10% slower with `--tokio`
- Parallel execution with blocks ~ threads is as bad as 50% slower
- Parallel execution with blocks >> threads is as much as 2x faster

The last configuration with `--tokio` approaches ~70% scaling on a single node. On a 28-core Mac Pro workstation, it tops 90 million zone-updates per second, per RK step (Mzps), with the block size set to 64. For comparison, message passing with blocks ~ physical cores tops out at ~100 Mzps with message passing.

Since the optimal mode depends on the mesh configuration, we are tentatively planning to maintain both parallelization strategies. That may not mean that all runtime configurations are supported in both parallelization modes. For example runs with tracer particles may require `--tokio`.



# Developer guidelines

The main branch may updated incrementally with improvements to the code performance, documentation, and non-breaking changes to the user interface or analysis scripts. New features which are expected to remain incomplete for days or weeks are developed on separate _feature branches_. Presently, the work-in-progess features are on the following branches:

- `energy-eqn`: Includes thermodynamics of viscous heating, radiative cooling, etc.
- `tracers`: Massless tracer particles which help to visualize the accretion flow
- `orbital-ev`: Evolution of the binary orbital parameters under gravitational and accretion forces
