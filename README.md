# The Circumbinary Disk Code (CDC)
_Simulates accretion from circumbinary gas disks onto stellar and black hole binaries._

__Overview__: This is a Rust port of the Mara3 binary setup, with considerable performance enhancements and more physics, including thermodynamics and massless tracer particles. Unlike the Mara3 version, this version does not yet support fixed mesh refinement (FMR), although it will soon. It is parallelized for multi-threaded execution. Hybrid distributed memory parallelism with MPI will be added in the near future.

CDC is written and maintained by the [Computational Astrophysics Lab](https://jzrake.people.clemson.edu) at the [Clemson University Department of Physics and Astronomy](http://www.clemson.edu/science/departments/physics-astro). Core developers include:

- Jonathan Zrake (Clemson)
- Chris Tiede (NYU)
- Ryan Westernacher-Schneider (Clemson)

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
