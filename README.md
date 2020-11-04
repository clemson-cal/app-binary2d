# binary2d
A rust project written by Dr. Jonathan Zrake that simulates orbital evolution of a binary system 
with a circumbinary disk. 

## Intro
This project produces snapshots of binary orbits and the circumbinary disk over time using 
runge-kutta methods. The output file type is hdf5 and can be plotted through python, and a 
python plotting script is included. 

## Setting up the suite

## Initializing the environment

## Using the suite
The suite has many parameters, and they are listed below with a description. There are two types
of parameters: arguments and options. Arguments have the syntax "[argument name]=[value]", 
and options are used with "--[option]=[value]". Both kinds are used on the command line or in 
your pbs batch job script. An example of command is:

cargo run --release num_blocks=20 domain_radius=5.0 --outdir=data/temp --fold=1000

--------------------------------------------------------------------------------------------------------------

Arguments, [default value]:

num_blocks, 1: Number of blocks per (per direction)
block_size, 100: Number of grid cells (per direction, per block)
buffer_rate, 1e3: Rate of damping in the buffer region [orbital frequency @ domain radius]
buffer_scale, 1.0: Length scale of the buffer transition region
one_body, false: Collapse the binary to a single body (validation of central potential)
cfl, 0.4: CFL parameter
cpi, 1.0: Checkpoint interval [Orbits]
domain_radius, 24.0: Half-size of the domain
mach_number, 10.0: Orbital Mach number of the disk
nu, 0.1: Kinematic viscosity [Omega a^2]
plm, 1.5: PLM parameter theta [1.0, 2.0] (0.0 reverts to PCM)
rk_order, 1: Runge-Kutta time integration order
sink_radius, 0.05: Radius of the sink region
sink_rate, 10.0: Sink rate to model accretion
softening_length, 0.05: Gravitational softening length
tfinal, 0.0: Time at which to stop the simulation [Orbits]

--------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------

Options, [default value]:
outdir, "data": path of output directory. The root of the of the directory will be the project directory.
fold, 1: Number of iterations between side effects. 
restart: Restart file or directory [use latest checkpoint if directory]

--------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------
