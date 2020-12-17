use std::collections::HashMap;
use hdf5::{File, Group, H5Type};
use io_logical::verified;
use io_logical::nicer_hdf5;
use io_logical::nicer_hdf5::{H5Read, H5Write};
use crate::Tasks;

use crate::traits::{
    Hydrodynamics,
    Conserved,
};

use crate::physics::{
    ItemizedChange,
};

use crate::scheme::{
    State,
    BlockSolution,
    BlockData,
};




// ============================================================================
impl nicer_hdf5::H5Read for Tasks
{
    fn read(group: &Group, name: &str) -> hdf5::Result<Self>
    {
        nicer_hdf5::read_as_keyed_vec(group, name)
    }    
}
impl nicer_hdf5::H5Write for Tasks
{
    fn write(&self, group: &Group, name: &str) -> hdf5::Result<()>
    {
        nicer_hdf5::write_as_keyed_vec(self.clone(), group, name)
    }
}




// ============================================================================
fn write_state<C: Conserved>(group: &Group, state: &State<C>, block_data: &Vec<BlockData<C>>) -> hdf5::Result<()>
{
    type E = kepler_two_body::OrbitalElements;

    let state_group = group.create_group("state")?;
    let solution_group = state_group.create_group("solution")?;

    for (b, s) in block_data.iter().zip(&state.solution)
    {
        let block_group = solution_group.create_group(&format!("0:{:03}-{:03}", b.index.0, b.index.1))?;
        s.conserved.write(&block_group, "conserved")?;
        block_group.new_dataset::<ItemizedChange<C>>().create("integrated_source_terms", ())?.write_scalar(&s.integrated_source_terms)?;
        block_group.new_dataset::<ItemizedChange<E>>().create("orbital_elements_change", ())?.write_scalar(&s.orbital_elements_change)?;
    }
    state.time.write(&state_group, "time")?;
    state.iteration.write(&state_group, "iteration")?;
    Ok(())
}

pub fn read_state<H: Hydrodynamics<Conserved=C>, C: Conserved>(file: &verified::File, _: &H) -> hdf5::Result<State<C>>
{
    let file = File::open(file.as_str())?;
    let state_group = file.group("state")?;
    let solution_group = state_group.group("solution")?;
    let mut solution = Vec::new();

    for key in solution_group.member_names()?
    {
        let block_group = solution_group.group(&key)?;
        let s = BlockSolution{
            conserved: ndarray::Array::read(&block_group, "conserved")?.to_shared(),
            integrated_source_terms: block_group.dataset("integrated_source_terms")?.read_scalar()?,
            orbital_elements_change: block_group.dataset("orbital_elements_change")?.read_scalar()?,
        };
        solution.push(s);
    }
    let time      = f64::read(&state_group, "time")?;
    let iteration = num::rational::Ratio::<i64>::read(&state_group, "iteration")?;

    let result = State{
        solution: solution,
        time: time,
        iteration: iteration,
    };
    Ok(result)
}

fn write_tasks(group: &Group, tasks: &Tasks) -> hdf5::Result<()>
{
    tasks.write(group, "tasks")
}

pub fn read_tasks(file: &verified::File) -> hdf5::Result<Tasks>
{
    let file = File::open(file.as_str())?;
    Tasks::read(&file, "tasks")
}

fn write_model(group: &Group, model: &HashMap::<String, kind_config::Value>) -> hdf5::Result<()>
{
    kind_config::io::write_to_hdf5(&group.create_group("model")?, &model)
}

pub fn read_model(file: &verified::File) -> hdf5::Result<HashMap::<String, kind_config::Value>>
{
    kind_config::io::read_from_hdf5(&File::open(file.as_str())?.group("model")?)
}

pub fn write_time_series<T: H5Type>(filename: &str, time_series: &Vec<T>) -> hdf5::Result<()>
{
    let file = File::create(filename)?;
    time_series.write(&file, "time_series")
}

pub fn read_time_series<T: H5Type>(file: verified::File) -> hdf5::Result<Vec<T>>
{
    let file = File::open(file.to_string())?;
    Vec::<T>::read(&file, "time_series")
}

fn write_build(group: &Group) -> hdf5::Result<()> {
    use hdf5::types::VarLenAscii;

    group.new_dataset::<VarLenAscii>()
        .create("version", ())?
        .write_scalar(&VarLenAscii::from_ascii(crate::VERSION_AND_BUILD).unwrap())?;

    Ok(())
}




// ============================================================================
pub fn write_checkpoint<C: Conserved>(
    filename: &str,
    state: &State<C>,
    block_data: &Vec<BlockData<C>>,
    model: &HashMap::<String, kind_config::Value>,
    tasks: &Tasks) -> hdf5::Result<()>
{
    let file = File::create(filename)?;

    write_state(&file, &state, block_data)?;
    write_tasks(&file, &tasks)?;
    write_model(&file, &model)?;
    write_build(&file)?;

    Ok(())
}
