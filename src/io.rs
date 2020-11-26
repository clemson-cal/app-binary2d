use std::collections::HashMap;
use hdf5::{File, Group, H5Type};
use io_logical::verified;
use io_logical::nicer_hdf5;
use io_logical::nicer_hdf5::{H5Read, H5Write};




// ============================================================================
impl nicer_hdf5::H5Read for crate::Tasks
{
    fn read(group: &Group, name: &str) -> hdf5::Result<Self>
    {
        nicer_hdf5::read_as_keyed_vec(group, name)
    }    
}
impl nicer_hdf5::H5Write for crate::Tasks
{
    fn write(&self, group: &Group, name: &str) -> hdf5::Result<()>
    {
        nicer_hdf5::write_as_keyed_vec(self.clone(), group, name)
    }
}




// ============================================================================
fn write_state(group: &Group, state: &crate::State, block_data: &Vec<crate::BlockData>) -> hdf5::Result<()>
{
    let cons = group.create_group("conserved")?;

    for (b, u) in block_data.iter().zip(&state.conserved)
    {
        u.mapv(<[f64; 3]>::from).write(&cons, &format!("0:{:03}-{:03}", b.index.0, b.index.1))?
    }

    state.time.write(group, "time")?;
    state.iteration.write(group, "iteration")?;
    Ok(())
}

pub fn read_state(file: verified::File) -> hdf5::Result<crate::State>
{
    let file = File::open(file.to_string())?;
    let cons = file.group("conserved")?;
    let mut conserved = Vec::new();

    for key in cons.member_names()?
    {
        let u = ndarray::Array2::<[f64; 3]>::read(&cons, &key)?;
        conserved.push(u.mapv(hydro_iso2d::Conserved::from).to_shared());
    }

    let time      = f64::read(&file, "time")?;
    let iteration = num::rational::Ratio::<i64>::read(&file, "iteration")?;

    let result = crate::State{
        conserved: conserved,
        time: time,
        iteration: iteration,
    };
    Ok(result)
}

fn write_tasks(group: &Group, tasks: &crate::Tasks) -> hdf5::Result<()>
{
    tasks.write(group, "tasks")
}

pub fn read_tasks(file: verified::File) -> hdf5::Result<crate::Tasks>
{
    let file = File::open(file.to_string())?;
    crate::Tasks::read(&file, "tasks")
}

fn write_model(group: &Group, model: &HashMap::<String, kind_config::Value>) -> hdf5::Result<()>
{
    kind_config::io::write_to_hdf5(&group.create_group("model")?, &model)
}

pub fn read_model(file: verified::File) -> hdf5::Result<HashMap::<String, kind_config::Value>>
{
    kind_config::io::read_from_hdf5(&File::open(file.to_string())?.group("model")?)
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




// ============================================================================
pub fn write_checkpoint(
    filename: &str,
    state: &crate::State,
    block_data: &Vec<crate::BlockData>,
    model: &HashMap::<String, kind_config::Value>,
    tasks: &crate::Tasks) -> hdf5::Result<()>
{
    let file = File::create(filename)?;

    write_state(&file, &state, block_data)?;
    write_tasks(&file, &tasks)?;
    write_model(&file, &model)?;

    Ok(())
}
