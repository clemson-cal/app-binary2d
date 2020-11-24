use std::collections::HashMap;
use hdf5::{File, H5Type};
use io_logical::verified;




// ============================================================================
fn write_group<E: H5Type, T: Into<Vec<(String, E)>>>(group: &hdf5::Group, tasks: T) -> Result<(), hdf5::Error>
{
    let task_vec: Vec<_> = tasks.into();
    for (name, task) in task_vec {
        group.new_dataset::<E>().create(&name, ())?.write_scalar(&task)?;
    }
    Ok(())
}

fn read_group<E: H5Type, T: From<Vec<(String, E)>>>(group: &hdf5::Group) -> Result<T, hdf5::Error>
{
    let task_vec: Vec<_> = group.member_names()?
        .into_iter()
        .map(|name| group.dataset(&name)?.read_scalar::<E>().map(|task| (name, task)))
        .filter_map(Result::ok)
        .collect();
    Ok(task_vec.into())
}




// ============================================================================
fn write_state(group: &hdf5::Group, state: &crate::State, block_data: &Vec<crate::BlockData>) -> Result<(), hdf5::Error>
{
    let cons = group.create_group("conserved")?;

    for (b, u) in block_data.iter().zip(&state.conserved)
    {
        let dname = format!("0:{:03}-{:03}", b.index.0, b.index.1);
        let udata = u.mapv(Into::<[f64; 3]>::into);
        cons.new_dataset::<[f64; 3]>().create(&dname, u.dim())?.write(&udata)?;
    }

    group.new_dataset::<i64>().create("iteration", ())?.write_scalar(&state.iteration.to_integer())?;
    group.new_dataset::<f64>().create("time",      ())?.write_scalar(&state.time)?;
    Ok(())
}

pub fn read_state(file: verified::File) -> Result<crate::State, hdf5::Error>
{
    let file = File::open(file.to_string())?;
    let cons = file.group("conserved")?;
    let mut conserved = Vec::new();

    for key in cons.member_names()?
    {
        let u = cons.dataset(&key)?
            .read_dyn::<[f64; 3]>()?
            .into_dimensionality::<ndarray::Ix2>()?
            .mapv(Into::<hydro_iso2d::Conserved>::into)
            .to_shared();
        conserved.push(u);
    }

    let time      = file.dataset("time")     ?.read_scalar::<f64>()?;
    let iteration = file.dataset("iteration")?.read_scalar::<usize>()?;

    let result = crate::State{
        conserved: conserved,
        time: time,
        iteration: (iteration as i64).into(),
    };
    Ok(result)
}




// ============================================================================
fn write_tasks(group: &hdf5::Group, tasks: &crate::Tasks) -> Result<(), hdf5::Error>
{
    write_group(group, tasks.clone())
}

pub fn read_tasks(file: verified::File) -> Result<crate::Tasks, hdf5::Error>
{
    read_group(&File::open(file.to_string())?.group("tasks")?)
}




// ============================================================================
fn write_model(group: &hdf5::Group, model: &HashMap::<String, kind_config::Value>) -> Result<(), hdf5::Error>
{
    kind_config::io::write_to_hdf5(&group, &model)
}

pub fn read_model(file: verified::File) -> Result<HashMap::<String, kind_config::Value>, hdf5::Error>
{
    kind_config::io::read_from_hdf5(&File::open(file.to_string())?.group("model")?)
}




// ============================================================================
pub fn write_time_series<T: H5Type>(filename: &str, time_series: &Vec<T>) -> Result<(), hdf5::Error>
{
    File::create(filename)?.new_dataset::<T>().create("time_series", time_series.len())?.write_raw(time_series)
}

pub fn read_time_series<T: H5Type>(file: verified::File) -> Result<Vec<T>, hdf5::Error>
{
    File::open(file.to_string())?.dataset("time_series")?.read_raw::<T>()
}




// ============================================================================
pub fn write_checkpoint(filename: &str, state: &crate::State, block_data: &Vec<crate::BlockData>, model: &HashMap::<String, kind_config::Value>, tasks: &crate::Tasks) -> Result<(), hdf5::Error>
{
    let file = File::create(filename)?;
    let tasks_group = file.create_group("tasks")?;
    let model_group = file.create_group("model")?;

    write_state(&file, &state, block_data)?;
    write_tasks(&tasks_group, &tasks)?;
    write_model(&model_group, &model)?;

    Ok(())
}
