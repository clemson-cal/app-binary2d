use std::collections::HashMap;
use hdf5::{File, H5Type};




// ============================================================================
fn write_state(group: &hdf5::Group, state: &crate::State, block_data: &Vec<crate::BlockData>) -> Result<(), hdf5::Error>
{
    let cons = group.create_group("conserved")?;

    for (b, u) in block_data.iter().zip(&state.conserved)
    {
        let gname = format!("0:{:03}-{:03}", b.index.0, b.index.1);
        let udata = u.mapv(Into::<[f64; 3]>::into);
        cons.new_dataset::<[f64; 3]>().create(&gname, u.dim())?.write(&udata)?;
    }
    group.new_dataset::<i64>().create("iteration",  ())?.write_scalar(&state.iteration.to_integer())?;
    group.new_dataset::<f64>().create("time",       ())?.write_scalar(&state.time)?;

    Ok(())
}

pub fn read_state(filename: &str) -> Result<crate::State, hdf5::Error>
{
    let file = File::open(filename)?;
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
fn write_group<E: H5Type, T: Into<Vec<(String, E)>>>(group: &hdf5::Group, tasks: T) -> Result<(), hdf5::Error>
{
    for (name, task) in Into::<Vec<(String, E)>>::into(tasks)
    {
        group.new_dataset::<E>().create(&name, ())?.write_scalar(&task)?;
    }
    Ok(())
}

fn read_group<E: H5Type, T: From<Vec<(String, E)>>>(group: &hdf5::Group) -> Result<T, hdf5::Error>
{
    let task_iter = group.member_names()?
        .into_iter()
        .map(|name| group.dataset(&name)?.read_scalar::<E>().map(|task| (name, task)))
        .filter_map(Result::ok);

    Ok(From::<Vec<(String, E)>>::from(task_iter.collect()))
}




// ============================================================================
fn write_tasks(group: &hdf5::Group, tasks: &crate::Tasks) -> Result<(), hdf5::Error>
{
    write_group(group, tasks.clone())
}

pub fn read_tasks(filename: &str) -> Result<crate::Tasks, hdf5::Error>
{
    let file = File::open(filename)?;
    let group = file.group("tasks")?;
    read_group(&group)
}




// ============================================================================
fn write_model(group: &hdf5::Group, model: &HashMap::<String, kind_config::Value>) -> Result<(), hdf5::Error>
{
    kind_config::io::write_to_hdf5(&group, &model)?;
    Ok(())
}

pub fn read_model(filename: &str) -> Result<HashMap::<String, kind_config::Value>, hdf5::Error>
{
    let file = File::open(filename)?;
    let group = file.group("model")?;
    kind_config::io::read_from_hdf5(&group)
}




// ============================================================================
pub fn write_time_series(_rundir: &str, _time_series: &Vec<crate::TimeSeriesSample>) -> Result<(), hdf5::Error>
{
    Ok(())
}

pub fn read_time_series(_rundir: &str) -> Result<Vec<crate::TimeSeriesSample>, hdf5::Error>
{
    Ok(Vec::new())
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
