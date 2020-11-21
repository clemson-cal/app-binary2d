use std::collections::HashMap;
use std::time::Instant;
use hdf5::File;




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
fn write_tasks(group: &hdf5::Group, tasks: &crate::Tasks) -> Result<(), hdf5::Error>
{
    group.new_dataset::<f64>  ().create("checkpoint_next_time", ())?.write_scalar(&tasks.checkpoint_next_time)?;
    group.new_dataset::<usize>().create("checkpoint_count",     ())?.write_scalar(&tasks.checkpoint_count)?;
    Ok(())
}

pub fn read_tasks(filename: &str) -> Result<crate::Tasks, hdf5::Error>
{
    let file = File::open(filename)?;
    let group = file.group("tasks")?;

    let result = crate::Tasks{
        checkpoint_next_time: group.dataset("checkpoint_next_time")?.read_scalar::<f64>()?,
        checkpoint_count:     group.dataset("checkpoint_count")    ?.read_scalar::<usize>()?,
        call_count_this_run: 0,
        tasks_last_performed: Instant::now(),
    };
    Ok(result)
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
