// ============================================================================
pub fn write_state(group: &hdf5::Group, state: &crate::State, block_data: &Vec<crate::BlockData>) -> Result<(), hdf5::Error>
{
    let cons = group.create_group("conserved")?;

    for (b, u) in block_data.iter().zip(&state.conserved)
    {
        let gname = format!("{:03}-{:03}", b.index.0, b.index.1);
        let udata = u.mapv(Into::<[f64; 3]>::into);
        cons.new_dataset::<[f64; 3]>().create(&gname, u.dim())?.write(&udata)?;
    }
    group.new_dataset::<i64>().create("iteration",  ())?.write_scalar(&state.iteration.to_integer())?;
    group.new_dataset::<f64>().create("time",       ())?.write_scalar(&state.time)?;

    Ok(())
}




// ============================================================================
pub fn write_checkpoint(filename: &str, state: &crate::State, block_data: &Vec<crate::BlockData>, run_config: &kind_config::Form) -> Result<(), hdf5::Error>
{
    use hdf5::File;
    use kind_config::io;

    let file = File::create(filename)?;
    let cfg_group = file.create_group("run_config")?;
    write_state(&file, &state, block_data)?;
    io::write_to_hdf5(&cfg_group, &run_config.value_map())?;

    Ok(())
}
