// use crate::SolutionState;


// // ============================================================================
// fn to_tuple(inp: &[usize]) -> (usize, usize)
// {
//     match inp
//     {
//         [a, b] => (*a, *b),
//         _ => panic!(),
//     }
// }


// // ============================================================================
// pub fn write_hdf5(state: &SolutionState, filename: &str) -> Result<(), hdf5::Error>
// {
//     use hdf5::types::VarLenAscii;
//     use hdf5::File;

//     let file = File::create(filename)?;
//     let cons = state.conserved.mapv(Into::<[f64; 3]>::into);

//     file.new_dataset::<[f64; 3]>()   .create("conserved",  to_tuple(cons.shape())) ?.write(&cons)?;
//     file.new_dataset::<i64>()        .create("iteration",  ())                     ?.write_scalar(&state.iteration.to_integer())?;
//     file.new_dataset::<f64>()        .create("time",       ())                     ?.write_scalar(&state.time)?;
//     file.new_dataset::<VarLenAscii>().create("app",        ())                     ?.write_scalar(&VarLenAscii::from_ascii("app-binary2d").unwrap())?;

//     Ok(())
// }
