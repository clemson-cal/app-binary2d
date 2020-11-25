use std::collections::HashMap;
use hdf5::{File, Group, H5Type};
use io_logical::verified;




// ============================================================================
fn write_as_keyed_vec<E: H5Type, T: Into<Vec<(String, E)>>>(item: T, group: &Group, name: &str) -> Result<(), hdf5::Error>
{
    let task_vec: Vec<_> = item.into();
    let target_group = group.create_group(name)?;
    for (name, task) in task_vec {
        target_group.new_dataset::<E>().create(&name, ())?.write_scalar(&task)?;
    }
    Ok(())
}

fn read_as_keyed_vec<E: H5Type, T: From<Vec<(String, E)>>>(group: &Group, name: &str) -> Result<T, hdf5::Error>
{
    let task_vec: Vec<_> = group
        .group(name)?
        .member_names()?
        .into_iter()
        .map(|name| group.dataset(&name)?.read_scalar::<E>().map(|task| (name, task)))
        .filter_map(Result::ok)
        .collect();
    Ok(task_vec.into())
}




/**
 * This trait bridges the ndarray and hdf5 Dimension traits.
 */
trait Dimension: ndarray::Dimension
{
    type Shape: hdf5::Dimension;

    fn hdf5_shape<P: ndarray::RawData>(array: &ndarray::ArrayBase<P, Self>) -> Self::Shape;
}

impl Dimension for ndarray::Ix1
{
    type Shape = usize;
    fn hdf5_shape<P: ndarray::RawData>(array: &ndarray::ArrayBase<P, Self>) -> Self::Shape { array.dim() }
}

impl Dimension for ndarray::Ix2
{
    type Shape = (usize, usize);
    fn hdf5_shape<P: ndarray::RawData>(array: &ndarray::ArrayBase<P, Self>) -> Self::Shape { array.dim() }
}

impl Dimension for ndarray::Ix3
{
    type Shape = (usize, usize, usize);
    fn hdf5_shape<P: ndarray::RawData>(array: &ndarray::ArrayBase<P, Self>) -> Self::Shape { array.dim() }
}




// ============================================================================
trait H5Write
{
    fn write(&self, group: &Group, name: &str) -> Result<(), hdf5::Error>;
}

trait H5Read: Sized
{
    fn read(group: &Group, name: &str) -> Result<Self, hdf5::Error>;
}




// ============================================================================
impl H5Write for f64
{
    fn write(&self, group: &Group, name: &str) -> hdf5::Result<()>
    {
        group.new_dataset::<Self>().create(name, ())?.write_scalar(self)
    }
}
impl H5Write for i64
{
    fn write(&self, group: &Group, name: &str) -> hdf5::Result<()>
    {
        group.new_dataset::<Self>().create(name, ())?.write_scalar(self)
    }
}
impl<T> H5Write for num::rational::Ratio<T> where T: num::Integer + H5Type + Copy
{
    fn write(&self, group: &Group, name: &str) -> hdf5::Result<()>
    {
        group.new_dataset::<[T; 2]>().create(name, ())?.write_scalar(&[*self.numer(), *self.denom()])
    }
}
impl<P, T, I> H5Write for ndarray::ArrayBase<P, I>
    where
    P: ndarray::RawData<Elem=T> + ndarray::Data,
    T: H5Type,
    I: Dimension,
{
    fn write(&self, group: &Group, name: &str) -> hdf5::Result<()>
    {
        group.new_dataset::<T>().create(name, I::hdf5_shape(self))?.write(self)
    }
}
impl<T> H5Write for Vec<T>
    where
    T: H5Type,
{
    fn write(&self, group: &Group, name: &str) -> hdf5::Result<()>
    {
        group.new_dataset::<T>().create(name, self.len())?.write_raw(self)
    }
}




// ============================================================================
impl H5Read for f64
{
    fn read(group: &Group, name: &str) -> hdf5::Result<Self>
    {
        group.dataset(name)?.read_scalar::<Self>()
    }
}
impl H5Read for i64
{
    fn read(group: &Group, name: &str) -> hdf5::Result<Self>
    {
        group.dataset(name)?.read_scalar::<Self>()
    }
}
impl<T> H5Read for num::rational::Ratio<T> where T: num::Integer + H5Type + Copy
{
    fn read(group: &Group, name: &str) -> hdf5::Result<Self>
    {
        let nd = group.dataset(name)?.read_scalar::<[T; 2]>()?;
        Ok(Self::new(nd[0], nd[1]))
    }
}
impl<T, I> H5Read for ndarray::Array<T, I>
    where
    T: H5Type,
    I: Dimension,
{
    fn read(group: &Group, name: &str) -> hdf5::Result<Self>
    {
        Ok(group.dataset(name)?.read_dyn::<T>()?.into_dimensionality::<I>()?)
    }
}
impl<T, I> H5Read for ndarray::ArcArray<T, I>
    where
    T: H5Type + Clone,
    I: Dimension,
{
    fn read(group: &Group, name: &str) -> hdf5::Result<Self>
    {
        Ok(ndarray::Array::<T, I>::read(group, name)?.to_shared())
    }
}
impl<T> H5Read for Vec<T>
    where
    T: H5Type,
{
    fn read(group: &Group, name: &str) -> hdf5::Result<Self>
    {
        group.dataset(name)?.read_raw()
    }
}




// ============================================================================
impl H5Read for crate::Tasks
{
    fn read(group: &Group, name: &str) -> hdf5::Result<Self>
    {
        read_as_keyed_vec(group, name)
    }    
}
impl H5Write for crate::Tasks
{
    fn write(&self, group: &Group, name: &str) -> hdf5::Result<()>
    {
        write_as_keyed_vec(self.clone(), group, name)
    }
}




// ============================================================================
fn write_state(group: &Group, state: &crate::State, block_data: &Vec<crate::BlockData>) -> Result<(), hdf5::Error>
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

pub fn read_state(file: verified::File) -> Result<crate::State, hdf5::Error>
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




// ============================================================================
fn write_tasks(group: &Group, tasks: &crate::Tasks) -> Result<(), hdf5::Error>
{
    tasks.write(group, "tasks")
}

pub fn read_tasks(file: verified::File) -> Result<crate::Tasks, hdf5::Error>
{
    let file = File::open(file.to_string())?;
    crate::Tasks::read(&file, "tasks")
}




// ============================================================================
fn write_model(group: &Group, model: &HashMap::<String, kind_config::Value>) -> Result<(), hdf5::Error>
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
pub fn write_checkpoint(
    filename: &str,
    state: &crate::State,
    block_data: &Vec<crate::BlockData>,
    model: &HashMap::<String, kind_config::Value>,
    tasks: &crate::Tasks) -> Result<(), hdf5::Error>
{
    let file = File::create(filename)?;
    let tasks_group = file.create_group("tasks")?;
    let model_group = file.create_group("model")?;

    write_state(&file, &state, block_data)?;
    write_tasks(&tasks_group, &tasks)?;
    write_model(&model_group, &model)?;

    Ok(())
}
