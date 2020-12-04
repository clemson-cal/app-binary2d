use hdf5::{Group, H5Type};




/**
 * This trait bridges the ndarray and hdf5 Dimension traits.
 */
pub trait Dimension: ndarray::Dimension
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
pub trait H5Write
{
    fn write(&self, group: &Group, name: &str) -> Result<(), hdf5::Error>;
}

pub trait H5Read: Sized
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




/**
 * Write a type that is Into a Vec of (String, T) pairs where T: H5Type. This
 * function can be used to easily implement H5Write for a type from your crate
 * which can be converted into a Vec of tuples.
 */
pub fn write_as_keyed_vec<E: H5Type, T: Into<Vec<(String, E)>>>(item: T, group: &Group, name: &str) -> Result<(), hdf5::Error>
{
    let item_vec: Vec<_> = item.into();
    let target_group = group.create_group(name)?;
    for (key, item) in item_vec {
        target_group.new_dataset::<E>().create(&key, ())?.write_scalar(&item)?;
    }
    Ok(())
}




/**
 * Read a type that is From a Vec of (String, T) pairs where T: H5Type. This
 * function can be used to easily implement H5Read for a type from your crate
 * which can be converted into a Vec of tuples.
 */
pub fn read_as_keyed_vec<E: H5Type, T: From<Vec<(String, E)>>>(group: &Group, name: &str) -> Result<T, hdf5::Error>
{
    let item_group = group.group(name)?;
    let item_vec: Vec<_> = item_group
        .member_names()?
        .into_iter()
        .map(|key| item_group.dataset(&key)?.read_scalar::<E>().map(|item| (key, item)))
        .filter_map(Result::ok)
        .collect();
    Ok(item_vec.into())
}
