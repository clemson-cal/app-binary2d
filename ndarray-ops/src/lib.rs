use std::ops::{Add, Sub, Mul};
use ndarray::*;




// ============================================================================
pub fn cartesian_product2<T: Default + Copy, U: Default + Copy>(x: Array1<T>, y: Array1<U>) -> Array2<(T, U)>
{
    Array::from_shape_fn((x.len(), y.len()), |(i, j)| (x[i], y[j]))
}




// ============================================================================
pub fn cartesian_product3<T: Default + Copy, U: Default + Copy, V: Default + Copy>(x: Array1<T>, y: Array1<U>, z: Array1<V>) -> Array3<(T, U, V)>
{
    Array::from_shape_fn((x.len(), y.len(), z.len()), |(i, j, k)| (x[i], y[j], z[k]))
}




// ============================================================================
pub fn map_stencil2<T, U, P, D, F>(x: &ArrayBase<P, D>, axis: Axis, f: F) -> Array<U, D>
    where
        T: Copy,
        U: Copy,
        P: RawData<Elem=T> + Data,
        D: Dimension,
        F: FnMut(&T, &T) -> U,
{
    let n = x.len_of(axis);
    let a = x.slice_axis(axis, Slice::from(0..n-1));
    let b = x.slice_axis(axis, Slice::from(1..n-0));
    azip![a, b].apply_collect(f)
}

pub fn map_stencil3<T, U, P, D, F>(x: &ArrayBase<P, D>, axis: Axis, f: F) -> Array<U, D>
    where
        T: Copy,
        U: Copy,
        P: RawData<Elem=T> + Data,
        D: Dimension,
        F: FnMut(&T, &T, &T) -> U,
{
    let n = x.len_of(axis);
    let a = x.slice_axis(axis, Slice::from(0..n-2));
    let b = x.slice_axis(axis, Slice::from(1..n-1));
    let c = x.slice_axis(axis, Slice::from(2..n-0));
    azip![a, b, c].apply_collect(f)
}

pub fn map_stencil4<T, U, P, D, F>(x: &ArrayBase<P, D>, axis: Axis, f: F) -> Array<U, D>
    where
        T: Copy,
        U: Copy,
        P: RawData<Elem=T> + Data,
        D: Dimension,
        F: FnMut(&T, &T, &T, &T) -> U,
{
    let n = x.len_of(axis);
    let a = x.slice_axis(axis, Slice::from(0..n-3));
    let b = x.slice_axis(axis, Slice::from(1..n-2));
    let c = x.slice_axis(axis, Slice::from(2..n-1));
    let d = x.slice_axis(axis, Slice::from(3..n-0));
    azip![a, b, c, d].apply_collect(f)
}




// ============================================================================
pub fn adjacent_mean<T, U, P, D>(x: &ArrayBase<P, D>, axis: Axis) -> Array<U, D>
    where
        T: Copy + Add<Output=U>,
        U: Copy + Mul<f64, Output=U>,
        P: RawData<Elem=T> + Data,
        D: Dimension
{
    map_stencil2(x, axis, |&a, &b| (a + b) * 0.5)
}

pub fn adjacent_diff<T, U, P, D>(x: &ArrayBase<P, D>, axis: Axis) -> Array<U, D>
    where
        T: Copy + Sub<Output=U>,
        U: Copy,
        P: RawData<Elem=T> + Data,
        D: Dimension
{
    map_stencil2(x, axis, |&a, &b| b - a)
}




// ============================================================================
pub fn extend_periodic<T: Copy>(a: Array2<T>, ng: usize) -> Array2<T>
{
    let ni = a.len_of(Axis(0));
    let nj = a.len_of(Axis(1));
    let extended_shape = (ni + 2 * ng, nj + 2 * ng);

    Array::from_shape_fn(extended_shape, |(mut i, mut j)| -> T
    {
        if i < ng {
            i += ni;
        } else if i >= ni + ng {
            i -= ni;
        }
        if j < ng {
            j += nj;
        } else if j >= nj + ng {
            j -= nj;
        }
        unsafe {
            *a.uget([i - ng, j - ng])
        }
    })
}
