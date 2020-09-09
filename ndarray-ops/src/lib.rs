use std::ops::{Add, Sub, Mul};
use ndarray::*;




// ============================================================================
pub fn cartesian_product2<T: Copy, U: Copy>(x: Array1<T>, y: Array1<U>) -> Array2<(T, U)>
{
    Array::from_shape_fn((x.len(), y.len()), |(i, j)| (x[i], y[j]))
}




// ============================================================================
pub fn cartesian_product3<T: Copy, U: Copy, V: Copy>(x: Array1<T>, y: Array1<U>, z: Array1<V>) -> Array3<(T, U, V)>
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
        D: Dimension,
{
    map_stencil2(x, axis, |&a, &b| (a + b) * 0.5)
}

pub fn adjacent_diff<T, U, P, D>(x: &ArrayBase<P, D>, axis: Axis) -> Array<U, D>
    where
        T: Copy + Sub<Output=U>,
        U: Copy,
        P: RawData<Elem=T> + Data,
        D: Dimension,
{
    map_stencil2(x, axis, |&a, &b| b - a)
}




// ============================================================================
pub fn extend_periodic<T, P>(a: ArrayBase<P, Ix2>, ng: usize) -> Array2<T>
    where
        T: Copy,
        P: RawData<Elem=T> + Data
{
    let ni = a.dim().0;
    let nj = a.dim().1;
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




// ============================================================================
pub fn extend_from_neighbor_arrays<T, P>(a: &[[ArrayBase<P, Ix2>; 3]; 3], gli: usize, gri: usize, glj: usize, grj: usize) -> Array2<T>
    where
        T: Copy,
        P: RawData<Elem=T> + Data,
{
    fn block_index(i: usize, gl: usize, n0: usize, gr: usize) -> usize
    {
        if i < gl {
            0
        } else if i < gl + n0 {
            1
        } else if i < gl + n0 + gr {
            2
        } else {
            panic!();
        }
    }
    let ni = a[1][1].dim().0;
    let nj = a[1][1].dim().1;
    let extended_shape = (ni + gli + gri, nj + glj + grj);

    Array::from_shape_fn(extended_shape, |(i, j)| -> T
    {
        let bi = block_index(i, gli, ni, gri);
        let bj = block_index(j, glj, nj, grj);
        let i0 = match bi {
            0 => i + a[0][bj].dim().0 - gli,
            1 => i - gli,
            2 => i - a[1][bj].dim().0 - gli,
            _ => panic!(),
        };
        let j0 = match bj {
            0 => j + a[bi][0].dim().1 - glj,
            1 => j - glj,
            2 => j - a[bi][1].dim().1 - glj,
            _ => panic!(),
        };
        unsafe {
            *a[bi][bj].uget([i0, j0])
        }
    })
}




// ============================================================================
#[cfg(test)]
mod tests
{
    use ndarray::Array2;

    #[test]
    fn extend_from_neighbor_arrays_works_with_uniformly_shaped_arrays()
    {
        let x = Array2::<f64>::zeros((10, 10)).to_shared();
        let a = [
            [x.clone(), x.clone(), x.clone()],
            [x.clone(), x.clone(), x.clone()],
            [x.clone(), x.clone(), x.clone()]
        ];
        assert_eq!(crate::extend_from_neighbor_arrays(&a, 2, 2, 2, 2).dim(), (14, 14));
        assert_eq!(crate::extend_from_neighbor_arrays(&a, 3, 3, 2, 2).dim(), (16, 14));
        assert_eq!(crate::extend_from_neighbor_arrays(&a, 2, 2, 3, 3).dim(), (14, 16));
        assert_eq!(crate::extend_from_neighbor_arrays(&a, 3, 1, 2, 2).dim(), (14, 14));
        assert_eq!(crate::extend_from_neighbor_arrays(&a, 2, 2, 3, 1).dim(), (14, 14));
    }

    #[test]
    fn extend_from_neighbor_arrays_works_with_non_uniformly_shaped_arrays()
    {
        let x = Array2::<f64>::zeros((10, 10)).to_shared();
        let y = Array2::<f64>::zeros(( 3,  3)).to_shared(); // these are in the corner

        let a = [
            [y.clone(), x.clone(), y.clone()],
            [x.clone(), x.clone(), x.clone()],
            [y.clone(), x.clone(), y.clone()],
        ];
        assert_eq!(crate::extend_from_neighbor_arrays(&a, 1, 1, 1, 1).dim(), (12, 12));
        assert_eq!(crate::extend_from_neighbor_arrays(&a, 1, 2, 1, 2).dim(), (13, 13));
    }

    #[test]
    #[should_panic]
    fn extend_from_neighbor_arrays_panics_with_wrongly_shaped_arrays()
    {
        let x = Array2::<f64>::zeros((10, 10)).to_shared();
        let y = Array2::<f64>::zeros(( 3,  3)).to_shared();

        let a = [
            [y.clone(), y.clone(), y.clone()],
            [x.clone(), x.clone(), x.clone()],
            [y.clone(), y.clone(), y.clone()],
        ];
        crate::extend_from_neighbor_arrays(&a, 1, 1, 1, 1);
    }
}
