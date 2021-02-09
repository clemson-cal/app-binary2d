use ndarray::{Axis, Array, Ix1, Ix2};




// ============================================================================
pub type BlockIndex = (usize, usize);




// ============================================================================
#[derive(Clone)]
pub struct Mesh
{
    pub num_blocks: usize,
    pub block_size: usize,
    pub domain_radius: f64,
}

impl Mesh
{
    pub fn block_length(&self) -> f64
    {
        2.0 * self.domain_radius / (self.num_blocks as f64)
    }

    pub fn block_start(&self, block_index: BlockIndex) -> (f64, f64)
    {
        (
            -self.domain_radius + (block_index.0 as f64) * self.block_length(),
            -self.domain_radius + (block_index.1 as f64) * self.block_length(),
        )
    }

    pub fn block_vertices(&self, block_index: BlockIndex) -> (Array<f64, Ix1>, Array<f64, Ix1>)
    {
        let start = self.block_start(block_index);
        let xv = Array::linspace(start.0, start.0 + self.block_length(), self.block_size + 1);
        let yv = Array::linspace(start.1, start.1 + self.block_length(), self.block_size + 1);
        (xv, yv)
    }

    pub fn cell_centers(&self, block_index: BlockIndex) -> Array<(f64, f64), Ix2>
    {
        use ndarray_ops::{adjacent_mean, cartesian_product2};
        let (xv, yv) = self.block_vertices(block_index);
        let xc = adjacent_mean(&xv, Axis(0));
        let yc = adjacent_mean(&yv, Axis(0));
        return cartesian_product2(xc, yc);
    }

    pub fn face_centers_x(&self, block_index: BlockIndex) -> Array<(f64, f64), Ix2>
    {
        use ndarray_ops::{adjacent_mean, cartesian_product2};
        let (xv, yv) = self.block_vertices(block_index);
        let yc = adjacent_mean(&yv, Axis(0));
        return cartesian_product2(xv, yc);
    }

    pub fn face_centers_y(&self, block_index: BlockIndex) -> Array<(f64, f64), Ix2>
    {
        use ndarray_ops::{adjacent_mean, cartesian_product2};
        let (xv, yv) = self.block_vertices(block_index);
        let xc = adjacent_mean(&xv, Axis(0));
        return cartesian_product2(xc, yv);
    }

    pub fn cell_spacing_x(&self) -> f64
    {
        self.block_length() / (self.block_size as f64)
    }

    pub fn cell_spacing_y(&self) -> f64
    {
        self.block_length() / (self.block_size as f64)
    }

    pub fn total_zones(&self) -> usize
    {
        self.num_blocks * self.num_blocks * self.block_size * self.block_size
    }

    pub fn block_indexes(&self) -> Vec<BlockIndex>
    {
        (0..self.num_blocks)
        .map(|i| (0..self.num_blocks)
        .map(move |j| (i, j)))
        .flatten()
        .collect()
    }

    pub fn neighbor_block_indexes(&self, block_index: BlockIndex) -> [[BlockIndex; 3]; 3]
    {
        let b = self.num_blocks;
        let m = |i, j| ((i + b) % b, (j + b) % b);
        let (i, j) = block_index;
        [
            [m(i - 1, j - 1), m(i - 1, j + 0), m(i - 1, j + 1)],
            [m(i + 0, j - 1), m(i + 0, j + 0), m(i + 0, j + 1)],
            [m(i + 1, j - 1), m(i + 1, j + 0), m(i + 1, j + 1)],
        ]
    }
}
