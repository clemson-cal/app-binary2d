use ndarray::{Axis, Array, Ix1, Ix2};
use ndarray_ops::{adjacent_mean, cartesian_product2};




/**
 * Type alias to identify a block position in the mesh
 */
pub type BlockIndex = (usize, usize);




/**
 * The mesh. It is block-decomposed and presently at a uniform refinement level,
 * although this may change soon.
 */
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct Mesh {


    /// Number of mesh blocks per direction. For example if num_blocks=4 then
    /// the mesh contains a total of 16 blocks.
    pub num_blocks: usize,


    /// Number of grid cells, per block, per direction. For example if
    /// block_size=32 then there are 1024 zones in each block.
    pub block_size: usize,


    /// The distance from the origin (at the center of the domain) to the
    /// edge. The domain width and height are twice this value from
    /// edge-to-edge.
    pub domain_radius: f64,
}




// ============================================================================
impl Mesh {

    pub fn block_length(&self) -> f64 {
        2.0 * self.domain_radius / (self.num_blocks as f64)
    }

    pub fn block_start(&self, block_index: BlockIndex) -> (f64, f64) {
        (
            -self.domain_radius + (block_index.0 as f64) * self.block_length(),
            -self.domain_radius + (block_index.1 as f64) * self.block_length(),
        )
    }

    pub fn contains(&self, block_index: BlockIndex) -> bool {
        block_index.0 < self.num_blocks && block_index.1 < self.num_blocks
    }

    pub fn block_vertices(&self, block_index: BlockIndex) -> (Array<f64, Ix1>, Array<f64, Ix1>) {
        let start = self.block_start(block_index);
        let xv = Array::linspace(start.0, start.0 + self.block_length(), self.block_size + 1);
        let yv = Array::linspace(start.1, start.1 + self.block_length(), self.block_size + 1);
        (xv, yv)
    }

    pub fn cell_centers(&self, block_index: BlockIndex) -> Array<(f64, f64), Ix2> {
        let (xv, yv) = self.block_vertices(block_index);
        let xc = adjacent_mean(&xv, Axis(0));
        let yc = adjacent_mean(&yv, Axis(0));
        return cartesian_product2(xc, yc);
    }

    pub fn face_centers_x(&self, block_index: BlockIndex) -> Array<(f64, f64), Ix2> {
        let (xv, yv) = self.block_vertices(block_index);
        let yc = adjacent_mean(&yv, Axis(0));
        return cartesian_product2(xv, yc);
    }

    pub fn face_centers_y(&self, block_index: BlockIndex) -> Array<(f64, f64), Ix2> {
        let (xv, yv) = self.block_vertices(block_index);
        let xc = adjacent_mean(&xv, Axis(0));
        return cartesian_product2(xc, yv);
    }

    pub fn cell_spacing_x(&self) -> f64 {
        self.block_length() / (self.block_size as f64)
    }

    pub fn cell_spacing_y(&self) -> f64 {
        self.block_length() / (self.block_size as f64)
    }

    pub fn total_zones(&self) -> usize {
        self.num_blocks * self.num_blocks * self.block_size * self.block_size
    }

    pub fn block_indexes<'a>(&'a self) -> impl Iterator<Item = BlockIndex> + 'a {
        (0..self.num_blocks)
        .map(move |i| (0..self.num_blocks)
        .map(move |j| (i, j)))
        .flatten()
    }

    pub fn neighbor_block_indexes(&self, block_index: BlockIndex) -> [[BlockIndex; 3]; 3] {
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
