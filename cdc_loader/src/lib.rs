use std::sync::Arc;
use std::collections::HashMap;
use pyo3::prelude::*;
use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::PyMappingProtocol;
use pyo3::wrap_pyfunction;
use numpy::ToPyArray;
use pythonize::pythonize;
use binary2d::app;
use binary2d::mesh;
use binary2d::state;
use binary2d::traits;




// ============================================================================
#[pyclass]
struct App {
    app: app::App
}

#[pyclass]
struct State {
    state: Arc<app::AnyState>,
    hydro: app::AnyHydro,
}

#[pyclass]
struct Mesh {
    mesh: mesh::Mesh,
}

#[pyclass]
struct MeshBlock {
    mesh: mesh::Mesh,
    index: (usize, usize),
}




// ============================================================================
#[pymethods]
impl App {

    /// The code version number
    #[getter]
    fn version(&self, py: Python) -> PyResult<PyObject> {
        Ok(pythonize(py, &self.app.version)?)
    }

    /// A dict of the runtime configuration. This dict will mirror the
    /// app::Configuration struct.
    #[getter]
    fn config(&self, py: Python) -> PyResult<PyObject> {
        Ok(pythonize(py, &self.app.config)?)
    }

    /// A dict of the task list
    #[getter]
    fn tasks(&self, py: Python) -> PyResult<PyObject> {
        Ok(pythonize(py, &self.app.tasks)?)
    }

    /// The simulation state
    #[getter]
    fn state(&self) -> State {
        State{state: Arc::new(self.app.state.clone()), hydro: self.app.config.hydro.clone()}
    }

    /// The simulation mesh
    #[getter]
    fn mesh(&self) -> Mesh {
        Mesh{mesh: self.app.config.mesh.clone()}
    }
}




// ============================================================================
#[pymethods]
impl State {

    /// The simulation time
    #[getter]
    fn time(&self) -> f64 {
        match self.state.as_ref() {
            app::AnyState::Euler(state) => state.time,
            app::AnyState::Isothermal(state) => state.time,
        }
    }

    /// The simulation time
    #[getter]
    fn iteration(&self) -> i64 {
        match self.state.as_ref() {
            app::AnyState::Euler(state) => *state.iteration.numer(),
            app::AnyState::Isothermal(state) => *state.iteration.numer(),
        }
    }

    /// X component of gas velocity
    #[getter]
    fn velocity_x(&self, py: Python) -> HashMap<(usize, usize), PyObject> {
        match self.state.as_ref() {
            app::AnyState::Euler(state)      => self.map_conserved(state, |u| u.1, py),
            app::AnyState::Isothermal(state) => self.map_conserved(state, |u| u.1, py),
        }
    }

    /// Y component of gas velocity
    #[getter]
    fn velocity_y(&self, py: Python) -> HashMap<(usize, usize), PyObject> {
        match self.state.as_ref() {
            app::AnyState::Euler(state)      => self.map_conserved(state, |u| u.2, py),
            app::AnyState::Isothermal(state) => self.map_conserved(state, |u| u.2, py),
        }
    }

    /// Surface density
    #[getter]
    fn sigma(&self, py: Python) -> HashMap<(usize, usize), PyObject> {
        match self.state.as_ref() {
            app::AnyState::Euler(state)      => self.map_conserved(state, |u| u.0, py),
            app::AnyState::Isothermal(state) => self.map_conserved(state, |u| u.0, py),
        }
    }

    /// Vertically integrated gas pressure
    #[getter]
    fn pressure(&self, py: Python) -> PyResult<HashMap<(usize, usize), PyObject>> {
        use traits::Hydrodynamics;

        match (&self.hydro, self.state.as_ref()) {
            (app::AnyHydro::Euler(hydro), app::AnyState::Euler(state)) => {
                Ok(self.map_conserved(state, |u| hydro.to_primitive(u).gas_pressure(), py))
            }
            (_, _) => {
                Err(PyErr::from_instance(PyValueError::new_err("not defined for isothermal hydro").instance(py)))
            }
        }
    }
}

impl State {
    fn map_conserved<C, F>(&self, state: &state::State<C>, f: F, py: Python) -> HashMap<(usize, usize), PyObject>
    where
        C: traits::Conserved,
        F: Fn(C) -> f64
    {
        state.solution.iter().map(|(&index, block)| {
            (index, block.conserved.mapv(&f).to_pyarray(py).to_object(py))
        }).collect()
    }
}




// ============================================================================
#[pymethods]
impl Mesh {

    #[getter]
    fn indexes(&self) -> Vec<(usize, usize)> {
        self.mesh.block_indexes().collect()
    }
}




// ============================================================================
#[pyproto]
impl PyMappingProtocol for Mesh {

    fn __len__(&self) -> usize {
        self.mesh.block_indexes().count()
    }

    fn __getitem__(&self, index: mesh::BlockIndex) -> PyResult<MeshBlock> {
        if self.mesh.contains(index) {
            Ok(MeshBlock{mesh: self.mesh.clone(), index})
        } else {
            pyo3::Python::with_gil(|py| {
                Err(PyErr::from_instance(PyKeyError::new_err("block index is out of bounds").instance(py)))
            })
        }
    }
}




// ============================================================================
#[pymethods]
impl MeshBlock {

    #[getter]
    fn vertices(&self, py: Python) -> (PyObject, PyObject) {
        let (xv, yv) = self.mesh.block_vertices(self.index);
        (xv.to_pyarray(py).to_object(py), yv.to_pyarray(py).to_object(py))
    }
}




// ============================================================================
#[pyfunction]
fn app(filename: &str) -> PyResult<App> {
    match app::App::from_file(filename) {
        Ok(app) => Ok(App{app}),
        Err(e)  => Err(PyValueError::new_err(format!("{}", e))),
    }
}




// ============================================================================
#[pymodule]
fn cdc_loader(_: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(app, m)?)?;
    Ok(())
}
