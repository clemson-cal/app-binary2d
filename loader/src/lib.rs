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
use binary2d::physics;
use binary2d::state;
use binary2d::traits::{Conserved, Hydrodynamics};




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

    #[getter]
    fn time_series(&self, py: Python) -> PyResult<PyObject> {
        match &self.app.time_series {
            app::AnyTimeSeries::Euler     (time_series) => Ok(pythonize(py, time_series)?),
            app::AnyTimeSeries::Isothermal(time_series) => Ok(pythonize(py, time_series)?),
        }
    }
}




// ============================================================================
#[pymethods]
impl State {

    /// The simulation time
    #[getter]
    fn time(&self) -> f64 {
        match self.state.as_ref() {
            app::AnyState::Euler     (state) => state.time,
            app::AnyState::Isothermal(state) => state.time,
        }
    }

    /// The simulation time
    #[getter]
    fn iteration(&self) -> i64 {
        match self.state.as_ref() {
            app::AnyState::Euler     (state) => *state.iteration.numer(),
            app::AnyState::Isothermal(state) => *state.iteration.numer(),
        }
    }

    /// X component of gas velocity
    #[getter]
    fn velocity_x(&self, py: Python) -> HashMap<(usize, usize), PyObject> {
        match self.state.as_ref() {
            app::AnyState::Euler(state)      => self.map_conserved(state, |u| u.1 / u.0, py),
            app::AnyState::Isothermal(state) => self.map_conserved(state, |u| u.1 / u.0, py),
        }
    }

    /// Y component of gas velocity
    #[getter]
    fn velocity_y(&self, py: Python) -> HashMap<(usize, usize), PyObject> {
        match self.state.as_ref() {
            app::AnyState::Euler(state)      => self.map_conserved(state, |u| u.2 / u.0, py),
            app::AnyState::Isothermal(state) => self.map_conserved(state, |u| u.2 / u.0, py),
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
        self.map_conserved_euler_only(|hydro, u| hydro.to_primitive(u).gas_pressure(), "pressure", py)
    }

    /// Gas thermal energy per unit mass
    #[getter]
    fn specific_internal_energy(&self, py: Python) -> PyResult<HashMap<(usize, usize), PyObject>> {
        let f = |hydro: &physics::Euler, u| hydro.to_primitive(u).specific_internal_energy(hydro.gamma_law_index);
        self.map_conserved_euler_only(f, "specific internal energy", py)
    }

    /// Gas Mach number
    #[getter]
    fn mach_number(&self, py: Python) -> PyResult<HashMap<(usize, usize), PyObject>> {
        let f = |hydro: &physics::Euler, u| hydro.to_primitive(u).mach_number(hydro.gamma_law_index);
        self.map_conserved_euler_only(f, "Mach number", py)
    }
}

impl State {
    fn map_conserved<C, F>(&self, state: &state::State<C>, f: F, py: Python) -> HashMap<(usize, usize), PyObject>
    where
        C: Conserved,
        F: Fn(C) -> f64
    {
        state.solution.iter().map(|(&index, block)| {
            (index, block.conserved.mapv(&f).to_pyarray(py).to_object(py))
        }).collect()
    }

    fn map_conserved_euler_only<F>(&self, f: F, field_name: &str, py: Python) -> PyResult<HashMap<(usize, usize), PyObject>>
    where
        F: Fn(&physics::Euler, hydro_euler::euler_2d::Conserved) -> f64
    {
        match (&self.hydro, self.state.as_ref()) {
            (app::AnyHydro::Euler(hydro), app::AnyState::Euler(state)) => {
                Ok(self.map_conserved(state, |u| f(hydro, u), py))
            }
            (_, _) => {
                Err(PyErr::from_instance(PyValueError::new_err(format!("{} not defined for isothermal hydro", field_name)).instance(py)))
            }
        }
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
        let (x, y) = self.mesh.block_vertices(self.index);
        (x.to_pyarray(py).to_object(py), y.to_pyarray(py).to_object(py))
    }
}




// ============================================================================
#[pyfunction]
fn app(filename: &str) -> PyResult<App> {
    match app::App::from_file(filename, Vec::new()) {
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
