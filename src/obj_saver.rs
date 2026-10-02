use std::{
    fs,
    io::{BufWriter, Write},
};

use pyo3::{pyclass, pymethods};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

#[gen_stub_pyclass]
#[pyclass]
pub struct ObjSaver {
    points: Vec<[f64; 3]>,
    as_line: bool,
}
#[gen_stub_pymethods]
#[pymethods]
impl ObjSaver {
    #[new]
    pub fn new() -> Self {
        Self {
            points: vec![],
            as_line: false,
        }
    }
    pub fn as_line(&mut self, as_line: bool) {
        self.as_line = as_line;
    }
    pub fn save(&self, path: &str) -> Result<(), std::io::Error> {
        let file = fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)
            .unwrap();
        let mut file = BufWriter::new(file);
        for &[x, y, z] in &self.points {
            writeln!(file, "v {} {} {}", x, y, z)?;
        }

        if self.as_line && self.points.len() > 1 {
            for i in 0..self.points.len() - 1 {
                writeln!(file, "l {} {}", i + 1, i + 2)?;
            }
        }

        Ok(())
    }
}
#[gen_stub_pymethods]
#[pymethods]
impl ObjSaver {
    pub fn point(&mut self, point: [f64; 3]) {
        self.points.push(point);
    }
}
