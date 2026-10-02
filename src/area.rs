use std::{
    fs,
    io::{self, BufWriter},
};

use box_gen::{self, cacl::point::PointTrait, support_type::Area};
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
#[gen_stub_pyclass]
#[pyclass(name = "Env")]
pub struct CraneEnv {
    pub(crate) area: box_gen::support_type::Area,
}
#[gen_stub_pyfunction]
#[pyfunction]
/// load env from obj file
pub fn load_env(
    path: &str,
    split: usize,
    offs_high: f64,
    offs_low: f64,
    offs: f64,
    wider: f64,
) -> PyResult<CraneEnv> {
    CraneEnv::new(path, split, offs_high, offs_low, offs, wider)
}
impl CraneEnv {
    pub fn new(
        path: &str,
        split: usize,
        offs_high: f64,
        offs_low: f64,
        offs: f64,
        wider: f64,
    ) -> PyResult<Self> {
        let a = Area::gen_from_obj_file(
            path,
            split,
            offs_high,
            offs_low,
            offs,
            wider,
            0.0,
            [112.01, 37.01, 752.0],
        );
        Ok(Self { area: a })
    }
}
#[gen_stub_pymethods]
#[pymethods]
impl CraneEnv {
    pub fn debug(&self) -> String {
        format!("{:#?}", self.area)
    }
    pub fn collide(&self, pos: [f64; 3]) -> bool {
        self.area.collide_point(pos)
    }
    pub fn collide_line(&self, a: [f64; 3], b: [f64; 3]) -> bool {
        self.area.collide_line(a, b)
    }
    pub fn direction_length(&self, pos: [f64; 3], direction: [f64; 3], max: f64) -> f64 {
        let p0 = pos;
        let p1 = pos.add(direction.scale(max / direction.length()));
        let delta = self.area.collide_delta(p0, p1).unwrap_or(1.);
        // (p0, p1).lerp(delta)
        delta * max
    }
    pub fn direction_length_multi(&self, args: Vec<([f64; 3], [f64; 3], f64)>) -> Vec<f64> {
        args.into_iter()
            .map(|(pos, direction, max)| self.direction_length(pos, direction, max))
            .collect::<Vec<f64>>()
    }
    pub fn direction_length_multi_dir(
        &self,
        pos: [f64; 3],
        directions: Vec<[f64; 3]>,
        max: f64,
    ) -> Vec<f64> {
        let device = pulp::Arch::new();
        // let device = pulp::Arch::Scalar;
        // println!("{}", directions.len());
        // return vec![];
        directions
            .into_par_iter()
            // .into_iter()
            .map(|direction| device.dispatch(|| self.direction_length(pos, direction, max)))
            .collect()
    }

    pub fn random_point(&self) -> [f64; 3] {
        self.area.random_point()
    }
    pub fn min(&self) -> [f64; 3] {
        self.area.min()
    }
    pub fn max(&self) -> [f64; 3] {
        self.area.max()
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl CraneEnv {
    pub fn save(&self, path: &str) -> Result<(), io::Error> {
        self.area.write_to_obj(&mut BufWriter::new(
            fs::OpenOptions::new()
                .create(true)
                .truncate(true)
                .write(true)
                .open(path)?,
        ))
    }
    pub fn block_width(&self) -> f64 {
        self.area.block_width()
    }
}
