use area::CraneEnv;
use crane::CraneM;
use obj_saver::ObjSaver;
use pyo3::prelude::*;
use pyo3_stub_gen::{
    define_stub_info_gatherer,
    derive::{gen_stub_pyclass, gen_stub_pyfunction},
    type_info::PyFunctionInfo,
};
use rrt::RRTConfig;

define_stub_info_gatherer!(stub_info);
mod area;
mod crane;
mod obj_saver;
mod rrt;
#[gen_stub_pyfunction]
#[pyfunction]
fn print_on_drop_new(msg: String) -> PyResult<PrintOnDrop> {
    // PyCl
    // Ok(())
    // Ok(Py::new(py, todo!())?)
    Ok(PrintOnDrop(msg))
}

#[pyfunction]
const fn version() -> &'static str {
    const _: () = {
        pyo3_stub_gen::inventory::submit! {
            PyFunctionInfo{ name: "version", args:&[], r#return:<&str as pyo3_stub_gen::PyStubType>::type_output, doc:concat!("version = ", include_str!("version")), module: None }
        };
    };
    include_str!("version")
}

/// A Python module implemented in Rust.
#[pymodule]
fn crane_env_pyo3(m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyo3_log::init();
    m.add_function(wrap_pyfunction!(print_on_drop_new, m)?)?;
    m.add_function(wrap_pyfunction!(area::load_env, m)?)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(dim_to_sample, m)?)?;
    m.add_class::<CraneM>()?;
    m.add_class::<CraneEnv>()?;
    m.add_class::<ObjSaver>()?;
    m.add_class::<RRTConfig>()?;
    Ok(())
}

#[gen_stub_pyclass]
#[pyclass]
struct PrintOnDrop(String);
impl Drop for PrintOnDrop {
    fn drop(&mut self) {
        println!("{}", self.0);
    }
}

#[pymethods]
impl PrintOnDrop {
    pub fn msg(&self) -> &str {
        &self.0
    }
}

#[gen_stub_pyfunction]
#[pyfunction]
fn dim_to_sample(x: [f64; 3], y: [f64; 3], z: [f64; 3], count: usize) -> Vec<[f64; 3]> {
    let sum_l = 8 + 12 * count + 6 * count * count;
    let x = nalgebra::Vector3::from(x);
    let y = nalgebra::Vector3::from(y);
    let z = nalgebra::Vector3::from(z);
    let mut out: Vec<nalgebra::Vector3<f64>> = Vec::new();
    out.reserve(sum_l);

    {
        // 八个角
        out.push(x + y + z);
        out.push(x + y - z);
        out.push(x - y + z);
        out.push(x - y - z);
        out.push(-x + y + z);
        out.push(-x + y - z);
        out.push(-x - y + z);
        out.push(-x - y - z);
    }
    {
        //   def appendline(neg0: bool, neg1: bool, neg2: bool):
        let mut appendline = |neg0: bool, neg1: bool, neg2: bool| {
            for i in 0..count {
                let fix0 = if neg0 { -1.0 } else { 1.0 };
                let fix1 = if neg1 { -1.0 } else { 1.0 };
                let fix2 = if neg2 { -1.0 } else { 1.0 };
                let start_point = (x * fix0 + y * fix1 + z * fix2) * (count + 1) as f64 / 2.;
                out.push(start_point - x * (i + 1) as f64 * fix0);
                out.push(start_point - y * (i + 1) as f64 * fix1);
                out.push(start_point - z * (i + 1) as f64 * fix2);
            }
        };
        appendline(false, false, false);
        appendline(true, true, false);
        appendline(true, false, true);
        appendline(false, true, true);
    }
    {
        let mut appendface = |base, sur1, sur2| {
            for x in 0..count {
                for y in 0..count {
                    out.push(
                        base * (count + 1) as f64 / 2.
                            - sur1 * (count + 1) as f64 / 2.
                            - sur2 * (count + 1) as f64 / 2.
                            + sur1 * (x + 1) as f64
                            + sur2 * (y + 1) as f64,
                    )
                }
            }
        };
        let [n0, n1, n2] = [x, y, z];
        appendface(n0, n1, n2);
        appendface(-n0, n1, n2);
        appendface(n1, n0, n2);
        appendface(-n1, n0, n2);
        appendface(n2, n0, n1);
        appendface(-n2, n0, n1);
    }

    out.into_iter()
        .map(|v| {
            let v = v.normalize();
            v.into()
        })
        .collect()
}
