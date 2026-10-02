use std::convert::Infallible;

use box_gen::{disable, path_planning::Crane};
use log::info;
use pyo3::{pyclass, pymethods};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
#[gen_stub_pyclass]
#[pyclass(name = "Crane")]
pub struct CraneM(pub(crate) Crane);
#[gen_stub_pymethods]
#[pymethods]
impl CraneM {
    #[new]
    #[pyo3(signature = (position, yaw_offs, length, max_yaw = 80.0, min_string_length = 0.0))]
    pub fn new(
        position: [f64; 3],
        yaw_offs: f64,
        length: f64,
        max_yaw: f64,
        min_string_length: f64,
    ) -> Self {
        let mut s = Self(Crane::new(&(length, yaw_offs, position, 0, max_yaw)));
        s.0.set_min_string_length(min_string_length);
        s
    }

    pub fn position(&self) -> [f64; 3] {
        self.0.position()
    }
    pub fn yaw_offs(&self) -> f64 {
        self.0.yaw_offs
    }
    pub fn max_yaw(&self) -> f64 {
        self.0.max变幅
    }
    pub fn min_string_length(&self) -> f64 {
        self.0.min_string_length
    }
}
#[gen_stub_pymethods]
#[pymethods]
impl CraneM {
    pub fn position_to_pose(&self, position: [f64; 3]) -> (f64, f64, f64) {
        self.0.position_to_pose(position)
    }
    pub fn pose_to_position(&self, pose: (f64, f64, f64)) -> [f64; 3] {
        self.0.pose_to_position(pose)
    }
    pub fn step_length_at(&self, pos: [f64; 3]) -> (f64, f64, f64) {
        self.0.step_at(pos)
    }
}
use box_gen::path_planning::AsMove;

use crate::area::CraneEnv;
#[gen_stub_pymethods]
#[pymethods]
impl CraneM {
    pub fn normals(&self, pos: [f64; 3]) -> ([f64; 3], [f64; 3], [f64; 3]) {
        let [t0, t1, l] = self.0.normals(pos);
        (t0, t1, l)
    }
    pub fn normals_as_move(&self, pos: [f64; 3]) -> [[f64; 3]; 3] {
        AsMove::normals(&self.0, pos)
    }

    pub fn apply_step(
        &self,
        position: [f64; 3],
        step_idx: usize,
        step_length_and_direction: f64,
    ) -> [f64; 3] {
        self.0.apply(position, step_idx, step_length_and_direction)
    }

    pub fn crane_length(&self) -> f64 {
        self.0.crane_length()
    }
    pub fn yaw_length(&self, position: [f64; 3]) -> f64 {
        self.0.yaw_length(position)
    }
    pub fn valid(&self, env: &CraneEnv, pos: [f64; 3]) -> bool {
        self.0.valid(&env.area, pos)
    }
    pub fn neared(
        &self,
        pos: [f64; 3],
        target: [f64; 3],
        from_move: usize,
        threshold: f64,
    ) -> bool {
        self.0.neared(pos, target, from_move, threshold)
    }
    pub fn mercy(&self, env: &CraneEnv, pos: [f64; 3], next_step: usize) -> bool {
        self.0.mercy(&env.area, pos, next_step as isize)
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl CraneM {
    pub fn max_step(&self, env: &CraneEnv, pos: [f64; 3], step_idx: usize, direction: f64) -> f64 {
        assert!(direction.is_normal());

        let mut step_length = env.area.block_width() / 1.2 * direction.signum();
        let target_step_length = step_length / 10.;
        assert!(step_length.is_normal());
        info!("max step start");
        info!("step_length: {}", step_length);
        let mut accumulate = 0.0;
        let mut now_pos = pos;
        // let (t10, t20, l0) = self.0.position_to_pose(pos);
        disable!(pos);
        let mut counter = 0;
        loop {
            let next_pos = self.apply_step(now_pos, step_idx, step_length);
            if env.area.collide_line(now_pos, next_pos) || !self.0.valid(&env.area, now_pos) {
                info!(
                    "break conditioo: {}, {}",
                    env.area.collide_line(now_pos, next_pos),
                    !self.0.valid(&env.area, now_pos)
                );
                let _unr: Infallible = if step_length > target_step_length {
                    break;
                } else {
                    step_length /= 2.0;
                    continue;
                };
            }
            accumulate += step_length.abs();
            if accumulate > self.0.crane_length() * 2.0 {
                break;
            }
            now_pos = next_pos;
            // info!("accumulate: {}", accumulate);
            counter += 1;
        }
        info!("counter: {}", counter);
        // let (t1, t2, l) = self.0.position_to_pose(now_pos);
        // let pose_diff = [t1 - t10, t2 - t20, l - l0];
        // pose_diff[step_idx]
        accumulate
    }
}
