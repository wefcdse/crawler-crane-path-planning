// path_planning.rs: 定义运动学抽象与塔吊（Crane）实现
// - `AsMove` 为通用的“可应用动作”接口，支持返回法向量集、按索引应用动作、合法性判断等
// - `Crane` 给出塔吊的运动学实现（回转/变幅/绳长），并实现 AsMove<3>
use std::{f64::consts::PI, ops::Sub};

use crate::{cacl::point::PointTrait, disable, support_type::Area};

/// 通用移动模型接口：
/// L 表示每个位置可能的动作数（法向量个数），实现者需提供动作法向量、apply、合法性检查等。
pub trait AsMove<const L: usize> {
    type Config;
    fn new(config: &Self::Config) -> Self;
    fn normals(&self, pos: [f64; 3]) -> [[f64; 3]; L];
    fn apply(&self, pos: [f64; 3], sel_idx: usize, step: f64) -> [f64; 3];
    fn default_move(&self) -> usize;
    fn valid(&self, area: &Area, pos: [f64; 3]) -> bool;
    fn mercy(&self, area: &Area, pos: [f64; 3], next_step: isize) -> bool;
    fn neared(&self, pos: [f64; 3], end: [f64; 3], from_move: usize, th: f64) -> bool {
        let _ = (pos, end, from_move, th);
        unimplemented!()
    }
}

pub struct Crane {
    pub l: f64,
    pub yaw_offs: f64,
    pub base: [f64; 3],
    pub default_move: usize,
    pub max变幅: f64,
    pub min_string_length: f64,
}

impl Crane {
    pub fn new(config: &(f64, f64, [f64; 3], usize, f64)) -> Self {
        Self {
            l: config.0,
            yaw_offs: config.1,
            base: config.2,
            default_move: config.3,
            max变幅: config.4,
            min_string_length: 0.0,
        }
    }
    // 返回用于限制绳长的最小值（可由外部设置）
    pub fn set_min_string_length(&mut self, l: f64) {
        self.min_string_length = l;
    }
    // 返回吊臂长度
    pub fn crane_length(&self) -> f64 {
        self.l
    }
    // 计算水平平面上从基准点到给定位置的投影长度（用于回转角速或步长估算）
    pub fn yaw_length(&self, pos: [f64; 3]) -> f64 {
        // const PI: f64 = std::f64::consts::PI;
        let [x, y, _] = pos.sub(self.base);
        return (x * x + y * y).sqrt();
    }
    // 返回塔吊基准位置
    pub fn position(&self) -> [f64; 3] {
        self.base
    }
    // 将目标位置 pos 转换为姿态 (theta1: 回转角, theta2: 变幅角, l: 绳长差)
    pub fn position_to_pose(&self, pos: [f64; 3]) -> (f64, f64, f64) {
        // y
        //
        // 0    x
        let [x, y, z] = pos.sub(self.base);
        disable!(pos);
        let theta1 = { y.atan2(x) };

        let xy = (x * x + y * y).sqrt().sub(self.yaw_offs);

        let theta2 = (xy / self.l).acos();

        let l = (self.l * self.l - xy * xy).sqrt() - z;
        (theta1, theta2, l)
    }
    pub fn pose_to_position(&self, (theta1, theta2, l): (f64, f64, f64)) -> [f64; 3] {
        let z = self.l * theta2.sin() - l;
        let xy = self.l * theta2.cos() + self.yaw_offs;
        let x = xy * theta1.cos();
        let y = xy * theta1.sin();
        [x, y, z].add(self.base)
    }
}
#[test]
fn a() {
    use crate::utils::index_xyz::*;
    let d = Crane {
        l: 40.,
        yaw_offs: 60.,
        base: [0., 0., 100.],

        default_move: 0,
        max变幅: 80.,
        min_string_length: 0.,
    };
    let p = [70., 70., -12.];
    assert!([p[X] - d.base[X], p[Y] - d.base[Y], 0.].length() < (d.l + d.yaw_offs));
    let mut a = dbg!(d.position_to_pose(p));
    let b = dbg!(d.pose_to_position(a));
    assert!(p.loose_eq(b));

    let mut point_vec = Vec::new();
    point_vec.push(d.base);
    point_vec.push(d.base.add([0., 0., d.l]));
    point_vec.push(p);
    for _ in 0..100 {
        a.0 += std::f64::consts::PI / 180. * 3.;
        point_vec.push(d.pose_to_position(a));
    }
    for _ in 0..80 {
        a.1 += std::f64::consts::PI / 180. * 3.;
        point_vec.push(d.pose_to_position(a));
    }
    for _ in 0..100 {
        a.0 += std::f64::consts::PI / 180. * 1.;
        point_vec.push(d.pose_to_position(a));
    }
    for _ in 0..30 {
        a.2 += 1.;
        point_vec.push(d.pose_to_position(a));
    }

    for _ in 0..100 {
        a.1 += std::f64::consts::PI / 180. * 3.;
        point_vec.push(d.pose_to_position(a));
    }
    use std::fs;
    use std::io::BufWriter;
    crate::utils::write_line_to_obj(
        &mut BufWriter::new(
            fs::OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open("temp/cyc.obj")
                .unwrap(),
        ),
        point_vec.iter().copied(),
    )
    .unwrap();
}
impl Crane {
    pub fn step_at(&self, pos: [f64; 3]) -> (f64, f64, f64) {
        let [x, y, _] = pos.sub(self.base);
        let step_rad1 = 1. / (x * x + y * y).sqrt();
        let step_rad2 = 1. / self.l;
        (step_rad1, step_rad2, 1.)
    }
}
impl AsMove<3> for Crane {
    type Config = (f64, f64, [f64; 3], usize, f64);

    fn new(config: &Self::Config) -> Self {
        Self {
            l: config.0,
            yaw_offs: config.1,
            base: config.2,
            default_move: config.3,
            max变幅: config.4,
            min_string_length: 0.,
        }
    }

    fn normals(&self, pos: [f64; 3]) -> [[f64; 3]; 3] {
        let (t1, t2, _) = self.position_to_pose(pos);
        let t1pos_normal = [-t1.sin(), t1.cos(), 0.];
        let t2pos_normal = [-t2.sin() * t1.cos(), -t2.sin() * t1.sin(), t2.cos()];

        [t1pos_normal, t2pos_normal, [0., 0., -1.]]
    }

    fn apply(&self, pos: [f64; 3], sel_idx: usize, step: f64) -> [f64; 3] {
        // const PI: f64 = std::f64::consts::PI;
        let [x, y, _] = pos.sub(self.base);
        let step_rad1 = step / (x * x + y * y).sqrt();
        let step_rad2 = step / self.l;
        let (t1, t2, l) = self.position_to_pose(pos);
        let pose1 = match sel_idx {
            0 => (t1 + step_rad1, t2, l),
            1 => (t1, t2 + step_rad2, l),
            2 => (t1, t2, l + step),
            _ => unreachable!(),
        };
        self.pose_to_position(pose1)
    }

    fn default_move(&self) -> usize {
        self.default_move
    }

    fn valid(&self, area: &Area, pos: [f64; 3]) -> bool {
        let (t1, t2, l) = self.position_to_pose(pos);
        let start_p = self.pose_to_position((t1, PI / 2., self.l));

        let end_p = self.pose_to_position((t1, t2, 0.));

        log::debug!(
            "conditions: {}, {}, {}, {}",
            l > self.min_string_length,
            t2 > 0.,
            (t2 < (PI / 180. * self.max变幅)),
            (!area.collide_line(start_p, end_p))
        );
        l > self.min_string_length
            && (t2 > 0. && t2 < (PI / 180. * self.max变幅)) & (!area.collide_line(start_p, end_p))
    }
    fn mercy(&self, _area: &Area, pos: [f64; 3], next_step: isize) -> bool {
        let (t1, t2, l) = self.position_to_pose(pos);
        let _start_p = self.pose_to_position((t1, PI / 2., self.l));

        let _end_p = self.pose_to_position((t1, t2, 0.));

        l > 0. && (t2 > 0. && t2 < (PI / 180. * self.max变幅)) && (next_step.abs() == 2)
    }
    fn neared(&self, pos: [f64; 3], end: [f64; 3], from_move: usize, th: f64) -> bool {
        let [x, y, _] = pos.sub(self.base);
        let th_rad1 = th / (x * x + y * y).sqrt();
        let th_rad2 = th / self.l;
        let (t1, t2, l) = self.position_to_pose(pos);
        let (t1_end, t2_end, l_end) = self.position_to_pose(end);
        match from_move {
            0 => (t1 - t1_end).abs() < th_rad1,
            1 => (t2 - t2_end).abs() < th_rad2,
            2 => (l - l_end).abs() < th,
            _ => unreachable!(),
        }
    }
}
