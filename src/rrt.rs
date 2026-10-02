use box_gen::{
    cacl::point::{Point2Trait, PointTrait},
    path_planning::AsMove,
    support_type::Area,
    utils::{index_xyz, Counter},
};
use log::info;
use pyo3::{pyclass, pymethods};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use rand::random;
use rayon::{
    iter::{IntoParallelIterator, ParallelIterator},
    vec,
};
use stupid_utils::select::DotSelect;

use crate::{area::CraneEnv, crane::CraneM};

#[derive(Debug, Clone)]
struct SearchDone {
    path: Vec<([f64; 3], usize, f64)>,
    tree: Vec<Node>,
    counter: usize,
}
#[derive(Debug, Clone)]
struct Node {
    pos: [f64; 3],
    root: usize,
    from_move_idx: usize,
    #[allow(unused)]
    from_move_step: f64,
}

#[gen_stub_pyclass]
#[pyclass]
#[derive(Debug)]
pub struct RRTConfig {
    rrt步长: f64,
    rrt动作保持强度: f64,
    rrt终止点最大距离: f64,
    rrt动作切换惩罚系数: f64,
    rrt路径生成次数: usize,
    rrt最大尝试采样次数: usize,
    rrt初始动作: usize,
}
#[gen_stub_pymethods]
#[pymethods]
impl RRTConfig {
    #[new]
    pub fn new(
        rrt步长: f64,
        rrt动作保持强度: f64,
        rrt终止点最大距离: f64,
        rrt动作切换惩罚系数: f64,
        rrt路径生成次数: usize,
        rrt最大尝试采样次数: usize,
        rrt初始动作: usize,
    ) -> Self {
        Self {
            rrt步长,
            rrt动作保持强度,
            rrt终止点最大距离,
            rrt动作切换惩罚系数,
            rrt路径生成次数,
            rrt最大尝试采样次数,
            rrt初始动作,
        }
    }
    pub fn rrt_once(
        &self,
        crane: &CraneM,
        env: &CraneEnv,
        start: [f64; 3],
        end: [f64; 3],
    ) -> Option<Vec<([f64; 3], usize, f64)>> {
        let moveset = &crane.0;
        let area = &env.area;
        info!("{:?}", self);
        let search_result = {
            info!("start {:?}, end {:?}", start, end);

            self.rrt_move::<3, box_gen::path_planning::Crane>(
                moveset,
                &area,
                start,
                end,
                self.rrt最大尝试采样次数,
            )
        };

        let path = search_result.map(|e| e.path);
        path
    }

    pub fn rrt(
        &self,
        crane: &CraneM,
        env: &CraneEnv,
        start: [f64; 3],
        end: [f64; 3],
    ) -> Option<Vec<([f64; 3], usize, f64)>> {
        let 起始点 = start;
        let 目标点 = end;

        // let config = &(18.981, -1., [-15.6384, -10.2941, -11.8289]);

        let moveset = &crane.0;
        let area = &env.area;
        info!("{:?}", self);
        let (search_result, _all) = {
            let (start, end) = (起始点, 目标点);
            // let offsstart: [f64; 3] = random();
            // dbg!(offsstart);

            // let start = start.add(offsstart);

            info!("start {:?}, end {:?}", start, end);
            info!("start collide {}", area.collide_point(start));
            info!("end collide {}", area.collide_point(end));

            // let path = rrt_move::<3, XYZ>(&(), &area, start, end);
            // let mut best_path = path;
            // for _ in 0..10 {
            //     dbg!();
            //     let path = rrt_move::<3, XYZ>(&(), &area, start, end);
            //     if path.len() < best_path.len() {
            //         best_path = path;
            //     }
            // }
            let counter = Counter::new();
            let paths = (0..self.rrt路径生成次数)
                .into_par_iter()
                .map(|i| {
                    // info!("start {}", i);
                    let p = self.rrt_move::<3, box_gen::path_planning::Crane>(
                        moveset,
                        &area,
                        start,
                        end,
                        self.rrt最大尝试采样次数,
                    );
                    // info!("done {}", i);
                    counter.count();
                    if counter.show() % 10 == 0 {
                        // println!("done {}", counter.show());
                    }
                    p
                })
                .filter_map(|p| p)
                // .map(|p| p.path)
                .collect::<Vec<_>>();
            info!("{}", paths.len());
            // best_path
            // return vec![];

            (
                paths
                    .iter()
                    .map(|a| (a, self.eval2(&a.path, end)))
                    .min_by(|(_, a), (_, b)| a.total_cmp(b))
                    .map(|(a, _)| a.clone())?,
                paths,
            )
        };

        let path = search_result.path;
        Some(path)
    }
}

impl RRTConfig {
    fn rrt_move<const L: usize, M: AsMove<L> + Send + Sync>(
        &self,
        moveset: &M,
        area: &Area,
        start: [f64; 3],
        end: [f64; 3],
        max_count: usize,
    ) -> Option<SearchDone> {
        let moveset = moveset;

        let step_length = self.rrt步长;
        // info!("step length{}", step_length);
        assert!(step_length > 0.);
        // dbg!(step_length);
        let mut route = Vec::from([Node {
            pos: start,
            root: 0,
            from_move_idx: self.rrt初始动作,
            from_move_step: 0.0,
        }]);
        let mut counter = 0;
        loop {
            // info!("rrt sample count {}", counter);
            let nearest_idx = |p: [f64; 3]| {
                let mut nearest_idx = 0;
                let mut l2 = f64::MAX;
                for (idx, p1) in route.iter().map(|e| e.pos).enumerate() {
                    // let rand: f64 = random();
                    // let rand = rand * 0. + 1.;
                    let direction_fix = end.sub(p1).normal().dot(p.sub(p1).normal());
                    // let direction_fix = 0.;
                    // dbg!(direction_fix);
                    let l = p1.sub(p).length() * (2.03 - direction_fix);
                    assert!(l > 0.);
                    if l < l2 {
                        l2 = l;
                        nearest_idx = idx;
                    }
                }
                nearest_idx
            };

            let p = area.random_point();
            // if area.collide_point(p) {
            //     continue;
            // }
            let (base_point, base_idx, from_move_idx) = {
                let ni = nearest_idx(p);
                (route[ni].pos, ni, route[ni].from_move_idx)
            };

            let direction = p.sub(base_point).normal();
            let moves = moveset.normals(base_point);
            // 选择最接近的方向
            let nearest_idx = moves
                .into_iter()
                .enumerate()
                .fold((f64::MIN, 0), |(max_abs, max_idx), (idx, normal)| {
                    let abs = normal.dot(direction).abs()
                        + if idx == from_move_idx
                        // && !moveset.neared(base_point, end, from_move_idx, step_length / 2.)
                        {
                            // random::<f64>() * 0.9
                            // 1. * random::<f64>()
                            self.rrt动作保持强度 * random::<f64>()
                        } else {
                            // 0.
                            moveset
                                .neared(base_point, end, from_move_idx, step_length / 2.)
                                .select(self.rrt动作保持强度 * random::<f64>(), 0.)
                        };
                    if abs > max_abs {
                        (abs, idx)
                    } else {
                        (max_abs, max_idx)
                    }
                })
                .1;
            // dbg!(nearest_idx);
            let step = moves[nearest_idx].dot(direction).signum() * step_length;
            // dbg!(step);
            let next_p = moveset.apply(base_point, nearest_idx, step);
            // let next_p = base_point.add(direction.scale(step_length.min((base_point, p).length())));
            let next_step = step.signum() as isize * nearest_idx as isize;
            if !moveset.valid(area, next_p)
                || area.collide_point(next_p)
                || area.collide_line(base_point, next_p)
            {
                if !moveset.mercy(area, next_p, next_step) {
                    // info!(
                    //     "!moveset.valid(area, next_p) {}",
                    //     !moveset.valid(area, next_p)
                    // );
                    // info!("area.collide_point(next_p) {}", area.collide_point(next_p));
                    // info!(
                    //     "area.collide_line(base_point, next_p) {}",
                    //     area.collide_line(base_point, next_p)
                    // );
                    // counter += 1;
                    // if counter > max_count {
                    //     return None;
                    // }
                    continue;
                } else {
                    // continue;
                    // println!(
                    //     "valid: {},{},{}",
                    //     moveset.valid(area, next_p),
                    //     moveset.mercy(area, next_p, next_step),
                    //     next_step
                    // );
                }
                // let len = route.len();
                // dbg!P
                //     len,
                //     next_p,
                //     moveset.valid(area, next_p),
                //     area.collide_point(next_p),
                //     area.collide_line(base_point, next_p)
                // );
                // println!(
                //     "p {},{}",
                //     area.collide_point(next_p),
                //     area.collide_line(base_point, next_p),
                //     next_step
                // );
            }
            // route.push((base_idx, next_p));
            route.push(Node {
                pos: next_p,
                root: base_idx,
                from_move_idx: nearest_idx,
                from_move_step: step,
            });
            counter += 1;
            use index_xyz::*;
            if ([next_p[X], next_p[Y], 0.], [end[X], end[Y], 0.]).length2()
                <= self.rrt终止点最大距离 * self.rrt终止点最大距离
                && (next_p[Z] - end[Z]).abs() < self.rrt终止点最大距离 * 2.
                && (!area.collide_line(next_p, end)
                    || moveset.mercy(area, next_p, next_step)
                    || false)
            // && ((next_p[Z] - end[Z]).abs() < step_length * 5.)
            // && next_p[Z] >= end[Z]
            {
                break;
            }
            if counter > max_count {
                return None;
            }
        }
        // dbg!(counter);
        let mut path = Vec::from([(end, 0, 0.)]);
        let mut now_idx = route.len() - 1;
        loop {
            let Node {
                root: parent,
                pos: point,
                from_move_idx,
                from_move_step,
                ..
            } = route[now_idx];
            path.push((point, from_move_idx, from_move_step));
            if parent == now_idx {
                break;
            }
            now_idx = parent;
        }
        // path.push(start);
        path.reverse();
        Some(SearchDone {
            path,
            tree: route,
            counter,
        })
    }
    fn eval(&self, path: &[([f64; 3], usize, f64)]) -> usize {
        let pos = path.last().unwrap().0;
        let mut fix: isize = 0;
        let mut last_move = 2;
        for (_, m, _) in path.iter().copied() {
            if last_move != m {
                fix += self.rrt动作切换惩罚系数 as isize;
            }
            last_move = m;
            if m == 2 {
                fix -= 3;
            }
        }
        (path.len() as isize + fix).clamp(0, isize::MAX) as usize
    }
    fn eval2(&self, path: &[([f64; 3], usize, f64)], end: [f64; 3]) -> f64 {
        // let pos = path[path.len() - 2].0;
        // let l = pos.sub(end).length2() * 1000.;
        // dbg!(l);
        // return l as usize;
        let mut fix: f64 = 0.;
        let mut last_move1: usize = 2;
        let mut last_move: usize = 2;
        for (_, m, _) in path.iter().copied() {
            if last_move != m {
                fix += self.rrt动作切换惩罚系数;
            }
            last_move1 = last_move;
            last_move = m;
            if m == 2 {
                fix -= 0.1;
            }
        }
        // dbg!(last_move1);
        path.len() as f64 + fix + (last_move1 == 0).select(0., 0.)
    }
}
fn write_nodetree<W: std::io::Write>(file: &mut W, tree: &[Node]) -> std::io::Result<()> {
    use index_xyz::*;
    for node in tree {
        let point = node.pos;
        writeln!(file, "v {} {} {}", point[X], point[Y], point[Z])?;
    }
    for (idx, node) in tree.iter().enumerate() {
        writeln!(file, "l {} {}", idx + 1, node.root + 1)?;
    }

    Ok(())
}

fn eval_sw(path: &[([f64; 3], usize, f64)]) -> f64 {
    let mut count: f64 = 0.;
    let mut last_move: usize = path[0].1;
    for (_, m, _) in path.iter().copied() {
        if last_move != m {
            // dbg!(last_move, m);
            count += 1.;
        }
        last_move = m;
    }
    count.max(1.) - 1.
}
fn eval_length(path: &[([f64; 3], usize, f64)]) -> f64 {
    let mut count: f64 = 0.;
    let mut last_pos = path[0].0;
    for (p, _, _) in path.iter().copied() {
        count += (last_pos.sub(p)).length();
        last_pos = p
    }
    count
}
