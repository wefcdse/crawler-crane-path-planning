// 本文件是基于 RRT (Rapidly-exploring Random Tree) 的路径搜索程序入口。
// 主要用于为塔吊（Crane）生成可行的运动轨迹，并将结果输出为 OBJ/CSV 供可视化或控制使用。
// 注释尽量准确说明每个函数和重要步骤的用途与输入/输出含义。
use core::f64;
use std::{f64::consts::PI, fs, io::BufWriter, isize};
#[path = "../config.rs"]
pub mod config;
use box_gen::{
    cacl::{
        point::{Point2Trait, PointTrait},
        到经纬度,
    },
    path_planning::{AsMove, Crane},
    support_type::Area,
    time,
    utils::{
        index_xyz::{self},
        write_line_to_obj, Counter,
    },
};
use config::*;
use rand::random;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::io::Write;
use stupid_utils::select::DotSelect;

fn main() {
    // 确保输出目录存在，并执行主配置流程
    // 输出文件会写入 ./temp 目录
    std::fs::create_dir_all("./temp").unwrap();
    time!(main);
    // 运行基于配置的搜索与输出（核心流程）
    run_config();
    time!(main, "main");
}

fn run_config() {
    // 读取全局 CONFIG，建立 Area（环境网格）与 Crane（运动学模型），
    // 并发多次运行 RRT 搜索以挑选最优解，最终将路径/树/动作写入文件。

    let rad_to_deg = 180. / PI;
    let base = CONFIG.基准点经纬度高度.to_xyz();
    let 吊车回转中心水平坐标和变幅中心垂直坐标 = CONFIG
        .吊车回转中心水平经纬度和变幅中心高度
        .to_xyz()
        .sub(base);
    let 起始点 = CONFIG.起始点经纬度高度.to_xyz().sub(base);
    let 目标点 = CONFIG.目标点经纬度高度.to_xyz().sub(base);

    let config = &(
        CONFIG.吊臂长度,
        CONFIG.变幅中心对回转中心偏移,
        吊车回转中心水平坐标和变幅中心垂直坐标,
        CONFIG.初始动作,
        CONFIG.吊车最大变幅角度,
    );
    // 根据配置构造 Crane（运动学），并打印起始点对应的姿态
    let moveset = Crane::new(config);
    let (_t0, t1, _l) = dbg!(moveset.position_to_pose(起始点));
    dbg!(t1 * rad_to_deg);
    let area = Area::gen_from_obj_file(
        &CONFIG.文件名称,
        CONFIG.网格分割数量,
        CONFIG.上方偏移,
        CONFIG.下方偏移,
        CONFIG.偏移,
        CONFIG.包围盒扩大距离,
        CONFIG.起重机忽略半径,
        吊车回转中心水平坐标和变幅中心垂直坐标,
    );

    // 将网格化的地形写出为 OBJ，以便三维查看地形/障碍
    area.write_to_obj(&mut BufWriter::new(
        fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open("temp/rrt_area.obj")
            .unwrap(),
    ))
    .unwrap();
    write_line_to_obj(
        &mut BufWriter::new(
            fs::OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open("temp/info_crane_start_end.obj")
                .unwrap(),
        ),
        [吊车回转中心水平坐标和变幅中心垂直坐标, 起始点, 目标点],
    )
    .unwrap();

    // 保存 Area 的元信息（最低点、基点、分辨率等）到 temp/info.txt
    area.write_info(
        &mut fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open("temp/info.txt")
            .unwrap(),
    )
    .unwrap();
    // 启动计时并并行执行多次 RRT（随机化）以统计成功率与挑选最优路径
    time!(search);
    let (search_result, all) = {
        let (start, end) = (起始点, 目标点);

        dbg!(start, end);

        // 并行生成多条路径，使用 Counter 记录完成进度
        let counter = Counter::new();
        let paths = (0..CONFIG.rrt路径生成次数)
            .into_par_iter()
            .map(|_i| {
                let p = rrt_move::<3, Crane>(config, &area, start, end, CONFIG.rrt最大尝试采样次数);
                counter.count();
                if counter.show() % 10 == 0 {
                    println!("done {}", counter.show());
                }
                p
            })
            .filter_map(|p| p)
            .collect::<Vec<_>>();
        (
            // 从所有成功的路径中按 eval2 评分选择最优路径；若无成功路径则构造一个直接(start->end)的占位解
            paths
                .iter()
                .map(|a| (a, eval(&a.path, end)))
                .min_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(a, _)| a.clone())
                .unwrap_or(SearchDone {
                    path: Vec::from([(start, 0, 0.), (end, 0, 0.)]),
                    tree: Vec::new(),
                    counter: 0,
                }),
            paths,
        )
    };
    // 搜索总耗时（秒）
    let 耗时 = dbg!(search.elapsed().as_secs_f32());

    let 搜索成功次数 = dbg!(all.len());
    let 总搜索次数 = CONFIG.rrt路径生成次数;
    let 耗时 = 耗时;
    let 最大搜索节点数 = CONFIG.rrt最大尝试采样次数;
    let 成功时平均搜索次数 =
        dbg!(all.iter().map(|a| a.counter as f64).sum::<f64>() / all.len() as f64);
    let 路径长度 = dbg!(eval_length(&search_result.path));
    let 动作切换次数 = dbg!(eval_sw(&search_result.path));
    let 成功时平均动作切换次数 =
        dbg!(all.iter().map(|a| eval_sw(&a.path)).sum::<f64>() / all.len() as f64);
    let 成功时平均路径长度 =
        dbg!(all.iter().map(|a| eval_length(&a.path)).sum::<f64>() / all.len() as f64);
    let 搜索次数 = dbg!(search_result.counter);

    println!(
        "{} {} {} {} {} {} {} {} {} {}",
        搜索成功次数,
        总搜索次数,
        耗时,
        最大搜索节点数,
        成功时平均搜索次数,
        路径长度,
        动作切换次数,
        成功时平均动作切换次数,
        成功时平均路径长度,
        搜索次数
    );
    // 将最优路径写为 OBJ（顶点 + 连线），以及将搜索树写出用于分析
    write_line_to_obj(
        &mut BufWriter::new(
            fs::OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open("temp/rrt_path.obj")
                .unwrap(),
        ),
        search_result.path.iter().map(|(a, _, _)| *a),
    )
    .unwrap();
    write_nodetree(
        &mut BufWriter::new(
            fs::OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open("temp/rrt_tree.obj")
                .unwrap(),
        ),
        &search_result.tree,
    )
    .unwrap();
    let path = search_result.path;
    // 将路径转换为离散动作序列并写入 CSV（回转、变幅、绳长）以便控制器使用
    {
        let moveset = Crane::new(config);
        let mut v = Vec::new();
        let mut last_pos = path[0].0;
        v.push(last_pos);
        let mut last_m = path[0].1;
        for (i, (pos, m, _)) in path.iter().copied().enumerate() {
            if m != last_m || i == path.len() - 1 {
                v.push(last_pos);
            }
            last_pos = pos;
            last_m = m;
        }

        let mut of = fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open("temp/rrt_move.csv")
            .unwrap();
        let rad_to_deg = 180. / PI;

        let (t01, t02, l) = moveset.position_to_pose(起始点);
        let 输出回转修正 = CONFIG.初始回转 - t01 * rad_to_deg;
        let 输出变幅修正 = CONFIG.初始变幅 - t02 * rad_to_deg;
        let 输出绳长修正 = CONFIG.初始绳长 - l;
        for p in v {
            let (t1, t2, l) = moveset.position_to_pose(p);
            writeln!(
                of,
                "{:.4},{:.4},{:.4}",
                t1 * rad_to_deg + 输出回转修正 + CONFIG.回转修正参数,
                t2 * rad_to_deg + 输出变幅修正,
                l + 输出绳长修正
            )
            .unwrap();
        }
    }

    // 将路径上每个点的姿态（弧度/绳长）全部写出，便于调试和可视化
    {
        let moveset = Crane::new(config);
        let mut of = fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open("temp/rrt_move_all.obj")
            .unwrap();

        for (p, _, _) in path.iter() {
            let (t1, t2, l) = moveset.position_to_pose(*p);
            writeln!(of, "{:.4} {:.4} {:.4}", t1, t2, l).unwrap();
        }
    }
}
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
fn rrt_move<const L: usize, M: AsMove<L>>(
    config: &M::Config,
    area: &Area,
    start: [f64; 3],
    end: [f64; 3],
    max_count: usize,
) -> Option<SearchDone> {
    let moveset = M::new(config);

    let step_length = area.block_width() * CONFIG.rrt步长对block倍数;
    assert!(step_length > 0.);
    let mut route = Vec::from([Node {
        pos: start,
        root: 0,
        from_move_idx: moveset.default_move(),
        from_move_step: 0.0,
    }]);
    let mut counter = 0;
    loop {
        let nearest_idx = |p: [f64; 3]| {
            let mut nearest_idx = 0;
            let mut l2 = f64::MAX;
            for (idx, p1) in route.iter().map(|e| e.pos).enumerate() {
                let direction_fix = end.sub(p1).normal().dot(p.sub(p1).normal());
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
                    + if idx == from_move_idx {
                        CONFIG.rrt动作保持强度 * random::<f64>()
                    } else {
                        moveset
                            .neared(base_point, end, from_move_idx, step_length / 2.)
                            .select(CONFIG.rrt动作保持强度 * random::<f64>(), 0.)
                    };
                if abs > max_abs {
                    (abs, idx)
                } else {
                    (max_abs, max_idx)
                }
            })
            .1;
        let step = moves[nearest_idx].dot(direction).signum() * step_length;
        let next_p = moveset.apply(base_point, nearest_idx, step);
        let next_step = step.signum() as isize * nearest_idx as isize;
        if !moveset.valid(area, next_p)
            || area.collide_point(next_p)
            || area.collide_line(base_point, next_p)
        {
            if !moveset.mercy(area, next_p, next_step) {
                continue;
            }
        }
        route.push(Node {
            pos: next_p,
            root: base_idx,
            from_move_idx: nearest_idx,
            from_move_step: step,
        });
        counter += 1;
        use index_xyz::*;
        if ([next_p[X], next_p[Y], 0.], [end[X], end[Y], 0.]).length2()
            <= area.block_width()
                * area.block_width()
                * CONFIG.rrt终止点最大距离对block倍数
                * CONFIG.rrt终止点最大距离对block倍数
            && (next_p[Z] - end[Z]).abs()
                < CONFIG.rrt终止点最大距离对block倍数 * area.block_width() * 2.
            && (!area.collide_line(next_p, end) || moveset.mercy(area, next_p, next_step) || false)
        {
            break;
        }
        if counter > max_count {
            return None;
        }
    }
    dbg!(counter);
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
    path.reverse();
    Some(SearchDone {
        path,
        tree: route,
        counter,
    })
}

fn eval(path: &[([f64; 3], usize, f64)], _end: [f64; 3]) -> f64 {
    let mut fix: f64 = 0.;
    let mut last_move1: usize = 2;
    let mut last_move: usize = 2;
    for (_, m, _) in path.iter().copied() {
        if last_move != m {
            fix += CONFIG.rrt动作切换惩罚系数;
        }
        last_move1 = last_move;
        last_move = m;
        if m == 2 {
            fix -= 0.1;
        }
    }
    path.len() as f64 + fix + (last_move1 == 0).select(0., 0.)
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
