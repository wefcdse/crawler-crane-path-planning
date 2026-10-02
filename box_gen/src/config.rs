use serde::{Deserialize, Serialize};
use std::io::Write;
use std::{fs, path::PathBuf};

use box_gen::utils::StringErr;

// 全局配置结构体：AppConfig
// 该结构由 config.json 反序列化生成，包含 RRT 参数、网格/文件路径、以及塔吊（Crane）的初始参数。
// 字段使用中文变量名以便于与原作者的配置项对应。

#[derive(Debug, Serialize, Deserialize)]
pub struct AppConfig {
    pub rrt步长对block倍数: f64,
    pub rrt最大尝试采样次数: usize,
    pub rrt路径生成次数: usize,
    pub rrt终止点最大距离对block倍数: f64,
    pub rrt动作切换惩罚系数: f64,
    pub rrt动作保持强度: f64,

    pub 文件名称: PathBuf,
    pub 网格分割数量: usize,
    pub 上方偏移: f64,
    pub 下方偏移: f64,
    pub 偏移: f64,
    pub 包围盒扩大距离: f64,

    pub 基准点经纬度高度: [f64; 3],
    pub 起始点经纬度高度: [f64; 3],
    pub 目标点经纬度高度: [f64; 3],
    pub 吊臂长度: f64,
    pub 起重机忽略半径: f64,
    pub 吊车回转中心水平经纬度和变幅中心高度: [f64; 3],
    pub 吊车最大变幅角度: f64,
    pub 变幅中心对回转中心偏移: f64,

    pub 初始回转: f64,
    pub 初始变幅: f64,
    pub 初始绳长: f64,

    pub 回转修正参数: f64,
    pub 初始动作: usize,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            rrt步长对block倍数: 5.0,
            rrt最大尝试采样次数: 5000,
            rrt路径生成次数: 1024,
            rrt终止点最大距离对block倍数: 2.0,
            rrt动作切换惩罚系数: 100.,
            rrt动作保持强度: 2.,

            文件名称: "".into(),
            网格分割数量: 300,
            上方偏移: 20.,
            下方偏移: 20.,
            偏移: 0.02,
            包围盒扩大距离: 1.,

            基准点经纬度高度: [7.107685, 0.592520, 0.928000],
            起始点经纬度高度: [7.011840, -2.013177, 0.397000],
            目标点经纬度高度: [7.107685, 0.592520, 0.928000],
            吊臂长度: 5.15744,
            起重机忽略半径: 1.5,
            吊车回转中心水平经纬度和变幅中心高度: [2.493617, -1.038677, 1.021000],
            吊车最大变幅角度: 75.,
            变幅中心对回转中心偏移: -0.252292,

            初始回转: 0.0,
            初始变幅: 0.0,
            初始绳长: 0.0,

            回转修正参数: 1.2,
            初始动作: 2,
        }
    }
}

// 懒加载的全局 CONFIG：程序启动时读取 config.json，若不存在或解析失败则写入默认配置并返回默认值
lazy_static::lazy_static!(
    pub static ref CONFIG:AppConfig = new_config();
);

fn new_config() -> AppConfig {
    let file = "config.json";
    let f: Result<AppConfig, String> = (|| {
        let f = fs::read_to_string(file).to_msg()?;
        dbg!(&f);
        let a = serde_json::from_str(&f).to_msg()?;
        Ok(a)
    })();
    match f {
        Ok(f) => f,
        Err(e) => {
            dbg!(e);
            let c = AppConfig::default();
            let s = serde_json::to_string_pretty(&c).unwrap();
            let mut f = fs::OpenOptions::new()
                .write(true)
                .truncate(true)
                .create(true)
                .open(file)
                .unwrap();
            write!(f, "{}", s).unwrap();
            // 解析失败时写回默认配置文件，并返回默认值供程序继续运行
            c
        }
    }
}
