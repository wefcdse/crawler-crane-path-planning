// cacl.rs: 几何与坐标转换辅助函数集合
// 提供：三角形/矩形相交检测、点内判定、基于 OBJ 面片生成地形网格、高度插值、经纬度到平面坐标等
// 注：注释尽量精确，避免改变现有行为。
use impl_here::impl_here;
use nalgebra::Vector2;
use obj::Position;
use rayon::{iter::ParallelIterator, slice::ParallelSlice};

use crate::{support_type::Area, time};

// 计算两个位置的最小值
pub fn min_p(a: Position, b: Position) -> Position {
    Position {
        position: [
            a.position[0].min(b.position[0]), // 3D空间中X轴方向上的最小值
            a.position[1].min(b.position[1]), // 3D空间中Y轴方向上的最小值
            a.position[2].min(b.position[2]), // 3D空间中Z轴方向上的最小值
        ],
    }
}

// 计算两个位置的最大值
pub fn max_p(a: Position, b: Position) -> Position {
    Position {
        position: [
            a.position[0].max(b.position[0]), // 3D空间中X轴方向上的最大值
            a.position[1].max(b.position[1]), // 3D空间中Y轴方向上的最大值
            a.position[2].max(b.position[2]), // 3D空间中Z轴方向上的最大值
        ],
    }
}

// 判断由三角形 (p1,p2,p3) 与矩形 (d1,d2,d3,d4) 在平面投影上是否可能相交。
// 返回 true 表示存在重叠（注意：函数实现使用投影分离轴检测，返回值含义为“未被明显分离”）。

pub fn crossed(
    p1: [impl Into<f64> + Copy; 2],
    p2: [impl Into<f64> + Copy; 2],
    p3: [impl Into<f64> + Copy; 2],
    d1: [impl Into<f64> + Copy; 2],
    d2: [impl Into<f64> + Copy; 2],
    d3: [impl Into<f64> + Copy; 2],
    d4: [impl Into<f64> + Copy; 2],
) -> bool {
    let p1: [f64; 2] = [p1[0].into(), p1[1].into()];
    let p2: [f64; 2] = [p2[0].into(), p2[1].into()];
    let p3: [f64; 2] = [p3[0].into(), p3[1].into()];
    let d1: [f64; 2] = [d1[0].into(), d1[1].into()];
    let d2: [f64; 2] = [d2[0].into(), d2[1].into()];
    let d3: [f64; 2] = [d3[0].into(), d3[1].into()];
    let d4: [f64; 2] = [d4[0].into(), d4[1].into()];
    let p1 = nalgebra::Vector2::from(p1);
    let p2 = nalgebra::Vector2::from(p2);
    let p3 = nalgebra::Vector2::from(p3);
    let d1 = nalgebra::Vector2::from(d1);
    let d2 = nalgebra::Vector2::from(d2);
    let d3 = nalgebra::Vector2::from(d3);
    let d4 = nalgebra::Vector2::from(d4);

    let vec_tri1 = p2 - p1;
    let vec_tri2 = p3 - p1;
    let vec_tri3 = p3 - p2;

    let vec_rect1 = d2 - d1;
    let vec_rect2 = d3 - d2;
    let vec_rect3 = d4 - d3;
    let vec_rect4 = d1 - d4;

    let check_with_line = |v: Vector2<f64>| {
        let v = Vector2::from([v[1], v[0]]);
        let mut min_tri = v.dot(&p1);
        let mut max_tri = v.dot(&p1);

        min_tri = min_tri.min(v.dot(&p2));
        max_tri = max_tri.max(v.dot(&p2));

        min_tri = min_tri.min(v.dot(&p3));
        max_tri = max_tri.max(v.dot(&p3));

        let mut min_rect = v.dot(&d1);
        min_rect = min_rect.min(v.dot(&d2));
        min_rect = min_rect.min(v.dot(&d3));
        min_rect = min_rect.min(v.dot(&d4));

        let mut max_rect = v.dot(&d1);
        max_rect = max_rect.max(v.dot(&d2));
        max_rect = max_rect.max(v.dot(&d3));
        max_rect = max_rect.max(v.dot(&d4));

        if (max_tri < min_rect) || (max_rect < min_tri) {
            true
        } else {
            false
        }
    };

    !(check_with_line(vec_tri1)
        || check_with_line(vec_tri2)
        || check_with_line(vec_tri3)
        || check_with_line(vec_rect1)
        || check_with_line(vec_rect2)
        || check_with_line(vec_rect3)
        || check_with_line(vec_rect4))
}
pub fn inside(
    p1: [impl Into<f64> + Copy; 2],
    p2: [impl Into<f64> + Copy; 2],
    p3: [impl Into<f64> + Copy; 2],
    target: [impl Into<f64> + Copy; 2],
) -> bool {
    let p1: [f64; 2] = [p1[0].into(), p1[1].into()];
    let p2 = [p2[0].into(), p2[1].into()];
    let p3 = [p3[0].into(), p3[1].into()];
    let target: [f64; 2] = [target[0].into(), target[1].into()];
    let p1 = nalgebra::Vector2::from(p1);
    let p2 = nalgebra::Vector2::from(p2);
    let p3 = nalgebra::Vector2::from(p3);
    let target = nalgebra::Vector2::from(target);
    let vec1 = p2 - p1;
    let vec2 = p3 - p1;
    let target = target - p1;

    let mat = nalgebra::Matrix2::from_columns(&[vec1, vec2]);

    let Some(mati) = mat.try_inverse() else {
        return true;
    };

    let t1 = mati * target;

    t1[0] >= 0. && t1[1] >= 0. && t1[0] + t1[1] <= 1.
}
#[test]
fn t() {
    assert_eq!(false, inside([1, 1], [5, 2], [4, 6], [3, 0]));
    assert_eq!(true, inside([1, 1], [5, 2], [4, 6], [4, 3]));
    assert_eq!(true, inside([1, 1], [5, 2], [4, 6], [4, 4]));
    assert_eq!(false, inside([1, 1], [5, 2], [4, 6], [2, 3]));
    assert_eq!(true, inside([1, 1], [5, 2], [4, 6], [2, 2]));
    assert_eq!(true, inside([1, 1], [5, 2], [4, 6], [4, 2]));
}
#[impl_here(PositionXY)]
impl Position {
    fn xy(&self) -> [f64; 2] {
        let [x, y, ..] = self.position;
        [x as f64, y as f64]
    }
}
// 生成区域的函数
pub fn gen_area(
    area: &mut Area,
    faces: &[(usize, usize, usize)],
    vertices: &[Position],
    offs_high: f64,
    offs: f64,
    max: &Position,
    包围盒扩大距离: f64,
    起重机忽略半径: f64,
    吊车回转中心水平坐标和变幅中心垂直坐标: [f64; 3],
) {
    time!(gen_area);

    let wider = 0;
    let data = faces
        .par_chunks((faces.len() / rayon::current_num_threads()).max(rayon::current_num_threads()))
        .map(|f| {
            let mut data = area.data.clone();
            for (a, b, c) in f.iter().copied() {
                let min = min_p(min_p(vertices[a], vertices[b]), vertices[c]);
                let max = max_p(max_p(vertices[a], vertices[b]), vertices[c]);
                let min_xy = [min.position[0] as f64, min.position[1] as f64];
                let max_xy = [max.position[0] as f64, max.position[1] as f64];

                let min_idx = area.in_block(min_xy);
                let max_idx = area.in_block(max_xy);
                assert!(min_idx[0] <= max_idx[0]);
                assert!(min_idx[1] <= max_idx[1]);

                for x in (min_idx[0].max(wider) - wider)
                    ..=(max_idx[0].min(area.data.x() - 1 - wider) + wider)
                {
                    for y in (min_idx[1].max(wider) - wider)
                        ..=(max_idx[1].min(area.data.y() - 1 - wider) + wider)
                    {
                        {
                            let [px, py] = area.pos([x, y]);
                            if !crossed(
                                vertices[a].xy(),
                                vertices[b].xy(),
                                vertices[c].xy(),
                                [px, py],
                                [px + area.block_width(), py],
                                [px + area.block_width(), py + area.block_width()],
                                [px, py + area.block_width()],
                            ) {
                                continue;
                            }
                        }

                        let value = vertices[a].position[2] as f64 + offs;
                        let value = value.max(vertices[b].position[2] as f64 + offs);
                        let value = value.max(vertices[c].position[2] as f64 + offs);
                        data[[x, y]] = data[[x, y]].max(value);
                    }
                }
            }
            data
        })
        .reduce(
            || area.data.clone(),
            |mut a, b| {
                for idx in a.iter_index() {
                    a[idx] = a[idx].max(b[idx]);
                }
                a
            },
        );
    area.data = data;

    dbg!(area.max(), max.position[2] as f64 + offs_high);

    let max_z_flag = max.position[2] as f64 + 48.; // magic number

    // 处理网格的空洞
    area.data.iter_mut().for_each(|(_, height)| {
        if *height == f64::NEG_INFINITY {
            *height = max_z_flag;
        } else {
            *height += offs;
        }
    });

    {
        // 处理起重机忽略
        // 先获取最低点z值
        let min_z = area
            .data
            .iter()
            .map(|(_, z)| *z)
            .fold(f64::INFINITY, |a, b| a.min(b));
        // 将起重机忽略半径内的点高度设置为 min_z - 起重机忽略半径
        let center_xy = [
            吊车回转中心水平坐标和变幅中心垂直坐标[0],
            吊车回转中心水平坐标和变幅中心垂直坐标[1],
        ];
        let r = 起重机忽略半径;
        let area_tmp = area.clone();
        area.data.iter_mut().for_each(|((x, y), height)| {
            let pos = area_tmp.pos([x, y]);
            let dx = pos[0] - center_xy[0];
            let dy = pos[1] - center_xy[1];
            if dx * dx + dy * dy <= r * r {
                *height = min_z - 5.0; // magic number
            }
        });
    }
    {
        let min_z = area
            .data
            .iter()
            .map(|(_, z)| *z)
            .fold(f64::INFINITY, |a, b| a.min(b));
        area.data.iter_mut().for_each(|(_, height)| {
            if *height == max_z_flag {
                *height = min_z;
            }
        });
    }
    // 处理wider
    let old_data = area.data.clone();
    let new_data = area.data.clone();
    let wider = (包围盒扩大距离 / area.block_width()).ceil() as isize;
    dbg!(wider);
    let indexs = old_data.iter_index().collect::<Vec<_>>();
    if true {
        let r = indexs
            .par_chunks(indexs.len() / rayon::current_num_threads())
            .map(|c| {
                let mut d = new_data.clone();
                for &(x, y) in c {
                    let xmin = (x as isize - wider).max(0) as usize;
                    let xmax = (x as isize + wider).min(area.data.x() as isize - 1) as usize;
                    let ymin = (y as isize - wider).max(0) as usize;
                    let ymax = (y as isize + wider).min(area.data.y() as isize - 1) as usize;
                    let height = old_data[(x, y)];
                    for x1 in xmin..=xmax {
                        for y1 in ymin..=ymax {
                            if (x1 as isize - x as isize) * (x1 as isize - x as isize)
                                + (y1 as isize - y as isize) * (y1 as isize - y as isize)
                                > (wider * wider)
                            {
                                continue;
                            }
                            d[(x1, y1)] = d[(x1, y1)].max(height);
                        }
                    }
                }
                d
            })
            .reduce(
                || new_data.clone(),
                |mut a, b| {
                    for idx in a.iter_index() {
                        a[idx] = a[idx].max(b[idx]);
                    }
                    a
                },
            );
        area.data = r;
    }
    if false {
        for (x, y) in indexs {
            let xmin = (x as isize - wider).max(0) as usize;
            let xmax = (x as isize + wider).min(area.data.x() as isize - 1) as usize;
            let ymin = (y as isize - wider).max(0) as usize;
            let ymax = (y as isize + wider).min(area.data.y() as isize - 1) as usize;
            let height = old_data[(x, y)];
            for x1 in xmin..=xmax {
                for y1 in ymin..=ymax {
                    if (x1 as isize - x as isize) * (x1 as isize - x as isize)
                        + (y1 as isize - y as isize) * (y1 as isize - y as isize)
                        > (wider * wider)
                    {
                        continue;
                    }
                    area.data[(x1, y1)] = area.data[(x1, y1)].max(height);
                }
            }
        }
    }
    time!(gen_area, "gen area");
}

// 插值工具：对标量/向量/数组提供线性插值接口
pub mod lerp {
    pub trait Lerp: Copy {
        type Inner;
        type Float;
        fn lerp(self, delta: Self::Float) -> Self::Inner;
    }
    impl<T: ExtractInner2<f64>> Lerp for T {
        type Inner = f64;

        type Float = f64;

        fn lerp(self, delta: Self::Float) -> Self::Inner {
            let (a, b) = self.extract();
            a * (1. - delta) + b * delta
        }
    }

    impl Lerp for ([f64; 2], [f64; 2]) {
        type Inner = [f64; 2];

        type Float = f64;

        fn lerp(self, delta: Self::Float) -> Self::Inner {
            let ([x1, y1], [x2, y2]) = self;
            [(x1, x2).lerp(delta), (y1, y2).lerp(delta)]
        }
    }
    impl Lerp for ([f64; 3], [f64; 3]) {
        type Inner = [f64; 3];

        type Float = f64;

        fn lerp(self, delta: Self::Float) -> Self::Inner {
            let ([x1, y1, z1], [x2, y2, z2]) = self;
            [
                (x1, x2).lerp(delta),
                (y1, y2).lerp(delta),
                (z1, z2).lerp(delta),
            ]
        }
    }
    trait ExtractInner2<Inner>: Copy {
        fn extract(self) -> (Inner, Inner);
    }

    macro_rules! impl_into {
        ($a:tt) => {
            impl ExtractInner2<f64> for ($a, $a) {
                fn extract(self) -> (f64, f64) {
                    let (a, b) = self;
                    (a.into(), b.into())
                }
            }
            impl ExtractInner2<f64> for [$a; 2] {
                fn extract(self) -> (f64, f64) {
                    let [a, b] = self;
                    (a.into(), b.into())
                }
            }
        };
    }
    impl_into!(i8);
    impl_into!(i16);
    impl_into!(i32);

    impl_into!(u8);
    impl_into!(u16);
    impl_into!(u32);

    impl_into!(f32);
    impl_into!(f64);

    #[test]
    fn l() {
        assert_eq!((1i16, 0).lerp(0.5), 0.5);
        assert!(((1.3, -2.5).lerp(0.45) - -0.41) < 0.00000001);
    }
}
// 取模工具：确保对浮点取模返回正值（用于网格/周期计算）
pub mod rempos {
    pub fn rempos(a: f64, b: f64) -> f64 {
        assert!(b > 0.);
        if a.is_sign_positive() {
            a % b
        } else {
            a % b + b
        }
    }
    #[test]
    fn fp() {
        let eq = |a, b| {
            assert!(rempos(a, b) > 0.);
            assert_eq!(rempos(a, b), a % b);
        };
        eq(1.412, 351.1);
        eq(1231.412, 351.1);
        let eqneg = |a, b| {
            assert!(rempos(a, b) > 0.);
            assert_eq!(rempos(a, b), a % b + b);
        };
        eqneg(-1.412, 351.1);
        eqneg(-1231.412, 351.1);
        eqneg(-1231.41432, 11.184838);
    }
}

// 点/向量工具：为 [f64;3] 提供常见向量运算接口和实现
pub mod point {

    use crate::utils::index_xyz::{X, Y, Z};

    pub trait PointTrait {
        fn length(&self) -> f64 {
            self.length2().sqrt()
        }
        fn length2(&self) -> f64;
        fn sub(&self, rhs: Self) -> Self;
        fn add(&self, rhs: Self) -> Self;
        fn dot(&self, rhs: Self) -> f64;
        fn scale(&self, rhs: f64) -> Self;
        fn normal(&self) -> Self;
        fn loose_eq(&self, rhs: Self) -> bool;
        fn cross(&self, rhs: Self) -> Self;
    }
    impl PointTrait for [f64; 3] {
        fn length2(&self) -> f64 {
            self[X] * self[X] + self[Y] * self[Y] + self[Z] * self[Z]
        }

        fn sub(&self, rhs: Self) -> Self {
            [(self[X] - rhs[X]), (self[Y] - rhs[Y]), (self[Z] - rhs[Z])]
        }

        fn add(&self, rhs: Self) -> Self {
            [(self[X] + rhs[X]), (self[Y] + rhs[Y]), (self[Z] + rhs[Z])]
        }

        fn dot(&self, rhs: Self) -> f64 {
            self[X] * rhs[X] + self[Y] * rhs[Y] + self[Z] * rhs[Z]
        }

        fn normal(&self) -> Self {
            [
                self[X] / self.length(),
                self[Y] / self.length(),
                self[Z] / self.length(),
            ]
        }

        fn scale(&self, rhs: f64) -> Self {
            [self[X] * rhs, self[Y] * rhs, self[Z] * rhs]
        }

        fn loose_eq(&self, rhs: Self) -> bool {
            let b = 1e-10;
            (self[X] - rhs[X]).abs() <= b
                && (self[Y] - rhs[Y]).abs() <= b
                && (self[Z] - rhs[Z]).abs() <= b
        }

        fn cross(&self, rhs: Self) -> Self {
            [
                self[1] * rhs[2] - self[2] * rhs[1],
                self[2] * rhs[0] - self[0] * rhs[2],
                self[0] * rhs[1] - self[1] * rhs[0],
            ]
        }
    }
    pub trait Point2Trait {
        fn length(&self) -> f64 {
            self.length2().sqrt()
        }
        fn length2(&self) -> f64;
    }
    impl Point2Trait for ([f64; 3], [f64; 3]) {
        fn length2(&self) -> f64 {
            self.0.sub(self.1).length2()
        }
    }
    impl Point2Trait for [[f64; 3]; 2] {
        fn length2(&self) -> f64 {
            self[0].sub(self[1]).length2()
        }
    }
    #[test]
    fn t() {
        use rand::random;
        assert_eq!([1., 1., 1.,].sub([0.3, 0.2, -0.6]), [0.7, 0.8, 1.6]);
        dbg!([1., 1., 1.,].dot([0.3, 0.2, 0.6]),);
        dbg!([213., 413., 13.].normal().length());

        let test = [
            (
                [0.6952728948550719, 0.8856387014784037, 0.07952342062781792],
                [0.7684203493004392, 0.5531224343830885, 0.36618075592317245],
                [0.2803176611740497, -0.19348813955452, -0.2959717641812078],
            ),
            (
                [0.6026368171302262, -0.491278785842444, 0.22342359116365107],
                [0.18030142487852108, -0.4911784234672233, 0.847900458499978],
                [-0.3068146604939731, -0.470692441715403, -0.207423936661347],
            ),
        ];
        for (a, b, r) in test {
            assert!(a.cross(b).loose_eq(r));
        }
        dbg!(random::<[f64; 3]>());
        dbg!(random::<[f64; 3]>());
    }
}

// 将经纬度（经度, 纬度, 单位度）转换为平面 XY（UTM）坐标
pub fn jwd经纬度到xy(经度: f64, 纬度: f64) -> [f64; 2] {
    assert!(经度 > 0., "别去西半球");
    let z1 = (经度.floor() / 6.).floor() as u8 + 31;
    dbg!(z1);
    let (y, x, _) = utm::to_utm_wgs84(纬度, 经度, z1);
    [x, y]
}

#[impl_here(到经纬度)]
impl [f64; 3] {
    pub fn to_xyz(&self) -> [f64; 3] {
        let [x, y] = jwd经纬度到xy(self[0], self[1]);
        [x, y, self[2]]
    }
}
