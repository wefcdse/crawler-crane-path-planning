// utils.rs: 通用工具集合
// - 宏 `time!`：快速测时打印
// - 宏 `disable!`：用于在代码中抑制未使用变量的警告（用于占位）
// - `index_xyz`：通过类型标记访问数组的 X/Y/Z 元素（可读性增强）
// - 随机点、OBJ 写出、计数器等实用函数
// 本文件主要是提供小且常用的工具函数/类型，便于其他模块复用。



use std::sync::atomic::AtomicUsize;

use local_wrap::Wrap;

#[macro_export]
macro_rules! time {
    // ($st:ident) => {
    
    // };
    ($($st:ident),*) => {
        $(let $st = ::std::time::Instant::now();)*
    };
    ($st:ident, $info:literal) => {
        ::std::println!("{}: {:?}", $info, ::std::time::Instant::elapsed(&$st));
    };
}
#[macro_export]
macro_rules! disable {
    ($($a:ident),*) => {
        $(
        #[allow(unused_variables)]
        let $a = {
            struct Disabled;
            Disabled
        };
        {
            fn d<T>(_v:T){}
            d($a);
            // $a;
        }
        )*
    };
}

pub mod index_xyz {
    use std::ops::{Index, IndexMut};

    #[derive(Debug, Clone, Copy, Default, Hash, PartialEq, Eq, PartialOrd, Ord)]
    pub struct X;
    #[derive(Debug, Clone, Copy, Default, Hash, PartialEq, Eq, PartialOrd, Ord)]
    pub struct Y;
    #[derive(Debug, Clone, Copy, Default, Hash, PartialEq, Eq, PartialOrd, Ord)]
    pub struct Z;
    
    macro_rules! implthis {
        ($a:tt, $b:literal) => {
            impl<const L: usize, T> Index<$a> for [T; L] {
                type Output = T;

                fn index(&self, _: $a) -> &Self::Output {
                    &self[$b]
                }
            }
            impl<const L: usize, T> IndexMut<$a> for [T; L] {
                fn index_mut(&mut self, _: $a) -> &mut Self::Output {
                    &mut self[$b]
                }
            }
        };
    }
    implthis!(X, 0);
    implthis!(Y, 1);
    implthis!(Z, 2);
    #[test]
    #[allow(clippy::deref_addrof)]
    fn t() {
        let mut a = [318, 15, 5];
        assert_eq!(a[0], a[X]);
        assert_eq!(a[1], a[Y]);
        assert_eq!(a[2], a[Z]);
        assert!(*(&mut a[0]) == *(&mut a[X]));
        assert!(*(&mut a[1]) == *(&mut a[Y]));
        assert!(*(&mut a[2]) == *(&mut a[Z]));

        let mut a = [318, 15];
        assert_eq!(a[0], a[X]);
        assert_eq!(a[1], a[Y]);
        assert!(*(&mut a[0]) == *(&mut a[X]));
        assert!(*(&mut a[1]) == *(&mut a[Y]));
    }
}

pub fn random_point_in(min: [f64; 3], max: [f64; 3]) -> [f64; 3] {
    use index_xyz::*;
    let (xs, ys, zs) = (max[X] - min[X], max[Y] - min[Y], max[Z] - min[Z]);
    assert!(xs >= 0.);
    assert!(ys >= 0.);
    assert!(zs >= 0.);
    [
        min[X] + rand::random::<f64>() * xs,
        min[Y] + rand::random::<f64>() * ys,
        min[Z] + rand::random::<f64>() * zs,
    ]
}

// 将一系列 3D 点写为 OBJ 格式的顶点与按顺序相连的线段（v + l）
pub fn write_line_to_obj<W: std::io::Write>(
    file: &mut W,
    line: impl IntoIterator<Item = [f64; 3]>,
) -> std::io::Result<()> {
    use index_xyz::*;
    let mut len = 0;
    for point in line {
        writeln!(file, "v {} {} {}", point[X], point[Y], point[Z])?;
        len += 1;
    }
    for i in 1..len {
        writeln!(file, "l {} {}", i, i + 1)?;
    }

    Ok(())
}

pub fn write_line_to_obj_r90<W: std::io::Write>(
    file: &mut W,
    line: impl Iterator<Item = [f64; 3]>,
) -> std::io::Result<()> {
    use index_xyz::*;
    let mut len = 0;
    for point in line {
        writeln!(file, "v {} {} {}", -point[Y], point[X], point[Z])?;
        len += 1;
    }
    for i in 1..len {
        writeln!(file, "l {} {}", i, i + 1)?;
    }

    Ok(())
}

pub trait StringErr<T> {
    fn to_msg(self) -> Result<T, String>;
}
impl<T, E: ToString> StringErr<T> for Result<T, E> {
    fn to_msg(self) -> Result<T, String> {
        self.map_err(|e| e.to_string())
    }
}

pub struct Counter(AtomicUsize);
impl Counter {
    pub fn new() -> Self {
        Self(AtomicUsize::new(0))
    }
    pub fn show(&self) -> usize {
        self.0.load(std::sync::atomic::Ordering::Relaxed)
    }
    pub fn count(&self) {
        self.0.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}

pub mod local_wrap {
    use std::{
        fmt::{Debug, Display},
        ops::{Deref, DerefMut},
    };

    #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
    #[repr(transparent)]
    pub struct Wrap<T: ?Sized>(T);

    impl<T: ?Sized> Wrap<T> {
        #[allow(unused)]
        pub fn wrap_ref(&self) -> &Self {
            self
        }
        #[allow(unused)]
        pub fn wrap_mut(&mut self) -> &mut Self {
            self
        }
    }

    impl<T> Wrap<T> {
        #[allow(unused)]
        pub fn wrap(self) -> Self {
            self
        }
        #[allow(unused)]
        pub fn into_inner(self) -> T {
            self.0
        }
    }
    impl<T: Copy> Wrap<T> {
        #[allow(unused)]
        pub fn copy(&self) -> Self {
            *self
        }
    }

    impl<T: ?Sized + Debug> Debug for Wrap<T> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            self.0.fmt(f)
        }
    }
    impl<T: ?Sized + Display> Display for Wrap<T> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            self.0.fmt(f)
        }
    }

    impl<T: ?Sized> Deref for Wrap<T> {
        type Target = T;

        fn deref(&self) -> &Self::Target {
            &self.0
        }
    }
    impl<T: ?Sized> DerefMut for Wrap<T> {
        fn deref_mut(&mut self) -> &mut Self::Target {
            &mut self.0
        }
    }

    #[allow(unused)]
    pub trait WrapSelf: Sized {
        fn wrap(self) -> Wrap<Self> {
            Wrap(self)
        }
    }

    #[allow(unused)]
    pub trait WrapRef {
        fn wrap_ref(&self) -> &Wrap<Self> {
            // SAFETY: Wrap is #[repr(transparent)]
            unsafe { std::mem::transmute(self) }
        }
        fn wrap_mut(&mut self) -> &mut Wrap<Self> {
            // SAFETY: Wrap is #[repr(transparent)]
            unsafe { std::mem::transmute(self) }
        }
    }

    impl<T> WrapSelf for T {}
    impl<T: ?Sized> WrapRef for T {}

    #[test]
    fn t() {
        let a = "fff";
        let aa = a.wrap_ref();

        let b = "fffffff".to_owned();
        let bb = b.wrap_ref();
        // drop(b);
        dbg!(bb);
        let b = b.wrap();
        dbg!(aa);
        dbg!(b);
        let a = vec![12, 14, 15, 17];
        let mut a = a.wrap();
        dbg!(a[2]);
        a[1] = 51;
        dbg!(a[1]);

        let a = vec![1, 2, 3];
        // assert_eq!()
        let a1 = a.wrap_ref();
        assert_eq!(&raw const a1[1], &raw const a[1]);

        // aa.0;
    }
}

impl<T> Wrap<[T; 2]> {
    pub fn extend_one(self, next: T) -> [T; 3] {
        let [a, b] = self.into_inner();
        [a, b, next]
    }
}
#[impl_here::impl_here(Extend)]
impl [f64; 2] {
    pub fn extend_one(self, next: f64) -> [f64; 3] {
        let [a, b] = self;
        [a, b, next]
    }
}
