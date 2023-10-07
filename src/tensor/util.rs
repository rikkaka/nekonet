#[macro_export]
macro_rules! into_2d {
    ($($param:expr),* $(,)? ) => {
        (
            $(
                $param.into_dimensionality::<ndarray::Ix2>().unwrap(),
            )*
        )
    };
}
pub use into_2d;
