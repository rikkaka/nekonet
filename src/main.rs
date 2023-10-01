use nekonet::{
    layer::{self, Layer},
    tensor::{operation::matmul, Tensor},
};

fn main() {
    let x = Tensor::new(vec![1., 2., 3., 4., 5., 6.], vec![2, 3]);

    let fc1 = layer::Linear::new(3, 2);
    let y = fc1.input(x.clone());
    y.all_require_grad(true);
    x.require_grad(false);
    y.all_init_grad();

    y.forward();

    y.set_grad(1.);
    y.backward().unwrap();

    dbg!(x);
    dbg!(fc1);
}
