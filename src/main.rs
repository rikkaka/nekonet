use nekonet::tensor::{
    layer::{self, Layer},
    Tensor,
};

fn main() {
    let x = Tensor::new(vec![1., 2., 3., 4., 5., 6.], vec![2, 3]);

    let fc1 = layer::Linear::new(3, 2);
    let fc2 = layer::Linear::new(2, 1);
    let y = fc1.output(x.clone());
    let y = fc2.output(y);
    y.all_require_grad(true);
    x.require_grad(false);
    y.all_init_grad();

    y.forward();

    y.set_grad_1();
    y.backward().unwrap();

    dbg!(x);
    dbg!(fc1);
    dbg!(fc2);
}
