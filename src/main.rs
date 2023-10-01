use nekonet::tensor::{operation::matmul, Tensor};

fn main() {
    let a = Tensor::new(vec![1., 2., 3., 4., 5., 6.], vec![2, 3]);
    a.init_grad();
    let b = Tensor::new(vec![1., 4., 2., 5., 3., 6.], vec![3, 2]);
    b.init_grad();

    let c = matmul(a.clone(), b.clone());
    c.init_grad();

    c.forward();
    c.set_grad(1.);
    c.backward().unwrap();

    c.dbg();
    a.dbg();
    b.dbg();
}
