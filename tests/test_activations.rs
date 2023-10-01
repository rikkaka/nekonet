use nekonet::tensor::{activation, Tensor};

#[test]
fn test_relu() {
    let x1 = Tensor::new(vec![-1., 2., -3., 4.], vec![2, 2]);
    let y = activation::relu(x1.clone());

    y.forward();
    y.all_require_grad(true);
    y.all_init_grad();
    y.set_grad_1();
    y.backward().unwrap();

    assert_eq!(y.data().borrow().as_slice(), &[0., 2., 0., 4.]);
    assert_eq!(x1.grad().unwrap().borrow().as_slice(), &[0., 1., 0., 1.]);
}

#[test]
fn test_softmax() {
    let x1 = Tensor::new(vec![1., 1., 2., 2.], vec![2, 2]);
    let y = activation::softmax(x1.clone());

    y.forward();
    y.all_require_grad(true);
    y.all_init_grad();
    y.set_grad_1();
    y.backward().unwrap();

    assert_eq!(y.data().borrow().as_slice(), &[0.5, 0.5, 0.5, 0.5]);
    assert_eq!(x1.grad().unwrap().borrow().as_slice(), &[0., 0., 0., 0.]);
}
