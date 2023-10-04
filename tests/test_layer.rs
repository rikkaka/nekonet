use nekonet::{layer, tensor::Tensor};

#[test]
fn test_linear() {
    let x = Tensor::new(vec![1., 2., 3., 4., 5., 6.], vec![2, 3]);

    let fc1 = layer::Linear::new(3, 2);
    let y = fc1.output(x.clone());
    y.all_require_grad(true);
    x.require_grad(false);
    y.all_init_grad();

    y.forward();

    y.one_grad();
    y.backward().unwrap();

    assert_eq!(y.shape(), &[2, 2]);
    assert_eq!(
        fc1.weight().grad().unwrap().borrow().as_slice(),
        &[5.0, 5.0, 7.0, 7.0, 9.0, 9.0]
    );
    assert_eq!(fc1.bias().grad().unwrap().borrow().as_slice(), &[2.0, 2.0]);
}
