use nekonet::tensor::{operation, Tensor};

#[test]
fn test_add() {
    let x1 = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    x1.init_grad();

    let y1 = operation::add(x1.clone(), x1.clone());
    y1.init_grad();
    let y2 = operation::add(x1.clone(), x1.clone());
    y2.init_grad();

    let z = operation::add(y1.clone(), y2.clone());
    z.init_grad();

    z.forward();
    z.one_grad();
    z.all_require_grad(true);
    z.backward().unwrap();

    assert_eq!(z.data().borrow().as_slice(), &[4., 8., 12., 16.]);
    assert_eq!(y1.grad().unwrap().borrow().as_slice(), &[1., 1., 1., 1.]);
    assert_eq!(x1.grad().unwrap().borrow().as_slice(), &[4., 4., 4., 4.]);
}

#[test]
fn test_opposite() {
    let x1 = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    x1.init_grad();

    let y = operation::opposite(x1.clone());
    y.init_grad();

    y.forward();
    y.one_grad();
    y.all_require_grad(true);
    y.backward().unwrap();

    assert_eq!(y.data().borrow().as_slice(), &[-1., -2., -3., -4.]);
    assert_eq!(
        x1.grad().unwrap().borrow().as_slice(),
        &[-1., -1., -1., -1.]
    );
}

#[test]
fn test_reciprol() {
    let x1 = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    x1.init_grad();

    let y = operation::reciprocal(x1.clone());
    y.init_grad();

    y.forward();
    y.one_grad();
    y.all_require_grad(true);
    y.backward().unwrap();

    assert_eq!(y.data().borrow().as_slice(), &[1., 0.5, 0.33333334, 0.25]);
    assert_eq!(
        x1.grad().unwrap().borrow().as_slice(),
        &[-1., -0.25, -0.11111111, -0.0625]
    );
}

#[test]
fn test_scalar_mul() {
    let x1 = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    x1.init_grad();

    let y = operation::scalar_mul(x1.clone(), x1.clone());
    y.init_grad();

    y.forward();
    y.one_grad();
    y.all_require_grad(true);
    y.backward().unwrap();

    assert_eq!(y.data().borrow().as_slice(), &[1., 4., 9., 16.]);
    assert_eq!(x1.grad().unwrap().borrow().as_slice(), &[2., 4., 6., 8.]);
}

#[test]
fn test_pow() {
    let x1 = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    x1.init_grad();

    let y = operation::pow(x1.clone(), x1.clone());
    y.init_grad();

    y.forward();
    y.one_grad();
    y.all_require_grad(true);
    y.backward().unwrap();

    assert_eq!(y.data().borrow().as_slice(), &[1., 4., 27., 256.]);
    assert_eq!(
        x1.grad().unwrap().borrow().as_slice(),
        &[
            1.,
            4. * (1. + 2_f32.ln()),
            27. * (1. + 3_f32.ln()),
            256. * (1. + 4_f32.ln())
        ]
    );
}

#[test]
fn test_ln() {
    let x1 = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    x1.init_grad();

    let y = operation::ln(x1.clone());
    y.init_grad();

    y.forward();
    y.one_grad();
    y.all_require_grad(true);
    y.backward().unwrap();

    assert_eq!(
        y.data().borrow().as_slice(),
        &[0., 2_f32.ln(), 3_f32.ln(), 4_f32.ln()]
    );
    assert_eq!(
        x1.grad().unwrap().borrow().as_slice(),
        &[1., 1. / 2., 1. / 3., 1. / 4.]
    );
}

#[test]
fn test_matmul() {
    let a = Tensor::new(vec![1., 2., 3., 4., 5., 6.], vec![2, 3]);
    a.init_grad();
    let b = Tensor::new(vec![1., 2., 3., 4., 5., 6.], vec![3, 2]);
    b.init_grad();

    let c = operation::matmul(a.clone(), b.clone());
    c.init_grad();

    c.forward();
    c.one_grad();
    c.all_require_grad(true);
    c.backward().unwrap();

    assert_eq!(c.data().borrow().as_slice(), &[22., 28., 49., 64.]);
    assert_eq!(
        a.grad().unwrap().borrow().as_slice(),
        &[3., 7., 11., 3., 7., 11.]
    );
    assert_eq!(
        b.grad().unwrap().borrow().as_slice(),
        &[5., 5., 7., 7., 9., 9.]
    );
}

#[test]
fn test_concat() {
    let a = Tensor::new(vec![1., 2., 3., 4.], vec![2, 2]);
    let b = Tensor::new(vec![1., 2., 3., 4.], vec![2, 2]);

    let c = operation::concat(vec![a.clone(), b.clone()]);

    c.forward();
    c.all_require_grad(true);
    c.all_init_grad();
    c.one_grad();
    c.backward().unwrap();

    assert_eq!(c.data().borrow().as_slice(), &[1., 2., 3., 4., 1., 2., 3., 4.]);
    assert_eq!(c.shape().as_slice(), &[4, 2]);
    assert_eq!(a.grad().unwrap().borrow().as_slice(), &[1., 1., 1., 1.]);
    assert_eq!(b.grad().unwrap().borrow().as_slice(), &[1., 1., 1., 1.]);
}

#[test]
fn test_split_rows() {
    let a = Tensor::new(vec![1., 2., 3., 4., 1., 2., 3., 4.], vec![2, 4]);
    
    let outputs = operation::split_rows(a.clone());
    let b = outputs[0].clone();
    let c = outputs[1].clone();

    b.forward();
    c.forward();
    b.all_require_grad(true);
    c.all_require_grad(true);
    b.all_init_grad();
    c.all_init_grad();
    b.one_grad();
    c.one_grad();
    b.backward().unwrap();
    c.backward().unwrap();

    assert_eq!(b.data().borrow().as_slice(), &[1., 2., 3., 4.]);
    assert_eq!(c.data().borrow().as_slice(), &[1., 2., 3., 4.]);
    assert_eq!(a.grad().unwrap().borrow().as_slice(), &[1., 1., 1., 1., 1., 1., 1., 1.]);
    assert_eq!(b.shape().as_slice(), &[1, 4]);
    assert_eq!(c.shape().as_slice(), &[1, 4]);


}