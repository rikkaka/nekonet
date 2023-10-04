use nekonet::{tensor::{Tensor, operation}, optimizer::{SGD, Optimizer}};

#[test]
fn test_sgd() {
    let x1 = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let x2 = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

    let y1 = operation::matmul(x1.clone(), x2.clone());
    let y2 = Tensor::new(vec![1.0, 2.0, 1.0, 2.0], vec![2, 2]);

    let z = operation::add(y1.clone(), y2.clone());

    z.forward();

    z.all_require_grad(true);
    x1.require_grad(false);

    z.all_init_grad();
    z.one_grad();
    z.backward().unwrap();

    let mut sgd = SGD::new(0.01);
    sgd.add_params(vec![x2.clone(), y2.clone()]);
    sgd.step();

    assert_eq!(z.data().borrow().as_slice(), &[8., 12., 16., 24.]);
    assert_eq!(x2.data().borrow().as_slice(), &[0.96, 1.96, 2.94, 3.94]);
    assert_eq!(y2.data().borrow().as_slice(), &[0.99, 1.99, 0.99, 1.99]);

    z.forward();
    assert_eq!(z.data().borrow().as_slice(), &[7.83, 11.83, 15.63, 23.63]);

    let mut last_data = z.data().borrow().clone();
    for _ in 0..10 {
        z.all_zero_grad();
        z.one_grad();
        z.backward().unwrap();
        
        sgd.step();

        z.forward();
        let now_data = z.data().borrow().clone();
        assert!(now_data < last_data);
        last_data = now_data;
    }

}