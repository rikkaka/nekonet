use nekonet::{
    layer::criterion::{self, Reduction, Criterion},
    tensor::Tensor,
};

#[test]
fn test_cross_entropy_loss() {
    let predict = Tensor::new(vec![0.5, 0.5, 0.5, 0.5], vec![2, 2]);
    let target = Tensor::new(vec![0., 1., 0., 1.], vec![2, 2]);

    let ce = criterion::CrossEntropyLoss::new().reduction(Reduction::None);
    let loss = ce.output(predict.clone(), target.clone());

    loss.all_require_grad(true);
    target.require_grad(false);

    loss.forward();
    loss.all_init_grad();
    loss.one_grad();
    loss.backward().unwrap();

    assert_eq!(loss.data().borrow().as_slice(), &[0.6931472, 0.6931472]);

    let ce = criterion::CrossEntropyLoss::new().reduction(Reduction::Mean);
    let loss = ce.output(predict.clone(), target.clone());

    loss.forward();
    loss.all_init_grad();
    loss.one_grad();
    loss.backward().unwrap();

    assert_eq!(loss.data().borrow().as_slice(), &[0.6931472]);

    let ce = criterion::CrossEntropyLoss::new().reduction(Reduction::Sum);
    let loss = ce.output(predict.clone(), target.clone());

    loss.forward();
    loss.all_init_grad();
    loss.one_grad();
    loss.backward().unwrap();

    assert_eq!(loss.data().borrow().as_slice(), &[1.3862944]);
}
