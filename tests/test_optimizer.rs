use nekonet::{
    graph::Graph,
    optimizer::{Optimizer, SGD},
    tensor::{operation, Tensor},
};

#[test]
fn test_sgd() {
    let x1 = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let x2 = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

    let y1 = operation::matmul(x1.clone(), x2.clone());
    let y2 = Tensor::new(vec![1.0, 2.0, 1.0, 2.0], vec![2, 2]);

    let z = operation::add(y1.clone(), y2.clone());

    let graph = Graph::from_output(z.clone());

    graph.init_grad();

    graph.forward();
    graph.zero_grad();
    graph.backward();

    let mut sgd = SGD::new(0.01);
    sgd.add_params(vec![x2.clone(), y2.clone()]);
    sgd.step();

    assert_eq!(z.raw_data().as_slice(), &[8., 12., 16., 24.]);
    assert_eq!(x2.raw_data().as_slice(), &[0.96, 1.96, 2.94, 3.94]);
    assert_eq!(y2.raw_data().as_slice(), &[0.99, 1.99, 0.99, 1.99]);

    graph.forward();
    assert_eq!(z.raw_data().as_slice(), &[7.83, 11.83, 15.63, 23.63]);

    let mut last_data = z.raw_data();
    for _ in 0..10 {
        graph.zero_grad();
        graph.backward();

        sgd.step();

        graph.forward();
        let now_data = z.raw_data();
        assert!(now_data < last_data);
        last_data = now_data;
    }
}
