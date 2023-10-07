use nekonet::{graph::Graph, layer::activation, tensor::Tensor};

#[test]
fn test_relu() {
    let x1 = Tensor::new(vec![-1., 2., -3., 4.], vec![2, 2]);
    let y = activation::ReLU().output(x1.clone());

    let graph = Graph::from_output(y.clone());
    graph.init_grad();

    graph.forward();
    graph.backward();

    assert_eq!(y.raw_data().as_slice(), &[0., 2., 0., 4.]);
    assert_eq!(x1.raw_grad().unwrap().as_slice(), &[0., 1., 0., 1.]);
}

#[test]
fn test_softmax() {
    let x1 = Tensor::new(vec![1., 2., 3., 4.], vec![1, 4]);
    let y = activation::Softmax().output(x1.clone());

    let graph = Graph::from_output(y.clone());
    graph.init_grad();

    graph.forward();
    graph.backward();

    assert!(y
        .raw_data()
        .iter()
        .zip([0.0320586, 0.08714432, 0.23688284, 0.6439143])
        .all(|(a, b)| (a - b).abs() < 1e-6));
    assert!(x1
        .raw_grad()
        .unwrap()
        .iter()
        .zip([0., 0., 0., 0.])
        .all(|(a, b)| (a - b).abs() < 1e-6));
}
