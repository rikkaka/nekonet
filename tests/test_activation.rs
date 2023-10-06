use nekonet::{graph::Graph, layer::activation, tensor::Tensor};

#[test]
fn test_relu() {
    let x1 = Tensor::new(vec![-1., 2., -3., 4.], vec![2, 2]);
    let y = activation::ReLU().output(x1.clone());

    let graph = Graph::from_output(y.clone());
    graph.init_grad();

    graph.forward();
    graph.backward();

    assert_eq!(y.data().borrow().as_slice(), &[0., 2., 0., 4.]);
    assert_eq!(x1.grad().unwrap().borrow().as_slice(), &[0., 1., 0., 1.]);
}

#[test]
fn test_softmax() {
    let x1 = Tensor::new(vec![1., 1., 2., 2.], vec![2, 2]);
    let y = activation::Softmax().output(x1.clone());

    let graph = Graph::from_output(y.clone());
    graph.init_grad();

    graph.forward();
    graph.backward();

    assert_eq!(y.data().borrow().as_slice(), &[0.5, 0.5, 0.5, 0.5]);
    assert_eq!(x1.grad().unwrap().borrow().as_slice(), &[0., 0., 0., 0.]);
}
