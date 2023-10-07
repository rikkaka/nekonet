use nekonet::{graph::Graph, layer, tensor::Tensor};

#[test]
fn test_linear() {
    let x = Tensor::new(vec![1., 2., 3., 4., 5., 6.], vec![2, 3]);

    let fc1 = layer::Linear::new(3, 2);
    let y = fc1.output(x.clone());

    let graph = Graph::from_output(y.clone());
    graph.init_grad();

    graph.forward();
    graph.backward();

    assert_eq!(y.shape(), &[2, 2]);
    assert_eq!(
        fc1.weight().raw_grad().unwrap().as_slice(),
        &[5.0, 5.0, 7.0, 7.0, 9.0, 9.0]
    );
    assert_eq!(fc1.bias().raw_grad().unwrap().as_slice(), &[2.0, 2.0]);
}
