use nekonet::{
    graph::Graph,
    layer::criterion::{self, Criterion, Reduction},
    tensor::Tensor,
};

#[test]
fn test_cross_entropy_loss() {
    let predict = Tensor::new(vec![0.5, 0.5, 0.5, 0.5], vec![2, 2]);
    let target = Tensor::new(vec![0., 1., 0., 1.], vec![2, 2]).no_grad();

    let ce = criterion::CrossEntropyLoss::new().reduction(Reduction::None);
    let loss = ce.output(predict.clone(), target.clone());

    let graph = Graph::from_output(loss.clone());
    graph.init_grad();

    graph.forward();
    graph.backward();

    assert_eq!(loss.data().borrow().as_slice(), &[0.6931472, 0.6931472]);

    let ce = criterion::CrossEntropyLoss::new().reduction(Reduction::Mean);
    let loss = ce.output(predict.clone(), target.clone());

    let graph = Graph::from_output(loss.clone());
    graph.init_grad();

    graph.forward();
    graph.backward();

    assert_eq!(loss.data().borrow().as_slice(), &[0.6931472]);

    let ce = criterion::CrossEntropyLoss::new().reduction(Reduction::Sum);
    let loss = ce.output(predict.clone(), target.clone());

    let graph = Graph::from_output(loss.clone());
    graph.init_grad();

    graph.forward();
    graph.backward();

    assert_eq!(loss.data().borrow().as_slice(), &[1.3862944]);
}
