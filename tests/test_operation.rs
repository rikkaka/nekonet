use nekonet::{
    graph::Graph,
    tensor::{
        operation::{self, sum},
        Tensor,
    },
};

#[test]
fn test_add() {
    let x1 = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let y1 = operation::add(x1.clone(), x1.clone());
    let y2 = operation::add(x1.clone(), x1.clone());

    let z = operation::add(y1.clone(), y2.clone());

    let graph = Graph::from_output(z.clone());

    graph.init_grad();

    graph.forward();
    graph.backward();

    assert_eq!(z.raw_data().as_slice(), &[4., 8., 12., 16.]);
    assert_eq!(y1.raw_grad().unwrap().as_slice(), &[1., 1., 1., 1.]);
    assert_eq!(x1.raw_grad().unwrap().as_slice(), &[4., 4., 4., 4.]);
}

#[test]
fn test_opposite() {
    let x1 = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let y = operation::opposite(x1.clone());

    let graph = Graph::from_output(y.clone());

    graph.init_grad();

    graph.forward();
    graph.zero_grad();
    graph.backward();

    assert_eq!(y.raw_data().as_slice(), &[-1., -2., -3., -4.]);
    assert_eq!(x1.raw_grad().unwrap().as_slice(), &[-1., -1., -1., -1.]);
}

#[test]
fn test_reciprol() {
    let x1 = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let y = operation::reciprocal(x1.clone());

    let graph = Graph::from_output(y.clone());

    graph.init_grad();

    graph.forward();
    graph.zero_grad();
    graph.backward();

    assert_eq!(y.raw_data().as_slice(), &[1., 0.5, 0.33333334, 0.25]);
    assert_eq!(
        x1.raw_grad().unwrap().as_slice(),
        &[-1., -0.25, -0.11111112, -0.0625]
    );
}

#[test]
fn test_scalar_mul() {
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let c = Tensor::new(vec![2.0], vec![1]);

    let y = operation::mul_scalar(x.clone(), c.clone());

    let graph = Graph::from_output(y.clone());

    graph.init_grad();

    graph.forward();
    graph.backward();

    assert_eq!(y.raw_data().as_slice(), &[2., 4., 6., 8.]);
    assert_eq!(x.raw_grad().unwrap().as_slice(), &[2., 2., 2., 2.]);
    assert_eq!(c.raw_grad().unwrap().as_slice(), &[10.]);
}

#[test]
fn test_pow() {
    let x1 = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let y = operation::pow(x1.clone(), x1.clone());

    let graph = Graph::from_output(y.clone());

    graph.init_grad();

    graph.forward();
    graph.backward();

    assert_eq!(y.raw_data().as_slice(), &[1., 4., 27., 256.]);
    assert_eq!(
        x1.raw_grad().unwrap().as_slice(),
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
    let y = operation::ln(x1.clone());

    let graph = Graph::from_output(y.clone());

    graph.init_grad();

    graph.forward();
    graph.backward();

    assert_eq!(
        y.raw_data().as_slice(),
        &[0., 2_f32.ln(), 3_f32.ln(), 4_f32.ln()]
    );
    assert_eq!(
        x1.raw_grad().unwrap().as_slice(),
        &[1., 1. / 2., 1. / 3., 1. / 4.]
    );
}

#[test]
fn test_matmul() {
    let a = Tensor::new(vec![1., 2., 3., 4., 5., 6.], vec![2, 3]);
    let b = Tensor::new(vec![1., 2., 3., 4., 5., 6.], vec![3, 2]);
    let c = operation::matmul(a.clone(), b.clone());

    let graph = Graph::from_output(c.clone());

    graph.init_grad();

    graph.forward();
    graph.backward();

    assert_eq!(c.raw_data().as_slice(), &[22., 28., 49., 64.]);
    assert_eq!(
        a.raw_grad().unwrap().as_slice(),
        &[3., 7., 11., 3., 7., 11.]
    );
    assert_eq!(b.raw_grad().unwrap().as_slice(), &[5., 5., 7., 7., 9., 9.]);
}

#[test]
fn test_concat() {
    let a = Tensor::new(vec![1., 2.], vec![1, 2]);
    let b = Tensor::new(vec![1., 2., 3., 4.], vec![2, 2]);

    let c = operation::concat(vec![a.clone(), b.clone()], 0);

    let graph = Graph::from_output(c.clone());
    graph.init_grad();

    graph.forward();
    graph.backward();

    assert_eq!(c.raw_data().as_slice(), &[1., 2., 1., 2., 3., 4.]);
    assert_eq!(c.shape().as_slice(), &[3, 2]);
    assert_eq!(a.raw_grad().unwrap().as_slice(), &[1., 1.]);
    assert_eq!(b.raw_grad().unwrap().as_slice(), &[1., 1., 1., 1.]);
}

#[test]
fn test_split() {
    let a = Tensor::new(vec![1., 2., 3., 4., 1., 2., 3., 4.], vec![2, 4]);

    let outputs = operation::split(a.clone(), 0);
    let b = outputs[0].clone();
    let c = outputs[1].clone();

    let graph = Graph::from_outputs(vec![b.clone(), c.clone()]);
    graph.init_grad();

    graph.forward();
    graph.backward();

    assert_eq!(b.raw_data().as_slice(), &[1., 2., 3., 4.]);
    assert_eq!(c.raw_data().as_slice(), &[1., 2., 3., 4.]);
    assert_eq!(
        a.raw_grad().unwrap().as_slice(),
        &[1., 1., 1., 1., 1., 1., 1., 1.]
    );
    assert_eq!(b.shape().as_slice(), &[1, 4]);
    assert_eq!(c.shape().as_slice(), &[1, 4]);

    let a = Tensor::new(vec![1., 2., 3., 4., 1., 2., 3., 4.], vec![2, 4]);

    let outputs = operation::split(a.clone(), 1);
    let b = outputs[0].clone();
    let c = outputs[1].clone();

    let graph = Graph::from_outputs(vec![b.clone(), c.clone()]);
    graph.init_grad();

    graph.forward();
    graph.backward();

    assert_eq!(b.raw_data().as_slice(), &[1., 1.]);
    assert_eq!(c.raw_data().as_slice(), &[2., 2.]);
    assert_eq!(
        a.raw_grad().unwrap().as_slice(),
        &[1., 1., 0., 0., 1., 1., 0., 0.]
    );
    assert_eq!(b.shape().as_slice(), &[2, 1]);
    assert_eq!(c.shape().as_slice(), &[2, 1]);
}

#[test]
fn test_sum() {
    let a = Tensor::new(vec![1., 2., 3., 4.], vec![1, 4]);

    let output = sum(a.clone());

    let graph = Graph::from_output(output.clone());
    graph.init_grad();

    graph.forward();
    graph.backward();

    assert_eq!(output.raw_data().as_slice(), &[10.]);
    assert_eq!(a.raw_grad().unwrap().as_slice(), &[1., 1., 1., 1.]);
}

#[test]
fn test_mean() {
    let a = Tensor::new(vec![1., 2., 3., 4.], vec![1, 4]);

    let output = operation::mean(a.clone());

    let graph = Graph::from_output(output.clone());
    graph.init_grad();

    graph.forward();
    graph.backward();

    assert_eq!(output.raw_data().as_slice(), &[2.5]);
    assert_eq!(a.raw_grad().unwrap().as_slice(), &[0.25, 0.25, 0.25, 0.25]);
}
