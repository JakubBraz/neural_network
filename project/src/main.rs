use crate::image::{get_training_data, read, HEIGHT, WIDTH};
use crate::network_interface::{create, learn, load, save, test_data};
use crate::neural_network::NeuralNetwork;

mod network_interface;
mod neural_network;
mod network_math;
mod image;

fn main() {
    // let mut neural_network = NeuralNetwork::new(&[(WIDTH * HEIGHT) as u32, 100, 10]);
    let mut neural_network = NeuralNetwork::new(&[(WIDTH * HEIGHT) as u32, 800, 10]);
    save(&neural_network, "new_network");
    test_data(&mut neural_network);
    println!("training...");
    learn(&mut neural_network);
    test_data(&mut neural_network);
    save(&neural_network, "after_learn_network");
}

// fn main() {
//     let network_file = "networks/network1";
//     // let network = load(network_file);
//     // let network = NeuralNetwork::new(&[2, 3, 2, 1]);
//     let mut network = NeuralNetwork::new(&[5, 10, 10]);
//
//     println!("{:?}", network);
//     // println!("{:?}", network.process(&[0.1, 0.2, 0.3]));
//     println!("{:?}", network.process(&[0.1, 0.2, 0.3, 0.2, 0.1]));
//
//     println!("learning...");
//     for i in 0..100 {
//         network.training_step(&[0.1, 0.2, 0.3, 0.2, 0.1], &[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 10.0);
//     }
//     println!("{:?}", network);
//     println!("{:?}", network.process(&[0.1, 0.2, 0.3, 0.2, 0.1]));
// }

// fn main() {
//     read();
//     let (input, target) = get_training_data(5, 3754);
//     println!("{:?}", input);
//     println!("{}", input.len());
//     println!("{:?}", target);
// }