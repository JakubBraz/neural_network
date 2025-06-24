use crate::network_interface::{create, load};
use crate::neural_network::NeuralNetwork;

mod network_interface;
mod neural_network;
mod network_math;

fn main() {
    let network_file = "networks/network1";
    // let network = load(network_file);
    // let network = NeuralNetwork::new(&[2, 3, 2, 1]);
    let network = NeuralNetwork::new(&[5, 10, 10]);

    println!("{:?}", network);
    // println!("{:?}", network.process(&[0.1, 0.2, 0.3]));
    println!("{:?}", network.process(&[0.1, 0.2, 0.3, 0.2, 0.1]));
}
