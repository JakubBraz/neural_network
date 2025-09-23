use std::fs::read_to_string;
use std::num::ParseFloatError;
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
    // let mut neural_network = NeuralNetwork::new(&[(WIDTH * HEIGHT) as u32, 1000, 800, 10]);
    let old_network = neural_network.clone();
    save(&neural_network, "new_network");
    test_data(&mut neural_network);
    println!("training...");
    learn(&mut neural_network);
    test_data(&mut neural_network);
    save(&neural_network, "after_learn_network");

    // let old_network = load("networks/new_network");
    let mut total_biases = 0.0;
    let mut total_weights = 0.0;
    let mut changed_biases = 0.0;
    let mut changed_weights = 0.0;
    for i in 0..neural_network.weights.len() {
        for j in 0..neural_network.weights[i].len() {
            for k in 0..neural_network.weights[i][j].len() {
                total_weights += 1.0;
                changed_weights += (neural_network.weights[i][j][k] != old_network.weights[i][j][k]) as u32 as f64;
            }
            total_biases += 1.0;
            changed_biases += (neural_network.biases[i][j] != old_network.biases[i][j]) as u32 as f64;
        }
    }
    println!("*");
    println!("biases; changed {}; total {}; {}%", changed_biases, total_biases, changed_biases / total_biases * 100.0);
    println!("weights; changed {}; total {}; {}%", changed_weights, total_weights, changed_weights / total_weights * 100.0);
    println!("TOTAL; changed {}; total {}; {}%", changed_biases + changed_weights, total_biases + total_weights, (changed_biases + changed_weights) / (total_biases + total_weights) * 100.0);
}
