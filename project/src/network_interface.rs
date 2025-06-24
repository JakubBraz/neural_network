use std::fs::File;
use std::io::{BufRead, BufReader};
use rand::random;
use crate::neural_network::NeuralNetwork;

pub fn process(input: &[f32], network: &NeuralNetwork) {

}

pub fn create(layers: &[u32], name: &str)  {

}

pub fn load(file_name: &str) -> NeuralNetwork {
//     let file = File::open(file_name).unwrap();
//     let reader = BufReader::new(file);
//
//     let mut result: Vec<Vec<f32>> = vec![Vec::new()];
//
//     for line in reader.lines() {
//         match line.unwrap().trim().parse::<f32>() {
//             Ok(v) => result.last_mut().unwrap().push(v),
//             Err(_e) => if result.last().unwrap().len() > 0 { result.push(Vec::new()); }
//         }
//     }
//
//     while result.last().unwrap().len() == 0 {
//         result.pop();
//     }
//
    NeuralNetwork::empty()
}
