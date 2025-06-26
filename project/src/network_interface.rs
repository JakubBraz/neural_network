use std::fs::{write, File};
use std::io::{BufRead, BufReader};
use rand::random;
use crate::image::get_training_data;
use crate::neural_network::NeuralNetwork;

pub fn test_data(neural_network: &mut NeuralNetwork) {
    check_digits(neural_network, "verification_dataset");
    check_digits(neural_network, "dataset");
}

fn check_digits(neural_network: &mut NeuralNetwork, dataset: &str) {
    for digit in 0..10 {
        let (input, _target) = get_training_data(dataset, digit, 0);
        let result = neural_network.process(&input);
        // println!("INPUT: {:?}", &input[260..270]);
        println!("{dataset} {digit}: {result:?}");
    }
}

pub fn learn(neural_network: &mut NeuralNetwork) {
    //todo add random training rate, sometimes high, for example 10.0, in rare cases even 100.0, often casual like 10.0 or 1.0

    // let training_rate = 1.0;
    // let training_rate = 10.0;
    let training_rate = 0.01;
    // let training_rate = 10.0;
    // let training_rate = 0.001;

    // for i in 0..1_000 {
    // for i in 0..500 {
    // for i in 0..60_000 {
    // for i in 0..300_000 {
    // for i in 0..1_000_000 {
    for i in 0..3_000_000 {
        if i % 500 == 0 {
            println!("{}; iteration {};", chrono::Local::now(), i);
        }
        let digit: u8 = random::<u8>() % 10;
        let index: u16 = random::<u16>() % 10773;
        let (input, target) = get_training_data("dataset", digit, index);
        neural_network.training_step(&input, &target, training_rate);
    }
}

pub fn save(neural_network: &NeuralNetwork, name: &str) {
    let serialized = neural_network.serialize();
    write(format!("networks/{name}"), serialized).unwrap();
}

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
