use std::fs::{read, read_to_string, write, File};
use std::io::{BufRead, BufReader};
use std::time::{Duration, Instant};
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
        let result = neural_network.process_mutable(&input);
        // println!("INPUT: {:?}", &input[260..270]);
        println!("{dataset} {digit}: {result:?}");
    }
}

pub fn learn(neural_network: &mut NeuralNetwork) {
    //todo add random training rate, sometimes high, for example 10.0, in rare cases even 100.0, often casual like 10.0 or 1.0

    // let training_rate = 1.0;
    // let training_rate = 10.0;
    // let training_rate = 10.0;
    // let training_rate = 0.001;
    // let training_rate = 1.0;
    // let training_rate = 0.01;
    // let training_rate = 0.1;
    let training_rate = 0.5;

    // let duration = Duration::from_secs(60 * 60 * 10);
    let duration = Duration::from_secs(60 * 10);

    let mut i = 0;
    let time = Instant::now();

    let minutes = duration.as_secs_f32() / 60.0;
    let msg = match minutes {
        x if x > 60.0 => format!("{} hours", minutes / 60.0),
        x if x > 1.0 => format!("{} minutes", minutes),
        _ => format!("{:?}", duration)
    };
    println!("{}; the training will last {}", chrono::Local::now(), msg);
    while time.elapsed() < duration {
    // for i in 0..10_000 {
    // for i in 0..100 {
    // for i in 0..500 {
    // for i in 0..60_000 {
    // for i in 0..300_000 {
    // for i in 0..1_000_000 {
    // for i in 0..3_000_000 {
        if i % 500 == 0 {
            println!("{}; iteration {};", chrono::Local::now(), i);
        }
        let digit: u8 = random::<u8>() % 10;
        let index: u16 = random::<u16>() % 10773;
        let (input, target) = get_training_data("dataset", digit, index);
        neural_network.training_step(&input, &target, training_rate);
        i += 1;
    }
    println!("{}; iteration {};", chrono::Local::now(), i);
}

pub fn save(neural_network: &NeuralNetwork, name: &str) {
    let serialized = neural_network.serialize();
    write(format!("networks/{name}"), serialized).unwrap();
}

pub fn load(file_name: &str) -> NeuralNetwork {
    let network = read_to_string(file_name).unwrap();
    NeuralNetwork::deserialize(&network)
}

pub fn process(input: &[f32], network: &NeuralNetwork) {

}

pub fn create(layers: &[u32], name: &str)  {

}

