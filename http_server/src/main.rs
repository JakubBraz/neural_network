#[macro_use] extern crate rocket;

use std::fs::read_dir;
use neural_network_lib::network_interface;
use neural_network_lib::neural_network::NeuralNetwork;
use rocket::fs::FileServer;
use rocket::serde::Deserialize;
use rocket::serde::json::Json;
use rocket::State;

#[derive(Deserialize, Debug)]
#[serde(crate = "rocket::serde")]
struct TrainingData {
    digit: u8,
    image_data: Vec<f32>,
}

#[post("/predict", data = "<body>")]
fn predict(body: &str, neural_network: &State<NeuralNetwork>) -> String {
    let input: Vec<f32> = body[1..(body.len() - 1)]
        .split(",")
        .map(|x| x.parse::<f32>().unwrap())
        .collect();
    let result = neural_network.process(&input);
    format!("[{}]", result.iter().map(|x| x.to_string()).collect::<Vec<String>>().join(","))
}

#[post("/training", data = "<data>")]
fn training(data: Json<TrainingData>) -> String {
    let old_count = 10773;
    let digit_count = count_files();
    neural_network_lib::image::save_training_data("training_data", data.digit, &data.image_data, digit_count[data.digit as usize] + old_count);
    format!("{}", digit_count.iter().sum::<u32>() + 1)
}

#[get("/count")]
fn count() -> String {
    format!("{}", count_files().iter().sum::<u32>())
}

fn count_files() -> Vec<u32> {
    (0..10).map(|x| read_dir(format!("training_data/{}/{}/", x, x)).unwrap().count() as u32)
        .collect()
}

#[launch]
fn rocket() -> _ {
    // let neural_network = network_interface::load("networks/3");
    // let neural_network = network_interface::load("../project/networks/after_learn_network");
    let neural_network = network_interface::load("networks/manual_dataset_5min");

    rocket::build()
        .manage(neural_network)
        .mount("/", FileServer::from("static")) // serve the index.html on / GET
        .mount("/", routes![predict, training, count]) // handle HTTP endpoints
}
