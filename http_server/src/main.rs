#[macro_use] extern crate rocket;

use neural_network_lib::network_interface;
use neural_network_lib::neural_network::NeuralNetwork;
use rocket::fs::FileServer;
use rocket::State;


#[post("/predict", data = "<body>")]
fn predict(body: &str, neural_network: &State<NeuralNetwork>) -> String {
    let input: Vec<f64> = body[1..(body.len() - 1)].split(",").map(|x| x.parse::<f64>().unwrap()).collect();
    let result = neural_network.process(&input);
    format!("[{}]", result.iter().map(|x| x.to_string()).collect::<Vec<String>>().join(","))
}

#[launch]
fn rocket() -> _ {
    let neural_network = network_interface::load("networks/3");

    rocket::build()
        .manage(neural_network)
        .mount("/", FileServer::from("static")) // serve the index.html on / GET
        .mount("/", routes![predict]) // handle /predict POST
}
