use rand::random;
use crate::network_math;

#[derive(Debug, Clone)]
pub struct NeuralNetwork {
    pub weights: Vec<Vec<Vec<f32>>>,
    pub biases: Vec<Vec<f32>>,
    pre_activations: Vec<Vec<f32>>,
    activations: Vec<Vec<f32>>,
}

impl PartialEq for NeuralNetwork {
    fn eq(&self, other: &Self) -> bool {
        self.weights == other.weights && self.biases == other.biases
    }
}

impl NeuralNetwork {
    fn activation(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    pub fn new(layers: &[u32]) -> Self {
        // first layer is input, the last one is output
        // there is no bias layer for first layer (input)
        let mut weights: Vec<Vec<Vec<f32>>> = Vec::new();
        let mut biases: Vec<Vec<f32>> = Vec::new();
        for i in 1..layers.len() {
            weights.push(Vec::new());
            biases.push(Vec::new());
            for _ in 0..layers[i] {
                let mut v: Vec<f32> = Vec::new();
                for _ in 0..layers[i - 1] {
                    //todo a lot depend on inital weights, very often all goes to 0 :/

                    // v.push((random::<f32>() * 2.0 - 1.0) * 0.1);
                    v.push((random::<f32>() * 2.0 - 1.0) * 0.2);
                    // v.push(random());
                }
                weights.last_mut().unwrap().push(v);
                // biases.last_mut().unwrap().push(random::<f32>() * 10.0 - 5.0);
                // biases.last_mut().unwrap().push(random::<f32>() * 100.0 - 50.0);
                // biases.last_mut().unwrap().push((random::<f32>() * 2.0 - 1.0) * 0.1);
                biases.last_mut().unwrap().push((random::<f32>() * 2.0 - 1.0) * 0.2);
            }
        }

        let mut pre_activations: Vec<Vec<f32>> = Vec::new();
        let mut activations: Vec<Vec<f32>> = Vec::new();

        for i in 0..biases.len() {
            let empty_vec = vec![0.0; biases[i].len()];
            pre_activations.push(empty_vec.clone());
            activations.push(empty_vec);
        }

        NeuralNetwork {
            weights, biases, pre_activations, activations
        }
    }

    pub fn empty() -> Self {
        Self { weights: vec![], biases: vec![], pre_activations: vec![], activations: vec![] }
    }

    pub fn process(&self, input: &[f32]) -> Vec<f32> {
        //todo improve it, parallelize and calculate product in sum in a one go instead of separate functions, limit allocations
        let mut prev = input.to_vec();
        let mut result = Vec::new();
        for i in 0..self.weights.len() {
            result = vec![0.0; self.weights[i].len()];
            network_math::product(&self.weights[i], &prev, &mut result);
            network_math::sum(&mut result, &self.biases[i]);
            for j in 0..result.len() {
                result[j] = NeuralNetwork::activation(result[j]);
            }
            prev = result.clone();
        }
        result
    }

    pub fn process_mutable(&mut self, input: &[f32]) -> Vec<f32> {
        //todo improve it, parallelize and calculate product in sum in a one go instead of separate functions
        for i in 0..self.weights.len() {
            let prev = if i==0 { input } else { &self.activations[i-1] };
            network_math::product(&self.weights[i], prev, &mut self.pre_activations[i]);
            network_math::sum(&mut self.pre_activations[i], &self.biases[i]);
            for j in 0..self.activations[i].len() {
                self.activations[i][j] = NeuralNetwork::activation(self.pre_activations[i][j]);
            }
        }
        self.activations.last().unwrap().clone()
    }

    pub fn training_step(&mut self, inputs: &[f32], targets: &[f32], learning_rate: f32) {
        let processed = self.process_mutable(inputs);

        let mut deltas: Vec<Vec<f32>> = self.biases.iter().map(|x| x.iter().map(|_| 0.0).collect()).collect();
        let mut gradients_weights = self.weights.clone();
        let mut gradients_biases = self.biases.clone();

        // calculate last layer gradients
        let layer = self.weights.len() - 1;
        deltas[layer] = processed.iter().zip(targets.iter()).map(|(&a, &y)| (a - y) * a * (1.0 - a)).collect();
        // deltas[layer] = processed.iter().zip(targets.iter()).map(|(&a, &y)| (y - a) * a * (1.0 - a)).collect();
        Self::update_gradients(layer, &deltas, inputs, &mut gradients_biases, &mut gradients_weights, &self.activations);
        // for i in 0..deltas[layer].len() {
        //     gradients_biases[layer][i] = deltas[layer][i];
        //     for j in 0..gradients_weights[layer][i].len() {
        //         let prev = if layer == 0 { inputs } else { &self.activations[layer-1] };
        //         gradients_weights[layer][i][j] = deltas[layer][i] * prev[j];
        //     }
        // }

        // println!("DELTAS OUTPUT {:?}", deltas[layer]);
        // panic!();

        // calculate gradients of the remaining layers
        for layer in (0 .. (self.weights.len() - 1)).rev() {
            for i in 0..self.biases[layer].len() {
            // for i in 0..self.weights[layer][0].len() {
            // for i in 0..self.weights[layer + 1][0].len() {
                let mut tmp = 0.0;
                for j in 0..self.biases[layer + 1].len() {
                    tmp += deltas[layer + 1][j] * self.weights[layer + 1][j][i];
                }
                deltas[layer][i] = tmp * &self.activations[layer][i] * (1.0 - &self.activations[layer][i]);
                // if deltas[layer][i] == 0.0 {
                //     println!("layer {}; i {}; tmp {}; activation {}", layer, i, tmp, self.activations[layer][i]);
                //     panic!();
                // }
            }

            Self::update_gradients(layer, &deltas, inputs, &mut gradients_biases, &mut gradients_weights, &self.activations);
        }

        // println!("weights {:?}", &self.weights[0][0][0..5]);
        // println!("gradients weight {:?}", &gradients_weights[0][0][0..5]);
        // println!("gradients biases {:?}", &gradients_biases[0][0..5]);
        // println!("deltas {} {:?}", deltas[1].iter().all(|&x| x == 0.0), &deltas[0][0..10]);
        // println!("deltas {:?}", deltas);

        // update weights and biases
        for layer in 0..gradients_biases.len() {
            for i in 0..gradients_biases[layer].len() {
                for j in 0..gradients_weights[layer][i].len() {
                    let prev_val = self.weights[layer][i][j];
                    self.weights[layer][i][j] -= learning_rate * gradients_weights[layer][i][j];
                    // if prev_val == self.weights[layer][i][j] {
                    //     println!("layer {}; i {}; j {}; weight {}", layer, i, j, gradients_weights[layer][i][j]);
                    //     panic!();
                    // }
                }
                self.biases[layer][i] -= learning_rate * gradients_biases[layer][i];
                // if self.biases[layer][i] == prev_val {
                //     println!("{deltas:?}");
                //     println!("{prev_val} {}", self.biases[layer][i]);
                //     println!("{}", gradients_biases == deltas);
                //     println!("layer {layer}; i {i}; gradient {}; {}", gradients_biases[layer][i], learning_rate * gradients_biases[layer][i]);
                //     panic!();
                // }
            }
        }
    }

    fn update_gradients(layer: usize, deltas: &Vec<Vec<f32>>, inputs: &[f32], gradients_biases: &mut Vec<Vec<f32>>, gradients_weights: &mut Vec<Vec<Vec<f32>>>, activations: &Vec<Vec<f32>>) {
        for i in 0..deltas[layer].len() {
            gradients_biases[layer][i] = deltas[layer][i];
            for j in 0..gradients_weights[layer][i].len() {
                let prev = if layer == 0 { inputs } else { &activations[layer-1] };
                gradients_weights[layer][i][j] = deltas[layer][i] * prev[j];
                // if gradients_weights[layer][i][j] == 0.0 {
                //     println!("layer {}; i {}; j {}; gradient {}; prev {:?}", layer, i, j, gradients_weights[layer][i][j], prev[j]);
                //     panic!();
                // }
            }
        }
    }

    pub fn serialize(&self) -> String {
        // todo serialize to binary instead of string
        let mut header: Vec<u32> = Vec::new();
        header.push(self.weights[0][0].len() as u32);
        for b in &self.biases {
            header.push(b.len() as u32);
        }

        let mut result: Vec<String> = Vec::new();
        for layer in 1..header.len() {
            result.push(format!("layer {}", layer));
            for current_layer_i in 0..header[layer] as usize {
                for prev_layer_i in 0..header[layer - 1] as usize {
                    result.push(self.weights[layer - 1][current_layer_i][prev_layer_i].to_string());
                }
                result.push(self.biases[layer - 1][current_layer_i].to_string());
            }
        }

        let header = header.iter().map(|x| x.to_string()).collect::<Vec<String>>().join(" ");
        let body = result.join("\n");
        header + "\n" + &body
    }

    pub fn deserialize(input: &str) -> Self {
        //todo handle errors, return Err
        let mut it = input.split("\n");
        // the first line with only numbers is considered the header, for example "2 3 2 1"
        let header: Vec<u32> = it.find_map(
            |s| {
                let split: Vec<&str> = s.split(' ').collect();
                if split.len() < 2 {
                    return None;
                }
                let values: Vec<u32> = split.iter().map_while(|&x| x.parse().ok()).collect();
                if values.len() == split.len() {
                    Some(values)
                }
                else {
                    None
                }
            }
        ).unwrap();
        let mut result = NeuralNetwork::new(&header);

        for layer in 1..header.len() {
            for current_layer_i in 0..header[layer] as usize {
                for prev_layer_i in 0..header[layer - 1] as usize {
                    let val: f32 = it.find_map(|x| x.parse().ok()).unwrap();
                    result.weights[layer - 1][current_layer_i][prev_layer_i] = val;
                }
                let val: f32 = it.find_map(|x| x.parse().ok()).unwrap();
                result.biases[layer - 1][current_layer_i] = val;
            }
        }

        result
    }
}

mod test {
    use std::cmp::PartialEq;
    use crate::neural_network::NeuralNetwork;

    fn get_network() -> NeuralNetwork {
        NeuralNetwork {
            weights: vec![
                vec![
                    vec![0.1, 0.2],
                    vec![0.2, 0.3],
                    vec![0.4, 0.5],
                ],
                vec![
                    vec![0.5, 0.6, 0.7],
                    vec![0.5, 0.1, 0.2],
                ],
                vec![
                    vec![0.3, 0.4],
                ]
            ],
            biases: vec![
                vec![-0.5, 0.3, 0.5],
                vec![0.2, -0.9],
                vec![0.6]
            ],
            pre_activations: vec![vec![0.0, 0.0, 0.0], vec![0.0, 0.0], vec![0.0]],
            activations: vec![vec![0.0, 0.0, 0.0], vec![0.0, 0.0], vec![0.0]],
        }
    }

    #[test]
    fn test_sigmoid_activation_function() {
        assert!((NeuralNetwork::activation(0.33) - 0.581759).abs() < 0.001);
        assert!((NeuralNetwork::activation(0.976) - 0.72631).abs() < 0.001);
        assert!((NeuralNetwork::activation(0.01) - 0.50249).abs() < 0.001);
        assert!((NeuralNetwork::activation(-0.75) - 0.32082).abs() < 0.001);
        //todo interesting, the precision is pretty low here, where is it lost?
        assert!((NeuralNetwork::activation(0.0) - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_process_network() {
        let mut network = get_network();

        /*
        f = lambda x: 1.0 / (1 + math.exp(-x))
        a11 = f(0.1 * 0.1 + 0.2 * 0.8 - 0.5)
        a12 = f(0.2 * 0.1 + 0.3 * 0.8 + 0.3)
        a13 = f(0.4 * 0.1 + 0.5 * 0.8 + 0.5)
        a21 = f(0.5 * a11 + 0.6 * a12 + 0.7 * a13 + 0.2)
        a22 = f(0.5 * a11 + 0.1 * a12 + 0.2 * a13 - 0.9)
        a3 = f(0.3 * a21 + 0.4 * a22 + 0.6)
        a3
        0.7287013674285573
         */
        let expected: f32 = 0.7287013674285573;
        let result1 = network.process(&[0.1, 0.8]);
        let result2 = network.process_mutable(&[0.1, 0.8]);

        assert_eq!(result1.len(), 1);
        assert_eq!(result2.len(), 1);
        assert_eq!(result1[0], expected);
        assert_eq!(result2[0], expected);
        // assert!((result[0] - expected).abs() < 0.001);
    }

    #[test]
    fn test_serialize() {
        let network = get_network();
        let result = network.serialize();
        let expected = "2 3 2 1\n\
        0.1\n0.2\n-0.5\n0.2\n0.3\n0.3\n0.4\n0.5\n0.5\n0.5\n0.6\n0.7\n0.2\n0.5\n0.1\n0.2\n-0.9\n0.3\n0.4\n0.6";

        assert_eq!(result, expected);
    }

    #[test]
    fn test_serialize_deserialize() {
        let network = NeuralNetwork::new(&[5, 7, 10, 10]);

        let serialized = network.serialize();
        let deserialized = NeuralNetwork::deserialize(&serialized);

        assert_eq!(deserialized, network);
    }

    #[test]
    fn test_deserialize_ignores_non_numeric_values() {
        let serialized = "\nlayers\n1 1 1\n\nlayer 1\n0.99\n0.33\n\noutput layer\n0.13\n3.14\n\nthis should be ignored\n";

        let deserialized = NeuralNetwork::deserialize(serialized);
        let expected = NeuralNetwork {
            weights: vec![vec![vec![0.99]], vec![vec![0.13]]],
            biases: vec![vec![0.33], vec![3.14]],
            pre_activations: vec![],
            activations: vec![],
        };

        assert_eq!(deserialized, expected);
    }

    //todo add at least invoking of "training_step", so it doesn't crash on different values
    //todo add a test for learning_step, calculate values manually
    #[test]
    fn test_learning_step() {
        // AI generated test vectors for me
        let mut network = NeuralNetwork::new(&[2, 2, 2]);
        network.weights = vec![
            vec![
                vec![0.15, 0.20],
                vec![0.25, 0.30],
            ],
            vec![
                vec![0.40, 0.45],
                vec![0.50, 0.55],
            ],
        ];

        network.biases = vec![
            vec![0.35, 0.35],
            vec![0.60, 0.60],
        ];

        let input = vec![0.05, 0.10];
        let target = vec![0.01, 0.99];

        let result = network.process_mutable(&input);

        println!("RESULT: {:?}", result);

        network.training_step(&input, &target, 0.5);

        println!("new weights: {:?}", network.weights);
        println!("new biases: {:?}", network.biases);
    }

    #[test]
    fn test_learning_step2() {
        let mut network = NeuralNetwork::new(&[2, 2, 1]);
        network.weights = vec![
            vec![
                vec![0.5, 0.3],
                vec![-0.2, 0.7],
            ],
            vec![
                vec![0.4, -0.6],
            ],
        ];
        network.biases = vec![
            vec![0.1, -0.3],
            vec![0.2],
        ];

        let input = vec![0.8, 0.4];
        let target = vec![0.9];

        println!("result {:?}", network.process_mutable(&input));
        network.training_step(&input, &target, 0.1);

        println!("new weights: {:?}", network.weights);
        println!("new biases: {:?}", network.biases);
    }
}
