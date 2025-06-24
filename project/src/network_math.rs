pub fn product(arr: &Vec<Vec<f32>>, vec: &Vec<f32>) -> Vec<f32> {
    //todo improve it, make it parallel or use a library for matrix product
    let mut result = Vec::new();
    for i in 0..arr.len() {
        let mut tmp = 0.0;
        for j in 0..arr[i].len() {
            tmp += arr[i][j] * vec[j];
        }
        result.push(tmp);
    }
    result
}

pub fn sum(vec1: &Vec<f32>, vec2: &Vec<f32>) -> Vec<f32> {
    //todo improve it, make it parallel or use a library for matrix product
    vec1.iter().enumerate().map(|(i, x)| vec2[i] + x).collect()
}
