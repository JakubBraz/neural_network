pub fn product(arr: &Vec<Vec<f32>>, vec: &[f32], result: &mut Vec<f32>) {
    //todo improve it, make it parallel or use a library for matrix product
    for i in 0..arr.len() {
        let mut tmp = 0.0;
        for j in 0..arr[i].len() {
            tmp += arr[i][j] * vec[j];
        }
        result[i] = tmp;
    }
}

pub fn sum(vec1_result: &mut Vec<f32>, vec2: &Vec<f32>) {
    //todo improve it, make it parallel or use a library for matrix product
    for i in 0..vec1_result.len() {
        vec1_result[i] = vec1_result[i] + vec2[i];
    }
}
