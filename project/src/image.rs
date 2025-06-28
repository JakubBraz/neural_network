use image::{GenericImageView, ImageBuffer, ImageReader, Rgba, RgbaImage};

pub const WIDTH: usize = 28;
pub const HEIGHT: usize = 28;

fn result_array(digit: u8) -> [f64; 10] {
    match digit {
        0 => [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        1 => [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        2 => [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        3 => [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        4 => [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        5 => [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        6 => [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        7 => [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        8 => [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        9 => [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        _ => [0.0; 10]
    }
}

pub fn get_training_data(catalog: &str, digit: u8, index: u16) -> ([f64; WIDTH * HEIGHT], [f64; 10]) {
    let path = format!("{catalog}/{digit}/{digit}/{index}.png");
    let img = ImageReader::open(path).unwrap().decode().unwrap();
    let zero_pixel = Rgba::from([0, 0, 0, 0]);
    let mut result = [0.0; WIDTH * HEIGHT];
    for row in 0..WIDTH {
        for col in 0..HEIGHT {
            //for now, I treat every pixel binary, if there is any shade of gray the pixel is considered "on"
            let p = img.get_pixel(row as u32, col as u32);
            // if p != zero_pixel {
            //     result[col * HEIGHT + row] = 1.0;
            // }
            result[col * HEIGHT + row] = p[3] as f64 / 255.0;
        }
    }
    (result, result_array(digit))
}

pub fn save_training_data(catalog: &str, digit: u8, image_data: &[f64], index: u32) {
    let path = format!("{catalog}/{digit}/{digit}/{index}.png");
    let mut img = RgbaImage::new(WIDTH as u32, HEIGHT as u32);
    for col in 0..HEIGHT {
        for row in 0..WIDTH {
            let pixel = Rgba::from([0, 0, 0, (image_data[col * HEIGHT + row] * 255.0) as u8]);
            img.put_pixel(row as u32, col as u32, pixel);
        }
    }
    img.save(path).unwrap();
}

pub fn read() {
    let path = "C:\\Users\\jakubbraz\\Downloads\\archive\\dataset\\3\\3\\999.png";
    let zero_pixel = Rgba::from([0, 0, 0, 0]);
    let reader = ImageReader::open(path).unwrap();
    let img = reader.decode().unwrap();
    println!("{:?}", img.dimensions());
    let p = img.get_pixel(0, 0);
    println!("{:?}", p);
    for row in 0..img.dimensions().1 {
        let mut buf: Vec<char> = Vec::new();
        for col in 0..img.dimensions().0 {
            let p = img.get_pixel(col, row);
            if p != zero_pixel {
                // println!("{p:?}");
                buf.push('X');
            } else {
                buf.push(' ');
            }

        }
        println!("{:?}", buf);
    }
}
