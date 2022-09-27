use std::f32::consts::LOG2_E;

const INVERSE_LOG2_E: f32 = 1. / LOG2_E;

const COEFF_0: f32 = 1.0;
const COEFF_1: f32 = 4.831794110;
const COEFF_2: f32 = 0.143440676;
const COEFF_3: f32 = 0.019890581;
const COEFF_4: f32 = 0.006935931;
const ONEBYLOG2: f32 = 1.442695041;
const OFFSET_F64: i64 = 1023;
const FRACTION_F64: u32 = 52;
const MIN_VAL: f32 = -500.0;

/// Fast approximation of exp() as shown by Kopcynski 2017:
/// https://eldorado.tu-dortmund.de/bitstream/2003/36203/1/Dissertation_Kopczynski.pdf
pub fn fast_exp(input: f32) -> f32 {
    if input > MIN_VAL {
        let mut x = ONEBYLOG2 * input;

        #[repr(C)]
        union F1 {
            i: i64,
            f: f32,
        }
        let mut f1 = F1 { i: x as i64 };

        x -= unsafe { f1.i } as f32;
        let mut f2 = x;
        let mut x_tmp = x;

        unsafe {
            f1.i += OFFSET_F64;
            f1.i <<= FRACTION_F64;
        }

        f2 *= COEFF_4;
        x_tmp += COEFF_1;
        f2 += COEFF_3;
        x_tmp *= x;
        f2 *= x;
        f2 += COEFF_2;
        f2 *= x_tmp;
        f2 += COEFF_0;

        unsafe { f1.f * f2 }
    } else {
        0.0
    }
}

pub fn logsumexp_2(x: f32, y:f32) -> f32 {
    if x < y {
        y + fast_log((x - y).exp() + 1.0)
    } else {
        x + fast_log((y - x).exp() + 1.0)
    }
}

pub fn softmax(vec: Vec<f32>) -> Vec<f32> {
    let exp_vec: Vec<f32> = vec.iter().map(|x| (*x as f32).exp()).collect();
    let sum: f32 = exp_vec.iter().sum();

    exp_vec.iter().map(|x| x / sum).collect::<Vec<f32>>()
}

pub fn normalize(vec: Vec<f32>) -> Vec<f32> {
    let max: f32 = *vec.iter().max_by(|x, y| x.partial_cmp(y).unwrap()).unwrap();
    vec.iter().map(|x| x - max).collect::<Vec<f32>>()
}

#[inline]
pub fn logsumexp(xs: &[f32], max: f32) -> f32 {
    fast_log(xs.iter().fold(0., |acc, &x| acc + fast_exp(x - max))) + max
}

#[inline]
pub fn fast_log(x: f32) -> f32 {
  x.log2() * INVERSE_LOG2_E
}
