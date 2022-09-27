use ndarray::{Array1, Array2};
use std::marker::{Send, Sync};
use std::error::Error;

#[derive(PartialEq, Eq)]
pub enum DecoderType {
    NemoRNNT,
}

pub trait AudioModel {
    fn get_next_tokens(&mut self, last_label: Option<usize>, state_index: usize) -> Result<(Array1<f32>, usize), Box<dyn Error + Send + Sync>>;
    fn is_terminal_state(&self, last_label: Option<usize>, state_index: usize) -> bool;
    fn get_type(&self) -> DecoderType;
}
