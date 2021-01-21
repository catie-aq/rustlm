use ndarray::Array2;
use std::marker::{Send, Sync};
use std::error::Error;

#[derive(PartialEq, Eq)]
pub enum LMType {
    FinalRescoring, // final rescoring of the last truncated beam (suited for large GPT style LM)
    Rescoring, // rescoring at each beam before truncation (for smaller language models) - NOT SUPPORTED
    ShallowFusion, // merge probabilities of LM with input probabilities (suited for character level LM) - NOT SUPPORTED
    NgramForecast, // n-gram based for summing all possible word given a prefix - NOT SUPPORTED
}

pub trait LanguageModel {
    fn get_sentence_likelihood(&mut self, input: Vec<String>) -> Result<Vec<f32>, Box<dyn Error + Send + Sync>>;
    fn get_next_letter(&mut self, input: Vec<String>, vocab: &[String], blank_id: usize, space_id: usize) -> Result<Array2<f32>, Box<dyn Error + Send + Sync>>;
    fn lm_type(&self) -> LMType;
}
