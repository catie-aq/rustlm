extern crate ndarray;

use numpy::{PyArray2, PyArray3};

use pyo3::{PyResult, PyErr};
use pyo3::exceptions::{PyValueError, PyRuntimeError};

use pyo3::prelude::*;
use pyo3::types::{PySequence, PyUnicode};
use std::fmt;
use std::cell::RefCell;

//mod bsearch;
mod bsearch_ctc;
mod bsearch_rnnt;
mod tree;
mod vec2d;
mod language_model;
mod dico_lm;
mod fast_math;
mod audio_model;
mod token_to_string;
mod final_gpt_lm;
mod nemo_rnnt_audio_model;

#[derive(Clone, Copy, Debug)]
pub enum SearchError {
    RanOutOfBeam,
    IncomparableValues,
    InvalidEnvelope,
}

impl fmt::Display for SearchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SearchError::RanOutOfBeam => {
                write!(f, "Ran out of search space (beam_cut_threshold too high)")
            }
            SearchError::IncomparableValues => {
                write!(f, "Failed to compare values (NaNs in input?)")
            }
            // TODO: document envelope constraints
            SearchError::InvalidEnvelope => write!(f, "Invalid envelope values"),
        }
    }
}

impl std::error::Error for SearchError {}

fn seq_to_vec(seq: &PySequence) -> PyResult<Vec<String>> {
    Ok(seq.tuple()?.iter().map(|x| x.to_string()).collect())
}

#[pyclass(unsendable)]
struct BeamSearchCTCNoLM {
    lm_model: Option<u32>,
}

#[pymethods]
impl BeamSearchCTCNoLM {

    #[new]
    fn new() -> Self {
        BeamSearchCTCNoLM {
            lm_model: None,
        }
    }

    pub fn beam_search(
        &mut self,
        network_output: &PyArray2<f32>,
        alphabet: &PySequence,
        alphabet_type: &PyUnicode,
        beam_width: usize,
        cutoff_prob: f32,
        alpha: f32,
        beta: f32,
        blank_id: usize,
        space_id: usize,
        sep_id: usize
    ) -> PyResult<(Vec<String>, Vec<Vec<usize>>, Vec<f32>)> {

        let alphabet = seq_to_vec(alphabet)?;

        if (alphabet.len() + 1) != network_output.shape()[1] {
            let err: PyErr = PyValueError::new_err(format!(
                "alphabet size {} does not match probability matrix inner dimension {}",
                alphabet.len(),
                network_output.shape()[1]
            ));
            Err(err)
        } else if beam_width == 0 {
            let err: PyErr = PyValueError::new_err("Beam_width cannot be 0");
            Err(err)
        } else if cutoff_prob < -0.0 {
            let err: PyErr = PyValueError::new_err("Cutoff_prob must be at least 0.0");
            Err(err)
        } else {
            bsearch_ctc::beam_search_ctc(
                unsafe { &network_output.as_array() }, // PyReadonlyArray2 missing trait
                &alphabet,
                alphabet_type.to_string(),
                beam_width,
                cutoff_prob,
                alpha,
                beta,
                blank_id,
                space_id,
                RefCell::new(None)
            )
            .map_err(|e| PyRuntimeError::new_err(format!("{}", e)))
        }
    }
}

#[pyclass(unsendable)]
struct BeamSearchRNNTNoLM {
    server_address: String,
    model_name: String,
    num_rnn_layers: usize,
    hidden_size: usize
}

#[pymethods]
impl BeamSearchRNNTNoLM {

    #[new]
    fn new(server_address: &PyUnicode, model_name: &PyUnicode, num_rnn_layers: usize, hidden_size: usize) -> Self {
        BeamSearchRNNTNoLM {
            server_address: server_address.to_string(),
            model_name: model_name.to_string(),
            num_rnn_layers: num_rnn_layers,
            hidden_size: hidden_size
        }
    }

    pub fn beam_search(
        &mut self,
        encoder_output: &PyArray3<f32>,
        alphabet: &PySequence,
        alphabet_type: &PyUnicode,
        beam_width: usize,
        cutoff_prob: f32,
        blank_id: usize,
        sep_id: usize,
        cls_id: usize
    ) -> PyResult<(Vec<String>, Vec<Vec<usize>>, Vec<f32>)> {

        let alphabet = seq_to_vec(alphabet)?;
        let mut audio_model = nemo_rnnt_audio_model::NemoRNNTAudioModel::load(&self.server_address, &self.model_name, unsafe { &encoder_output.as_array() },
                                    self.num_rnn_layers, self.hidden_size, encoder_output.shape()[1], alphabet.len(), blank_id, sep_id);

        if beam_width == 0 {
            let err: PyErr = PyValueError::new_err("Beam_width cannot be 0");
            Err(err)
        } else if cutoff_prob < -0.0 {
            let err: PyErr = PyValueError::new_err("Cutoff_prob must be at least 0.0");
            Err(err)
        } else {
            bsearch_rnnt::beam_search_rnnt(
                RefCell::new(&mut audio_model), // PyReadonlyArray2 missing trait
                &alphabet,
                alphabet_type.to_string(),
                beam_width,
                cutoff_prob,
                blank_id,
                RefCell::new(None)
            )
            .map_err(|e| PyRuntimeError::new_err(format!("{}", e)))
        }
    }
}

#[pyclass(unsendable)]
struct BeamSearchCTCDico {
    lm_model: dico_lm::DicoLanguageModel,
}

#[pymethods]
impl BeamSearchCTCDico {

    #[new]
    fn new(dico_path: &PyUnicode) -> Self {
        let dico_path_str = dico_path.to_string();

        BeamSearchCTCDico {
            lm_model: dico_lm::DicoLanguageModel::load_from_files(&dico_path_str).unwrap(),
        }
    }

    pub fn beam_search(
        &mut self,
        network_output: &PyArray2<f32>,
        alphabet: &PySequence,
        alphabet_type: &PyUnicode,
        beam_width: usize,
        cutoff_prob: f32,
        alpha: f32,
        beta: f32,
        blank_id: usize,
        space_id: usize,
        sep_id: usize
    ) -> PyResult<(Vec<String>, Vec<Vec<usize>>, Vec<f32>)> {

        let alphabet = seq_to_vec(alphabet)?;

        if (alphabet.len() + 1) != network_output.shape()[1] {
            let err: PyErr = PyValueError::new_err(format!(
                "alphabet size {} does not match probability matrix inner dimension {}",
                alphabet.len(),
                network_output.shape()[1]
            ));
            Err(err)
        } else if beam_width == 0 {
            let err: PyErr = PyValueError::new_err("Beam_width cannot be 0");
            Err(err)
        } else if cutoff_prob < -0.0 {
            let err: PyErr = PyValueError::new_err("Cutoff_prob must be at least 0.0");
            Err(err)
        } else {
            bsearch_ctc::beam_search_ctc(
                unsafe { &network_output.as_array() }, // PyReadonlyArray2 missing trait
                &alphabet,
                alphabet_type.to_string(),
                beam_width,
                cutoff_prob,
                alpha,
                beta,
                blank_id,
                space_id,
                RefCell::new(Some(&mut self.lm_model)),
            )
            .map_err(|e| PyRuntimeError::new_err(format!("{}", e)))
        }
    }
}

#[pyclass(unsendable)]
struct BeamSearchCTCGPTRescoring {
    lm_model: final_gpt_lm::GPTFinalLanguageModel,
}

#[pymethods]
impl BeamSearchCTCGPTRescoring {

    #[new]
    fn new(server_address: &PyUnicode, model_name: &PyUnicode, vocab_path: &PyUnicode, merges_path: &PyUnicode, max_batch_size:usize, max_length:usize, num_tokens:usize) -> Self {
        let server_address_str = server_address.to_string();
        let model_name_str = model_name.to_string();
        let vocab_path_str = vocab_path.to_string();
        let merges_path_str = merges_path.to_string();

        BeamSearchCTCGPTRescoring {
            lm_model: final_gpt_lm::GPTFinalLanguageModel::load_from_files(&server_address_str, &model_name_str, &vocab_path_str, &merges_path_str, max_batch_size, max_length, num_tokens).unwrap(),
        }
    }

    pub fn beam_search(
        &mut self,
        network_output: &PyArray2<f32>,
        alphabet: &PySequence,
        alphabet_type: &PyUnicode,
        beam_width: usize,
        cutoff_prob: f32,
        alpha: f32,
        beta: f32,
        blank_id: usize,
        space_id: usize,
        sep_id: usize
    ) -> PyResult<(Vec<String>, Vec<Vec<usize>>, Vec<f32>)> {

        let alphabet = seq_to_vec(alphabet)?;

        if (alphabet.len() + 1) != network_output.shape()[1] {
            let err: PyErr = PyValueError::new_err(format!(
                "alphabet size {} does not match probability matrix inner dimension {}",
                alphabet.len(),
                network_output.shape()[1]
            ));
            Err(err)
        } else if beam_width == 0 {
            let err: PyErr = PyValueError::new_err("Beam_width cannot be 0");
            Err(err)
        } else if cutoff_prob < -0.0 {
            let err: PyErr = PyValueError::new_err("Cutoff_prob must be at least 0.0");
            Err(err)
        } else {
            bsearch_ctc::beam_search_ctc(
                unsafe { &network_output.as_array() }, // PyReadonlyArray2 missing trait
                &alphabet,
                alphabet_type.to_string(),
                beam_width,
                cutoff_prob,
                alpha,
                beta,
                blank_id,
                space_id,
                RefCell::new(Some(&mut self.lm_model)),
            )
            .map_err(|e| PyRuntimeError::new_err(format!("{}", e)))
        }
    }
}

#[pymodule]
fn rustlm(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<BeamSearchCTCNoLM>()?;
    m.add_class::<BeamSearchCTCDico>()?;
    m.add_class::<BeamSearchCTCGPTRescoring>()?;
    m.add_class::<BeamSearchRNNTNoLM>()?;
    //m.add_wrapped(wrap_pyfunction!(beam_search))?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
