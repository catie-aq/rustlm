#![feature(static_nobundle)]
#![feature(test)] // benchmarking

#[macro_use(s)]
#[cfg_attr(test, macro_use(array))]
extern crate ndarray;
extern crate test; // benchmarking

use numpy::PyArray2;

use pyo3::{PyResult, PyErr};
use pyo3::exceptions::{PyValueError, PyRuntimeError};

use pyo3::prelude::*;
use pyo3::types::{PySequence, PyUnicode};
use std::fmt;
use std::cell::RefCell;

mod bsearch;
mod tree;
mod vec2d;
mod language_model;
mod dico_lm;
mod gpt2_lm;
mod fast_math;

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
struct GPT2BeamSearch {
    lm_model: gpt2_lm::GPT2LanguageModel,
}

#[pymethods]
impl GPT2BeamSearch {

    /// Build a GPT2BeamSearch
    ///
    /// Args:
    ///     model_path (String): Path to the onnx model.
    ///     vocab_path (String): Path to the tokenizer vocab file.
    ///     merge_path (String): Path to the tokenizer merge file.
    ///     space_id (u32): The index of the space character
    ///     pad_token (String): The token for padding
    #[new]
    fn new(model_path: &PyUnicode, vocab_path: &PyUnicode, merge_path: &PyUnicode, space_id: u32, pad_token: &PyUnicode) -> Self {

        let model_path_str = model_path.to_string();
        let vocab_path_str = vocab_path.to_string();
        let merge_path_str = merge_path.to_string();
        let pad_token_str = pad_token.to_string();

        GPT2BeamSearch {
            lm_model: gpt2_lm::GPT2LanguageModel::load_from_files(&model_path_str, &vocab_path_str, &merge_path_str, space_id, &pad_token_str).unwrap(),
        }
    }

    /// Perform a CTC beam search decode on an RNN output using a GPT2 based LM.
    ///
    /// This function does a beam search variant of the prefix search decoding mentioned (and described
    /// in fairly vague terms) in the original CTC paper (Graves et al, 2006, section 3.2).
    ///
    /// The paper mentioned above provides recursive equations that give an efficient way to find the
    /// probability for a specific labelling. A tree of possible labelling suffixes, together with
    /// their probabilities, can be built up by starting at one end and trying every possible label at
    /// each stage. The "beam" part of the search is how we keep the search space managable - at each
    /// step, we ignore all but the most-probable tree leaves (like searching with a torch beam). This
    /// means we may not actually find the most likely labelling, but it often works very well.
    ///
    /// See the module-level documentation for general requirements on `network_output` and `alphabet`.
    ///
    /// Args:
    ///     network_output (numpy.ndarray): The 2D array output of the neural network.
    ///     alphabet (sequence): The labels (including the blank label, which must be first) in the
    ///         order given on the inner axis of `network_output`.
    ///     beam_width (usize): How many search points should be kept at each step. Higher numbers are
    ///         less likely to discard the true labelling, but also make it slower and more memory
    ///         intensive. Must be at least 1.
    ///     cutoff_prob (float): Ignore any entries in `network_output` below this value. Must
    ///         be at least 0.0, and less than ``1/len(alphabet)``.
    ///     alpha (float): Language model weight
    ///     beta (float): Word insertion weight
    ///     blank_id (usize): The index of the blank (epsilon) character in the array
    ///     space_id (usize): The index of the space character in the array
    ///
    /// Returns:
    ///     tuple of (str, numpy.ndarray): The decoded sequences and an array of the final
    ///         timepoints of each label (as indices into the outer axis of `network_output`) and
    ///         an array of probabilities for each path.
    ///
    /// Raises:
    ///     ValueError: The constraints on the arguments have not been met.
    pub fn beam_search(
        &mut self,
        network_output: &PyArray2<f32>,
        alphabet: &PySequence,
        beam_width: usize,
        cutoff_prob: f32,
        alpha: f32,
        beta: f32,
        blank_id: usize,
        space_id: usize
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
            bsearch::beam_search(
                unsafe { &network_output.as_array() }, // PyReadonlyArray2 missing trait
                &alphabet,
                beam_width,
                cutoff_prob,
                alpha,
                beta,
                blank_id,
                space_id,
                RefCell::new(Some(&mut self.lm_model))
            )
            .map_err(|e| PyRuntimeError::new_err(format!("{}", e)))
        }
    }
}

#[pyclass(unsendable)]
struct NoLMBeamSearch {
    lm_model: Option<u32>,
}

#[pymethods]
impl NoLMBeamSearch {

    #[new]
    fn new() -> Self {
        NoLMBeamSearch {
            lm_model: None,
        }
    }

    pub fn beam_search(
        &mut self,
        network_output: &PyArray2<f32>,
        alphabet: &PySequence,
        beam_width: usize,
        cutoff_prob: f32,
        alpha: f32,
        beta: f32,
        blank_id: usize,
        space_id: usize
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
            bsearch::beam_search(
                unsafe { &network_output.as_array() }, // PyReadonlyArray2 missing trait
                &alphabet,
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
struct DicoBeamSearch {
    lm_model: dico_lm::DicoLanguageModel,
}

#[pymethods]
impl DicoBeamSearch {

    #[new]
    fn new(dico_path: &PyUnicode) -> Self {
        let dico_path_str = dico_path.to_string();

        DicoBeamSearch {
            lm_model: dico_lm::DicoLanguageModel::load_from_files(&dico_path_str).unwrap(),
        }
    }

    pub fn beam_search(
        &mut self,
        network_output: &PyArray2<f32>,
        alphabet: &PySequence,
        beam_width: usize,
        cutoff_prob: f32,
        alpha: f32,
        beta: f32,
        blank_id: usize,
        space_id: usize
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
            bsearch::beam_search(
                unsafe { &network_output.as_array() }, // PyReadonlyArray2 missing trait
                &alphabet,
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
    m.add_class::<GPT2BeamSearch>()?;
    m.add_class::<NoLMBeamSearch>()?;
    m.add_class::<DicoBeamSearch>()?;
    //m.add_wrapped(wrap_pyfunction!(beam_search))?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
