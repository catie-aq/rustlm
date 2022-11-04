use super::SearchError;
use std::collections::BTreeMap;
use crate::tree::{SuffixTree, ROOT_NODE};
use crate::language_model::{LanguageModel, LMType};
use crate::audio_model::AudioModel;
use crate::token_to_string::token_to_string;
use std::cell::RefCell;
use std::cmp::Ordering;

/// A node in the labelling tree to build from.
#[derive(Clone, Debug)]
struct SearchPoint {
    /// The prefix of the search point
    node: i32,
    /// The cumulative probability of the labelling so far for paths without any ending blank
    /// labels.
    label_prob: f32,
    /// A state space index
    state_index: usize,
    /// The last decoded label
    last_label: usize,
    /// A terminaison boolean
    is_end_point: bool
}

impl SearchPoint {
    /// The total probability of the labelling so far.
    ///
    /// This sums the probabilities of the paths with and without leading blank labels.
    fn probability(&self) -> f32 {
        self.label_prob
    }
}

pub fn beam_search_rnnt(
    audio_model: RefCell<&mut dyn AudioModel>,
    alphabet: &[String],
    alphabet_type: String,
    beam_width: usize,
    cutoff_prob: f32,
    blank_id: usize,
    language_model: RefCell<Option<&mut dyn LanguageModel>>
) -> Result<(Vec<String>, Vec<Vec<usize>>, Vec<f32>), SearchError> {

    let vocabulary_length = alphabet.len();

    // prefix probabilities as hashmaps
    let mut prefix_tree_prev = BTreeMap::<i32, SearchPoint>::new();
    let mut prefix_tree = BTreeMap::<i32, SearchPoint>::new();

    // prefix map
    let mut suffix_tree = SuffixTree::new(vocabulary_length);

    let empty_search_point = SearchPoint {
        node: ROOT_NODE,
        label_prob: 0.0,
        state_index: 0,
        last_label: blank_id,
        is_end_point: false
    };

    // initial state
    prefix_tree_prev.insert(0, empty_search_point);

    let mut beam = prefix_tree_prev.values().cloned().collect::<Vec<_>>();

    // boolean controlling the outside loop
    let mut has_ended = false;
    let mut idx = 0;

    // borrow the audio model
    let mut model = &mut *audio_model.borrow_mut();

    while has_ended != true {
        prefix_tree.clear();

        // iterate for every prefix
        for &SearchPoint {
            node,
            label_prob,
            state_index,
            last_label,
            is_end_point
        } in &beam
        {

            if is_end_point == true {
                prefix_tree.insert(node, SearchPoint {
                    node,
                    label_prob: label_prob,
                    state_index: state_index,
                    last_label: last_label,
                    is_end_point: is_end_point
                });
                continue;
            }

            let previous_label = suffix_tree.label(node);

            let (mut prob, new_state_index)  = model.get_next_tokens(Some(last_label), state_index).unwrap();

            for (label, &pr) in prob.iter().enumerate() {

                if (pr < cutoff_prob) || pr.is_nan() || (pr.log(2.0).is_infinite()) {
                    continue;
                }

                let end = model.is_terminal_state(Some(label), new_state_index);

                // empty token
                if label == blank_id {

                    if let Some(x) = prefix_tree.get_mut(&node) {
                        (*x).last_label = label;
                        (*x).state_index = new_state_index;
                        (*x).label_prob += pr.log(2.0);
                    } else {
                        prefix_tree.insert(node, SearchPoint {
                            node,
                            label_prob: label_prob + pr.log(2.0),
                            state_index: new_state_index,
                            last_label: label,
                            is_end_point: end
                        });
                    }
                } else {
                    // if a new character

                    let new_node_idx = suffix_tree
                        .get_child(node, label)
                        .unwrap_or_else(|| suffix_tree.add_node(node, label, idx));

                    if let Some(x) = prefix_tree.get_mut(&new_node_idx) {
                        (*x).label_prob += pr.log(2.0);
                        (*x).last_label = label;
                        (*x).state_index = new_state_index;
                    } else {
                        prefix_tree.insert(new_node_idx, SearchPoint {
                            node: new_node_idx,
                            label_prob: label_prob + pr.log(2.0),
                            state_index: new_state_index,
                            last_label: label,
                            is_end_point: end
                        });
                    }
                }
            }
        } // end SearchPoint

        beam = prefix_tree.values().cloned().collect::<Vec<_>>();

        let mut has_nans = false;
        beam.sort_unstable_by(|a, b| {
            (b.probability())
                .partial_cmp(&(a.probability()))
                .unwrap_or_else(|| {
                    has_nans = true;
                    std::cmp::Ordering::Equal // don't really care
                })
        });
        if has_nans {
            return Err(SearchError::IncomparableValues);
        }

        beam.truncate(beam_width);
        if beam.is_empty() {
            // we've run out of beam (probably the threshold is too high)
            return Err(SearchError::RanOutOfBeam);
        }

        std::mem::swap(&mut prefix_tree_prev, &mut prefix_tree);

        // Control end condition
        has_ended = true;
        for &SearchPoint {
            node,
            label_prob,
            state_index,
            last_label,
            is_end_point
        } in &beam
        {
            if is_end_point == false {
                has_ended = false;
            }
        }

        idx = idx + 1;
    }

    // Extract sentences
    let mut vec_tokens = Vec::<Vec<String>>::new();
    let mut vec_str = Vec::<String>::new();
    let mut vec_path = Vec::<Vec<usize>>::new();
    let mut vec_prob = Vec::<f32>::new();

    for &SearchPoint {
            node,
            label_prob,
            state_index,
            last_label,
            is_end_point
        } in &beam
    {

        let mut sequence = Vec::<String>::new();
        let mut path = Vec::<usize>::new();

        if node != ROOT_NODE {
            for (label, &time) in suffix_tree.iter_from(node) {
                path.push(time);
                sequence.push(alphabet[label].to_string());
            }
        }

        sequence.reverse();
        vec_tokens.push(sequence.to_owned());
        vec_str.push(token_to_string(sequence, &alphabet_type).unwrap());
        vec_path.push(path);
        vec_prob.push(label_prob);
    }

    // Rescore the probabilities from the language model if neccessary
    if let Some(ref mut model) = *language_model.borrow_mut() {
        // if model is of type "final rescoring"
        if model.lm_type() == LMType::FinalRescoring {
            let model_logprob = model.get_sentence_likelihood(vec_str.to_owned()).unwrap();
            let model_prob: Vec<f32> = model_logprob.iter().map(|x| x.exp()).collect();

            let permutation = permutation::sort_by(&model_prob[..], |a, b| b.partial_cmp(a).unwrap_or(Ordering::Equal));
            vec_prob = permutation.apply_slice(&model_prob[..]);
            vec_str = permutation.apply_slice(&vec_str[..]);
            vec_path = permutation.apply_slice(&vec_path[..]);
        }
    }

    Ok((vec_str, vec_path, vec_prob))
}
