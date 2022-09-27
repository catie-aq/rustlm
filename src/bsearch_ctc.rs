use super::SearchError;
use ndarray::{ArrayBase, Data, Ix2};
use std::collections::BTreeMap;
use crate::tree::{SuffixTree, ROOT_NODE};
use crate::language_model::{LanguageModel, LMType};
use crate::fast_math::{fast_exp, logsumexp, fast_log, logsumexp_2};
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
    /// The cumulative probability of the labelling so far for paths with one or more ending
    /// blank labels.
    blank_prob: f32,
    /// A space counter for weighting prefix word length
    space_counter: f32
}

impl SearchPoint {
    /// The total probability of the labelling so far.
    ///
    /// This sums the probabilities of the paths with and without leading blank labels.
    fn probability(&self, beta: f32) -> f32 {
        (self.label_prob + self.blank_prob) * self.space_counter.powf(beta)
    }
}

pub fn beam_search_ctc<D: Data<Elem = f32>>(
    network_output: &ArrayBase<D, Ix2>,
    alphabet: &[String],
    alphabet_type: String,
    beam_width: usize,
    cutoff_prob: f32,
    alpha: f32,
    beta: f32,
    blank_id: usize,
    space_id: usize,
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
        blank_prob: 1.0,
        space_counter: 0.0,
    };

    // initial state
    prefix_tree_prev.insert(0, empty_search_point);

    let mut beam = prefix_tree_prev.values().cloned().collect::<Vec<_>>();

    for (idx, pr) in network_output.outer_iter().enumerate() {
        prefix_tree.clear();

        // iterate for every prefix
        for &SearchPoint {
            node,
            label_prob,
            blank_prob,
            space_counter,
        } in &beam
        {

            let last_label = suffix_tree.label(node);
            let mut pr_copy = pr.to_owned();

            // Shallow fusion language model
            if let Some(ref mut model) = *language_model.borrow_mut() {
                // if model is of type "final rescoring"
                if model.lm_type() == LMType::ShallowFusion {
                    let mut sequence = String::new();
                    if node != ROOT_NODE {
                        for (label, &time) in suffix_tree.iter_from(node) {
                            sequence.push_str(&alphabet[label]);
                        }
                    }

                    let sequence_string = sequence.chars().rev().collect::<String>();
                    let lm_prob = model.get_next_letter(vec![sequence_string], alphabet, blank_id, space_id).unwrap();
                    pr_copy = pr_copy * lm_prob.row(0);
                    let sum_probs = pr_copy.sum();
                    pr_copy = pr_copy / sum_probs; // renormalize probabilities
                }
            }


            for (label, &pr) in pr_copy.iter().enumerate() {
                if pr < cutoff_prob {
                    continue;
                }

                // empty token
                if label == blank_id {

                    if let Some(x) = prefix_tree.get_mut(&node) {
                        (*x).blank_prob += (label_prob + blank_prob) * pr;
                    } else {
                        prefix_tree.insert(node, SearchPoint {
                            node,
                            label_prob: 0.0,
                            blank_prob: (label_prob + blank_prob) * pr,
                            space_counter: space_counter,
                        });
                    }
                } else {

                    // regular character

                    // if repetition of the last character
                    if Some(label) == last_label {
                        if let Some(x) = prefix_tree.get_mut(&node) {
                            (*x).label_prob += label_prob * pr;
                        } else {
                            prefix_tree.insert(node, SearchPoint {
                                node,
                                label_prob: label_prob * pr,
                                blank_prob: 0.0,
                                space_counter: space_counter,
                            });
                        }

                        let new_node_idx = suffix_tree.get_child(node, label).or_else(|| {
                            if blank_prob > 0.0 {
                                Some(suffix_tree.add_node(node, label, idx))
                            } else {
                                None
                            }
                        });

                        if let Some(idx) = new_node_idx {

                            if let Some(x) = prefix_tree.get_mut(&idx) {
                                (*x).label_prob += blank_prob * pr;
                            } else {
                                prefix_tree.insert(idx, SearchPoint {
                                    node: idx,
                                    label_prob: blank_prob * pr,
                                    blank_prob: 0.0,
                                    space_counter: space_counter,
                                });
                            }
                        }
                    } else if label == space_id {
                        // A space, increment space counter

                        let new_node_idx = suffix_tree
                            .get_child(node, label)
                            .unwrap_or_else(|| suffix_tree.add_node(node, label, idx));

                        let mut lm_prob = 1.0; // integrate language model probability here after spaces

                        if let Some(x) = prefix_tree.get_mut(&new_node_idx) {
                            (*x).label_prob += (label_prob + blank_prob) * pr * lm_prob;
                            (*x).space_counter += 1.0;
                        } else {
                            prefix_tree.insert(new_node_idx, SearchPoint {
                                node: new_node_idx,
                                label_prob: (label_prob + blank_prob) * pr * lm_prob,
                                blank_prob: 0.0,
                                space_counter: space_counter + 1.0,
                            });
                        }
                    } else {

                        // if a new character
                        let new_node_idx = suffix_tree
                            .get_child(node, label)
                            .unwrap_or_else(|| suffix_tree.add_node(node, label, idx));

                        if let Some(x) = prefix_tree.get_mut(&new_node_idx) {
                            (*x).label_prob += (label_prob + blank_prob) * pr;
                        } else {
                            prefix_tree.insert(new_node_idx, SearchPoint {
                                node: new_node_idx,
                                label_prob: (label_prob + blank_prob) * pr,
                                blank_prob: 0.0,
                                space_counter: space_counter,
                            });
                        }
                    }

                    // recover previous prefixes
                    if !prefix_tree.contains_key(&node) && prefix_tree.contains_key(&node) {
                        let new_node_idx = suffix_tree
                            .get_child(node, label)
                            .unwrap_or_else(|| suffix_tree.add_node(node, label, idx));

                        prefix_tree.insert(new_node_idx, SearchPoint {
                            node: new_node_idx,
                            label_prob: label_prob * pr,
                            blank_prob: (label_prob + blank_prob) * pr,
                            space_counter: space_counter,
                        });
                    }
                }
            }
        } // end SearchPoint

        beam = prefix_tree.values().cloned().collect::<Vec<_>>();

        let mut has_nans = false;
        beam.sort_unstable_by(|a, b| {
            (b.probability(beta))
                .partial_cmp(&(a.probability(beta)))
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
    }

    // Extract sentences
    let mut vec_tokens = Vec::<Vec<String>>::new();
    let mut vec_str = Vec::<String>::new();
    let mut vec_path = Vec::<Vec<usize>>::new();
    let mut vec_prob = Vec::<f32>::new();

    for &SearchPoint {
            node,
            label_prob,
            blank_prob,
            space_counter,
        } in &beam
    {

        let mut sequence = Vec::<String>::new();
        let mut path = Vec::<usize>::new();
        let prob = label_prob + blank_prob;

        if node != ROOT_NODE {
            for (label, &time) in suffix_tree.iter_from(node) {
                path.push(time);
                sequence.push(alphabet[label].to_string());
            }
        }

        sequence.reverse();
        vec_tokens.push(sequence.to_owned());
        vec_str.push(token_to_string(sequence, &alphabet_type).unwrap().to_owned());
        vec_path.push(path);
        vec_prob.push(prob);
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

