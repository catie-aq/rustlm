use ndarray::Array2;
use crate::language_model::{LanguageModel, LMType};
use std::unimplemented;
use std::{fs::File, io::{BufRead, BufReader}, error::Error};

extern crate patricia_tree;
use patricia_tree::PatriciaSet;

pub struct DicoLanguageModel {
    trie: PatriciaSet,
}

impl DicoLanguageModel {
    pub fn load_from_files(filename: &str) -> Result<DicoLanguageModel, Box<dyn Error + Send + Sync>> {

        let mut trie = PatriciaSet::new();

        let file = File::open(filename)?;
        let buf = BufReader::new(file);

        let lines: Vec<_> = buf.lines().map(|l| l.unwrap()).collect();

        for word in lines {
            trie.insert(word);
        }

        Ok(DicoLanguageModel {
            trie: trie,
        })
    }
}

impl LanguageModel for DicoLanguageModel {

    fn get_sentence_likelihood(&mut self, input: Vec<String>) -> Result<Vec<f32>, Box<dyn Error + Send + Sync>> {
        unimplemented!()
    }

    fn get_next_letter(&mut self, input: Vec<String>, vocab: &[String], blank_id: usize, space_id: usize) -> Result<Array2<f32>, Box<dyn Error + Send + Sync>> {
        let mut result_array = Array2::<f32>::zeros([input.len(), vocab.len() + 1]);

        for (i, prefix) in input.iter().enumerate() {
            let prefix_splitted: Vec<_> = prefix.split_whitespace().collect();
            if (prefix_splitted.len() == 0) ||  prefix.ends_with(' ') { // root node

                for k in self.trie.iter() {
                    let first_char = k[0];
                    let index = vocab.iter().position(|r| r.as_bytes()[0] == first_char).unwrap();
                    result_array[[i, index]] = 1.0;
                }

                // can always be blank between words
                result_array[[i, blank_id]] = 1.0;
                result_array[[i, space_id]] = 1.0;
            } else {
                let last_prefix = prefix_splitted[prefix_splitted.len() - 1].to_owned();
                let last_prefix_bytes = last_prefix.as_bytes();

                for n in self.trie.iter_prefix(last_prefix_bytes) {
                    if last_prefix_bytes == n { // word ended
                        result_array[[i, space_id]] = 1.0;
                        result_array[[i, blank_id]] = 1.0;
                    } else {
                        let index = vocab.iter().position(|r| r.as_bytes()[0] == n[last_prefix_bytes.len()]).unwrap();
                        result_array[[i, index]] = 1.0;

                        if n[last_prefix_bytes.len()] == last_prefix_bytes[last_prefix_bytes.len() - 1] {
                            // case of duplicates letters should allow blank
                            result_array[[i, blank_id]] = 1.0;
                        }

                        // marginally allow blank
                        //result_array[[i, blank_id]] = 0.01;
                    }
                }
            }
        }

        Ok(result_array)
    }

    fn lm_type(&self) -> LMType {
        LMType::ShallowFusion
    }
}
