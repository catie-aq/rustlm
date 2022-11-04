use ndarray::{Array, Array2, Axis};
use crate::language_model::{LanguageModel, LMType};
use std::unimplemented;
use std::{error::Error, boxed::Box};

extern crate tokenizers;
use tokenizers::tokenizer::Tokenizer;

extern crate triton_rust;
use triton_rust::TritonInference;
use triton_rust::inference::model_infer_request::{InferInputTensor, InferRequestedOutputTensor};
use triton_rust::system_shared_memory::SystemSharedMemoryRegionHandle;


pub struct CausalLMFinalLanguageModel {
    model_name: String,
    triton_inferer: TritonInference,
    tokenizer: Tokenizer,
    ids_mem: SystemSharedMemoryRegionHandle,
    mask_mem: SystemSharedMemoryRegionHandle,
    output_mem: SystemSharedMemoryRegionHandle,
    max_batch_size: usize,
    max_length: usize,
    model_output_size: usize,
}

impl CausalLMFinalLanguageModel {
    pub fn load_from_files(server_address: &str, model_name: &str, max_batch_size: usize, max_length: usize, model_output_size: usize) -> Result<CausalLMFinalLanguageModel, Box<dyn Error + Send + Sync>> {

        let server_add_str: &'static str = Box::leak(server_address.to_string().into_boxed_str());
        let mut triton_inferer = TritonInference::connect(&server_add_str).unwrap();

        let mut tokenizer = Tokenizer::from_pretrained("bert-base-cased", None).unwrap();

        /* Create shared memory zones */
        triton_inferer.unregister_system_shared_memory("input_ids_data").unwrap();
        triton_inferer.unregister_system_shared_memory("attention_mask_data").unwrap();
        triton_inferer.unregister_system_shared_memory("output_data").unwrap();
        let system_mem_zone_ids = triton_inferer.create_system_shared_memory("input_ids_data", "/input_ids_data", (max_length*8) as u64).unwrap();
        let system_mem_zone_mask = triton_inferer.create_system_shared_memory("attention_mask_data", "/attention_mask_data", (max_length*8) as u64).unwrap();
        let system_mem_zone_output = triton_inferer.create_system_shared_memory("output_data", "/output_data", (max_length*8*model_output_size) as u64).unwrap();

        Ok(CausalLMFinalLanguageModel {
            model_name: model_name.to_string(),
            triton_inferer: triton_inferer,
            tokenizer: tokenizer,
            ids_mem: system_mem_zone_ids,
            mask_mem: system_mem_zone_mask,
            output_mem: system_mem_zone_output,
            max_batch_size: max_batch_size,
            max_length: max_length,
            model_output_size: model_output_size
        })
    }
}

impl LanguageModel for CausalLMFinalLanguageModel {

    fn get_sentence_likelihood(&mut self, input: Vec<String>) -> Result<Vec<f32>, Box<dyn Error + Send + Sync>> {
        /* Tokenize sentences */
        let encoding = self.tokenizer.encode_batch(input.to_owned(), false).unwrap();

        let tokens: Vec<Vec<u32>> = encoding.iter().map(|x| x.get_ids().to_vec()).collect();
        let mask: Vec<Vec<u32>> = encoding.iter().map(|x| x.get_attention_mask().to_vec()).collect();
        let lengths: Vec<usize> = tokens.iter().map(|x| x.len()).collect();

        let max_length = *lengths.iter().max().unwrap();
        let batch_size = tokens.len();

        let mut tokens_array = Array2::<i64>::zeros((batch_size, max_length));
        for (i, vec_tok) in tokens.iter().enumerate() {
            for (j, tok) in vec_tok.iter().enumerate() {
                tokens_array[[i, j]] = (*tok) as i64;
            }
        }

        let mut mask_array = Array2::<i64>::zeros((batch_size, max_length));
        for (i, vec_tok) in mask.iter().enumerate() {
            for (j, tok) in vec_tok.iter().enumerate() {
                mask_array[[i, j]] = (*tok) as i64;
            }
        }

        /* Infer using shared memory */
        let size_of_input = (max_length * batch_size * 8) as u64;
        let mut infer_inputs = Vec::<InferInputTensor>::with_capacity(2);
        let ids_input_params = self.triton_inferer.get_system_shared_memory_params("input_ids_data", size_of_input, 0);
        infer_inputs.push(self.triton_inferer.get_infer_input("input_ids", "INT64", &[batch_size as i64, max_length as i64], ids_input_params));

        let mask_input_params = self.triton_inferer.get_system_shared_memory_params("attention_mask_data", size_of_input, 0);
        infer_inputs.push(self.triton_inferer.get_infer_input("attention_mask", "INT64", &[batch_size as i64, max_length as i64], mask_input_params));

        self.ids_mem.copy_array(&tokens_array, 0);
        self.mask_mem.copy_array(&mask_array, 0);

        let size_of_output = (batch_size * max_length * self.model_output_size * 4) as u64;
        let mut infer_outputs = Vec::<InferRequestedOutputTensor>::with_capacity(1);
        let output_params = self.triton_inferer.get_system_shared_memory_params("output_data", size_of_output, 0);
        infer_outputs.push(self.triton_inferer.get_infer_output("logits", output_params));

        let response  = self.triton_inferer.infer(&self.model_name, "1", "25", infer_inputs, infer_outputs, Vec::<Vec<u8>>::new()).unwrap();

        let output_class: Vec<f32> = self.output_mem.get_data(size_of_output, 0);
        let array_nd = Array::from_iter(output_class.into_iter()).into_shape((batch_size, max_length, self.model_output_size)).unwrap();
        let log_array = array_nd.map(|x| f32::log2(*x));

        // Compute log probabilities of the output
        let mut vec_log_prob = Vec::<f32>::with_capacity(batch_size);
        for (i, vec) in tokens_array.axis_iter(Axis(0)).enumerate() {
            let mut log_prob = 0.0;
            for (j, tok) in vec.iter().enumerate() {
                let token_prob = log_array[[i, j, *tok as usize]];
                if (mask_array[[i,j]] != 0) && !f32::is_nan(token_prob) {
                    log_prob = log_prob + token_prob;
                }
            }
            vec_log_prob.push(log_prob);
        }

        Ok(vec_log_prob)
    }

    fn get_next_letter(&mut self, input: Vec<String>, vocab: &[String], blank_id: usize, space_id: usize) -> Result<Array2<f32>, Box<dyn Error + Send + Sync>> {
        unimplemented!()
    }

    fn lm_type(&self) -> LMType {
        LMType::FinalRescoring
    }
}
