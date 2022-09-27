use crate::audio_model::{AudioModel, DecoderType};
use crate::fast_math::{softmax, normalize};
use ndarray::{Axis, Array, Array1, Array2, Array3, ArrayBase, Data, OwnedRepr, Ix2, Ix3, s};
use std::{error::Error, boxed::Box, collections::HashMap};

extern crate triton_rust;
use triton_rust::TritonInference;
use triton_rust::inference::model_infer_request::{InferInputTensor, InferRequestedOutputTensor};
use triton_rust::system_shared_memory::SystemSharedMemoryRegionHandle;

pub struct RNNTState {
    input_state_1: ArrayBase<OwnedRepr<f32>, Ix3>,
    input_state_2: ArrayBase<OwnedRepr<f32>, Ix3>,
    last_token_input_state_1: ArrayBase<OwnedRepr<f32>, Ix3>,
    last_token_input_state_2: ArrayBase<OwnedRepr<f32>, Ix3>,
    buffered_token: u32,
    encoder_index: usize
}

pub struct NemoRNNTAudioModel {
    model_name: String,
    triton_inferer: TritonInference,
    input_state_1_mem: SystemSharedMemoryRegionHandle,
    input_state_2_mem: SystemSharedMemoryRegionHandle,
    encoder_mem: SystemSharedMemoryRegionHandle,
    target_length_mem: SystemSharedMemoryRegionHandle,
    target_mem: SystemSharedMemoryRegionHandle,
    output_mem: SystemSharedMemoryRegionHandle,
    output_state_1_mem: SystemSharedMemoryRegionHandle,
    output_state_2_mem: SystemSharedMemoryRegionHandle,
    encoder_state: ArrayBase<OwnedRepr<f32>, Ix3>,
    state_map: HashMap<usize, RNNTState>,
    num_rnn_layers: usize,
    hidden_size: usize,
    encoder_size: usize,
    alphabet_length: usize,
    blank_token: usize,
    sep_token: usize
}

impl NemoRNNTAudioModel {
    pub fn load<D: Data<Elem = f32>>(server_address: &str, model_name: &str, encoder_array: &ArrayBase<D, Ix3>, num_rnn_layers: usize, hidden_size: usize, encoder_size: usize, alphabet_length: usize, blank_token: usize, sep_token: usize) -> NemoRNNTAudioModel {

        let server_add_str: &'static str = Box::leak(server_address.to_string().into_boxed_str());
        let mut triton_inferer = TritonInference::connect(&server_add_str).unwrap();

        /* Create shared memory zones */
        triton_inferer.unregister_system_shared_memory("input_state_1_data").unwrap();
        triton_inferer.unregister_system_shared_memory("input_state_2_data").unwrap();
        triton_inferer.unregister_system_shared_memory("encoder_data").unwrap();
        triton_inferer.unregister_system_shared_memory("target_length_data").unwrap();
        triton_inferer.unregister_system_shared_memory("target_data").unwrap();

        triton_inferer.unregister_system_shared_memory("outputs_data").unwrap();
        triton_inferer.unregister_system_shared_memory("output_state_1_data").unwrap();
        triton_inferer.unregister_system_shared_memory("output_state_2_data").unwrap();

        let system_mem_zone_input_state_1 = triton_inferer.create_system_shared_memory("input_state_1_data", "/input_state_1_data", (num_rnn_layers * hidden_size * 4) as u64).unwrap();
        let system_mem_zone_input_state_2 = triton_inferer.create_system_shared_memory("input_state_2_data", "/input_state_2_data", (num_rnn_layers * hidden_size * 4) as u64).unwrap();
        let system_mem_zone_encoder = triton_inferer.create_system_shared_memory("encoder_data", "/encoder_data", (encoder_size * 4) as u64).unwrap();
        let system_mem_zone_target_length = triton_inferer.create_system_shared_memory("target_length_data", "/target_length_data", 4).unwrap();
        let system_mem_zone_target = triton_inferer.create_system_shared_memory("target_data", "/target_data", 4).unwrap();

        let system_mem_zone_output = triton_inferer.create_system_shared_memory("outputs_data", "/outputs_data", (alphabet_length * 4) as u64).unwrap();
        let system_mem_zone_output_state_1 = triton_inferer.create_system_shared_memory("output_state_1_data", "/output_state_1_data", (num_rnn_layers * hidden_size * 4) as u64).unwrap();
        let system_mem_zone_output_state_2 = triton_inferer.create_system_shared_memory("output_state_2_data", "/output_state_2_data", (num_rnn_layers * hidden_size * 4) as u64).unwrap();

        /* Create the state map */
        let state_map = HashMap::<usize, RNNTState>::with_capacity(20);

        NemoRNNTAudioModel {
            model_name: model_name.to_string(),
            triton_inferer: triton_inferer,
            input_state_1_mem: system_mem_zone_input_state_1,
            input_state_2_mem: system_mem_zone_input_state_2,
            encoder_mem: system_mem_zone_encoder,
            target_length_mem: system_mem_zone_target_length,
            target_mem: system_mem_zone_target,
            output_mem: system_mem_zone_output,
            output_state_1_mem: system_mem_zone_output_state_1,
            output_state_2_mem: system_mem_zone_output_state_2,
            encoder_state: encoder_array.to_owned(),
            state_map: state_map,
            num_rnn_layers: num_rnn_layers,
            hidden_size: hidden_size,
            encoder_size: encoder_size,
            alphabet_length: alphabet_length,
            blank_token: blank_token,
            sep_token: sep_token
        }
    }
}

impl AudioModel for NemoRNNTAudioModel {
    fn get_next_tokens(&mut self, last_token: Option<usize>, state_index: usize) -> Result<(Array1<f32>, usize), Box<dyn Error + Send + Sync>> {

        let probs = false;

        let input_state_1_ref;
        let input_state_2_ref;
        let buffered_token: u32;
        let mut encoder_index;

        let input_state_1;
        let input_state_2;

        if state_index == 0 {
            input_state_1 = Array3::<f32>::zeros((self.num_rnn_layers, 1, self.hidden_size));
            input_state_2 = Array3::<f32>::zeros((self.num_rnn_layers, 1, self.hidden_size));
            input_state_1_ref = &input_state_1;
            input_state_2_ref = &input_state_2;
            buffered_token = self.blank_token as u32;
            encoder_index = 0;
        } else {
            let token = last_token.unwrap(); // should be defined
            let state = self.state_map.get(&state_index).unwrap();
            encoder_index = state.encoder_index;

            if (token != self.blank_token) && (token != self.sep_token) {
                buffered_token = token as u32;
                input_state_1_ref = &state.input_state_1;
                input_state_2_ref = &state.input_state_2;
            } else {
                buffered_token = state.buffered_token;
                input_state_1_ref = &state.last_token_input_state_1;
                input_state_2_ref = &state.last_token_input_state_2;
            }

            if (token == self.blank_token) || (token == self.sep_token) {
                encoder_index = encoder_index + 1;
            }
        }

        let encoder_vec = self.encoder_state.slice(s![.., .., encoder_index]).insert_axis(Axis(2)).to_owned();

        /* Infer using shared memory */
        let size_of_state = (self.num_rnn_layers * self.hidden_size * 4) as u64;
        let size_of_encoder = (self.encoder_size * 4) as u64;
        let mut infer_inputs = Vec::<InferInputTensor>::with_capacity(5);

        let state_1_params = self.triton_inferer.get_system_shared_memory_params("input_state_1_data", size_of_state, 0);
        infer_inputs.push(self.triton_inferer.get_infer_input("input-states-1", "FP32", &[self.num_rnn_layers as i64, 1, self.hidden_size as i64], state_1_params));

        let state_2_params = self.triton_inferer.get_system_shared_memory_params("input_state_2_data", size_of_state, 0);
        infer_inputs.push(self.triton_inferer.get_infer_input("input-states-2", "FP32", &[self.num_rnn_layers as i64, 1, self.hidden_size as i64], state_2_params));

        let encoder_input_params = self.triton_inferer.get_system_shared_memory_params("encoder_data", size_of_encoder, 0);
        infer_inputs.push(self.triton_inferer.get_infer_input("encoder_outputs", "FP32", &[1, self.encoder_size as i64, 1], encoder_input_params));

        let target_length_params = self.triton_inferer.get_system_shared_memory_params("target_length_data", 4, 0);
        infer_inputs.push(self.triton_inferer.get_infer_input("target_length", "INT32", &[1], target_length_params));

        let target_params = self.triton_inferer.get_system_shared_memory_params("target_data", 4, 0);
        infer_inputs.push(self.triton_inferer.get_infer_input("targets", "INT32", &[1, 1], target_params));

        self.input_state_1_mem.copy_array(input_state_1_ref, 0);
        self.input_state_2_mem.copy_array(input_state_2_ref, 0);
        self.encoder_mem.copy_array(&encoder_vec, 0);
        self.target_length_mem.copy_array(&Array1::<i32>::from_elem(1, 1), 0);
        self.target_mem.copy_array(&Array2::<i32>::from_elem((1,1), buffered_token as i32), 0);

        let size_of_output = (self.alphabet_length * 4) as u64;
        let mut infer_outputs = Vec::<InferRequestedOutputTensor>::with_capacity(3);

        let outputs_params = self.triton_inferer.get_system_shared_memory_params("outputs_data", size_of_output, 0);
        infer_outputs.push(self.triton_inferer.get_infer_output("outputs", outputs_params));

        let output_state_1_params = self.triton_inferer.get_system_shared_memory_params("output_state_1_data", size_of_state, 0);
        infer_outputs.push(self.triton_inferer.get_infer_output("output-states-1", output_state_1_params));

        let output_state_2_params = self.triton_inferer.get_system_shared_memory_params("output_state_2_data", size_of_state, 0);
        infer_outputs.push(self.triton_inferer.get_infer_output("output-states-2", output_state_2_params));

        let response  = self.triton_inferer.infer(&self.model_name, "1", "25", infer_inputs, infer_outputs, Vec::<Vec<u8>>::new()).unwrap();

        let outputs: Vec<f32> = self.output_mem.get_data(size_of_output, 0);
        let array_outputs_nd = Array::from_iter(softmax(normalize(outputs)).into_iter());

        let output_state_1: Vec<f32> = self.output_state_1_mem.get_data(size_of_state, 0);
        let array_output_state_1_nd = Array::from_iter(output_state_1.into_iter()).into_shape((self.num_rnn_layers, 1, self.hidden_size)).unwrap();

        let output_state_2: Vec<f32> = self.output_state_2_mem.get_data(size_of_state, 0);
        //let rounded_output_state_2: Vec<f32> = output_state_2.into_iter().map(|x| f32::trunc(x  * 10000.0) / 10000.0).collect();
        let array_output_state_2_nd = Array::from_iter(output_state_2.into_iter()).into_shape((self.num_rnn_layers, 1, self.hidden_size)).unwrap();

        let new_state_index = self.state_map.len() + 1;

        self.state_map.insert(new_state_index,
        RNNTState {
            input_state_1: array_output_state_1_nd.to_owned(),
            input_state_2: array_output_state_2_nd.to_owned(),
            last_token_input_state_1: input_state_1_ref.to_owned(),
            last_token_input_state_2: input_state_2_ref.to_owned(),
            buffered_token: buffered_token,
            encoder_index: encoder_index
        });

        Ok((array_outputs_nd.to_owned(), new_state_index))
    }

    fn is_terminal_state(&self, last_token: Option<usize>, state_index: usize) -> bool {
        if state_index == 0 {
            return false;
        }

        let state = self.state_map.get(&state_index).unwrap();

        if let Some(token) = last_token {
            if ((state.encoder_index == self.encoder_state.dim().2 - 1) && (token == self.blank_token)) || (token == self.sep_token) {
                return true;
            }
        }

        return false;
    }

    fn get_type(&self) -> DecoderType {
        return DecoderType::NemoRNNT;
    }

} // implt NemoRNNTAudioModel