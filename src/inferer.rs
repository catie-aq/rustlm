use tokenizers::tokenizer::{Result, Tokenizer, PaddingParams};
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::models::bpe::BPE;

use ndarray::{s, Axis, Array};

use onnxruntime::{
    environment::Environment, session::Session, tensor::OrtOwnedTensor, GraphOptimizationLevel, LoggingLevel,
};

pub struct GPT2Inferer {
    tokenizer: Tokenizer,
    environment: Environment, // /!\ needed for avoiding segfault
    session: Session,
}

impl GPT2Inferer {
    pub fn load_from_files(vocab_path: &str, merge_path: &str, model_path: &str, padding_id: u32, padding_string: &str) -> Result<GPT2Inferer> {

        let bpe_builder = BPE::from_file(vocab_path, merge_path);
        let bpe = bpe_builder
            .build()?;

        let mut _params = PaddingParams::default();
        _params.pad_id = padding_id;
        _params.pad_token = String::from(padding_string);

        let mut tokenizer = Tokenizer::new(bpe);
        tokenizer.with_padding(Some(_params));

        tokenizer.with_pre_tokenizer(ByteLevel::default().add_prefix_space(false))
            .with_decoder(ByteLevel::default())
            .with_post_processor(ByteLevel::default());

         let environment = Environment::builder()
            .with_name("test-gpt2")
            .with_log_level(LoggingLevel::Error)
            .build()?;

        let session = environment
            .new_session_builder()?
            .with_optimization_level(GraphOptimizationLevel::All)?
            .with_number_threads(4)?
            .with_model_from_file(model_path)?;

        Ok(GPT2Inferer {
            tokenizer: tokenizer,
            environment: environment,
            session: session,
        })
    }

    pub fn infer(&mut self, input: Vec<String>) -> Result<Vec<f32>> {
        let encoding = self.tokenizer.encode_batch(input, false)?;

        let length = encoding[0].get_ids().len();
        let batch_size = encoding.len();

        let array_emb = Array::from_shape_fn((batch_size, length), |(i, j)| encoding[i].get_ids()[j]);
        let array_attention = Array::from_shape_fn((batch_size, length), |(i, j)| encoding[i].get_attention_mask()[j]);
        let input_array = array_emb.mapv(|elem| elem as i64); // convert type

        let outputs: Vec<OrtOwnedTensor<f32, _>> = self.session.run(vec![input_array])?;

        let mut log_probs = outputs[0].slice(s![.., ..-1, ..]).to_owned();
        log_probs.par_mapv_inplace(f32::exp);

        let mut result: f32 = 0.0;
        let mut result_vec = Vec::<f32>::new();

        // compute cross entropy
        for (i, batch) in log_probs.axis_iter(Axis(0)).enumerate() {
            for (j, seq) in batch.axis_iter(Axis(0)).enumerate() {
                if array_attention[[i, j+1]] == 1 {
                    result += seq[array_emb[[i, j+1]] as usize].ln() - (seq.scalar_sum()).ln();
                }
            }
            result_vec.push(result);
            result = 0.0;
        }

        Ok(result_vec)
    }
}
