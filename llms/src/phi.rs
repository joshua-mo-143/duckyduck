use futures::stream::Stream;

use rand::Rng;
use serde::de::Deserializer;
use serde::Deserialize;
use std::collections::HashSet;

use crate::tokenizer::TokenOutputStream;

use candle_transformers::models::mistral::{Config, Model as Mistral};

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use tokenizers::Tokenizer;

use hf_hub::{api::sync::ApiBuilder, api::sync::ApiRepo, Repo, RepoType};

pub use ort::Result as OrtResult;

const PROMPT: &str = r#"
            You are a robot that specialises in turning natural language into JSON. Your area of expertise is the Rust programming language.

Using the Rust libraries available, you will decide what libraries the user requires.

Libraries available: [
    'tokio',
    'warp',
    'axum',
    'actix-web'
]

Response format:
{
    "libraries": ["libraries", "go", "here"]
}

You will answer using the JSON format only. Do not add anything else.
"#;

#[derive(Clone)]
pub struct TextGeneration {
    model: Mistral,
    device: Device,
    tokenizer: TokenOutputStream,
    seed: u64,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

impl TextGeneration {
    pub fn new() -> anyhow::Result<Self> {
        let token = std::env::var("HF_TOKEN").unwrap();

        let repo = get_repo(token)?;
        let tokenizer = get_tokenizer(&repo)?;
        let device = Device::Cpu;
        let filenames = hub_load_safetensors(&repo, "model.safetensors.index.json")?;

        let config = Config::config_7b_v0_1(false);

        let model = {
            let dtype = DType::F32;
            let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };
            Mistral::new(&config, vb)?
        };

        Ok(Self {
            model,
            tokenizer: TokenOutputStream::new(tokenizer),
            seed: 532532532532,
            repeat_penalty: 1.1, //repeat penalty
            repeat_last_n: 64,   // context window for repeat penalty
            device: device.clone(),
        })
    }

    pub fn run_inference(
        mut self,
        prompt: String,
        sample_len: usize,
    ) -> impl Stream<Item = anyhow::Result<String>> {
        self.tokenizer.clear();
        let mut tokens = self
            .tokenizer
            .tokenizer()
            .encode(prompt, true)
            .unwrap()
            .get_ids()
            .to_vec();

        println!("Got tokens!");

        let eos_token = match self.tokenizer.get_token("</s>") {
            Some(token) => token,
            None => panic!("cannot find the </s> token"),
        };

        let mut logits_processor = LogitsProcessor::new(self.seed, Some(0.0), None);

        async_stream::try_stream! {

        for index in 0..sample_len {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let start_pos = tokens.len().saturating_sub(context_size);
            let ctxt = &tokens[start_pos..];
            let input = Tensor::new(ctxt, &self.device)
                .unwrap()
                .unsqueeze(0)
                .unwrap();
            let logits = self.model.forward(&input, start_pos).unwrap();
            let logits = logits
                .squeeze(0)
                .unwrap()
                .squeeze(0)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap();
            let logits = if self.repeat_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(self.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.repeat_penalty,
                    &tokens[start_at..],
                )
                .unwrap()
            };

            let next_token = logits_processor.sample(&logits)?;
            tokens.push(next_token);

            if next_token == eos_token {
                break;
            }

            if let Some(t) = self.tokenizer.next_token(next_token)? {
                println!("Found a token!");
                yield t;
            }
        }
        }
    }
}

fn get_repo(token: String) -> anyhow::Result<ApiRepo> {
    let api = ApiBuilder::new().with_token(Some(token)).build()?;

    let model_id = "mistralai/Mistral-7B-v0.1".to_string();

    Ok(api.repo(Repo::with_revision(
        model_id,
        RepoType::Model,
        "main".to_string(),
    )))
}

fn get_tokenizer(repo: &ApiRepo) -> anyhow::Result<Tokenizer> {
    let tokenizer_filename = repo.get("tokenizer.json")?;

    Ok(Tokenizer::from_file(tokenizer_filename).unwrap())
}

#[derive(Debug, Deserialize)]
struct Weightmaps {
    #[serde(deserialize_with = "deserialize_weight_map")]
    weight_map: HashSet<String>,
}

// Custom deserializer for the weight_map to directly extract values into a HashSet
fn deserialize_weight_map<'de, D>(deserializer: D) -> Result<HashSet<String>, D::Error>
where
    D: Deserializer<'de>,
{
    let map = serde_json::Value::deserialize(deserializer)?;
    match map {
        serde_json::Value::Object(obj) => Ok(obj
            .values()
            .filter_map(|v| v.as_str().map(ToString::to_string))
            .collect::<HashSet<String>>()),
        _ => Err(serde::de::Error::custom(
            "Expected an object for weight_map",
        )),
    }
}

pub fn hub_load_safetensors(
    repo: &hf_hub::api::sync::ApiRepo,
    json_file: &str,
) -> anyhow::Result<Vec<std::path::PathBuf>> {
    let json_file = repo.get(json_file).map_err(candle_core::Error::wrap)?;
    let json_file = std::fs::File::open(json_file)?;
    let json: Weightmaps = serde_json::from_reader(&json_file).map_err(candle_core::Error::wrap)?;

    let pathbufs: Vec<std::path::PathBuf> = json
        .weight_map
        .iter()
        .map(|f| repo.get(f).unwrap())
        .collect();

    Ok(pathbufs)
}
