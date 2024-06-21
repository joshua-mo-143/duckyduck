use std::path::PathBuf;

pub mod qdrant;

use anyhow::Result;
use fastembed::{
    EmbeddingModel, InitOptions, InitOptionsUserDefined, TextEmbedding, TokenizerFiles,
    UserDefinedEmbeddingModel,
};
use hf_hub::{
    api::sync::{ApiBuilder, ApiRepo},
    Repo, RepoType,
};
use parser::TCode;
use std::collections::HashMap;

pub struct Embedding {
    code: TCode,
    jinabert_embedding: Vec<f32>,
    minilm_embedding: Vec<f32>,
}

impl Embedding {
    fn to_vectormap(&self) -> HashMap<String, Vec<f32>> {
        let mut map = HashMap::new();

        map.insert("code".to_string(), self.jinabert_embedding.clone());
        map.insert("nlp".to_string(), self.minilm_embedding.clone());

        map
    }
}

pub struct Embedder {
    minilm: TextEmbedding,
    jina_bert: TextEmbedding,
}

pub struct CombinedEmbedding(Vec<f32>, Vec<f32>);

impl CombinedEmbedding {
    pub fn into_vectormap(self) -> HashMap<String, Vec<f32>> {
        let mut map = HashMap::new();

        map.insert("code".to_string(), self.0);
        map.insert("nlp".to_string(), self.1);

        map
    }
}

impl Embedder {
    pub fn new() -> Result<Self> {
        // With custom InitOptions
        let minilm = TextEmbedding::try_new(InitOptions {
            model_name: EmbeddingModel::AllMiniLML6V2,
            show_download_progress: true,
            ..Default::default()
        })
        .unwrap();

        let token = std::env::var("HF_TOKEN").unwrap();
        let api = ApiBuilder::new().with_token(Some(token)).build()?;
        let model_id = "jinaai/jina-embeddings-v2-base-code".to_string();

        let repo = api.repo(Repo::with_revision(
            model_id,
            RepoType::Model,
            "main".to_string(),
        ));

        let jina_bert_code = TokenizerFilePaths::new(repo)?.into_model()?;

        let jina_bert = TextEmbedding::try_new_from_user_defined(
            jina_bert_code,
            InitOptionsUserDefined::default(),
        )?;

        Ok(Self { minilm, jina_bert })
    }

    pub fn embed_code(&self, documents: Vec<TCode>) -> Result<Vec<Embedding>> {
        let embeddings_as_text: Vec<String> = documents
            .iter()
            .map(|text| serde_json::to_string_pretty(text).unwrap())
            .collect();

        let jinabert_embeddings = self.jina_bert.embed(embeddings_as_text.clone(), None)?;
        let minilm_embeddings = self.minilm.embed(embeddings_as_text, None)?;

        let embeddings = itertools::izip!(documents, jinabert_embeddings, minilm_embeddings);

        let embeddings = embeddings
            .map(|(code, jinabert_embedding, minilm_embedding)| Embedding {
                jinabert_embedding,
                minilm_embedding,
                code,
            })
            .collect();

        Ok(embeddings)
    }

    pub fn embed_prompt(&self, prompt: String) -> Result<CombinedEmbedding> {
        let prompt = vec![prompt];

        let minilm_embedding = self
            .minilm
            .embed(prompt.clone(), None)?
            .into_iter()
            .next()
            .unwrap();

        let jinabert_embedding = self
            .jina_bert
            .embed(prompt, None)?
            .into_iter()
            .next()
            .unwrap();

        Ok(CombinedEmbedding(jinabert_embedding, minilm_embedding))
    }
}

struct TokenizerFilePaths {
    onnx_file: PathBuf,
    tokenizer_file: PathBuf,
    tokenizer_config_file: PathBuf,
    config_file: PathBuf,
    special_tokens_map_file: PathBuf,
}

impl TokenizerFilePaths {
    fn new(repo: ApiRepo) -> Result<Self> {
        let onnx_file = repo.get("onnx/model_quantized.onnx")?;
        let tokenizer_file = repo.get("tokenizer.json")?;
        let tokenizer_config_file = repo.get("tokenizer_config.json")?;
        let config_file = repo.get("config.json")?;
        let special_tokens_map_file = repo.get("special_tokens_map.json")?;

        Ok(Self {
            onnx_file,
            tokenizer_file,
            tokenizer_config_file,
            config_file,
            special_tokens_map_file,
        })
    }

    fn into_model(self) -> Result<UserDefinedEmbeddingModel> {
        let onnx_file = std::fs::read(self.onnx_file)?;
        let tokenizer_file = std::fs::read(self.tokenizer_file)?;
        let tokenizer_config_file = std::fs::read(self.tokenizer_config_file)?;
        let config_file = std::fs::read(self.config_file)?;
        let special_tokens_map_file = std::fs::read(self.special_tokens_map_file)?;

        let tokenizer_files = TokenizerFiles {
            tokenizer_file,
            tokenizer_config_file,
            config_file,
            special_tokens_map_file,
        };

        Ok(UserDefinedEmbeddingModel {
            onnx_file,
            tokenizer_files,
        })
    }
}
