use std::collections::HashMap;

use anyhow::Result;
use parser::TCode;
use qdrant_client::client::{QdrantClient, QdrantClientConfig};
use qdrant_client::qdrant::vectors_config::Config;
use qdrant_client::qdrant::{
    CreateCollection, Distance, PointStruct, SearchBatchPoints, SearchPoints, Value, VectorParams,
    VectorParamsMap, VectorsConfig,
};

use crate::{CombinedEmbedding, Embedding};

pub const COLLECTION_NAME: &str = "DUCKYDUCK";

pub struct Qdrant {
    qdrant: QdrantClient,
}

impl Qdrant {
    pub fn from_url(url: &str, api_key: Option<String>) -> Result<Self> {
        let cfg = QdrantClientConfig::from_url(url)
            .with_api_key(api_key)
            .build()?;

        Ok(Self { qdrant: cfg })
    }

    pub async fn create_collection(&self) -> Result<()> {
        let map = collection_params();

        self.qdrant
            .create_collection(&CreateCollection {
                collection_name: COLLECTION_NAME.to_string(),
                vectors_config: Some(VectorsConfig {
                    config: Some(Config::ParamsMap(map)),
                }),
                ..Default::default()
            })
            .await?;

        Ok(())
    }

    pub async fn insert_docs(&self, docs: Vec<Embedding>) -> Result<()> {
        for embedding in docs {
            let value = serde_json::to_string(&embedding.code)?;
            let value: Value = serde_json::from_str(&value)?;

            let mut payload: HashMap<String, Value> = HashMap::new();
            payload.insert("data".to_string(), value);

            let point = PointStruct {
                id: Some(uuid::Uuid::new_v4().to_string().into()), // unique u64 or String
                vectors: Some(embedding.to_vectormap().into()),
                payload,
            };

            self.qdrant
                .upsert_points(COLLECTION_NAME.to_string(), None, vec![point], None)
                .await?;
        }

        Ok(())
    }

    pub async fn search(&self, embedding: CombinedEmbedding) -> Result<Vec<TCode>> {
        let search_points = vec![
            SearchPoints {
                collection_name: COLLECTION_NAME.into(),
                vector: embedding.0,
                limit: 4,
                with_payload: Some(true.into()),
                vector_name: Some("code".to_string()),
                ..Default::default()
            },
            SearchPoints {
                collection_name: COLLECTION_NAME.into(),
                vector: embedding.1,
                limit: 4,
                with_payload: Some(true.into()),
                vector_name: Some("nlp".to_string()),
                ..Default::default()
            },
        ];
        let batch_results = self
            .qdrant
            .search_batch_points(&SearchBatchPoints {
                collection_name: COLLECTION_NAME.into(),
                search_points,
                ..Default::default()
            })
            .await?;

        let mut final_response = Vec::new();

        for result in batch_results.result {
            let mut search_result = result
                .result
                .into_iter()
                .map(|x| {
                    let payload = x.payload.get("data").unwrap().to_owned();
                    serde_json::from_value::<TCode>(payload.into()).unwrap()
                })
                .collect::<Vec<TCode>>();

            final_response.append(&mut search_result);
        }
        Ok(final_response)
    }
}

fn collection_params() -> VectorParamsMap {
    let mut map: HashMap<String, VectorParams> = HashMap::new();
    map.insert(
        "nlp".to_string(),
        VectorParams {
            size: 384,
            distance: Distance::Cosine as i32,
            ..Default::default()
        },
    );

    map.insert(
        "code".to_string(),
        VectorParams {
            size: 768,
            distance: Distance::Cosine as i32,
            ..Default::default()
        },
    );

    VectorParamsMap { map }
}
