use axum::{body::Body, extract::State, response::IntoResponse as AxumResponse, Form, Json};

use serde::Deserialize;

use parser::TCode;

use askama::Template;
use askama_axum::IntoResponse as AskamaResponse;

use crate::AppState;

#[derive(Deserialize)]
pub struct Prompt {
    prompt: String,
}

#[derive(Template)]
#[template(path = "index.html")]
struct IndexFile;

#[derive(Template)]
#[template(path = "table.html")]
struct TableResult {
    results: Vec<TCode>,
}

pub async fn prompt(
    State(state): State<AppState>,
    Form(Prompt { prompt }): Form<Prompt>,
) -> impl AskamaResponse {
    let embedding = state
        .embedder
        .embed_prompt(prompt)
        .inspect_err(|x| println!("Something went wrong: {x}"))
        .unwrap();

    let results = state
        .qdrant
        .search(embedding)
        .await
        .inspect_err(|x| println!("Something went wrong: {x}"))
        .unwrap();

    TableResult { results }
}

pub async fn prompt_text(
    State(state): State<AppState>,
    Json(Prompt { prompt }): Json<Prompt>,
) -> impl AxumResponse {
    println!("Request received!");
    Body::from_stream(state.ort.run_inference(prompt, 1024))
}

pub async fn homepage() -> impl AskamaResponse {
    IndexFile
}
