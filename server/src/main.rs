use axum::{
    routing::{get, post},
    Router,
};
use llms::{phi::TextGeneration, qdrant::Qdrant, Embedder};
use std::sync::Arc;
use tokio::net::TcpListener;

mod routes;
use routes::{homepage, prompt, prompt_text};

#[derive(Clone)]
pub struct AppState {
    qdrant: Arc<Qdrant>,
    embedder: Arc<Embedder>,
    ort: TextGeneration,
}

impl AppState {
    fn new() -> Self {
        let qdrant = Qdrant::from_url("http://localhost:6334", None)
            .expect("Running Qdrant gRPC at localhost:6334");
        let embedder = Embedder::new().unwrap();

        let qdrant = Arc::new(qdrant);
        let embedder = Arc::new(embedder);
        let ort = TextGeneration::new().unwrap();

        Self {
            qdrant,
            embedder,
            ort,
        }
    }
}

#[tokio::main]
async fn main() {
    let state = AppState::new();

    let rtr = Router::new()
        .route("/", get(homepage))
        .route("/prompt", post(prompt))
        .route("/prompt/text", post(prompt_text))
        .with_state(state);

    let tcp_listener = TcpListener::bind("127.0.0.1:8000").await.unwrap();

    println!("Starting server at port 8000...");
    axum::serve(tcp_listener, rtr).await.unwrap();
}
