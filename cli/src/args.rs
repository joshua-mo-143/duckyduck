use clap::{Parser, Subcommand};
use llms::{qdrant::Qdrant, Embedder};
use parser::github::GitHub;
use std::path::PathBuf;

use crate::error::Error;

#[derive(Parser)]
#[command(version, about, long_about = None)]
pub struct Args {
    #[command(subcommand)]
    command: Commands,
}

impl Args {
    pub async fn process(self) -> Result<(), Error> {
        match self.command {
            Commands::Embed { dir } => {
                let items = parser::process_dir(dir);

                let embedder = Embedder::new().map_err(|x| Error::Initialise(x.to_string()))?;

                let embeddings = embedder
                    .embed_code(items)
                    .map_err(|_| Error::Parsing)
                    .unwrap();

                let qdrant = Qdrant::from_url("http://localhost:6334", None).unwrap();

                // qdrant.create_collection().await.unwrap();

                qdrant.insert_docs(embeddings).await.unwrap();
            }
            Commands::Fetch { org, repo } => {
                println!("Fetching {org}/{repo}...");
                let github = GitHub::new();
                let bytes = github.fetch_repo("tokio-rs", "tokio").await;
                println!("Repo has been fetched.");

                github.unpack_repo("./things", bytes).await;
                println!("Repo has been unpacked.")
            }

            Commands::Search { prompt } => {
                let embedder = Embedder::new().map_err(|x| Error::Initialise(x.to_string()))?;

                let embedding = embedder.embed_prompt(prompt).unwrap();

                let qdrant = Qdrant::from_url("http://localhost:6334", None).unwrap();

                let res = qdrant.search(embedding).await.unwrap();
                println!("{:?}", res);
            }
        }

        Ok(())
    }
}

#[derive(Subcommand)]
pub enum Commands {
    /// does testing things
    Embed {
        #[arg(short, long, value_name = "FILE", default_value = ".")]
        dir: PathBuf,
    },

    /// Fetch a public repo from GitHub
    Fetch {
        #[arg(short, long)]
        org: String,
        #[arg(short, long)]
        repo: String,
    },
    Search {
        #[arg(short, long)]
        prompt: String,
    },
}
