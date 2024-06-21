mod args;

use args::Args;
use clap::Parser;
use error::Error;

mod error;

#[tokio::main]
async fn main() -> Result<(), Error> {
    Args::parse().process().await?;

    Ok(())
}
