use flate2::read::GzDecoder;
use std::io::{Cursor, Read};
use std::{path::Path, sync::Arc};
use tokio_tar::Archive;

use http_body_util::BodyExt;
use octocrab::Octocrab;

pub struct GitHub {
    ctx: Arc<Octocrab>,
}

impl GitHub {
    pub fn new() -> Self {
        Self {
            ctx: octocrab::instance(),
        }
    }

    pub async fn fetch_repo(&self, org: &str, repo: &str) -> Vec<u8> {
        let repo = self.ctx.repos(org, repo);

        let latest_commit = repo
            .list_commits()
            .send()
            .await
            .unwrap()
            .into_iter()
            .next()
            .unwrap()
            .sha;

        repo.download_tarball(latest_commit)
            .await
            .unwrap()
            .into_body()
            .collect()
            .await
            .unwrap()
            .to_bytes()
            .to_vec()
    }

    pub async fn unpack_repo<P: AsRef<Path>>(&self, path: P, bytes: Vec<u8>) {
        let mut gzip = GzDecoder::new(Cursor::new(bytes));
        let mut decompressed_bytes = Vec::new();
        gzip.read_to_end(&mut decompressed_bytes).unwrap();

        let mut ar = Archive::new(Cursor::new(decompressed_bytes));

        ar.unpack(path).await.unwrap();
    }
}

impl Default for GitHub {
    fn default() -> Self {
        Self::new()
    }
}
