//! Experimental native Metal embedding backend. Its vectors must never be
//! mixed with the older MLX bf16 index, even though both have 768 dimensions.
use anyhow::{Context, Result, ensure};
use sha2::{Digest, Sha256};
use std::{fs::File, io::Read, path::Path};

pub const MODEL_SHA256: &str = "b5ce9d77a3fc4b3b39ccb5643c36777911cc4eb46a66962eadfa3f5f60490d63";
pub const MODEL_ID: &str = "embeddinggemma-300m-q8-gguf:b5ce9d77:crispembed-0.16.1:metal-v1";

pub struct Embedder(crispembed::CrispEmbed);
impl Embedder {
    pub fn load(path: &Path) -> Result<Self> {
        let mut file =
            File::open(path).with_context(|| format!("Cannot open model {}", path.display()))?;
        let mut digest = Sha256::new();
        let mut buf = [0_u8; 1024 * 1024];
        loop {
            let n = file.read(&mut buf)?;
            if n == 0 {
                break;
            }
            digest.update(&buf[..n]);
        }
        ensure!(
            format!("{:x}", digest.finalize()) == MODEL_SHA256,
            "Model checksum differs from the pinned embedding backend"
        );
        crispembed::set_gpu_backend("metal");
        let model =
            crispembed::CrispEmbed::new(path.to_str().context("Model path is not UTF-8")?, 0)
                .map_err(anyhow::Error::msg)?;
        Ok(Self(model))
    }

    pub fn embed(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let vectors = self.0.encode_batch(texts);
        ensure!(vectors.len() == texts.len(), "Embedding count mismatch");
        for vector in &vectors {
            ensure!(
                vector.len() == 768 && vector.iter().all(|v| v.is_finite()),
                "Invalid model vector"
            );
        }
        Ok(vectors)
    }
}
