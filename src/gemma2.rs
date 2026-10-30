//! EmbeddingGemma 2 text-only inference on Apple's MLX. The graph follows the
//! mlx-vlm reference implementation (MIT), revision 3d87e884; see THIRD_PARTY_NOTICES.
use anyhow::{Context, Result, bail, ensure};
use mlx_rs::{
    Array, Device, Dtype, fast, nn,
    ops::{self, indexing::IndexOp},
};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

pub const MODEL_ID: &str =
    "embeddinggemma2-bf16:ffaf2604:tokenizer-4d777ef5:mlx-rs-0.33.0:ctx2048-truncate-b512:v1";
// Both files contain identical text tensors. The unchanged model ID names the
// upstream checkpoint lineage and allows reuse of proven-equivalent vectors.
const FULL_FILES: &[(&str, &str)] = &[
    (
        "model.safetensors",
        "ffaf2604b011ad0634022529bac930a4f6a2feba22687d8d5890434d186b459c",
    ),
    (
        "tokenizer.json",
        "4d777ef5bdc1aa36227abdfb77c3e49e7b9c892d16e1b6bda41c393504828be4",
    ),
    (
        "config.json",
        "24bc60679eecb0a61bcf9f5ecf7b01bbd9603b8e47aa469fe9f0569e17c55ed9",
    ),
];
const TEXT_FILES: &[(&str, &str)] = &[
    (
        "model.safetensors",
        "13fcb8ca7db3636f9efadee3c803a4b185b9165963c4022a3a1d2c52fb8e1ab3",
    ),
    (
        "tokenizer.json",
        "4d777ef5bdc1aa36227abdfb77c3e49e7b9c892d16e1b6bda41c393504828be4",
    ),
    (
        "config.json",
        "24bc60679eecb0a61bcf9f5ecf7b01bbd9603b8e47aa469fe9f0569e17c55ed9",
    ),
];
// Match the existing model context limit while bounding quadratic attention.
const MAX_TOKENS: usize = 2048;
const MAX_BATCH: usize = 8;
const MAX_PADDED_TOKENS: usize = 512;
const MAX_CACHE_BYTES: usize = 256 * 1024 * 1024;

pub struct Embedder {
    weights: HashMap<String, Array>,
    tokenizer: tokenizers::Tokenizer,
}
impl Embedder {
    pub fn load(dir: &Path) -> Result<Self> {
        let size = std::fs::metadata(dir.join("model.safetensors"))?.len();
        let files = if size == 542_056_448 {
            TEXT_FILES
        } else {
            FULL_FILES
        };
        crate::model_integrity::verify(dir, files)?;
        let installed = std::env::current_exe()?
            .parent()
            .context("Executable has no parent")?
            .join("../lib/bsearch/mlx-rs-0.33.metallib");
        let model_library = dir.join("mlx.metallib");
        let user_library = std::env::var_os("HOME")
            .map(PathBuf::from)
            .map(|home| home.join(".local/lib/bsearch/mlx-rs-0.33.metallib"));
        let library = if installed.is_file() {
            Some(installed)
        } else if model_library.is_file() {
            Some(model_library)
        } else {
            user_library.filter(|path| path.is_file())
        };
        if let Some(library) = library {
            if std::env::var_os("BSEARCH_MLX_MEMORY_TRACE").is_some() {
                eprintln!("mlx_library {}", library.display());
            }
            mlx_rs::metal::set_metallib_path(
                library
                    .to_str()
                    .context("Metal library path is not UTF-8")?,
            )?;
        }
        Device::set_default(&Device::gpu());
        mlx_rs::memory::set_cache_limit(MAX_CACHE_BYTES)?;
        let weights = Array::load_safetensors(dir.join("model.safetensors"))?;
        let mut tokenizer = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("Tokenizer: {e}"))?;
        tokenizer
            .with_truncation(Some(tokenizers::TruncationParams {
                max_length: MAX_TOKENS,
                ..Default::default()
            }))
            .map_err(|e| anyhow::anyhow!("Tokenizer truncation: {e}"))?;
        // Refuse incomplete or unexpected weights before beginning a scan.
        for k in [
            "language_model.embed_tokens.weight",
            "language_model.ple.per_layer_model_projection.weight",
            "language_model.ple.per_layer_projection_norm.weight",
            "language_model.norm.weight",
            "language_model.embedding_projection.weight",
        ] {
            ensure!(weights.contains_key(k), "Model lacks {k}");
        }
        for i in 0..24 {
            for k in [
                "input_layernorm.weight",
                "layer_scalar",
                "mlp.down_proj.weight",
                "mlp.gate_proj.weight",
                "mlp.up_proj.weight",
                "ple_block.per_layer_input_gate.weight",
                "ple_block.per_layer_projection.weight",
                "ple_block.post_per_layer_input_norm.weight",
                "post_attention_layernorm.weight",
                "post_feedforward_layernorm.weight",
                "pre_feedforward_layernorm.weight",
                "self_attn.k_norm.weight",
                "self_attn.k_proj.weight",
                "self_attn.o_proj.weight",
                "self_attn.q_norm.weight",
                "self_attn.q_proj.weight",
                "self_attn.v_proj.weight",
            ] {
                let key = format!("language_model.layers.{i}.{k}");
                ensure!(weights.contains_key(&key), "Model lacks {key}");
            }
        }
        Ok(Self { weights, tokenizer })
    }
    pub fn embed(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let mut encoded = Vec::with_capacity(texts.len());
        for (i, text) in texts.iter().enumerate() {
            let ids = self
                .tokenizer
                .encode(*text, true)
                .map_err(|e| anyhow::anyhow!("Tokenizer: {e}"))?
                .get_ids()
                .to_vec();
            ensure!(
                !ids.is_empty() && ids.len() <= MAX_TOKENS,
                "Input {i} has {} tokens; maximum is {MAX_TOKENS}",
                ids.len()
            );
            encoded.push((i, ids));
        }
        encoded.sort_by_key(|x| x.1.len());
        let mut result = vec![Vec::new(); texts.len()];
        let mut pos = 0;
        while pos < encoded.len() {
            let mut end = pos + 1;
            while end < encoded.len()
                && end - pos < MAX_BATCH
                && (end - pos + 1) * encoded[end].1.len() <= MAX_PADDED_TOKENS
            {
                end += 1;
            }
            let batch = &encoded[pos..end];
            let ids: Vec<&[u32]> = batch.iter().map(|x| x.1.as_slice()).collect();
            let vectors = self.forward(&ids)?;
            ensure!(vectors.len() == batch.len(), "Embedding count mismatch");
            for ((index, _), v) in batch.iter().zip(vectors) {
                result[*index] = v;
            }
            mlx_rs::memory::clear_cache()?;
            if std::env::var_os("BSEARCH_MLX_MEMORY_TRACE").is_some() {
                eprintln!(
                    "mlx_memory active={} cache={} peak={}",
                    mlx_rs::memory::active_memory()?,
                    mlx_rs::memory::cache_memory()?,
                    mlx_rs::memory::peak_memory()?
                );
            }
            pos = end;
        }
        Ok(result)
    }
    fn forward(&self, ids: &[&[u32]]) -> Result<Vec<Vec<f32>>> {
        let weights = &self.weights;
        let batch = ids.len();
        let n = ids.iter().map(|x| x.len()).max().context("Empty batch")?;
        let mut padded = vec![0u32; batch * n];
        for (i, t) in ids.iter().enumerate() {
            padded[i * n..i * n + t.len()].copy_from_slice(t);
        }
        let idx = Array::from_slice(&padded, &[batch as i32, n as i32]);
        let e = weights["language_model.embed_tokens.weight"].take_axis(&idx, 0)?;
        let e = &e * &Array::from_f32(512f32.sqrt()).as_dtype(Dtype::Bfloat16)?;
        let ple = linear(
            &e,
            &weights["language_model.ple.per_layer_model_projection.weight"],
        )?;
        let ple = (&ple * &Array::from_f32(512f32.powf(-0.5)).as_dtype(Dtype::Bfloat16)?)
            .reshape(&[batch as i32, n as i32, 24, 512])?;
        let ple = norm(
            &ple,
            Some(&weights["language_model.ple.per_layer_projection_norm.weight"]),
        )?;
        let mut h = e;
        // Keys are masked for padding. Masked query positions are ignored in pooling.
        let mask = |full: bool| {
            let mut v = Vec::with_capacity(batch * n * n);
            for t in ids {
                for q in 0..n {
                    for k in 0..n {
                        v.push(k < t.len() && (full || q.abs_diff(k) <= 512));
                    }
                }
            }
            Array::from_slice(&v, &[batch as i32, 1, n as i32, n as i32])
        };
        let full_mask = mask(true);
        let sliding_mask = mask(false);
        for i in 0..24 {
            let prefix = format!("language_model.layers.{i}.");
            let get = |k: &str| -> &Array { &weights[&(prefix.clone() + k)] };
            let x = norm(&h, Some(get("input_layernorm.weight")))?;
            let full = i % 6 == 5;
            let dim = if full { 512 } else { 256 };
            let kv = if full { 1 } else { 2 };
            let base = if full { 1_000_000.0 } else { 10_000.0 };
            let b = batch as i32;
            let n = n as i32;
            let q = norm(
                &linear(&x, get("self_attn.q_proj.weight"))?.reshape(&[b, n, 4, dim])?,
                Some(get("self_attn.q_norm.weight")),
            )?
            .transpose_axes(&[0, 2, 1, 3])?;
            let k = norm(
                &linear(&x, get("self_attn.k_proj.weight"))?.reshape(&[b, n, kv, dim])?,
                Some(get("self_attn.k_norm.weight")),
            )?
            .transpose_axes(&[0, 2, 1, 3])?;
            let v = norm(
                &linear(&x, get("self_attn.v_proj.weight"))?.reshape(&[b, n, kv, dim])?,
                None,
            )?
            .transpose_axes(&[0, 2, 1, 3])?;
            let q = rope(&q, n, dim, base)?;
            let k = rope(&k, n, dim, base)?;
            let attn_mask = if full { &full_mask } else { &sliding_mask };
            let a = fast::scaled_dot_product_attention(
                &q,
                &k,
                &v,
                1.0,
                Some(fast::ScaledDotProductAttentionMask::Array(attn_mask)),
                None,
            )?
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[b, n, 4 * dim])?;
            let a = norm(
                &linear(&a, get("self_attn.o_proj.weight"))?,
                Some(get("post_attention_layernorm.weight")),
            )?;
            h = &h + &a;
            let x = norm(&h, Some(get("pre_feedforward_layernorm.weight")))?;
            let gate = nn::gelu_approximate(linear(&x, get("mlp.gate_proj.weight"))?)?;
            let up = linear(&x, get("mlp.up_proj.weight"))?;
            let a = norm(
                &linear(&(&gate * &up), get("mlp.down_proj.weight"))?,
                Some(get("post_feedforward_layernorm.weight")),
            )?;
            h = &h + &a;
            let p = ple.index((.., .., i as i32, ..));
            let gate =
                nn::gelu_approximate(linear(&h, get("ple_block.per_layer_input_gate.weight"))?)?;
            let a = norm(
                &linear(&(&gate * &p), get("ple_block.per_layer_projection.weight"))?,
                Some(get("ple_block.post_per_layer_input_norm.weight")),
            )?;
            h = &(&h + &a) * get("layer_scalar");
        }
        let h = norm(&h, Some(&weights["language_model.norm.weight"]))?;
        let h = linear(&h, &weights["language_model.embedding_projection.weight"])?
            .as_dtype(Dtype::Float32)?;
        let pool: Vec<f32> = ids
            .iter()
            .flat_map(|t| {
                (0..n).map(move |i| {
                    if i < t.len() {
                        1.0 / t.len() as f32
                    } else {
                        0.0
                    }
                })
            })
            .collect();
        let h =
            (&h * &Array::from_slice(&pool, &[batch as i32, n as i32, 1])).sum_axis(1, false)?;
        let length = h.square()?.sum_axis(-1, true)?.sqrt()?;
        let out = (&h / &length).as_dtype(Dtype::Float32)?;
        out.eval()?;
        let raw = out.as_slice::<f32>();
        ensure!(raw.len() == batch * 768, "Invalid vector shape");
        let mut ret = Vec::with_capacity(batch);
        for v in raw.chunks_exact(768) {
            if !v.iter().all(|x| x.is_finite()) {
                bail!("Nonfinite model vector")
            }
            ret.push(v.to_vec());
        }
        Ok(ret)
    }
}
fn linear(x: &Array, w: &Array) -> Result<Array> {
    Ok(x.matmul(w.t())?)
}
fn norm(x: &Array, w: Option<&Array>) -> Result<Array> {
    Ok(fast::rms_norm(x, w, 1e-6)?)
}
fn rope(x: &Array, n: i32, dim: i32, base: f32) -> Result<Array> {
    let half = dim / 2;
    let mut angles = Vec::with_capacity((n * dim) as usize);
    for pos in 0..n {
        let a: Vec<f32> = (0..half)
            .map(|i| pos as f32 / base.powf((2 * i) as f32 / dim as f32))
            .collect();
        angles.extend_from_slice(&a);
        angles.extend_from_slice(&a);
    }
    let angles = Array::from_slice(&angles, &[1, 1, n, dim]);
    let cos = angles.cos()?.as_dtype(Dtype::Bfloat16)?;
    let sin = angles.sin()?.as_dtype(Dtype::Bfloat16)?;
    let first = x.index((.., .., .., 0..half));
    let second = x.index((.., .., .., half..dim));
    let negative = &second * &Array::from_f32(-1.0).as_dtype(Dtype::Bfloat16)?;
    let rot = ops::concatenate(&[&negative, &first], -1)?;
    Ok(&(x * &cos) + &(&rot * &sin))
}
