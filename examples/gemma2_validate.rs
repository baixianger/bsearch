#[path = "../src/model_integrity.rs"]
mod model_integrity;
// Explicit real-model test; accepts model directory and synthetic reference JSON.
#[path = "../src/gemma2.rs"]
mod gemma2;
use anyhow::{Result, ensure};
use serde_json::Value;
use std::{path::Path, time::Instant};
fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("model directory");
    let fixture = args.next().expect("reference json");
    let rows: Vec<Value> = serde_json::from_slice(&std::fs::read(fixture)?)?;
    let texts: Vec<&str> = rows.iter().map(|v| v["text"].as_str().unwrap()).collect();
    let t = Instant::now();
    let mut model = gemma2::Embedder::load(Path::new(&path))?;
    let load = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let batches = model.embed(&texts)?;
    let seconds = t.elapsed().as_secs_f64();
    let mut min = 1.0f64;
    let mut max = 0.0f64;
    for (i, (v, row)) in batches.iter().zip(&rows).enumerate() {
        let mut dot = 0f64;
        let mut a = 0f64;
        let mut b = 0f64;
        for (x, y) in v.iter().zip(row["vector"].as_array().unwrap()) {
            let x = *x as f64;
            let y = y.as_f64().unwrap();
            dot += x * y;
            a += x * x;
            b += y * y;
            max = max.max((x - y).abs());
        }
        let cosine = dot / (a * b).sqrt();
        min = min.min(cosine);
        ensure!(cosine > 0.999, "sample {i} differs: {cosine}");
    }
    for (i, text) in texts.iter().enumerate() {
        let single = model.embed(&[*text])?.remove(0);
        let cosine = single
            .iter()
            .zip(&batches[i])
            .map(|(a, b)| (*a as f64) * (*b as f64))
            .sum::<f64>();
        ensure!(
            cosine > 0.999,
            "batch/single differ for sample {i}: {cosine}"
        );
    }
    let long = " cat".repeat(2047);
    let truncated = model.embed(&[&long])?.remove(0);
    let reference = batches.last().unwrap();
    let cosine = truncated
        .iter()
        .zip(reference)
        .map(|(a, b)| (*a as f64) * (*b as f64))
        .sum::<f64>();
    ensure!(cosine > 0.999, "over-limit truncation differs: {cosine}");
    let score = |a: usize, b: usize| {
        batches[a]
            .iter()
            .zip(&batches[b])
            .map(|(a, b)| a * b)
            .sum::<f32>()
    };
    ensure!(score(0, 1) > score(0, 2));
    ensure!(score(3, 4) > score(3, 5));
    println!(
        "{}",
        serde_json::json!({"samples":texts.len(),"long_tokens":626,"min_cosine":min,"max_abs_error":max,"batch_seconds":seconds,"load_seconds":load,"max_input_tokens":2048,"batch_single":"pass","retrieval":"pass"})
    );
    Ok(())
}
