use anyhow::{Context, Result, ensure};
use memmap2::MmapOptions;
use safetensors::{SafeTensors, tensor::serialize_to_file};
use sha2::{Digest, Sha256};
use std::{fs::File, io::Read, path::Path};
fn hash(path: &Path) -> Result<String> {
    let mut f = File::open(path)?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1024 * 1024];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(format!("{:x}", h.finalize()))
}
fn main() -> Result<()> {
    let mut a = std::env::args_os().skip(1);
    let src = a.next().context("source")?;
    let dst = a.next().context("target")?;
    let src = Path::new(&src);
    let dst = Path::new(&dst);
    ensure!(!dst.exists(), "target already exists");
    ensure!(
        hash(src)? == "ffaf2604b011ad0634022529bac930a4f6a2feba22687d8d5890434d186b459c",
        "source hash mismatch"
    );
    let file = File::open(src)?;
    let mmap = unsafe { MmapOptions::new().map(&file)? };
    let tensors = SafeTensors::deserialize(&mmap)?;
    let text: Vec<_> = tensors
        .tensors()
        .into_iter()
        .filter(|(name, _)| name.starts_with("language_model."))
        .collect();
    ensure!(!text.is_empty(), "no text tensors");
    let (_, metadata) = SafeTensors::read_metadata(&mmap)?;
    serialize_to_file(text.clone(), metadata.metadata().clone(), dst)?;
    let out = File::open(dst)?;
    let output = unsafe { MmapOptions::new().map(&out)? };
    let saved = SafeTensors::deserialize(&output)?;
    ensure!(saved.names().len() == text.len(), "tensor count mismatch");
    for (name, original) in &text {
        let new = saved.tensor(name)?;
        ensure!(
            original.dtype() == new.dtype()
                && original.shape() == new.shape()
                && original.data() == new.data(),
            "tensor mismatch: {name}"
        );
    }
    println!(
        "text_tensors={} total_tensors={} bytes={} sha256={} all_text_tensors_byte_identical=true",
        text.len(),
        tensors.names().len(),
        out.metadata()?.len(),
        hash(dst)?
    );
    Ok(())
}
