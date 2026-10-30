//! Cache verified file identities without accepting a changed file as verified.
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    os::unix::fs::{MetadataExt, OpenOptionsExt},
    path::Path,
};
#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct Stamp {
    name: String,
    sha256: String,
    device: u64,
    inode: u64,
    size: u64,
    mtime: i64,
    mtime_ns: i64,
    ctime: i64,
    ctime_ns: i64,
}
fn stamp(dir: &Path, name: &str, sha: &str) -> Result<Stamp> {
    let meta =
        fs::metadata(dir.join(name)).with_context(|| format!("Cannot stat model file {name}"))?;
    ensure!(meta.is_file(), "Model component is not a file: {name}");
    Ok(Stamp {
        name: name.into(),
        sha256: sha.into(),
        device: meta.dev(),
        inode: meta.ino(),
        size: meta.size(),
        mtime: meta.mtime(),
        mtime_ns: meta.mtime_nsec(),
        ctime: meta.ctime(),
        ctime_ns: meta.ctime_nsec(),
    })
}
/// Returns true if full hashes were recomputed. A cache miss or unreadable cache
/// never bypasses verification. Atomic best-effort caching is an optimization.
pub fn verify(dir: &Path, files: &[(&str, &str)]) -> Result<bool> {
    let before: Vec<_> = files
        .iter()
        .map(|(n, h)| stamp(dir, n, h))
        .collect::<Result<_>>()?;
    let cache = dir.join(".bsearch-model-verification-v1.json");
    if let Ok(data) = fs::read(&cache) {
        if let Ok(saved) = serde_json::from_slice::<Vec<Stamp>>(&data) {
            if saved == before {
                return Ok(false);
            }
        }
    }
    for (name, expected) in files {
        let mut f =
            File::open(dir.join(name)).with_context(|| format!("Cannot open model file {name}"))?;
        let mut h = Sha256::new();
        let mut buf = vec![0u8; 1024 * 1024];
        loop {
            let n = f.read(&mut buf)?;
            if n == 0 {
                break;
            }
            h.update(&buf[..n]);
        }
        ensure!(
            format!("{:x}", h.finalize()) == *expected,
            "Model file checksum mismatch: {name}"
        );
    }
    let after: Vec<_> = files
        .iter()
        .map(|(n, h)| stamp(dir, n, h))
        .collect::<Result<_>>()?;
    ensure!(before == after, "Model file changed during verification");
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_nanos();
    let temp = dir.join(format!(
        ".bsearch-model-verification-{}-{nonce}.tmp",
        std::process::id()
    ));
    if let Ok(mut f) = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(&temp)
    {
        let result = (|| -> Result<()> {
            serde_json::to_writer(&mut f, &after)?;
            f.write_all(b"\n")?;
            f.sync_all()?;
            fs::rename(&temp, &cache)?;
            Ok(())
        })();
        if result.is_err() {
            let _ = fs::remove_file(&temp);
        }
    }
    Ok(true)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn verified_cache_invalidates_on_same_size_change_and_corruption() {
        let d = tempfile::tempdir().unwrap();
        let path = d.path().join("weights");
        fs::write(&path, b"abc").unwrap();
        let good = format!("{:x}", Sha256::digest(b"abc"));
        let files = [("weights", good.as_str())];
        assert!(verify(d.path(), &files).unwrap());
        assert!(!verify(d.path(), &files).unwrap());
        fs::write(&path, b"abd").unwrap();
        assert!(verify(d.path(), &files).is_err());
        fs::write(&path, b"abc").unwrap();
        assert!(verify(d.path(), &files).unwrap());
        fs::write(
            d.path().join(".bsearch-model-verification-v1.json"),
            "broken",
        )
        .unwrap();
        assert!(verify(d.path(), &files).unwrap());
    }
}
