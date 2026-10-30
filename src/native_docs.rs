//! Staged Rust document indexer. The caller must select a separate database;
//! this module refuses to mix vectors from different model versions.
use anyhow::{Context, Result, ensure};
use chrono::Utc;
use globset::GlobSet;
use rusqlite::{Connection, OptionalExtension, params};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::HashSet,
    fs,
    io::Read,
    path::{Component, Path, PathBuf},
};

pub trait Encoder {
    fn embed(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
}

pub struct Settings<'a> {
    pub root: &'a Path,
    pub roots: &'a [String],
    pub includes: &'a GlobSet,
    pub excludes: &'a GlobSet,
    pub model_id: &'a str,
    pub metadata_fingerprint: &'a str,
}

#[derive(Debug, Default)]
pub struct ScanResult {
    pub seen: usize,
    pub changed: usize,
    pub embedded: usize,
    pub errors: usize,
}

fn digest(data: &[u8]) -> String {
    format!("{:x}", Sha256::digest(data))
}
fn now() -> String {
    Utc::now().to_rfc3339()
}
pub(crate) fn allowed(rel: &str, cfg: &Settings<'_>) -> bool {
    let parts: Vec<_> = rel.split('/').collect();
    if parts
        .first()
        .is_some_and(|p| p.eq_ignore_ascii_case("archive"))
        || !rel.to_lowercase().ends_with(".md")
        || parts.iter().any(|p| {
            p.starts_with('.')
                || [
                    "credentials",
                    "bookkeeping",
                    "node_modules",
                    "target",
                    "build",
                    "dist",
                    "venv",
                    "__pycache__",
                    "_large-assets",
                ]
                .contains(&p.to_lowercase().as_str())
        })
    {
        return false;
    }
    cfg.roots
        .iter()
        .any(|r| rel.starts_with(&format!("{}/", r.trim_end_matches('/'))))
        && cfg.includes.is_match(rel.to_lowercase())
        && !cfg.excludes.is_match(rel)
}
fn visit(dir: &Path, cfg: &Settings<'_>, out: &mut Vec<PathBuf>) -> Result<()> {
    let mut entries = fs::read_dir(dir)
        .with_context(|| format!("Cannot scan {}", dir.display()))?
        .map(|e| e.map(|e| e.path()))
        .collect::<std::io::Result<Vec<_>>>()?;
    entries.sort();
    for p in entries {
        let info = fs::symlink_metadata(&p)?;
        if info.file_type().is_symlink() {
            continue;
        }
        let name = p
            .file_name()
            .and_then(|n| n.to_str())
            .context("Non-UTF-8 filename")?;
        if name.starts_with('.')
            || [
                "credentials",
                "bookkeeping",
                "node_modules",
                "target",
                "build",
                "dist",
                "venv",
                "__pycache__",
                "_large-assets",
            ]
            .contains(&name.to_lowercase().as_str())
        {
            continue;
        }
        if info.is_dir() {
            visit(&p, cfg, out)?;
        } else if info.is_file() {
            let rel = p
                .strip_prefix(cfg.root)?
                .to_str()
                .context("Non-UTF-8 path")?;
            if allowed(rel, cfg) {
                out.push(p);
            }
        }
    }
    Ok(())
}
pub fn inventory(cfg: &Settings<'_>) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for area in cfg.roots {
        let path = Path::new(area);
        ensure!(
            !path.is_absolute() && !path.components().any(|c| c == Component::ParentDir),
            "Invalid document root"
        );
        if area
            .split('/')
            .next()
            .is_some_and(|v| v.eq_ignore_ascii_case("archive"))
        {
            continue;
        }
        let base = cfg.root.join(path);
        ensure!(
            !fs::symlink_metadata(&base)?.file_type().is_symlink()
                && base.is_dir()
                && base.canonicalize()?.starts_with(cfg.root),
            "Document root unavailable or outside workspace"
        );
        visit(&base, cfg, &mut files)?;
    }
    files.sort();
    files.dedup();
    Ok(files)
}
pub fn hash_file(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path)?;
    let mut hash = Sha256::new();
    let mut buf = [0_u8; 1024 * 1024];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hash.update(&buf[..n]);
    }
    Ok(format!("{:x}", hash.finalize()))
}
pub(crate) fn describe(rel: &str, text: &str, registry: &Value) -> (String, String, String) {
    let title = text
        .lines()
        .find_map(|l| l.strip_prefix("# ").map(str::trim))
        .unwrap_or_else(|| {
            Path::new(rel)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("")
        });
    let mut summary = text
        .lines()
        .find_map(|l| l.strip_prefix("summary:").map(str::trim))
        .or_else(|| {
            text.lines()
                .find(|l| !l.trim().is_empty() && !l.starts_with('#') && !l.starts_with("---"))
                .map(str::trim)
        })
        .unwrap_or("")
        .to_owned();
    let parts: Vec<_> = rel.split('/').collect();
    let mut category = if parts.len() > 2 {
        format!("{}/{}", parts[0], parts[1])
    } else {
        parts[0].to_owned()
    };
    if parts.first() == Some(&"entities") && parts.len() > 2 {
        let rest = parts[2..].join("/");
        let found = registry["subjects"][parts[1]]["holds"]
            .as_object()
            .and_then(|holds| {
                holds
                    .keys()
                    .filter(|c| rest.starts_with(&format!("{}/", c.trim_end_matches('/'))))
                    .max_by_key(|c| c.len())
            });
        category = found
            .map(|c| format!("entities/{}/{c}", parts[1]))
            .unwrap_or_else(|| {
                Path::new(rel)
                    .parent()
                    .unwrap_or(Path::new(""))
                    .to_string_lossy()
                    .into_owned()
            });
    }
    if summary.is_empty() {
        if let Some(entries) = registry["entries"].as_array() {
            if let Some(e) = entries
                .iter()
                .filter(|e| {
                    e["where"].as_str().is_some_and(|w| {
                        rel == w || rel.starts_with(&format!("{}/", w.trim_end_matches('/')))
                    })
                })
                .max_by_key(|e| e["where"].as_str().unwrap_or("").len())
            {
                summary = e["title"].as_str().unwrap_or("").to_owned();
            }
        }
    }
    (
        title.chars().take(300).collect(),
        summary.chars().take(600).collect(),
        category,
    )
}
pub(crate) fn pieces(rel: &str, title: &str, text: &str) -> Vec<(usize, usize, String, String)> {
    let chars: Vec<char> = text.chars().collect();
    (0..chars.len())
        .step_by(600)
        .enumerate()
        .filter_map(|(ordinal, start)| {
            let body: String = chars[start..(start + 700).min(chars.len())]
                .iter()
                .collect();
            let body = body.trim();
            if body.is_empty() {
                return None;
            }
            let value = format!("title: {title} | text: {rel}\n{body}");
            let line = 1 + chars[..start].iter().filter(|c| **c == '\n').count();
            let hash = digest(value.as_bytes());
            Some((ordinal, line, value, hash))
        })
        .collect()
}
pub fn initialize(db: &Connection, model_id: &str) -> Result<()> {
    let has_vectors: bool = db.query_row(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='vectors')",
        [],
        |r| r.get(0),
    )?;
    if has_vectors {
        let incompatible: usize = db.query_row(
            "SELECT count(*) FROM vectors WHERE model<>?",
            [model_id],
            |r| r.get(0),
        )?;
        ensure!(
            incompatible == 0,
            "Existing database contains vectors from a different model; use a separate staging database"
        );
    }
    db.execute_batch("PRAGMA journal_mode=WAL; CREATE TABLE IF NOT EXISTS files(path TEXT PRIMARY KEY,sha256 TEXT,title TEXT,summary TEXT,category TEXT,status TEXT,error TEXT,updated TEXT,area TEXT,kind TEXT DEFAULT 'document'); CREATE TABLE IF NOT EXISTS chunks(path TEXT,ordinal INTEGER,line INTEGER,text TEXT,hash TEXT,PRIMARY KEY(path,ordinal)); CREATE TABLE IF NOT EXISTS vectors(hash TEXT,model TEXT,vector BLOB,PRIMARY KEY(hash,model)); CREATE TABLE IF NOT EXISTS state(key TEXT PRIMARY KEY,value TEXT);")?;
    Ok(())
}
pub fn scan<F>(
    db: &mut Connection,
    cfg: &Settings<'_>,
    no_embed: bool,
    model_factory: F,
) -> Result<ScanResult>
where
    F: FnOnce() -> Result<Box<dyn Encoder>>,
{
    // Enumerate before mutating. An unreadable/missing root cannot mark files missing.
    let paths = inventory(cfg)?;
    initialize(db, cfg.model_id)?;
    let registry_path = cfg.root.join("entities/registry.json");
    let registry: Value = if registry_path.exists() {
        serde_json::from_slice(&fs::read(registry_path)?)?
    } else {
        serde_json::json!({})
    };
    let old_meta: Option<String> = db
        .query_row(
            "SELECT value FROM state WHERE key='registry_sha'",
            [],
            |r| r.get(0),
        )
        .optional()?;
    let refresh = old_meta.as_deref() != Some(cfg.metadata_fingerprint);
    let mut result = ScanResult {
        seen: paths.len(),
        ..Default::default()
    };
    let mut seen = HashSet::new();
    for path in paths {
        let rel = path
            .strip_prefix(cfg.root)?
            .to_str()
            .context("Non-UTF-8 path")?
            .to_owned();
        seen.insert(rel.clone());
        let process=||->Result<(String,String,String,String,Vec<(usize,usize,String,String)>,String)>{
            let hash=hash_file(&path)?;
            let bytes=if fs::metadata(&path)?.len()<=16*1024*1024{fs::read(&path)?}else{Vec::new()};
            if !bytes.is_empty(){ensure!(digest(&bytes)==hash,"File changed while scanning");}
            let text=String::from_utf8_lossy(&bytes).into_owned();
            let (title,summary,category)=describe(&rel,&text,&registry);
            let chunks=pieces(&rel,&title,&text);
            let status=if chunks.is_empty(){"metadata-only"}else{"pending"}.to_owned();
            Ok((hash,title,summary,category,chunks,status))
        };
        match process() {
            Ok((hash, title, summary, category, chunks, status)) => {
                let old: Option<(String, String)> = db
                    .query_row(
                        "SELECT sha256,status FROM files WHERE path=?",
                        [&rel],
                        |r| Ok((r.get(0)?, r.get(1)?)),
                    )
                    .optional()?;
                if !refresh
                    && old
                        .as_ref()
                        .is_some_and(|(h, s)| h == &hash && s != "error" && s != "missing")
                {
                    continue;
                }
                let tx = db.transaction()?;
                tx.execute("DELETE FROM chunks WHERE path=?", [&rel])?;
                tx.execute("INSERT OR REPLACE INTO files(path,sha256,title,summary,category,status,error,updated,area,kind) VALUES (?,?,?,?,?,?,?,?,?,'document')",params![rel,hash,title,summary,category,status,if status=="metadata-only"{"oversize or empty Markdown"}else{""},now(),rel.split('/').next().unwrap_or("")])?;
                for (ordinal, line, text, hash) in chunks {
                    tx.execute(
                        "INSERT INTO chunks VALUES (?,?,?,?,?)",
                        params![rel, ordinal, line, text, hash],
                    )?;
                }
                tx.commit()?;
                result.changed += 1;
            }
            Err(err) => {
                let tx = db.transaction()?;
                tx.execute("DELETE FROM chunks WHERE path=?", [&rel])?;
                tx.execute("INSERT OR REPLACE INTO files(path,sha256,title,summary,category,status,error,updated,area,kind) VALUES (?,'','','','','error',?,?,?,'document')",params![rel,err.to_string(),now(),rel.split('/').next().unwrap_or("")])?;
                tx.commit()?;
                result.errors += 1;
            }
        }
    }
    let existing: Vec<String> = {
        let mut q = db.prepare("SELECT path FROM files")?;
        q.query_map([], |r| r.get(0))?
            .collect::<rusqlite::Result<_>>()?
    };
    {
        let tx = db.transaction()?;
        for rel in existing {
            if !allowed(&rel, cfg) {
                tx.execute("DELETE FROM files WHERE path=?", [&rel])?;
                tx.execute("DELETE FROM chunks WHERE path=?", [&rel])?;
            } else if !seen.contains(&rel) {
                tx.execute("UPDATE files SET status='missing' WHERE path=?", [&rel])?;
                tx.execute("DELETE FROM chunks WHERE path=?", [&rel])?;
            }
        }
        tx.commit()?;
    }
    let pending: Vec<(String, String)> = {
        let mut q=db.prepare("SELECT DISTINCT c.hash,c.text FROM chunks c JOIN files f ON f.path=c.path LEFT JOIN vectors v ON v.hash=c.hash AND v.model=? WHERE f.status='pending' AND v.hash IS NULL")?;
        q.query_map([cfg.model_id], |r| Ok((r.get(0)?, r.get(1)?)))?
            .collect::<rusqlite::Result<_>>()?
    };
    ensure!(
        !no_embed || pending.is_empty(),
        "{} chunks require embedding; --no-embed forbids model use",
        pending.len()
    );
    if !pending.is_empty() {
        let mut model = model_factory()?;
        for batch in pending.chunks(32) {
            let texts: Vec<_> = batch.iter().map(|r| r.1.clone()).collect();
            let vectors = model.embed(&texts)?;
            ensure!(vectors.len() == batch.len(), "Embedding count mismatch");
            let tx = db.transaction()?;
            for ((hash, _), vector) in batch.iter().zip(vectors) {
                ensure!(
                    !vector.is_empty() && vector.iter().all(|x| x.is_finite()),
                    "Invalid vector"
                );
                let bytes: Vec<u8> = vector.iter().flat_map(|x| x.to_le_bytes()).collect();
                tx.execute(
                    "INSERT OR REPLACE INTO vectors VALUES (?,?,?)",
                    params![hash, cfg.model_id, bytes],
                )?;
            }
            tx.commit()?;
            result.embedded += batch.len();
        }
    }
    let tx = db.transaction()?;
    tx.execute("UPDATE files SET status='ready' WHERE status='pending' AND NOT EXISTS (SELECT 1 FROM chunks c LEFT JOIN vectors v ON v.hash=c.hash AND v.model=? WHERE c.path=files.path AND v.hash IS NULL)",[cfg.model_id])?;
    tx.execute(
        "DELETE FROM vectors WHERE hash NOT IN (SELECT hash FROM chunks)",
        [],
    )?;
    tx.execute(
        "INSERT OR REPLACE INTO state VALUES ('registry_sha',?)",
        [cfg.metadata_fingerprint],
    )?;
    tx.execute(
        "INSERT OR REPLACE INTO state VALUES ('completed',?)",
        [now()],
    )?;
    tx.commit()?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use globset::Glob;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    struct Fake;
    impl Encoder for Fake {
        fn embed(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            Ok(texts.iter().map(|t| vec![1.0, t.len() as f32]).collect())
        }
    }
    #[test]
    fn markdown_incremental_scope_failure_and_noop() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().canonicalize().unwrap();
        for dir in [
            "library/notes",
            "library/credentials",
            "archive",
            "tools",
            "tools/sample/target/deps",
            "tools/sample/build/deps",
        ] {
            fs::create_dir_all(root.join(dir)).unwrap();
        }
        let path = root.join("library/notes/a.md");
        fs::write(&path, "# A\nfirst text").unwrap();
        fs::write(root.join("library/notes/a.pdf"), "excluded").unwrap();
        fs::write(root.join("library/credentials/password.md"), "excluded").unwrap();
        fs::write(root.join("archive/a.md"), "excluded").unwrap();
        fs::write(root.join("tools/sample/target/deps/README.md"), "excluded").unwrap();
        fs::write(root.join("tools/sample/build/deps/README.md"), "excluded").unwrap();
        let roots = vec!["library".to_owned(), "tools".to_owned()];
        let includes = Glob::new("**/*.md").unwrap().compile_matcher();
        let mut include = globset::GlobSetBuilder::new();
        include.add(Glob::new("**/*.md").unwrap());
        let include = include.build().unwrap();
        assert!(includes.is_match("library/notes/a.md"));
        let exclude = globset::GlobSetBuilder::new().build().unwrap();
        let cfg = Settings {
            root: &root,
            roots: &roots,
            includes: &include,
            excludes: &exclude,
            model_id: "test-v2",
            metadata_fingerprint: "m1",
        };
        let calls = Arc::new(AtomicUsize::new(0));
        let mut db = Connection::open_in_memory().unwrap();
        let factory = || {
            calls.fetch_add(1, Ordering::SeqCst);
            Ok(Box::new(Fake) as Box<dyn Encoder>)
        };
        let first = scan(&mut db, &cfg, false, factory).unwrap();
        assert_eq!((first.seen, first.changed, first.embedded), (1, 1, 1));
        let old: String = db
            .query_row("SELECT hash FROM chunks", [], |r| r.get(0))
            .unwrap();
        let second = scan(&mut db, &cfg, false, factory).unwrap();
        assert_eq!(
            (
                second.changed,
                second.embedded,
                calls.load(Ordering::SeqCst)
            ),
            (0, 0, 1)
        );
        fs::write(&path, "# A\nchanged text").unwrap();
        assert!(scan(&mut db, &cfg, true, factory).is_err());
        assert_eq!(
            db.query_row("SELECT status FROM files", [], |r| r.get::<_, String>(0))
                .unwrap(),
            "pending"
        );
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        let third = scan(&mut db, &cfg, false, factory).unwrap();
        assert_eq!(third.embedded, 1);
        assert_eq!(
            db.query_row("SELECT count(*) FROM vectors WHERE hash=?", [old], |r| {
                r.get::<_, usize>(0)
            })
            .unwrap(),
            0
        );
        fs::remove_file(path).unwrap();
        scan(&mut db, &cfg, false, factory).unwrap();
        assert_eq!(
            db.query_row("SELECT status FROM files", [], |r| r.get::<_, String>(0))
                .unwrap(),
            "missing"
        );
        assert_eq!(
            db.query_row("SELECT count(*) FROM chunks", [], |r| r.get::<_, usize>(0))
                .unwrap(),
            0
        );
    }

    #[test]
    fn rejects_incompatible_vectors_and_incomplete_traversal_without_removing_records() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().canonicalize().unwrap();
        fs::create_dir_all(root.join("library")).unwrap();
        fs::write(root.join("library/a.md"), "# A").unwrap();
        let roots = vec!["library".to_owned()];
        let mut builder = globset::GlobSetBuilder::new();
        builder.add(Glob::new("**/*.md").unwrap());
        let include = builder.build().unwrap();
        let exclude = globset::GlobSetBuilder::new().build().unwrap();
        let cfg = Settings {
            root: &root,
            roots: &roots,
            includes: &include,
            excludes: &exclude,
            model_id: "old",
            metadata_fingerprint: "m",
        };
        let mut db = Connection::open_in_memory().unwrap();
        scan(&mut db, &cfg, false, || Ok(Box::new(Fake))).unwrap();
        let new = Settings {
            model_id: "new",
            ..cfg
        };
        assert!(scan(&mut db, &new, false, || Ok(Box::new(Fake))).is_err());
        assert_eq!(
            db.query_row("SELECT status FROM files", [], |r| r.get::<_, String>(0))
                .unwrap(),
            "ready"
        );
        fs::rename(root.join("library"), root.join("offline")).unwrap();
        assert!(scan(&mut db, &new, false, || Ok(Box::new(Fake))).is_err());
        assert_eq!(
            db.query_row("SELECT status FROM files", [], |r| r.get::<_, String>(0))
                .unwrap(),
            "ready"
        );
    }
}
