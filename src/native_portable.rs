//! Validated portable snapshots for the native model; legacy MLX bundles are
//! rejected because their vectors occupy a different embedding space.
use crate::{native_docs, native_skills};
use anyhow::{Context, Result, ensure};
use rusqlite::{Connection, OpenFlags, OptionalExtension, params};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
};

#[derive(Serialize, Deserialize)]
struct Manifest {
    format: u32,
    document_model: String,
    skill_model: String,
    settings_sha256: String,
    documents: BTreeMap<String, String>,
    skills: BTreeMap<String, String>,
    chunks: BTreeSet<String>,
    files: BTreeMap<String, String>,
}
fn inventory(cfg: &native_docs::Settings<'_>) -> Result<BTreeMap<String, String>> {
    let mut result = BTreeMap::new();
    for path in native_docs::inventory(cfg)? {
        let rel = path
            .strip_prefix(cfg.root)?
            .to_str()
            .context("Non-UTF-8 path")?
            .to_owned();
        result.insert(rel, native_docs::hash_file(&path)?);
    }
    Ok(result)
}
fn skill_hashes(root: &Path) -> Result<BTreeMap<String, String>> {
    Ok(native_skills::discover(root)?
        .into_iter()
        .map(|(name, s)| (name, s.md5))
        .collect())
}
fn verify_skill_index(
    path: &Path,
    cfg: &native_docs::Settings<'_>,
    expected: &BTreeMap<String, String>,
) -> Result<native_skills::Index> {
    let index = native_skills::read_index(path, cfg.model_id)?;
    let actual: BTreeMap<_, _> = index
        .skills
        .iter()
        .map(|(name, s)| (name.clone(), s.md5.clone()))
        .collect();
    ensure!(
        &actual == expected,
        "Skill index differs from current skill files"
    );
    for skill in index.skills.values() {
        for chunk in &skill.chunks {
            ensure!(
                chunk.vector.len() == 768 && chunk.vector.iter().all(|v| v.is_finite()),
                "Skill index has an invalid vector"
            );
        }
    }
    Ok(index)
}
fn vector_rows(db: &Connection, model: &str) -> Result<Vec<(String, Vec<u8>)>> {
    let mut q = db.prepare("SELECT hash,model,vector FROM vectors ORDER BY hash")?;
    let rows = q.query_map([], |r| {
        Ok((
            r.get::<_, String>(0)?,
            r.get::<_, String>(1)?,
            r.get::<_, Vec<u8>>(2)?,
        ))
    })?;
    let mut output = Vec::new();
    for row in rows {
        let (hash, version, blob) = row?;
        ensure!(
            version == model
                && blob.len() == 768 * 4
                && blob
                    .chunks_exact(4)
                    .all(|b| f32::from_le_bytes(b.try_into().unwrap()).is_finite()),
            "Snapshot contains an invalid or incompatible document vector"
        );
        output.push((hash, blob));
    }
    Ok(output)
}
fn index_documents(db: &Connection) -> Result<BTreeMap<String, String>> {
    let mut q = db.prepare("SELECT path,sha256,status FROM files WHERE status<>'missing'")?;
    let rows = q.query_map([], |r| {
        Ok((
            r.get::<_, String>(0)?,
            r.get::<_, String>(1)?,
            r.get::<_, String>(2)?,
        ))
    })?;
    let mut result = BTreeMap::new();
    for row in rows {
        let (path, hash, status) = row?;
        ensure!(
            status == "ready" || status == "metadata-only",
            "Document index has pending or failed files"
        );
        result.insert(path, hash);
    }
    Ok(result)
}
fn references(db: &Connection) -> Result<BTreeSet<String>> {
    let mut q = db.prepare("SELECT DISTINCT hash FROM chunks")?;
    Ok(q.query_map([], |r| r.get(0))?
        .collect::<rusqlite::Result<_>>()?)
}
fn create_private(path: &Path) -> Result<()> {
    fs::create_dir(path)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(path, fs::Permissions::from_mode(0o700))?;
    }
    Ok(())
}
fn private_file(path: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(path, fs::Permissions::from_mode(0o600))?;
    }
    Ok(())
}

pub fn export<F: Fn() -> Result<String>>(
    cfg: &native_docs::Settings<'_>,
    db_path: &Path,
    skill_path: &Path,
    dest: &Path,
    fingerprint: F,
) -> Result<()> {
    ensure!(
        fingerprint()? == cfg.metadata_fingerprint,
        "Configuration changed before export"
    );
    ensure!(!dest.exists(), "Snapshot destination already exists");
    let db = Connection::open_with_flags(db_path, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
    let documents = inventory(cfg)?;
    ensure!(
        index_documents(&db)? == documents,
        "Document index differs from files; refresh first"
    );
    let metadata: Option<String> = db
        .query_row(
            "SELECT value FROM state WHERE key='registry_sha'",
            [],
            |r| r.get(0),
        )
        .optional()?;
    ensure!(
        metadata.as_deref() == Some(cfg.metadata_fingerprint),
        "Document metadata is stale"
    );
    let skills = skill_hashes(cfg.root)?;
    verify_skill_index(skill_path, cfg, &skills)?;
    let chunks = references(&db)?;
    let vectors = vector_rows(&db, cfg.model_id)?;
    ensure!(
        chunks == vectors.iter().map(|r| r.0.clone()).collect(),
        "Document vectors are incomplete or contain stale entries"
    );
    create_private(dest)?;
    let create = (|| -> Result<Manifest> {
        let output_path = dest.join("vectors.sqlite3");
        let mut output = Connection::open(&output_path)?;
        output.execute_batch(
            "CREATE TABLE vectors(hash TEXT, model TEXT, vector BLOB, PRIMARY KEY(hash,model))",
        )?;
        {
            let tx = output.transaction()?;
            for (hash, blob) in &vectors {
                tx.execute(
                    "INSERT INTO vectors VALUES (?,?,?)",
                    params![hash, cfg.model_id, blob],
                )?;
            }
            tx.commit()?;
        }
        drop(output);
        private_file(&output_path)?;
        let skill_dest = dest.join("skills.json");
        fs::copy(skill_path, &skill_dest)?;
        private_file(&skill_dest)?;
        ensure!(
            inventory(cfg)? == documents
                && skill_hashes(cfg.root)? == skills
                && fingerprint()? == cfg.metadata_fingerprint,
            "Sources changed during export; retry"
        );
        let files = BTreeMap::from([
            (
                "vectors.sqlite3".to_owned(),
                native_docs::hash_file(&output_path)?,
            ),
            (
                "skills.json".to_owned(),
                native_docs::hash_file(&skill_dest)?,
            ),
        ]);
        let manifest = Manifest {
            format: 2,
            document_model: cfg.model_id.into(),
            skill_model: cfg.model_id.into(),
            settings_sha256: cfg.metadata_fingerprint.into(),
            documents,
            skills,
            chunks,
            files,
        };
        let path = dest.join("manifest.json");
        fs::write(&path, serde_json::to_vec(&manifest)?)?;
        private_file(&path)?;
        Ok(manifest)
    })();
    match create {
        Ok(manifest) => {
            println!(
                "{}",
                serde_json::json!({"documents":manifest.documents.len(),"skills":manifest.skills.len(),"snapshot":dest})
            );
            Ok(())
        }
        Err(e) => {
            let _ = fs::remove_dir_all(dest);
            Err(e)
        }
    }
}

pub fn import<F: Fn() -> Result<String>>(
    cfg: &native_docs::Settings<'_>,
    db_path: &Path,
    skill_path: &Path,
    src: &Path,
    fingerprint: F,
) -> Result<()> {
    ensure!(
        fingerprint()? == cfg.metadata_fingerprint,
        "Configuration changed before import"
    );
    let value: serde_json::Value = serde_json::from_slice(&fs::read(src.join("manifest.json"))?)?;
    ensure!(
        value["format"].as_u64() == Some(2),
        "Snapshot format or model is incompatible; legacy MLX vectors cannot be imported"
    );
    let manifest: Manifest = serde_json::from_value(value)?;
    ensure!(
        manifest.document_model == cfg.model_id && manifest.skill_model == cfg.model_id,
        "Snapshot format or model is incompatible"
    );
    for name in ["vectors.sqlite3", "skills.json"] {
        ensure!(
            manifest.files.get(name) == Some(&native_docs::hash_file(&src.join(name))?),
            "Snapshot checksum mismatch: {name}"
        );
    }
    ensure!(
        manifest.settings_sha256 == cfg.metadata_fingerprint
            && manifest.documents == inventory(cfg)?
            && manifest.skills == skill_hashes(cfg.root)?,
        "Local files or configuration differ from snapshot"
    );
    let index = verify_skill_index(&src.join("skills.json"), cfg, &manifest.skills)?;
    let source = Connection::open_with_flags(
        src.join("vectors.sqlite3"),
        OpenFlags::SQLITE_OPEN_READ_ONLY,
    )?;
    let integrity: String = source.query_row("PRAGMA integrity_check", [], |r| r.get(0))?;
    ensure!(integrity == "ok", "Snapshot SQLite integrity check failed");
    let rows = vector_rows(&source, cfg.model_id)?;
    ensure!(
        manifest.chunks == rows.iter().map(|r| r.0.clone()).collect(),
        "Snapshot is missing document vectors"
    );
    ensure!(
        manifest.documents == inventory(cfg)?
            && manifest.skills == skill_hashes(cfg.root)?
            && fingerprint()? == cfg.metadata_fingerprint,
        "Sources changed during import; retry"
    );
    if skill_path.exists() {
        native_skills::read_index(skill_path, cfg.model_id)?;
    }
    let mut target = Connection::open(db_path)?;
    native_docs::initialize(&target, cfg.model_id)?;
    let tx = target.transaction()?;
    for (hash, blob) in &rows {
        tx.execute(
            "INSERT OR IGNORE INTO vectors VALUES (?,?,?)",
            params![hash, cfg.model_id, blob],
        )?;
    }
    // Both generations represent the same model and identical source hashes;
    // publishing a matching skill index before SQLite commit cannot mix models.
    native_skills::atomic_write(skill_path, &index)?;
    tx.commit()?;
    println!(
        "{}",
        serde_json::json!({"vectors_available":rows.len(),"skills":index.skills.len()})
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native_docs::Encoder;
    use globset::{Glob, GlobSetBuilder};
    struct Fake;
    impl Encoder for Fake {
        fn embed(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            Ok(texts.iter().map(|_| vec![1.0; 768]).collect())
        }
    }

    #[test]
    fn portable_snapshot_reuses_vectors_and_rejects_changes_without_mutating_target() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().canonicalize().unwrap();
        git2::Repository::init(&root).unwrap();
        fs::create_dir_all(root.join("library")).unwrap();
        fs::write(root.join("library/a.md"), "# A\nbody").unwrap();
        fs::create_dir_all(root.join("skills/example")).unwrap();
        fs::write(root.join("skills/example/SKILL.md"), "# Example\nbody").unwrap();
        let roots = vec!["library".to_owned()];
        let mut globs = GlobSetBuilder::new();
        globs.add(Glob::new("**/*.md").unwrap());
        let includes = globs.build().unwrap();
        let excludes = GlobSetBuilder::new().build().unwrap();
        let cfg = native_docs::Settings {
            root: &root,
            roots: &roots,
            includes: &includes,
            excludes: &excludes,
            model_id: "test-model",
            metadata_fingerprint: "metadata",
        };
        let source = root.join("source.sqlite");
        let mut db = Connection::open(&source).unwrap();
        native_docs::scan(&mut db, &cfg, false, || Ok(Box::new(Fake))).unwrap();
        drop(db);
        let skills = root.join("skills-source.json");
        native_skills::scan(&root, &skills, cfg.model_id, || Ok(Box::new(Fake))).unwrap();
        let bundle = root.join("bundle");
        export(&cfg, &source, &skills, &bundle, || Ok("metadata".into())).unwrap();
        let changed = std::sync::atomic::AtomicUsize::new(0);
        let rejected = root.join("config-race-bundle");
        assert!(
            export(&cfg, &source, &skills, &rejected, || Ok(if changed
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
                == 0
            {
                "metadata"
            } else {
                "changed"
            }
            .into()))
            .is_err()
        );
        assert!(!rejected.exists());
        let target = root.join("target.sqlite");
        let target_skills = root.join("skills-target.json");
        import(&cfg, &target, &target_skills, &bundle, || {
            Ok("metadata".into())
        })
        .unwrap();
        let mut db = Connection::open(&target).unwrap();
        let result = native_docs::scan(&mut db, &cfg, true, || {
            anyhow::bail!("no model should load")
        })
        .unwrap();
        assert_eq!(result.embedded, 0);
        let before = db
            .query_row("SELECT count(*) FROM vectors", [], |r| r.get::<_, usize>(0))
            .unwrap();
        drop(db);
        let old_skill_bytes = fs::read(&target_skills).unwrap();
        fs::write(root.join("library/a.md"), "changed").unwrap();
        assert!(
            import(&cfg, &target, &target_skills, &bundle, || Ok(
                "metadata".into()
            ))
            .is_err()
        );
        assert_eq!(
            Connection::open(&target)
                .unwrap()
                .query_row("SELECT count(*) FROM vectors", [], |r| r.get::<_, usize>(0))
                .unwrap(),
            before
        );
        assert_eq!(fs::read(&target_skills).unwrap(), old_skill_bytes);
        fs::write(root.join("library/a.md"), "# A\nbody").unwrap();
        fs::write(bundle.join("skills.json"), "tampered").unwrap();
        assert!(
            import(&cfg, &target, &target_skills, &bundle, || Ok(
                "metadata".into()
            ))
            .is_err()
        );
        assert_eq!(fs::read(&target_skills).unwrap(), old_skill_bytes);
    }
}

/// Export the workspace collections in the existing portable snapshot format.
pub fn export_collections<F: Fn() -> Result<String>>(
    cfg: &native_docs::Settings<'_>,
    db_path: &Path,
    dest: &Path,
    fingerprint: F,
) -> Result<()> {
    use crate::collections;
    ensure!(
        fingerprint()? == cfg.metadata_fingerprint,
        "Configuration changed before export"
    );
    ensure!(!dest.exists(), "Snapshot destination already exists");
    let db = Connection::open_with_flags(db_path, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
    collections::require_workspace_root(&db, "workspace", cfg.root)?;
    collections::require_skill_root(&db, "skills", cfg.root)?;
    collections::require_model(&db, "workspace", cfg.model_id)?;
    collections::require_model(&db, "skills", cfg.model_id)?;
    let metadata: Option<String> = db.query_row("SELECT value FROM collection_state WHERE collection='workspace' AND key='registry_sha'",[],|r|r.get(0)).optional()?;
    ensure!(
        metadata.as_deref() == Some(cfg.metadata_fingerprint),
        "Workspace metadata is stale; scan before exporting"
    );
    let documents = inventory(cfg)?;
    let indexed: BTreeMap<String, String> = {
        let mut q = db.prepare(
            "SELECT path,hash,status FROM files WHERE collection='workspace' AND status<>'missing'",
        )?;
        let mut out = BTreeMap::new();
        for row in q.query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
            ))
        })? {
            let (path, hash, status) = row?;
            ensure!(
                status == "ready" || status == "metadata-only",
                "Workspace has incomplete documents"
            );
            out.insert(path, hash);
        }
        out
    };
    ensure!(
        indexed == documents,
        "Document index differs from files; refresh first"
    );
    let skill_hashes = skill_hashes(cfg.root)?;
    let index = collections::load_skill_index(&db, "skills", cfg.model_id)?;
    let actual_skills: BTreeMap<_, _> = index
        .skills
        .iter()
        .map(|(n, s)| (n.clone(), s.md5.clone()))
        .collect();
    ensure!(
        actual_skills == skill_hashes,
        "Skill index differs from files; refresh first"
    );
    let invalid:usize=db.query_row("SELECT count(*) FROM files f WHERE f.collection='workspace' AND f.status='ready' AND NOT EXISTS(SELECT 1 FROM chunks c WHERE c.collection=f.collection AND c.path=f.path)",[],|r|r.get(0))?;
    ensure!(invalid == 0, "Workspace contains incomplete chunks");
    let mut vectors: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    {
        use sha2::{Digest, Sha256};
        let mut q=db.prepare("SELECT c.source_hash,c.text,c.vector,f.status FROM chunks c JOIN files f ON f.collection=c.collection AND f.path=c.path WHERE c.collection='workspace'")?;
        for row in q.query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, Vec<u8>>(2)?,
                r.get::<_, String>(3)?,
            ))
        })? {
            let (hash, text, vector, status) = row?;
            ensure!(
                status == "ready"
                    && format!("{:x}", Sha256::digest(text.as_bytes())) == hash
                    && vector.len() == 768 * 4
                    && vector
                        .chunks_exact(4)
                        .all(|v| f32::from_le_bytes(v.try_into().unwrap()).is_finite()),
                "Workspace contains an invalid vector or chunk"
            );
            if let Some(old) = vectors.get(&hash) {
                ensure!(old == &vector, "Conflicting vectors for one chunk");
            } else {
                vectors.insert(hash, vector);
            }
        }
    }
    create_private(dest)?;
    let create = (|| -> Result<Manifest> {
        let output_path = dest.join("vectors.sqlite3");
        let mut output = Connection::open(&output_path)?;
        output.execute_batch(
            "CREATE TABLE vectors(hash TEXT,model TEXT,vector BLOB,PRIMARY KEY(hash,model))",
        )?;
        {
            let tx = output.transaction()?;
            for (hash, vector) in &vectors {
                tx.execute(
                    "INSERT INTO vectors VALUES(?,?,?)",
                    params![hash, cfg.model_id, vector],
                )?;
            }
            tx.commit()?;
        }
        drop(output);
        private_file(&output_path)?;
        let skill_dest = dest.join("skills.json");
        native_skills::atomic_write(&skill_dest, &index)?;
        ensure!(
            documents == inventory(cfg)?
                && skill_hashes == self::skill_hashes(cfg.root)?
                && fingerprint()? == cfg.metadata_fingerprint,
            "Sources changed during export; retry"
        );
        let files = BTreeMap::from([
            (
                "vectors.sqlite3".to_owned(),
                native_docs::hash_file(&output_path)?,
            ),
            (
                "skills.json".to_owned(),
                native_docs::hash_file(&skill_dest)?,
            ),
        ]);
        let manifest = Manifest {
            format: 2,
            document_model: cfg.model_id.into(),
            skill_model: cfg.model_id.into(),
            settings_sha256: cfg.metadata_fingerprint.into(),
            documents,
            skills: skill_hashes,
            chunks: vectors.keys().cloned().collect(),
            files,
        };
        fs::write(dest.join("manifest.json"), serde_json::to_vec(&manifest)?)?;
        private_file(&dest.join("manifest.json"))?;
        Ok(manifest)
    })();
    match create {
        Ok(m) => {
            println!(
                "{}",
                serde_json::json!({"documents":m.documents.len(),"skills":m.skills.len(),"snapshot":dest})
            );
            Ok(())
        }
        Err(e) => {
            let _ = fs::remove_dir_all(dest);
            Err(e)
        }
    }
}

/// Import a validated portable snapshot into a collection database. On a fresh
/// receiver, vectors are retained in the cache until the first document scan.
pub fn import_collections<F: Fn() -> Result<String>>(
    cfg: &native_docs::Settings<'_>,
    db_path: &Path,
    src: &Path,
    fingerprint: F,
) -> Result<()> {
    use crate::collections;
    ensure!(
        fingerprint()? == cfg.metadata_fingerprint,
        "Configuration changed before import"
    );
    let manifest: Manifest = serde_json::from_slice(&fs::read(src.join("manifest.json"))?)?;
    ensure!(
        manifest.format == 2
            && manifest.document_model == cfg.model_id
            && manifest.skill_model == cfg.model_id,
        "Snapshot format or model is incompatible"
    );
    for name in ["vectors.sqlite3", "skills.json"] {
        ensure!(
            manifest.files.get(name) == Some(&native_docs::hash_file(&src.join(name))?),
            "Snapshot checksum mismatch: {name}"
        );
    }
    ensure!(
        manifest.settings_sha256 == cfg.metadata_fingerprint
            && manifest.documents == inventory(cfg)?
            && manifest.skills == skill_hashes(cfg.root)?,
        "Local files or configuration differ from snapshot"
    );
    let index = verify_skill_index(&src.join("skills.json"), cfg, &manifest.skills)?;
    let source = Connection::open_with_flags(
        src.join("vectors.sqlite3"),
        OpenFlags::SQLITE_OPEN_READ_ONLY,
    )?;
    let integrity: String = source.query_row("PRAGMA integrity_check", [], |r| r.get(0))?;
    ensure!(integrity == "ok", "Snapshot SQLite integrity check failed");
    let rows = vector_rows(&source, cfg.model_id)?;
    ensure!(
        manifest.chunks == rows.iter().map(|r| r.0.clone()).collect(),
        "Snapshot is missing document vectors"
    );
    ensure!(
        manifest.documents == inventory(cfg)?
            && manifest.skills == skill_hashes(cfg.root)?
            && fingerprint()? == cfg.metadata_fingerprint,
        "Sources changed during import; retry"
    );
    let skill_root = cfg.root.join("skills").canonicalize()?;
    ensure!(
        skill_root == cfg.root.join("skills"),
        "Skill root changed or became a symlink"
    );
    let mut target = collections::open(db_path)?;
    let tx = target.transaction()?;
    tx.execute("INSERT OR IGNORE INTO collections(name,root,include,exclude,result,depth,returns,model) VALUES('workspace',?,'**/*.md','[]','workspace',1,'paths',?)",params![cfg.root.to_str().context("Non-UTF-8 workspace root")?,cfg.model_id])?;
    tx.execute("INSERT OR IGNORE INTO collections(name,root,include,exclude,result,depth,returns,model) VALUES('skills',?,'**/*.md','[]','skill',1,'paths',?)",params![skill_root.to_str().context("Non-UTF-8 skill root")?,cfg.model_id])?;
    collections::require_workspace_root(&tx, "workspace", cfg.root)?;
    collections::require_skill_root(&tx, "skills", cfg.root)?;
    collections::require_model(&tx, "workspace", cfg.model_id)?;
    collections::require_model(&tx, "skills", cfg.model_id)?;
    for (hash, blob) in &rows {
        tx.execute(
            "INSERT OR IGNORE INTO vector_cache(hash,model,vector) VALUES(?,?,?)",
            params![hash, cfg.model_id, blob],
        )?;
    }
    collections::write_skill_index(&tx, "skills", &index, cfg.model_id)?;
    ensure!(
        manifest.documents == inventory(cfg)?
            && manifest.skills == skill_hashes(cfg.root)?
            && fingerprint()? == cfg.metadata_fingerprint,
        "Sources changed during import; retry"
    );
    tx.commit()?;
    println!(
        "{}",
        serde_json::json!({"vectors_available":rows.len(),"skills":index.skills.len()})
    );
    Ok(())
}

#[cfg(test)]
mod collection_tests {
    use super::*;
    use crate::{collections, native_docs::Encoder};
    use globset::{Glob, GlobSetBuilder};
    struct Fake;
    impl Encoder for Fake {
        fn embed(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            Ok(texts.iter().map(|_| vec![0.25; 768]).collect())
        }
    }
    #[test]
    fn collection_snapshot_restores_fresh_receiver_without_reembedding() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().canonicalize().unwrap();
        git2::Repository::init(&root).unwrap();
        fs::create_dir_all(root.join("library")).unwrap();
        fs::write(root.join("library/a.md"), "# A\ncontent").unwrap();
        fs::create_dir_all(root.join("skills/example")).unwrap();
        fs::write(root.join("skills/example/SKILL.md"), "# Skill\ncontent").unwrap();
        let roots = vec!["library".to_owned()];
        let mut builder = GlobSetBuilder::new();
        builder.add(Glob::new("**/*.md").unwrap());
        let includes = builder.build().unwrap();
        let excludes = GlobSetBuilder::new().build().unwrap();
        let cfg = native_docs::Settings {
            root: &root,
            roots: &roots,
            includes: &includes,
            excludes: &excludes,
            model_id: "model",
            metadata_fingerprint: "metadata",
        };
        let mut legacy = Connection::open_in_memory().unwrap();
        native_docs::scan(&mut legacy, &cfg, false, || Ok(Box::new(Fake))).unwrap();
        let old_skills = root.join("old-skills.json");
        native_skills::scan(&root, &old_skills, "model", || Ok(Box::new(Fake))).unwrap();
        let source_path = root.join("collections-source.sqlite");
        let mut source = collections::open(&source_path).unwrap();
        collections::migrate_workspace_documents(&mut source, "workspace", &legacy, &cfg, || {
            Ok("metadata".into())
        })
        .unwrap();
        collections::migrate_workspace_skills(&mut source, "skills", &root, &old_skills, "model")
            .unwrap();
        drop(source);
        let bundle = root.join("bundle");
        export_collections(&cfg, &source_path, &bundle, || Ok("metadata".into())).unwrap();
        let target_path = root.join("collections-target.sqlite");
        import_collections(&cfg, &target_path, &bundle, || Ok("metadata".into())).unwrap();
        let mut target = collections::open(&target_path).unwrap();
        let loaded = collections::load_skill_index(&target, "skills", "model").unwrap();
        assert_eq!(loaded.skills.len(), 1);
        let result = collections::scan_workspace(
            &mut target,
            "workspace",
            &cfg,
            || -> Result<Box<dyn Encoder>> { panic!("cached vectors must avoid model loading") },
            || Ok("metadata".into()),
        )
        .unwrap();
        assert_eq!(result["embedded"], 0);
        assert_eq!(result["seen"], 1);
        let before: Vec<u8> = target
            .query_row(
                "SELECT vector FROM chunks WHERE collection='workspace'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        drop(target);
        fs::write(bundle.join("skills.json"), "tampered").unwrap();
        assert!(import_collections(&cfg, &target_path, &bundle, || Ok("metadata".into())).is_err());
        let target = collections::open(&target_path).unwrap();
        assert_eq!(
            target
                .query_row(
                    "SELECT vector FROM chunks WHERE collection='workspace'",
                    [],
                    |r| r.get::<_, Vec<u8>>(0)
                )
                .unwrap(),
            before
        );
    }
}
