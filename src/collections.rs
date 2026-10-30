//! Independent collection storage. The legacy workspace index is never opened for writing here.
use anyhow::{Context, Result, ensure};
use globset::{Glob, GlobSet, GlobSetBuilder};
use rusqlite::{Connection, OptionalExtension, params};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    fs,
    path::{Component, Path, PathBuf},
};

#[derive(Debug)]
pub struct Collection {
    pub name: String,
    pub root: PathBuf,
    pub include: String,
    pub exclude: Vec<String>,
    pub result: String,
    pub depth: usize,
    pub returns: String,
}
#[derive(Serialize)]
pub struct Hit {
    pub collection: String,
    pub name: String,
    pub path: String,
    pub relative_path: String,
    pub score: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub area: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub category: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub line: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}
fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn glob(patterns: &[String]) -> Result<GlobSet> {
    let mut b = GlobSetBuilder::new();
    for p in patterns {
        b.add(Glob::new(p)?);
    }
    Ok(b.build()?)
}
fn relative(s: &str) -> Result<()> {
    ensure!(
        !Path::new(s).is_absolute()
            && Path::new(s)
                .components()
                .all(|p| matches!(p, Component::Normal(_) | Component::CurDir)),
        "Scope must be a relative path inside the collection"
    );
    Ok(())
}
pub fn open(path: &Path) -> Result<Connection> {
    fs::create_dir_all(path.parent().context("Database has no parent")?)?;
    let mut db = Connection::open(path)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(path, fs::Permissions::from_mode(0o600))?;
    }
    let version: u32 = db.pragma_query_value(None, "user_version", |row| row.get(0))?;
    ensure!(
        version <= 3,
        "Collection database schema is newer than this binary"
    );
    db.execute_batch("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;")?;
    if version < 2 {
        let tx = db.transaction()?;
        tx.execute_batch("CREATE TABLE IF NOT EXISTS collections(name TEXT PRIMARY KEY, root TEXT NOT NULL, include TEXT NOT NULL, exclude TEXT NOT NULL, result TEXT NOT NULL, depth INTEGER NOT NULL, returns TEXT NOT NULL, model TEXT); CREATE TABLE IF NOT EXISTS files(collection TEXT NOT NULL, path TEXT NOT NULL, hash TEXT NOT NULL, PRIMARY KEY(collection,path), FOREIGN KEY(collection) REFERENCES collections(name)); CREATE TABLE IF NOT EXISTS chunks(collection TEXT NOT NULL, path TEXT NOT NULL, ordinal INTEGER NOT NULL, text TEXT NOT NULL, vector BLOB NOT NULL, PRIMARY KEY(collection,path,ordinal), FOREIGN KEY(collection,path) REFERENCES files(collection,path) ON DELETE CASCADE);")?;
        tx.execute_batch("ALTER TABLE files ADD COLUMN title TEXT; ALTER TABLE files ADD COLUMN summary TEXT; ALTER TABLE files ADD COLUMN category TEXT; ALTER TABLE files ADD COLUMN status TEXT; ALTER TABLE files ADD COLUMN error TEXT; ALTER TABLE files ADD COLUMN updated TEXT; ALTER TABLE files ADD COLUMN area TEXT; ALTER TABLE files ADD COLUMN kind TEXT; ALTER TABLE chunks ADD COLUMN line INTEGER; ALTER TABLE chunks ADD COLUMN source_hash TEXT; CREATE TABLE skill_indexes(collection TEXT PRIMARY KEY, model TEXT NOT NULL, algorithm TEXT NOT NULL, backend TEXT NOT NULL, model_name TEXT NOT NULL, FOREIGN KEY(collection) REFERENCES collections(name) ON DELETE CASCADE); CREATE TABLE skills(collection TEXT NOT NULL, name TEXT NOT NULL, hash TEXT NOT NULL, kind TEXT NOT NULL, area TEXT NOT NULL, path TEXT NOT NULL, category TEXT NOT NULL, PRIMARY KEY(collection,name), FOREIGN KEY(collection) REFERENCES skill_indexes(collection) ON DELETE CASCADE); CREATE TABLE skill_chunks(collection TEXT NOT NULL, skill TEXT NOT NULL, ordinal INTEGER NOT NULL, file TEXT NOT NULL, heading TEXT NOT NULL, text TEXT NOT NULL, vector BLOB NOT NULL, PRIMARY KEY(collection,skill,ordinal), FOREIGN KEY(collection,skill) REFERENCES skills(collection,name) ON DELETE CASCADE); PRAGMA user_version=2;")?;
        tx.commit()?;
    }
    if version < 3 {
        let tx = db.transaction()?;
        tx.execute_batch("CREATE TABLE collection_state(collection TEXT NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL, PRIMARY KEY(collection,key), FOREIGN KEY(collection) REFERENCES collections(name) ON DELETE CASCADE); CREATE TABLE vector_cache(hash TEXT NOT NULL, model TEXT NOT NULL, vector BLOB NOT NULL, PRIMARY KEY(hash,model)); PRAGMA user_version=3;")?;
        tx.commit()?;
    }
    Ok(db)
}
pub fn add(db: &Connection, c: &Collection, protected: &Path) -> Result<()> {
    ensure!(
        !c.name.is_empty()
            && c.name
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_'),
        "Invalid collection name"
    );
    ensure!(c.depth > 0, "Directory depth must be positive");
    glob(&[c.include.clone()])?;
    glob(&c.exclude)?;
    let root = c
        .root
        .canonicalize()
        .context("Collection directory unavailable")?;
    ensure!(root.is_dir(), "Collection root must be a directory");
    ensure!(
        !root.starts_with(protected) && !protected.starts_with(&root),
        "Collection overlaps the protected workspace; use the existing workspace index"
    );
    db.execute("INSERT INTO collections(name,root,include,exclude,result,depth,returns) VALUES(?,?,?,?,?,?,?)", params![c.name,root.to_string_lossy(),c.include,serde_json::to_string(&c.exclude)?,c.result,c.depth,c.returns])?;
    Ok(())
}
pub fn list(db: &Connection) -> Result<Vec<serde_json::Value>> {
    let mut s=db.prepare("SELECT name,root,include,exclude,result,depth,returns,model FROM collections ORDER BY name")?;
    let rows=s.query_map([],|r| Ok(serde_json::json!({"name":r.get::<_,String>(0)?,"root":r.get::<_,String>(1)?,"include":r.get::<_,String>(2)?,"exclude":serde_json::from_str::<Vec<String>>(&r.get::<_,String>(3)?).unwrap_or_default(),"result":r.get::<_,String>(4)?,"depth":r.get::<_,usize>(5)?,"return":r.get::<_,String>(6)?,"model":r.get::<_,Option<String>>(7)?})))?;
    Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
}
fn get(db: &Connection, name: &str) -> Result<(Collection, Option<String>)> {
    db.query_row(
        "SELECT root,include,exclude,result,depth,returns,model FROM collections WHERE name=?",
        [name],
        |r| {
            Ok((
                Collection {
                    name: name.into(),
                    root: PathBuf::from(r.get::<_, String>(0)?),
                    include: r.get(1)?,
                    exclude: serde_json::from_str(&r.get::<_, String>(2)?).unwrap_or_default(),
                    result: r.get(3)?,
                    depth: r.get(4)?,
                    returns: r.get(5)?,
                },
                r.get(6)?,
            ))
        },
    )
    .optional()?
    .context("Collection not found")
}
pub fn file_stat(meta: &fs::Metadata) -> (u64, i64, u32) {
    let len = meta.len();
    let (secs, nanos) = match meta.modified() {
        Ok(t) => match t.duration_since(std::time::UNIX_EPOCH) {
            Ok(d) => (d.as_secs() as i64, d.subsec_nanos()),
            Err(e) => (
                -(e.duration().as_secs() as i64),
                e.duration().subsec_nanos(),
            ),
        },
        Err(_) => (0, 0),
    };
    (len, secs, nanos)
}

pub fn compute_stat_fingerprint(stats: &[(String, u64, i64, u32)]) -> String {
    let mut items: Vec<&(String, u64, i64, u32)> = stats.iter().collect();
    items.sort_by(|a, b| a.0.cmp(&b.0));
    let mut hasher = Sha256::new();
    for (rel, len, sec, nano) in items {
        hasher.update(rel.as_bytes());
        hasher.update(&[0]);
        hasher.update(&len.to_le_bytes());
        hasher.update(&sec.to_le_bytes());
        hasher.update(&nano.to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

fn inventory_stats(
    root: &Path,
    includes: &GlobSet,
    excludes: &GlobSet,
) -> Result<Vec<(String, u64, i64, u32)>> {
    let mut result = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in fs::read_dir(&dir).with_context(|| format!("Cannot read {}", dir.display()))? {
            let entry = entry?;
            let path = entry.path();
            let ty = entry.file_type()?;
            if ty.is_symlink() {
                continue;
            }
            let rel = path
                .strip_prefix(root)?
                .to_str()
                .context("Non UTF-8 path")?
                .replace('\\', "/");
            if excludes.is_match(&rel) {
                continue;
            }
            if ty.is_dir() {
                if !entry.file_name().to_string_lossy().starts_with('.') {
                    stack.push(path);
                }
            } else if ty.is_file() && includes.is_match(&rel) {
                let meta = entry.metadata()?;
                let (len, sec, nano) = file_stat(&meta);
                result.push((rel, len, sec, nano));
            }
        }
    }
    result.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(result)
}

fn inventory(
    root: &Path,
    includes: &GlobSet,
    excludes: &GlobSet,
) -> Result<Vec<(String, String, String)>> {
    let mut result = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in fs::read_dir(&dir).with_context(|| format!("Cannot read {}", dir.display()))? {
            let entry = entry?;
            let path = entry.path();
            let ty = entry.file_type()?;
            if ty.is_symlink() {
                continue;
            }
            let rel = path
                .strip_prefix(root)?
                .to_str()
                .context("Non UTF-8 path")?
                .replace('\\', "/");
            if excludes.is_match(&rel) {
                continue;
            }
            if ty.is_dir() {
                if !entry.file_name().to_string_lossy().starts_with('.') {
                    stack.push(path)
                }
            } else if ty.is_file() && includes.is_match(&rel) {
                ensure!(
                    fs::metadata(&path)?.len() <= 16 * 1024 * 1024,
                    "File too large: {rel}"
                );
                let bytes = fs::read(&path)?;
                let hash = digest(&bytes);
                let text = String::from_utf8(bytes)
                    .with_context(|| format!("File is not UTF-8 text: {rel}"))?;
                ensure!(!text.contains('\0'), "Binary file rejected: {rel}");
                result.push((rel, hash, text));
            }
        }
    }
    result.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(result)
}
pub fn scan<F>(
    db: &mut Connection,
    name: &str,
    model: &str,
    mut embed: F,
) -> Result<serde_json::Value>
where
    F: FnMut(&[String]) -> Result<Vec<Vec<f32>>>,
{
    let (c, old_model) = get(db, name)?;
    ensure!(
        c.result != "workspace",
        "Workspace collection requires its protected scanner"
    );
    ensure!(
        old_model.as_deref().is_none_or(|m| m == model),
        "Collection model changed; use a separate database"
    );
    ensure!(
        c.root.canonicalize()? == c.root,
        "Collection root changed or became a symlink"
    );
    let inc_glob = glob(&[c.include.clone()])?;
    let exc_glob = glob(&c.exclude)?;
    let stats = inventory_stats(&c.root, &inc_glob, &exc_glob)?;
    let stat_fingerprint = compute_stat_fingerprint(&stats);

    let stored_stat: Option<String> = db
        .query_row(
            "SELECT value FROM collection_state WHERE collection=? AND key='stat_fingerprint'",
            [name],
            |r| r.get(0),
        )
        .optional()?;

    if stored_stat.as_deref() == Some(&stat_fingerprint) && old_model.as_deref() == Some(model) {
        let seen: usize = db.query_row(
            "SELECT count(*) FROM files WHERE collection=?",
            [name],
            |r| r.get(0),
        )?;
        return Ok(serde_json::json!({"collection":name,"seen":seen,"changed":0,"embedded":0}));
    }

    let files = inventory(&c.root, &inc_glob, &exc_glob)?;
    let mut old = HashMap::new();
    {
        let mut s = db.prepare("SELECT path,hash FROM files WHERE collection=?")?;
        for r in s.query_map([name], |r| {
            Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
        })? {
            let (p, h) = r?;
            old.insert(p, h);
        }
    }
    let mut updates = Vec::new();
    let mut count = 0;
    for (path, hash, text) in &files {
        if old.get(path) == Some(hash) {
            continue;
        }
        let chars: Vec<char> = text.chars().collect();
        let chunks: Vec<String> = (0..chars.len())
            .step_by(600)
            .filter_map(|start| {
                let body: String = chars[start..(start + 700).min(chars.len())]
                    .iter()
                    .collect();
                if body.trim().is_empty() {
                    None
                } else {
                    Some(format!(
                        "title: {} | text: {}\n{}",
                        Path::new(path)
                            .file_stem()
                            .unwrap_or_default()
                            .to_string_lossy(),
                        path,
                        body.trim()
                    ))
                }
            })
            .collect();
        let mut vectors = Vec::new();
        for batch in chunks.chunks(32) {
            let out = embed(batch)?;
            ensure!(out.len() == batch.len(), "Embedding count mismatch");
            for v in out {
                ensure!(
                    !v.is_empty() && v.iter().all(|f| f.is_finite()),
                    "Invalid vector"
                );
                vectors.push(v)
            }
        }
        count += vectors.len();
        updates.push((path.clone(), hash.clone(), chunks, vectors));
    }
    let tx = db.transaction()?;
    for (path, hash, chunks, vectors) in &updates {
        tx.execute(
            "DELETE FROM files WHERE collection=? AND path=?",
            params![name, path],
        )?;
        tx.execute(
            "INSERT INTO files(collection,path,hash) VALUES(?,?,?)",
            params![name, path, hash],
        )?;
        for (i, (text, v)) in chunks.iter().zip(vectors).enumerate() {
            let bytes: Vec<u8> = v.iter().flat_map(|f| f.to_le_bytes()).collect();
            tx.execute(
                "INSERT INTO chunks(collection,path,ordinal,text,vector) VALUES(?,?,?,?,?)",
                params![name, path, i, text, bytes],
            )?;
        }
    }
    for path in old
        .keys()
        .filter(|p| !files.iter().any(|(f, _, _)| f == *p))
    {
        tx.execute(
            "DELETE FROM files WHERE collection=? AND path=?",
            params![name, path],
        )?;
    }
    tx.execute(
        "UPDATE collections SET model=? WHERE name=?",
        params![model, name],
    )?;
    tx.execute(
        "INSERT OR REPLACE INTO collection_state(collection,key,value) VALUES(?,'stat_fingerprint',?)",
        params![name, stat_fingerprint],
    )?;
    tx.commit()?;
    Ok(
        serde_json::json!({"collection":name,"seen":files.len(),"changed":updates.len(),"embedded":count}),
    )
}
pub fn search(
    db: &Connection,
    name: &str,
    scope: &str,
    returns: Option<&str>,
    limit: usize,
    model: &str,
    q: &[f32],
    eligible: Option<&dyn Fn(&str) -> bool>,
) -> Result<Vec<Hit>> {
    relative(scope)?;
    ensure!(limit > 0, "Limit must be positive");
    let (c, index_model) = get(db, name)?;
    ensure!(
        c.result != "workspace" || eligible.is_some(),
        "Workspace searches require the protected filter"
    );
    ensure!(
        index_model.as_deref() == Some(model),
        "Collection is not indexed with this model"
    );
    let mode = returns.unwrap_or(&c.returns);
    let mut best: HashMap<String, Hit> = HashMap::new();
    let mut s = db.prepare("SELECT c.path,c.text,c.vector,c.line,f.title,f.summary,f.category,f.area,f.kind FROM chunks c JOIN files f ON f.collection=c.collection AND f.path=c.path WHERE c.collection=? AND (f.status IS NULL OR f.status='ready')")?;
    for row in s.query_map([name], |r| {
        Ok((
            r.get::<_, String>(0)?,
            r.get::<_, String>(1)?,
            r.get::<_, Vec<u8>>(2)?,
            r.get::<_, Option<usize>>(3)?,
            r.get::<_, Option<String>>(4)?,
            r.get::<_, Option<String>>(5)?,
            r.get::<_, Option<String>>(6)?,
            r.get::<_, Option<String>>(7)?,
            r.get::<_, Option<String>>(8)?,
        ))
    })? {
        let (path, text, bytes, line, title, summary, category, area, kind) = row?;
        if !super::under(&path, scope) || eligible.is_some_and(|filter| !filter(&path)) {
            continue;
        }
        ensure!(bytes.len() % 4 == 0, "Malformed vector");
        let v: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let score = super::cosine(q, &v)?;
        let key = if c.result == "directory" {
            let parts: Vec<_> = path.split('/').collect();
            if parts.len() <= c.depth {
                continue;
            }
            parts[..c.depth].join("/")
        } else {
            path.clone()
        };
        let content = match mode {
            "paths" => None,
            "snippets" => Some(text),
            "full" => {
                ensure!(
                    c.result == "document",
                    "Full content is only available for document results"
                );
                let actual = c.root.join(&path).canonicalize()?;
                ensure!(actual.starts_with(&c.root), "Content escaped collection");
                Some(fs::read_to_string(actual)?)
            }
            _ => anyhow::bail!("Invalid return mode"),
        };
        let hit = Hit {
            collection: name.into(),
            name: title.filter(|s| !s.is_empty()).unwrap_or_else(|| {
                Path::new(&key)
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .into_owned()
            }),
            path: c.root.join(&key).display().to_string(),
            relative_path: key.clone(),
            score,
            area,
            category,
            summary,
            kind,
            line: if mode == "paths" { None } else { line },
            content,
        };
        if best.get(&key).is_none_or(|h| score > h.score) {
            best.insert(key, hit);
        }
    }
    let mut hits: Vec<_> = best.into_values().collect();
    hits.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.relative_path.cmp(&b.relative_path))
    });
    hits.truncate(limit);
    Ok(hits)
}

#[cfg(test)]
mod schema_tests {
    use super::*;
    #[test]
    fn upgrades_unversioned_database_without_losing_existing_vectors() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("collections.sqlite3");
        let db = Connection::open(&path).unwrap();
        db.execute_batch("CREATE TABLE collections(name TEXT PRIMARY KEY, root TEXT NOT NULL, include TEXT NOT NULL, exclude TEXT NOT NULL, result TEXT NOT NULL, depth INTEGER NOT NULL, returns TEXT NOT NULL, model TEXT); CREATE TABLE files(collection TEXT NOT NULL, path TEXT NOT NULL, hash TEXT NOT NULL, PRIMARY KEY(collection,path), FOREIGN KEY(collection) REFERENCES collections(name)); CREATE TABLE chunks(collection TEXT NOT NULL, path TEXT NOT NULL, ordinal INTEGER NOT NULL, text TEXT NOT NULL, vector BLOB NOT NULL, PRIMARY KEY(collection,path,ordinal), FOREIGN KEY(collection,path) REFERENCES files(collection,path) ON DELETE CASCADE); INSERT INTO collections VALUES('notes','/tmp','**/*.md','[]','document',1,'paths','test'); INSERT INTO files VALUES('notes','a.md','hash'); INSERT INTO chunks VALUES('notes','a.md',0,'text',x'0000803f');").unwrap();
        drop(db);
        let upgraded = open(&path).unwrap();
        assert_eq!(
            upgraded
                .pragma_query_value(None, "user_version", |r| r.get::<_, u32>(0))
                .unwrap(),
            3
        );
        assert_eq!(
            upgraded
                .query_row("SELECT hex(vector) FROM chunks", [], |r| r
                    .get::<_, String>(0))
                .unwrap(),
            "0000803F"
        );
        assert_eq!(
            upgraded
                .query_row(
                    "SELECT count(*) FROM pragma_table_info('chunks') WHERE name='line'",
                    [],
                    |r| r.get::<_, u32>(0)
                )
                .unwrap(),
            1
        );
        assert_eq!(
            upgraded
                .query_row("PRAGMA integrity_check", [], |r| r.get::<_, String>(0))
                .unwrap(),
            "ok"
        );
        drop(upgraded);
        open(&path).unwrap();
    }
    #[test]
    fn rejects_future_schema_without_modifying_it() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("collections.sqlite3");
        let db = Connection::open(&path).unwrap();
        db.execute_batch("PRAGMA user_version=99; CREATE TABLE future(value TEXT); INSERT INTO future VALUES('keep');").unwrap();
        drop(db);
        assert!(open(&path).is_err());
        let db = Connection::open(path).unwrap();
        assert_eq!(
            db.query_row("SELECT value FROM future", [], |r| r.get::<_, String>(0))
                .unwrap(),
            "keep"
        );
    }
    #[test]
    fn skill_index_round_trip_and_rejected_replacement_preserve_vectors() {
        let dir = tempfile::tempdir().unwrap();
        let mut db = open(&dir.path().join("index.sqlite3")).unwrap();
        db.execute("INSERT INTO collections VALUES('skills','/tmp/skills','**/*.md','[]','skill',1,'paths','model')", []).unwrap();
        let index = crate::native_skills::Index {
            backend: "mlx-v2".into(),
            model: "model".into(),
            model_id: "model".into(),
            algorithm: "skill-v1-md5-sections-1000-max48".into(),
            skills: std::collections::BTreeMap::from([(
                "example".into(),
                crate::native_skills::Skill {
                    md5: "hash".into(),
                    kind: "skill".into(),
                    area: "skills".into(),
                    path: "skills/example".into(),
                    category: "skills".into(),
                    chunks: vec![crate::native_skills::Chunk {
                        file: "SKILL.md".into(),
                        heading: "intro".into(),
                        text: "instructions".into(),
                        vector: vec![0.25; 768],
                    }],
                },
            )]),
        };
        store_skill_index(&mut db, "skills", &index, "model").unwrap();
        let loaded = load_skill_index(&db, "skills", "model").unwrap();
        assert_eq!(
            serde_json::to_value(&loaded).unwrap(),
            serde_json::to_value(&index).unwrap()
        );
        assert!(store_skill_index(&mut db, "skills", &index, "other").is_err());
        assert_eq!(
            serde_json::to_value(load_skill_index(&db, "skills", "model").unwrap()).unwrap(),
            serde_json::to_value(index).unwrap()
        );
    }
    #[test]
    fn workspace_skill_migration_validates_sources_and_preserves_rollback() {
        use crate::native_docs::Encoder;
        struct Fake;
        impl Encoder for Fake {
            fn embed(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
                Ok(texts.iter().map(|_| vec![0.25; 768]).collect())
            }
        }
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().canonicalize().unwrap();
        git2::Repository::init(&root).unwrap();
        let skills = root.join("skills");
        fs::create_dir_all(skills.join("alpha")).unwrap();
        fs::write(skills.join("alpha/SKILL.md"), "# Alpha\nfirst").unwrap();
        let (old, _) =
            crate::native_skills::build_collection(&skills, None, "model", || Ok(Box::new(Fake)))
                .unwrap();
        let source = root.join("old.json");
        crate::native_skills::atomic_write(&source, &old).unwrap();
        let mut db = open(&root.join("new.sqlite3")).unwrap();
        assert_eq!(
            migrate_workspace_skills(&mut db, "skills", &root, &source, "model").unwrap(),
            1
        );
        assert_eq!(
            serde_json::to_value(load_skill_index(&db, "skills", "model").unwrap()).unwrap(),
            serde_json::to_value(&old).unwrap()
        );
        assert_eq!(
            scan_skills(
                &mut db,
                "skills",
                "model",
                || -> Result<Box<dyn Encoder>> { panic!("unchanged skill must not load model") }
            )
            .unwrap()["embedded"],
            0
        );
        fs::write(skills.join("alpha/SKILL.md"), "# Alpha\nchanged").unwrap();
        assert_eq!(
            scan_skills(&mut db, "skills", "model", || Ok(Box::new(Fake))).unwrap()["changed"],
            1
        );
        assert_eq!(
            serde_json::to_value(crate::native_skills::read_index(&source, "model").unwrap())
                .unwrap(),
            serde_json::to_value(&old).unwrap()
        );
        assert!(migrate_workspace_skills(&mut db, "second", &root, &source, "model").is_err());
        assert_eq!(
            db.query_row(
                "SELECT count(*) FROM collections WHERE name='second'",
                [],
                |r| r.get::<_, i64>(0)
            )
            .unwrap(),
            0
        );
    }
    #[test]
    fn workspace_document_migration_preserves_metadata_and_rejects_stale_sources() {
        use crate::native_docs::{Encoder, Settings};
        struct Fake;
        impl Encoder for Fake {
            fn embed(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
                Ok(texts.iter().map(|_| vec![0.125; 768]).collect())
            }
        }
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().canonicalize().unwrap();
        fs::create_dir_all(root.join("library")).unwrap();
        fs::create_dir_all(root.join("archive")).unwrap();
        let path = root.join("library/a.md");
        fs::write(
            &path,
            format!("# Title\nsummary: Example\n{}", "line\n".repeat(180)),
        )
        .unwrap();
        fs::write(root.join("library/not.txt"), "not indexed").unwrap();
        fs::write(root.join("archive/old.md"), "not indexed").unwrap();
        let roots = vec!["library".to_owned()];
        let includes = glob(&["**/*.md".to_owned()]).unwrap();
        let excludes = glob(&[]).unwrap();
        let cfg = Settings {
            root: &root,
            roots: &roots,
            includes: &includes,
            excludes: &excludes,
            model_id: "model",
            metadata_fingerprint: "fingerprint",
        };
        let mut source = Connection::open_in_memory().unwrap();
        crate::native_docs::scan(&mut source, &cfg, false, || Ok(Box::new(Fake))).unwrap();
        let mut target = open(&root.join("target.sqlite3")).unwrap();
        let migrated = migrate_workspace_documents(&mut target, "workspace", &source, &cfg, || {
            Ok("fingerprint".into())
        })
        .unwrap();
        assert_eq!(migrated["files"], 1);
        let old: (String, String, String) = source
            .query_row("SELECT title,summary,category FROM files", [], |r| {
                Ok((r.get(0)?, r.get(1)?, r.get(2)?))
            })
            .unwrap();
        let new: (String, String, String) = target
            .query_row("SELECT title,summary,category FROM files", [], |r| {
                Ok((r.get(0)?, r.get(1)?, r.get(2)?))
            })
            .unwrap();
        assert_eq!(old, new);
        let source_chunks: Vec<(i64, String, Vec<u8>)> = {
            let mut q=source.prepare("SELECT c.line,c.hash,v.vector FROM chunks c JOIN vectors v ON v.hash=c.hash ORDER BY c.ordinal").unwrap();
            q.query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))
                .unwrap()
                .collect::<rusqlite::Result<_>>()
                .unwrap()
        };
        let dest_chunks: Vec<(i64, String, Vec<u8>)> = {
            let mut q = target
                .prepare("SELECT line,source_hash,vector FROM chunks ORDER BY ordinal")
                .unwrap();
            q.query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))
                .unwrap()
                .collect::<rusqlite::Result<_>>()
                .unwrap()
        };
        assert_eq!(source_chunks, dest_chunks);
        assert!(source_chunks.len() > 1 && source_chunks[1].0 > 1);
        let noop = scan_workspace(
            &mut target,
            "workspace",
            &cfg,
            || -> Result<Box<dyn Encoder>> {
                panic!("unchanged workspace must not load the model")
            },
            || Ok("fingerprint".into()),
        )
        .unwrap();
        assert_eq!(noop["changed"], 0);
        assert_eq!(noop["embedded"], 0);
        fs::write(&path, "# changed").unwrap();
        assert!(
            migrate_workspace_documents(&mut target, "stale", &source, &cfg, || Ok(
                "fingerprint".into()
            ))
            .is_err()
        );
        assert_eq!(
            target
                .query_row(
                    "SELECT count(*) FROM collections WHERE name='stale'",
                    [],
                    |r| r.get::<_, i64>(0)
                )
                .unwrap(),
            0
        );
        assert_eq!(
            source
                .query_row("SELECT title FROM files", [], |r| r.get::<_, String>(0))
                .unwrap(),
            "Title"
        );
        assert!(
            scan_workspace(
                &mut target,
                "workspace",
                &cfg,
                || -> Result<Box<dyn Encoder>> { anyhow::bail!("model unavailable") },
                || Ok("fingerprint".into())
            )
            .is_err()
        );
        assert_eq!(
            target
                .query_row(
                    "SELECT title FROM files WHERE collection='workspace'",
                    [],
                    |r| r.get::<_, String>(0)
                )
                .unwrap(),
            "Title"
        );
        let updated = scan_workspace(
            &mut target,
            "workspace",
            &cfg,
            || Ok(Box::new(Fake)),
            || Ok("fingerprint".into()),
        )
        .unwrap();
        assert_eq!(updated["changed"], 1);
        assert_eq!(updated["embedded"], 1);
        assert_eq!(
            target
                .query_row(
                    "SELECT title FROM files WHERE collection='workspace'",
                    [],
                    |r| r.get::<_, String>(0)
                )
                .unwrap(),
            "changed"
        );
        assert_eq!(target.query_row("SELECT count(*) FROM files WHERE collection='workspace' AND (lower(path) NOT LIKE '%.md' OR lower(path) LIKE 'archive/%')",[],|r|r.get::<_,i64>(0)).unwrap(),0);
        fs::remove_file(&path).unwrap();
        let removed = scan_workspace(
            &mut target,
            "workspace",
            &cfg,
            || -> Result<Box<dyn Encoder>> { panic!("deletion must not load the model") },
            || Ok("fingerprint".into()),
        )
        .unwrap();
        assert_eq!(removed["removed"], 1);
        let repeat = scan_workspace(
            &mut target,
            "workspace",
            &cfg,
            || -> Result<Box<dyn Encoder>> {
                panic!("unchanged missing file must not load the model")
            },
            || Ok("fingerprint".into()),
        )
        .unwrap();
        assert_eq!(repeat["removed"], 0);
    }
}

/// Replace a complete skill snapshot atomically after validating its model and vectors.
#[allow(dead_code)]
#[cfg(any(test, feature = "metal-eval", feature = "test-backend"))]
pub fn store_skill_index(
    db: &mut Connection,
    collection: &str,
    index: &crate::native_skills::Index,
    model: &str,
) -> Result<()> {
    let tx = db.transaction()?;
    write_skill_index(&tx, collection, index, model)?;
    tx.commit()?;
    Ok(())
}

#[cfg(any(test, feature = "metal-eval", feature = "test-backend"))]
pub(crate) fn write_skill_index(
    tx: &Connection,
    collection: &str,
    index: &crate::native_skills::Index,
    model: &str,
) -> Result<()> {
    ensure!(index.model_id == model, "Skill model mismatch");
    ensure!(
        index.algorithm == "skill-v1-md5-sections-1000-max48",
        "Unsupported skill chunking algorithm"
    );
    let (settings, _) = get(tx, collection)?;
    ensure!(
        settings.result == "skill",
        "Collection is not in skill mode"
    );
    let root_name = settings
        .root
        .file_name()
        .and_then(|s| s.to_str())
        .context("Invalid skill collection root")?;
    for skill in index.skills.values() {
        ensure!(
            skill.path.starts_with(&format!("{root_name}/"))
                && !skill
                    .path
                    .split('/')
                    .any(|p| p == ".." || p.starts_with('.')),
            "Invalid skill root"
        );
        for chunk in &skill.chunks {
            ensure!(
                chunk.vector.len() == 768 && chunk.vector.iter().all(|v| v.is_finite()),
                "Invalid skill vector"
            );
        }
    }
    tx.execute("DELETE FROM skill_indexes WHERE collection=?", [collection])?;
    tx.execute("INSERT INTO skill_indexes(collection,model,algorithm,backend,model_name) VALUES(?,?,?,?,?)", params![collection,index.model_id,index.algorithm,index.backend,index.model])?;
    for (name, skill) in &index.skills {
        tx.execute("INSERT INTO skills(collection,name,hash,kind,area,path,category) VALUES(?,?,?,?,?,?,?)", params![collection,name,skill.md5,skill.kind,skill.area,skill.path,skill.category])?;
        for (ordinal, chunk) in skill.chunks.iter().enumerate() {
            let vector: Vec<u8> = chunk.vector.iter().flat_map(|v| v.to_le_bytes()).collect();
            tx.execute("INSERT INTO skill_chunks(collection,skill,ordinal,file,heading,text,vector) VALUES(?,?,?,?,?,?,?)",params![collection,name,ordinal,chunk.file,chunk.heading,chunk.text,vector])?;
        }
    }
    tx.execute(
        "UPDATE collections SET model=? WHERE name=?",
        params![model, collection],
    )?;
    Ok(())
}

#[cfg(any(test, feature = "metal-eval", feature = "test-backend"))]
pub fn load_skill_index(
    db: &Connection,
    collection: &str,
    model: &str,
) -> Result<crate::native_skills::Index> {
    use crate::native_skills::{Chunk, Index, Skill};
    let (model_id, algorithm, backend, model_name): (String, String, String, String) = db
        .query_row(
            "SELECT model,algorithm,backend,model_name FROM skill_indexes WHERE collection=?",
            [collection],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
        )?;
    ensure!(
        model_id == model && algorithm == "skill-v1-md5-sections-1000-max48",
        "Incompatible skill snapshot"
    );
    let mut skills = std::collections::BTreeMap::new();
    let mut statement = db.prepare(
        "SELECT name,hash,kind,area,path,category FROM skills WHERE collection=? ORDER BY name",
    )?;
    let rows = statement.query_map([collection], |r| {
        Ok((
            r.get::<_, String>(0)?,
            Skill {
                md5: r.get(1)?,
                kind: r.get(2)?,
                area: r.get(3)?,
                path: r.get(4)?,
                category: r.get(5)?,
                chunks: Vec::new(),
            },
        ))
    })?;
    for row in rows {
        let (name, mut skill) = row?;
        let mut chunks=db.prepare("SELECT file,heading,text,vector FROM skill_chunks WHERE collection=? AND skill=? ORDER BY ordinal")?;
        for row in chunks.query_map(params![collection, name], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
                r.get::<_, Vec<u8>>(3)?,
            ))
        })? {
            let (file, heading, text, bytes) = row?;
            ensure!(bytes.len() == 768 * 4, "Invalid skill vector size");
            let vector: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            ensure!(vector.iter().all(|v| v.is_finite()), "Invalid skill vector");
            skill.chunks.push(Chunk {
                file,
                heading,
                text,
                vector,
            });
        }
        skills.insert(name, skill);
    }
    Ok(Index {
        backend,
        model: model_name,
        model_id,
        algorithm,
        skills,
    })
}

#[cfg(any(test, feature = "metal-eval", feature = "test-backend"))]
pub fn scan_skills<F>(
    db: &mut Connection,
    name: &str,
    model: &str,
    factory: F,
) -> Result<serde_json::Value>
where
    F: FnOnce() -> Result<Box<dyn crate::native_docs::Encoder>>,
{
    let (collection, old_model) = get(db, name)?;
    ensure!(
        collection.result == "skill",
        "Collection is not in skill mode"
    );
    ensure!(
        collection.root.canonicalize()? == collection.root,
        "Collection root changed or became a symlink"
    );
    ensure!(
        old_model.as_deref().is_none_or(|m| m == model),
        "Collection model changed; use a separate database"
    );
    let stats = crate::native_skills::skill_file_stats(&collection.root)?;
    let stat_fingerprint = compute_stat_fingerprint(&stats);

    let stored_stat: Option<String> = db
        .query_row(
            "SELECT value FROM collection_state WHERE collection=? AND key='stat_fingerprint'",
            [name],
            |r| r.get(0),
        )
        .optional()?;

    if stored_stat.as_deref() == Some(&stat_fingerprint) && old_model.as_deref() == Some(model) {
        let count: usize = db.query_row(
            "SELECT count(*) FROM skills WHERE collection=?",
            [name],
            |r| r.get(0),
        )?;
        return Ok(serde_json::json!({"collection":name,"skills":count,"changed":0,"embedded":0}));
    }

    let exists: bool = db.query_row(
        "SELECT EXISTS(SELECT 1 FROM skill_indexes WHERE collection=?)",
        [name],
        |r| r.get(0),
    )?;
    let previous = if exists {
        Some(load_skill_index(db, name, model)?)
    } else {
        None
    };
    let (index, result) =
        crate::native_skills::build_collection(&collection.root, previous, model, factory)?;
    let tx = db.transaction()?;
    write_skill_index(&tx, name, &index, model)?;
    tx.execute(
        "INSERT OR REPLACE INTO collection_state(collection,key,value) VALUES(?,'stat_fingerprint',?)",
        params![name, stat_fingerprint],
    )?;
    tx.commit()?;
    Ok(
        serde_json::json!({"collection":name,"skills":result.skills,"changed":result.changed,"embedded":result.embedded}),
    )
}

#[cfg(any(test, feature = "metal-eval", feature = "test-backend"))]
pub fn search_skills(
    db: &Connection,
    name: &str,
    scope: &str,
    limit: usize,
    model: &str,
    query: &[f32],
) -> Result<Vec<Hit>> {
    relative(scope)?;
    ensure!(limit > 0, "Limit must be positive");
    let (collection, _) = get(db, name)?;
    ensure!(
        collection.result == "skill",
        "Collection is not in skill mode"
    );
    let prefix = collection
        .root
        .file_name()
        .and_then(|s| s.to_str())
        .context("Invalid skill root")?;
    let index = load_skill_index(db, name, model)?;
    let mut hits = Vec::new();
    for (skill_name, skill) in index.skills {
        let rel = skill
            .path
            .strip_prefix(&format!("{prefix}/"))
            .context("Skill path is outside its collection")?;
        if !super::under(rel, scope) {
            continue;
        }
        let actual = collection.root.join(rel);
        if !actual.join("SKILL.md").is_file()
            || !actual.canonicalize()?.starts_with(&collection.root)
        {
            continue;
        }
        let mut best: Option<f32> = None;
        for chunk in skill.chunks {
            let score = super::cosine(query, &chunk.vector)?;
            best = Some(best.map_or(score, |s| s.max(score)));
        }
        if let Some(score) = best {
            hits.push(Hit {
                collection: name.into(),
                name: skill_name,
                path: actual.display().to_string(),
                relative_path: rel.into(),
                score,
                area: None,
                category: None,
                summary: None,
                kind: None,
                line: None,
                content: None,
            });
        }
    }
    hits.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.relative_path.cmp(&b.relative_path))
    });
    hits.truncate(limit);
    Ok(hits)
}

pub fn mode(db: &Connection, name: &str) -> Result<String> {
    Ok(get(db, name)?.0.result)
}

pub fn require_workspace_root(db: &Connection, name: &str, root: &Path) -> Result<()> {
    let (collection, _) = get(db, name)?;
    ensure!(
        collection.result == "workspace" && collection.root == root,
        "Workspace collection root differs from protected workspace"
    );
    Ok(())
}

pub fn require_skill_root(db: &Connection, name: &str, root: &Path) -> Result<()> {
    let (collection, _) = get(db, name)?;
    ensure!(
        collection.result == "skill" && collection.root == root.join("skills"),
        "Skill collection root differs from protected workspace"
    );
    Ok(())
}

pub fn require_model(db: &Connection, name: &str, model: &str) -> Result<()> {
    let (_, stored) = get(db, name)?;
    ensure!(
        stored.as_deref() == Some(model),
        "Collection model is incompatible with the active model"
    );
    Ok(())
}

/// Migrate the existing workspace skill index without re-embedding. Both the
/// source inventory and every stored chunk must still match the live files.
#[cfg(any(test, feature = "metal-eval", feature = "test-backend"))]
pub fn migrate_workspace_skills(
    db: &mut Connection,
    name: &str,
    root: &Path,
    source: &Path,
    model: &str,
) -> Result<usize> {
    ensure!(
        !super::reserved_nickname(name),
        "Collection name conflicts with a bsearch command"
    );
    ensure!(
        !name.is_empty()
            && name
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_'),
        "Invalid collection name"
    );
    let root = root.canonicalize()?;
    let skill_root = root.join("skills").canonicalize()?;
    ensure!(
        skill_root.starts_with(&root) && skill_root.is_dir(),
        "Skill root outside workspace"
    );
    let index = crate::native_skills::read_index(source, model)?;
    verify_workspace_skills(&root, &index)?;
    let tx = db.transaction()?;
    tx.execute("INSERT INTO collections(name,root,include,exclude,result,depth,returns,model) VALUES(?,?,'**/*.md','[]','skill',1,'paths',?)", params![name, skill_root.to_str().context("Non-UTF-8 skill root")?, model])?;
    write_skill_index(&tx, name, &index, model)?;
    verify_workspace_skills(&root, &index)?;
    tx.commit()?;
    Ok(index.skills.len())
}

#[cfg(any(test, feature = "metal-eval", feature = "test-backend"))]
fn verify_workspace_skills(root: &Path, index: &crate::native_skills::Index) -> Result<()> {
    let current = crate::native_skills::discover(root)?;
    ensure!(
        current.len() == index.skills.len(),
        "Skill inventory changed; refresh the old index first"
    );
    for (name, fresh) in current {
        let old = index
            .skills
            .get(&name)
            .context("Skill inventory differs from old index")?;
        ensure!(
            fresh.md5 == old.md5
                && fresh.kind == old.kind
                && fresh.area == old.area
                && fresh.path == old.path
                && fresh.category == old.category
                && fresh.chunks.len() == old.chunks.len()
                && fresh
                    .chunks
                    .iter()
                    .zip(&old.chunks)
                    .all(|(a, b)| a.file == b.file && a.heading == b.heading && a.text == b.text),
            "Skill content differs from old index; refresh before migrating"
        );
    }
    Ok(())
}

#[cfg(any(test, feature = "metal-eval", feature = "test-backend"))]
fn workspace_inventory(
    cfg: &crate::native_docs::Settings<'_>,
) -> Result<std::collections::BTreeMap<String, String>> {
    let mut inventory = std::collections::BTreeMap::new();
    for path in crate::native_docs::inventory(cfg)? {
        let rel = path
            .strip_prefix(cfg.root)?
            .to_str()
            .context("Non-UTF-8 document path")?
            .to_owned();
        inventory.insert(rel, crate::native_docs::hash_file(&path)?);
    }
    Ok(inventory)
}

/// Import a workspace snapshot only when all eligible sources and vector
/// references match. A failed or stale import leaves the destination unchanged.
#[cfg(any(test, feature = "metal-eval", feature = "test-backend"))]
pub fn migrate_workspace_documents<F>(
    db: &mut Connection,
    name: &str,
    source: &Connection,
    cfg: &crate::native_docs::Settings<'_>,
    fingerprint: F,
) -> Result<serde_json::Value>
where
    F: Fn() -> Result<String>,
{
    use crate::native_docs;
    use std::collections::BTreeMap;
    ensure!(
        !super::reserved_nickname(name)
            && !name.is_empty()
            && name
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_'),
        "Invalid or reserved collection name"
    );
    let integrity: String = source.query_row("PRAGMA integrity_check", [], |r| r.get(0))?;
    ensure!(integrity == "ok", "Source database integrity check failed");
    let metadata: Option<String> = source
        .query_row(
            "SELECT value FROM state WHERE key='registry_sha'",
            [],
            |r| r.get(0),
        )
        .optional()?;
    ensure!(
        metadata.as_deref() == Some(cfg.metadata_fingerprint),
        "Source document metadata is stale; refresh before migrating"
    );
    ensure!(
        fingerprint()? == cfg.metadata_fingerprint,
        "Workspace configuration changed before migration"
    );
    let mismatched: usize = source.query_row(
        "SELECT count(*) FROM vectors WHERE model <> ?",
        [cfg.model_id],
        |r| r.get(0),
    )?;
    ensure!(mismatched == 0, "Source contains incompatible vectors");
    let missing_vectors: usize = source.query_row("SELECT count(*) FROM chunks c LEFT JOIN files f ON f.path=c.path LEFT JOIN vectors v ON v.hash=c.hash AND v.model=? WHERE f.path IS NULL OR f.status <> 'ready' OR v.hash IS NULL", [cfg.model_id], |r| r.get(0))?;
    ensure!(
        missing_vectors == 0,
        "Source has incomplete or invalid chunk references"
    );
    let inventory = workspace_inventory(cfg)?;
    let mut indexed = BTreeMap::new();
    {
        let mut q = source.prepare("SELECT path,sha256,status FROM files")?;
        for row in q.query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
            ))
        })? {
            let (path, hash, status) = row?;
            ensure!(
                native_docs::allowed(&path, cfg),
                "Source contains a document excluded by workspace rules"
            );
            ensure!(
                status == "ready" || status == "metadata-only" || status == "missing",
                "Source contains pending or failed documents"
            );
            if status != "missing" {
                indexed.insert(path, hash);
            }
        }
    }
    ensure!(
        indexed == inventory,
        "Source documents changed; refresh old index before migrating"
    );
    let paths = crate::native_docs::inventory(cfg)?;
    let mut stats = Vec::with_capacity(paths.len());
    for p in &paths {
        let rel = p
            .strip_prefix(cfg.root)?
            .to_str()
            .context("Non-UTF-8 document path")?;
        let meta = fs::metadata(p)?;
        let (l, sec, nano) = file_stat(&meta);
        stats.push((rel.to_owned(), l, sec, nano));
    }
    let stat_fingerprint = compute_stat_fingerprint(&stats);
    let tx = db.transaction()?;
    tx.execute("INSERT INTO collections(name,root,include,exclude,result,depth,returns,model) VALUES(?,?,'**/*.md','[]','workspace',1,'paths',?)", params![name,cfg.root.to_str().context("Non-UTF-8 workspace root")?,cfg.model_id])?;
    tx.execute(
        "INSERT INTO collection_state(collection,key,value) VALUES(?,'registry_sha',?)",
        params![name, cfg.metadata_fingerprint],
    )?;
    tx.execute(
        "INSERT INTO collection_state(collection,key,value) VALUES(?,'stat_fingerprint',?)",
        params![name, stat_fingerprint],
    )?;
    let mut files = 0usize;
    {
        let mut q = source.prepare("SELECT path,sha256,title,summary,category,status,error,updated,area,kind FROM files ORDER BY path")?;
        let rows = q.query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, Option<String>>(2)?,
                r.get::<_, Option<String>>(3)?,
                r.get::<_, Option<String>>(4)?,
                r.get::<_, String>(5)?,
                r.get::<_, Option<String>>(6)?,
                r.get::<_, Option<String>>(7)?,
                r.get::<_, Option<String>>(8)?,
                r.get::<_, Option<String>>(9)?,
            ))
        })?;
        for row in rows {
            let (path, hash, title, summary, category, status, error, updated, area, kind) = row?;
            tx.execute("INSERT INTO files(collection,path,hash,title,summary,category,status,error,updated,area,kind) VALUES(?,?,?,?,?,?,?,?,?,?,?)",params![name,path,hash,title,summary,category,status,error,updated,area,kind])?;
            files += 1;
        }
    }
    let mut chunks = 0usize;
    {
        let mut q=source.prepare("SELECT c.path,c.ordinal,c.line,c.text,c.hash,v.vector FROM chunks c JOIN vectors v ON v.hash=c.hash AND v.model=? ORDER BY c.path,c.ordinal")?;
        let rows = q.query_map([cfg.model_id], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, usize>(1)?,
                r.get::<_, usize>(2)?,
                r.get::<_, String>(3)?,
                r.get::<_, String>(4)?,
                r.get::<_, Vec<u8>>(5)?,
            ))
        })?;
        for row in rows {
            let (path, ordinal, line, text, hash, vector) = row?;
            ensure!(
                line > 0 && digest(text.as_bytes()) == hash,
                "Invalid source chunk metadata"
            );
            ensure!(
                vector.len() == 768 * 4
                    && vector
                        .chunks_exact(4)
                        .all(|v| f32::from_le_bytes(v.try_into().unwrap()).is_finite()),
                "Invalid source vector"
            );
            tx.execute("INSERT INTO chunks(collection,path,ordinal,text,vector,line,source_hash) VALUES(?,?,?,?,?,?,?)",params![name,path,ordinal,text,vector,line,hash])?;
            chunks += 1;
        }
    }
    ensure!(
        workspace_inventory(cfg)? == inventory,
        "Sources changed while migrating; retry"
    );
    ensure!(
        fingerprint()? == cfg.metadata_fingerprint,
        "Workspace configuration changed during migration; retry"
    );
    tx.commit()?;
    Ok(serde_json::json!({"collection":name,"files":files,"chunks":chunks,"embedded":0}))
}

/// Incrementally refresh the protected workspace collection. Reuses the exact
/// legacy chunk hashes and defers all database writes until embedding succeeds.
#[cfg(any(test, feature = "metal-eval", feature = "test-backend"))]
pub fn scan_workspace<F, G>(
    db: &mut Connection,
    name: &str,
    cfg: &crate::native_docs::Settings<'_>,
    model_factory: F,
    fingerprint: G,
) -> Result<serde_json::Value>
where
    F: FnOnce() -> Result<Box<dyn crate::native_docs::Encoder>>,
    G: Fn() -> Result<String>,
{
    use crate::native_docs;
    use std::collections::{BTreeMap, BTreeSet};
    let (collection, model) = get(db, name)?;
    ensure!(
        collection.result == "workspace" && collection.root == cfg.root,
        "Workspace collection root differs from protected workspace"
    );
    ensure!(
        model.as_deref() == Some(cfg.model_id),
        "Workspace collection model mismatch"
    );
    ensure!(
        fingerprint()? == cfg.metadata_fingerprint,
        "Workspace configuration changed before scanning"
    );
    let paths = crate::native_docs::inventory(cfg)?;
    let mut stats = Vec::with_capacity(paths.len());
    for p in &paths {
        let rel = p
            .strip_prefix(cfg.root)?
            .to_str()
            .context("Non-UTF-8 document path")?;
        let meta = fs::metadata(p)?;
        let (l, sec, nano) = file_stat(&meta);
        stats.push((rel.to_owned(), l, sec, nano));
    }
    let stat_fingerprint = compute_stat_fingerprint(&stats);

    let stored_stat: Option<String> = db
        .query_row(
            "SELECT value FROM collection_state WHERE collection=? AND key='stat_fingerprint'",
            [name],
            |r| r.get(0),
        )
        .optional()?;
    let stored_registry: Option<String> = db
        .query_row(
            "SELECT value FROM collection_state WHERE collection=? AND key='registry_sha'",
            [name],
            |r| r.get(0),
        )
        .optional()?;

    if stored_stat.as_deref() == Some(&stat_fingerprint)
        && stored_registry.as_deref() == Some(&cfg.metadata_fingerprint)
    {
        let seen: usize = db.query_row(
            "SELECT count(*) FROM files WHERE collection=?",
            [name],
            |r| r.get(0),
        )?;
        return Ok(
            serde_json::json!({"collection":name,"seen":seen,"changed":0,"embedded":0,"removed":0}),
        );
    }
    let mut inventory = std::collections::BTreeMap::new();
    for p in paths {
        let rel = p
            .strip_prefix(cfg.root)?
            .to_str()
            .context("Non-UTF-8 document path")?
            .to_owned();
        inventory.insert(rel, crate::native_docs::hash_file(&p)?);
    }
    let registry_path = cfg.root.join("entities/registry.json");
    let registry: serde_json::Value = if registry_path.exists() {
        serde_json::from_slice(&fs::read(registry_path)?)?
    } else {
        serde_json::json!({})
    };
    let mut old: BTreeMap<
        String,
        (
            String,
            Option<String>,
            Option<String>,
            Option<String>,
            Option<String>,
        ),
    > = BTreeMap::new();
    {
        let mut q = db.prepare(
            "SELECT path,hash,title,summary,category,status FROM files WHERE collection=?",
        )?;
        for row in q.query_map([name], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, Option<String>>(2)?,
                r.get::<_, Option<String>>(3)?,
                r.get::<_, Option<String>>(4)?,
                r.get::<_, Option<String>>(5)?,
            ))
        })? {
            let (path, hash, title, summary, category, status) = row?;
            old.insert(path, (hash, title, summary, category, status));
        }
    }
    let mut vectors: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    {
        let mut q = db.prepare("SELECT source_hash,text,vector FROM chunks WHERE collection=?")?;
        for row in q.query_map([name], |r| {
            Ok((
                r.get::<_, Option<String>>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, Vec<u8>>(2)?,
            ))
        })? {
            let (hash, text, vector) = row?;
            let hash = hash.context("Workspace chunk has no source hash")?;
            ensure!(
                digest(text.as_bytes()) == hash
                    && vector.len() == 768 * 4
                    && vector
                        .chunks_exact(4)
                        .all(|v| f32::from_le_bytes(v.try_into().unwrap()).is_finite()),
                "Invalid workspace vector or chunk"
            );
            if let Some(existing) = vectors.get(&hash) {
                ensure!(existing == &vector, "Conflicting vectors for one chunk");
            } else {
                vectors.insert(hash, vector);
            }
        }
    }
    {
        let mut q = db.prepare("SELECT hash,vector FROM vector_cache WHERE model=?")?;
        for row in q.query_map([cfg.model_id], |r| {
            Ok((r.get::<_, String>(0)?, r.get::<_, Vec<u8>>(1)?))
        })? {
            let (hash, vector) = row?;
            ensure!(
                vector.len() == 768 * 4
                    && vector
                        .chunks_exact(4)
                        .all(|v| f32::from_le_bytes(v.try_into().unwrap()).is_finite()),
                "Invalid cached vector"
            );
            vectors.entry(hash).or_insert(vector);
        }
    }
    let missing_chunks: usize=db.query_row("SELECT count(*) FROM files f WHERE f.collection=? AND f.status='ready' AND NOT EXISTS(SELECT 1 FROM chunks c WHERE c.collection=f.collection AND c.path=f.path)",[name],|r|r.get(0))?;
    ensure!(missing_chunks == 0, "Ready workspace file has no chunks");
    struct Update {
        path: String,
        hash: String,
        title: String,
        summary: String,
        category: String,
        status: String,
        area: String,
        pieces: Vec<(usize, usize, String, String)>,
    }
    let mut updates = Vec::new();
    let mut needed: BTreeMap<String, String> = BTreeMap::new();
    for (path, hash) in &inventory {
        let full = cfg.root.join(path);
        let bytes = if fs::metadata(&full)?.len() <= 16 * 1024 * 1024 {
            fs::read(&full)?
        } else {
            Vec::new()
        };
        if !bytes.is_empty() {
            ensure!(digest(&bytes) == *hash, "Document changed while scanning");
        }
        let text = String::from_utf8_lossy(&bytes);
        let (title, summary, category) = native_docs::describe(path, &text, &registry);
        let pieces = native_docs::pieces(path, &title, &text);
        let status = if pieces.is_empty() {
            "metadata-only"
        } else {
            "ready"
        };
        if old.get(path).is_some_and(|(h, t, s, c, st)| {
            h == hash
                && t.as_deref() == Some(&title)
                && s.as_deref() == Some(&summary)
                && c.as_deref() == Some(&category)
                && st.as_deref() == Some(status)
        }) {
            continue;
        }
        for (_, _, text, chunk_hash) in &pieces {
            if !vectors.contains_key(chunk_hash) {
                needed
                    .entry(chunk_hash.clone())
                    .or_insert_with(|| text.clone());
            }
        }
        updates.push(Update {
            path: path.clone(),
            hash: hash.clone(),
            title,
            summary,
            category,
            status: status.into(),
            area: path.split('/').next().unwrap_or("").into(),
            pieces,
        });
    }
    let mut embedded = 0usize;
    if !needed.is_empty() {
        let mut encoder = model_factory()?;
        let all: Vec<_> = needed.into_iter().collect();
        for batch in all.chunks(32) {
            let texts: Vec<_> = batch.iter().map(|(_, t)| t.clone()).collect();
            let output = encoder.embed(&texts)?;
            ensure!(output.len() == batch.len(), "Embedding count mismatch");
            for ((hash, _), vector) in batch.iter().zip(output) {
                ensure!(
                    vector.len() == 768 && vector.iter().all(|v| v.is_finite()),
                    "Invalid workspace embedding"
                );
                vectors.insert(
                    hash.clone(),
                    vector.iter().flat_map(|v| v.to_le_bytes()).collect(),
                );
                embedded += 1;
            }
        }
    }
    ensure!(
        workspace_inventory(cfg)? == inventory,
        "Documents changed while scanning; retry"
    );
    ensure!(
        fingerprint()? == cfg.metadata_fingerprint,
        "Workspace configuration changed while scanning; retry"
    );
    let present: BTreeSet<_> = inventory.keys().cloned().collect();
    let tx = db.transaction()?;
    for update in &updates {
        tx.execute(
            "DELETE FROM files WHERE collection=? AND path=?",
            params![name, update.path],
        )?;
        let error = if update.status == "metadata-only" {
            "oversize or empty Markdown"
        } else {
            ""
        };
        tx.execute("INSERT INTO files(collection,path,hash,title,summary,category,status,error,updated,area,kind) VALUES(?,?,?,?,?,?,?,?,?,?, 'document')",params![name,update.path,update.hash,update.title,update.summary,update.category,update.status,error,chrono::Utc::now().to_rfc3339(),update.area])?;
        for (ordinal, line, text, hash) in &update.pieces {
            let vector = vectors.get(hash).context("Missing workspace vector")?;
            tx.execute("INSERT INTO chunks(collection,path,ordinal,text,vector,line,source_hash) VALUES(?,?,?,?,?,?,?)",params![name,update.path,ordinal,text,vector,line,hash])?;
        }
    }
    let mut removed = 0usize;
    for (path, (_, _, _, _, status)) in &old {
        if !native_docs::allowed(path, cfg) {
            tx.execute(
                "DELETE FROM files WHERE collection=? AND path=?",
                params![name, path],
            )?;
            removed += 1;
        } else if !present.contains(path) && status.as_deref() != Some("missing") {
            tx.execute(
                "DELETE FROM chunks WHERE collection=? AND path=?",
                params![name, path],
            )?;
            tx.execute(
                "UPDATE files SET status='missing' WHERE collection=? AND path=?",
                params![name, path],
            )?;
            removed += 1;
        }
    }
    tx.execute(
        "INSERT OR REPLACE INTO collection_state(collection,key,value) VALUES(?,'registry_sha',?)",
        params![name, cfg.metadata_fingerprint],
    )?;
    tx.execute(
        "INSERT OR REPLACE INTO collection_state(collection,key,value) VALUES(?,'stat_fingerprint',?)",
        params![name, stat_fingerprint],
    )?;
    tx.execute("DELETE FROM vector_cache WHERE model=? AND hash NOT IN (SELECT source_hash FROM chunks WHERE collection=? AND source_hash IS NOT NULL)",params![cfg.model_id,name])?;
    tx.commit()?;
    Ok(
        serde_json::json!({"collection":name,"seen":inventory.len(),"changed":updates.len(),"embedded":embedded,"removed":removed}),
    )
}
