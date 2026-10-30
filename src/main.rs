use anyhow::{Context, Result, ensure};
use clap::{CommandFactory, Parser, Subcommand, ValueEnum};
use globset::{Glob, GlobSet, GlobSetBuilder};
use rusqlite::{Connection, OpenFlags, OptionalExtension};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, HashMap},
    fs,
    io::{self, Read, Write},
    path::{Component, Path, PathBuf},
    process::{Command, Stdio},
};
#[cfg(any(test, feature = "metal-eval", feature = "test-backend"))]
mod collections;
#[cfg(feature = "metal-eval")]
mod embedding;
#[cfg(feature = "gemma2-mlx")]
mod gemma2;
#[cfg(feature = "gemma2-mlx")]
mod model_integrity;
#[cfg(any(test, feature = "metal-eval", feature = "test-backend"))]
mod native_docs;
#[cfg(any(test, feature = "metal-eval"))]
mod native_portable;
#[cfg(any(test, feature = "metal-eval", feature = "test-backend"))]
mod native_skills;
#[cfg(feature = "test-backend")]
struct FixtureEncoder(Vec<f32>);
#[cfg(feature = "test-backend")]
impl native_docs::Encoder for FixtureEncoder {
    fn embed(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|_| self.0.clone()).collect())
    }
}
#[cfg(feature = "metal-eval")]
impl native_docs::Encoder for embedding::Embedder {
    fn embed(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let refs: Vec<_> = texts.iter().map(String::as_str).collect();
        self.embed(&refs)
    }
}

#[cfg(feature = "gemma2-mlx")]
impl native_docs::Encoder for gemma2::Embedder {
    fn embed(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let refs: Vec<_> = texts.iter().map(String::as_str).collect();
        self.embed(&refs)
    }
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, ValueEnum, PartialEq)]
#[serde(rename_all = "lowercase")]
enum Kind {
    #[default]
    Document,
    Skill,
}
#[derive(Clone, Copy, Debug, Default, Deserialize, ValueEnum, PartialEq)]
#[serde(rename_all = "lowercase")]
enum Method {
    #[default]
    Vector,
}
#[derive(Clone, Copy, Debug, Default, Deserialize, ValueEnum)]
#[serde(rename_all = "lowercase")]
enum Format {
    #[default]
    Json,
    Text,
    Markdown,
}
#[derive(Clone, Copy, Debug, Default, Deserialize, ValueEnum, PartialEq)]
#[serde(rename_all = "lowercase")]
enum Return {
    #[default]
    Paths,
    Snippets,
    Full,
}

#[derive(Parser)]
#[command(
    version,
    about = "Local Markdown and skill search with directory scopes"
)]
struct Cli {
    #[arg(long, global = true)]
    config: Option<PathBuf>,
    #[arg(long, global = true, help = "Override the document SQLite database")]
    db: Option<PathBuf>,
    #[command(subcommand)]
    command: Action,
}
#[derive(Subcommand)]
enum Action {
    /// Manage independently rooted collections in a separate vector database.
    Collection {
        #[command(subcommand)]
        action: CollectionAction,
    },
    #[command(hide = true)]
    HookWorker,
    Search {
        query: String,
        #[arg(long)]
        scope: Option<String>,
        #[arg(long, value_enum)]
        kind: Option<Kind>,
        #[arg(long, value_enum)]
        method: Option<Method>,
        #[arg(long = "return", value_enum)]
        returns: Option<Return>,
        #[arg(long, value_enum)]
        format: Option<Format>,
        #[arg(short = 'k', long, default_value_t = 5)]
        limit: usize,
        #[arg(long, help = "Do not record skill retrieval usage")]
        no_log: bool,
    },
    Get {
        path: String,
        #[arg(long, default_value_t = 1)]
        from: usize,
        #[arg(long)]
        lines: Option<usize>,
        #[arg(long, value_enum)]
        format: Option<Format>,
    },
    List {
        #[arg(long)]
        scope: Option<String>,
        #[arg(long, value_enum)]
        kind: Option<Kind>,
        #[arg(long, value_enum)]
        format: Option<Format>,
    },
    Scan {
        #[arg(long, value_enum, default_value = "document")]
        kind: Kind,
        #[arg(long)]
        background: bool,
        #[arg(
            long,
            help = "Fail if any document needs embedding; do not load the model"
        )]
        no_embed: bool,
    },
    /// Export or import vectors after reconciling the same source files on both machines.
    Vectors {
        #[command(subcommand)]
        action: VectorAction,
    },
    /// Record that a skill was actually adopted.
    Used {
        skill: String,
    },
    /// Summarize historical skill retrieval and adoption counts.
    Report {
        #[arg(long, default_value_t = 8)]
        weeks: u32,
    },
    Status,
    #[command(about = "Print resolved configuration, without modifying it")]
    Config,
    #[command(about = "Project SessionStart dispatcher; reads hook JSON from stdin")]
    Hook,
}
#[derive(Subcommand)]
enum CollectionAction {
    /// Import the existing workspace document index into the collection database.
    MigrateDocuments {
        #[arg(long, default_value = "workspace")]
        name: String,
    },
    /// Import the existing workspace skill index into the collection database.
    MigrateSkills {
        #[arg(long, default_value = "skills")]
        name: String,
    },
    Add {
        name: String,
        path: PathBuf,
        #[arg(
            long,
            help = "Allow a reserved command name; search via collection search only"
        )]
        no_shortcut: bool,
        #[arg(long, default_value = "**/*.md")]
        include: String,
        #[arg(long)]
        exclude: Vec<String>,
        #[arg(long, visible_alias = "mode", default_value = "document", value_parser = ["document", "directory", "skill"])]
        result: String,
        #[arg(long, default_value_t = 1)]
        depth: usize,
        #[arg(long = "return", value_enum, default_value = "paths")]
        returns: Return,
    },
    List,
    Scan {
        name: String,
        #[arg(long, help = "Fail if any new embeddings are needed")]
        no_embed: bool,
    },
    Search {
        name: String,
        query: String,
        #[arg(long)]
        scope: Option<String>,
        #[arg(long = "return", value_enum)]
        returns: Option<Return>,
        #[arg(long, value_enum)]
        format: Option<Format>,
        #[arg(short = 'k', long, default_value_t = 5)]
        limit: usize,
        #[arg(long, help = "Do not record skill retrieval usage")]
        no_log: bool,
    },
}
#[derive(Subcommand)]
enum VectorAction {
    Export { path: PathBuf },
    Import { path: PathBuf },
}
#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct Defaults {
    #[serde(default)]
    method: Method,
    #[serde(default)]
    format: Format,
}
#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct Entry {
    #[serde(default)]
    kind: Kind,
    #[serde(default, rename = "return")]
    returns: Return,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Settings {
    root: PathBuf,
    database: PathBuf,
    #[serde(default)]
    skill_index: PathBuf,
    #[serde(default)]
    workspace_collections: bool,
    document_roots: Vec<String>,
    include: Vec<String>,
    exclude: Vec<String>,
    #[serde(default)]
    defaults: Defaults,
    #[serde(default)]
    entries: HashMap<String, Entry>,
    #[serde(default)]
    embedding_command: Vec<String>,
    #[serde(default, rename = "adapter")]
    _legacy_adapter: Option<PathBuf>,
    #[serde(default)]
    backend: Option<String>,
    #[serde(default)]
    model_path: Option<PathBuf>,
    #[cfg(feature = "test-backend")]
    #[serde(default)]
    fixture_vector: Option<Vec<f32>>,
    #[cfg(feature = "test-backend")]
    #[serde(default)]
    fixture_worker_log: Option<PathBuf>,
}
struct App {
    file: PathBuf,
    cfg: Settings,
    root: PathBuf,
    db: PathBuf,
    skills: PathBuf,
    includes: GlobSet,
    excludes: GlobSet,
}

fn normalize(path: &Path) -> Result<PathBuf> {
    let mut result = PathBuf::new();
    for part in path.components() {
        match part {
            Component::ParentDir => {
                ensure!(result.pop(), "Path escapes its root");
            }
            Component::CurDir => (),
            _ => result.push(part.as_os_str()),
        }
    }
    Ok(result)
}
fn globs(patterns: &[String], lower: bool) -> Result<GlobSet> {
    let mut builder = GlobSetBuilder::new();
    for p in patterns {
        builder.add(Glob::new(&if lower {
            p.to_lowercase()
        } else {
            p.clone()
        })?);
    }
    Ok(builder.build()?)
}
impl App {
    fn collection_path(&self) -> Result<PathBuf> {
        if self.cfg.workspace_collections {
            Ok(self.db.clone())
        } else {
            Ok(self
                .db
                .parent()
                .context("Database has no parent")?
                .join("collections.sqlite3"))
        }
    }

    fn collection_db_readonly(&self) -> Result<Connection> {
        let path = self.collection_path()?;
        ensure!(path.is_file(), "Collection database is absent");
        Ok(Connection::open_with_flags(
            path,
            OpenFlags::SQLITE_OPEN_READ_ONLY,
        )?)
    }
    fn native_enabled(&self) -> bool {
        matches!(
            self.cfg.backend.as_deref().unwrap_or("metal"),
            "metal" | "mlx-v2"
        )
    }
    #[cfg(feature = "metal-eval")]
    fn model_id(&self) -> &'static str {
        #[cfg(feature = "gemma2-mlx")]
        if self.cfg.backend.as_deref() == Some("mlx-v2") {
            return gemma2::MODEL_ID;
        }
        embedding::MODEL_ID
    }
    #[cfg(feature = "metal-eval")]
    fn model(&self) -> Result<Box<dyn native_docs::Encoder>> {
        let path = self.native_model_path()?;
        #[cfg(feature = "gemma2-mlx")]
        if self.cfg.backend.as_deref() == Some("mlx-v2") {
            return Ok(Box::new(gemma2::Embedder::load(&path)?));
        }
        Ok(Box::new(embedding::Embedder::load(&path)?))
    }
    #[cfg(feature = "metal-eval")]
    fn native_fingerprint(&self) -> Result<String> {
        use sha2::{Digest, Sha256};
        let mut hash = Sha256::new();
        hash.update(fs::read(&self.file)?);
        let registry = self.root.join("entities/registry.json");
        if registry.exists() {
            hash.update(fs::read(registry)?);
        }
        Ok(format!("staged-native-v1:{:x}", hash.finalize()))
    }
    #[cfg(feature = "metal-eval")]
    fn native_settings<'a>(&'a self, fingerprint: &'a str) -> native_docs::Settings<'a> {
        native_docs::Settings {
            root: &self.root,
            roots: &self.cfg.document_roots,
            includes: &self.includes,
            excludes: &self.excludes,
            model_id: self.model_id(),
            metadata_fingerprint: fingerprint,
        }
    }
    #[cfg(feature = "metal-eval")]
    fn native_model_path(&self) -> Result<PathBuf> {
        if let Some(path) = &self.cfg.model_path {
            return Ok(if path.is_absolute() {
                path.clone()
            } else {
                self.file.parent().unwrap().join(path)
            });
        }
        let home = PathBuf::from(std::env::var_os("HOME").context("HOME is not set")?);
        let models = home.join("Library/Caches/bsearch/models");
        if self.cfg.backend.as_deref() == Some("mlx-v2") {
            let text_only = models.join("embeddinggemma-2-bf16-text");
            return Ok(if text_only.is_dir() {
                text_only
            } else {
                models.join("embeddinggemma-2-bf16")
            });
        }
        Ok(models.join("embeddinggemma-300M-Q8_0.gguf"))
    }
    #[cfg(feature = "metal-eval")]
    fn native_scan(&self, kind: Kind, no_embed: bool, wait_for_lock: bool) -> Result<()> {
        let dir = self.db.parent().context("Database has no parent")?;
        fs::create_dir_all(dir)?;
        let _lock = lock_writer(dir, wait_for_lock)?;
        if self.cfg.workspace_collections {
            let mut db = collections::open(&self.collection_path()?)?;
            let result = if kind == Kind::Skill {
                ensure!(!no_embed, "--no-embed applies only to document scanning");
                collections::scan_skills(&mut db, "skills", self.model_id(), || self.model())?
            } else {
                let fingerprint = self.native_fingerprint()?;
                let cfg = self.native_settings(&fingerprint);
                collections::scan_workspace(
                    &mut db,
                    "workspace",
                    &cfg,
                    || {
                        ensure!(
                            !no_embed,
                            "New document embeddings required; --no-embed forbids model use"
                        );
                        self.model()
                    },
                    || self.native_fingerprint(),
                )?
            };
            println!("{result}");
            return Ok(());
        }
        if kind == Kind::Skill {
            ensure!(!no_embed, "--no-embed applies only to document scanning");
            let result =
                native_skills::scan(&self.root, &self.skills, self.model_id(), || self.model())?;
            println!(
                "{}",
                json!({"skills":result.skills,"changed":result.changed,"embedded":result.embedded,"model":self.model_id()})
            );
        } else {
            let fingerprint = self.native_fingerprint()?;
            let cfg = self.native_settings(&fingerprint);
            let mut db = Connection::open(&self.db)?;
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                fs::set_permissions(&self.db, fs::Permissions::from_mode(0o600))?;
            }
            let result = native_docs::scan(&mut db, &cfg, no_embed, || self.model())?;
            println!(
                "{}",
                json!({"seen":result.seen,"changed":result.changed,"embedded":result.embedded,"errors":result.errors,"model":self.model_id()})
            );
            ensure!(result.errors == 0, "Some documents could not be indexed");
        }
        Ok(())
    }
    fn load(cli: &Cli) -> Result<Self> {
        let file = if let Some(p) = &cli.config {
            p.canonicalize()?
        } else if let Some(path) = std::env::var_os("BSEARCH_CONFIG") {
            PathBuf::from(path)
                .canonicalize()
                .context("BSEARCH_CONFIG is not a readable configuration path")?
        } else {
            let cwd = std::env::current_dir()?;
            let local = cwd
                .ancestors()
                .flat_map(|p| {
                    [
                        (p.join("tools/document-vector/config.json"), true),
                        (p.join("config.json"), false),
                    ]
                })
                .find_map(|(path, workspace_config)| {
                    if !path.is_file()
                        || !path.parent().is_some_and(|p| p.join("Cargo.toml").exists())
                    {
                        return None;
                    }
                    if workspace_config {
                        return Some(path);
                    }
                    let value: Value = serde_json::from_slice(&fs::read(&path).ok()?).ok()?;
                    ["root", "database", "document_roots", "include", "exclude"]
                        .iter()
                        .all(|key| value.get(key).is_some())
                        .then_some(path)
                });
            match local {
                Some(path) => path.canonicalize()?,
                None => {
                    let home =
                        std::env::var_os("HOME").context("Config not found and HOME is not set")?;
                    PathBuf::from(home).join(".config/bsearch/config.json").canonicalize().context("Config not found; use --config or create ~/.config/bsearch/config.json")?
                }
            }
        };
        let cfg: Settings = serde_json::from_str(&fs::read_to_string(&file)?)?;
        ensure!(
            cfg.embedding_command.is_empty(),
            "External embedding commands are no longer supported"
        );
        let backend = cfg.backend.as_deref().unwrap_or("metal");
        #[cfg(feature = "test-backend")]
        ensure!(
            backend == "metal"
                || backend == "fixture"
                || cfg!(feature = "gemma2-mlx") && backend == "mlx-v2",
            "Unsupported backend: {backend}"
        );
        #[cfg(not(feature = "test-backend"))]
        ensure!(
            backend == "metal" || cfg!(feature = "gemma2-mlx") && backend == "mlx-v2",
            "Unsupported backend: {backend}"
        );
        let parent = file.parent().unwrap();
        let root = parent
            .join(&cfg.root)
            .canonicalize()
            .context("Invalid workspace root")?;
        for p in &cfg.document_roots {
            ensure!(
                !Path::new(p).is_absolute()
                    && !Path::new(p).components().any(|c| c == Component::ParentDir),
                "Document roots must be workspace-relative directories"
            );
        }
        let db = normalize(
            &cli.db
                .clone()
                .map(|p| {
                    if p.is_absolute() {
                        p
                    } else {
                        std::env::current_dir().unwrap().join(p)
                    }
                })
                .unwrap_or_else(|| parent.join(&cfg.database)),
        )?;
        let skills = if cfg.workspace_collections {
            db.clone()
        } else {
            ensure!(
                !cfg.skill_index.as_os_str().is_empty(),
                "Legacy skill_index is required"
            );
            normalize(&parent.join(&cfg.skill_index))?
        };
        let includes = globs(&cfg.include, true)?;
        let excludes = globs(&cfg.exclude, false)?;
        Ok(Self {
            file,
            cfg,
            root,
            db,
            skills,
            includes,
            excludes,
        })
    }
    fn scope(&self, value: Option<&str>) -> Result<String> {
        let joined = self.root.join(value.unwrap_or("."));
        let path = if joined.exists() {
            joined.canonicalize()?
        } else {
            normalize(&joined)?
        };
        ensure!(
            path.starts_with(&self.root),
            "Scope must stay inside the workspace"
        );
        ensure!(!path.is_file(), "Scope must be a directory");
        Ok(path
            .strip_prefix(&self.root)?
            .to_string_lossy()
            .replace('\\', "/"))
    }
    fn eligible(&self, path: &str) -> bool {
        let parts: Vec<_> = path.split('/').collect();
        if parts
            .first()
            .is_some_and(|p| p.eq_ignore_ascii_case("archive"))
            || !path.to_lowercase().ends_with(".md")
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
        self.cfg.document_roots.iter().any(|r| under(path, r))
            && self.includes.is_match(path.to_lowercase())
            && !self.excludes.is_match(path)
    }
    fn kind_return(
        &self,
        scope: &str,
        kind: Option<Kind>,
        returns: Option<Return>,
    ) -> (Kind, Return) {
        let entry = entry_for_scope(&self.cfg.entries, scope);
        let kind = kind.unwrap_or_else(|| entry.map(|e| e.kind).unwrap_or_default());
        let returns = returns.unwrap_or_else(|| {
            entry
                .filter(|e| e.kind == kind)
                .map(|e| e.returns)
                .unwrap_or_default()
        });
        (kind, returns)
    }
    fn db(&self) -> Result<Connection> {
        ensure!(
            self.db.exists(),
            "Document database is absent; run bsearch scan"
        );
        Ok(Connection::open_with_flags(
            &self.db,
            OpenFlags::SQLITE_OPEN_READ_ONLY,
        )?)
    }
    fn query_vector(&self, query: &str, kind: Kind) -> Result<Vec<f32>> {
        #[cfg(feature = "test-backend")]
        if self.cfg.backend.as_deref() == Some("fixture") {
            return self
                .cfg
                .fixture_vector
                .clone()
                .context("Fixture vector missing");
        }
        #[cfg(feature = "metal-eval")]
        {
            let mut model = self.model()?;
            let text = if kind == Kind::Document || self.cfg.backend.as_deref() == Some("mlx-v2") {
                format!("task: search result | query: {query}")
            } else {
                query.to_owned()
            };
            Ok(model.embed(&[text])?.remove(0))
        }
        #[cfg(not(feature = "metal-eval"))]
        anyhow::bail!("Metal backend is unavailable in this build")
    }
    fn chunks(&self, kind: Kind, scope: &str) -> Result<Vec<Chunk>> {
        #[cfg(any(test, feature = "metal-eval", feature = "test-backend"))]
        if self.cfg.workspace_collections {
            let db = self.collection_db_readonly()?;
            let mut out = Vec::new();
            if kind == Kind::Document {
                collections::require_workspace_root(&db, "workspace", &self.root)?;
                #[cfg(feature = "metal-eval")]
                if self.native_enabled() {
                    collections::require_model(&db, "workspace", self.model_id())?;
                }
                let mut q = db.prepare("SELECT f.path,f.title,f.area,f.category,c.text,c.line,c.vector FROM files f JOIN chunks c ON f.collection=c.collection AND f.path=c.path WHERE f.collection='workspace' AND f.status='ready'")?;
                let rows = q.query_map([], |r| {
                    Ok((
                        r.get::<_, String>(0)?,
                        r.get::<_, String>(1)?,
                        r.get::<_, String>(2)?,
                        r.get::<_, String>(3)?,
                        r.get::<_, String>(4)?,
                        r.get::<_, usize>(5)?,
                        r.get::<_, Vec<u8>>(6)?,
                    ))
                })?;
                for row in rows {
                    let (path, name, area, category, text, line, bytes) = row?;
                    if !self.eligible(&path) || !under(&path, scope) {
                        continue;
                    }
                    ensure!(bytes.len() % 4 == 0, "Malformed vector blob");
                    let vector = bytes
                        .chunks_exact(4)
                        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                        .collect();
                    out.push(Chunk {
                        path,
                        name,
                        area,
                        category,
                        text,
                        line: Some(line),
                        vector,
                    });
                }
            } else {
                collections::require_skill_root(&db, "skills", &self.root)?;
                #[cfg(feature = "metal-eval")]
                let model = self.model_id();
                #[cfg(not(feature = "metal-eval"))]
                let model = "fixture-v1";
                let index = collections::load_skill_index(&db, "skills", model)?;
                for (name, skill) in index.skills {
                    let path = &skill.path;
                    if !under(path, scope)
                        || self.excludes.is_match(path)
                        || self.excludes.is_match(format!("{path}/SKILL.md"))
                    {
                        continue;
                    }
                    let root = self.root.join(path);
                    if !root.join("SKILL.md").is_file()
                        || !root.canonicalize()?.starts_with(&self.root)
                    {
                        continue;
                    }
                    for chunk in skill.chunks {
                        out.push(Chunk {
                            path: path.clone(),
                            name: name.clone(),
                            area: skill.area.clone(),
                            category: skill.category.clone(),
                            text: chunk.text,
                            line: None,
                            vector: chunk.vector,
                        });
                    }
                }
            }
            return Ok(out);
        }
        let mut out = Vec::new();
        if kind == Kind::Document {
            let db = self.db()?;
            let model: Option<String> = db
                .query_row(
                    "SELECT model FROM vectors ORDER BY rowid DESC LIMIT 1",
                    [],
                    |r| r.get(0),
                )
                .optional()?;
            let Some(model) = model else {
                return Ok(out);
            };
            #[cfg(feature = "metal-eval")]
            if self.native_enabled() {
                ensure!(
                    model == self.model_id(),
                    "Document index model is incompatible with the Metal backend"
                );
                let other: usize = db.query_row(
                    "SELECT count(*) FROM vectors WHERE model<>?",
                    [self.model_id()],
                    |r| r.get(0),
                )?;
                ensure!(other == 0, "Document index contains mixed model versions");
            }
            let prefix = if scope.is_empty() {
                String::new()
            } else {
                format!("{scope}/")
            };
            let mut stmt=db.prepare("SELECT f.path,f.title,f.area,f.category,c.text,c.line,v.vector FROM chunks c JOIN files f ON f.path=c.path JOIN vectors v ON v.hash=c.hash AND v.model=?1 WHERE f.status='ready' AND substr(f.path,1,length(?2))=?2")?;
            let rows = stmt.query_map([&model, &prefix], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, String>(2)?,
                    r.get::<_, String>(3)?,
                    r.get::<_, String>(4)?,
                    r.get::<_, usize>(5)?,
                    r.get::<_, Vec<u8>>(6)?,
                ))
            })?;
            for row in rows {
                let (path, name, area, category, text, line, bytes) = row?;
                if !self.eligible(&path) {
                    continue;
                }
                ensure!(bytes.len() % 4 == 0, "Malformed vector blob");
                let vector = bytes
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect();
                out.push(Chunk {
                    path,
                    name,
                    area,
                    category,
                    text,
                    line: Some(line),
                    vector,
                });
            }
        } else {
            let state: Value = serde_json::from_str(
                &fs::read_to_string(&self.skills)
                    .context("Skill index absent; run bsearch scan --kind skill")?,
            )?;
            #[cfg(feature = "metal-eval")]
            if self.native_enabled() {
                ensure!(
                    state["model_id"].as_str() == Some(self.model_id()),
                    "Skill index model is incompatible with the Metal backend"
                );
            }
            for (name, s) in state["skills"]
                .as_object()
                .context("Malformed skill index")?
            {
                let path = s["path"]
                    .as_str()
                    .context("Skill path missing; refresh skill index metadata")?;
                if !under(path, scope) || path.split('/').next() != Some("skills") {
                    continue;
                }
                if self.excludes.is_match(path)
                    || self.excludes.is_match(format!("{path}/SKILL.md"))
                {
                    continue;
                }
                let root = self.root.join(path);
                if !root.join("SKILL.md").is_file() {
                    continue;
                }
                if !root.canonicalize()?.starts_with(&self.root) {
                    continue;
                }
                for c in s["chunks"].as_array().context("Malformed skill chunks")? {
                    if !c["vector"].is_array() {
                        continue;
                    }
                    out.push(Chunk {
                        path: path.into(),
                        name: name.clone(),
                        area: "skills".into(),
                        category: s["category"].as_str().unwrap_or("skills").into(),
                        text: c["text"].as_str().unwrap_or("").into(),
                        line: None,
                        vector: serde_json::from_value(c["vector"].clone())?,
                    });
                }
            }
        }
        Ok(out)
    }
    fn objects(&self, kind: Kind, scope: &str) -> Result<Vec<Chunk>> {
        if kind == Kind::Skill {
            return self.chunks(kind, scope);
        }
        #[cfg(any(test, feature = "metal-eval", feature = "test-backend"))]
        if self.cfg.workspace_collections {
            let db = self.collection_db_readonly()?;
            collections::require_workspace_root(&db, "workspace", &self.root)?;
            let mut stmt=db.prepare("SELECT path,title,area,category FROM files WHERE collection='workspace' AND status<>'missing' ORDER BY path")?;
            let rows = stmt.query_map([], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, String>(2)?,
                    r.get::<_, String>(3)?,
                ))
            })?;
            let mut out = Vec::new();
            for row in rows {
                let (path, name, area, category) = row?;
                if self.eligible(&path) && under(&path, scope) {
                    out.push(Chunk {
                        path,
                        name,
                        area,
                        category,
                        text: String::new(),
                        line: None,
                        vector: Vec::new(),
                    });
                }
            }
            return Ok(out);
        }
        let db = self.db()?;
        let mut stmt = db.prepare(
            "SELECT path,title,area,category FROM files WHERE status<>'missing' ORDER BY path",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
                r.get::<_, String>(3)?,
            ))
        })?;
        let mut out = Vec::new();
        for row in rows {
            let (path, name, area, category) = row?;
            if self.eligible(&path) && under(&path, scope) {
                out.push(Chunk {
                    path,
                    name,
                    area,
                    category,
                    text: String::new(),
                    line: None,
                    vector: Vec::new(),
                });
            }
        }
        Ok(out)
    }
    fn content(&self, path: &str, kind: Kind) -> Result<String> {
        let path = if kind == Kind::Skill {
            self.root.join(path).join("SKILL.md")
        } else {
            self.root.join(path)
        };
        let actual = path.canonicalize()?;
        ensure!(
            actual.starts_with(&self.root),
            "Content path escapes the workspace"
        );
        ensure!(
            actual
                .extension()
                .is_some_and(|s| s.eq_ignore_ascii_case("md")),
            "Only Markdown content may be returned"
        );
        Ok(fs::read_to_string(actual)?)
    }
    fn dispatch(&self, event: &Value) -> Result<()> {
        if !session_start_in_workspace(event, &self.root) {
            println!("{{}}");
            return Ok(());
        }
        let dir = self.db.parent().context("Database has no parent")?;
        fs::create_dir_all(dir)?;
        let log = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(dir.join("worker.log"))?;
        let mut cmd = Command::new(std::env::current_exe()?);
        cmd.arg("--config")
            .arg(&self.file)
            .arg("--db")
            .arg(&self.db)
            .arg("hook-worker")
            .current_dir(&self.root)
            .stdin(Stdio::null())
            .stdout(Stdio::from(log.try_clone()?))
            .stderr(Stdio::from(log));
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            // Detach before exec so an agent ending its session cannot kill the worker.
            unsafe {
                cmd.pre_exec(|| {
                    if libc::setsid() == -1 {
                        Err(io::Error::last_os_error())
                    } else {
                        Ok(())
                    }
                });
            }
        }
        cmd.spawn()
            .context("Unable to start background index worker")?;
        println!("{{}}");
        Ok(())
    }
}
fn session_start_in_workspace(event: &Value, root: &Path) -> bool {
    if matches!(
        event.get("source").and_then(Value::as_str),
        Some("compact" | "clear")
    ) {
        return false;
    }
    let within = |path: &str| {
        Path::new(path).is_absolute()
            && Path::new(path)
                .canonicalize()
                .is_ok_and(|p| p.starts_with(root))
    };
    if let Some(name) = event.get("hook_event_name") {
        return name.as_str() == Some("SessionStart")
            && event.get("cwd").and_then(Value::as_str).is_some_and(within);
    }
    event
        .get("conversationId")
        .and_then(Value::as_str)
        .is_some_and(|s| !s.is_empty())
        && event
            .get("workspacePaths")
            .and_then(Value::as_array)
            .is_some_and(|paths| paths.iter().filter_map(Value::as_str).any(within))
}
fn under(path: &str, scope: &str) -> bool {
    scope.is_empty() || path == scope || path.starts_with(&format!("{scope}/"))
}
// A nested policy overrides its parent only inside that directory, never a
// sibling whose name merely shares a prefix.
fn entry_for_scope<'a>(entries: &'a HashMap<String, Entry>, scope: &str) -> Option<&'a Entry> {
    entries
        .iter()
        .filter(|(path, _)| under(scope, path.trim_end_matches('/')))
        .max_by_key(|(path, _)| path.trim_end_matches('/').len())
        .map(|(_, entry)| entry)
}
#[cfg(any(test, feature = "metal-eval", feature = "test-backend"))]
fn lock_writer(dir: &Path, wait: bool) -> Result<fs::File> {
    fs::create_dir_all(dir)?;
    let lock = fs::OpenOptions::new()
        .create(true)
        .write(true)
        .open(dir.join("writer.lock"))?;
    if wait {
        lock.lock()?;
    } else {
        lock.try_lock()
            .map_err(|_| anyhow::anyhow!("An indexing operation is already running"))?;
    }
    Ok(lock)
}

#[cfg(test)]
mod lock_tests {
    use super::*;
    #[test]
    fn nested_entry_policy_uses_longest_directory_match() {
        let entries = HashMap::from([
            (
                "library".to_owned(),
                Entry {
                    kind: Kind::Document,
                    returns: Return::Snippets,
                },
            ),
            (
                "library/notes".to_owned(),
                Entry {
                    kind: Kind::Document,
                    returns: Return::Full,
                },
            ),
        ]);
        assert_eq!(
            entry_for_scope(&entries, "library/notes/work")
                .unwrap()
                .returns,
            Return::Full
        );
        assert_eq!(
            entry_for_scope(&entries, "library/notes-old")
                .unwrap()
                .returns,
            Return::Snippets
        );
        assert!(entry_for_scope(&entries, "library2").is_none());
    }
    #[test]
    fn background_waits_until_a_running_writer_finishes() {
        let temp = tempfile::tempdir().unwrap();
        let held = lock_writer(temp.path(), false).unwrap();
        assert!(lock_writer(temp.path(), false).is_err());
        let path = temp.path().to_path_buf();
        let (tx, rx) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || {
            let guard = lock_writer(&path, true).unwrap();
            tx.send(()).unwrap();
            drop(guard);
        });
        assert!(
            rx.recv_timeout(std::time::Duration::from_millis(100))
                .is_err()
        );
        drop(held);
        rx.recv_timeout(std::time::Duration::from_secs(2)).unwrap();
        worker.join().unwrap();
    }
    #[test]
    #[cfg(unix)]
    fn writer_lock_interoperates_with_existing_flock() {
        use std::os::fd::AsRawFd;
        let temp = tempfile::tempdir().unwrap();
        let old = fs::OpenOptions::new()
            .create(true)
            .write(true)
            .open(temp.path().join("writer.lock"))
            .unwrap();
        assert_eq!(
            unsafe { libc::flock(old.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) },
            0
        );
        assert!(lock_writer(temp.path(), false).is_err());
        assert_eq!(unsafe { libc::flock(old.as_raw_fd(), libc::LOCK_UN) }, 0);
        assert!(lock_writer(temp.path(), false).is_ok());
    }
}
struct Chunk {
    path: String,
    name: String,
    area: String,
    category: String,
    text: String,
    line: Option<usize>,
    vector: Vec<f32>,
}
#[derive(Serialize)]
struct Hit {
    kind: Kind,
    name: String,
    area: String,
    category: String,
    path: String,
    relative_path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    score: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    line: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
}
fn cosine(a: &[f32], b: &[f32]) -> Result<f32> {
    ensure!(
        a.len() == b.len(),
        "Embedding dimensions differ; use the index's model or rebuild"
    );
    ensure!(b.iter().all(|x| x.is_finite()), "Nonfinite index vector");
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm = (a.iter().map(|x| x * x).sum::<f32>() * b.iter().map(|x| x * x).sum::<f32>()).sqrt();
    Ok(if norm > 0. { dot / norm } else { 0. })
}
fn scores(chunks: &[Chunk], q: &[f32]) -> Result<Vec<f32>> {
    chunks.iter().map(|c| cosine(q, &c.vector)).collect()
}

fn record_usage(root: &Path, mut event: Value) -> Result<()> {
    let mut hostbuf = [0_u8; 256];
    #[cfg(unix)]
    ensure!(
        unsafe { libc::gethostname(hostbuf.as_mut_ptr().cast(), hostbuf.len()) } == 0,
        "Unable to identify usage-log host: {}",
        io::Error::last_os_error()
    );
    let host = std::ffi::CStr::from_bytes_until_nul(&hostbuf)?
        .to_str()?
        .split('.')
        .next()
        .unwrap_or("local")
        .to_owned();
    event["ts"] = json!(chrono::Utc::now().to_rfc3339());
    event["project"] = json!(
        std::env::current_dir()?
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
    );
    let dir = root.join("tools/skill-vector/usage");
    fs::create_dir_all(&dir)?;
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(dir.join(format!("{host}.jsonl")))?;
    writeln!(file, "{event}")?;
    Ok(())
}

fn report_usage(root: &Path, weeks: u32) -> Result<()> {
    ensure!(weeks > 0, "--weeks must be positive");
    use chrono::{Datelike, NaiveDate, Utc};
    let cutoff = Utc::now().date_naive() - chrono::Duration::weeks(i64::from(weeks));
    let usage = root.join("tools/skill-vector/usage");
    let mut paths = if usage.exists() {
        fs::read_dir(&usage)?
            .map(|e| e.map(|e| e.path()))
            .collect::<std::io::Result<Vec<_>>>()?
    } else {
        Vec::new()
    };
    paths.retain(|p| p.extension().is_some_and(|e| e == "jsonl"));
    paths.sort();
    let mut totals: BTreeMap<String, [usize; 2]> = BTreeMap::new();
    let mut weekly: BTreeMap<(String, String), [usize; 2]> = BTreeMap::new();
    let mut projects: BTreeMap<(String, String), [usize; 2]> = BTreeMap::new();
    for path in paths {
        for (line_num, line) in fs::read_to_string(&path)?.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let context = || format!("Invalid usage event in {}:{}", path.display(), line_num + 1);
            let event: Value = serde_json::from_str(line).with_context(context)?;
            let date = event["ts"]
                .as_str()
                .and_then(|s| s.get(..10))
                .and_then(|s| NaiveDate::parse_from_str(s, "%Y-%m-%d").ok())
                .with_context(context)?;
            let skill = event["skill"].as_str().with_context(context)?;
            let kind = event["event"].as_str().with_context(context)?;
            if date < cutoff {
                continue;
            }
            let index = match kind {
                "retrieved" => 0,
                "used" => 1,
                _ => continue,
            };
            let project = match &event["project"] {
                Value::Null => "?".to_owned(),
                Value::String(s) if s.is_empty() => "?".to_owned(),
                Value::String(s) => s.clone(),
                Value::Object(o) if o.get("Unix").and_then(Value::as_array).is_some() => {
                    let bytes = o["Unix"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|v| {
                            v.as_u64()
                                .and_then(|b| u8::try_from(b).ok())
                                .with_context(context)
                        })
                        .collect::<Result<Vec<_>>>()?;
                    String::from_utf8_lossy(&bytes).into_owned()
                }
                _ => anyhow::bail!("{}", context()),
            };
            let iso = date.iso_week();
            let week = format!("{}-W{:02}", iso.year(), iso.week());
            totals.entry(skill.to_owned()).or_default()[index] += 1;
            weekly.entry((week, skill.to_owned())).or_default()[index] += 1;
            projects.entry((project, skill.to_owned())).or_default()[index] += 1;
        }
    }
    let esc = |s: &str| s.replace('|', "\\|");
    println!("近 {weeks} 周，含所有机器；retrieved 是返回次数，used 是显式登记次数。\n");
    println!("按技能\n| skill | retrieved | used |\n|---|---|---|");
    for (skill, n) in totals {
        println!("| {} | {} | {} |", esc(&skill), n[0], n[1]);
    }
    println!("\n按周与技能\n| week | skill | retrieved | used |\n|---|---|---|---|");
    for ((week, skill), n) in weekly {
        println!("| {} | {} | {} | {} |", esc(&week), esc(&skill), n[0], n[1]);
    }
    println!("\n按项目与技能\n| project | skill | retrieved | used |\n|---|---|---|---|");
    for ((project, skill), n) in projects {
        println!(
            "| {} | {} | {} | {} |",
            esc(&project),
            esc(&skill),
            n[0],
            n[1]
        );
    }
    println!();
    Ok(())
}

fn render(hits: &[Hit], format: Format) -> Result<()> {
    match format {
        Format::Json => println!("{}", serde_json::to_string_pretty(hits)?),
        Format::Text | Format::Markdown => {
            for h in hits {
                if matches!(format, Format::Markdown) {
                    println!("- [{}]({}) ({})", h.name, h.path, h.category);
                } else {
                    println!(
                        "{}\t{}\t{}",
                        h.score.map(|v| format!("{v:.4}")).unwrap_or_default(),
                        h.name,
                        h.path
                    );
                }
                if let Some(text) = &h.content {
                    println!("{text}");
                }
            }
        }
    }
    Ok(())
}
fn hit(app: &App, c: &Chunk, kind: Kind, score: Option<f32>, returns: Return) -> Result<Hit> {
    Ok(Hit {
        kind,
        name: c.name.clone(),
        area: c.area.clone(),
        category: c.category.clone(),
        path: app.root.join(&c.path).display().to_string(),
        relative_path: c.path.clone(),
        score,
        line: if returns == Return::Paths {
            None
        } else {
            c.line
        },
        content: match returns {
            Return::Paths => None,
            Return::Snippets => Some(c.text.clone()),
            Return::Full => Some(app.content(&c.path, kind)?),
        },
    })
}
fn run() -> Result<()> {
    let cli = Cli::parse_from(expand_nickname(std::env::args_os().collect()));
    let app = App::load(&cli)?;
    match cli.command {
        Action::Collection { action } => {
            #[cfg(any(feature = "metal-eval", feature = "test-backend"))]
            {
                let path = app.collection_path()?;
                let mut db = collections::open(&path)?;
                #[cfg(feature = "metal-eval")]
                let model_id = app.model_id();
                #[cfg(not(feature = "metal-eval"))]
                let model_id = "fixture-v1";
                match action {
                    CollectionAction::MigrateDocuments { name } => {
                        ensure!(
                            !app.cfg.workspace_collections,
                            "Migration requires the legacy configuration"
                        );
                        #[cfg(feature = "metal-eval")]
                        {
                            let _lock = lock_writer(
                                app.db.parent().context("Database has no parent")?,
                                true,
                            )?;
                            let source = Connection::open_with_flags(
                                &app.db,
                                OpenFlags::SQLITE_OPEN_READ_ONLY,
                            )?;
                            source.execute_batch("BEGIN")?;
                            let fingerprint = app.native_fingerprint()?;
                            let cfg = app.native_settings(&fingerprint);
                            let result = collections::migrate_workspace_documents(
                                &mut db,
                                &name,
                                &source,
                                &cfg,
                                || app.native_fingerprint(),
                            )?;
                            source.execute_batch("COMMIT")?;
                            println!("{result}");
                        }
                        #[cfg(not(feature = "metal-eval"))]
                        {
                            let _ = name;
                            anyhow::bail!("Native backend is unavailable");
                        }
                    }
                    CollectionAction::MigrateSkills { name } => {
                        ensure!(
                            !app.cfg.workspace_collections,
                            "Migration requires the legacy configuration"
                        );
                        let _lock =
                            lock_writer(app.db.parent().context("Database has no parent")?, true)?;
                        let count = collections::migrate_workspace_skills(
                            &mut db,
                            &name,
                            &app.root,
                            &app.skills,
                            model_id,
                        )?;
                        println!("{}", json!({"collection":name,"skills":count,"embedded":0}));
                    }
                    CollectionAction::Add {
                        name,
                        path,
                        no_shortcut,
                        include,
                        exclude,
                        result,
                        depth,
                        returns,
                    } => {
                        let reserved = reserved_nickname(&name);
                        ensure!(
                            !no_shortcut || reserved,
                            "--no-shortcut is only needed for reserved names"
                        );
                        ensure!(
                            !reserved || no_shortcut,
                            "Nickname '{name}' conflicts with a bsearch command. Reserved names: {}. Choose another nickname, or add this collection with --no-shortcut and search it with 'bsearch collection search {name} <query>'.",
                            reserved_names().join(", ")
                        );
                        ensure!(
                            result != "skill" || returns == Return::Paths,
                            "Skill collections return skill paths"
                        );
                        let returns = match returns {
                            Return::Paths => "paths",
                            Return::Snippets => "snippets",
                            Return::Full => "full",
                        };
                        let collection = collections::Collection {
                            name,
                            root: path,
                            include,
                            exclude,
                            result,
                            depth,
                            returns: returns.into(),
                        };
                        collections::add(&db, &collection, &app.root)?;
                        println!("{}", json!({"added":collection.name}));
                    }
                    CollectionAction::List => {
                        println!("{}", serde_json::to_string(&collections::list(&db)?)?)
                    }
                    CollectionAction::Scan { name, no_embed } => {
                        let _lock =
                            lock_writer(app.db.parent().context("Database has no parent")?, true)?;
                        #[cfg(feature = "metal-eval")]
                        let mut encoder: Option<
                            Box<dyn native_docs::Encoder>,
                        > = None;
                        let mode = collections::mode(&db, &name)?;
                        let result = if mode == "workspace" {
                            #[cfg(feature = "metal-eval")]
                            {
                                let fingerprint = app.native_fingerprint()?;
                                let cfg = app.native_settings(&fingerprint);
                                collections::scan_workspace(
                                    &mut db,
                                    &name,
                                    &cfg,
                                    || {
                                        ensure!(
                                            !no_embed,
                                            "New document embeddings required; --no-embed forbids model use"
                                        );
                                        app.model()
                                    },
                                    || app.native_fingerprint(),
                                )?
                            }
                            #[cfg(not(feature = "metal-eval"))]
                            {
                                anyhow::bail!("Native backend is unavailable")
                            }
                        } else if mode == "skill" {
                            collections::scan_skills(&mut db, &name, model_id, || {
                                ensure!(
                                    !no_embed,
                                    "New skill embeddings required; --no-embed forbids model use"
                                );
                                #[cfg(feature = "test-backend")]
                                if app.cfg.backend.as_deref() == Some("fixture") {
                                    return Ok(Box::new(FixtureEncoder(
                                        app.cfg
                                            .fixture_vector
                                            .clone()
                                            .context("Fixture vector missing")?,
                                    )));
                                }
                                #[cfg(feature = "metal-eval")]
                                {
                                    return app.model();
                                }
                                #[cfg(not(feature = "metal-eval"))]
                                anyhow::bail!("Metal backend is unavailable")
                            })?
                        } else {
                            collections::scan(&mut db, &name, model_id, |batch| {
                                ensure!(
                                    !no_embed,
                                    "New embeddings required; --no-embed forbids model use"
                                );
                                #[cfg(feature = "test-backend")]
                                if app.cfg.backend.as_deref() == Some("fixture") {
                                    return Ok(batch
                                        .iter()
                                        .map(|_| app.cfg.fixture_vector.clone().unwrap_or_default())
                                        .collect());
                                }
                                #[cfg(feature = "metal-eval")]
                                {
                                    if encoder.is_none() {
                                        encoder = Some(app.model()?);
                                    }
                                    return encoder.as_mut().unwrap().embed(batch);
                                }
                                #[cfg(not(feature = "metal-eval"))]
                                anyhow::bail!("Metal backend is unavailable")
                            })?
                        };
                        println!("{result}");
                    }
                    CollectionAction::Search {
                        name,
                        query,
                        scope,
                        returns,
                        format,
                        limit,
                        no_log,
                    } => {
                        ensure!(!query.trim().is_empty(), "Query cannot be empty");
                        let mode = collections::mode(&db, &name)?;
                        let q = app.query_vector(
                            &query,
                            if mode == "skill" {
                                Kind::Skill
                            } else {
                                Kind::Document
                            },
                        )?;
                        if mode == "workspace" {
                            collections::require_workspace_root(&db, &name, &app.root)?;
                        }
                        let returns = returns
                            .or_else(|| {
                                if mode == "workspace" {
                                    Some(
                                        app.kind_return(
                                            scope.as_deref().unwrap_or(""),
                                            Some(Kind::Document),
                                            None,
                                        )
                                        .1,
                                    )
                                } else {
                                    None
                                }
                            })
                            .map(|r| match r {
                                Return::Paths => "paths",
                                Return::Snippets => "snippets",
                                Return::Full => "full",
                            });
                        let hits = if mode == "skill" {
                            ensure!(
                                returns.is_none() || returns == Some("paths"),
                                "Skill collections return skill paths"
                            );
                            collections::search_skills(
                                &db,
                                &name,
                                scope.as_deref().unwrap_or(""),
                                limit,
                                model_id,
                                &q,
                            )?
                        } else {
                            let filter = |path: &str| app.eligible(path);
                            collections::search(
                                &db,
                                &name,
                                scope.as_deref().unwrap_or(""),
                                returns,
                                limit,
                                model_id,
                                &q,
                                if mode == "workspace" {
                                    Some(&filter)
                                } else {
                                    None
                                },
                            )?
                        };
                        if app.cfg.workspace_collections
                            && name == "skills"
                            && mode == "skill"
                            && !no_log
                        {
                            for (rank, hit) in hits.iter().enumerate() {
                                record_usage(
                                    &app.root,
                                    json!({"event":"retrieved","skill":hit.name,"query":query,"score":hit.score,"rank":rank+1,"chunk":format!("skills/{}",hit.relative_path),"source":"bsearch"}),
                                )?;
                            }
                        }
                        match format.unwrap_or(app.cfg.defaults.format) {
                            Format::Json => println!("{}", serde_json::to_string(&hits)?),
                            Format::Text | Format::Markdown => {
                                for hit in hits {
                                    println!(
                                        "{:.4}\t{}{}",
                                        hit.score,
                                        hit.path,
                                        hit.content.map(|c| format!("\n{c}")).unwrap_or_default()
                                    );
                                }
                            }
                        }
                    }
                }
            }
            #[cfg(not(any(feature = "metal-eval", feature = "test-backend")))]
            {
                let _ = action;
                anyhow::bail!("Collection backend unavailable");
            }
        }
        Action::HookWorker => {
            let dir = app.db.parent().context("Database has no parent")?;
            fs::create_dir_all(dir)?;
            let lock = fs::OpenOptions::new()
                .create(true)
                .write(true)
                .open(dir.join("hook-worker.lock"))?;
            if lock.try_lock().is_err() {
                return Ok(());
            }
            #[cfg(feature = "test-backend")]
            if app.cfg.backend.as_deref() == Some("fixture") {
                if let Some(path) = &app.cfg.fixture_worker_log {
                    let mut f = fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(path)?;
                    writeln!(f, "scan")?;
                    std::thread::sleep(std::time::Duration::from_millis(150));
                    writeln!(f, "skill-index")?;
                    std::thread::sleep(std::time::Duration::from_millis(150));
                }
                return Ok(());
            }
            #[cfg(feature = "metal-eval")]
            {
                let doc = app.native_scan(Kind::Document, false, true);
                let skill = app.native_scan(Kind::Skill, false, true);
                doc.context("Document refresh failed")?;
                skill.context("Skill refresh failed")?;
            }
            #[cfg(not(feature = "metal-eval"))]
            anyhow::bail!("Metal backend is unavailable in this build");
        }
        Action::Search {
            query,
            scope,
            kind,
            method,
            returns,
            format,
            limit,
            no_log,
        } => {
            ensure!(
                limit > 0 && !query.trim().is_empty(),
                "Query and positive limit required"
            );
            let scope = app.scope(scope.as_deref())?;
            let (kind, returns) = app.kind_return(&scope, kind, returns);
            let chunks = app.chunks(kind, &scope)?;
            if chunks.is_empty() {
                return render(&[], format.unwrap_or(app.cfg.defaults.format));
            }
            let _method = method.unwrap_or(app.cfg.defaults.method);
            let q = app.query_vector(&query, kind)?;
            let scores = scores(&chunks, &q)?;
            let mut best: HashMap<&str, (usize, f32)> = HashMap::new();
            for (i, c) in chunks.iter().enumerate() {
                let entry = best.entry(&c.path).or_insert((i, scores[i]));
                if scores[i] > entry.1 {
                    *entry = (i, scores[i]);
                }
            }
            let mut ranked: Vec<_> = best.values().copied().collect();
            ranked.sort_by(|a, b| {
                b.1.total_cmp(&a.1)
                    .then_with(|| chunks[a.0].path.cmp(&chunks[b.0].path))
            });
            let hits = ranked
                .iter()
                .take(limit)
                .map(|(i, s)| hit(&app, &chunks[*i], kind, Some(*s), returns))
                .collect::<Result<Vec<_>>>()?;
            if kind == Kind::Skill && !no_log {
                for (rank, h) in hits.iter().enumerate() {
                    record_usage(
                        &app.root,
                        json!({"event":"retrieved","skill":h.name,"query":query,
                        "score":h.score,"rank":rank+1,"chunk":h.relative_path,"source":"bsearch"}),
                    )?;
                }
            }
            render(&hits, format.unwrap_or(app.cfg.defaults.format))?;
        }
        Action::List {
            scope,
            kind,
            format,
        } => {
            let scope = app.scope(scope.as_deref())?;
            let (kind, _) = app.kind_return(&scope, kind, None);
            let chunks = app.objects(kind, &scope)?;
            let mut seen = HashMap::new();
            for c in &chunks {
                seen.entry(c.path.clone()).or_insert(c);
            }
            let mut paths: Vec<_> = seen.keys().collect();
            paths.sort();
            let hits = paths
                .iter()
                .map(|p| hit(&app, seen[*p], kind, None, Return::Paths))
                .collect::<Result<Vec<_>>>()?;
            render(&hits, format.unwrap_or(app.cfg.defaults.format))?;
        }
        Action::Get {
            path,
            from,
            lines,
            format,
        } => {
            ensure!(
                from > 0 && lines != Some(0),
                "Line numbers and counts must be positive"
            );
            let absolute = app.root.join(&path).canonicalize()?;
            ensure!(absolute.starts_with(&app.root), "Path outside workspace");
            let rel = absolute
                .strip_prefix(&app.root)?
                .to_string_lossy()
                .replace('\\', "/");
            let kind = if absolute.is_dir() {
                Kind::Skill
            } else {
                Kind::Document
            };
            let chunks = app.objects(kind, "")?;
            let c = chunks
                .iter()
                .find(|c| c.path == rel)
                .context("Path is not an indexed object")?;
            let mut item = hit(&app, c, kind, None, Return::Full)?;
            if from != 1 || lines.is_some() {
                item.content = Some(
                    item.content
                        .unwrap()
                        .lines()
                        .skip(from - 1)
                        .take(lines.unwrap_or(usize::MAX))
                        .collect::<Vec<_>>()
                        .join("\n"),
                );
            }
            render(&[item], format.unwrap_or(app.cfg.defaults.format))?;
        }
        Action::Scan {
            kind,
            background,
            no_embed,
        } => {
            if no_embed {
                ensure!(
                    kind == Kind::Document && !background,
                    "--no-embed requires a synchronous document scan"
                );
            }
            if background {
                ensure!(
                    kind == Kind::Document,
                    "Background scan starts both indexes; omit --kind skill"
                );
                app.dispatch(&json!({"hook_event_name":"SessionStart","cwd":app.root}))?;
            } else {
                #[cfg(feature = "metal-eval")]
                app.native_scan(kind, no_embed, false)?;
                #[cfg(not(feature = "metal-eval"))]
                anyhow::bail!("Metal backend is unavailable in this build");
            }
        }
        Action::Vectors { action } => {
            #[cfg(feature = "metal-eval")]
            {
                let dir = app.db.parent().context("Database has no parent")?;
                fs::create_dir_all(dir)?;
                let lock = fs::OpenOptions::new()
                    .create(true)
                    .write(true)
                    .open(dir.join("writer.lock"))?;
                lock.try_lock()
                    .map_err(|_| anyhow::anyhow!("An indexing operation is already running"))?;
                let fingerprint = app.native_fingerprint()?;
                let cfg = app.native_settings(&fingerprint);
                match action {
                    VectorAction::Export { path } => {
                        if app.cfg.workspace_collections {
                            native_portable::export_collections(&cfg, &app.db, &path, || {
                                app.native_fingerprint()
                            })?
                        } else {
                            native_portable::export(&cfg, &app.db, &app.skills, &path, || {
                                app.native_fingerprint()
                            })?
                        }
                    }
                    VectorAction::Import { path } => {
                        if app.cfg.workspace_collections {
                            native_portable::import_collections(&cfg, &app.db, &path, || {
                                app.native_fingerprint()
                            })?
                        } else {
                            native_portable::import(&cfg, &app.db, &app.skills, &path, || {
                                app.native_fingerprint()
                            })?
                        }
                    }
                }
            }
            #[cfg(not(feature = "metal-eval"))]
            anyhow::bail!("This binary was built without the experimental Metal backend");
        }
        Action::Used { skill } => {
            ensure!(
                app.objects(Kind::Skill, "")?
                    .iter()
                    .any(|c| c.name == skill),
                "Skill not present in index: {skill}"
            );
            record_usage(
                &app.root,
                json!({"event":"used","skill":skill,"source":"bsearch"}),
            )?;
            println!("Recorded skill usage: {skill}");
        }
        Action::Report { weeks } => report_usage(&app.root, weeks)?,
        Action::Status => {
            #[cfg(feature = "metal-eval")]
            {
                if app.cfg.workspace_collections {
                    let db = app.collection_db_readonly()?;
                    collections::require_workspace_root(&db, "workspace", &app.root)?;
                    collections::require_skill_root(&db, "skills", &app.root)?;
                    let dir = app.db.parent().context("Database has no parent")?;
                    let running = lock_writer(dir, false).is_err();
                    let counts = {
                        let mut q=db.prepare("SELECT status,count(*) FROM files WHERE collection='workspace' GROUP BY status")?;
                        q.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, usize>(1)?)))?
                            .collect::<rusqlite::Result<BTreeMap<_, _>>>()?
                    };
                    let areas = {
                        let mut q=db.prepare("SELECT area,count(*) FROM files WHERE collection='workspace' GROUP BY area")?;
                        q.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, usize>(1)?)))?
                            .collect::<rusqlite::Result<BTreeMap<_, _>>>()?
                    };
                    let chunks: usize = db.query_row(
                        "SELECT count(*) FROM chunks WHERE collection='workspace'",
                        [],
                        |r| r.get(0),
                    )?;
                    let vectors:usize=db.query_row("SELECT count(DISTINCT source_hash) FROM chunks WHERE collection='workspace'",[],|r|r.get(0))?;
                    let skills: usize = db.query_row(
                        "SELECT count(*) FROM skills WHERE collection='skills'",
                        [],
                        |r| r.get(0),
                    )?;
                    println!(
                        "{}",
                        serde_json::to_string_pretty(
                            &json!({"database":app.db,"running":running,"states":{"storage":"collections-v3"},"files":counts,"areas":areas,"chunks":chunks,"vectors_ready":vectors,"vectors_total":vectors,"skill_index":app.db,"skills":skills})
                        )?
                    );
                    return Ok(());
                }
                let db = app.db()?;
                let dir = app.db.parent().context("Database has no parent")?;
                let running = lock_writer(dir, false).is_err();
                let counts = {
                    let mut q = db.prepare("SELECT status,count(*) FROM files GROUP BY status")?;
                    q.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, usize>(1)?)))?
                        .collect::<rusqlite::Result<BTreeMap<_, _>>>()?
                };
                let chunks: usize =
                    db.query_row("SELECT count(*) FROM chunks", [], |r| r.get(0))?;
                let vectors: usize =
                    db.query_row("SELECT count(*) FROM vectors", [], |r| r.get(0))?;
                let areas = {
                    let mut q = db.prepare("SELECT area,count(*) FROM files GROUP BY area")?;
                    q.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, usize>(1)?)))?
                        .collect::<rusqlite::Result<BTreeMap<_, _>>>()?
                };
                let states = {
                    let mut q = db.prepare("SELECT key,value FROM state")?;
                    q.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))?
                        .collect::<rusqlite::Result<BTreeMap<_, _>>>()?
                };
                let total: usize =
                    db.query_row("SELECT count(DISTINCT hash) FROM chunks", [], |r| r.get(0))?;
                println!(
                    "{}",
                    serde_json::to_string_pretty(
                        &json!({"database":app.db,"running":running,"states":states,"files":counts,"areas":areas,"chunks":chunks,"vectors_ready":vectors,"vectors_total":total,"skill_index":app.skills})
                    )?
                );
            }
            #[cfg(not(feature = "metal-eval"))]
            anyhow::bail!("This binary was built without the experimental Metal backend");
        }
        Action::Config => println!(
            "{}",
            serde_json::to_string_pretty(
                &json!({"config":app.file,"root":app.root,"database":app.db,"skill_index":app.skills,"settings":serde_json::from_str::<Value>(&fs::read_to_string(&app.file)?)?})
            )?
        ),
        Action::Hook => {
            let mut input = String::new();
            io::stdin().read_to_string(&mut input)?;
            app.dispatch(&serde_json::from_str(&input)?)?;
        }
    }
    Ok(())
}

fn reserved_nickname(name: &str) -> bool {
    reserved_names().iter().any(|n| n == name)
}

fn reserved_names() -> Vec<String> {
    let mut names: Vec<_> = Cli::command()
        .get_subcommands()
        .map(|c| c.get_name().to_owned())
        .collect();
    names.push("help".to_owned());
    names.sort();
    names.dedup();
    names
}

// Translate `bsearch <nickname> <query>` into the existing collection search
// command, without intercepting options or built-in commands.
fn expand_nickname(mut args: Vec<std::ffi::OsString>) -> Vec<std::ffi::OsString> {
    let mut i = 1;
    while i < args.len() {
        let Some(arg) = args[i].to_str() else {
            return args;
        };
        if arg == "--config" || arg == "--db" {
            i += 2;
        } else if arg.starts_with("--config=") || arg.starts_with("--db=") {
            i += 1;
        } else if arg.starts_with('-') || reserved_nickname(arg) {
            return args;
        } else {
            args.insert(i, "collection".into());
            args.insert(i + 1, "search".into());
            return args;
        }
    }
    args
}
fn main() {
    if let Err(error) = run() {
        eprintln!("bsearch: {error:#}");
        std::process::exit(1);
    }
}
