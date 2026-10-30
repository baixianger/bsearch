//! Staged skill discovery and chunking; paths are relative to skill roots.
use crate::native_docs::Encoder;
use anyhow::{Context, Result, ensure};
use ignore::WalkBuilder;
use md5::{Digest, Md5};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fs,
    io::Write,
    path::{Path, PathBuf},
};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Chunk {
    pub file: String,
    pub heading: String,
    pub text: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub vector: Vec<f32>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Skill {
    pub md5: String,
    pub kind: String,
    pub area: String,
    pub path: String,
    pub category: String,
    pub chunks: Vec<Chunk>,
}

#[derive(Serialize, Deserialize)]
pub struct Index {
    pub backend: String,
    pub model: String,
    pub model_id: String,
    pub algorithm: String,
    pub skills: BTreeMap<String, Skill>,
}
const ALGORITHM: &str = "skill-v1-md5-sections-1000-max48";
fn backend_for_model(model_id: &str) -> &'static str {
    if model_id.starts_with("embeddinggemma2-") {
        "mlx-v2"
    } else {
        "metal"
    }
}

pub fn read_index(path: &Path, model_id: &str) -> Result<Index> {
    let value: Index = serde_json::from_slice(&fs::read(path)?)
        .with_context(|| format!("Invalid skill index {}", path.display()))?;
    ensure!(
        value.model_id == model_id && value.algorithm == ALGORITHM,
        "Skill index uses a different model or chunking algorithm"
    );
    Ok(value)
}
pub fn atomic_write(path: &Path, index: &Index) -> Result<()> {
    let parent = path.parent().context("Skill index has no parent")?;
    fs::create_dir_all(parent)?;
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_nanos();
    let tmp = parent.join(format!(".skill-index-{}-{nonce}.tmp", std::process::id()));
    let mut opts = fs::OpenOptions::new();
    opts.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }
    let mut file = opts.open(&tmp)?;
    let write = (|| -> Result<()> {
        serde_json::to_writer(&mut file, index)?;
        file.write_all(b"\n")?;
        file.sync_all()?;
        fs::rename(&tmp, path)?;
        Ok(())
    })();
    if write.is_err() {
        let _ = fs::remove_file(&tmp);
    }
    write
}

#[derive(Debug, Default)]
pub struct ScanResult {
    pub skills: usize,
    pub changed: usize,
    pub embedded: usize,
}

/// Build a complete snapshot for a collection without touching its live index.
/// Reuses an existing chunk only when its source fingerprint and text match.
pub fn build_collection<F>(
    directory: &Path,
    previous: Option<Index>,
    model_id: &str,
    model_factory: F,
) -> Result<(Index, ScanResult)>
where
    F: FnOnce() -> Result<Box<dyn Encoder>>,
{
    let discovered = discover_directory(directory)?;
    let mut result = ScanResult {
        skills: discovered.len(),
        ..Default::default()
    };
    let mut done = BTreeMap::new();
    let mut pending = Vec::new();
    if let Some(index) = &previous {
        ensure!(
            index.model_id == model_id && index.algorithm == ALGORITHM,
            "Incompatible skill snapshot"
        );
    }
    for (name, current) in &discovered {
        let reuse = previous
            .as_ref()
            .and_then(|i| i.skills.get(name))
            .filter(|s| {
                s.md5 == current.md5
                    && s.chunks.len() == current.chunks.len()
                    && s.chunks.iter().zip(&current.chunks).all(|(a, b)| {
                        a.file == b.file
                            && a.heading == b.heading
                            && a.text == b.text
                            && a.vector.len() == 768
                            && a.vector.iter().all(|v| v.is_finite())
                    })
            });
        if let Some(old) = reuse {
            let mut ready = old.clone();
            ready.path = current.path.clone();
            ready.category = current.category.clone();
            done.insert(name.clone(), ready);
        } else {
            pending.push((name.clone(), current.clone()));
        }
    }
    if !pending.is_empty() {
        let mut model = model_factory()?;
        for (name, mut skill) in pending {
            let texts: Vec<_> = skill.chunks.iter().map(|c| c.text.clone()).collect();
            let vectors = if texts.is_empty() {
                Vec::new()
            } else {
                model.embed(&texts)?
            };
            ensure!(
                vectors.len() == texts.len(),
                "Skill embedding count mismatch"
            );
            for (chunk, vector) in skill.chunks.iter_mut().zip(vectors) {
                ensure!(
                    vector.len() == 768 && vector.iter().all(|v| v.is_finite()),
                    "Invalid skill vector"
                );
                chunk.vector = vector;
            }
            result.changed += 1;
            result.embedded += skill.chunks.len();
            done.insert(name, skill);
        }
    }
    let current = discover_directory(directory)?;
    ensure!(
        current.len() == discovered.len()
            && current
                .iter()
                .all(|(n, s)| discovered.get(n).is_some_and(|x| x.md5 == s.md5)),
        "Skill sources changed during indexing; retry"
    );
    Ok((
        Index {
            backend: backend_for_model(model_id).into(),
            model: model_id.into(),
            model_id: model_id.into(),
            algorithm: ALGORITHM.into(),
            skills: done,
        },
        result,
    ))
}

/// Rebuild only changed skills, write checkpoints separately, then atomically
/// publish after a second source-inventory check. The caller owns the lock.
pub fn scan<F>(root: &Path, target: &Path, model_id: &str, model_factory: F) -> Result<ScanResult>
where
    F: FnOnce() -> Result<Box<dyn Encoder>>,
{
    let discovered = discover(root)?;
    let partial = target.with_extension("partial.json");
    let mut old = if target.exists() {
        read_index(target, model_id)?.skills
    } else {
        BTreeMap::new()
    };
    if partial.exists() {
        old.extend(read_index(&partial, model_id)?.skills);
    }
    let mut result = ScanResult {
        skills: discovered.len(),
        ..Default::default()
    };
    let mut done = BTreeMap::new();
    let mut pending = Vec::new();
    for (name, current) in &discovered {
        if let Some(previous) = old.get(name).filter(|s| {
            s.md5 == current.md5
                && s.chunks.len() == current.chunks.len()
                && s.chunks.iter().zip(&current.chunks).all(|(a, b)| {
                    a.file == b.file
                        && a.heading == b.heading
                        && a.text == b.text
                        && a.vector.len() == 768
                        && a.vector.iter().all(|v| v.is_finite())
                })
        }) {
            let mut ready = previous.clone();
            ready.path = current.path.clone();
            ready.category = current.category.clone();
            done.insert(name.clone(), ready);
        } else {
            pending.push((name.clone(), current.clone()));
        }
    }
    if !pending.is_empty() {
        let mut model = model_factory()?;
        for (i, (name, mut skill)) in pending.into_iter().enumerate() {
            let texts: Vec<_> = skill.chunks.iter().map(|c| c.text.clone()).collect();
            let vectors = if texts.is_empty() {
                Vec::new()
            } else {
                model.embed(&texts)?
            };
            ensure!(
                vectors.len() == texts.len(),
                "Skill embedding count mismatch"
            );
            for (chunk, vector) in skill.chunks.iter_mut().zip(vectors) {
                ensure!(
                    vector.len() == 768 && vector.iter().all(|v| v.is_finite()),
                    "Invalid skill vector"
                );
                chunk.vector = vector;
            }
            result.embedded += skill.chunks.len();
            result.changed += 1;
            done.insert(name, skill);
            if (i + 1) % 10 == 0 {
                atomic_write(
                    &partial,
                    &Index {
                        backend: backend_for_model(model_id).into(),
                        model: model_id.into(),
                        model_id: model_id.into(),
                        algorithm: ALGORITHM.into(),
                        skills: done.clone(),
                    },
                )?;
            }
        }
    }
    let current = discover(root)?;
    ensure!(
        current.len() == discovered.len()
            && current
                .iter()
                .all(|(n, s)| discovered.get(n).is_some_and(|x| x.md5 == s.md5)),
        "Skill sources changed during indexing; retry"
    );
    atomic_write(
        target,
        &Index {
            backend: backend_for_model(model_id).into(),
            model: model_id.into(),
            model_id: model_id.into(),
            algorithm: ALGORITHM.into(),
            skills: done,
        },
    )?;
    if partial.exists() {
        fs::remove_file(partial)?;
    }
    Ok(result)
}

fn sections(text: &str) -> Vec<(String, String)> {
    let mut parts = Vec::new();
    let mut current = String::new();
    for line in text.split_inclusive('\n') {
        if line.starts_with("## ") {
            parts.push(std::mem::take(&mut current));
            parts.push(line.trim_end_matches(['\r', '\n']).to_owned());
        } else {
            current.push_str(line);
        }
    }
    parts.push(current);
    let mut result = Vec::new();
    let mut intro = String::new();
    for (i, part) in parts.iter().enumerate() {
        if part.starts_with("## ") {
            result.push((
                part.trim().to_owned(),
                parts.get(i + 1).map(|s| s.trim()).unwrap_or("").to_owned(),
            ));
        } else if !part.trim().is_empty() {
            intro = part.trim().to_owned();
        }
    }
    if !intro.is_empty() {
        result.insert(0, ("(intro)".to_owned(), intro));
    }
    result
}
fn truncate(s: String) -> String {
    s.chars().take(1000).collect()
}
fn chunks(dir: &Path, files: &[PathBuf]) -> Result<Vec<Chunk>> {
    let name = dir
        .file_name()
        .and_then(|s| s.to_str())
        .context("Skill name must be UTF-8")?;
    let skill_md = dir.join("SKILL.md");
    let text = fs::read_to_string(&skill_md)?;
    let mut body = text.as_str();
    let mut description = "";
    if let Some(front) = text.strip_prefix("---\n") {
        if let Some((head, tail)) = front.split_once("\n---\n") {
            body = tail;
            description = head
                .lines()
                .find_map(|l| l.strip_prefix("description:"))
                .unwrap_or("")
                .trim();
        }
    }
    let mut out = Vec::new();
    if !description.is_empty() {
        out.push(Chunk {
            file: "SKILL.md".into(),
            heading: "(description)".into(),
            text: truncate(format!("{name} — {description}")),
            vector: Vec::new(),
        });
    }
    for (heading, content) in sections(body) {
        if !content.is_empty() {
            out.push(Chunk {
                file: "SKILL.md".into(),
                text: truncate(format!("{name} {heading}\n{content}")),
                heading,
                vector: Vec::new(),
            });
        }
    }
    for p in files {
        if *p == skill_md {
            continue;
        }
        let rel = p
            .strip_prefix(dir)?
            .to_str()
            .context("Non-UTF-8 skill path")?
            .to_owned();
        if p.extension()
            .and_then(|s| s.to_str())
            .is_some_and(|s| s.eq_ignore_ascii_case("md") || s.eq_ignore_ascii_case("txt"))
        {
            let data = fs::read(p)?;
            let text = String::from_utf8_lossy(&data);
            let mut parts = sections(&text);
            if parts.is_empty() {
                parts.push(("(no heading)".into(), text.to_string()));
            }
            for (heading, content) in parts {
                if !content.trim().is_empty() {
                    out.push(Chunk {
                        file: rel.clone(),
                        text: truncate(format!("{name} [{rel}] {heading}\n{content}")),
                        heading,
                        vector: Vec::new(),
                    });
                }
            }
        } else {
            out.push(Chunk {
                file: rel.clone(),
                heading: "(file)".into(),
                text: truncate(format!("{name} 附属文件: {rel}")),
                vector: Vec::new(),
            });
        }
    }
    out.truncate(48);
    Ok(out)
}
pub fn discover(root: &Path) -> Result<BTreeMap<String, Skill>> {
    discover_at(root, &root.join("skills"), true)
}

/// Discover a named skill directory outside a workspace; paths are relative to
/// its parent, and Git ignore rules apply when the directory is in a worktree.
#[allow(dead_code)]
pub fn discover_directory(skills_root: &Path) -> Result<BTreeMap<String, Skill>> {
    let skills_root = skills_root.canonicalize()?;
    let root = skills_root
        .parent()
        .context("Skill directory has no parent")?;
    discover_at(root, &skills_root, false)
}

fn discover_at(
    root: &Path,
    skills_root: &Path,
    require_git: bool,
) -> Result<BTreeMap<String, Skill>> {
    ensure!(skills_root.is_dir(), "Skill directory unavailable");
    let mut files = Vec::new();
    let mut builder = WalkBuilder::new(&skills_root);
    builder
        .hidden(true)
        .follow_links(false)
        .ignore(false)
        .git_ignore(true)
        .git_global(true)
        .git_exclude(true)
        .filter_entry(|e| e.file_name() != "__pycache__");
    for item in builder.build() {
        let item = item?;
        if item.file_type().is_some_and(|t| t.is_file())
            && !item
                .path()
                .strip_prefix(&skills_root)?
                .components()
                .any(|c| {
                    c.as_os_str()
                        .to_str()
                        .is_none_or(|p| p.starts_with('.') || p == "__pycache__")
                })
        {
            files.push(item.into_path());
        }
    }
    // `git check-ignore` (the previous implementation) does not report tracked
    // files, even if a later ignore rule matches them. The ignore walker alone
    // would drop them, so merge tracked paths from the Git index without running Git.
    let repo = match git2::Repository::discover(if require_git { root } else { skills_root }) {
        Ok(repo) => Some(repo),
        Err(e) if !require_git && e.code() == git2::ErrorCode::NotFound => None,
        Err(e) => return Err(e).context("Cannot open workspace Git repository"),
    };
    if let Some(repo) = repo {
        let workdir = repo
            .workdir()
            .context("A worktree is required for skill discovery")?;
        for entry in repo.index()?.iter() {
            let relative = std::str::from_utf8(&entry.path).context("Non-UTF-8 tracked path")?;
            let path = workdir.join(relative);
            if !path.starts_with(&skills_root) {
                continue;
            }
            let rel = path.strip_prefix(&skills_root)?;
            if rel.components().any(|c| {
                c.as_os_str()
                    .to_str()
                    .is_none_or(|p| p.starts_with('.') || p == "__pycache__")
            }) {
                continue;
            }
            if let Ok(meta) = fs::symlink_metadata(&path) {
                if meta.file_type().is_file()
                    && path
                        .canonicalize()?
                        .starts_with(skills_root.canonicalize()?)
                {
                    files.push(path);
                }
            }
        }
    }
    files.sort();
    files.dedup();
    let mut dirs: Vec<PathBuf> = files
        .iter()
        .filter(|p| p.file_name().is_some_and(|s| s == "SKILL.md"))
        .filter_map(|p| p.parent().map(Path::to_path_buf))
        .collect();
    dirs.sort();
    // Once a directory is itself a skill, nested SKILL.md files belong to it.
    let mut roots: Vec<PathBuf> = Vec::new();
    for dir in dirs {
        if !roots.iter().any(|p| dir.starts_with(p)) {
            roots.push(dir);
        }
    }
    let mut result = BTreeMap::new();
    for dir in roots {
        let name = dir
            .file_name()
            .and_then(|s| s.to_str())
            .context("Non-UTF-8 skill name")?
            .to_owned();
        let associated: Vec<_> = files
            .iter()
            .filter(|p| p.starts_with(&dir))
            .cloned()
            .collect();
        let mut hash = Md5::new();
        for p in &associated {
            hash.update(
                p.strip_prefix(&dir)?
                    .to_str()
                    .context("Non-UTF-8 skill path")?
                    .as_bytes(),
            );
            hash.update(fs::read(p)?);
        }
        let rel = dir
            .strip_prefix(root)?
            .to_str()
            .context("Non-UTF-8 skill path")?
            .to_owned();
        let category = Path::new(&rel)
            .parent()
            .and_then(|p| p.to_str())
            .unwrap_or("skills")
            .to_owned();
        let previous = result.insert(
            name.clone(),
            Skill {
                md5: format!("{:x}", hash.finalize()),
                kind: "skill".into(),
                area: "skills".into(),
                path: rel,
                category,
                chunks: chunks(&dir, &associated)?,
            },
        );
        ensure!(
            previous.is_none(),
            "Duplicate skill name '{name}'; rename one skill directory"
        );
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn duplicate_skill_names_fail_instead_of_silently_dropping_one() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("skills");
        for group in ["a", "b"] {
            let dir = root.join(group).join("same");
            fs::create_dir_all(&dir).unwrap();
            fs::write(dir.join("SKILL.md"), "# Same\ntext").unwrap();
        }
        let error = discover_directory(&root).unwrap_err().to_string();
        assert!(error.contains("Duplicate skill name"));
    }
    #[test]
    fn skill_library_at_git_root_keeps_tracked_ignored_files() {
        let temp = tempfile::tempdir().unwrap();
        let library = temp.path().join("scale");
        fs::create_dir_all(library.join("alpha")).unwrap();
        let repo = git2::Repository::init(&library).unwrap();
        fs::write(library.join(".gitignore"), "*.md\n").unwrap();
        fs::write(library.join("alpha/SKILL.md"), "# Alpha\ntracked").unwrap();
        fs::write(library.join("alpha/ignored.md"), "ignored content").unwrap();
        let mut index = repo.index().unwrap();
        index.add_path(Path::new("alpha/SKILL.md")).unwrap();
        index.write().unwrap();
        let skills = discover_directory(&library).unwrap();
        let skill = skills.get("alpha").unwrap();
        assert!(skill.chunks.iter().any(|c| c.text.contains("tracked")));
        assert!(
            skill
                .chunks
                .iter()
                .all(|c| !c.text.contains("ignored content"))
        );
    }
    #[test]
    fn discovers_named_non_git_skill_library_without_embedding_code() {
        let temp = tempfile::tempdir().unwrap();
        let library = temp.path().join("scale");
        fs::create_dir_all(library.join("alpha/scripts")).unwrap();
        fs::write(library.join("alpha/SKILL.md"), "# Alpha\nFind photos").unwrap();
        fs::write(library.join("alpha/scripts/helper.py"), "private code body").unwrap();
        let skills = discover_directory(&library).unwrap();
        let skill = skills.get("alpha").unwrap();
        assert_eq!(skill.path, "scale/alpha");
        assert!(
            skill
                .chunks
                .iter()
                .all(|chunk| !chunk.text.contains("private code body"))
        );
        assert!(
            skill
                .chunks
                .iter()
                .any(|chunk| chunk.text.contains("helper.py"))
        );
    }
    #[test]
    fn ignores_gitignored_data_tracks_text_and_does_not_embed_code_body() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().canonicalize().unwrap();
        git2::Repository::init(&root).unwrap();
        fs::write(root.join(".gitignore"), "*.secret\ncache/\n").unwrap();
        let skill = root.join("skills/example");
        fs::create_dir_all(&skill).unwrap();
        fs::write(
            skill.join("SKILL.md"),
            "---\ndescription: searchable purpose\n---\n# Title\n\n## Detail\nUseful instructions",
        )
        .unwrap();
        fs::write(skill.join("reference.md"), "Useful reference").unwrap();
        let before = discover(&root).unwrap().remove("example").unwrap();
        fs::write(skill.join("private.secret"), "sensitive content").unwrap();
        fs::create_dir(skill.join("cache")).unwrap();
        fs::write(skill.join("cache/local.md"), "cached data").unwrap();
        fs::create_dir(skill.join("profiles")).unwrap();
        fs::write(skill.join("profiles/.gitignore"), "*\n!.gitignore\n").unwrap();
        let after = discover(&root).unwrap().remove("example").unwrap();
        assert_eq!(before.md5, after.md5);
        assert_eq!(before.chunks, after.chunks);
        // Git's check-ignore skips tracked files even when their names match an ignore rule.
        fs::write(skill.join("tracked.secret"), "tracked fixture").unwrap();
        let repo = git2::Repository::open(&root).unwrap();
        let mut index = repo.index().unwrap();
        index
            .add_path(Path::new("skills/example/tracked.secret"))
            .unwrap();
        index.write().unwrap();
        let tracked = discover(&root).unwrap().remove("example").unwrap();
        assert!(tracked.chunks.iter().any(|c| c.file == "tracked.secret"));
        fs::write(skill.join("code.py"), "not to be embedded").unwrap();
        let changed = discover(&root).unwrap().remove("example").unwrap();
        assert_ne!(before.md5, changed.md5);
        let code = changed.chunks.iter().find(|c| c.file == "code.py").unwrap();
        assert_eq!(code.heading, "(file)");
        assert!(!code.text.contains("not to be embedded"));
        fs::write(skill.join("reference.md"), "Changed reference").unwrap();
        assert_ne!(changed.md5, discover(&root).unwrap()["example"].md5);
    }

    struct Fake;
    impl Encoder for Fake {
        fn embed(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            Ok(texts.iter().map(|_| vec![1.0; 768]).collect())
        }
    }
    struct Broken;
    impl Encoder for Broken {
        fn embed(&mut self, _: &[String]) -> Result<Vec<Vec<f32>>> {
            anyhow::bail!("injected failure")
        }
    }
    #[test]
    fn staged_index_is_incremental_and_failure_never_replaces_old_target() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().canonicalize().unwrap();
        git2::Repository::init(&root).unwrap();
        let skill = root.join("skills/example");
        fs::create_dir_all(&skill).unwrap();
        fs::write(skill.join("SKILL.md"), "# Example\nfirst").unwrap();
        let target = root.join("cache/skills.json");
        let first = scan(&root, &target, "model-a", || Ok(Box::new(Fake))).unwrap();
        assert_eq!((first.skills, first.changed), (1, 1));
        let second = scan(&root, &target, "model-a", || {
            anyhow::bail!("model must not load")
        })
        .unwrap();
        assert_eq!(second.changed, 0);
        let before = fs::read(&target).unwrap();
        fs::write(skill.join("SKILL.md"), "# Example\nchanged").unwrap();
        assert!(scan(&root, &target, "model-a", || Ok(Box::new(Broken))).is_err());
        assert_eq!(before, fs::read(&target).unwrap());
        assert!(scan(&root, &target, "model-b", || Ok(Box::new(Fake))).is_err());
        assert_eq!(before, fs::read(&target).unwrap());
        let updated = scan(&root, &target, "model-a", || Ok(Box::new(Fake))).unwrap();
        assert_eq!(updated.changed, 1);
        assert_ne!(before, fs::read(&target).unwrap());
    }
}
