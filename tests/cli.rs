#![cfg(feature = "test-backend")]
use rusqlite::{Connection, params};
use serde_json::{Value, json};
use std::{
    fs,
    io::Write,
    path::PathBuf,
    process::{Command, Output},
};
use tempfile::TempDir;

struct Fixture {
    dir: TempDir,
    config: PathBuf,
    root: PathBuf,
}
impl Fixture {
    fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("workspace");
        fs::create_dir(&root).unwrap();
        for (path, text) in [
            (
                "library/notes/a.md",
                "# Alpha\nFull document body\nthird line",
            ),
            ("library/notes-old/b.md", "Outside selected notes"),
            ("archive/old.md", "Excluded archive"),
            ("library/notes/a.pdf", "Excluded format"),
            ("tools/sample/README.md", "Tool description"),
            ("skills/A/SKILL.md", "# Skill A\nSkill instructions"),
            ("skills/B/SKILL.md", "# Skill B"),
            (
                "skills/A/scripts/helper.py",
                "internal code must not become an object",
            ),
        ] {
            let file = root.join(path);
            fs::create_dir_all(file.parent().unwrap()).unwrap();
            fs::write(file, text).unwrap();
        }
        let db = Connection::open(dir.path().join("docs.sqlite3")).unwrap();
        db.execute_batch("CREATE TABLE files(path TEXT PRIMARY KEY,title TEXT,area TEXT,category TEXT,status TEXT);
          CREATE TABLE chunks(path TEXT,ordinal INTEGER,line INTEGER,text TEXT,hash TEXT);
          CREATE TABLE vectors(hash TEXT,model TEXT,vector BLOB);").unwrap();
        for (i, (path, text, v)) in [
            ("library/notes/a.md", "irrelevant fragment", [0f32, 1.]),
            ("library/notes/a.md", "alpha fragment", [1., 0.]),
            ("library/notes-old/b.md", "alpha outside", [1., 0.]),
            ("archive/old.md", "alpha archive", [1., 0.]),
            ("library/notes/a.pdf", "alpha pdf", [1., 0.]),
            ("tools/sample/README.md", "tool keyword", [1., 0.]),
        ]
        .iter()
        .enumerate()
        {
            let area = path.split('/').next().unwrap();
            db.execute(
                "INSERT OR IGNORE INTO files VALUES (?1,'title',?2,?3,'ready')",
                params![path, area, path.rsplit_once('/').unwrap().0],
            )
            .unwrap();
            let hash = format!("hash{i}");
            db.execute(
                "INSERT INTO chunks VALUES (?1,?2,1,?3,?4)",
                params![path, i, text, hash],
            )
            .unwrap();
            let bytes: Vec<u8> = v.iter().flat_map(|f| f.to_le_bytes()).collect();
            db.execute(
                "INSERT INTO vectors VALUES (?1,'test',?2)",
                params![hash, bytes],
            )
            .unwrap();
        }
        fs::write(dir.path().join("skills.json"),json!({"skills":{
            "A":{"path":"skills/A","category":"skills","chunks":[
                {"file":"SKILL.md","text":"unrelated","vector":[0.,1.]},
                {"file":"scripts/helper.py","text":"best fragment","vector":[1.,0.]}]},
            "B":{"path":"skills/B","category":"skills","chunks":[{"text":"other","vector":[0.7,0.7]}]}
        }}).to_string()).unwrap();
        let config = dir.path().join("config.json");
        fs::write(&config,json!({"root":"workspace","database":"docs.sqlite3","skill_index":"skills.json",
            "document_roots":["library","tools"],"include":["**/*.md"],"exclude":["archive/**"],
            "entries":{"skills":{"kind":"skill","return":"paths"},"library":{"kind":"document","return":"snippets"}},
            "backend":"fixture","fixture_vector":[1,0]
        }).to_string()).unwrap();
        Self { dir, config, root }
    }
    fn run(&self, args: &[&str]) -> Output {
        Command::new(env!("CARGO_BIN_EXE_bsearch"))
            .arg("--config")
            .arg(&self.config)
            .args(args)
            .output()
            .unwrap()
    }
    fn json(&self, args: &[&str]) -> Value {
        let out = self.run(args);
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        serde_json::from_slice(&out.stdout).unwrap()
    }
}

#[test]
fn directory_scope_filters_before_top_k_and_respects_boundary() {
    let f = Fixture::new();
    let rows = f.json(&[
        "search",
        "alpha",
        "--scope",
        "library/notes",
        "--return",
        "paths",
    ]);
    assert_eq!(rows.as_array().unwrap().len(), 1);
    assert_eq!(rows[0]["relative_path"], "library/notes/a.md");
    assert!(rows[0]["content"].is_null());
    assert_eq!(rows[0]["category"], "library/notes");
    assert_eq!(rows[0]["score"], 1.0);
}
#[test]
fn excludes_archive_and_non_markdown_even_if_old_database_contains_them() {
    let f = Fixture::new();
    let rows = f.json(&["search", "alpha", "--return", "paths"]);
    assert_eq!(rows.as_array().unwrap().len(), 3);
    assert!(
        rows.as_array().unwrap().iter().all(|r| r["relative_path"]
            .as_str()
            .unwrap()
            .ends_with(".md")
            && !r["relative_path"].as_str().unwrap().starts_with("archive/"))
    );
    assert_eq!(
        f.json(&["search", "alpha", "--scope", "archive"]),
        json!([])
    );
}
#[test]
fn skill_uses_max_chunk_score_and_returns_only_fixed_skill_root() {
    let f = Fixture::new();
    let rows = f.json(&["search", "query", "--scope", "skills"]);
    assert_eq!(rows.as_array().unwrap().len(), 2);
    assert_eq!(rows[0]["name"], "A");
    assert_eq!(rows[0]["score"], 1.0);
    assert_eq!(rows[0]["relative_path"], "skills/A");
    assert!(rows[0]["content"].is_null() && rows[0]["line"].is_null());
    assert_eq!(
        f.json(&["search", "query", "--scope", "skills/A/scripts"]),
        json!([])
    );
}
#[test]
fn full_document_and_get_line_ranges_are_actual_markdown_content() {
    let f = Fixture::new();
    let rows = f.json(&[
        "search",
        "alpha",
        "--scope",
        "library/notes",
        "--return",
        "full",
    ]);
    assert_eq!(
        rows[0]["content"],
        "# Alpha\nFull document body\nthird line"
    );
    let rows = f.json(&["get", "library/notes/a.md", "--from", "2", "--lines", "1"]);
    assert_eq!(rows[0]["content"], "Full document body");
    assert_eq!(
        f.json(&["get", "skills/A"])[0]["content"],
        "# Skill A\nSkill instructions"
    );
}
#[test]
fn vector_search_and_config_defaults_apply() {
    let f = Fixture::new();
    let rows = f.json(&["search", "alpha", "--scope", "library/notes"]);
    assert_eq!(rows[0]["content"], "alpha fragment");
    assert!(
        !f.run(&["search", "alpha", "--method", "keyword"])
            .status
            .success()
    );
    assert!(
        !f.run(&["search", "alpha", "--method", "hybrid"])
            .status
            .success()
    );
    assert_eq!(
        f.json(&["list", "--scope", "tools"])[0]["relative_path"],
        "tools/sample/README.md"
    );
}

#[test]
fn rejects_outside_scope_and_unindexed_content_and_wrong_vector_dimension() {
    let f = Fixture::new();
    assert!(
        !f.run(&["search", "query", "--scope", "../../"])
            .status
            .success()
    );
    assert!(!f.run(&["get", "library/notes/a.pdf"]).status.success());
    let outside = f.dir.path().join("outside.md");
    fs::write(&outside, "outside").unwrap();
    assert!(!f.run(&["get", outside.to_str().unwrap()]).status.success());
    let mut cfg: Value = serde_json::from_str(&fs::read_to_string(&f.config).unwrap()).unwrap();
    cfg["fixture_vector"] = json!([1, 0, 0]);
    fs::write(&f.config, cfg.to_string()).unwrap();
    let out = f.run(&["search", "query"]);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("dimensions differ"));
    assert!(f.root.is_dir());
}

#[test]
fn session_hooks_support_both_agents_and_refresh_both_indexes_in_background() {
    let f = Fixture::new();
    let marker = f.dir.path().join("worker-events");
    let mut config: Value = serde_json::from_str(&fs::read_to_string(&f.config).unwrap()).unwrap();
    config["fixture_worker_log"] = json!(marker);
    fs::write(&f.config, config.to_string()).unwrap();
    let dispatch = |event: Value| {
        let mut child = Command::new(env!("CARGO_BIN_EXE_bsearch"))
            .arg("--config")
            .arg(&f.config)
            .arg("hook")
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        child
            .stdin
            .take()
            .unwrap()
            .write_all(event.to_string().as_bytes())
            .unwrap();
        let output = child.wait_with_output().unwrap();
        assert!(output.status.success());
        assert_eq!(
            serde_json::from_slice::<Value>(&output.stdout).unwrap(),
            json!({})
        );
    };
    dispatch(json!({"conversationId":"test","workspacePaths":[f.dir.path()]}));
    assert!(!marker.exists(), "outside workspace must not start worker");
    dispatch(json!({"hook_event_name":"SessionStart","cwd":f.root,"source":"compact"}));
    dispatch(json!({"hook_event_name":"SessionStart","cwd":f.root,"source":"clear"}));
    assert!(
        !marker.exists(),
        "compaction or clear must not start another scan"
    );
    dispatch(json!({"conversationId":"test","workspacePaths":[f.root]}));
    dispatch(json!({"hook_event_name":"SessionStart","cwd":f.root}));
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    loop {
        let text = fs::read_to_string(&marker).unwrap_or_default();
        if text.lines().count() >= 2 {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "background worker did not refresh both indexes: {text}"
        );
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    let worker_lock = fs::OpenOptions::new()
        .write(true)
        .open(f.dir.path().join("hook-worker.lock"))
        .unwrap();
    loop {
        if worker_lock.try_lock().is_ok() {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "worker did not finish"
        );
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    let events = fs::read_to_string(&marker).unwrap();
    assert_eq!(
        events.lines().collect::<Vec<_>>(),
        vec!["scan", "skill-index"]
    );
}

#[test]
fn report_counts_recent_usage_and_legacy_project_encoding() {
    let f = Fixture::new();
    let usage = f.root.join("tools/skill-vector/usage");
    fs::create_dir_all(&usage).unwrap();
    let today = chrono::Utc::now()
        .format("%Y-%m-%dT12:00:00+00:00")
        .to_string();
    let events = [
        json!({"ts":today,"skill":"A","event":"retrieved","project":{"Unix":[65]}}),
        json!({"ts":today,"skill":"A","event":"used","project":"A"}),
        json!({"ts":"2000-01-01T00:00:00Z","skill":"old","event":"used"}),
    ];
    fs::write(
        usage.join("host.jsonl"),
        events
            .iter()
            .map(Value::to_string)
            .collect::<Vec<_>>()
            .join("\n"),
    )
    .unwrap();
    let out = f.run(&["report", "--weeks", "2"]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8(out.stdout).unwrap();
    assert!(text.contains("| A | 1 | 1 |"));
    assert!(text.contains("| A | A | 1 | 1 |"));
    assert!(!text.contains("| old |"));
    fs::write(usage.join("host.jsonl"), "invalid json\n").unwrap();
    let out = f.run(&["report"]);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("host.jsonl:1"));
}

#[test]
fn skill_usage_and_report_preserve_retrieval_and_used_counts() {
    let f = Fixture::new();
    let dir = f.root.join("tools/skill-vector/usage");
    fs::create_dir_all(&dir).unwrap();
    fs::write(
        dir.join("old.jsonl"),
        "{\"ts\":\"2099-01-01T00:00:00Z\",\"project\":{\"Unix\":[108,101,103,97,99,121]},\"event\":\"retrieved\",\"skill\":\"A\"}\n{\"ts\":\"2099-01-01T00:00:00Z\",\"project\":\"legacy\",\"event\":\"used\",\"skill\":\"A\"}\n",
    )
    .unwrap();
    assert!(
        f.run(&["search", "query", "--scope", "skills", "-k", "1"])
            .status
            .success()
    );
    assert!(f.run(&["used", "A"]).status.success());
    assert!(!f.run(&["used", "missing"]).status.success());
    let report = f.run(&["report"]);
    assert!(report.status.success());
    let report = String::from_utf8(report.stdout).unwrap();
    assert!(report.contains("| A | 2 | 2 |"));
    assert!(report.contains("| legacy | A | 1 | 1 |"));
    assert!(report.contains("| 2099-W01 | A | 1 | 1 |"));
}

#[test]
fn independent_collections_filter_increment_and_aggregate_directories() {
    let f = Fixture::new();
    let one = f.dir.path().join("notes");
    let two = f.dir.path().join("more");
    fs::create_dir_all(one.join("A/deep")).unwrap();
    fs::create_dir_all(one.join("B")).unwrap();
    fs::create_dir_all(&two).unwrap();
    fs::write(one.join("A/deep/readme.md"), "alpha markdown").unwrap();
    fs::write(one.join("A/deep/note.txt"), "alpha text").unwrap();
    fs::write(one.join("B/ignore.md"), "excluded").unwrap();
    fs::write(two.join("other.md"), "second collection").unwrap();
    assert!(
        f.run(&[
            "collection",
            "add",
            "first",
            one.to_str().unwrap(),
            "--include",
            "**/*.{md,txt}",
            "--exclude",
            "B/**",
            "--result",
            "directory"
        ])
        .status
        .success()
    );
    assert!(
        f.run(&["collection", "add", "second", two.to_str().unwrap()])
            .status
            .success()
    );
    assert!(
        !f.run(&["collection", "add", "protected", f.root.to_str().unwrap()])
            .status
            .success()
    );
    let first = f.json(&["collection", "scan", "first"]);
    assert_eq!(first["seen"], 2);
    assert_eq!(first["embedded"], 2);
    assert_eq!(f.json(&["collection", "scan", "first"])["embedded"], 0);
    assert_eq!(f.json(&["collection", "scan", "second"])["seen"], 1);
    let hits = f.json(&["collection", "search", "first", "alpha"]);
    assert_eq!(hits.as_array().unwrap().len(), 1);
    assert_eq!(hits[0]["relative_path"], "A");
    assert!(hits[0]["content"].is_null());
    assert_eq!(f.json(&["first", "alpha"]), hits);
    let conflict = f.run(&["collection", "add", "search", two.to_str().unwrap()]);
    assert!(!conflict.status.success());
    let error = String::from_utf8_lossy(&conflict.stderr);
    assert!(
        error.contains("Reserved names:")
            && error.contains("search")
            && error.contains("--no-shortcut")
    );
    assert!(
        f.run(&[
            "collection",
            "add",
            "search",
            two.to_str().unwrap(),
            "--no-shortcut"
        ])
        .status
        .success()
    );
    assert_eq!(f.json(&["collection", "scan", "search"])["seen"], 1);
    assert_eq!(
        f.json(&["collection", "search", "search", "other"])[0]["relative_path"],
        "other.md"
    );
    assert_eq!(
        f.json(&["collection", "search", "first", "alpha", "--scope", "B"]),
        json!([])
    );
    assert!(
        !f.run(&[
            "collection",
            "search",
            "first",
            "alpha",
            "--scope",
            "../other"
        ])
        .status
        .success()
    );
    let hits = f.json(&[
        "collection",
        "search",
        "second",
        "other",
        "--return",
        "full",
    ]);
    assert_eq!(hits[0]["content"], "second collection");
    fs::write(one.join("A/deep/readme.md"), "changed").unwrap();
    assert_eq!(f.json(&["collection", "scan", "first"])["changed"], 1);
    fs::write(one.join("A/deep/note.txt"), [0xff, 0xfe]).unwrap();
    assert!(!f.run(&["collection", "scan", "first"]).status.success());
    assert_eq!(
        f.json(&["collection", "search", "first", "alpha"])
            .as_array()
            .unwrap()
            .len(),
        1
    );
}

#[test]
fn skill_collection_uses_nickname_incremental_scan_and_skill_roots() {
    let f = Fixture::new();
    let mut cfg: Value = serde_json::from_slice(&fs::read(&f.config).unwrap()).unwrap();
    cfg["fixture_vector"] = json!(vec![1.0_f32; 768]);
    fs::write(&f.config, cfg.to_string()).unwrap();
    let root = f.dir.path().join("external-skills");
    fs::create_dir_all(root.join("alpha/scripts")).unwrap();
    fs::create_dir_all(root.join("alpha/nested")).unwrap();
    fs::create_dir_all(root.join("beta")).unwrap();
    fs::write(root.join("alpha/SKILL.md"), "# Alpha\nfind photos").unwrap();
    fs::write(root.join("alpha/nested/SKILL.md"), "nested content").unwrap();
    fs::write(root.join("alpha/scripts/a.py"), "private code").unwrap();
    fs::write(root.join("beta/SKILL.md"), "# Beta\nwrite notes").unwrap();
    let add = f.run(&[
        "collection",
        "add",
        "my-skills",
        root.to_str().unwrap(),
        "--mode",
        "skill",
    ]);
    assert!(
        add.status.success(),
        "{}",
        String::from_utf8_lossy(&add.stderr)
    );
    let first = f.json(&["collection", "scan", "my-skills"]);
    assert_eq!(first["skills"], 2);
    assert_eq!(first["changed"], 2);
    assert_eq!(f.json(&["collection", "scan", "my-skills"])["embedded"], 0);
    let hits = f.json(&["my-skills", "photos"]);
    assert_eq!(hits.as_array().unwrap().len(), 2);
    assert_eq!(hits[0]["relative_path"], "alpha");
    assert!(hits[0]["content"].is_null());
    assert_eq!(
        f.json(&["my-skills", "photos", "--scope", "alpha/nested"]),
        json!([])
    );
    fs::write(root.join("alpha/SKILL.md"), "# Alpha\nchanged").unwrap();
    assert_eq!(f.json(&["collection", "scan", "my-skills"])["changed"], 1);
    assert_eq!(
        f.json(&["my-skills", "photos"]).as_array().unwrap().len(),
        2
    );
}

#[test]
#[cfg(unix)]
fn global_configuration_allows_search_outside_the_workspace() {
    let f = Fixture::new();
    let home = f.dir.path().join("home");
    let config_dir = home.join(".config/bsearch");
    fs::create_dir_all(&config_dir).unwrap();
    std::os::unix::fs::symlink(&f.config, config_dir.join("config.json")).unwrap();
    let project = f.dir.path().join("unrelated-project");
    fs::create_dir_all(&project).unwrap();
    fs::write(
        project.join("Cargo.toml"),
        "[package]\nname='other'\nversion='0.1.0'\n",
    )
    .unwrap();
    fs::write(project.join("config.json"), "{\"unrelated\":true}").unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_bsearch"))
        .current_dir(&project)
        .env("HOME", &home)
        .env_remove("BSEARCH_CONFIG")
        .args(["search", "alpha", "--scope", "library/notes"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let rows: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(rows[0]["relative_path"], "library/notes/a.md");
}
