---
summary: Set up local, configurable document and skill collections with the Rust bSearch CLI and a separately licensed MLX model.
---
# bSearch

A local Rust CLI for semantic search over personal text collections on Apple silicon. It uses MLX/Metal and a separately obtained EmbeddingGemma 2 model; indexing and queries do not send document content to a model API. No Python runtime is required.

## What it does

- Maintain multiple named collections in one local SQLite database. Include and exclude paths using globs; files must be UTF-8 text. Markdown is the default. PDF, image, audio and video parsing are not supported.
- Return matching documents, excerpts or full content, or aggregate by a fixed directory depth. The built-in `skill` mode ranks by the best chunk and returns the skill root; it embeds Markdown/TXT text and attachment paths, not code bodies.
- Search with `bsearch notes 'query'` or `bsearch collection search notes 'query'`. Nicknames cannot shadow reserved commands; use `--no-shortcut` and the long form if a name conflicts.
- Use hashes for incremental scanning. Scan a collection after its files change; automatic workspace hooks do not scan independent collections.

The existing workspace integration is a separate configuration. In particular, workspaces like brainstorm may continue to index only Markdown and exclude archive. Search supplements human/agent maintained catalogs; it does not register documents for you.

## Requirements and model

Currently tested on macOS Apple silicon with Xcode and its Metal toolchain, Rust, and CMake. Other platforms are not established as supported. Build dependencies may download upstream sources. Source and model weights are separate distributions. Review the model license and terms at [Google](https://huggingface.co/google/embeddinggemma-2) and [mlx-community](https://huggingface.co/mlx-community/embeddinggemma-2-bf16).

Download `model.safetensors`, `tokenizer.json`, and `config.json` from the [pinned MLX checkpoint](https://huggingface.co/mlx-community/embeddinggemma-2-bf16/tree/1a4ffddb7905d3f63486748deabe091a01fb6201). The inference code verifies the expected file hashes. For the text-only variant, use the included Rust extractor; its destination must not already exist:

```sh
mkdir -p "$HOME/Library/Caches/bsearch/models/embeddinggemma-2-bf16-text"
DEVELOPER_DIR="${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}" cargo run --release --locked --example extract_text_weights -- \
  /path/to/download/model.safetensors \
  "$HOME/Library/Caches/bsearch/models/embeddinggemma-2-bf16-text/model.safetensors"
cp /path/to/download/tokenizer.json /path/to/download/config.json \
  "$HOME/Library/Caches/bsearch/models/embeddinggemma-2-bf16-text/"
```

Alternatively set `model_path` in the config to the full checkpoint directory. Use only the exact supported checkpoint; changing model, tokenizer or inference semantics requires a separate compatible index or a rebuild. Do not mix incompatible vectors.

## Installation and first collection

```sh
cp config.example.json config.json
mkdir -p workspace
./install.sh
# Restart your shell or put ~/.local/bin on PATH.
bsearch --config "$PWD/config.json" collection add notes "$HOME/Documents/notes" --include '**/*.md' --exclude '**/archive/**' --return snippets
bsearch --config "$PWD/config.json" collection scan notes
bsearch --config "$PWD/config.json" notes 'design decisions'
bsearch --config "$PWD/config.json" collection search notes 'design decisions' --scope projects --return full --format json
```

Create the source directory first. `root` is a reserved workspace root and independent collection roots must not overlap it. Config paths are relative to the config file. The installer creates a global config link only when no config exists and a local `config.json` is present; it does not replace an existing configuration. You can instead use `BSEARCH_CONFIG` or `--config`. `--config` has priority.

## Skill Library Registration System

bSearch includes built-in support for managing a centralized Agent Skill Library via `--mode skill`:

### 1. Designate a Central Skills Directory
Choose a single source-of-truth directory for all your skills (e.g., `~/skills` or `~/.agents/skills`). Every immediate subdirectory represents an independent skill unit containing a standard `SKILL.md`:

```text
~/skills/
├── git-helper/
│   └── SKILL.md
├── search-assistant/
│   └── SKILL.md
└── danish-bookkeeping/
    └── SKILL.md
```

### 2. Register and Index the Skill Collection
```sh
bsearch collection add skills "$HOME/skills" --mode skill
bsearch collection scan skills
```

### 3. Skill Retrieval Characteristics
- **Directory Aggregation**: Ranks each skill by the highest cosine similarity among all its text chunks.
- **Code-Body Exclusion**: Embeds `SKILL.md`, documentation text, and referenced asset paths while **strictly ignoring code implementations**. This avoids code noise and focuses retrieval on agent trigger descriptions and workflows.
- **Skill Name Output**: Directly returns the skill root path and nickname (e.g. `git-helper`), allowing agents to immediately identify and adopt the required skill.

```sh
# Search for relevant skills by task description
bsearch skills 'review git commit history and assist with rebase'
```

---

## Adding Custom Collections & Knowledge Bases

You can register arbitrary text collections (personal knowledge bases, project documentation, meeting notes, etc.):

### 1. Register a Knowledge Base
```sh
bsearch collection add notes "$HOME/Documents/notes" \
  --include '**/*.md' \
  --exclude '**/archive/**' \
  --return snippets
```

Key options:
- `--include '<glob>'`: File matching pattern (defaults to `**/*.md`, files must be valid UTF-8).
- `--exclude '<glob>'`: Exclude patterns (e.g., `archive/**`, `node_modules/**`, private stores).
- `--return <paths|snippets|full>`: Choose between returning file paths, high-scoring snippets, or full file contents.
- `--format <json|markdown|text>`: Choose output format (`json` for agents, `markdown`/`text` for human reading).

### 2. Scan and Search
```sh
# Incremental scan (SHA-256 hash checks skip unchanged files)
bsearch collection scan notes

# Search using the generated shortcut command
bsearch notes 'distributed consensus Paxos vs Raft'

# Scope search to a subdirectory
bsearch collection search notes 'database schema' --scope 'architecture/database'
```

---

## Agent SessionStart Hooks (Codex, Claude Code & Anti-Gravity)

`bsearch hook` is a non-blocking SessionStart dispatcher. When a new session opens in your workspace, it reads event metadata from stdin and spawns an asynchronous background worker (`bsearch hook-worker`) to refresh indexes without delaying the agent. It automatically ignores context compaction/clear events.

### 1. Codex (`.codex/hooks.json`)
```json
{
  "hooks": {
    "SessionStart": [
      {
        "matcher": "^startup$",
        "hooks": [
          {
            "type": "command",
            "command": "\"$HOME/.local/bin/bsearch\" --config \"$HOME/.config/bsearch/config.json\" hook",
            "async": true
          }
        ]
      }
    ]
  }
}
```

### 2. Claude Code (`.claude/settings.json`)
```json
{
  "hooks": {
    "SessionStart": [
      {
        "matcher": "startup",
        "hooks": [
          {
            "type": "command",
            "command": "\"$HOME/.local/bin/bsearch\" --config \"$HOME/.config/bsearch/config.json\" hook",
            "async": true
          }
        ]
      }
    ]
  }
}
```

### 3. Anti-Gravity (`.agents/hooks.json` or `~/.gemini/config/hooks.json`)
```json
{
  "bsearch": {
    "SessionStart": [
      {
        "type": "command",
        "command": "\"$HOME/.local/bin/bsearch\" --config \"$HOME/.config/bsearch/config.json\" hook",
        "timeout": 10
      }
    ]
  }
}
```

---

## Bundled Agent Skill

For autonomous agent configuration and operation, a companion skill is available in `skills/bsearch/SKILL.md`. It teaches agents how to manage collections, perform semantic lookups, and configure workspace hooks automatically.

---

## Verify and license

`tests/run.sh` runs deterministic tests, checks formatting and runs a model parity fixture when the model is available. It defaults to `/Applications/Xcode.app`; set `DEVELOPER_DIR` if Xcode is elsewhere. This source is Apache-2.0; see `LICENSE`. The referenced mlx-vlm implementation is MIT; attribution is in `THIRD_PARTY_NOTICES`. Models and dependencies have separate licenses. This is a source release; binary distributors must audit their actual native and Rust dependencies and provide their required notices.
