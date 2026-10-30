---
name: bsearch
description: 本地向量检索工具 bSearch 的配置、使用与维护技能。用于引导用户指定并注册中央技能目录、添加管理自定义文档库（如个人知识库/笔记/项目文档）、执行精准语义检索与技能匹配，以及配置 Codex、Claude 和 Anti-Gravity 的 SessionStart 自动索引 Hook。
---

# bSearch 技能与知识库检索管理指南

bSearch 是专为 Apple Silicon 设计的本地 Rust 语义检索工具，采用本地 MLX/Metal 与 EmbeddingGemma 2 文本模型，完全在本地 SQLite 运行，不向任何外部模型接口发送文档正文。

---

## 一、 技能注册系统（Central Skill Library）

bSearch 支持专门的 `skill` 聚合模式。通过建立一个单一事实源目录管理全部 Agent 技能，配合 bSearch 的技能语义索引，Agent 能够根据任务需求一键找到最契合的技能名称。

### 1. 指定中央技能根目录
引导用户确定一个专门的技能存放目录，例如：
- 个人全局技能库：`~/skills/`
- 工作区集中技能库：`~/brainstorm/skills/`
- 客户端软链根：`~/.agents/skills/`（或 `~/.claude/skills/`, `~/.codex/skills/`）

在此目录下，**每一个直接子目录即为一个独立的技能单元**，且必须包含规范的 `SKILL.md`：
```text
~/skills/
├── git-helper/
│   └── SKILL.md
├── search-assistant/
│   └── SKILL.md
└── danish-bookkeeping/
    └── SKILL.md
```

### 2. 注册技能集合到 bSearch
在终端中执行以下命令将该目录注册为技能集合：

```bash
bsearch collection add skills "$HOME/skills" --mode skill
bsearch collection scan skills
```

> [!TIP]
> 也可以在 `config.json` 中配置，其内部映射为 `"result": "directory"`, `"depth": 1`, `"returns": "paths"`, `"mode": "skill"`。

### 3. bSearch 技能检索特点
- **按技能根聚合**：以一级子目录为独立单位，同一技能下可能包含多个分块，但最终取同技能所有片段中的**最高余弦相似度**作为该技能的得分。
- **排除代码实现，聚焦文本语义**：仅对 `SKILL.md`、说明文档文本及附件路径进行分块嵌入，**严格忽略代码体（code bodies）**。这样能确保搜索结果不被具体实现细节的代码片段带偏，专注匹配技能的触发意图（trigger description）。
- **直接输出技能名称**：检索返回的结果直接为该技能的一级目录路径或名称（例如 `git-helper`），Agent 可以直接依据输出决定激活哪个技能。

### 4. Agent 技能发现与调用规范
当用户向 Agent 提出复杂任务或寻求某种专门能力时，Agent 应主动调用 bSearch 进行技能检索：

```bash
# 检索能处理当前任务的技能
bsearch skills "帮我审查 git 提交历史并进行 rebase"
# 或完整命令
bsearch collection search skills "帮我审查 git 提交历史并进行 rebase"
```

输出示例：
```json
[
  {
    "collection": "skills",
    "name": "git-helper",
    "path": "/Users/.../skills/git-helper",
    "relative_path": "git-helper",
    "score": 0.812
  }
]
```
Agent 依据最高分直接加载并阅读 `/Users/.../skills/git-helper/SKILL.md` 开展工作。

---

## 二、 添加与管理自定义目录（以个人知识库为例）

除技能外，用户可以向 bSearch 注册任意数量的文本集合，例如个人知识库、会议纪要、项目文档库、博客草稿等。

### 1. 注册新集合（如知识库）
使用 `collection add` 命令：

```bash
# 添加名为 notes 的个人笔记集合
bsearch collection add notes "$HOME/Documents/notes" \
  --include '**/*.md' \
  --exclude '**/archive/**' \
  --return snippets
```

#### 参数说明：
- `<name>`：集合唯一名称（如 `notes`, `knowledge`, `workdocs`），注册后自动生成同名快捷命令 `bsearch notes "query"`。
- `<path>`：待索引的绝对目录路径。
- `--include '<glob>'`：包含的文件匹配规则（默认为 `**/*.md`，文件必须为 UTF-8 文本）。
- `--exclude '<glob>'`：排除规则（如排除归档 `archive/**`、依赖 `node_modules/**`、敏感目录等）。
- `--return <paths|snippets|full>`：
  - `paths`：仅返回文件路径（Token 开销最小）。
  - `snippets`：返回与 Query 最相关的匹配高亮片段（推荐，适合作为知识库索引检索）。
  - `full`：返回整个 Markdown 文档全文。
- `--format <json|markdown|text>`：输出格式（默认 `json`，适合 Agent 结构化解析；终端人类查阅可设为 `markdown`）。

### 2. 扫描与增量更新
每当知识库新增或修改文档后，触发扫描：

```bash
bsearch collection scan notes
```
bSearch 内部对文件计算 SHA-256，未变动的文件秒级跳过，仅对新增或变动的文档进行分块与向量计算。

### 3. 知识库检索示例
```bash
# 使用快捷短命令查询
bsearch notes "分布式系统 Paxos 与 Raft 对比"

# 限制在某一子目录下查询
bsearch collection search notes "数据库迁移记录" --scope "architecture/database"

# 获取格式化的 Markdown 结果
bsearch notes "设计决策" --format markdown
```

### 4. 集合管理命令速查
- **查看已注册集合**：`bsearch collection list`
- **移除集合**：`bsearch collection remove <name>`
- **查看索引状态与文件数**：`bsearch status`

---

## 三、 Agent 会话启动 Hook（SessionStart Hook 配置）

bSearch 内置了跨 Agent 的 `bsearch hook` 命令。在 Agent 启动新会话时，后台自动以孤儿子进程方式触发增量扫描，确保向量索引与磁盘文件永远保持同步，且**不卡顿前台会话交互**，并在上下文压缩（`compact`/`clear`）时自动静默忽略。

在各个主流 Agent 工具中配置如下：

### 1. Codex（ChatGPT CLI / IDE）
在项目根目录 `.codex/hooks.json`（或全局 `~/.codex/hooks.json`）中添加：

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

### 2. Claude Code
在项目根目录 `.claude/settings.json`（或全局 `~/.claude/settings.json`）中添加：

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

### 3. Anti-Gravity (AGY)
在项目根目录 `.agents/hooks.json`（或用户级 `~/.gemini/config/hooks.json`）中添加：

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

> [!NOTE]
> `bsearch hook` 接收来自各个 Agent 标准输入的 JSON 事件。若事件中的工作区路径与配置匹配，将派生后台工作进程 `bsearch hook-worker` 并立即返回 `{}`，对 Agent 无任何阻塞。

---

## 四、 常见问题与维护

1. **向量数据库文件位置**：
   数据库由配置中的 `"database"` 指定（默认在 `.cache/collections-*.sqlite3`）。数据库损坏或需要全量重建时，删除该 SQLite 文件并重新运行 `bsearch collection scan <name>` 即可。
2. **快捷名称与内置命令冲突**：
   集合名称避免与内置命令（如 `collection`, `search`, `scan`, `hook`, `status`）重名。若发生冲突，短命令将自动禁用，此时仍可使用完整长命令：`bsearch collection search <name> "query"`。
