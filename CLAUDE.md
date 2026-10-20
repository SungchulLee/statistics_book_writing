# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

MkDocs Material documentation site for a mathematical textbook. Content uses LaTeX (MathJax)
and includes Python example scripts. Three specialized review/write agents live in `agents/`;
their full instructions are in `agents/SKILL.md`.

## Build Commands

```bash
mkdocs serve          # local dev with live reload
mkdocs build          # build static site
mkdocs build --strict # used in CI — all warnings are errors
pip install -r requirements.txt
```

## Deployment

GitHub Actions (`.github/workflows/deploy-mkdocs.yml`) auto-deploys to GitHub Pages on push
to `main`. Build uses `--strict` mode.

## Repository Structure

```
book_name/
├── CLAUDE.md
├── README.md
├── mkdocs.yml
├── requirements.txt
├── agents/
│   ├── SKILL.md            ← agent orchestration hub (read this first)
│   ├── MATH_REVIEWER.md
│   ├── WRITING_REVIEWER.md
│   └── WRITER.md
├── .github/workflows/deploy-mkdocs.yml
└── docs/
    ├── index.md
    ├── assets/favicon.ico
    ├── stylesheets/extra.css
    ├── javascripts/mathjax.js
    └── chapter_name/
        ├── index.md
        └── section_name/
            ├── topic.md
            ├── topic.py
            ├── module/__init__.py
            └── figures/
```

## Agent Commands (Quick Reference)

> Full command semantics, file conventions, and agent prompts: **read `agents/SKILL.md` first**.

| Command | What it does |
|---|---|
| `review <file\|folder\|all>` | Freeze vN snapshot, run both reviewers, print to stdout. No writes. |
| `write <file\|folder\|all> [if score < N]` | Requires prior `review`. Runs reviewers in-memory, writes improved file, updates score. |
| `update <file\|folder\|all> [if score < N]` | `review` + `write` in one step. |

**Path convention**: paths are relative to `docs/` — omit the `docs/` prefix.

**Execution rules**: sequential only · batch size = 1 · commit after each file.

## File Management

| File | Git | GitHub Pages |
|---|---|---|
| `<name>.md` | ✅ committed | ✅ published |
| `<name>_score.md` | ✅ committed | ❌ excluded via `mkdocs.yml` |
| `<name>_v[0-9]*.md` | ❌ gitignored | ❌ never built |

`.gitignore` entry: `docs/**/*_v[0-9]*.md`

`mkdocs.yml` exclusion:
```yaml
exclude_docs: |
  *_score.md
  *_v[0-9]*.md
```

Commit after `update` — stage only `<name>.md` and `<name>_score.md`:
```bash
git add docs/path/to/<name>.md docs/path/to/<name>_score.md
git commit -m "update: <name>"
```

## Navigation Structure

Nav hierarchy: **Parts → Chapters → Sections → Pages**. Nav entries point to `.md` and `.py`
files only.

```yaml
nav:
  - I Part Title:
    - 1 Chapter Title:
      - Chapter Overview: ch01/index.md
      - 1.1 Section Title:
        - Topic Title: ch01/section_title/topic_title.md
```

YAML quoting: quote any title containing `:`, `#`, `*`, `&`.

## Content Conventions (Summary)

Full MathJax/admonition rules are in `agents/SKILL.md`. Key points:

- `$...$` inline math, `$$...$$` display — always blank lines above and below display math
- No blank lines *inside* `$$...$$` blocks
- No LaTeX in `#` headings (breaks TOC)
- `\$` for currency, never bare `$`
- QED: `$\square$`
- Every content page ends with `## Exercises` (interleaved solutions, collapsible)
- Python: module docstring · `# ===` dividers · `if __name__ == "__main__":` guard
- 두 연재 상자 — `<div class="storybox">` 「그림으로 보는 통계학」(호박·점선),
  `<div class="aibox">` 「AI로 보는 통계학」(청록·왼쪽 띠). 증명·문항 사이클 밖이며
  이야기하는 정의/정리 **바로 뒤**에 놓는다. 전체 규칙은 `agents/SKILL.md`
- **출처 줄** — 상자가 책 바깥의 사실(사람·연도·학술지·쪽수·현실의 수치)을 하나라도
  주장하면 마지막 줄은 `_확인한 곳 — ⟨확인한 것⟩. 그림과 표의 수는 모두 직접 계산한 값이다_`
  로 닫는다. **확인한 것만 적는다** — 확인하지 않은 사실은 출처 줄에 올리지 말고 본문에서
  뺀다. 수학만 하는 상자는 이 줄을 붙이지 않는다

## Common Tasks

### Add a new section
1. Create `docs/chapter_name/section_name/` with `.md`/`.py` files
2. Add nav entries to `mkdocs.yml` (`.md` and `.py` only)

### Add a new chapter
1. Create `docs/chapter_name/` with `index.md` and section subdirectories
2. Add chapter block to `mkdocs.yml` under the correct Part

### 장·절을 옮기거나 합칠 때 (점검 목록)

`mkdocs build --strict` 는 **링크와 파일만** 검사한다. 4~6 번은 글자라서 빌드가
잡지 못하고, 실제로 매번 여기서 누락이 난다. 순서대로 전수로 훑을 것.

1. **nav** — `mkdocs.yml` 의 경로·제목·절 번호
2. **상대링크와 그림** — `git mv` 뒤 `](../..)` 가 새 위치 기준으로 맞는지.
   쪽이 쓰는 `./img/*.png` 도 함께 옮겼는지
3. **스크립트 출력 경로** — `scripts/make_*.py` 의 `OUT = "docs/chNN/.../img/"`.
   PNG 는 커밋되어 있어 빌드는 통과하지만, 다시 그리면 엉뚱한 자리에 저장된다.
   한 스크립트의 그림이 두 장으로 갈리면 변수를 나눌 것
4. **본문의 번호 언급** — `N장`, `N.x절`, 「다음 절 ○○」. 링크가 아니라 글자다
5. **장 개요** — 「장의 구성」 절 목록, 「선수 지식」, 「핵심 요약」.
   다른 장들의 「선수 지식」이 옮겨 간 내용을 옛 장 이름으로 부르고 있지 않은지
6. **범위 문장** — 「이 장은 ~까지다」처럼 경계를 말하는 문장
7. **빈 디렉터리와 고아 그림** — `find docs -type d -empty`

확인 명령:

```bash
# 본문의 번호 언급 (ch05 를 옮겼다면)
grep -rn "5장\|5\.[0-9]*절" docs --include='*.md'
# 스크립트 출력 경로가 실제 그림 위치와 맞는지
grep -rn 'OUT *= *"docs/' scripts/*.py
```

### Add a Python example
1. Create `.py` in the relevant section directory
2. Educational style: module docstring, `# ===` dividers, `if __name__ == "__main__":` guard
