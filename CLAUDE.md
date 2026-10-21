# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

MkDocs Material documentation site for a mathematical textbook. Content uses LaTeX (MathJax).
Python 코드는 **쪽 안의 펜스 블록에만** 있다 — `docs/` 아래에 `.py` 파일은 없다
(아래 「코드는 쪽 안에」). Three specialized review/write agents live in `agents/`;
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
            ├── topic.md          ← 코드도 이 안에 (출력까지)
            └── img/*.png         ← scripts/make_*.py 가 그린 그림
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
| `docs/**/*.py` | ❌ 만들지 않는다 | ❌ excluded via `mkdocs.yml` |

`.gitignore` entry: `docs/**/*_v[0-9]*.md`

`mkdocs.yml` exclusion:
```yaml
exclude_docs: |
  *_score.md
  *_v[0-9]*.md
  *.py
```

마지막 줄은 **안전망이다.** nav 에 없는 파일도 MkDocs 는 `site/` 로 그대로 복사하므로,
검증용으로 잠깐 만든 `docs/ch11/_c.py` 같은 스크래치가 `/ch11/_c.py` 로 공개된 적이 있다.
검증 스크립트는 `docs/` 밖(스크래치 디렉터리)에서 돌리고, 끝나면 코드와 출력을 쪽에 옮긴다.

Commit after `update` — stage only `<name>.md` and `<name>_score.md`:
```bash
git add docs/path/to/<name>.md docs/path/to/<name>_score.md
git commit -m "update: <name>"
```

## Navigation Structure

Nav hierarchy: **Parts → Chapters → Sections → Pages**. Nav entries point to `.md` files only.

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
- **머리글과 대체텍스트에는 LaTeX 를 쓰지 않는다. 유니코드 글자를 쓴다**
  (`μ`, `σ²`, `τ`, `λ`, `F₁`, `Oₚ`, `χ²`, `X̄`, `p̂`). 까닭은
  `docs/javascripts/mathjax.js` 에 있다 — `ignoreHtmlClass: ".*|"` 와
  `processHtmlClass: "arithmatex"` 로 **`arithmatex` 클래스가 붙은 것만** 처리한다.
  본문 머리글은 그 래퍼를 받지만 **오른쪽 목차 항목은 받지 못해서**, 거기에는
  `두 기준은 같은 \(\lambda\)를 고르지 않는다` 가 글자 그대로 찍힌다. 그림의
  `alt` 속성도 HTML 속성이라 애초에 수식이 될 수 없다. **빌드는 통과하고
  본문만 보면 멀쩡해서 목차를 봐야 잡힌다**

      ### 두 기준은 같은 $\lambda$를 고르지 않는다     ← 목차가 깨진다
      ### 두 기준은 같은 λ를 고르지 않는다              ← 이렇게
      ![alpha는 되풀이했을 때의 비율이다](...)          ← alt 는 LaTeX 불가
      ![α는 되풀이했을 때의 비율이다](...)              ← 이렇게

  반대로 **쪽지 제목과 `<figcaption>` 은 LaTeX 가 맞다.** 둘은 본문 안이라
  `arithmatex` 래퍼를 받아 정상으로 렌더된다(`!!! warning "$\sigma^2$의 추정"`).
  ASCII 로 `sigma` 라 적지 말 것. 다만 sklearn 인자 이름처럼 **식별자**를
  가리킬 때는 역따옴표가 맞다(`` `alpha` 라는 이름의 두 가지 의미 ``)
- `\$` for currency, never bare `$`
- QED: `$\square$`
- Every content page ends with `## Exercises` (interleaved solutions, collapsible)
- **코드는 쪽 안에** — `docs/` 아래에 `.py` 파일을 만들지 않는다. 코드는 쪽의
  ```` ```python ```` 펜스에 넣고, **찍는 것이 있으면 바로 아래에 출력을 붙인다**:

      ```python
      print(f"T = {T:.4f}, p = {p:.4f}")
      ```

      출력:

      ```
      T = 1.3070, p = 0.5202
      ```

  출력은 **실제로 돌려 받은 것**을 붙인다. 손으로 적으면 코드를 고칠 때 어긋나고,
  어긋나도 빌드는 통과한다. 검사 도구는 `tools/verify_outputs`·`tools/resync_output`
- **출력 표의 머리글에 한글을 넣을 때 f-string 자리맞춤을 믿지 말 것.** 한글은
  고정폭 글꼴에서 **두 칸**을 차지하는데 `{'대응 폭':>9s}` 는 **글자 수**로 센다.
  그래서 코드가 내는 머리글이 자료 열과 어긋나고, 보기 좋게 **손으로 다듬는 순간
  코드와 출력이 갈린다.** 실제로 그렇게 갈린 곳이 둘 있었고(`ci_mu_paired`,
  `sample_size`) 한쪽은 다듬는 과정에서 줄 셋이 빠져 있었다. 머리글은 **공백을
  직접 적고** 까닭을 주석으로 남긴다

      # 한글은 고정폭에서 두 칸을 차지하므로 f-string 자리맞춤이 보이는 열과 어긋난다
      print("   n     ρ   대응 폭   독립 폭      비")      ← 이렇게
      print(f"{'n':>4s} {'ρ':>5s} {'대응 폭':>9s} ...")   ← 어긋난다

  이미 있는 `{'한글':>Ns}` 가 3,000 곳 넘지만 **그 자체는 결함이 아니다** — 출력을
  그대로 받아 적었으면 표가 들쭉날쭉할 뿐 코드와 일치한다. 결함은 손으로 다듬은
  곳뿐이고 그것은 `tools/verify_outputs` 만 잡는다
- 그림 그리는 코드만 예외로 `scripts/make_*.py` 에 둔다. 쪽에는 PNG 만 들어간다
- 두 연재 상자 — `<div class="storybox">` 「그림으로 보는 통계학」(호박·점선),
  `<div class="aibox">` 「AI로 보는 통계학」(청록·왼쪽 띠). 증명·문항 사이클 밖이며
  이야기하는 정의/정리 **바로 뒤**에 놓는다. 전체 규칙은 `agents/SKILL.md`
- **출처 줄** — 상자가 책 바깥의 사실(사람·연도·학술지·쪽수·현실의 수치)을 하나라도
  주장하면 마지막 줄은 `_확인한 곳 — ⟨확인한 것⟩. 그림과 표의 수는 모두 직접 계산한 값이다_`
  로 닫는다. **확인한 것만 적는다** — 확인하지 않은 사실은 출처 줄에 올리지 말고 본문에서
  뺀다. 수학만 하는 상자는 이 줄을 붙이지 않는다
- **`\text{}` 안에서는 escape 가 깨진다** — MathJax 3 은 `\text{}` 안의
  `\%` `\_` `\&` `\#` `\~` `\^` 를 **역슬래시째 찍는다.** 살아남는 것은
  `\$` `\{` `\}` 셋뿐이다. 퍼센트는 수식 모드로 빼고(`95\%\ \text{신뢰구간}`),
  밑줄은 역슬래시를 떼어 쓴다(`\text{petal_width}`). 다만 `\mathrm{}` 과
  `\operatorname{}` 은 수식 모드라 `\_` 가 **맞다**(`\mathrm{Sex\_int}`).
  빌드는 통과하므로 눈으로만 잡힌다. 검사 명령은 아래 점검 목록에 있다
- **다른 쪽의 항목을 가리킬 때는 쪽을 링크로 적는다** — 연습문제·보기·정의·정리의
  번호는 **쪽마다 1 부터 다시 시작**하고 한 절에 쪽이 열까지 있다. 그래서 「15.4절
  연습문제 4」는 가리키는 곳이 넷이고, 넷 다 존재하므로 **어떤 검사로도 틀렸는지 알 수
  없다.** 반드시 `15.4절 [유도와 카이제곱 근사](derivation.md) 연습문제 2` 꼴로 쪽을
  링크한다. 링크로 적어야 기계가 검증한다. 같은 쪽 안이면 `연습문제 4` 로 족하다

## Common Tasks

### Add a new section
1. Create `docs/chapter_name/section_name/` with `.md` files (코드는 그 안의 펜스에)
2. Add nav entries to `mkdocs.yml` (`.md` only)

### Add a new chapter
1. Create `docs/chapter_name/` with `index.md` and section subdirectories
2. Add chapter block to `mkdocs.yml` under the correct Part

### 장·절을 옮기거나 합칠 때 (점검 목록)

`mkdocs build --strict` 는 **링크와 파일만** 검사한다. 4~7 번은 글자라서 빌드가
잡지 못하고, 실제로 매번 여기서 누락이 난다. 순서대로 전수로 훑을 것.

1. **nav** — `mkdocs.yml` 의 경로·제목·절 번호
2. **상대링크와 그림** — `git mv` 뒤 `](../..)` 가 새 위치 기준으로 맞는지.
   쪽이 쓰는 `./img/*.png` 도 함께 옮겼는지
3. **스크립트 출력 경로** — `scripts/make_*.py` 의 `OUT = "docs/chNN/.../img/"`.
   PNG 는 커밋되어 있어 빌드는 통과하지만, 다시 그리면 엉뚱한 자리에 저장된다.
   한 스크립트의 그림이 두 장으로 갈리면 변수를 나눌 것
4. **본문의 번호 언급** — `N장`, `N.x절`, 「다음 절 ○○」. 링크가 아니라 글자다
5. **항목 번호** — 연습문제·보기·정의·정리·상자. **쪽을 합치면 한쪽 번호가 통째로
   밀리는데**, 본문 참조는 따라가지 않는다. 밀린 뒤에도 그 번호는 *존재하므로*
   「없는 번호」 검사에 걸리지 않는다. 합칠 때 더해진 offset 을 적어 두고 본문
   참조를 그만큼 옮길 것. 상자 번호가 한 쪽에 둘이 되지 않았는지도 본다
6. **장 개요** — 「장의 구성」 절 목록, 「선수 지식」, 「핵심 요약」.
   다른 장들의 「선수 지식」이 옮겨 간 내용을 옛 장 이름으로 부르고 있지 않은지
7. **범위 문장** — 「이 장은 ~까지다」처럼 경계를 말하는 문장
8. **빈 디렉터리와 고아 그림** — `find docs -type d -empty`

확인 명령:

```bash
# 본문의 번호 언급 (ch05 를 옮겼다면)
grep -rn "5장\|5\.[0-9]*절" docs --include='*.md'
# 스크립트 출력 경로가 실제 그림 위치와 맞는지
grep -rn 'OUT *= *"docs/' scripts/*.py
# 쪽 이름 없이 절 번호만 적은 참조 — 가리키는 곳이 여럿이라 검증되지 않는다
grep -rnP '\d+\.\d+\s*절\s*(?!\[)[^가-힣\n]{0,3}연습문제 \d' docs --include='*.md'
# 한 쪽에 같은 상자 번호가 둘
grep -rhoE '\*\*(그림으로 보는|AI로 보는) 통계학 [0-9]+\.' docs --include='*.md' | sort | uniq -c
# \text{} 안에서 깨지는 escape (\$ \{ \} 만 살아남는다)
grep -rnoP '\\(?:text|textbf|textrm|textit|mbox)\{[^{}]*\\[%_&#~^][^{}]*\}' docs --include='*.md'
```

**고아 그림·없는 그림 검사** — `python3 tools/lint_figures.py [chNN]`. 양쪽 방향을
모두 본다. 쪽에서 절을 걷어내면 **고아**가 생기고 그림을 옮기면 **없는 그림**이
생기는데, PNG 가 커밋되어 있으면 `--strict` 가 아무 말도 하지 않는다. 이 세션에
두 번 놓쳤다 — 5.1 의 338줄을 덜어 낼 때와 `coverage.md` 의 보기 둘을 걷어낼 때다.
직접 쓸 때는 대체텍스트에 `]` 가 들어가면(`E[D]`, `100 t CIs | n=10` 등) 소박한
정규식이 끊기므로 `!\[.*?\]\(` 로 비탐욕 매칭할 것.

### Add a Python example
1. 쪽 안의 ```` ```python ```` 펜스에 쓴다. `.py` 파일을 만들지 않는다
2. 돌려서 실제 출력을 받고, 「출력:」 블록으로 바로 아래에 붙인다
3. 검증용 스크립트가 필요하면 `docs/` **밖**(스크래치)에서 돌리고 쪽으로 옮긴 뒤 버린다

확인 명령:

```bash
# docs/ 아래에 .py 가 생기지 않았는지 — 하나도 없어야 한다
find docs -name '*.py'
# nav 에 .py 항목이 없는지
grep -n '\.py' mkdocs.yml
# 찍는 블록인데 출력이 없는 곳 — 0 이어야 한다 (종료코드로도 알린다)
python3 tools/lint_outputs.py
# 고아 그림과 없는 그림 — 둘 다 0 이어야 한다
python3 tools/lint_figures.py
# 머리글의 LaTeX 와 ASCII 그리스 문자 — 목차에 날 문자열로 새어 나간다
grep -rnE '^#{1,6} .*\$' docs --include='*.md'
grep -rnE '^#{1,4} .*(^|[^\\A-Za-z_])(mu|sigma|tau|rho|alpha|beta|lambda|chi|theta)([^A-Za-z_]|$)' docs --include='*.md'
# 그림 대체텍스트의 ASCII 그리스 문자 (alt 속성은 수식이 될 수 없다)
grep -rnoE '!\[[^]]*(^|[^\\A-Za-z_])(mu|sigma|tau|rho|alpha|beta|lambda|chi|theta)([^A-Za-z_]|$)[^]]*\]' docs --include='*.md'
# nav 제목과 쪽의 H1 이 어긋나는 곳. 160 곳이 나오지만 거의 다 의도된 것이다
# (nav 가 'Bernoulli(p)'·'(코드)'·'(재론)' 처럼 구별을 붙인다). 실질이 다른
# 것만 고를 것 — ASCII 그리스 문자가 한쪽에만 있는 경우가 전형이다
```

`lint_outputs.py` 가 거르는 함정 셋을 기억할 것. `def` 안의 `print` 는 정의만으로
찍히지 않고(다음 블록이 부른다), 문제 블록의 출력은 `??? success "풀이"` 안에
있을 수 있고, 한 줄짜리는 `print(a, b)   # 4.0  3.0` 처럼 주석에 값을 적는다.
이 셋을 거르지 않으면 열셋이 잡히는데 그 가운데 **여덟이 거짓이다.**

일부러 출력을 싣지 않는 블록은 안에 `# 출력 없음: <까닭>` 을 적는다 — 독자에게
알려 주는 줄이고 `lint_outputs.py` 도 그 줄을 보고 넘어간다. 까닭은 둘뿐이어야
한다: 독자가 자기 자료를 끼우는 **골격**, 또는 환경이 없어 **확보하지 못한** 출력
(`yfinance` 한도·`pymc` 미설치 — 이때는 `!!! note` 로 맞춰 볼 값까지 적는다).
