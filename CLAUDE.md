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
  밑줄은 역슬래시를 떼어 쓴다(`\text{petal_width}`). **`\texttt` 도 똑같이 깨진다** —
  MathJax 3 은 텍스트 모드 escape 를 **하나도** 구현하지 않으므로 `\text` 계열 전부가
  그렇다. 2026-10-23 에 `\texttt{l1\_ratio}` 꼴로 열 곳이 있었고, 그때까지 이 줄이
  명령 다섯만 적고 있어서 검사에 걸리지 않았다. 다만 `\mathrm{}` 과
  `\operatorname{}` 은 수식 모드라 `\_` 가 **맞다**(`\mathrm{Sex\_int}`).
  빌드는 통과하므로 눈으로만 잡힌다. 검사 명령은 아래 점검 목록에 있다
- **다른 쪽의 항목을 가리킬 때는 쪽을 링크로 적는다** — 연습문제·보기·정의·정리의
  번호는 **쪽마다 1 부터 다시 시작**하고 한 절에 쪽이 열까지 있다. 그래서 「15.4절
  연습문제 4」는 가리키는 곳이 넷이고, 넷 다 존재하므로 **어떤 검사로도 틀렸는지 알 수
  없다.** 반드시 `15.4절 [유도와 카이제곱 근사](derivation.md) 연습문제 2` 꼴로 쪽을
  링크한다. 링크로 적어야 기계가 검증한다 — 그 기계가 `tools/lint_item_refs.py` 이고,
  링크를 풀어 **그 쪽이 그 번호를 정말 갖고 있는지** 본다. 같은 쪽 안이면
  `연습문제 4` 로 족하다.

      링크를 **빼먹은** 자리를 찾는 것은 `tools/lint_itemref_links.py` 다. 판정이
      「번호가 맞나」가 아니라 **「링크가 있나」**라서 기계로 끝난다. 2026-10-23 현재
      **46 곳**이고 ch03 에 20, ch05 에 15 가 있다(ch14·ch15 는 0). 처음에 61 로
      셌는데 **열다섯이 거짓 양성이었다** — 번호와 항목이 링크 텍스트 **안에** 함께
      있는 규약대로의 꼴(`[13.5절 보기 5](../assumptions/checking_linearity.md)`)을
      세고 있었다. 「절」 바로 뒤를 보는 `(?!\[)` 로는 못 가린다(그때 대괄호는 훨씬
      앞에 있다). 링크 구간을 미리 떠서 그 안에서 시작하는 매치를 버려야 한다.
      **너무 넓은 패턴은 수를 부풀리고, 부풀린 수가 그대로 이 파일에 적힌다.** 이 꼴이
      가장 오래 숨는다 — 같은 날 ch15 세션이 「11.5절 보기 3」 셋을 찾았는데
      **11.5 는 쪽이 다섯이고 그중 넷에 보기 3 이 있었다.** 절 번호도 항목 번호도
      존재하는데 가리키는 내용이 어디에도 없었고, 하나는 자기만 갖고 있는 수 둘을
      다른 쪽이 재어 두었다고 했다. 번호 검증을 전부 통과해도 남는다

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
# 쪽 이름 없이 절 번호만 적은 항목 참조 — 가리키는 곳이 여럿이라 검증되지 않는다
# (아래 grep 은 `연습문제` 하나만 찾아 보기·정의·정리를 놓친다. 도구를 쓸 것)
python3 tools/lint_itemref_links.py [chNN]
# 한 쪽에 같은 상자 번호가 둘
grep -rhoE '\*\*(그림으로 보는|AI로 보는) 통계학 [0-9]+\.' docs --include='*.md' | sort | uniq -c
# \text{} 안에서 깨지는 escape (\$ \{ \} 만 살아남는다)
grep -rnoP '\\(?:text|textbf|textrm|textit|textsf|texttt|textnormal|textsc|mbox|hbox)\{[^{}]*\\[%_&#~^][^{}]*\}' docs --include='*.md'
```

**절을 넘는 중복 검사** — `python3 tools/lint_cross_section_pairs.py [chNN] [--all]`.
한 장 안에서 **같은 장·다른 절** 두 쪽이 특징적 소제목을 둘 이상 공유하는 자리를 찾는다.
절 안의 중복은 눈에 띄지만 절을 넘는 것은 nav 에서 멀리 떨어져 함께 열리지 않는다 —
ch14 의 14.1 과 14.7 이 「정형화된 사실」 넷을 소제목까지 같게 적고 있었고 서로 링크도
없었다. **후보만 낸다**: 아홉 가운데 넷이 의도된 평행 구조였고(검정마다 자기 p 값 유도가
있어야 한다), 하나는 틀이 달라 양쪽이 다 있어야 했다(오염 모형 대 두꺼운 꼬리 모형).
판정이 끝난 쌍은 도구 안 `SETTLED` 에 까닭과 함께 적혀 있으니 **같은 쌍을 다시 열지
말 것.** 소제목을 고쳐 겹침이 풀린 쌍은 「이제 안 걸림」으로 보고된다.

**거짓 절 번호 검사** — `python3 tools/lint_section_refs.py [chNN]`. 본문의
`N.M절 [쪽](경로)` 에서 **링크는 맞고 번호만 거짓인** 자리를 찾는다. 눌러 보면 제자리에
가고 `--strict` 도 통과하므로 틀린 것은 **독자가 읽는 번호뿐**이다. 장·절을 옮긴 뒤
본문 참조가 따라가지 않으면 이렇게 남는다 — 2026-10-23 에 서른둘이 나왔다(ch13 에 일곱).
진실의 원천은 `mkdocs.yml` 의 nav 다. 직접 grep 으로 하려 들면 거짓 양성 셋에 걸린다:
**절이 주어인 꼴**(`4.3절은 특히 [결합분포](../ch03/rv/joint.md)를 전제한다`),
**번호가 링크 뒤 괄호에 오는 꼴**(`[신뢰구간](ch08/…)(8.2절)과 [검정](ch09/…)(9.2절)` —
이걸 잘못 읽어 멀쩡한 8.2 를 9.2 로 고친 적이 있다), 그리고 **번호와 링크가 줄바꿈으로
나뉜 꼴**(이것을 빼면 둘을 놓친다).

꼴은 다섯이고 **마지막 하나는 nav 가 필요 없다.** A~D 는 다른 쪽을 가리키는 참조이므로
번호 뒤에 링크나 쪽 이름을 요구하는데, **같은 쪽 안의 절 참조는 뒤에 아무 이름도
따라오지 않는다**(`§6`, `$\S 6$`, `4절의 Cornish–Fisher 전개로`). 그래서 A~D 가 넷 다
놓치고, 쪽을 합치거나 절을 덜어 내면 똑같이 틀어진다. 꼴 E 가 그 쪽의 `## N.` 머리글과
맞춘다 — 책 전체 1780 곳을 검사해 셋이 나왔고, 셋 다 **내가 14.1↔14.7 을 합친 자국**
이었다(`financial_context` 가 §4 를 세 번 가리키는데 그 쪽은 §3 까지다). 이 꼴이 특히
오래 숨는 까닭은 **틀어진 번호도 존재하는 일이 많아서**다 — §4 가 §3 으로 밀려도 다른
쪽에는 §4 가 있으니 「없는 번호」 검사에 걸리지 않는다. 직접 grep 할 때 걸러야 하는 것
둘: 장을 넘는 `2장 5절`, 그리고 **링크 뒤의 `§N`**(`[카이제곱분포](../…/chi_square.md) 쪽
§6 의` — 그 `§6` 은 **그 쪽**의 절이다). 바깥 문헌의 절을 가리키는 자리는 기계로
가릴 수 없으므로 도구의 `E_ALLOW` 에 까닭과 함께 적는다(보르트키에비치 원서의 §12).

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
# 돌리지 않고 찾는 「없는 이름」 — 0 이어야 한다. 책 전체가 몇 초다.
# verify_outputs 는 돌려서 잡으므로 한 장에 25 분이다. 이것을 먼저 돌릴 것
python3 tools/lint_undefined.py
# 고아 그림과 없는 그림 — 둘 다 0 이어야 한다
python3 tools/lint_figures.py
# 링크는 맞고 절 번호만 거짓인 참조 — 0 이어야 한다 (빌드는 통과한다)
python3 tools/lint_section_refs.py
# 링크로 적은 항목 번호가 그 쪽에 정말 있는지 — 0 이어야 한다.
# 「링크로 적어야 기계가 검증한다」고 적어 둔 그 기계다
python3 tools/lint_item_refs.py
# 절을 넘어 같은 일을 하는 쪽 쌍 — 판정이 안 된 것이 0 이어야 한다
python3 tools/lint_cross_section_pairs.py
# 머리글의 LaTeX 와 ASCII 그리스 문자 — 목차에 날 문자열로 새어 나간다
grep -rnE '^#{1,6} .*\$' docs --include='*.md'
grep -rnE '^#{1,4} .*(^|[^\\A-Za-z_])(mu|sigma|tau|rho|alpha|beta|lambda|chi|theta)([^A-Za-z_]|$)' docs --include='*.md'
# 그림 대체텍스트의 ASCII 그리스 문자 (alt 속성은 수식이 될 수 없다)
grep -rnoE '!\[[^]]*(^|[^\\A-Za-z_])(mu|sigma|tau|rho|alpha|beta|lambda|chi|theta)([^A-Za-z_]|$)[^]]*\]' docs --include='*.md'
# 그림 대체텍스트 안의 $수식$ — 위 검사가 못 보는 꼴이다. 0 이어야 한다
grep -rnoE '!\[[^]]*\$[^]]*\]' docs --include='*.md'
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

**애초에 돌아갈 수 없는 블록은 `# verify: noexec — <까닭>` 을 적는다.** 꾸러미
안쪽을 옮긴 **인용**(`sklearn` 의 `chi2` 속 — 그 `X`, `Y` 는 그 함수가 인자로
받는 것이다), 독자가 끼우는 **골격**, 그리고 한 블록이 곧 한 `.py` 파일인
**여러 파일 보기**(5.3절 [표준오차](docs/ch05/applications/standard_error.md))가 그렇다.
이미 있는 `# verify: skip` 으로는 안 된다 — 그것은 「**돌리되** 출력은 맞춰 보지
않는다」여서 NameError 가 그대로 남는다. 주의할 점 셋:

- **스스로 닫힌 블록에만** 쓴다. 돌리지 않으므로 뒤 블록이 그 변수를 쓰면 거기서 터진다
- 건너뛴 블록은 `SKIP` 으로 찍히고 쪽을 BAD 로 세지 않는다. 그래도 **찍히기는
  한다** — 조용히 사라지면 결함을 덮는 표시가 되어 버린다
- 출력 블록이 딸려 있어도 **맞춰 보지 않는다.** 그 수는 이제 기계가 지켜 주지
  않으므로, 돌릴 수 있는 블록이라면 noexec 를 쓰지 말 것
