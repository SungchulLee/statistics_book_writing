# 원본 강의노트 복구 안내

2026-03-08 ~ 2026-05-12 사이의 `review:` / `update:` 커밋들이 여러 페이지의
서술형 예시를 요약·삭제했다. **원본은 모두 git 히스토리에 그대로 남아 있다.**
이 문서는 그것을 되찾는 방법을 기록한다.

## 태그

| 태그 | 커밋 | 시점 | 내용 |
|---|---|---|---|
| `original-notes-2026-02-09` | `dc680cc` | 2026-02-09 | 예시가 가장 풍부한 초기 스냅숏 |
| `original-notes-pre-review` | `e725ddb` | 2026-03-06 | review 패스 직전의 마지막 상태 |

두 태그는 주석 태그이므로 `git gc`가 절대 회수하지 않는다.

## 전체 트리를 다시 꺼내기

```bash
rm -rf recovered && mkdir -p recovered
git archive original-notes-pre-review docs | tar -x -C recovered
# -> recovered/docs/... 아래에 461개의 .md 원본이 복원된다
```

`recovered/` 는 `.gitignore` 에 등록되어 있다. 원본은 git 안에 있으므로
이 디렉터리는 언제든 위 명령으로 재생성할 수 있는 편의용 사본일 뿐이다.

## 특정 파일 하나만 보기

```bash
git show original-notes-pre-review:docs/ch01/classical/survivorship_bias.md
git diff original-notes-pre-review HEAD -- docs/ch01/classical/
```

## 무엇이 사라졌는지 확인하기

연도가 붙은 역사적 사례는 번역 후에도 연도가 남으므로 가장 잘 잡힌다.

```bash
git show original-notes-pre-review:docs/ch01/classical/bias_nonresponse.md \
  | grep -nE '\b(18|19|20)[0-9]{2}\b'
```
