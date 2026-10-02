"""출력 블록을 실제 출력에 맞추되, **공백만 다른 것만** 자동으로 고친다.

    python3 tools/sync_outputs.py docs/ch14          # 고친다
    python3 tools/sync_outputs.py --dry docs/ch14    # 무엇이 바뀔지만 본다

`verify_outputs.py` 는 어긋남을 **알려 주기만** 하고, `resync_output.py` 는
시키는 블록을 **묻지 않고 덮어쓴다**. 이 도구는 그 사이에 있다.

왜 나누는가. 어긋남에는 두 갈래가 있고 처리가 정반대다.

- **공백만 다른 것** — 한글이 화면에서 두 칸을 차지하는데 파이썬의 `:>N` 은
  글자 수로 채운다. 그래서 표 머리글의 자리가 밀린다. 쪽이 적은 것이 손으로
  맞춘 모양이고 코드가 내는 것이 실제다. **자동으로 실제에 맞춘다.**
- **글자가 다른 것** — 수가 다르다. 이때 **코드가 옳다는 보장이 없다.**
  실제로 ch08 margin_of_error.md 에서는 쪽이 옳고 코드가 틀렸다(올림을
  반올림으로 찍고 있었다). 그런 자리를 덮어쓰면 결함을 지우는 것이 된다.
  **손대지 않고 보고만 한다.**

사람이 봐야 할 것(REVIEW)만 남으므로, 그것만 따져 보면 된다.

스레드를 묶어 돌릴 것:

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \\
    MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MPLBACKEND=Agg \\
    python3 tools/sync_outputs.py docs
"""
import json
import pathlib
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from verify_outputs import FENCE, RUNNER, blocks, preamble  # noqa: E402


def norm(lines):
    return [re.sub(r"\s+", "", x) for x in lines]


def process(path, dry=False, timeout=1800):
    bs = blocks(path)
    codes, meta = [], []
    pre = preamble(path)
    for k, (lang, ln, code) in enumerate(bs):
        if lang not in ("python", "py"):
            continue
        exp_idx = k + 1 if (k + 1 < len(bs) and bs[k + 1][0] in ("", "text")) else None
        codes.append((pre + "\n" + code) if (pre and not codes) else code)
        meta.append((ln, exp_idx))
    if not codes:
        return []

    with tempfile.TemporaryDirectory() as d:
        spec = pathlib.Path(d, "spec.json"); spec.write_text(json.dumps(codes), encoding="utf-8")
        res = pathlib.Path(d, "res.json")
        run = pathlib.Path(d, "run.py"); run.write_text(RUNNER, encoding="utf-8")
        try:
            subprocess.run([sys.executable, str(run), str(spec), str(res)],
                           capture_output=True, text=True, timeout=timeout,
                           cwd=str(pathlib.Path.cwd()))
            got = json.loads(res.read_text(encoding="utf-8")) if res.exists() else None
        except subprocess.TimeoutExpired:
            got = None
    if got is None:
        return [("TIMEOUT", meta[0][0], f"{timeout}s 초과")]

    lines = path.read_text(encoding="utf-8").split("\n")
    edits, report = [], []
    for (ln, exp_idx), r in zip(meta, got):
        if exp_idx is None:
            if not r["ok"]:
                report.append(("ERROR", ln, r["err"][:90]))
            continue
        if not r["ok"]:
            report.append(("ERROR", ln, r["err"][:90]))
            continue
        exp_start = bs[exp_idx][1]                      # ```text 울타리 줄 (1 기준)
        ind = len(FENCE.match(lines[exp_start - 1]).group(1))
        body_from = exp_start                            # 0 기준으로 울타리 다음 줄
        body_to = body_from
        while not FENCE.match(lines[body_to]):
            body_to += 1
        old = lines[body_from:body_to]
        new = [(" " * ind + x).rstrip() for x in r["stdout"].strip("\n").split("\n")]
        if old == new:
            continue
        if norm([x.strip() for x in old]) == norm([x.strip() for x in new]):
            edits.append((body_from, body_to, new))
            report.append(("WS", ln, f"{len(old)}줄 공백 맞춤"))
        else:
            d1 = next((f"코드[{(new[i] if i < len(new) else '<없음>').strip()[:46]}] "
                       f"쪽[{(old[i] if i < len(old) else '<없음>').strip()[:46]}]"
                       for i in range(max(len(old), len(new)))
                       if (new[i] if i < len(new) else None) != (old[i] if i < len(old) else None)), "")
            report.append(("REVIEW", ln, d1))
    if edits and not dry:
        for a, b, new in sorted(edits, reverse=True):
            lines[a:b] = new
        path.write_text("\n".join(lines), encoding="utf-8")
    return report


def main(argv):
    dry = "--dry" in argv
    argv = [a for a in argv if a != "--dry"]
    t = pathlib.Path(argv[0])
    files = sorted(t.rglob("*.md")) if t.is_dir() else [t]
    tally = {}
    for p in files:
        rep = process(p, dry=dry)
        if not rep:
            continue
        for kind, _, _ in rep:
            tally[kind] = tally.get(kind, 0) + 1
        print(f"{p}", flush=True)
        for kind, ln, msg in rep:
            print(f"      {kind:7s}{ln:5d}  {msg}", flush=True)
    print("\n" + "  ".join(f"{k}={v}" for k, v in sorted(tally.items())))


if __name__ == "__main__":
    main(sys.argv[1:])
