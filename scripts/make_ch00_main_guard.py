r"""가드가 없으면 무슨 일이 일어나는가 — spawn 과 __main__ 의 재수입.

이 쪽은 `if __name__ == "__main__":` 가드를 `__name__` 의 값으로 설명한다. 맞는
설명이지만, 그 가드가 **무엇을 막아 주는지**는 따로 보아야 보인다.

파이썬이 새 프로세스를 띄우는 방법은 셋이고(fork, spawn, forkserver) **spawn 은
파이썬 해석기를 새로 띄운다.** 새 해석기는 자식이 돌릴 함수를 찾으려고 `__main__`
모듈을 **다시 수입한다.** 가드가 없으면 그때 모듈의 맨 위 코드가 다시 돌아간다.

① spawn 이 fork 보다 얼마나 느린가 — 프로세스를 띄우는 값.
② 가드가 없으면 맨 위 코드가 몇 번 도는가.
③ 그래서 병렬이 이득이 되는 자리는 어디부터인가.

만드는 파일: docs/ch00/tools/img/main_guard.png
"""

import multiprocessing as mp
import os
import subprocess
import sys
import tempfile
import textwrap
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"

def _noop():
    pass


def _busy(ms):
    t0 = time.perf_counter()
    x = 0.0
    while (time.perf_counter() - t0) * 1000 < ms:
        x += 1.0
    return x


def main():
    TMP = tempfile.mkdtemp(prefix="guard_")
    print(f"파이썬 {sys.version.split()[0]}, 플랫폼 {sys.platform}")
    print(f"이 플랫폼의 기본 시작 방법: {mp.get_start_method()}")
    print(f"쓸 수 있는 방법: {mp.get_all_start_methods()}")


    # === ① 프로세스를 띄우는 값 =========================================
    def start_cost(method, k=4):
        ctx = mp.get_context(method)
        t0 = time.perf_counter()
        ps = [ctx.Process(target=_noop) for _ in range(k)]
        for p in ps:
            p.start()
        for p in ps:
            p.join()
        return (time.perf_counter() - t0) / k * 1000     # 하나당 ms


    METHODS = [m for m in ("fork", "spawn", "forkserver")
               if m in mp.get_all_start_methods()]
    print(f"\n① 빈 프로세스 하나를 띄우고 거두는 데 드는 시간")
    # 한글은 고정폭에서 두 칸을 차지하므로 자리맞춤을 직접 적는다
    print("   방법          하나당(ms)")
    COST = []
    for m in METHODS:
        try:
            c = start_cost(m)
        except Exception as e:                      # fork 가 막힌 환경도 있다
            print(f"   {m:<12} 쓸 수 없음 ({type(e).__name__})")
            continue
        COST.append((m, c))
        print(f"   {m:<12} {c:>10.2f}")
    if len(COST) >= 2:
        fast = min(COST, key=lambda x: x[1])
        slow = max(COST, key=lambda x: x[1])
        print(f"   {slow[0]} 가 {fast[0]} 보다 {slow[1]/fast[1]:.1f} 배 느리다 — "
              f"해석기를 새로 띄우기 때문이다.")

    # === ② 가드가 없으면 맨 위 코드가 몇 번 도는가 ======================
    BODY = textwrap.dedent("""\
        import multiprocessing as mp, os, sys

        print("TOP", os.getpid(), flush=True)   # 모듈 맨 위 — 몇 번 찍히는가

        def work():
            print("CHILD", os.getpid(), flush=True)

        {guard}
            mp.set_start_method("spawn", force=True)
            ps = [mp.Process(target=work) for _ in range(3)]
            for p in ps: p.start()
            for p in ps: p.join()
        """)

    cases = {
        "가드 있음": BODY.format(guard='if __name__ == "__main__":'),
        "가드 없음": BODY.format(guard='if True:'),
    }
    print("\n② 자식 셋을 spawn 으로 띄울 때 모듈 맨 위가 몇 번 도는가")
    print("   경우          TOP 줄   CHILD 줄   결과")
    TOPS = {}
    for name, src in cases.items():
        path = os.path.join(TMP, f"case_{len(TOPS)}.py")
        with open(path, "w") as f:
            f.write(src)
        r = subprocess.run([sys.executable, path], capture_output=True,
                           text=True, timeout=120)
        out = r.stdout.split("\n")
        top = sum(1 for l in out if l.startswith("TOP"))
        child = sum(1 for l in out if l.startswith("CHILD"))
        bad = "RuntimeError" in r.stderr or r.returncode != 0
        TOPS[name] = (top, child, bad)
        tag = "터짐" if bad else "정상"
        print(f"   {name:<12} {top:>6}   {child:>8}   {tag}")
        if bad:
            last = [l for l in r.stderr.strip().split("\n") if l.strip()][-1]
            print(f"      → {last[:92]}")

    g = TOPS["가드 있음"]
    n = TOPS["가드 없음"]
    print(f"   가드가 있어도 맨 위는 {g[0]} 번 돈다 — 부모 한 번, 자식 셋이 "
          f"다시 수입하며 세 번이다.")
    print(f"   **모듈 맨 위에 둔 일은 자식 수만큼 되풀이된다.**")
    print(f"   가드가 없으면 자식이 {n[1]} 개 돌고 RuntimeError 로 멈춘다 — "
          f"파이썬이 알아보고 거부한다.")

    # === ③ 병렬이 이득이 되는 자리 ======================================
    SIZES = [1, 10, 100, 300, 1000, 3000]
    NTASK, NWORK = 8, 4
    print(f"\n③ 일 {NTASK} 개를 혼자 vs 일꾼 {NWORK} 로 (spawn)")
    print("   일 하나(ms)   혼자(ms)   병렬(ms)      빠르기")
    SPEED = []
    ctx = mp.get_context("spawn")
    for ms in SIZES:
        t0 = time.perf_counter()
        for _ in range(NTASK):
            _busy(ms)
        solo = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        with ctx.Pool(NWORK) as pool:
            pool.map(_busy, [ms] * NTASK)
        par = (time.perf_counter() - t0) * 1000
        SPEED.append(solo / par)
        print(f"   {ms:>11}   {solo:>8.1f}   {par:>8.1f}   {solo/par:>10.2f}x")
    cross = next((s for s, r in zip(SIZES, SPEED) if r > 1), None)
    print(f"   일 하나가 {cross} ms 는 되어야 병렬이 이득이다 "
          f"(프로세스 띄우는 값이 그만큼이다)." if cross else
          "   이 범위에서는 병렬이 이득이 되지 않는다.")

    # === 그림 ============================================================
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.6, 5.2))
    C_A, C_B, C_C = "#d62728", "#1f77b4", "#2ca02c"

    names1 = [m for m, _ in COST]
    vals1 = [c for _, c in COST]
    bars = ax1.bar(names1, vals1, color=[C_C if m == "fork" else C_A for m in names1],
                   alpha=0.9)
    for b, v in zip(bars, vals1):
        ax1.text(b.get_x() + b.get_width() / 2, v + max(vals1) * 0.02,
                 f"${v:.1f}$ ms", ha="center", fontsize=11)
    ax1.set_ylabel("프로세스 하나를 띄우는 시간 (ms)")
    ax1.set_ylim(0, max(vals1) * 1.18)
    ax1.set_title("① spawn 은 해석기를 새로 띄운다", fontsize=12)
    ax1.grid(alpha=0.25, axis="y")

    names2 = list(TOPS)
    xs2 = np.arange(len(names2))
    tops = [TOPS[k][0] for k in names2]
    kids = [TOPS[k][1] for k in names2]
    ax2.bar(xs2 - 0.19, tops, width=0.36, color="#9467bd", alpha=0.9,
            label="모듈 맨 위가 돈 횟수")
    ax2.bar(xs2 + 0.19, kids, width=0.36, color=C_C, alpha=0.9,
            label="제대로 돈 자식 수")
    for x, v in zip(xs2 - 0.19, tops):
        ax2.text(x, v + 0.12, f"${v}$", ha="center", fontsize=12)
    for x, v, k in zip(xs2 + 0.19, kids, names2):
        ax2.text(x, v + 0.12, f"${v}$", ha="center", fontsize=12,
                 color=C_A if v == 0 else "#333333")
    ax2.text(1, 1.6, "RuntimeError\n로 멈춤", ha="center", fontsize=10.5,
             color=C_A)
    ax2.set_xticks(xs2)
    ax2.set_xticklabels(names2, fontsize=11)
    ax2.set_ylabel("횟수")
    ax2.set_ylim(0, max(tops) * 1.45)
    ax2.legend(loc="upper left", fontsize=9.5)
    ax2.set_title("② 자식 셋을 spawn 으로 띄우면", fontsize=12)
    ax2.grid(alpha=0.25, axis="y")

    ax3.semilogx(SIZES, SPEED, "o-", color=C_B, lw=2.8, ms=11)
    ax3.axhline(1.0, color="#333333", lw=2.0, ls="--")
    ax3.text(1.2, 1.05, "여기 아래면 혼자가 빠르다", fontsize=10.5, color="#333333")
    ax3.set_xlabel("일 하나의 크기 (ms)")
    ax3.set_ylabel(f"혼자 / 일꾼 {NWORK} (빠르기)")
    ax3.set_xticks(SIZES)
    ax3.set_xticklabels([f"${s}$" for s in SIZES])
    ax3.minorticks_off()
    ax3.set_title(f"③ 일 {NTASK} 개를 나눌 때", fontsize=12)
    ax3.grid(alpha=0.25)

    fig.tight_layout()
    out = "docs/ch00/tools/img/main_guard.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("\nsaved", out)


if __name__ == "__main__":
    main()
