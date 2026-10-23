r"""13.6절 「그림으로 보는 통계학」 — 걷어낸 뒤의 p 값은 그 p 값이 아니다.

이 쪽의 $2$ 절은 쿡 거리로 **이상점을 걸러 내고 다시 적합하는 코드**를 싣는다. 그
뒤에 나오는 $t$ 값과 $p$ 값은 어떤 것인가. 걸러 내는 규칙이 $y$ 를 들여다보고
정해지므로, 그 수는 더 이상 약속한 눈금 위에 있지 않다.

참 기울기를 $0$ 으로 두고(정규 오차) 셋을 견준다.

  삭제 없음          — 그대로 검정
  스튜던트화 잔차 2   — |외부 스튜던트화 잔차| > 2 인 점을 빼고 다시 적합
  쿡 4/n             — D_i > 4/n 인 점을 빼고 다시 적합

이 스크립트가 재는 값(되풀이 4000 번, 유의수준 0.05):

    n   규칙            기각률   평균 삭제 개수   삭제 비율   s 의 축소비
   30   삭제 없음        0.051        0.00        0.000      1.000
   30   스튜던트화 잔차 2  0.118        1.65        0.055      0.859
   30   쿡 4/n           0.092        1.77        0.059      0.899
   60   삭제 없음        0.044        0.00        0.000      1.000
   60   스튜던트화 잔차 2  0.106        3.03        0.051      0.868
   60   쿡 4/n           0.085        3.19        0.053      0.902
  200   삭제 없음        0.053        0.00        0.000      1.000
  200   스튜던트화 잔차 2  0.117        9.34        0.047      0.877
  200   쿡 4/n           0.106       10.15        0.051      0.903

곧 **$5\%$ 로 약속한 검정이 $10$–$12\%$ 가 된다.** 그리고 $n$ 이 커져도 사라지지
않는다 — 규칙이 늘 **같은 비율**(정규표본의 $5\%$ 쯤)을 잘라 내기 때문이다.

만드는 파일:

  ch13/diagnostics/img/delete_then_test.png

실행:  python3 scripts/make_ch13_delete_then_test.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch13/diagnostics/img/"

INK = "#37474F"
AMBER = "#B07C2C"
TEAL = "#00838F"
RED = "#D32F2F"
BLUE = "#1565C0"
MUTED = "#90A4AE"

NS = (30, 60, 200)
RULES = (("삭제 없음", MUTED), ("스튜던트화 잔차 $2$", AMBER), ("쿡 $4/n$", RED))
REPS = 4000
ALPHA = 0.05
SEED = 31


def summarize(X, y):
    XtXi = np.linalg.inv(X.T @ X)
    b = XtXi @ X.T @ y
    e = y - X @ b
    s2 = (e ** 2).sum() / (len(y) - 2)
    return b[1], np.sqrt(s2 * XtXi[1, 1]), e, s2, XtXi


def one(n, rule, r):
    rr = np.random.default_rng([SEED, n, r])
    x = rr.normal(size=n)
    y = rr.normal(size=n)                       # 참 기울기 0
    X = np.column_stack([np.ones(n), x])
    _, _, e, s2, XtXi = summarize(X, y)
    h = np.einsum("ij,jk,ik->i", X, XtXi, X)
    if rule == 0:
        keep = np.ones(n, bool)
    elif rule == 1:
        s2i = ((e ** 2).sum() - e ** 2 / (1 - h)) / (n - 3)
        keep = np.abs(e / np.sqrt(s2i * (1 - h))) <= 2
    else:
        keep = e ** 2 * h / (2 * s2 * (1 - h) ** 2) <= 4 / n
    if keep.sum() < 5:
        keep = np.ones(n, bool)
    b1, se, _, s2b, _ = summarize(X[keep], y[keep])
    rej = abs(b1 / se) > stats.t.ppf(1 - ALPHA / 2, int(keep.sum()) - 2)
    return bool(rej), n - int(keep.sum()), float(np.sqrt(s2b / s2))


def main():
    res = {}
    # 한글은 고정폭에서 두 칸을 차지하므로 자리맞춤은 공백을 직접 적는다
    print(f"되풀이 {REPS} 번, 유의수준 {ALPHA}, 참 기울기 0")
    print("    n   규칙              기각률   평균 삭제   삭제 비율   s 의 축소비")
    for n in NS:
        for j, (nm, _) in enumerate(RULES):
            out = [one(n, j, r) for r in range(REPS)]
            rej = float(np.mean([o[0] for o in out]))
            cut = float(np.mean([o[1] for o in out]))
            shr = float(np.mean([o[2] for o in out]))
            res[(n, j)] = (rej, cut, shr)
            flat = nm.replace("$", "")
            print(f"  {n:4d}   {flat:16s} {rej:7.3f} {cut:10.2f} "
                  f"{cut / n:11.3f} {shr:12.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), constrained_layout=True)

    # ── ① 기각률 ──
    ax = axes[0]
    xs = np.arange(len(NS))
    w = 0.26
    for j, (nm, col) in enumerate(RULES):
        v = [res[(n, j)][0] for n in NS]
        ax.bar(xs + (j - 1) * w, v, width=w * 0.9, color=col, alpha=0.9,
               label=nm)
        for i, u in enumerate(v):
            ax.text(i + (j - 1) * w, u + 0.007, f"${u:.3f}$", ha="center",
                    fontsize=10.2, color=col, fontweight="bold")
    ax.axhline(ALPHA, color=INK, lw=1.8, ls="--", zorder=0, alpha=0.7)
    ax.text(2.44, ALPHA - 0.011, f"약속한 ${ALPHA}$", ha="right", fontsize=11,
            color=INK)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"$n = {n}$" for n in NS], fontsize=11.6)
    ax.set_ylim(0, 0.19)
    ax.set_ylabel("참 기울기가 $0$ 인데 기각한 비율", fontsize=11.5)
    ax.set_title("① 걷어낸 뒤의 $5\\%$ 검정은 $5\\%$ 가 아니다", fontsize=12.6,
                 color=INK)
    ax.grid(alpha=0.22, axis="y")
    ax.legend(fontsize=10.2, framealpha=0.93, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)

    # ── ② 왜 사라지지 않는가 ──
    ax = axes[1]
    for j, (nm, col) in enumerate(RULES[1:], start=1):
        frac = [res[(n, j)][1] / n for n in NS]
        shr = [res[(n, j)][2] for n in NS]
        ax.plot(NS, frac, "o-", ms=11, lw=2.6, color=col,
                label=f"{nm} — 잘라 낸 비율")
        ax.plot(NS, shr, "s--", ms=9, lw=2.2, color=col, alpha=0.6,
                label=f"{nm} — $s$ 의 축소비")
    ax.axhline(1.0, color=MUTED, lw=1.0, ls=":")
    ax.annotate("둘 다 $n$ 과 무관하다 —\n그래서 기각률도 줄지 않는다",
                xy=(130, 0.89), xytext=(40, 0.63), fontsize=11.2, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.5))
    ax.set_xscale("log")
    ax.set_xticks(list(NS))
    ax.set_xticklabels([f"${n}$" for n in NS])
    ax.minorticks_off()
    ax.set_xlabel("관측 수 $n$ (로그 눈금)", fontsize=11.5)
    ax.set_ylabel("비율", fontsize=11.5)
    ax.set_ylim(0, 1.12)
    ax.set_title("② 규칙은 늘 같은 몫을 잘라 낸다", fontsize=12.6, color=INK)
    ax.grid(alpha=0.22)
    ax.legend(fontsize=9.6, framealpha=0.93, loc="center right")
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("무엇을 뺄지 $y$ 를 보고 정하면", fontsize=14.5)
    fig.text(0.5, -0.05,
             f"참 기울기를 ${0}$ 으로 두고 정규 오차로 만든 자료를 "
             f"${REPS}$ 번씩 돌렸다. 계수 추정값은 어느 규칙에서도 불편이고, "
             f"달라지는 것은 그 곁의 눈금뿐이다. "
             f"모든 수는 이 스크립트가 계산한 값이다.",
             ha="center", fontsize=10.4, color=INK)

    os.makedirs(OUT, exist_ok=True)
    path = OUT + "delete_then_test.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
