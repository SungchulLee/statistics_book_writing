r"""9.4절 「그림으로 보는 통계학」 — 비틀스를 들으면 어려진다.

시먼스·넬슨·시몬손은 2011 년에 『거짓양성 심리학』을 냈다. 그 논문의
첫 실험에서, 비틀스의 「When I'm Sixty-Four」를 들은 사람들이 대조군보다
한 살 반 **어려졌다**. p = 0.04 였다.

그들은 자료를 조작하지 않았다. 그때 누구나 쓰던 분석의 자유를 썼을
뿐이다 — 결과변수를 둘 재어 두고 맞는 쪽을 쓰고, 유의하지 않으면
표본을 더 모으고, 공변량을 넣었다 빼고, 조건 셋 중 둘만 보고한다.

네 가지를 각각, 그리고 함께 썼을 때의 거짓양성률을 직접 잰다.
조건 세 개에 칸당 20 명(필요하면 30 명), 결과변수 둘의 상관 0.5,
참 효과는 0 이다.

                        이 쪽의 모의    시먼스 등(2011)
  정직하게 하나만              0.053          0.050
  A  결과변수를 둘 재어 둔다    0.093          0.095
  B  모자라면 더 모은다        0.081          0.077
  C  성별을 넣었다 뺀다        0.126          0.117
  D  조건 셋 중 둘만 쓴다      0.121          0.126
  A+B                      0.138          0.144
  A+B+C                    0.313          0.309
  A+B+C+D                  0.601          0.607

하나씩은 다 작다. 넷을 함께 쓰면 스무 번에 한 번이 아니라 **열 번에
여섯 번** 유의해진다.

만드는 파일:

  ch09/multiple_testing/img/researcher_dof.png

실행:  python3 scripts/make_ch09_researcher_dof.py   (저장소 최상위에서)
       한 번에 몇 분 걸린다.
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import numpy as np
from scipy.stats import t as tdist

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch09/multiple_testing/img/"

INK = "#37474F"
MUTED = "#90A4AE"
GREY = "#78909C"
RED = "#D32F2F"
TEAL = "#00838F"
AMBER = "#B07C2C"
BLUE = "#1565C0"

NCELL, ADD, RHO, REPS, SEED = 20, 10, 0.5, 8000, 2011
PAIRS = ((0, 1), (0, 2), (1, 2))

#       이름                  고를 수 있는 길         시먼스 등이 보고한 값
SPECS = (("정직하게\n하나만", lambda k: k == (0, 0, 0, 0), 0.05),
         ("A\n결과변수 둘", lambda k: k[1] == 0 and k[2] == 0 and k[3] == 0, 0.095),
         ("B\n더 모은다", lambda k: k[0] == 0 and k[2] == 0 and k[3] == 0, 0.077),
         ("C\n성별", lambda k: k[0] == 0 and k[1] == 0 and k[2] == 0, 0.117),
         ("D\n조건 둘만", lambda k: k[0] == 0 and k[1] == 0 and k[3] == 0, 0.126),
         ("A$+$B", lambda k: k[2] == 0 and k[3] == 0, 0.144),
         ("A$+$B$+$C", lambda k: k[2] == 0, 0.309),
         ("A$+$B$+$C$+$D", lambda k: True, 0.607))


def tp(X, y):
    """조건 더미(둘째 열)에 붙는 양쪽꼬리 p."""
    n, k = X.shape
    if n - k <= 1:
        return 1.0
    XtX = X.T @ X
    b = np.linalg.solve(XtX, X.T @ y)
    r = y - X @ b
    se = np.sqrt((r @ r) / (n - k) * np.linalg.inv(XtX)[1, 1])
    return float(2 * tdist.sf(abs(b[1] / se), n - k))


def one_study(rr):
    """연구 한 건에서 연구자가 고를 수 있는 모든 분석의 p 를 모은다."""
    n2 = NCELL + ADD
    cond = np.repeat([0, 1, 2], n2)
    g = rr.integers(0, 2, 3 * n2).astype(float)          # 성별
    L = np.array([[1, 0], [RHO, np.sqrt(1 - RHO ** 2)]])
    Y = rr.normal(size=(3 * n2, 2)) @ L.T                # 결과변수 둘, 효과는 0
    out = {}
    for dv in (0, 1):
        for ni, ncell in enumerate((NCELL, n2)):
            keep = np.concatenate([np.where(cond == c)[0][:ncell]
                                   for c in range(3)])
            for pi, (a, b) in enumerate(PAIRS):
                m = keep[np.isin(cond[keep], (a, b))]
                d = (cond[m] == b).astype(float)
                gg, y, o = g[m], Y[m, dv], np.ones(len(m))
                out[(dv, ni, pi, 0)] = tp(np.c_[o, d], y)            # 그대로
                out[(dv, ni, pi, 1)] = tp(np.c_[o, d, gg], y)        # 성별 통제
                out[(dv, ni, pi, 2)] = tp(np.c_[o, d, gg, d * gg], y)  # 교호작용
                for s in (0, 1):                                      # 성별 안에서
                    q = gg == s
                    out[(dv, ni, pi, 3 + s)] = (tp(np.c_[o[q], d[q]], y[q])
                                                if q.sum() > 6 else 1.0)
    return out


def main():
    rr = np.random.default_rng(SEED)
    mins = np.empty((len(SPECS), REPS))
    for i in range(REPS):
        p = one_study(rr)
        for j, (_, sel, _) in enumerate(SPECS):
            mins[j, i] = min(v for k, v in p.items() if sel(k))
    rates = (mins < 0.05).mean(1)

    print(f"{'':>20}{'이 쪽의 모의':>14}{'시먼스 등':>12}")
    for (nm, _, ref), r in zip(SPECS, rates):
        print(f"{nm.replace(chr(10), ' '):>20}{r:>14.3f}{ref:>12.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.2), constrained_layout=True)

    # ── 왼쪽: 하나씩은 작고 함께 쓰면 크다 ──
    ax = axes[0]
    cols = [GREY] + [TEAL] * 4 + [AMBER, AMBER, RED]
    xs = np.arange(len(SPECS))
    ax.axhline(5, color=INK, lw=2.0, ls="--", label="약속한 $5\\%$")
    for x, (r, c) in enumerate(zip(rates, cols)):
        ax.bar(x, r * 100, width=0.6, color=c, alpha=0.9, zorder=3)
        ax.text(x, r * 100 + 1.4, f"${r*100:.1f}$", ha="center", fontsize=11.6,
                color=c, fontweight="bold")
    refs = [s[2] * 100 for s in SPECS]
    ax.plot(xs, refs, lw=0, marker="_", ms=22, mew=2.6, color=INK, zorder=5,
            label="시먼스 등$(2011)$ 이 보고한 값")
    ax.set_xticks(xs)
    ax.set_xticklabels([s[0] for s in SPECS], fontsize=9.8)
    ax.set_ylabel("효과가 없는데 유의하다고 할 확률", fontsize=11.5)
    ax.set_title("하나씩은 작고, 함께 쓰면 크다", fontsize=12.8, color=INK,
                 pad=8)
    ax.set_ylim(0, 72)
    ax.set_yticks([0, 5, 20, 40, 60])
    ax.set_yticklabels(["$0\\%$", "$5\\%$", "$20\\%$", "$40\\%$", "$60\\%$"])
    ax.legend(fontsize=10.4, framealpha=0.93, loc="upper left")
    ax.grid(alpha=0.25, axis="y")
    ax.spines[["top", "right"]].set_visible(False)

    # ── 오른쪽: 보고된 수는 더 이상 p-값이 아니다 ──
    ax = axes[1]
    grid = np.linspace(0, 1, 400)
    hon = np.searchsorted(np.sort(mins[0]), grid) / REPS
    hak = np.searchsorted(np.sort(mins[-1]), grid) / REPS
    ax.plot(grid, hon, lw=3.2, color=GREY, ls="--",
            label="정직하게 분석 하나만 — 고르게 퍼진다")
    ax.plot(grid, hak, lw=3.6, color=RED,
            label="네 자유를 다 쓰고 가장 작은 것을 보고")
    ax.axvline(0.05, color=INK, lw=1.8, ls=":")
    ax.plot([0.05], [rates[-1]], marker="o", ms=13, color=RED, zorder=5,
            mec="white", mew=1.7)
    ax.plot([0.05], [rates[0]], marker="D", ms=11, color=GREY, zorder=5,
            mec="white", mew=1.7)
    ax.annotate(f"${rates[-1]*100:.1f}\\%$", xy=(0.05, rates[-1]),
                xytext=(0.23, 0.56), fontsize=12.4, color=RED, ha="center",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))
    ax.annotate(f"${rates[0]*100:.1f}\\%$", xy=(0.05, rates[0]),
                xytext=(0.20, 0.13), fontsize=12.4, color=GREY, ha="center",
                arrowprops=dict(arrowstyle="->", color=GREY, lw=1.5))
    ax.text(0.063, 0.30, "$p = 0.05$", fontsize=10.8, color=INK)
    ax.set_xlabel("보고된 $p$-값", fontsize=11.5)
    ax.set_ylabel("그보다 작을 확률 (누적)", fontsize=11.5)
    ax.set_title("보고된 수는 더 이상 $p$-값이 아니다", fontsize=12.8,
                 color=INK, pad=8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.03)
    ax.legend(fontsize=10.4, framealpha=0.93, loc="lower right")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("비틀스를 들으면 어려진다", fontsize=14.5)
    fig.text(0.5, -0.07,
             f"조건 세 개에 칸당 ${NCELL}$ 명(모자라면 ${NCELL+ADD}$ 명), "
             f"결과변수 둘의 상관 ${RHO}$, 참 효과는 $0$ 이다. "
             f"연구 ${REPS//1000}{{,}}000$ 건을 모의했다. 자료를 조작한 분석은 하나도 "
             "없다 — 모두 그 자체로는 흔히 쓰이던 선택들이다.",
             ha="center", fontsize=10.4, color=INK)

    path = OUT + "researcher_dof.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
