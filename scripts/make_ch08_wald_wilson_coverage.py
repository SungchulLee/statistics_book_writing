r"""5.5절 「그림으로 보는 통계학」 — 왈드 구간이 어디서 무너지는가.

n = 40, 명목 95% 에서 참 p 를 0.01 부터 0.99 까지 옮겨 가며 실제 포함률을
정확히 센다(이항은 유한하므로 모의가 아니라 합으로 셀 수 있다).

  왈드  p 가 0 이나 1 에 가까울수록 급격히 무너진다. p=0.05 에서 86.8%.
  윌슨  같은 자리에서 95.2% 로 버틴다.

톱니처럼 보이는 것은 k/n 이 이산이라 구간의 양 끝이 뛰기 때문이다.

만드는 파일:

  ch05/applications/img/wald_wilson_coverage.png

실행:  python3 scripts/make_ch05_wald_wilson_coverage.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch08/one_sample_intervals/img/"

INK = "#37474F"
RED = "#D32F2F"
TEAL = "#00838F"
MUTED = "#90A4AE"

N, ZC = 40, 1.96


def coverage(ps):
    """각 p 에서 두 구간의 실제 포함률을 정확히 센다."""
    k = np.arange(N + 1)
    ph = k / N
    lo_w = ph - ZC * np.sqrt(ph * (1 - ph) / N)
    hi_w = ph + ZC * np.sqrt(ph * (1 - ph) / N)
    d = 1 + ZC**2 / N
    c = (ph + ZC**2 / (2 * N)) / d
    hw = ZC / d * np.sqrt(ph * (1 - ph) / N + ZC**2 / (4 * N**2))
    lo_s, hi_s = c - hw, c + hw

    w, s = [], []
    for p in ps:
        pm = stats.binom.pmf(k, N, p)
        w.append(pm[(lo_w <= p) & (p <= hi_w)].sum())
        s.append(pm[(lo_s <= p) & (p <= hi_s)].sum())
    return np.array(w), np.array(s)


def main():
    ps = np.linspace(0.01, 0.99, 981)
    w, s = coverage(ps)

    fig, ax = plt.subplots(figsize=(10.2, 5.2), constrained_layout=True)

    ax.axhline(95, color=MUTED, lw=1.8, ls="--", zorder=2)
    ax.text(0.985, 95.35, "명목 95%", ha="right", fontsize=10.5, color=MUTED)

    ax.plot(ps, s * 100, lw=2.0, color=TEAL, zorder=4, label="윌슨 (1927)")
    ax.plot(ps, w * 100, lw=1.6, color=RED, zorder=3, label="왈드 — 교과서가 먼저 가르치는 쪽")

    for p in (0.05, 0.5):
        wv = coverage([p])[0][0] * 100
        ax.plot([p], [wv], marker="o", ms=9, color=RED, zorder=6)
        ax.annotate(f"p = {p}\n{wv:.1f}%", xy=(p, wv), xytext=(p + 0.055, wv - 9),
                    fontsize=10.5, color=RED,
                    arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))

    ax.set_xlabel("참 비율 p", fontsize=11.5)
    ax.set_ylabel("명목 95% 구간의 실제 포함률 (%)", fontsize=11.5)
    ax.set_title(f"같은 95% 인데 한쪽만 무너진다  (n = {N})",
                 fontsize=13.5, color=INK, pad=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(60, 101)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=10.5, loc="lower center", framealpha=0.93)
    ax.spines[["top", "right"]].set_visible(False)

    fig.text(0.5, -0.045,
             "톱니 모양은 k/n 이 이산이라 생기는 것이고, 왈드가 양쪽 끝에서 깊이 꺼지는 것은 "
             "p-hat 이 0 이나 1 에 닿으면 구간의 폭이 0 이 되어 버리기 때문이다.",
             ha="center", fontsize=10.5, color=INK)

    path = OUT + "wald_wilson_coverage.png"
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")
    for p in (0.05, 0.10, 0.20, 0.50):
        a, b = coverage([p])
        print(f"    p={p:.2f}   왈드 {a[0]*100:5.1f}%   윌슨 {b[0]*100:5.1f}%")


if __name__ == "__main__":
    main()
