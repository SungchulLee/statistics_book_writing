r"""5.7절 「그림으로 보는 통계학」 — 정확한 답이 없는 문제.

두 모평균을 견줄 때 분산이 다르면 정확한 분포가 없다. 베렌스-피셔 문제다.
합동 t 는 등분산을 전제하므로, 분산이 다르고 표본크기가 불균형하면
명목 95% 가 65% 까지 내려앉는다. 웰치의 어림은 그 자리에서 94.7% 를 지킨다.

가로축에 분산비를 두고 두 방법의 실제 포함률을 그린다. 작은 표본 쪽의
분산이 클수록(오른쪽) 합동 t 가 무너진다.

만드는 파일:

  ch05/applications/img/behrens_fisher.png

실행:  python3 scripts/make_ch05_behrens_fisher.py   (저장소 최상위에서)
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

OUT = "docs/ch05/applications/img/"
INK, RED, TEAL, MUTED = "#37474F", "#D32F2F", "#00838F", "#90A4AE"

N1, N2, B, SEED = 5, 25, 40_000, 12
RATIOS = [0.25, 0.5, 1, 2, 4, 8, 16]


def main():
    rng = np.random.default_rng(SEED)
    pooled, welch = [], []
    for r in RATIOS:
        s1 = np.sqrt(r)                       # 작은 표본 쪽의 표준편차
        a = rng.normal(0, s1, (B, N1)); b = rng.normal(0, 1.0, (B, N2))
        m1, m2 = a.mean(1), b.mean(1)
        v1, v2 = a.var(1, ddof=1), b.var(1, ddof=1)
        sp = ((N1-1)*v1 + (N2-1)*v2) / (N1+N2-2)
        tp = (m1-m2) / np.sqrt(sp*(1/N1+1/N2))
        pooled.append(np.mean(np.abs(tp) <= stats.t.ppf(.975, N1+N2-2)))
        se = np.sqrt(v1/N1 + v2/N2)
        df = se**4 / ((v1/N1)**2/(N1-1) + (v2/N2)**2/(N2-1))
        welch.append(np.mean(np.abs((m1-m2)/se) <= stats.t.ppf(.975, df)))

    fig, ax = plt.subplots(figsize=(9.8, 5.2), constrained_layout=True)
    ax.axhline(95, color=MUTED, lw=1.8, ls="--")
    ax.text(RATIOS[-1], 95.6, "명목 95%", ha="right", fontsize=10.5, color=MUTED)
    ax.plot(RATIOS, np.array(welch)*100, marker="o", ms=9, lw=2.6, color=TEAL,
            label="웰치 — 자유도를 자료에서 고친다")
    ax.plot(RATIOS, np.array(pooled)*100, marker="s", ms=9, lw=2.6, color=RED,
            label="합동 t — 등분산을 전제한다")
    ax.axvline(1, color=INK, lw=1.0, ls=":")
    ax.text(1.06, 62, "분산이 같은 자리", fontsize=9.8, color=INK)

    i = RATIOS.index(16)
    ax.annotate(f"{pooled[i]*100:.0f}%", xy=(16, pooled[i]*100),
                xytext=(9, pooled[i]*100-7), fontsize=12, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.6))

    ax.set_xscale("log")
    ax.set_xticks(RATIOS); ax.set_xticklabels([str(r) for r in RATIOS])
    ax.minorticks_off()
    ax.set_xlabel(f"분산비 — 작은 표본(n={N1}) 쪽이 큰 표본(n={N2})보다 몇 배 퍼져 있는가",
                  fontsize=11)
    ax.set_ylabel("명목 95% 구간의 실제 포함률 (%)", fontsize=11.5)
    ax.set_title("분산이 다르면 정확한 답이 없다 — 어림이 전제보다 낫다",
                 fontsize=13.5, color=INK, pad=10)
    ax.set_ylim(58, 100)
    ax.grid(alpha=0.25); ax.legend(fontsize=10.5, loc="lower left", framealpha=0.93)
    ax.spines[["top", "right"]].set_visible(False)

    path = OUT + "behrens_fisher.png"
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")
    for r, p_, w_ in zip(RATIOS, pooled, welch):
        print(f"    분산비 {r:5}  합동 t {p_*100:5.1f}%   웰치 {w_*100:5.1f}%")


if __name__ == "__main__":
    main()
