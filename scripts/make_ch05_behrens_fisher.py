r"""5.7절 「그림으로 보는 통계학」 — 평균에는 맞고 내 경우에는 틀린다.

이 쪽은 웰치의 자유도 어림을 유도하고 그 포함확률이 좋다는 것을 보인다.
맞는 말이다. 그런데 피셔는 바로 그 점을 두고 웰치를 공격했다.

베렌스(1929)가 두 분산이 다른 두 평균의 비교에 답을 내놓았고, 피셔가
1935 년에 그것을 피두셜 논증으로 다시 세웠다. 웰치(1938, 1947)는 다른
길로 갔다 — 자유도를 자료에서 어림해 t 분포에 맞추는 것이다.

피셔의 반론은 이랬다. 웰치의 구간은 **상상할 수 있는 모든 실험에 걸쳐**
평균을 내면 95% 를 담는다. 그러나 내가 지금 손에 쥔 자료에서 s1 과 s2 를
보고 나면, 나는 내가 어느 무리에 속하는지 알 수 있다. 그 무리 안에서의
포함확률이 95% 가 아니라면 「이 구간이 95%」라는 말은 무슨 뜻인가.

재 보면 피셔가 옳다.

  n = (6, 24), sigma 가 같을 때   전체 포함확률 0.9459
    관측 분산비가 작은 쪽 20%        0.8534
    관측 분산비가 큰 쪽 20%         0.9953

평균은 맞는데 어느 층에서도 맞지 않는다. 표본크기가 같으면(n = 10, 10)
이 흔들림이 사라진다 — 어느 층에서나 0.950 이다. 범인은 **불균형**이다.

그렇다고 베렌스-피셔 쪽이 답인 것도 아니다. 리닉(1966)을 비롯한 결과로
이 문제에는 정확한 해가 **없다**는 것이 알려져 있다. 없는 것을 찾는 것이
아니라 무엇을 포기할지 고르는 문제다. 오늘날 소프트웨어는 대부분 웰치를
기본값으로 쓴다 — 전체 포함확률을 지키는 쪽을 고른 것이다.

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

INK = "#37474F"
MUTED = "#90A4AE"
GREY = "#78909C"
RED = "#D32F2F"
TEAL = "#00838F"
AMBER = "#B07C2C"
BLUE = "#1565C0"

REPS, SEED, NBIN = 400_000, 1947, 14

CASES = [
    ("$n = (10,\\ 10)$,  $\\sigma$ 같음", 10, 10, 1.0, 1.0, TEAL),
    ("$n = (6,\\ 24)$,  $\\sigma$ 같음", 6, 24, 1.0, 1.0, RED),
    ("$n = (24,\\ 6)$,  $\\sigma_2 = 3\\sigma_1$", 24, 6, 1.0, 3.0, AMBER),
]


def welch(n1, n2, s1, s2, rng):
    a = rng.normal(0, s1, (REPS, n1))
    b = rng.normal(0, s2, (REPS, n2))
    v1 = a.var(1, ddof=1)
    v2 = b.var(1, ddof=1)
    se = np.sqrt(v1 / n1 + v2 / n2)
    nu = (v1 / n1 + v2 / n2) ** 2 / ((v1 / n1) ** 2 / (n1 - 1)
                                     + (v2 / n2) ** 2 / (n2 - 1))
    cov = np.abs(a.mean(1) - b.mean(1)) <= stats.t.ppf(0.975, nu) * se
    return cov, v1 / v2


def main():
    rng = np.random.default_rng(SEED)
    out = []
    for lab, n1, n2, s1, s2, col in CASES:
        cov, ratio = welch(n1, n2, s1, s2, rng)
        q = np.quantile(ratio, np.linspace(0, 1, NBIN + 1))
        mid, cc = [], []
        for i in range(NBIN):
            m = (ratio >= q[i]) & (ratio < q[i + 1] if i < NBIN - 1
                                   else ratio <= q[i + 1])
            mid.append(np.sqrt(q[i] * q[i + 1]))
            cc.append(cov[m].mean())
        out.append((lab, col, float(cov.mean()), np.array(mid), np.array(cc)))
        plain = lab.replace("$", "").replace("\\", "").replace("sigma", "σ")
        print(f"    {plain:<28} 전체 {cov.mean():.4f}   "
              f"층별 {min(cc):.4f} ~ {max(cc):.4f}   폭 {max(cc)-min(cc):.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.1), constrained_layout=True)

    # ── 왼쪽: 관측한 분산비로 층을 가르면 ──
    ax = axes[0]
    ax.axhline(0.95, color=INK, lw=2.2, ls="--")
    ax.text(0.012, 0.9555, "약속한 $0.95$", fontsize=11, color=INK)
    for lab, col, tot, mid, cc in out:
        ax.plot(mid, cc, lw=2.8, color=col, marker="o", ms=5, label=lab)
    ax.set_xscale("log")
    ax.set_xticks([0.02, 0.1, 0.5, 2, 10])
    ax.set_xticklabels(["0.02", "0.1", "0.5", "2", "10"])
    ax.minorticks_off()
    lab, col, tot, mid, cc = out[1]
    ax.annotate(f"같은 절차인데\n이 층에서는 {cc[0]:.3f}",
                xy=(mid[0], cc[0]), xytext=(mid[0] * 2.6, 0.745),
                fontsize=10.8, color=RED, ha="left",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.4))
    ax.annotate(f"이 층에서는 {cc[-1]:.3f}",
                xy=(mid[-1], cc[-1]), xytext=(mid[-1] * 0.22, 1.012),
                fontsize=10.8, color=RED, ha="right",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.4))
    ax.set_xlabel("내 자료에서 관측한 분산비  $s_1^2 / s_2^2$", fontsize=11.5)
    ax.set_ylabel("그런 자료들만 모았을 때의 포함확률", fontsize=11.5)
    ax.set_title("자료를 보고 나면 내가 어느 무리인지 알 수 있다",
                 fontsize=12.6, color=INK, pad=8)
    ax.set_ylim(0.695, 1.025)
    ax.legend(fontsize=10.2, framealpha=0.93, loc="lower right")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    # ── 오른쪽: 전체는 맞는다 ──
    ax = axes[1]
    ys = np.arange(len(out))[::-1]
    for y, (lab, col, tot, mid, cc) in zip(ys, out):
        ax.plot([cc.min(), cc.max()], [y, y], lw=9, color=col, alpha=0.35,
                solid_capstyle="round")
        ax.plot([tot], [y], marker="o", ms=13, color=col, zorder=5)
        ax.text(cc.min() - 0.004, y, f"{cc.min():.3f}", ha="right", va="center",
                fontsize=10.4, color=col)
        ax.text(cc.max() + 0.004, y, f"{cc.max():.3f}", ha="left", va="center",
                fontsize=10.4, color=col)
        ax.text(0.702, y + 0.30, lab, fontsize=11.0, color=col, ha="left")
        ax.text(tot, y - 0.36, f"전체 {tot:.4f}", ha="center", va="top",
                fontsize=10.2, color=INK)
    ax.axvline(0.95, color=INK, lw=2.2, ls="--")
    ax.text(0.9545, len(out) - 0.44, "약속한 $0.95$", fontsize=11, color=INK)
    ax.set_yticks([])
    ax.set_ylim(-0.75, len(out) - 0.35)
    ax.set_xlim(0.695, 1.035)
    ax.set_xlabel("포함확률 — 점은 전체, 띠는 층별 범위", fontsize=11.5)
    ax.set_title("평균을 맞추는 일과 내 경우를 맞추는 일은 다르다",
                 fontsize=12.6, color=INK, pad=8)
    ax.grid(alpha=0.25, axis="x")
    ax.spines[["top", "right", "left"]].set_visible(False)

    fig.suptitle("웰치의 구간은 평균에는 맞고 내 경우에는 틀릴 수 있다",
                 fontsize=14.5)
    fig.text(0.5, -0.05,
             f"세 경우 모두 참 평균차가 $0$ 이고 모의실험은 각 {REPS//10000} 만 번이다. "
             "관측한 분산비로 $14$ 개 층을 같은 크기로 갈랐다.\n"
             "표본크기가 같으면(청록) 층을 갈라도 흔들리지 않는다 — "
             "흔들림을 만드는 것은 표본크기의 불균형이다.",
             ha="center", fontsize=10.4, color=INK)

    path = OUT + "behrens_fisher.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
