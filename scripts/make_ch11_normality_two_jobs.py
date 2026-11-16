r"""13.3절 「그림으로 보는 통계학」 — 정규성은 두 자리에서 다르게 쓰인다.

회귀에서 「오차가 정규」라는 가정은 두 곳에 들어가는데, 둘의 운명이
완전히 다르다.

  계수의 신뢰구간   b-hat 은 오차의 **평균 꼴**이라 중심극한정리가 받친다.
                    어떤 오차분포에서도 n = 10 에서 이미 95% 다.
  새 관측의 예측구간 새로 올 y 는 평균이 아니라 **오차 하나**다. 받쳐 줄
                    정리가 없고, n 을 키우면 95% 가 아니라 그 분포가
                    정한 다른 값으로 간다.

극한값은 손으로 적을 수 있다 — P(|eps| <= 1.96 sigma) 다.

  정규 0.9500 · 균등 1.0000 · 라플라스 0.9375 · 지수 0.9482 · t(3) 0.9574

만드는 파일:

  ch13/assumptions/img/normality_two_jobs.png

실행:  python3 scripts/make_ch13_normality_two_jobs.py   (저장소 최상위에서)
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

OUT = "docs/ch13/assumptions/img/"

INK = "#37474F"
MUTED = "#90A4AE"
Z = stats.norm.ppf(0.975)
REPS, SEED = 40_000, 5
NS = (10, 20, 50, 100, 200, 500)

KINDS = [
    ("정규", "#37474F", lambda r, s: r.normal(0, 1, s), 2 * stats.norm.cdf(Z) - 1),
    ("균등", "#1565C0", lambda r, s: r.uniform(-np.sqrt(3), np.sqrt(3), s), 1.0),
    ("라플라스", "#D32F2F", lambda r, s: r.laplace(0, 1 / np.sqrt(2), s),
     1 - np.exp(-Z * np.sqrt(2))),
    ("지수", "#B07C2C", lambda r, s: r.exponential(1, s) - 1.0, 1 - np.exp(-(Z + 1))),
    ("t(3)", "#00838F", lambda r, s: r.standard_t(3, s) / np.sqrt(3),
     2 * stats.t.cdf(Z * np.sqrt(3), 3) - 1),
]


def main():
    rng = np.random.default_rng(SEED)
    ci = {k: [] for k, *_ in KINDS}
    pi = {k: [] for k, *_ in KINDS}

    for n in NS:
        x = np.linspace(-1.5, 1.5, n)
        xc = x - x.mean()
        sxx = float((xc ** 2).sum())
        tc = stats.t.ppf(0.975, n - 2)
        for name, _, draw, _lim in KINDS:
            y = 1 + 2 * x + draw(rng, (REPS, n))
            ybar = y.mean(axis=1)
            yc = y - ybar[:, None]
            b = (yc * xc).sum(axis=1) / sxx
            e = yc - b[:, None] * xc
            s2 = (e ** 2).sum(axis=1) / (n - 2)

            ci[name].append(float(np.mean(np.abs(b - 2) <= tc * np.sqrt(s2 / sxx))))

            yhat = ybar + b * (0 - x.mean())                 # x0 = 0 에서의 예측
            ynew = 1.0 + draw(rng, REPS)
            half = tc * np.sqrt(s2 * (1 + 1 / n + (0 - x.mean()) ** 2 / sxx))
            pi[name].append(float(np.mean(np.abs(ynew - yhat) <= half)))

    print("    기울기 신뢰구간")
    for name, *_ in KINDS:
        print(f"      {name:6} " + " ".join(f"{v:.4f}" for v in ci[name]))
    print("    새 관측의 예측구간")
    for name, _, _, lim in KINDS:
        print(f"      {name:6} " + " ".join(f"{v:.4f}" for v in pi[name])
              + f"   극한 {lim:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.9), sharex=True,
                             constrained_layout=True)

    ax = axes[0]
    ax.axhline(95, color=MUTED, lw=2.0, ls="--")
    for name, col, _, _ in KINDS:
        ax.plot(NS, np.array(ci[name]) * 100, marker="o", ms=6, lw=2.2,
                color=col, label=name)
    ax.set_xscale("log")
    ax.set_xticks(NS)
    ax.set_xticklabels([str(n) for n in NS])
    ax.set_ylim(91, 101)
    ax.set_xlabel("표본크기 $n$", fontsize=11.5)
    ax.set_ylabel("명목 95% 의 실제 포함률 (%)", fontsize=11.5)
    ax.set_title("기울기의 신뢰구간 — 다섯이 겹친다",
                 fontsize=12.8, color=INK, pad=8)
    ax.text(12, 96.4, "$n = 10$ 에서 이미 다 맞는다", fontsize=10.4, color=MUTED)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=10, framealpha=0.93, ncol=2, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    ax.axhline(95, color=MUTED, lw=2.0, ls="--")
    for name, col, _, lim in KINDS:
        ax.plot(NS, np.array(pi[name]) * 100, marker="s", ms=6, lw=2.2, color=col)
        ax.axhline(lim * 100, color=col, lw=1.3, ls=":")
        ax.text(560, lim * 100, f" {lim*100:.1f}", fontsize=10, color=col,
                va="center")
    ax.set_xscale("log")
    ax.set_xlim(8, 900)
    ax.set_ylim(91, 101)
    for a in axes:                         # sharex 라 눈금을 한 번 더 박아 준다
        a.set_xticks(NS)
        a.set_xticklabels([str(v) for v in NS])
        a.minorticks_off()
    ax.set_xlabel("표본크기 $n$", fontsize=11.5)
    ax.set_title("새 관측의 예측구간 — 다섯이 갈라진다",
                 fontsize=12.8, color=INK, pad=8)
    ax.text(10.5, 92.0, "점선은 $n \\to \\infty$ 의 극한값\n"
                        "$P(|\\varepsilon| \\leq 1.96\\sigma)$ 그 자체다",
            fontsize=10.2, color=INK)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("같은 가정인데 한쪽은 표본이 구해 주고 한쪽은 아니다", fontsize=14.5)
    fig.text(0.5, -0.05,
             f"오차분포만 바꾸고 나머지는 같게 두었다({REPS//1000} 천 번). "
             "왼쪽이 구해지는 까닭은 $\\hat\\beta$ 이 오차의 평균 꼴이기 때문이고, "
             "오른쪽이 안 되는 까닭은 새로 올 $y$ 가 평균이 아니라 오차 하나이기 때문이다.",
             ha="center", fontsize=10.4, color=INK)

    path = OUT + "normality_two_jobs.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
