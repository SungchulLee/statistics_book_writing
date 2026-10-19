r"""13.4절 「그림으로 보는 통계학」 — 띠가 셋인데 묻는 것이 다 다르다.

회귀직선 둘레에 그릴 수 있는 띠가 하나가 아니다. 무엇을 95% 로 담고
싶으냐에 따라 폭이 달라진다.

  점별 신뢰띠   정해 둔 x0 한 곳에서 **평균** E[y|x0] 을 담는다
  동시 신뢰띠   **직선 전체**를 한꺼번에 담는다 (워킹-호텔링)
  예측띠        그 자리에 올 **새 관측 하나**를 담는다

점별 신뢰띠로 직선 전체를 말하면 약속이 깨진다. n = 30 에서 그 띠가
직선을 통째로 담을 확률은 95% 가 아니라 85.8% 이고, 이 값은 정확히
P(F(2, n-2) <= t^2/2) 다.

만드는 파일:

  ch13/estimation_inference/img/three_bands.png

실행:  python3 scripts/make_ch13_three_bands.py   (저장소 최상위에서)
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

OUT = "docs/ch13/estimation_inference/img/"

INK = "#37474F"
AMBER = "#B07C2C"
TEAL = "#00838F"
RED = "#D32F2F"
MUTED = "#90A4AE"

N, REPS, SEED = 30, 40_000, 44
A0, B0, SIGMA = 1.0, 2.0, 1.0
X = np.linspace(-2, 2, N)
XC = X - X.mean()
SXX = float((XC ** 2).sum())
TC = stats.t.ppf(0.975, N - 2)
WH = float(np.sqrt(2 * stats.f.ppf(0.95, 2, N - 2)))      # 워킹-호텔링 배수


def main():
    rng = np.random.default_rng(SEED)
    y = A0 + B0 * X + rng.normal(0, SIGMA, (REPS, N))
    ybar = y.mean(axis=1)
    b = ((y - ybar[:, None]) * XC).sum(axis=1) / SXX
    a = ybar - b * X.mean()
    e = y - (a[:, None] + b[:, None] * X)
    s = np.sqrt((e ** 2).sum(axis=1) / (N - 2))

    g = np.linspace(-2, 2, 81)
    lev = 1 / N + (g - X.mean()) ** 2 / SXX                 # 평균의 분산 인자
    fit = a[:, None] + b[:, None] * g
    truth = A0 + B0 * g
    sl = s[:, None] * np.sqrt(lev)

    mid = len(g) // 2                                       # x0 = 0
    cov_point = float(np.mean(np.abs(fit[:, mid] - truth[mid]) <= TC * sl[:, mid]))
    cov_all_pt = float(np.mean((np.abs(fit - truth) <= TC * sl).all(axis=1)))
    cov_all_wh = float(np.mean((np.abs(fit - truth) <= WH * sl).all(axis=1)))
    ynew = A0 + B0 * 0.0 + rng.normal(0, SIGMA, REPS)
    half_pi = TC * s * np.sqrt(1 + lev[mid])
    cov_pred = float(np.mean(np.abs(ynew - fit[:, mid]) <= half_pi))
    exact_all_pt = float(stats.f.cdf(TC ** 2 / 2, 2, N - 2))

    print(f"    점별 신뢰띠 · 한 점의 평균     {cov_point:.4f}")
    print(f"    점별 신뢰띠 · 직선 전체       {cov_all_pt:.4f}   (관측 범위 안)")
    print(f"       같은 띠를 x 전체로 늘리면    {exact_all_pt:.4f}   (정확값)")
    print(f"    워킹-호텔링 · 직선 전체       {cov_all_wh:.4f}")
    print(f"    예측띠 · 새 관측 하나         {cov_pred:.4f}")
    print(f"    배수   t = {TC:.4f}   워킹-호텔링 = {WH:.4f}  ({WH/TC:.2f} 배 넓다)")

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.9), constrained_layout=True)

    # ── 왼쪽: 한 표본에 세 띠 ──
    ax = axes[0]
    r2 = np.random.default_rng(9)
    ys = A0 + B0 * X + r2.normal(0, SIGMA, N)
    yb = ys.mean()
    bs = float(((ys - yb) * XC).sum() / SXX)
    as_ = yb - bs * X.mean()
    ss = float(np.sqrt(((ys - (as_ + bs * X)) ** 2).sum() / (N - 2)))
    f = as_ + bs * g
    sg = ss * np.sqrt(lev)

    ax.fill_between(g, f - TC * ss * np.sqrt(1 + lev), f + TC * ss * np.sqrt(1 + lev),
                    color=AMBER, alpha=0.18, label="예측띠 — 새 관측 하나")
    ax.fill_between(g, f - WH * sg, f + WH * sg, color=TEAL, alpha=0.28,
                    label="동시 신뢰띠 — 직선 전체")
    ax.fill_between(g, f - TC * sg, f + TC * sg, color=RED, alpha=0.35,
                    label="점별 신뢰띠 — 한 점의 평균")
    ax.plot(g, f, lw=2.4, color=INK)
    ax.scatter(X, ys, s=22, color=INK, zorder=5)
    ax.set_xlabel("$x$", fontsize=12)
    ax.set_ylabel("$y$", fontsize=12)
    ax.set_title(f"같은 자료에 그린 세 띠  (n = {N})",
                 fontsize=12.8, color=INK, pad=8)
    ax.legend(fontsize=9.8, framealpha=0.93, loc="upper left")
    ax.grid(alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)

    # ── 오른쪽: 각각이 실제로 담는 비율 ──
    ax = axes[1]
    labels = ["점별 신뢰띠로\n한 점의 평균", "점별 신뢰띠로\n직선 전체\n(관측 범위 안)",
              "동시 신뢰띠로\n직선 전체", "예측띠로\n새 관측 하나"]
    vals = [cov_point, cov_all_pt, cov_all_wh, cov_pred]
    cols = [RED, "#B71C1C", TEAL, AMBER]
    bars = ax.bar(range(4), np.array(vals) * 100, color=cols, alpha=0.85, width=0.62)
    ax.axhline(95, color=MUTED, lw=2.0, ls="--")
    for i, v in enumerate(vals):
        ax.text(i, v * 100 + 0.6, f"{v*100:.1f}%", ha="center",
                fontsize=11.5, color=INK)
    bars[1].set_hatch("//")
    ax.annotate("쓰려는 곳이 어긋났다", xy=(1, cov_all_pt * 100),
                xytext=(1.6, 88.0), fontsize=10.8, color="#B71C1C",
                arrowprops=dict(arrowstyle="->", color="#B71C1C", lw=1.4))
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(80, 99)
    ax.set_ylabel("실제로 담는 비율 (%)", fontsize=11.5)
    ax.set_title(f"넷 다 「명목 95%」라 부른다  ({REPS//1000} 천 번)",
                 fontsize=12.8, color=INK, pad=8)
    ax.grid(alpha=0.22, axis="y")
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("무엇을 담고 싶은지 정하지 않으면 폭을 정할 수 없다", fontsize=14.5)
    fig.text(0.5, -0.05,
             f"둘째 막대가 흔한 실수다. 점별 띠를 그려 놓고 「참 직선이 이 안에 있다」고 "
             f"읽으면 관측 범위 안에서도 ${cov_all_pt*100:.1f}\\%$ 이고, "
             f"$x$ 를 끝까지 늘리면 $P(F_{{2,{N-2}}} \\leq t^2/2) = {exact_all_pt*100:.1f}\\%$ 로 "
             "더 내려간다.",
             ha="center", fontsize=10.4, color=INK)

    path = OUT + "three_bands.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
