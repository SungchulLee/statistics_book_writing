r"""11장 실무 응용 두 쪽의 개념 그림을 생성한다.

만드는 파일:
  ch11/practical_applications/img/ab_peeking.png       엿보기가 제1종 오류를 부풀린다
  ch11/practical_applications/img/finance_regime.png   주효과가 없어도 교호작용이 있을 수 있다

실행:  python3 scripts/make_ch11_practical_figures.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib, statsmodels — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""
import os

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

INK = "#37474F"
BLUE, BLUE_F = "#1565C0", "#DCEBFB"
ORANGE, ORANGE_F = "#E65100", "#FFE0B2"
GREEN, GREEN_F = "#33691E", "#C5E1A5"
PURPLE = "#6A1B9A"
MUTED = "#90A4AE"
RED = "#D32F2F"

OUT = "docs/ch11/practical_applications/img/"
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    path = OUT + name
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def clean(ax):
    ax.tick_params(labelsize=10, colors=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)


# =====================================================================
# 1. 엿보기 — ab_testing.md
# =====================================================================
def fig_peeking():
    rng = np.random.default_rng(2024)
    N = 5_000                      # 팔당 최종 표본크기
    L = 50                         # 최대 확인 횟수
    idx = np.arange(1, L + 1) * (N // L) - 1     # 확인 시점의 인덱스
    ns = idx + 1
    M = 4_000
    chunk = 250

    pmat = np.empty((M, L))
    done = 0
    while done < M:
        b = min(chunk, M - done)
        x = rng.standard_normal((b, N))
        y = rng.standard_normal((b, N))          # 두 팔이 정확히 같은 분포
        out = []
        for a in (x, y):
            cs = np.cumsum(a, axis=1)[:, idx]
            cq = np.cumsum(a ** 2, axis=1)[:, idx]
            m = cs / ns
            v = (cq - cs ** 2 / ns) / (ns - 1)
            out.append((m, v))
        (mx, vx), (my, vy) = out
        sp2 = (vx + vy) / 2
        t = (mx - my) / np.sqrt(sp2 * 2 / ns)
        pmat[done:done + b] = 2 * stats.t.sf(np.abs(t), 2 * ns - 2)
        done += b

    running_min = np.minimum.accumulate(pmat, axis=1)
    Ks = [1, 2, 5, 10, 25, 50]
    rates = []
    for K in Ks:
        step = L // K
        cols = np.arange(step - 1, L, step)
        rates.append(np.mean(pmat[:, cols].min(axis=1) < 0.05))
        print(f"  확인 {K:2d} 회: 실제 제1종 오류율 {rates[-1]:.4f}")
    print(f"  한 번만 보면 {rates[0]:.4f}, 50 번 보면 {rates[-1]:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.5))

    ax = axes[0]
    show = 18
    hit = running_min[:show, -1] < 0.05
    for i in range(show):
        c = RED if hit[i] else MUTED
        ax.plot(ns, pmat[i], color=c, lw=1.5 if hit[i] else 1.0,
                alpha=0.95 if hit[i] else 0.5)
    ax.axhline(0.05, color=BLUE, lw=2, ls="--", label="$p$ = 0.05")
    ax.set_xlabel("지금까지 모은 팔당 표본크기", fontsize=10.5, color=INK)
    ax.set_ylabel("그 시점의 $p$ 값", fontsize=10.5, color=INK)
    ax.set_xlim(0, N)
    ax.set_ylim(0, 1.0)
    ax.set_title(f"두 팔이 완전히 같은 실험 {show} 개의 $p$ 값 궤적",
                 fontsize=11.5, color=INK)
    ax.plot([], [], color=RED, lw=1.8,
            label=f"도중에 0.05 아래로 내려간 실험 {hit.sum()} 개")
    ax.plot([], [], color=MUTED, lw=1.2, label="한 번도 내려가지 않은 실험")
    h, lbs = ax.get_legend_handles_labels()
    ax.legend(h[1:] + h[:1], lbs[1:] + lbs[:1], fontsize=9.5,
              loc="upper right")
    clean(ax)

    ax = axes[1]
    ax.plot(Ks, rates, "o-", color=RED, lw=2.4, ms=9)
    for k, r in zip(Ks, rates):
        ax.annotate(f"{r:.3f}", (k, r), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=10, color=RED)
    ax.axhline(0.05, color=BLUE, lw=1.8, ls="--")
    ax.text(50, 0.062, "명목 0.05", color=BLUE, fontsize=10.5, ha="right")
    ax.set_xscale("log")
    ax.set_xticks(Ks)
    ax.set_xticklabels([str(k) for k in Ks], fontsize=10)
    ax.minorticks_off()
    ax.set_xlabel("실험 도중 결과를 확인한 횟수 (로그 눈금)", fontsize=10.5,
                  color=INK)
    ax.set_ylabel("실제 제1종 오류율", fontsize=10.5, color=INK)
    ax.set_ylim(0, 0.34)
    ax.set_title("효과가 전혀 없는데도 '유의'가 나올 확률", fontsize=11.5,
                 color=INK)
    clean(ax)

    fig.tight_layout()
    save(fig, "ab_peeking.png")


# =====================================================================
# 2. 국면과 전략의 교호작용 — finance_applications.md
# =====================================================================
def fig_regime():
    import pandas as pd
    from statsmodels.formula.api import ols
    import statsmodels.api as sm

    rng = np.random.default_rng(7)
    regimes = ["상승장", "횡보장", "하락장"]
    strats = ["모멘텀", "방어형"]
    cell = {("모멘텀", "상승장"): 3.0, ("모멘텀", "횡보장"): 0.0,
            ("모멘텀", "하락장"): -2.4,
            ("방어형", "상승장"): 0.5, ("방어형", "횡보장"): 0.2,
            ("방어형", "하락장"): -0.1}
    n, sd = 20, 1.5
    rows = []
    for s in strats:
        for r in regimes:
            for v in rng.normal(cell[(s, r)], sd, n):
                rows.append({"strategy": s, "regime": r, "ret": v})
    df = pd.DataFrame(rows)

    model = ols("ret ~ C(strategy) * C(regime)", data=df).fit()
    tab = sm.stats.anova_lm(model, typ=2)
    Fs = [tab.loc["C(strategy)", "F"], tab.loc["C(regime)", "F"],
          tab.loc["C(strategy):C(regime)", "F"]]
    ps = [tab.loc["C(strategy)", "PR(>F)"], tab.loc["C(regime)", "PR(>F)"],
          tab.loc["C(strategy):C(regime)", "PR(>F)"]]
    dfn = [1, 2, 2]
    print(tab.round(4))

    a = df.loc[df.strategy == "모멘텀", "ret"].to_numpy()
    b = df.loc[df.strategy == "방어형", "ret"].to_numpy()
    t_naive = stats.ttest_ind(a, b)
    print(f"  국면을 무시한 비교: 평균 {a.mean():.4f} 대 {b.mean():.4f}, "
          f"t = {t_naive.statistic:.4f}, p = {t_naive.pvalue:.4f}")
    means = {(s, r): df[(df.strategy == s) & (df.regime == r)].ret.mean()
             for s in strats for r in regimes}
    ses = {(s, r): df[(df.strategy == s) & (df.regime == r)].ret.std(ddof=1)
           / np.sqrt(n) for s in strats for r in regimes}
    for kk, vv in means.items():
        print(f"  {kk}: 표본평균 {vv:.3f}")

    fig = plt.figure(figsize=(12.8, 4.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 0.85, 1], wspace=0.38)

    # (a) 교호작용 그림
    ax = fig.add_subplot(gs[0, 0])
    xs = np.arange(3)
    cols = {"모멘텀": ORANGE, "방어형": BLUE}
    mk = {"모멘텀": "o", "방어형": "s"}
    for s in strats:
        jj = rng.uniform(-0.09, 0.09, (3, n))
        for j, r in enumerate(regimes):
            vals = df[(df.strategy == s) & (df.regime == r)].ret
            ax.scatter(xs[j] + jj[j], vals, s=14, color=cols[s], alpha=0.3,
                       edgecolor="none")
        m = [means[(s, r)] for r in regimes]
        e = [ses[(s, r)] for r in regimes]
        ax.errorbar(xs, m, yerr=e, fmt=mk[s] + "-", color=cols[s], lw=2.6,
                    ms=10, capsize=5, label=s, zorder=4)
    ax.axhline(0, color=MUTED, lw=1, ls=":")
    ax.set_xticks(xs)
    ax.set_xticklabels(regimes, fontsize=11)
    ax.set_xlim(-0.4, 2.4)
    ax.set_ylabel("월 수익률 (%)", fontsize=10.5, color=INK)
    ax.set_title("국면마다 순위가 뒤바뀐다", fontsize=11.5, color=INK)
    ax.legend(fontsize=10.5, loc="lower left")
    clean(ax)

    # (b) 국면을 무시하면
    ax = fig.add_subplot(gs[0, 1])
    mm = [a.mean(), b.mean()]
    ee = [a.std(ddof=1) / np.sqrt(len(a)), b.std(ddof=1) / np.sqrt(len(b))]
    ax.bar([0, 1], mm, yerr=ee, width=0.5, capsize=7,
           color=[ORANGE_F, BLUE_F], edgecolor=[ORANGE, BLUE], lw=1.8,
           error_kw=dict(ecolor=INK, lw=1.6))
    for i, v in enumerate(mm):
        off = ee[i] + 0.06 if v >= 0 else -(ee[i] + 0.10)
        ax.text(i, v + off, f"{v:.3f}", ha="center", fontsize=11, color=INK)
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(strats, fontsize=11)
    ax.set_ylim(-0.75, 0.8)
    ax.set_ylabel("전체 기간 평균 수익률 (%)", fontsize=10.5, color=INK)
    ax.set_title(f"두 전략이 똑같아 보인다\n$p$ = {t_naive.pvalue:.3f}",
                 fontsize=11.5, color=INK)
    clean(ax)

    # (c) 이원배치 분산분석
    ax = fig.add_subplot(gs[0, 2])
    bars = ax.bar(range(3), Fs, width=0.55,
                  color=[ORANGE_F, GREEN_F, "#EDE7F6"],
                  edgecolor=[ORANGE, GREEN, PURPLE], lw=1.8)
    for i, (bb, f_, p_) in enumerate(zip(bars, Fs, ps)):
        lab = f"$F$ = {f_:.2f}\n" + ("$p < 10^{-4}$" if p_ < 1e-4
                                     else f"$p$ = {p_:.3f}")
        ax.text(bb.get_x() + bb.get_width() / 2, max(f_ + 2.5, 8.5), lab,
                ha="center", fontsize=10, color=INK)
    for i, d in enumerate(dfn):
        c = stats.f.ppf(0.95, d, 114)
        ax.hlines(c, i - 0.36, i + 0.36, color=RED, lw=2)
    ax.text(2.45, 3.0, "5 % 임계값", color=RED, fontsize=9.5, ha="left",
            va="center")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["전략\n(주효과)", "국면\n(주효과)", "교호작용"],
                       fontsize=10.5)
    ax.set_xlim(-0.6, 3.5)
    ax.set_ylim(0, 92)
    ax.set_ylabel("$F$ 통계량", fontsize=10.5, color=INK)
    ax.set_title("분해하면 구조가 드러난다", fontsize=11.5, color=INK)
    clean(ax)

    save(fig, "finance_regime.png")


if __name__ == "__main__":
    fig_peeking()
    fig_regime()
