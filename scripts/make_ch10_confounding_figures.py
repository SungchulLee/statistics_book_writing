r"""12장 교란 네 쪽의 개념 그림을 생성한다.

만드는 파일:
  ch12/confounding/img/confounding_directions.png  교란은 부풀리거나 가리거나 뒤집는다
  ch12/confounding/img/lurking_firefighters.png    보이지 않는 공통원인이 부호를 뒤집는다
  ch12/confounding/img/trend_spurious.png          추세만으로 r = 0.97 이 나온다
  ch12/confounding/img/ovb_and_simpson.png         누락변수 편향과 심슨의 역설

실행:  python3 scripts/make_ch12_confounding_figures.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
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

OUT = "docs/ch12/confounding/img/"
os.makedirs(OUT, exist_ok=True)

STRAT = ["#BBDEFB", "#64B5F6", "#1E88E5", "#0D47A1"]


def save(fig, name):
    path = OUT + name
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def clean(ax):
    ax.tick_params(labelsize=9, colors=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)


def strata(z, k=4):
    qs = np.quantile(z, np.linspace(0, 1, k + 1))
    lab = np.zeros(len(z), dtype=int)
    for i in range(k):
        lo, hi = qs[i], qs[i + 1]
        m = (z >= lo) & (z <= hi) if i == k - 1 else (z >= lo) & (z < hi)
        lab[m] = i
    return lab


def headroom(ax, frac=0.30):
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + frac * (hi - lo))


# === 그림 1. 교란의 세 방향 ===
def fig_confounding_directions():
    rng = np.random.default_rng(11)
    n = 700
    BETA = 1.0
    cases = [("양의 교란", 2.0), ("음의 교란", -1.5), ("부호까지 뒤집는 교란", -4.0)]

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.3))
    for ax, (name, a) in zip(axes, cases):
        Z = rng.normal(0, 1, n)
        X = Z + rng.normal(0, 1, n)
        Y = BETA * X + a * Z + rng.normal(0, 0.7, n)

        crude = np.polyfit(X, Y, 1)[0]
        M = np.column_stack([np.ones(n), X, Z])
        adj = np.linalg.lstsq(M, Y, rcond=None)[0][1]

        lab = strata(Z)
        clean(ax)
        for i in range(4):
            m = lab == i
            ax.scatter(X[m], Y[m], s=12, color=STRAT[i], alpha=0.8,
                       edgecolor="none")
            s, b = np.polyfit(X[m], Y[m], 1)
            t = np.linspace(np.quantile(X[m], 0.03), np.quantile(X[m], 0.97), 8)
            ax.plot(t, b + s * t, color=STRAT[i], lw=1.7)
        s, b = np.polyfit(X, Y, 1)
        t = np.linspace(X.min(), X.max(), 10)
        ax.plot(t, b + s * t, color=RED, lw=2.3, ls="--")

        headroom(ax, 0.34)
        ax.text(0.03, 0.97,
                f"조야한 기울기  {crude:+.2f}\n"
                f"Z 를 통제하면  {adj:+.2f}\n"
                f"참 효과      {BETA:+.2f}",
                transform=ax.transAxes, fontsize=10.2, color=INK, va="top",
                linespacing=1.5)
        ax.set_xlabel("X", fontsize=10.5, color=INK)
        ax.set_ylabel("Y", fontsize=10.5, color=INK)
        ax.set_title(name, fontsize=12.5, color=INK, pad=8)
        print(f"  {name}: 조야={crude:+.3f}, 조정={adj:+.3f}")

    fig.suptitle("참 효과는 셋 다 " + r"$+1$" +
                 " 인데 교란변수를 빼놓은 직선(붉은 파선)만 달라진다",
                 fontsize=12.5, color=INK, y=1.02)
    save(fig, "confounding_directions.png")


# === 그림 2. 소방관과 화재 규모 ===
def fig_lurking_firefighters():
    rng = np.random.default_rng(404)
    n = 600
    # 화재 규모는 네 등급의 잠복변수다. 등급 안에서는 소방관이 많을수록
    # 피해가 줄어든다(기울기 -1.5).
    S = rng.choice([3.0, 5.0, 7.0, 9.0], size=n)
    X = 3 * S + rng.normal(0, 1.6, n)              # 출동 소방관 수
    Y = 20 * S - 1.5 * X + rng.normal(0, 4, n)     # 재산 피해액

    r_obs, _ = stats.pearsonr(X, Y)
    crude = np.polyfit(X, Y, 1)[0]
    M = np.column_stack([np.ones(n), X, S])
    adj = np.linalg.lstsq(M, Y, rcond=None)[0][1]
    eX = X - np.polyval(np.polyfit(S, X, 1), S)
    eY = Y - np.polyval(np.polyfit(S, Y, 1), S)
    r_res, _ = stats.pearsonr(eX, eY)

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.3))

    ax = axes[0]
    clean(ax)
    ax.scatter(X, Y, s=14, color=MUTED, alpha=0.75, edgecolor="none")
    s, b = np.polyfit(X, Y, 1)
    t = np.linspace(X.min(), X.max(), 10)
    ax.plot(t, b + s * t, color=RED, lw=2.3)
    headroom(ax, 0.26)
    ax.text(0.03, 0.97, f"r = {r_obs:+.2f}\n기울기 {crude:+.2f}",
            transform=ax.transAxes, fontsize=10.5, color=RED, va="top")
    ax.set_xlabel("출동 소방관 수", fontsize=10.5, color=INK)
    ax.set_ylabel("재산 피해액", fontsize=10.5, color=INK, labelpad=8)
    ax.set_title("자료에 보이는 것", fontsize=12.5, color=INK, pad=8)

    ax = axes[1]
    clean(ax)
    levels = [3.0, 5.0, 7.0, 9.0]
    names = ["소형", "중형", "대형", "특대"]
    for i, lv in enumerate(levels):
        m = S == lv
        ax.scatter(X[m], Y[m], s=14, color=STRAT[i], alpha=0.85,
                   edgecolor="none", label=names[i])
        s, b = np.polyfit(X[m], Y[m], 1)
        t = np.linspace(np.quantile(X[m], 0.03), np.quantile(X[m], 0.97), 8)
        ax.plot(t, b + s * t, color=STRAT[i], lw=1.9)
    t = np.linspace(X.min(), X.max(), 10)
    s, b = np.polyfit(X, Y, 1)
    ax.plot(t, b + s * t, color=RED, lw=2.0, ls="--")
    headroom(ax, 0.26)
    ax.text(0.03, 0.97, "규모가 같은 화재끼리는 내려간다", transform=ax.transAxes,
            fontsize=10.5, color=INK, va="top")
    ax.legend(fontsize=9.5, frameon=False, loc="lower right", ncol=2,
              title="화재 규모", title_fontsize=9.5)
    ax.set_xlabel("출동 소방관 수", fontsize=10.5, color=INK)
    ax.set_title("숨은 변수를 알고 나면", fontsize=12.5, color=INK, pad=8)

    ax = axes[2]
    clean(ax)
    ax.scatter(eX, eY, s=14, color=GREEN, alpha=0.7, edgecolor="none")
    s, b = np.polyfit(eX, eY, 1)
    t = np.linspace(eX.min(), eX.max(), 10)
    ax.plot(t, b + s * t, color=GREEN, lw=2.3)
    headroom(ax, 0.26)
    ax.text(0.03, 0.97, f"r = {r_res:+.2f}\n기울기 {adj:+.2f}",
            transform=ax.transAxes, fontsize=10.5, color=GREEN, va="top")
    ax.set_xlabel("소방관 수의 잔차", fontsize=10.5, color=INK)
    ax.set_ylabel("피해액의 잔차", fontsize=10.5, color=INK)
    ax.set_title("화재 규모를 통제한 뒤", fontsize=12.5, color=INK, pad=8)

    save(fig, "lurking_firefighters.png")
    print(f"  r_obs={r_obs:.3f} 조야기울기={crude:.3f} 조정기울기={adj:.3f} "
          f"r_res={r_res:.3f}")


# === 그림 3. 공통 추세가 만드는 허위상관 ===
def fig_trend_spurious():
    T = 120

    def walk(gen, drift=0.35):
        return np.cumsum(gen.normal(drift, 1.0, T))

    # 보기 좋은 한 쌍을 고른다.
    pick = None
    for seed in range(4000):
        g = np.random.default_rng(seed)
        a, b = walk(g), walk(g)
        r = stats.pearsonr(a, b)[0]
        rd = stats.pearsonr(np.diff(a), np.diff(b))[0]
        if r > 0.965 and abs(rd) < 0.05:
            pick = (seed, a, b, r)
            break
    seed, A, B, r_lvl = pick
    dA, dB = np.diff(A), np.diff(B)
    r_dif = stats.pearsonr(dA, dB)[0]

    reps = 4000
    g = np.random.default_rng(1234)
    r_levels = np.empty(reps)
    r_diffs = np.empty(reps)
    for i in range(reps):
        a, b = walk(g), walk(g)
        r_levels[i] = stats.pearsonr(a, b)[0]
        r_diffs[i] = stats.pearsonr(np.diff(a), np.diff(b))[0]
    big_lvl = float((np.abs(r_levels) > 0.7).mean())
    big_dif = float((np.abs(r_diffs) > 0.7).mean())

    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.2))

    ax = axes[0]
    clean(ax)
    ax.plot(np.arange(T), A, color=BLUE, lw=2.0, label="계열 A")
    ax.plot(np.arange(T), B, color=ORANGE, lw=2.0, label="계열 B")
    ax.set_xlabel("시간", fontsize=10.5, color=INK)
    ax.set_ylabel("수준", fontsize=10.5, color=INK)
    ax.legend(fontsize=10, frameon=False, loc="upper left")
    ax.set_title("서로 완전히 독립인 두 계열", fontsize=12.5, color=INK, pad=8)

    ax = axes[1]
    clean(ax)
    ax.scatter(A, B, s=16, color=MUTED, alpha=0.8, edgecolor="none")
    s, b = np.polyfit(A, B, 1)
    t = np.linspace(A.min(), A.max(), 10)
    ax.plot(t, b + s * t, color=RED, lw=2.2)
    headroom(ax, 0.22)
    ax.text(0.03, 0.97, f"수준끼리  r = {r_lvl:.3f}\n차분끼리  r = {r_dif:+.3f}",
            transform=ax.transAxes, fontsize=10.5, color=INK, va="top",
            linespacing=1.5)
    ax.set_xlabel("계열 A", fontsize=10.5, color=INK)
    ax.set_ylabel("계열 B", fontsize=10.5, color=INK)
    ax.set_title("시간을 지우고 흩뿌리면", fontsize=12.5, color=INK, pad=8)

    ax = axes[2]
    clean(ax)
    bins = np.linspace(-1, 1, 61)
    ax.hist(r_levels, bins=bins, color=RED, alpha=0.6, edgecolor="none",
            label="수준 자료")
    ax.hist(r_diffs, bins=bins, color=BLUE, alpha=0.75, edgecolor="none",
            label="차분 자료")
    ax.set_xlabel("독립인 두 계열의 상관계수", fontsize=10.5, color=INK)
    ax.set_ylabel("쌍의 수", fontsize=10.5, color=INK)
    ax.legend(fontsize=10, frameon=False, loc="upper left")
    ax.set_title(f"{reps:,}쌍을 새로 만들어 보면", fontsize=12.5, color=INK,
                 pad=8)
    ax.text(0.97, 0.62,
            f"|r| > 0.7 인 비율\n수준 {100 * big_lvl:.0f}%  ·  차분 {100 * big_dif:.1f}%",
            transform=ax.transAxes, fontsize=10.2, color=INK, ha="right",
            va="top", linespacing=1.5)

    save(fig, "trend_spurious.png")
    print(f"  seed={seed} r_level={r_lvl:.4f} r_diff={r_dif:+.4f}")
    print(f"  |r|>0.7 수준 {big_lvl:.4f}, 차분 {big_dif:.4f}")


# === 그림 4. 누락변수 편향과 심슨의 역설 ===
def fig_ovb_and_simpson():
    # 본문과 똑같은 난수 순서를 재현한다.
    np.random.seed(42)
    n1 = 500
    tc = np.random.multivariate_normal([0, 0], [[1, 0.8], [0.8, 1]], n1)
    t, c = tc[:, 0], tc[:, 1]
    y = c + np.random.normal(0, 1, n1)
    short = stats.linregress(t, y).slope
    M = np.column_stack([np.ones(n1), t, c])
    long_t = np.linalg.lstsq(M, y, rcond=None)[0][1]

    n2 = 1000
    sev = np.random.binomial(1, 0.5, n2)
    p_t = np.where(sev == 1, 0.7, 0.3)
    tr = np.random.binomial(1, p_t)
    Y = 50 - 20 * sev + 5 * tr + np.random.normal(0, 5, n2)

    naive = Y[tr == 1].mean() - Y[tr == 0].mean()
    mild = (Y[(tr == 1) & (sev == 0)].mean() - Y[(tr == 0) & (sev == 0)].mean())
    severe = (Y[(tr == 1) & (sev == 1)].mean()
              - Y[(tr == 0) & (sev == 1)].mean())
    ps = sev.mean()
    adj = (1 - ps) * mild + ps * severe

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))

    # --- 왼쪽: 짧은 회귀와 긴 회귀 ---
    ax = axes[0]
    clean(ax)
    lab = strata(c)
    for i in range(4):
        m = lab == i
        ax.scatter(t[m], y[m], s=13, color=STRAT[i], alpha=0.85,
                   edgecolor="none")
        s, b = np.polyfit(t[m], y[m], 1)
        tt = np.linspace(np.quantile(t[m], 0.04), np.quantile(t[m], 0.96), 8)
        ax.plot(tt, b + s * tt, color=STRAT[i], lw=1.8)
    tt = np.linspace(t.min(), t.max(), 10)
    s, b = np.polyfit(t, y, 1)
    ax.plot(tt, b + s * tt, color=RED, lw=2.4, ls="--")
    headroom(ax, 0.32)
    ax.text(0.03, 0.97,
            f"짧은 회귀  {short:+.3f}\n"
            f"긴 회귀    {long_t:+.3f}\n"
            f"참 효과    {0.0:+.3f}",
            transform=ax.transAxes, fontsize=10.5, color=INK, va="top",
            linespacing=1.5)
    ax.text(0.97, 0.06, "색 = 교란변수 C 의 층", transform=ax.transAxes,
            fontsize=10, color=INK, ha="right")
    ax.set_xlabel("처치 T", fontsize=10.5, color=INK)
    ax.set_ylabel("결과 Y", fontsize=10.5, color=INK)
    ax.set_title("1부: T 는 Y 에 아무 효과가 없다", fontsize=12.5, color=INK,
                 pad=8)

    # --- 오른쪽: 심슨의 역설 ---
    ax = axes[1]
    clean(ax)
    groups = ["경증", "중증", "합친 전체"]
    ctrl = [Y[(tr == 0) & (sev == 0)].mean(), Y[(tr == 0) & (sev == 1)].mean(),
            Y[tr == 0].mean()]
    trt = [Y[(tr == 1) & (sev == 0)].mean(), Y[(tr == 1) & (sev == 1)].mean(),
           Y[tr == 1].mean()]
    diffs = [mild, severe, naive]
    xs = np.arange(3)
    ax.scatter(xs - 0.09, ctrl, s=130, color=MUTED, zorder=3, label="비처치군")
    ax.scatter(xs + 0.09, trt, s=130, color=ORANGE, zorder=3, marker="D",
               label="처치군")
    for i in range(3):
        up = diffs[i] > 0
        ax.annotate("", xy=(xs[i] + 0.09, trt[i]), xytext=(xs[i] - 0.09, ctrl[i]),
                    arrowprops=dict(arrowstyle="-|>", lw=2.0,
                                    color=GREEN if up else RED,
                                    shrinkA=7, shrinkB=7))
        ax.text(xs[i] + 0.16, 0.5 * (ctrl[i] + trt[i]), f"{diffs[i]:+.2f}",
                fontsize=11.5, color=GREEN if up else RED, va="center")
    ax.set_xticks(xs)
    ax.set_xticklabels(groups, fontsize=11, color=INK)
    ax.set_xlim(-0.45, 2.6)
    ax.set_ylabel("평균 결과", fontsize=10.5, color=INK)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo - 0.08 * (hi - lo), hi + 0.22 * (hi - lo))
    ax.legend(fontsize=10, frameon=False, loc="lower left")
    ax.set_title("2부: 두 층 모두 오르는데 합치면 내려간다", fontsize=12.5,
                 color=INK, pad=8)

    save(fig, "ovb_and_simpson.png")
    print(f"  short={short:.4f} long={long_t:.4f}")
    print(f"  naive={naive:.2f} mild={mild:.2f} severe={severe:.2f} "
          f"adj={adj:.2f}")
    print(f"  ctrl={np.round(ctrl, 2)} trt={np.round(trt, 2)}")
    print(f"  처치군 중증비율={sev[tr == 1].mean():.3f}, "
          f"비처치군 중증비율={sev[tr == 0].mean():.3f}")


if __name__ == "__main__":
    fig_confounding_directions()
    fig_lurking_firefighters()
    fig_trend_spurious()
    fig_ovb_and_simpson()
