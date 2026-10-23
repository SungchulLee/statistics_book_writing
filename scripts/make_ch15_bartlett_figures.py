r"""15장 바틀렛 검정 네 쪽의 개념 그림을 생성한다.

만드는 파일:
  ch15/bartlett_test/img/bartlett_amgm.png            산술평균 ≥ 기하평균이 통계량을 만든다
  ch15/bartlett_test/img/bartlett_size_vs_kurtosis.png  첨도가 오류율을 결정한다
  ch15/bartlett_test/img/bartlett_equals_f.png        k = 2 에서는 F 검정과 같은 검정
  ch15/bartlett_test/img/bartlett_lognormal_null.png  대수정규에서 귀무분포가 통째로 밀린다

실행:  python3 scripts/make_ch15_bartlett_figures.py   (저장소 최상위에서)
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

OUT = "docs/ch15/bartlett_test/img/"


def save(fig, name):
    path = OUT + name
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def clean(ax):
    ax.tick_params(labelsize=10, colors=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)


def bare(ax):
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=10, colors=INK, length=3)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)


def bartlett_T(s2, nu, k):
    """표본분산 벡터에서 바틀렛 통계량을 직접 계산한다 (집단 크기 동일)."""
    s2 = np.asarray(s2, float)
    N_k = nu * k
    sp = s2.mean()
    num = N_k * np.log(sp) - nu * np.log(s2).sum()
    C = 1 + (1 / (3 * (k - 1))) * (k / nu - 1 / N_k)
    return num / C, num, C


# === 그림 3. 첨도가 오류율을 결정한다 ===
def fig_size_vs_kurtosis():
    rows = [("균등", -1.2, 0.0, 0.003), ("정규", 0.0, 0.0, 0.049),
            ("$t_{10}$", 1.0, 0.0, 0.109), ("$\\chi^2_4$", 3.0, 1.41, 0.232),
            ("오염 정규", 5.3, 0.0, 0.335), ("$t_5$", 6.0, 0.0, 0.217),
            ("지수", 6.0, 2.0, 0.383)]

    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    for name, g2, skew, size in rows:
        c = GREEN if size < 0.06 else (ORANGE if size < 0.25 else RED)
        mk = "D" if skew > 0.5 else "o"
        ax.plot([g2], [size], marker=mk, ms=11, color=c, zorder=5)
        dy = 0.022 if name != "$t_5$" else -0.032
        ax.text(g2, size + dy, name, fontsize=10.5, color=INK, ha="center")
    ax.axhline(0.05, color=INK, lw=1.3, ls=(0, (5, 3)))
    ax.text(-1.15, 0.062, "명목 0.05", fontsize=10.5, color=INK)
    ax.annotate("", xy=(6.0, 0.383), xytext=(6.0, 0.217),
                arrowprops=dict(arrowstyle="<->", color=PURPLE, lw=1.6))
    ax.text(5.75, 0.30, "같은 첨도 6,\n치우침만 다르다", fontsize=10.5,
            color=PURPLE, ha="right", va="center")
    ax.plot([], [], marker="o", ms=9, color=MUTED, lw=0, label="대칭분포")
    ax.plot([], [], marker="D", ms=9, color=MUTED, lw=0, label="치우친 분포")
    ax.set_xlim(-2.0, 7.2)
    ax.set_ylim(-0.02, 0.44)
    ax.set_xlabel("모집단의 초과첨도 $\\gamma_2$", fontsize=11, color=INK)
    ax.set_ylabel("실제 제1종 오류율", fontsize=11, color=INK)
    clean(ax)
    ax.legend(fontsize=10.5, frameon=False, loc="upper left")
    ax.set_title("바틀렛 검정: 집단 3개, 각 $n = 20$, 명목 $\\alpha = 0.05$",
                 fontsize=12.5, color=INK, pad=10)
    save(fig, "bartlett_size_vs_kurtosis.png")


# === 그림 4. k = 2 에서는 F 검정과 같은 검정 ===
def fig_equals_f():
    obs = [(1.00, 0.0000, 1.0000, 1.000), (1.05, 0.2344, 0.9070, 0.628),
           (1.10, 0.8934, 0.8264, 0.345), (1.15, 1.9179, 0.7561, 0.166),
           (1.20, 3.2564, 0.6944, 0.071)]
    crit = stats.chi2.ppf(0.95, 1)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11.6, 4.5), gridspec_kw={"width_ratios": [1.2, 1]})

    x = np.linspace(0.02, 9, 800)
    y = stats.chi2.pdf(x, 1)
    ax1.fill_between(x[x <= crit], y[x <= crit], color=BLUE_F, alpha=0.95)
    ax1.fill_between(x, y, where=x >= crit, color=RED, alpha=0.45)
    ax1.plot(x, y, color=INK, lw=1.8)
    ax1.vlines(crit, 0, 0.42, color=RED, lw=1.4, ls=(0, (5, 3)))
    ax1.text(crit + 0.12, 0.42, f"임계값 {crit:.3f}", fontsize=10.5,
             color=RED)
    for i, (sig, T, F, p) in enumerate(obs):
        c = RED if p < 0.10 else MUTED
        yy = 0.82 - 0.075 * i
        ax1.plot([T], [yy], marker="o", ms=7, color=c, zorder=5)
        ax1.plot([T, T], [0, yy], color=c, lw=1.0, alpha=0.4)
        ax1.text(T + 0.16, yy,
                 f"$\\sigma_y$ = {sig:.2f}   $T$ = {T:.4f}   $p$ = {p:.3f}",
                 fontsize=10, color=c, va="center")
    ax1.set_xlim(0, 9)
    ax1.set_ylim(0, 0.95)
    ax1.set_xlabel("바틀렛 통계량 $T$   (귀무분포 $\\chi^2_1$)", fontsize=11,
                   color=INK)
    bare(ax1)
    ax1.set_title("$k = 2$, $n_1 = n_2 = 100$ — 본문 보기의 다섯 값",
                  fontsize=11.5, color=INK, pad=9)

    n = 100
    nu, k = n - 1, 2
    Fg = np.linspace(0.4, 2.6, 400)
    Ts = []
    for f in Fg:
        Ts.append(bartlett_T(np.array([f, 1.0]), nu, k)[0])
    ax2.plot(Fg, Ts, color=PURPLE, lw=2.4)
    ax2.axhline(crit, color=RED, lw=1.3, ls=(0, (5, 3)))
    lo, hi = stats.f.ppf(0.025, nu, nu), stats.f.ppf(0.975, nu, nu)
    for v in (lo, hi):
        ax2.vlines(v, 0, 12, color=RED, lw=1.2, ls=(0, (5, 3)))
    ax2.text(lo, 12.4, f"{lo:.3f}", fontsize=10, color=RED, ha="center")
    ax2.text(hi, 12.4, f"{hi:.3f}", fontsize=10, color=RED, ha="center")
    ax2.text(1.05, 6.6, f"가로선: $\\chi^2_{{0.95,1}}$ = {crit:.3f}\n"
                        f"세로선: $F$ 검정의 2.5% 임계값\n"
                        f"세 선이 한 점에서 만난다",
             fontsize=10, color=RED, ha="center", va="center")
    for sig, T, F, p in obs:
        ax2.plot([F], [T], marker="o", ms=7, color=BLUE, zorder=5)
    ax2.set_xlim(0.4, 2.6)
    ax2.set_ylim(0, 14)
    ax2.set_xlabel("$F = s_x^2/s_y^2$", fontsize=11, color=INK)
    ax2.set_ylabel("바틀렛 통계량 $T$", fontsize=11, color=INK)
    clean(ax2)
    ax2.set_title("$T$ 는 $|\\ln F|$ 의 증가함수 — 두 기각역이 정확히 겹친다",
                  fontsize=11.5, color=INK, pad=9)

    fig.tight_layout()
    save(fig, "bartlett_equals_f.png")
    print(f"  chi2 임계 {crit:.4f}, F 임계 {lo:.4f} / {hi:.4f}")
    for v in (lo, hi):
        print(f"  F={v:.4f} 에서 T = {bartlett_T(np.array([v, 1.0]), nu, k)[0]:.4f}")


# === 그림 5. 대수정규에서 귀무분포가 통째로 밀린다 ===
def fig_lognormal_null():
    rng = np.random.default_rng(7)
    k, n, R = 3, 20, 20000
    nu, N = n - 1, k * n
    C = 1 + (1 / (3 * (k - 1))) * (k / nu - 1 / (N - k))
    crit = stats.chi2.ppf(0.95, k - 1)

    def stat(x):
        s2 = x.var(axis=2, ddof=1)
        num = (N - k) * np.log(s2.mean(axis=1)) - nu * np.log(s2).sum(axis=1)
        return num / C

    T_norm = stat(rng.normal(0, 1, size=(R, k, n)))
    T_log = stat(rng.lognormal(0, 1, size=(R, k, n)))
    size_n = float((T_norm > crit).mean())
    size_l = float((T_log > crit).mean())

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11.6, 4.5), gridspec_kw={"width_ratios": [1.35, 1]})

    grid = np.linspace(0.02, 40, 600)
    ax1.fill_between(grid, stats.chi2.pdf(grid, k - 1), color=BLUE_F,
                     alpha=0.9)
    ax1.plot(grid, stats.chi2.pdf(grid, k - 1), color=INK, lw=1.8,
             label="검정이 믿는 기준분포 $\\chi^2_2$")
    for s, color, lab in [(T_norm, GREEN, f"정규 자료의 실제 분포"),
                          (T_log, RED, f"대수정규 자료의 실제 분포")]:
        kde = stats.gaussian_kde(s[s < 90], bw_method=0.20)
        ax1.plot(grid, kde(grid), color=color, lw=2.2, label=lab)
    ax1.vlines(crit, 0, 0.30, color=INK, lw=1.4, ls=(0, (5, 3)))
    ax1.text(crit + 0.7, 0.305, f"임계값 {crit:.3f}", fontsize=10.5,
             color=INK)
    ax1.set_xlim(0, 40)
    ax1.set_ylim(0, 0.52)
    ax1.set_xlabel("바틀렛 통계량 $T$", fontsize=11, color=INK)
    ax1.set_ylabel("밀도", fontsize=11, color=INK)
    clean(ax1)
    ax1.legend(fontsize=10, frameon=False, loc="upper right")
    ax1.text(13.5, 0.20,
             f"임계값 오른쪽 넓이\n정규 {size_n:.3f}   대수정규 {size_l:.3f}",
             fontsize=10.5, color=INK, va="top")
    ax1.set_title("$H_0$ 이 참인데도 — 집단 3개, 각 $n = 20$", fontsize=11.5,
                  color=INK, pad=9)

    names = ["바틀렛", "레빈 (평균)", "플리그너–킬린"]
    vals = [0.6748, 0.2470, 0.1028]
    cols = [RED, ORANGE, PURPLE]
    ys = np.arange(3)
    ax2.barh(ys, vals, 0.5, color=cols, alpha=0.85)
    for y, v in zip(ys, vals):
        ax2.text(v + 0.012, y, f"{v:.4f}", va="center", fontsize=10.5,
                 color=INK)
    ax2.axvline(0.05, color=INK, lw=1.4, ls=(0, (5, 3)))
    ax2.text(0.05, -0.45, "명목 0.05", fontsize=10, color=INK, ha="center")
    ax2.set_yticks(ys)
    ax2.set_yticklabels(names, fontsize=10.5, color=INK)
    ax2.invert_yaxis()
    ax2.set_ylim(2.8, -0.65)
    ax2.set_xlim(0, 0.85)
    ax2.set_xlabel("거짓 양성률", fontsize=11, color=INK)
    clean(ax2)
    ax2.spines["left"].set_visible(False)
    ax2.set_title("$\\text{Lognormal}(0,1)$ 자료에서", fontsize=11.5,
                  color=INK, pad=9)

    fig.tight_layout()
    save(fig, "bartlett_lognormal_null.png")
    print(f"  정규 크기 {size_n:.4f}, 대수정규 크기 {size_l:.4f}, "
          f"대수정규 중앙값 T = {np.median(T_log):.3f}")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_amgm()
    fig_size_vs_kurtosis()
    fig_equals_f()
    fig_lognormal_null()
