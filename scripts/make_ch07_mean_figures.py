r"""7.1 표본평균 네 쪽의 그림을 생성한다.

네 쪽 모두 그림이 없던 곳이다. 쪽마다 그 쪽의 핵심 주장 하나를 그렸다.

만드는 파일:
  ch07/mean/img/unbiased_all_n_rate.png   모든 n 에서 편향 0, 그리고 1/√n 속도
  ch07/mean/img/mean_vs_median_efficiency.png  효율 — 같은 자료, 좁은 쪽과 넓은 쪽
  ch07/mean/img/robust_means_breakdown.png     오염 한 점이 평균을 끌고 간다
  ch07/mean/img/sample_mean_linear_blue.png    선형·불편 중에서 가장 좁다

실행:  python3 scripts/make_ch07_mean_figures.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

INK = "#37474F"
BLUE, BLUE_F = "#1565C0", "#DCEBFB"
ORANGE, ORANGE_F = "#E65100", "#FFE0B2"
GREEN, GREEN_F = "#33691E", "#C5E1A5"
PURPLE = "#6A1B9A"
MUTED = "#90A4AE"
RED = "#D32F2F"

OUT = "docs/ch07/mean/img/"


def save(fig, path):
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def bare_axis(ax):
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=10, colors=INK, length=3)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)


def kde(samples, grid, bw):
    """가우스 커널 밀도 — scipy 없이 쓰기 위해 직접 짠다."""
    d = (grid[:, None] - samples[None, :]) / bw
    return np.exp(-d ** 2 / 2).sum(axis=1) / (len(samples) * bw * np.sqrt(2 * np.pi))


# ==================================================================
# 1. 모든 n 에서 편향이 0 이다 — 그리고 좁아지는 속도는 1/√n
# ==================================================================
def unbiased_all_n_rate():
    rng = np.random.default_rng(20250929)
    mu = 1.0                      # Exp(1) 의 평균
    reps = 200_000
    ns = [2, 10, 50]
    colors = [ORANGE, PURPLE, BLUE]

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 4.9),
                             gridspec_kw={"width_ratios": [1.15, 1]})

    # --- (a) 치우친 모집단에서도 X̄ 의 중심은 언제나 μ ---
    ax = axes[0]
    grid = np.linspace(0, 3.2, 700)
    means = {}
    for n, c in zip(ns, colors):
        xbar = rng.exponential(1.0, size=(reps, n)).mean(axis=1)
        means[n] = xbar.mean()
        y = kde(xbar[:20000], grid, bw=0.35 / np.sqrt(n))
        ax.plot(grid, y, color=c, linewidth=2.1, zorder=5,
                label=f"$n = {n}$   평균 $= {means[n]:.4f}$")
        ax.fill_between(grid, y, color=c, alpha=0.11, zorder=3)
        print(f"  n={n:3d}  E[Xbar] 추정 = {means[n]:.4f}   "
              f"SD = {xbar.std():.4f}  (이론 {1/np.sqrt(n):.4f})")

    top = 3.0
    # 세로선을 곡선 꼭대기보다 조금만 높게 세우고 라벨을 그 위에 둔다.
    # 왼쪽 위에 두면 범례와 겹친다.
    ax.plot([mu, mu], [0, top * 0.99], color=RED, linewidth=1.7, zorder=6)
    ax.text(mu - 0.06, top * 1.02, r"참값 $\mu = 1$", fontsize=11.5,
            color=RED, ha="right", va="bottom")
    ax.text(1.95, top * 0.52,
            "모집단은 오른쪽으로\n심하게 치우쳤지만\n세 곡선의 무게중심은\n"
            "모두 붉은 선 위에 있다",
            fontsize=10.8, color=INK, ha="left", va="top", linespacing=1.55)

    ax.set_xlim(0, 3.2)
    ax.set_ylim(0, top * 1.25)
    ax.set_xticks([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    bare_axis(ax)
    ax.legend(fontsize=10.5, frameon=False, loc="upper right",
              handlelength=1.4)
    ax.set_title(r"지수분포 $\mathrm{Exp}(1)$ 에서 뽑은 $\bar X$ 의 표본분포",
                 fontsize=13, pad=10)

    # --- (b) 좁아지는 속도 — 정밀도 2배에 자료 4배 ---
    ax = axes[1]
    # n 을 10 부터 그린다. 1 부터 그리면 세로축이 1.0 까지 늘어나 곡선 아래의
    # 빈 자리가 눌리고, 아래쪽 화살표 라벨이 곡선에 닿는다.
    n = np.arange(10, 431)
    se = 1.0 / np.sqrt(n)
    ax.plot(n, se, color=BLUE, linewidth=2.4, zorder=5)
    ax.fill_between(n, se, color=BLUE, alpha=0.10, zorder=3)

    label_xy = {25: (42, 0.215), 100: (116, 0.116), 400: (292, 0.063)}
    for nn, c in [(25, GREEN), (100, ORANGE), (400, PURPLE)]:
        s = 1.0 / np.sqrt(nn)
        ax.plot([nn, nn], [0, s], color=c, linewidth=1.3, linestyle="--",
                zorder=4)
        ax.plot([0, nn], [s, s], color=c, linewidth=1.3, linestyle="--",
                zorder=4)
        ax.plot([nn], [s], "o", color=c, markersize=7.5, zorder=7)
        tx, ty = label_xy[nn]
        ax.text(tx, ty, f"$n = {nn}$,  $\\mathrm{{SE}} = {s:.2f}$",
                fontsize=11, color=c, ha="left", va="bottom")

    for x0, x1, tag in [(25, 100, "자료 4배"), (100, 400, "또 4배")]:
        ax.add_patch(FancyArrowPatch((x0, 0.017), (x1, 0.017),
                                     arrowstyle="-|>", mutation_scale=13,
                                     color=INK, linewidth=1.4, zorder=8))
        ax.text((x0 + x1) / 2, 0.021, tag, fontsize=10.5, color=INK,
                ha="center", va="bottom")

    ax.text(172, 0.345, r"$\mathrm{SE} = \sigma / \sqrt{n}$",
            fontsize=13, color=BLUE, ha="left", va="top")
    ax.text(172, 0.295, "정밀도를 두 배로 하려면\n자료를 네 배로 늘려야 한다",
            fontsize=10.8, color=INK, ha="left", va="top", linespacing=1.55)

    ax.set_xlim(0, 435)
    ax.set_ylim(0, 0.37)
    ax.set_xlabel("표본크기 $n$", fontsize=12, color=INK)
    ax.set_ylabel(r"$\bar X$ 의 표준오차  ($\sigma = 1$)", fontsize=11.5,
                  color=INK)
    ax.tick_params(labelsize=10, colors=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.set_title("중심은 $n$ 과 무관하고, 폭만 $n$ 을 따른다", fontsize=13,
                 pad=10)

    fig.suptitle("편향은 모든 $n$ 에서 정확히 0, 일치성은 $1/\\sqrt{n}$ 의 속도로",
                 fontsize=14.5, y=1.03)
    fig.tight_layout()
    save(fig, OUT + "unbiased_all_n_rate.png")


# ==================================================================
# 2. 효율의 순위는 분포가 정한다 — 평균 대 중앙값
# ==================================================================
def mean_vs_median_efficiency():
    rng = np.random.default_rng(7)
    n, reps = 25, 60_000
    grid = np.linspace(-1.2, 1.2, 700)

    # 세 모집단 모두 중심이 0 이다. 정규와 라플라스는 분산을 1 로 맞춘다.
    pops = [
        ("정규분포  $N(0, 1)$",
         lambda size: rng.standard_normal(size)),
        ("라플라스분포  (분산 $1$)",
         lambda size: rng.laplace(0.0, 1 / np.sqrt(2), size)),
        ("코시분포",
         lambda size: rng.standard_cauchy(size)),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.8), sharey=True)
    for ax, (name, draw) in zip(axes, pops):
        x = draw((reps, n))
        xbar = x.mean(axis=1)
        med = np.median(x, axis=1)

        for vals, color, tag in [(xbar, BLUE, "표본평균 $\\bar X$"),
                                 (med, ORANGE, "표본중앙값")]:
            # 코시 표본의 중앙값은 퍼짐이 커서 같은 띠폭이면 울퉁불퉁하다.
            y = kde(vals[:40000], grid, bw=0.050 if "코시" in name else 0.035)
            ax.plot(grid, y, color=color, linewidth=2.2, zorder=5, label=tag)
            ax.fill_between(grid, y, color=color, alpha=0.13, zorder=3)

        out_bar = np.mean(np.abs(xbar) > 1.0)
        out_med = np.mean(np.abs(med) > 1.0)
        print(f"  {name}:  SD(Xbar)={xbar.std():.4f}  SD(med)={med.std():.4f}"
              f"   P(|Xbar|>1)={out_bar:.4f}  P(|med|>1)={out_med:.5f}")

        if "코시" in name:
            note = (f"표본평균은 절반({out_bar * 100:.0f}%)이\n"
                    "이 창 밖으로 나간다\n"
                    f"중앙값은 {out_med * 100:.1f}% 만 나간다")
        else:
            # 한글과 수식을 섞지 않는다. 전부 보통 글자로 적는다.
            are = med.var() / xbar.var()
            note = (f"표본평균의 표준편차   {xbar.std():.3f}\n"
                    f"중앙값의 표준편차     {med.std():.3f}\n"
                    f"ARE = {are:.2f}")

        ax.text(-1.13, 2.62, note, fontsize=11, color=INK, ha="left",
                va="top", linespacing=1.6)
        ax.plot([0, 0], [0, 2.75], color=MUTED, linewidth=1.1,
                linestyle=":", zorder=2)
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(0, 3.35)
        ax.set_xticks([-1, -0.5, 0, 0.5, 1])
        bare_axis(ax)
        ax.set_title(name, fontsize=12.5, pad=10)

    axes[0].legend(fontsize=11, frameon=False, loc="upper right",
                   handlelength=1.4)

    fig.text(0.5, -0.035,
             "세 그림 모두 $n = 25$, 중심은 $0$, 가로축의 눈금도 같다. "
             "좁은 곡선이 이긴 쪽이다 — 이기는 쪽이 분포를 따라 바뀐다.",
             fontsize=11.5, color=INK, ha="center", va="center")
    fig.suptitle("같은 표본크기, 같은 중심 — 그런데 누가 더 좁은지는 분포가 정한다",
                 fontsize=14.5, y=1.03)
    fig.tight_layout()
    save(fig, OUT + "mean_vs_median_efficiency.png")


# ==================================================================
# 3. 한 점이 얼마나 끌고 가는가 — 민감도 곡선과 그 대가
# ==================================================================
def robust_means_breakdown():
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.0),
                             gridspec_kw={"width_ratios": [1.3, 1]})

    # --- (a) 민감도 곡선 — 관측값 하나를 끌고 다닌다 ---
    ax = axes[0]
    # 가운데 순서통계량 사이 간격이 전형적인 표본을 고른다. 간격이 우연히
    # 크면 중앙값 곡선이 큰 계단 하나로 보여 오히려 덜 로버스트해 보인다.
    rng = np.random.default_rng(79)
    clean = rng.standard_normal(20)
    xs = np.linspace(-25, 25, 601)
    m = len(clean) + 1                     # n = 21

    def trimmed(sample, k):
        s = np.sort(sample)
        return s[k:len(s) - k].mean()

    curves = {
        "표본평균": np.array([np.append(clean, x).mean() for x in xs]),
        "10% 절사평균": np.array([trimmed(np.append(clean, x), 2) for x in xs]),
        "20% 절사평균": np.array([trimmed(np.append(clean, x), 4) for x in xs]),
        "중앙값": np.array([np.median(np.append(clean, x)) for x in xs]),
    }
    # 로버스트한 셋은 거의 같은 자리에서 평평해진다. 굵은 것부터 그리고
    # 굵기를 달리해야 셋이 다 보인다.
    order = [("중앙값", PURPLE, 3.6), ("20% 절사평균", ORANGE, 2.6),
             ("10% 절사평균", GREEN, 1.7), ("표본평균", BLUE, 2.3)]
    handles = {}
    for name, c, lw in order:
        y = curves[name]
        handles[name], = ax.plot(xs, y, color=c, linewidth=lw, zorder=5,
                                 label=name)
        print(f"  {name:10s}  x=-25 에서 {y[0]:+.3f},  x=+25 에서 {y[-1]:+.3f}")

    ax.axhline(clean.mean(), color=MUTED, linewidth=1.1, linestyle=":",
               zorder=2)
    # 이 설명은 곡선에서 충분히 떨어뜨려 둔다. 기준선 바로 밑에 두면
    # 평평해진 곡선들과 겹친다.
    ax.text(-24.5, -0.32, "점선은 오염이 없을 때의 값", fontsize=10.5,
            color=MUTED, ha="left", va="center")

    ax.annotate(f"기울기 $1/{m}$ — 멈추는 곳이 없다",
                xy=(19, curves["표본평균"][-60]), xytext=(4.0, 1.62),
                fontsize=11.5, color=BLUE, ha="center", va="center",
                arrowprops=dict(arrowstyle="->", color=BLUE, linewidth=1.3))
    ax.text(-24.5, -1.30,
            "절사평균과 중앙값은 평평해진다 —\n"
            "한 점이 아무리 멀리 가도 더는 끌려가지 않는다",
            fontsize=11, color=INK, ha="left", va="center", linespacing=1.6)

    ax.set_xlim(-26, 26)
    ax.set_ylim(-1.75, 1.95)
    ax.set_xlabel("오염된 관측값 하나가 놓인 자리", fontsize=12, color=INK)
    ax.set_ylabel("추정값", fontsize=12, color=INK)
    ax.set_xticks([-20, -10, 0, 10, 20])
    ax.tick_params(labelsize=10, colors=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.legend([handles[k] for k in
               ["표본평균", "10% 절사평균", "20% 절사평균", "중앙값"]],
              ["표본평균", "10% 절사평균", "20% 절사평균", "중앙값"],
              fontsize=10.8, frameon=False, loc="upper left", ncol=2,
              handlelength=1.5, columnspacing=1.4)
    ax.set_title(f"깨끗한 관측값 20개에 오염 하나를 더했을 때  ($n = {m}$)",
                 fontsize=13, pad=10)

    # --- (b) 그 대가와 그 이득 ---
    ax = axes[1]
    rng = np.random.default_rng(11)
    n, reps = 25, 80_000
    clean_s = rng.standard_normal((reps, n))
    u = rng.random((reps, n))
    cont_s = np.where(u < 0.05, rng.standard_normal((reps, n)) * 5,
                      rng.standard_normal((reps, n)))

    def sds(sample):
        s = np.sort(sample, axis=1)
        return [sample.mean(axis=1).std(),
                s[:, 2:n - 2].mean(axis=1).std(),
                s[:, 5:n - 5].mean(axis=1).std(),
                np.median(sample, axis=1).std()]

    names = ["표본평균", "10%\n절사평균", "20%\n절사평균", "중앙값"]
    a, b = sds(clean_s), sds(cont_s)
    print("  깨끗한 정규:", [round(v, 4) for v in a])
    print("  5% 오염    :", [round(v, 4) for v in b])

    pos = np.arange(4)
    w = 0.36
    ax.bar(pos - w / 2, a, w, color=BLUE, alpha=0.85, edgecolor="white",
           label="깨끗한 $N(0,1)$", zorder=4)
    ax.bar(pos + w / 2, b, w, color=ORANGE, alpha=0.85, edgecolor="white",
           label="5% 가 $N(0,5^2)$ 에서 온 자료", zorder=4)
    for p, v in zip(pos - w / 2, a):
        ax.text(p, v + 0.006, f"{v:.3f}", fontsize=10.2, color=BLUE,
                ha="center", va="bottom")
    for p, v in zip(pos + w / 2, b):
        ax.text(p, v + 0.006, f"{v:.3f}", fontsize=10.2, color=ORANGE,
                ha="center", va="bottom")

    ax.set_xticks(pos)
    ax.set_xticklabels(names, fontsize=10.8)
    ax.set_ylim(0, 0.40)
    ax.set_ylabel(f"추정값의 표준편차  ($n = {n}$)", fontsize=11.5, color=INK)
    ax.tick_params(labelsize=10, colors=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.legend(fontsize=10.5, frameon=False, loc="upper left")
    ax.set_title("깨끗할 때 치르는 값, 더러울 때 버는 값", fontsize=13,
                 pad=10)

    fig.tight_layout()
    save(fig, OUT + "robust_means_breakdown.png")


# ==================================================================
# 4. 선형·불편 중에서 동등가중이 가장 좁다 (가우스–마르코프)
# ==================================================================
def sample_mean_linear_blue():
    n = 10
    idx = np.arange(1, n + 1)

    schemes = [
        ("동등가중 — 표본평균", np.full(n, 1 / n), BLUE, "o", 2.6),
        ("최근 자료에 더 큰 가중치", idx / idx.sum(), GREEN, "s", 2.0),
        ("뒤쪽 절반만 사용", np.where(idx > 5, 0.2, 0.0), ORANGE, "^", 2.0),
        ("한 관측값에 절반을 몰아줌",
         np.where(idx == 1, 0.5, 0.5 / (n - 1)), PURPLE, "D", 2.0),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.9),
                             gridspec_kw={"width_ratios": [1.22, 1]})

    # --- (a) 합이 1 인 가중치 네 가지 ---
    ax = axes[0]
    for name, w, c, mk, lw in schemes:
        mult = n * np.sum(w ** 2)          # 분산이 sigma^2/n 의 몇 배인가
        ax.plot(idx, w, color=c, linewidth=lw, marker=mk, markersize=6.5,
                markeredgecolor="white", markeredgewidth=0.7, zorder=5,
                label=f"{name}   (분산 {mult:.2f} 배)")
        print(f"  {name:22s} sum(w)={w.sum():.3f}  분산배수={mult:.3f}")

    ax.set_xlim(0.4, 10.6)
    ax.set_ylim(-0.03, 0.62)
    ax.set_xticks(idx)
    ax.set_xlabel("관측값 번호", fontsize=12, color=INK)
    ax.set_ylabel("가중치 $w_i$", fontsize=12, color=INK)
    ax.tick_params(labelsize=10, colors=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.legend(fontsize=10.5, frameon=False, loc="upper center",
              handlelength=1.8)
    ax.set_title("네 가중치 모두 합이 $1$ 이므로 모두 불편이다  "
                 f"($n = {n}$)", fontsize=12.8, pad=10)

    # --- (b) n = 2 에서 본 골짜기 ---
    ax = axes[1]
    w = np.linspace(-0.15, 1.15, 500)
    v = w ** 2 + (1 - w) ** 2
    ax.plot(w, v, color=BLUE, linewidth=2.5, zorder=5)
    ax.fill_between(w, v, 1.5, color=BLUE, alpha=0.07, zorder=2)

    ax.plot([0.5], [0.5], "o", color=RED, markersize=9, zorder=7)
    ax.plot([0.5, 0.5], [0, 0.5], color=RED, linewidth=1.3, linestyle="--",
            zorder=4)
    # 라벨을 파선 왼쪽에 둔다. 가운데에 두면 파선이 글자를 세로로 가른다.
    ax.text(0.46, 0.44, "동등가중\n$w = 1/2$", fontsize=11.5, color=RED,
            ha="right", va="top", linespacing=1.5)

    for wv, c in [(0.8, ORANGE), (1.0, PURPLE)]:
        vv = wv ** 2 + (1 - wv) ** 2
        ax.plot([wv], [vv], "o", color=c, markersize=7.5, zorder=7)
        ax.text(wv + 0.035, vv, f"$w = {wv:g}$,  {vv / 0.5:.2f} 배",
                fontsize=11, color=c, ha="left", va="center")

    ax.text(0.03, 1.43,
            "$\\hat\\mu = w X_1 + (1-w) X_2$ 는\n"
            "모든 $w$ 에서 불편이다.\n"
            "분산만 $w$ 를 따라 움직인다.",
            fontsize=11, color=INK, ha="left", va="top", linespacing=1.6)

    ax.set_xlim(-0.15, 1.28)
    ax.set_ylim(0, 1.5)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("첫 관측값에 주는 가중치 $w$", fontsize=12, color=INK)
    ax.set_ylabel(r"$\mathrm{Var}(\hat\mu) \,/\, \sigma^2$", fontsize=12,
                  color=INK)
    ax.tick_params(labelsize=10, colors=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.set_title("$n = 2$ 로 줄여 본 같은 이야기", fontsize=12.8, pad=10)

    fig.suptitle("불편이라는 조건만으로는 하나로 정해지지 않는다 — "
                 "가장 좁은 것이 표본평균이다", fontsize=14.5, y=1.03)
    fig.tight_layout()
    save(fig, OUT + "sample_mean_linear_blue.png")


if __name__ == "__main__":
    unbiased_all_n_rate()
    mean_vs_median_efficiency()
    robust_means_breakdown()
    sample_mean_linear_blue()
