r"""7.2 최대가능도 세 쪽의 그림을 생성한다.

세 쪽 모두 그림이 없던 곳이다. 쪽마다 그 쪽의 핵심 주장 하나를 그렸다.

만드는 파일:
  ch07/mle/img/gaussian_loglik_surface.png   로그가능도 곡면과 꼭대기의 자리
  ch07/mle/img/mle_bias_shrinks.png          MLE 의 편향은 O(1/n) 으로 사라진다
  ch07/mle/img/sufficiency_compression.png   충분통계량은 무엇을 버리고 무엇을 남기나

실행:  python3 scripts/make_ch07_mle_figures.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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

OUT = "docs/ch07/mle/img/"


def save(fig, path):
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


# ==================================================================
# 1. 로그가능도 곡면 — 꼭대기가 (x̄, σ̂²) 에 있다
# ==================================================================
def gaussian_loglik_surface():
    # n 을 작게 잡아야 MLE 와 S^2 이 눈에 보일 만큼 벌어진다.
    # n = 12 면 비가 0.917 이라 두 점이 겹쳐 보인다.
    rng = np.random.default_rng(5)
    n = 5
    x = rng.normal(5.0, 2.0, size=n)
    xbar = x.mean()
    s2_mle = ((x - xbar) ** 2).mean()
    s2_unb = ((x - xbar) ** 2).sum() / (n - 1)
    print(f"  n={n}  xbar={xbar:.4f}  MLE={s2_mle:.4f}  S^2={s2_unb:.4f}"
          f"   비 {s2_mle / s2_unb:.4f} (= (n-1)/n = {(n-1)/n:.4f})")

    def loglik(mu, v):
        return (-n / 2 * np.log(2 * np.pi) - n / 2 * np.log(v)
                - ((x[:, None, None] - mu) ** 2).sum(axis=0) / (2 * v))

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.1),
                             gridspec_kw={"width_ratios": [1.18, 1]})

    # --- (a) 등고선 ---
    ax = axes[0]
    mus = np.linspace(xbar - 3.6, xbar + 3.6, 320)
    vs = np.linspace(0.8, 15.0, 320)
    MU, V = np.meshgrid(mus, vs)
    L = loglik(MU, V)
    peak = L.max()

    # 맨 안쪽 영역까지 색이 채워지도록 경계를 하나 더 준다.
    levels = peak - np.array([12, 8, 5, 3, 1.5, 0.6, 0.15])
    ax.contourf(MU, V, L,
                levels=np.concatenate([[peak - 40], levels, [peak + 0.01]]),
                colors=["#F7FAFC", "#EAF2F9", "#DBE9F5", "#C9DEF1",
                        "#B3D0EC", "#9AC1E7", "#7FB0E1", "#649FDB"],
                zorder=2)
    ax.contour(MU, V, L, levels=levels, colors=[BLUE], linewidths=0.9,
               alpha=0.8, zorder=3)

    ax.plot([xbar], [s2_mle], "o", color=RED, markersize=10, zorder=8)
    ax.plot([xbar, xbar], [0.8, s2_mle], color=RED, linewidth=1.2,
            linestyle="--", zorder=4)
    ax.annotate(f"MLE  $(\\bar x, \\hat\\sigma^2) = ({xbar:.2f}, "
                f"{s2_mle:.2f})$",
                xy=(xbar - 0.10, s2_mle - 0.15),
                xytext=(xbar - 3.45, 1.45),
                fontsize=11.5, color=RED, ha="left", va="center",
                bbox=dict(facecolor="white", alpha=0.82, edgecolor="none",
                          pad=2.0),
                arrowprops=dict(arrowstyle="->", color=RED, linewidth=1.3))

    ax.plot([xbar], [s2_unb], "s", color=GREEN, markersize=8.5, zorder=8)
    ax.annotate(f"불편추정값  $S^2 = {s2_unb:.2f}$\n(꼭대기가 아니다)",
                xy=(xbar + 0.07, s2_unb + 0.15), xytext=(xbar + 0.9, 10.6),
                fontsize=11.5, color=GREEN, ha="left", va="center",
                linespacing=1.5,
                bbox=dict(facecolor="white", alpha=0.82, edgecolor="none",
                          pad=2.0),
                arrowprops=dict(arrowstyle="->", color=GREEN, linewidth=1.3))

    ax.set_xlim(mus[0], mus[-1])
    ax.set_ylim(0.8, 15.0)
    ax.set_xlabel(r"$\mu$", fontsize=13, color=INK)
    ax.set_ylabel(r"$\sigma^2$", fontsize=13, color=INK)
    ax.tick_params(labelsize=10, colors=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.set_title(f"로그가능도 $\\ell(\\mu, \\sigma^2)$ 의 등고선  ($n = {n}$)",
                 fontsize=13, pad=10)

    # --- (b) mu = x̄ 에서 자른 단면 ---
    ax = axes[1]
    v = np.linspace(1.4, 8.5, 600)
    prof = loglik(np.array([[xbar]]), v[None, :].T)[:, 0]
    ax.plot(v, prof, color=BLUE, linewidth=2.5, zorder=5)

    lo = prof.min() - 0.35
    for val, c, name in [(s2_mle, RED, "MLE"), (s2_unb, GREEN, "S^2")]:
        pv = loglik(np.array([[xbar]]), np.array([[val]]))[0, 0]
        ax.plot([val, val], [lo, pv], color=c, linewidth=1.3, linestyle="--",
                zorder=4)
        ax.plot([val], [pv], "o", color=c, markersize=8.5, zorder=7)
        print(f"    {name}: sigma^2={val:.3f}  loglik={pv:.4f}")

    # 두 라벨을 각자의 파선 바깥쪽으로 밀어 서로 부딪히지 않게 한다.
    ax.text(s2_mle - 0.12, lo - 0.12, f"$\\hat\\sigma^2 = {s2_mle:.2f}$",
            fontsize=11.5, color=RED, ha="right", va="top")
    ax.text(s2_unb + 0.12, lo - 0.12, f"$S^2 = {s2_unb:.2f}$", fontsize=11.5,
            color=GREEN, ha="left", va="top")

    drop = (loglik(np.array([[xbar]]), np.array([[s2_mle]]))[0, 0]
            - loglik(np.array([[xbar]]), np.array([[s2_unb]]))[0, 0])
    ax.annotate(f"가능도가 {drop:.3f} 만큼 낮다",
                xy=(s2_unb + 0.05, loglik(np.array([[xbar]]),
                                          np.array([[s2_unb]]))[0, 0]),
                xytext=(6.0, prof.max() - 1.15), fontsize=11.5, color=GREEN,
                ha="center", va="center",
                arrowprops=dict(arrowstyle="->", color=GREEN, linewidth=1.3))
    ax.text(6.0, prof.max() + 0.12,
            "가능도를 최대로 만드는 값과\n편향이 0인 값은 서로 다르다",
            fontsize=11, color=INK, ha="center", va="center", linespacing=1.6)

    ax.set_xlim(1.4, 8.5)
    ax.set_ylim(lo - 1.1, prof.max() + 0.62)
    ax.set_xlabel(r"$\sigma^2$", fontsize=13, color=INK)
    ax.set_ylabel(r"$\ell(\bar x, \sigma^2)$", fontsize=12.5, color=INK)
    ax.tick_params(labelsize=10, colors=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.set_title(r"$\mu = \bar x$ 에서 자른 단면", fontsize=13, pad=10)

    fig.suptitle("곡면의 꼭대기가 최대가능도추정값이다", fontsize=14.5,
                 y=1.02)
    fig.tight_layout()
    save(fig, OUT + "gaussian_loglik_surface.png")


# ==================================================================
# 2. MLE 의 편향이 어디서 오는가 — x̄ 에서 재기 때문이다
# ==================================================================
def mle_bias_shrinks():
    rng = np.random.default_rng(17)
    n = 5

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.0),
                             gridspec_kw={"width_ratios": [1.05, 1.15]})

    # --- (a) g(a) = (1/n) Σ (x_i - a)^2 는 a = x̄ 에서 최소 ---
    ax = axes[0]
    x = rng.normal(0.0, 1.0, size=n)
    xbar = x.mean()
    a = np.linspace(-1.30, 0.42, 400)
    g = ((x[:, None] - a[None, :]) ** 2).mean(axis=0)
    g_at_mu = (x ** 2).mean()
    g_at_xbar = ((x - xbar) ** 2).mean()
    print(f"  (a) xbar={xbar:.4f}  g(mu)={g_at_mu:.4f}  "
          f"g(xbar)={g_at_xbar:.4f}  차이={g_at_mu - g_at_xbar:.4f} "
          f"(= (xbar-mu)^2 = {xbar ** 2:.4f})")

    ax.plot(a, g, color=BLUE, linewidth=2.5, zorder=5)

    lo_y = 0.95
    for val, c in [(0.0, RED), (xbar, GREEN)]:
        gv = ((x - val) ** 2).mean()
        ax.plot([val, val], [lo_y, gv], color=c, linewidth=1.4,
                linestyle="--", zorder=4)
        ax.plot([-1.28, val], [gv, gv], color=c, linewidth=1.0,
                linestyle=":", zorder=3)
        ax.plot([val], [gv], "o", color=c, markersize=9, zorder=7)
        ax.text(val, lo_y + 0.012,
                r"$\mu$" if c is RED else r"$\bar x$", fontsize=12.5,
                color=c, ha="center", va="bottom")

    ax.annotate(f"$\\hat\\sigma^2_{{\\mathrm{{MLE}}}} = {g_at_xbar:.2f}$",
                xy=(xbar, g_at_xbar), xytext=(xbar + 0.30, g_at_xbar - 0.11),
                fontsize=12, color=GREEN, ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color=GREEN, linewidth=1.3))
    ax.annotate(f"$\\mu$ 에서 쟀다면 ${g_at_mu:.2f}$",
                xy=(0.0, g_at_mu), xytext=(-1.26, g_at_mu + 0.30),
                fontsize=12, color=RED, ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color=RED, linewidth=1.3))

    # 라벨은 화살표 위, 포물선 안쪽의 빈 자리에 둔다. 화살표 오른쪽에 두면
    # 내려오는 곡선이 글자를 가로지른다.
    ax.add_patch(FancyArrowPatch((-1.13, g_at_xbar), (-1.13, g_at_mu),
                                 arrowstyle="<|-|>", mutation_scale=12,
                                 color=INK, linewidth=1.5, zorder=8))
    ax.text(-1.13, 1.12,
            f"덜 잰 만큼\n$(\\bar x - \\mu)^2 = {xbar ** 2:.2f}$",
            fontsize=11, color=INK, ha="left", va="top", linespacing=1.5)

    ax.set_xlim(-1.30, 0.42)
    ax.set_ylim(0.95, 1.98)
    ax.set_xlabel("편차를 재는 기준점 $a$", fontsize=12, color=INK)
    ax.set_ylabel("$(1/n)\\sum (x_i - a)^2$", fontsize=12, color=INK)
    ax.tick_params(labelsize=10, colors=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.set_title("제곱합은 $\\bar x$ 에서 최소가 된다", fontsize=13, pad=10)

    # --- (b) 그래서 표본분포의 중심이 밀린다 ---
    ax = axes[1]
    reps = 200_000
    xs = rng.normal(0.0, 1.0, size=(reps, n))
    ss = ((xs - xs.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
    mle = ss / n
    unb = ss / (n - 1)
    grid = np.linspace(0, 3.2, 700)

    def kde1(vals, bw):
        d = (grid[:, None] - vals[None, :]) / bw
        return np.exp(-d ** 2 / 2).sum(axis=1) / (len(vals) * bw
                                                  * np.sqrt(2 * np.pi))

    for vals, c, tag in [(mle, ORANGE, r"$\hat\sigma^2_{\mathrm{MLE}}$  "
                                       r"($n$ 으로 나눔)"),
                         (unb, BLUE, r"$S^2$  ($n-1$ 로 나눔)")]:
        y = kde1(vals[:40000], 0.085)
        ax.plot(grid, y, color=c, linewidth=2.3, zorder=5, label=tag)
        ax.fill_between(grid, y, color=c, alpha=0.13, zorder=3)
        print(f"  (b) {tag[:12]}: 평균={vals.mean():.4f}  "
              f"P(< 1)={np.mean(vals < 1.0):.4f}")

    top = 1.18
    ax.plot([1.0, 1.0], [0, top], color=RED, linewidth=1.7, zorder=6)
    ax.text(1.04, top, r"참값 $\sigma^2 = 1$", fontsize=11.5, color=RED,
            ha="left", va="top")

    # 두 평균이 붉은 세로선 양쪽에 바싹 붙으므로 라벨을 바깥쪽으로 민다.
    for vals, c, dy, ha, dx in [(mle, ORANGE, 0.30, "right", -0.045),
                                (unb, BLUE, 0.46, "left", 0.105)]:
        m = vals.mean()
        ax.plot([m], [dy], "v", color=c, markersize=10, zorder=8)
        ax.text(m + dx, dy, f"평균 {m:.3f}", fontsize=11, color=c,
                ha=ha, va="center",
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none",
                          pad=1.5))

    ax.text(1.55, 0.92,
            f"$n = {n}$ 에서 MLE 는 평균적으로\n"
            f"참값의 {(n - 1) / n:.0%} 밖에 되지 않는다.\n"
            f"표본의 {np.mean(mle < 1.0):.0%} 에서 참값보다 작다.",
            fontsize=11, color=INK, ha="left", va="top", linespacing=1.65)

    ax.set_xlim(0, 3.2)
    ax.set_ylim(0, 1.42)
    ax.set_yticks([])
    ax.set_xlabel(r"추정값", fontsize=12, color=INK)
    ax.tick_params(axis="x", labelsize=10, colors=INK)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.legend(fontsize=11, frameon=False, loc="upper left")
    ax.set_title(f"두 추정량의 표본분포  ($n = {n}$, 20만 회)", fontsize=13,
                 pad=10)

    fig.suptitle("한 자리를 자료에서 가져다 썼기 때문에 생기는 편향",
                 fontsize=14.5, y=1.02)
    fig.tight_layout()
    save(fig, OUT + "mle_bias_shrinks.png")


# ==================================================================
# 3. 충분통계량 — 같은 (x̄, s²) 를 가지면 가능도가 완전히 같다
# ==================================================================
def sufficiency_compression():
    rng = np.random.default_rng(404)
    n = 12
    target_mean, target_sd = 5.0, 2.0

    def force(y):
        """모양은 y 그대로 두고 (x̄, s) 만 목표값에 맞춘다."""
        z = (y - y.mean()) / y.std(ddof=1)
        return target_mean + target_sd * z

    datasets = [
        ("A  정규분포에서", force(rng.normal(0, 1, n)), BLUE),
        ("B  균등분포에서", force(rng.uniform(-1, 1, n)), GREEN),
        ("C  두 덩어리로", force(np.r_[rng.normal(-1, 0.18, n // 2),
                                   rng.normal(1, 0.18, n // 2)]), ORANGE),
        ("D  이상점 하나", force(np.r_[rng.normal(0, 0.28, n - 1), [6.0]]),
         PURPLE),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.0),
                             gridspec_kw={"width_ratios": [1.1, 1]})

    # --- (a) 생김새가 전혀 다른 네 자료 ---
    ax = axes[0]
    for k, (name, d, c) in enumerate(datasets):
        y = 3 - k
        ax.plot(d, np.full(n, y), "o", color=c, markersize=8.5, alpha=0.85,
                markeredgecolor="white", markeredgewidth=0.8, zorder=5)
        ax.plot([d.mean()], [y + 0.30], "v", color=INK, markersize=8,
                zorder=6)
        ax.text(-1.2, y, name, fontsize=11.5, color=c, ha="left",
                va="center")
        print(f"  {name}: xbar={d.mean():.6f}  s={d.std(ddof=1):.6f}  "
              f"최솟값={d.min():.2f}  최댓값={d.max():.2f}")

    ax.plot([target_mean, target_mean], [-0.35, 3.62], color=MUTED,
            linewidth=1.1, linestyle=":", zorder=3)
    ax.text(target_mean, 3.70, r"$\bar x = 5.00$", fontsize=11.5, color=INK,
            ha="center", va="bottom")
    ax.text(-1.2, -0.55,
            f"네 자료 모두 $n = {n}$, $\\bar x = 5.00$, $s^2 = 4.00$ 으로 똑같다.\n"
            "다른 것은 관측값들이 그 안에서 어떻게 흩어져 있는지뿐이다.",
            fontsize=11, color=INK, ha="left", va="top", linespacing=1.6)

    ax.set_xlim(-1.3, 11.8)
    ax.set_ylim(-1.35, 4.05)
    ax.set_yticks([])
    ax.set_xlabel("관측값", fontsize=12, color=INK)
    ax.tick_params(axis="x", labelsize=10, colors=INK)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.set_title("생김새가 전혀 다른 네 자료", fontsize=13, pad=10)

    # --- (b) 그런데 프로파일 로그가능도는 하나로 포개진다 ---
    ax = axes[1]
    mu = np.linspace(2.6, 7.4, 500)
    for k, (name, d, c) in enumerate(datasets):
        vhat = ((d[:, None] - mu[None, :]) ** 2).mean(axis=0)
        prof = -n / 2 * (np.log(2 * np.pi) + np.log(vhat) + 1)
        # 네 곡선이 완전히 겹치므로 파선의 위상만 어긋나게 준다.
        ax.plot(mu, prof, color=c, linewidth=3.0,
                linestyle=(k * 7, (7, 21)), solid_capstyle="butt", zorder=5,
                label=name.split()[0])

    top = -n / 2 * (np.log(2 * np.pi) + np.log(target_sd ** 2 * (n - 1) / n)
                    + 1)
    ax.plot([target_mean], [top], "o", color=RED, markersize=9, zorder=8)
    # 파선을 짧게 끊어 아래쪽 설명글과 부딪히지 않게 한다.
    ax.plot([target_mean, target_mean], [top - 3.2, top], color=RED,
            linewidth=1.2, linestyle="--", zorder=4)
    ax.text(target_mean, top + 0.35, r"$\hat\mu = \bar x = 5.00$",
            fontsize=11.5, color=RED, ha="center", va="bottom")

    ax.text(target_mean, top - 3.75,
            "네 곡선이 완전히 포개져 있다.\n"
            "가능도는 자료를 오직 $\\bar x$ 와\n"
            "$\\sum (x_i - \\bar x)^2$ 을 통해서만 본다.",
            fontsize=11, color=INK, ha="center", va="top", linespacing=1.65)

    ax.set_xlim(2.6, 7.4)
    ax.set_ylim(top - 7.5, top + 1.6)
    ax.set_xlabel(r"$\mu$", fontsize=13, color=INK)
    ax.set_ylabel("프로파일 로그가능도", fontsize=12, color=INK)
    ax.tick_params(labelsize=10, colors=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    # 파선의 위상을 어긋나게 준 탓에 기본 범례 표본에는 색이 안 보인다.
    # 실선 대역을 따로 만들어 넣는다.
    proxies = [Line2D([0], [0], color=c, linewidth=3.0)
               for _, _, c in datasets]
    ax.legend(proxies, [name.split()[0] for name, _, _ in datasets],
              fontsize=11, frameon=False, loc="upper right", ncol=4,
              handlelength=1.6, columnspacing=1.0)
    ax.set_title("네 자료의 로그가능도", fontsize=13, pad=10)

    fig.suptitle("버려도 되는 것과 버리면 안 되는 것", fontsize=14.5, y=1.02)
    fig.tight_layout()
    save(fig, OUT + "sufficiency_compression.png")


if __name__ == "__main__":
    gaussian_loglik_surface()
    mle_bias_shrinks()
    sufficiency_compression()
