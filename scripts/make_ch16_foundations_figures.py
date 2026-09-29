r"""16.1 비모수 검정의 기초 네 쪽에 들어가는 개념 그림을 만든다.

만드는 파일:
  ch16/foundations/img/t_skew_asymmetry.png      치우친 자료에서 t 의 기각이 한쪽 꼬리에 몰린다
  ch16/foundations/img/rank_transform.png        어떤 분포든 순위는 같은 격자로 간다
  ch16/foundations/img/are_power_curves.png      분포가 바뀌면 검정력 순서가 뒤집힌다
  ch16/foundations/img/paired_structure_lost.png 대응 구조를 버리면 신호가 사라진다

실행:  python3 scripts/make_ch16_foundations_figures.py   (저장소 최상위에서)
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

OUT = "docs/ch16/foundations/img/"
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    path = OUT + name
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def clean(ax):
    ax.tick_params(labelsize=9.5, colors=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)


def bare(ax):
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=9.5, colors=INK, length=3)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)


# ==================================================================
# 1. 치우친 자료에서 t 의 기각은 한쪽 꼬리에 몰린다
# ==================================================================
def t_skew_asymmetry():
    rng = np.random.default_rng(20240916)
    B = 200_000
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.1))

    for ax, n in zip(axes, (5, 30)):
        x = rng.exponential(1.0, (B, n))
        t = (x.mean(1) - 1.0) / (x.std(1, ddof=1) / np.sqrt(n))
        crit = stats.t.ppf(0.975, n - 1)
        left = (t < -crit).mean()
        right = (t > crit).mean()

        grid = np.linspace(-7.5, 4.0, 700)
        ax.hist(t, bins=np.linspace(-7.5, 4.0, 180), density=True,
                color=BLUE_F, edgecolor="none",
                label="실제로 나온 $t$")
        ax.plot(grid, stats.t.pdf(grid, n - 1), color=INK, lw=1.7,
                label=f"이론 귀무분포 $t_{{{n-1}}}$")

        # 꼬리 색칠
        gl = grid[grid <= -crit]
        gr = grid[grid >= crit]
        ax.fill_between(gl, 0, stats.t.pdf(gl, n - 1), color=RED, alpha=0.75)
        ax.fill_between(gr, 0, stats.t.pdf(gr, n - 1), color=RED, alpha=0.75)
        ax.set_xlim(-7.5, 4.0)
        top = max(stats.t.pdf(0, n - 1), np.histogram(
            t, bins=np.linspace(-7.5, 4.0, 180), density=True)[0].max())
        ax.set_ylim(0, top * 1.50)
        for c in (-crit, crit):
            ax.vlines(c, 0, top * 0.72, color=RED, lw=1.1, ls=(0, (4, 3)))
        bare(ax)
        ax.set_xlabel(r"$t = (\bar{X}-1)/(S/\sqrt{n})$", fontsize=10.5, color=INK)
        ax.set_title(f"n = {n}", fontsize=12, color=INK, pad=8)

        ax.annotate(f"왼쪽 꼬리 기각 {left:.3f}",
                    xy=(-crit - 0.30, stats.t.pdf(crit, n - 1) * 0.6),
                    xytext=(-7.3, top * 1.47),
                    fontsize=10.5, color=RED, ha="left", va="top",
                    arrowprops=dict(arrowstyle="->", color=RED, lw=1.1))
        ax.annotate(f"오른쪽 꼬리 기각 {right:.3f}",
                    xy=(crit + 0.25, stats.t.pdf(crit, n - 1) * 0.55),
                    xytext=(3.9, top * 1.47),
                    fontsize=10.5, color=RED, ha="right", va="top",
                    arrowprops=dict(arrowstyle="->", color=RED, lw=1.1))
        ax.text(-7.3, top * 1.12, f"실제 크기 {left + right:.3f}\n명목 0.050",
                fontsize=11, color=INK, ha="left", va="top")
        ax.legend(fontsize=9.5, loc="upper left",
                  bbox_to_anchor=(0.0, 0.50), frameon=False)

    fig.suptitle("지수분포 자료에 일표본 $t$ 검정을 쓰면 기각이 왼쪽으로만 쏟아진다",
                 fontsize=13, color=INK, y=1.01)
    fig.tight_layout()
    save(fig, "t_skew_asymmetry.png")


# ==================================================================
# 2. 순위변환은 어떤 분포든 같은 격자로 보낸다
# ==================================================================
def rank_transform():
    rng = np.random.default_rng(3)
    n = 12
    base = np.sort(rng.normal(0, 1, n))
    sets = [
        ("정규 자료", np.round(50 + 8 * base, 1), BLUE),
        ("오른쪽으로 치우친 자료", np.round(np.exp(3.9 + 0.55 * base), 1), GREEN),
        ("이상치 하나가 섞인 자료", None, ORANGE),
    ]
    contaminated = np.round(50 + 8 * base, 1).copy()
    contaminated[-1] = 900.0
    sets[2] = ("이상치 하나가 섞인 자료", contaminated, ORANGE)

    fig, axes = plt.subplots(3, 1, figsize=(10.4, 7.2))
    for ax, (label, vals, color) in zip(axes, sets):
        r = stats.rankdata(vals)
        # 위: 실제 값의 위치 (각 패널마다 자기 눈금)
        lo, hi = vals.min(), vals.max()
        pad = (hi - lo) * 0.06
        xs_val = (vals - lo) / (hi - lo)          # 0~1 로 정규화해 같은 폭에 그린다
        xs_rank = (r - 1) / (n - 1)

        ax.hlines(1.0, -0.04, 1.04, color=MUTED, lw=1.0)
        ax.hlines(0.0, -0.04, 1.04, color=MUTED, lw=1.0)
        ax.plot(xs_val, np.full(n, 1.0), "o", ms=8, color=color,
                mec="white", mew=1.0, zorder=3)
        ax.plot(xs_rank, np.zeros(n), "o", ms=8, color=color,
                mec="white", mew=1.0, zorder=3)
        for xv, xr in zip(xs_val, xs_rank):
            ax.plot([xv, xr], [1.0, 0.0], color=color, lw=0.8, alpha=0.45,
                    zorder=1)

        ax.text(-0.055, 1.0, "원값", fontsize=10, color=INK,
                ha="right", va="center")
        ax.text(-0.055, 0.0, "순위", fontsize=10, color=INK,
                ha="right", va="center")
        for xr, rr in zip(xs_rank, r):
            ax.text(xr, -0.28, f"{int(rr)}", fontsize=8.5, color=INK,
                    ha="center", va="center")
        # 원값 몇 개만 숫자로
        for idx in (0, n - 1):
            ax.text(xs_val[idx], 1.28, f"{vals[idx]:g}", fontsize=8.5,
                    color=color, ha="center", va="center")
        ax.set_title(label, fontsize=11.5, color=INK, loc="left", pad=4)
        ax.set_xlim(-0.175, 1.04)
        ax.set_ylim(-0.6, 1.62)
        ax.axis("off")

    fig.suptitle("세 표본은 원값의 모양이 전혀 다르지만 순위는 똑같은 등간 격자다",
                 fontsize=13, color=INK, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "rank_transform.png")
    return dict(
        normal=np.round(50 + 8 * base, 1),
        lognormal=np.round(np.exp(3.9 + 0.55 * base), 1),
        contaminated=contaminated,
    )


# ==================================================================
# 3. 분포가 바뀌면 검정력 순서가 뒤집힌다
# ==================================================================
def are_power_curves():
    rng = np.random.default_rng(1234)
    n, B = 20, 3000
    deltas = np.array([0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9])

    def gen(kind, delta, size):
        if kind == "normal":
            return rng.normal(delta, 1.0, size)
        if kind == "laplace":
            return rng.laplace(delta, 1 / np.sqrt(2), size)
        return rng.standard_t(3, size) / np.sqrt(3.0) + delta

    kinds = [("정규분포", "normal"), ("Laplace 분포", "laplace"),
             ("$t(3)$ 분포 (분산 1 로 맞춤)", "t3")]
    results = {}
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.2), sharey=True)
    for ax, (title, kind) in zip(axes, kinds):
        pt, pw, ps = [], [], []
        for d in deltas:
            x = gen(kind, d, (B, n))
            pt.append((stats.ttest_1samp(x, 0, axis=1).pvalue < 0.05).mean())
            wp = np.array([stats.wilcoxon(x[i]).pvalue for i in range(B)])
            pw.append((wp < 0.05).mean())
            npos = (x > 0).sum(1)
            sp = 2 * np.minimum(stats.binom.cdf(np.minimum(npos, n - npos), n, 0.5),
                                0.5)
            ps.append((sp < 0.05).mean())
        results[kind] = (np.array(pt), np.array(pw), np.array(ps))

        ax.plot(deltas, pt, "-o", color=INK, lw=1.9, ms=5, label="일표본 $t$ 검정")
        ax.plot(deltas, pw, "-s", color=BLUE, lw=1.9, ms=5,
                label="Wilcoxon 부호순위")
        ax.plot(deltas, ps, "-^", color=ORANGE, lw=1.9, ms=5, label="부호검정")
        ax.axhline(0.05, color=MUTED, lw=1.0, ls=(0, (4, 3)))
        clean(ax)
        ax.set_xlabel("위치 이동 $\\delta$", fontsize=10.5, color=INK)
        ax.set_title(title, fontsize=11.5, color=INK, pad=7)
        ax.set_ylim(0, 1.0)
        ax.set_xlim(-0.03, 0.93)
    axes[0].set_ylabel("검정력", fontsize=10.5, color=INK)
    axes[0].legend(fontsize=9.5, loc="upper left", frameon=False)
    axes[2].text(0.90, 0.095, "명목 $\\alpha = 0.05$",
                 fontsize=9.5, color=MUTED, ha="right", va="bottom")

    fig.suptitle("정규분포에서는 $t$ 가 앞서지만 꼬리가 두꺼워지면 순서가 뒤집힌다",
                 fontsize=13, color=INK, y=1.02)
    fig.tight_layout()
    save(fig, "are_power_curves.png")
    for k, (a, b, c) in results.items():
        print(k, "delta=0.45 ->", "t %.3f  W %.3f  sign %.3f"
              % (a[3], b[3], c[3]))
    return results, deltas


# ==================================================================
# 4. 대응 구조를 버리면 신호가 사라진다
# ==================================================================
def paired_structure_lost():
    data = np.array([
        [93, 76], [70, 72], [81, 75], [65, 68], [79, 65],
        [54, 54], [94, 88], [91, 81], [77, 65], [65, 57],
        [95, 86], [89, 87], [78, 78], [80, 77], [76, 76]])
    post, pre = data[:, 0].astype(float), data[:, 1].astype(float)
    d = post - pre

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6),
                             gridspec_kw=dict(width_ratios=[1.15, 1]))

    # (a) 대응으로 본 자료
    ax = axes[0]
    for i, (a, b) in enumerate(zip(pre, post)):
        col = BLUE if b > a else (RED if b < a else MUTED)
        ax.plot([0, 1], [a, b], color=col, lw=1.4, alpha=0.85, zorder=2)
    ax.plot(np.zeros(15), pre, "o", ms=6, color=INK, mec="white", mew=0.8,
            zorder=3)
    ax.plot(np.ones(15), post, "o", ms=6, color=INK, mec="white", mew=0.8,
            zorder=3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["처치 전", "처치 후"], fontsize=10.5, color=INK)
    ax.set_xlim(-0.35, 1.45)
    clean(ax)
    ax.set_ylabel("점수", fontsize=10.5, color=INK)
    nplus, nminus, nzero = (d > 0).sum(), (d < 0).sum(), (d == 0).sum()
    ax.text(1.08, 96, f"올라간 학생 {nplus}명", fontsize=10, color=BLUE,
            ha="left", va="center")
    ax.text(1.08, 91, f"내려간 학생 {nminus}명", fontsize=10, color=RED,
            ha="left", va="center")
    ax.text(1.08, 86, f"변화 없음 {nzero}명", fontsize=10, color=MUTED,
            ha="left", va="center")
    ax.set_title("(가) 같은 학생끼리 이어 보면 방향이 거의 한쪽이다",
                 fontsize=11.5, color=INK, loc="left", pad=8)

    # (b) 독립표본처럼 본 자료
    ax = axes[1]
    bins = np.arange(50, 101, 5)
    ax.hist(pre, bins=bins, color=ORANGE_F, edgecolor=ORANGE, lw=1.2,
            label="처치 전")
    ax.hist(post, bins=bins, facecolor="none", edgecolor=BLUE, lw=1.8,
            hatch="///", label="처치 후")
    clean(ax)
    ax.set_xlabel("점수", fontsize=10.5, color=INK)
    ax.set_ylabel("학생 수", fontsize=10.5, color=INK)
    ax.set_ylim(0, 6.6)
    ax.set_yticks([0, 1, 2, 3, 4, 5])
    ax.legend(fontsize=9.5, loc="upper left", frameon=False)
    ax.set_title("(나) 두 무더기로만 보면 크게 겹친다",
                 fontsize=11.5, color=INK, loc="left", pad=8)

    p_sr = stats.wilcoxon(post, pre, method="approx",
                          zero_method="pratt").pvalue
    p_rs = stats.ranksums(post, pre).pvalue
    ax.text(0.99, 0.99, f"부호순위 $p$ = {p_sr:.4f}\n순위합 $p$ = {p_rs:.4f}",
            transform=ax.transAxes, fontsize=10.5, color=INK,
            ha="right", va="top")

    fig.suptitle("같은 15명의 자료인데 대응 구조를 버리자 $p$ 값이 16배로 커졌다",
                 fontsize=13, color=INK, y=1.02)
    fig.tight_layout()
    save(fig, "paired_structure_lost.png")
    print("paired: n+ %d n- %d n0 %d, sr p=%.4f rs p=%.4f, ratio %.1f"
          % (nplus, nminus, nzero, p_sr, p_rs, p_rs / p_sr))
    print("mean d = %.3f, sd d = %.3f" % (d.mean(), d.std(ddof=1)))


if __name__ == "__main__":
    t_skew_asymmetry()
    rank_transform()
    are_power_curves()
    paired_structure_lost()
