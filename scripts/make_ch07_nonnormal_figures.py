r"""7.3 치우친 분포 쪽의 그림을 생성한다.

만드는 파일:
  ch07/non_normal/img/skew_center_and_clt.png   치우침이 중심을 가르고, 평균의 분포까지 끌고 간다

실행:  python3 scripts/make_ch07_nonnormal_figures.py   (저장소 최상위에서)
필요:  numpy, matplotlib — 문서 빌드에는 필요하지 않다.
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

OUT = "docs/ch07/non_normal/img/"


def save(fig, path):
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def bare_axis(ax):
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=10, colors=INK, length=3)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)


def kde(vals, grid, bw):
    d = (grid[:, None] - vals[None, :]) / bw
    return np.exp(-d ** 2 / 2).sum(axis=1) / (len(vals) * bw
                                              * np.sqrt(2 * np.pi))


# ==================================================================
# 치우침의 두 얼굴
# ==================================================================
def skew_center_and_clt():
    # LogNormal(0, 1)
    mean = np.exp(0.5)
    median = 1.0
    mode = np.exp(-1.0)
    sd = np.sqrt((np.exp(1) - 1) * np.exp(1))
    skew = (np.exp(1) + 2) * np.sqrt(np.exp(1) - 1)
    print(f"  LogNormal(0,1):  최빈 {mode:.4f}  중앙 {median:.4f}  "
          f"평균 {mean:.4f}  SD {sd:.4f}  왜도 {skew:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.0),
                             gridspec_kw={"width_ratios": [1, 1.1]})

    # --- (a) 세 중심이 갈라진다 ---
    ax = axes[0]
    x = np.linspace(0.005, 6.0, 900)
    pdf = np.exp(-(np.log(x) ** 2) / 2) / (x * np.sqrt(2 * np.pi))
    ax.plot(x, pdf, color=BLUE, linewidth=2.4, zorder=5)
    ax.fill_between(x, pdf, color=BLUE, alpha=0.13, zorder=3)

    # 세 라벨을 오른쪽으로 내려오는 계단처럼 배치한다. 최빈값 라벨을
    # 선 왼쪽에 두면 그림 왼쪽 경계 밖으로 잘린다.
    marks = [(mode, GREEN, "최빈값", 0.90), (median, PURPLE, "중앙값", 0.70),
             (mean, ORANGE, "평균", 0.50)]
    for val, c, name, h in marks:
        ax.plot([val, val], [0, h], color=c, linewidth=1.8,
                linestyle="--", zorder=6)
        ax.text(val + 0.10, h, f"{name}  {val:.2f}", fontsize=11.5, color=c,
                ha="left", va="center",
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="none",
                          pad=1.8))

    ax.add_patch(FancyArrowPatch((median, 0.26), (mean, 0.26),
                                 arrowstyle="-|>", mutation_scale=13,
                                 color=INK, linewidth=1.5, zorder=8))
    ax.text(mean + 0.16, 0.26, "긴 꼬리가\n평균을 끌어간다", fontsize=11,
            color=INK, ha="left", va="center", linespacing=1.55)

    ax.text(3.6, 0.72, f"왜도 $\\gamma_1 = {skew:.2f}$", fontsize=12.5,
            color=INK, ha="left", va="center")

    ax.set_xlim(0, 6.0)
    ax.set_ylim(0, 1.02)
    ax.set_xticks([0, 1, 2, 3, 4, 5, 6])
    bare_axis(ax)
    ax.set_title("로그정규분포 $\\mathrm{LogNormal}(0, 1)$ 의 세 중심",
                 fontsize=13, pad=10)

    # --- (b) 표준화한 표본평균 — n = 30 도 아직 멀었다 ---
    ax = axes[1]
    rng = np.random.default_rng(2024)
    grid = np.linspace(-3.4, 4.2, 800)
    reps = 300_000
    rows = []
    for n, c in [(5, ORANGE), (30, PURPLE), (100, BLUE)]:
        xb = rng.lognormal(0, 1, size=(reps, n)).mean(axis=1)
        z = (xb - mean) / (sd / np.sqrt(n))
        left = np.mean(z < -1.645)
        right = np.mean(z > 1.645)
        g = ((z - z.mean()) ** 3).mean() / z.std() ** 3
        rows.append((n, left, right, g))
        print(f"  n={n:3d}  P(z<-1.645)={left:.4f}  P(z>1.645)={right:.4f}"
              f"  왜도={g:.3f}  (γ₁/√n = {skew / np.sqrt(n):.3f})")
        ax.plot(grid, kde(z[:60000], grid, 0.085), color=c, linewidth=2.3,
                zorder=5, label=f"$n = {n}$   왜도 ${g:.2f}$")

    std_normal = np.exp(-grid ** 2 / 2) / np.sqrt(2 * np.pi)
    ax.plot(grid, std_normal, color=INK, linewidth=2.0, linestyle=(0, (5, 4)),
            zorder=6, label="$N(0, 1)$")

    ax.plot([-1.645, -1.645], [0, 0.60], color=RED, linewidth=1.4,
            linestyle=":", zorder=7)
    ax.text(-1.70, 0.615, "$z = -1.645$", fontsize=11, color=RED,
            ha="center", va="bottom")

    tail = "\n".join(
        [f"$n = {n}$ :  {left * 100:4.1f}%" for n, left, _, _ in rows])
    ax.text(-3.35, 0.545,
            "이 선 왼쪽의 실제 확률\n정규근사라면 5.0%\n" + tail,
            fontsize=10.5, color=INK, ha="left", va="top", linespacing=1.7)

    ax.set_xlim(-3.4, 4.2)
    ax.set_ylim(0, 0.84)
    ax.set_xlabel(r"$(\bar X - \mu) \,/\, (\sigma/\sqrt{n})$", fontsize=12,
                  color=INK)
    ax.set_xticks([-3, -2, -1, 0, 1, 2, 3, 4])
    bare_axis(ax)
    ax.legend(fontsize=10.8, frameon=False, loc="upper right",
              handlelength=1.8)
    ax.set_title("표준화한 표본평균의 분포", fontsize=13, pad=10)

    fig.suptitle("치우침은 중심을 갈라놓고, 표본평균의 분포까지 끌고 간다",
                 fontsize=14.5, y=1.02)
    fig.tight_layout()
    save(fig, OUT + "skew_center_and_clt.png")


if __name__ == "__main__":
    skew_center_and_clt()
