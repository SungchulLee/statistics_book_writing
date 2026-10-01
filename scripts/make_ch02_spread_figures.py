r"""2장 산포 절에서 그림이 없던 두 쪽의 그림을 생성한다.

  ch02/spread/img/why_square.png        편차를 왜 제곱하는가
  ch02/spread/img/mad_breakdown.png     이상치가 들어올 때 두 척도의 운명

mad_breakdown 의 왼쪽 칸은 그 쪽 보기 2 가 보고한 수치를 그대로 옮긴
것이다(원자료는 실행 때 내려받는 것이라 다시 읽지 않았다). 오른쪽 칸은
이 스크립트가 직접 모의실험한다.

실행:  python3 scripts/make_ch02_spread_figures.py   (저장소 최상위에서)
필요:  numpy, matplotlib — 문서 빌드에는 필요하지 않다.
       그림은 PNG로 커밋되므로 CI에서 다시 그리지 않는다.
"""

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# === 공통 설정 ===
plt.rcParams["font.family"] = "Apple SD Gothic Neo"   # 한글 폰트
plt.rcParams["axes.unicode_minus"] = False

INK = "#37474F"
BLUE, BLUE_F = "#1565C0", "#DCEBFB"
ORANGE, ORANGE_F = "#E65100", "#FFE0B2"
GREEN, GREEN_F = "#33691E", "#C5E1A5"
PURPLE = "#6A1B9A"
MUTED = "#90A4AE"
RED = "#D32F2F"

OUT = "docs/ch02/spread/img/"


def save(fig, path):
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def clean_axis(ax):
    ax.tick_params(labelsize=10, colors=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)


# ==================================================================
# 1. 왜 제곱하는가
# ==================================================================
def why_square():
    x = np.array([4.0, 5.0, 5.0, 6.0, 6.0, 7.0, 7.0, 14.0])
    xbar = x.mean()
    d = x - xbar
    share_abs = np.abs(d) / np.abs(d).sum()
    share_sq = d ** 2 / (d ** 2).sum()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4),
                             gridspec_kw={"width_ratios": [1.15, 1, 1.1]})

    # (a) 편차의 합은 언제나 0
    ax = axes[0]
    for i, (xi, di) in enumerate(zip(x, d)):
        c = BLUE if di < 0 else ORANGE
        ax.plot([xbar, xi], [i, i], color=c, linewidth=2.6,
                solid_capstyle="butt", zorder=4)
        ax.plot([xi], [i], "o", color=c, markersize=8, zorder=5)
        ax.text(xi + (0.35 if di >= 0 else -0.35), i, f"{di:+.2f}",
                fontsize=9.5, color=c, ha="left" if di >= 0 else "right",
                va="center")
    ax.axvline(xbar, color=INK, linewidth=1.8, zorder=6)
    ax.text(xbar, len(x) - 0.4, f"평균 {xbar:.2f}", fontsize=11, color=INK,
            ha="center", va="bottom")
    ax.text(xbar, -1.5,
            f"음의 편차 합 ${d[d < 0].sum():.2f}$  $+$  "
            f"양의 편차 합 ${d[d > 0].sum():.2f}$  $=$  $0$",
            fontsize=11, color=INK, ha="center", va="center")
    ax.set_xlim(2.0, 16.5)
    ax.set_ylim(-2.4, len(x) + 0.3)
    ax.set_yticks([])
    ax.set_xlabel("$x$", fontsize=11.5, color=INK)
    ax.tick_params(axis="x", labelsize=10, colors=INK, length=3)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.set_title("편차를 그냥 더하면 언제나 $0$ 이다", fontsize=12.5, pad=10)

    # (b) 두 벌점 함수의 모양
    ax = axes[1]
    t = np.linspace(-3, 3, 600)
    ax.plot(t, np.abs(t), color=GREEN, linewidth=2.6)
    ax.plot(t, t ** 2, color=BLUE, linewidth=2.6)
    ax.text(2.72, 2.55, "$|d|$", fontsize=14, color=GREEN, ha="left",
            va="center")
    ax.text(2.42, 6.2, "$d^2$", fontsize=14, color=BLUE, ha="left",
            va="center")
    ax.plot([0], [0], "o", color=RED, markersize=8, zorder=6)
    ax.annotate("$0$ 에서 꺾여 미분할 수 없다", xy=(0, 0), xytext=(1.95, 0.55),
                fontsize=10.5, color=GREEN, ha="center", va="center",
                arrowprops=dict(arrowstyle="->", color=GREEN, linewidth=1.2))
    ax.text(-2.85, 7.6, "$d^2$ 은 어디서나 매끄럽고\n먼 점에 훨씬 큰 벌점을 준다",
            fontsize=10.5, color=BLUE, ha="left", va="top", linespacing=1.5)
    ax.set_xlim(-3, 3)
    ax.set_ylim(0, 9)
    ax.set_xlabel("편차 $d = x_i - \\bar x$", fontsize=11.5, color=INK)
    clean_axis(ax)
    ax.set_title("부호를 없애는 두 방법", fontsize=12.5, pad=10)

    # (c) 한 점이 차지하는 몫
    ax = axes[2]
    idx = np.arange(len(x))
    ax.barh(idx + 0.19, share_abs * 100, height=0.36, color=GREEN_F,
            edgecolor=GREEN, linewidth=1.4, label="$|d|$ 를 쓸 때")
    ax.barh(idx - 0.19, share_sq * 100, height=0.36, color=BLUE_F,
            edgecolor=BLUE, linewidth=1.4, label="$d^2$ 을 쓸 때")
    j = int(np.argmax(np.abs(d)))
    ax.text(share_abs[j] * 100 + 1.5, j + 0.19, f"{share_abs[j]*100:.0f}%",
            fontsize=10.5, color=GREEN, ha="left", va="center")
    ax.text(share_sq[j] * 100 + 1.5, j - 0.19, f"{share_sq[j]*100:.0f}%",
            fontsize=10.5, color=BLUE, ha="left", va="center")
    ax.annotate(f"가장 먼 점 $x = {x[j]:g}$", xy=(share_sq[j] * 100, j - 0.19),
                xytext=(52, j - 2.6), fontsize=10.5, color=INK, ha="center",
                arrowprops=dict(arrowstyle="->", color=INK, linewidth=1.2))
    ax.set_yticks(idx)
    ax.set_yticklabels([f"{v:g}" for v in x], fontsize=9.5)
    ax.set_xlim(0, 95)
    ax.set_xlabel("흩어짐 전체에서 차지하는 몫 (%)", fontsize=11.5, color=INK)
    ax.set_ylabel("$x_i$", fontsize=11.5, color=INK)
    clean_axis(ax)
    ax.legend(fontsize=10.5, frameon=False, loc="lower right")
    ax.set_title("제곱하면 먼 점 하나가 지배한다", fontsize=12.5, pad=10)

    fig.suptitle("제곱은 부호를 없애는 김에 먼 점의 목소리를 키운다",
                 fontsize=13.5, y=1.03)
    fig.tight_layout()
    save(fig, OUT + "why_square.png")

    print(f"  x = {x.tolist()}, mean = {xbar}")
    print(f"  |d| 합 = {np.abs(d).sum():.2f}, d^2 합 = {(d**2).sum():.2f}")
    print(f"  가장 먼 점의 몫: |d| {share_abs[j]*100:.1f}%, "
          f"d^2 {share_sq[j]*100:.1f}%")


# ==================================================================
# 2. 붕괴점 — 오염이 늘어나면
# ==================================================================
def mad_breakdown():
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8),
                             gridspec_kw={"width_ratios": [1, 1.2]})

    # (a) 그 쪽 보기 2 의 수치를 그대로
    ax = axes[0]
    names = ["표준편차", "MAD"]
    before = np.array([6_848_235, 3_849_876])
    after = np.array([24_537_372, 4_273_462])
    xs = np.arange(2)
    ax.bar(xs - 0.19, before / 1e6, width=0.36, color=MUTED, alpha=0.55,
           edgecolor=INK, linewidth=1.4, label="원래 자료 (50개 주)")
    ax.bar(xs + 0.19, after / 1e6, width=0.36, color=RED, alpha=0.55,
           edgecolor=RED, linewidth=1.4, label="가상의 두 주를 넣은 뒤")
    for i in range(2):
        ax.text(xs[i] - 0.19, before[i] / 1e6 + 0.6, f"{before[i]/1e6:.1f}",
                fontsize=10.5, color=INK, ha="center", va="bottom")
        ax.text(xs[i] + 0.19, after[i] / 1e6 + 0.6, f"{after[i]/1e6:.1f}",
                fontsize=10.5, color=RED, ha="center", va="bottom")
        pct = 100 * (after[i] - before[i]) / before[i]
        ax.text(xs[i], after[i] / 1e6 + 3.2, f"{pct:+.1f}%", fontsize=12,
                color=RED if pct > 100 else GREEN, ha="center", va="bottom")
    ax.set_xticks(xs)
    ax.set_xticklabels(names, fontsize=12)
    ax.set_ylim(0, 31)
    ax.set_ylabel("백만 명", fontsize=11.5, color=INK)
    clean_axis(ax)
    ax.legend(fontsize=10, frameon=False, loc="upper right")
    ax.set_title("자료 52개 중 둘만 바꾸었을 때", fontsize=12.5, pad=10)

    # (b) 오염 비율을 늘려 가며
    ax = axes[1]
    rng = np.random.default_rng(0)
    n = 2000
    base = rng.normal(0, 1, n)
    fracs = np.arange(0, 0.66, 0.02)
    sds, mads = [], []
    for f in fracs:
        k = int(round(f * n))
        y = base.copy()
        if k:
            y[:k] = 1e6                       # 터무니없이 큰 값으로 오염
        sds.append(y.std(ddof=1))
        m = 1.4826 * np.median(np.abs(y - np.median(y)))
        # 절반을 넘기면 중앙값 자체가 오염값이 되어 MAD 가 정확히 0 이 된다.
        # 로그축에는 그릴 수 없으므로 선을 끊는다.
        mads.append(m if m > 0 else np.nan)
    ax.plot(fracs * 100, sds, color=RED, linewidth=2.6, label="표준편차")
    ax.plot(fracs * 100, mads, color=GREEN, linewidth=2.6,
            label="MAD (표준화)")
    ax.axvline(50, color=INK, linewidth=1.6, linestyle="--")
    ax.text(51.5, 3e4, "붕괴점 50%\n여기를 넘으면 MAD 가\n정확히 $0$ 이 된다",
            fontsize=10.5, color=INK, ha="left", va="center",
            linespacing=1.5)
    ax.axhline(1.0, color=MUTED, linewidth=1.2, linestyle=":")
    ax.text(0.8, 1.35, "참값 $\\sigma = 1$", fontsize=10.5, color=MUTED,
            ha="left", va="bottom")
    ax.text(6, 2e3,
            "표준편차는 한 점만 오염되어도\n곧바로 무너진다 (붕괴점 0%)",
            fontsize=10.5, color=RED, ha="left", va="center",
            linespacing=1.5)
    ax.text(14, 0.25,
            "MAD 는 절반이 오염될 때까지\n$\\sigma$ 근처에 머문다",
            fontsize=10.5, color=GREEN, ha="left", va="center",
            linespacing=1.5)
    ax.set_yscale("log")
    ax.set_yticks([0.1, 1, 10, 1e2, 1e3, 1e4, 1e5, 1e6])
    ax.set_yticklabels(["0.1", "1", "10", "100", "1천", "1만", "10만",
                        "100만"])
    ax.set_xlim(-1, 68)
    ax.set_ylim(0.1, 3e6)
    ax.set_xlabel(f"오염된 관측값의 비율 (%)   (정규자료 {n}개)",
                  fontsize=11.5, color=INK)
    ax.set_ylabel("추정한 산포 (로그 눈금)", fontsize=11.5, color=INK)
    clean_axis(ax)
    ax.legend(fontsize=11, frameon=False, loc="upper left")
    ax.set_title("오염을 늘려 가며 두 척도를 재면", fontsize=12.5, pad=10)

    fig.suptitle("같은 자료를 재는데 한쪽은 무너지고 한쪽은 버틴다",
                 fontsize=13.5, y=1.02)
    fig.tight_layout()
    save(fig, OUT + "mad_breakdown.png")

    for f in (0.0, 0.02, 0.10, 0.48, 0.52):
        k = int(round(f * n))
        y = base.copy()
        if k:
            y[:k] = 1e6
        print(f"  오염 {f:.0%}: SD = {y.std(ddof=1):.3g}, "
              f"MAD = {1.4826 * np.median(np.abs(y - np.median(y))):.3g}")


if __name__ == "__main__":
    why_square()
    mad_breakdown()
