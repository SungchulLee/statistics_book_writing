r"""8장 대응표본 신뢰구간 두 쪽의 개념 그림을 생성한다.

만드는 파일:
  ch08/paired_sample_intervals/img/pairing_gain.png       짝짓기의 이득은 1-rho 다
  ch08/paired_sample_intervals/img/mcnemar_discordant.png 일치 짝은 사라진다

실행:  python3 scripts/make_ch08_paired_figures.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

INK = "#37474F"
BLUE, BLUE_F = "#1565C0", "#DCEBFB"
ORANGE, ORANGE_F = "#E65100", "#FFE0B2"
GREEN, GREEN_F = "#33691E", "#C5E1A5"
PURPLE = "#6A1B9A"
MUTED = "#90A4AE"
RED = "#D32F2F"

OUT = "docs/ch08/paired_sample_intervals/img/"
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def clean_axis(ax):
    ax.tick_params(labelsize=10, colors=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)


# === 그림 1. 짝짓기의 이득 (paired_vs_independent.md) ============
def fig_pairing_gain():
    x1 = np.array([250.0, 310, 280, 340, 220])
    x2 = np.array([230.0, 290, 260, 325, 205])
    d = x1 - x2
    n = len(d)
    s1, s2 = x1.std(ddof=1), x2.std(ddof=1)
    sd = d.std(ddof=1)
    rho = np.corrcoef(x1, x2)[0, 1]
    se_p = sd / np.sqrt(n)
    sp = np.sqrt((s1 ** 2 + s2 ** 2) / 2)
    se_i = sp * np.sqrt(2 / n)
    print(f"  s1={s1:.2f} s2={s2:.2f} sD={sd:.2f} rho={rho:.4f}")
    print(f"  SE_paired={se_p:.3f} SE_indep={se_i:.3f} ratio={se_i/se_p:.1f}")
    print(f"  1/sqrt(1-rho)={1/np.sqrt(1-rho):.1f}")

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(12.6, 4.8), gridspec_kw={"width_ratios": [1.0, 1.1]}
    )

    # --- 왼쪽: 사람마다의 선 ---
    for i in range(n):
        axL.plot([0, 1], [x1[i], x2[i]], color=MUTED, lw=1.4, zorder=1)
        axL.plot([0], [x1[i]], marker="o", ms=7, color=BLUE, zorder=3)
        axL.plot([1], [x2[i]], marker="o", ms=7, color=ORANGE, zorder=3)
        axL.text(1.06, x2[i], f"$-{d[i]:.0f}$", fontsize=10, color=INK,
                 ha="left", va="center")
    axL.text(-0.38, 280, "사람 사이 산포\n120 ms", fontsize=10.5, color=INK,
             ha="right", va="center", linespacing=1.5)
    axL.annotate("", xy=(-0.30, 220), xytext=(-0.30, 340),
                 arrowprops=dict(arrowstyle="<->", color=INK, lw=1.3))

    axL.set_xlim(-1.12, 1.38)
    axL.set_ylim(190, 360)
    axL.set_xticks([0, 1])
    axL.set_xticklabels([f"미섭취\n(평균 {x1.mean():.0f})",
                         f"섭취\n(평균 {x2.mean():.0f})"],
                        fontsize=11, color=INK)
    axL.set_ylabel("반응시간 (ms)", fontsize=10.5, color=INK)
    axL.spines[["top", "right"]].set_visible(False)
    axL.spines[["left", "bottom"]].set_color(MUTED)
    axL.tick_params(labelsize=10, colors=INK)
    axL.set_title("사람 사이는 벌어져 있고 사람 안의 변화는 한결같다",
                  fontsize=12, color=INK, pad=10)
    axL.text(0.5, 197, f"차이는 15~20 ms 뿐이다  ($s_D={sd:.2f}$)",
             fontsize=10.5, color=GREEN, ha="center", va="center")

    # --- 오른쪽: 표준오차의 비 = 1/sqrt(1-rho) ---
    r = np.linspace(-0.5, 0.9985, 2000)
    ratio = 1 / np.sqrt(1 - r)
    axR.plot(r, ratio, color=BLUE, lw=2.2, zorder=3)
    axR.axhline(1.0, color=MUTED, lw=1.2, ls=(0, (5, 4)), zorder=1)
    axR.axvspan(-0.5, 0, color="#F2F2F2", zorder=0)
    axR.text(-0.33, 0.775, "짝짓기가\n오히려 손해", fontsize=10.5, color=MUTED,
             ha="center", va="center", linespacing=1.5)

    for rr, lab in ((0.5, "0.5"), (0.7, "0.7"), (0.85, "0.85")):
        v = 1 / np.sqrt(1 - rr)
        axR.plot([rr], [v], marker="o", ms=7, color=BLUE, zorder=4)
        axR.text(rr - 0.03, v * 1.06, rf"$\rho={lab}$ : {v:.2f}배",
                 fontsize=10.5, color=INK, ha="right", va="bottom")
    v_ex = 1 / np.sqrt(1 - rho)
    axR.plot([rho], [v_ex], marker="o", ms=9, color=GREEN, zorder=5)
    axR.annotate(f"위 예의 $\\rho={rho:.3f}$ : {v_ex:.1f}배",
                 xy=(rho, v_ex), xytext=(0.30, 19.0),
                 fontsize=10.5, color=GREEN, ha="left", va="center",
                 arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.2,
                                 shrinkA=2, shrinkB=5))

    axR.set_yscale("log")
    axR.set_yticks([0.8, 1, 2, 5, 10, 25])
    axR.set_yticklabels(["0.8", "1", "2", "5", "10", "25"])
    axR.minorticks_off()
    axR.set_xlim(-0.52, 1.02)
    axR.set_ylim(0.70, 33)
    axR.set_xticks([-0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0])
    axR.set_xticklabels(["-0.5", "-0.25", "0", "0.25", "0.5", "0.75", "1"])
    axR.set_xlabel(r"짝 내 상관 $\rho$", fontsize=10.5, color=INK)
    axR.set_ylabel("독립 분석의 표준오차 ÷ 대응 분석의 표준오차",
                   fontsize=10.5, color=INK)
    clean_axis(axR)
    axR.set_title(r"짝짓기로 줄어드는 양은 오직 $\rho$ 가 정한다",
                  fontsize=12, color=INK, pad=10)

    fig.tight_layout(w_pad=2.6)
    save(fig, "pairing_gain.png")


# === 그림 2. 일치 짝은 사라진다 (ci_paired_proportions.md) =======
def fig_mcnemar_discordant():
    a, b, c, d = 20, 50, 30, 100
    n = a + b + c + d
    z = 1.959963985

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(12.8, 4.8), gridspec_kw={"width_ratios": [1.0, 1.15]}
    )

    # --- 왼쪽: 2x2 표 ---
    cells = [
        (0, 1, a, "$a=20$", MUTED, "#EFF1F2", "일치"),
        (1, 1, b, "$b=50$", BLUE, BLUE_F, "불일치"),
        (0, 0, c, "$c=30$", ORANGE, ORANGE_F, "불일치"),
        (1, 0, d, "$d=100$", MUTED, "#EFF1F2", "일치"),
    ]
    for cx, cy, val, lab, col, fill, kind in cells:
        axL.add_patch(Rectangle((cx, cy), 1, 1, facecolor=fill,
                                edgecolor=col, lw=1.6, zorder=1))
        axL.text(cx + 0.5, cy + 0.62, lab, fontsize=15, color=col,
                 ha="center", va="center")
        axL.text(cx + 0.5, cy + 0.32, kind, fontsize=10.5, color=col,
                 ha="center", va="center")

    axL.text(-0.07, 1.5, "치료 전\n완화", fontsize=10.5, color=INK,
             ha="right", va="center", linespacing=1.5)
    axL.text(-0.07, 0.5, "치료 전\n미완화", fontsize=10.5, color=INK,
             ha="right", va="center", linespacing=1.5)
    axL.text(0.5, 2.09, "치료 후 완화", fontsize=10.5, color=INK,
             ha="center", va="center")
    axL.text(1.5, 2.09, "치료 후 미완화", fontsize=10.5, color=INK,
             ha="center", va="center")

    axL.text(1.0, -0.30,
             r"$\hat p_1-\hat p_2=\dfrac{b-c}{n}=\dfrac{50-30}{200}=0.10$",
             fontsize=13, color=INK, ha="center", va="center")
    axL.text(1.0, -0.72, f"일치 짝 {a + d}쌍은 식 어디에도 없다",
             fontsize=11, color=MUTED, ha="center", va="center")

    axL.set_xlim(-0.95, 2.1)
    axL.set_ylim(-0.95, 2.35)
    axL.axis("off")
    axL.set_title("네 칸 중 두 칸만 계산에 들어간다",
                  fontsize=12, color=INK, pad=10)

    # --- 오른쪽: 차이가 같아도 불일치가 많으면 넓어진다 ---
    cases = [(20, 0), (30, 10), (50, 30), (70, 50), (110, 90)]
    for i, (bb, cc) in enumerate(cases):
        m = bb + cc
        diff = (bb - cc) / n
        se = np.sqrt(m - (bb - cc) ** 2 / n) / n
        lo, hi = diff - z * se, diff + z * se
        y = len(cases) - 1 - i
        col = BLUE if lo > 0 else RED
        axR.plot([lo, hi], [y, y], color=col, lw=2.2, zorder=3)
        for e in (lo, hi):
            axR.plot([e, e], [y - 0.13, y + 0.13], color=col, lw=2.2, zorder=3)
        axR.plot([diff], [y], marker="o", ms=6, color=col, zorder=4)
        axR.text(-0.255, y, f"$b={bb},\\ c={cc}$", fontsize=10.5, color=INK,
                 ha="left", va="center")
        axR.text(-0.126, y, f"불일치 {m}", fontsize=10.5, color=MUTED,
                 ha="left", va="center")
        axR.text(0.30, y, f"±{z * se:.4f}", fontsize=10.5, color=col,
                 ha="left", va="center")
        print(f"  b={bb:3d} c={cc:3d} m={m:3d} ({lo:+.4f}, {hi:+.4f})")

    axR.plot([0, 0], [-0.55, 4.45], color=INK, lw=1.4, zorder=1)
    axR.text(0.0, -0.85, "차이 0", fontsize=10.5, color=INK,
             ha="center", va="center")
    axR.plot([0.10, 0.10], [-0.55, 4.45], color=MUTED, lw=1.1,
             ls=(0, (5, 4)), zorder=1)
    axR.text(0.10, 4.66, r"점추정값 $\hat p_1-\hat p_2=0.10$ (다섯 줄 모두)",
             fontsize=10.5, color=MUTED, ha="center", va="center")

    axR.set_xlim(-0.26, 0.40)
    axR.set_ylim(-1.05, 5.0)
    axR.set_yticks([])
    axR.set_xticks([-0.1, 0.0, 0.1, 0.2, 0.3])
    axR.set_xlabel(r"$p_1-p_2$", fontsize=10.5, color=INK)
    axR.spines[["top", "right", "left"]].set_visible(False)
    axR.spines["bottom"].set_color(MUTED)
    axR.tick_params(axis="x", labelsize=10, colors=INK, length=3)
    axR.set_title(r"$n=200$, 점추정값은 모두 0.10 — 그런데 결론이 갈린다",
                  fontsize=12, color=INK, pad=22)

    fig.tight_layout(w_pad=2.4)
    save(fig, "mcnemar_discordant.png")


if __name__ == "__main__":
    fig_pairing_gain()
    fig_mcnemar_discordant()
