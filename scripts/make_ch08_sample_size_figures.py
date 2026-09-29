r"""8장 표본크기 세 쪽의 개념 그림을 생성한다.

만드는 파일:
  ch08/sample_size/img/inverse_square_law.png   오차한계와 표본크기의 역제곱
  ch08/sample_size/img/planning_inputs.png      계획값이 틀리면 얼마나 틀어지나
  ch08/sample_size/img/two_group_design.png     효과크기와 배정비

실행:  python3 scripts/make_ch08_sample_size_figures.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import os

import numpy as np
from scipy.stats import norm

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

OUT = "docs/ch08/sample_size/img/"
os.makedirs(OUT, exist_ok=True)

Z95 = norm.ppf(0.975)


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def clean_axis(ax):
    ax.tick_params(labelsize=10, colors=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)


# === 그림 1. 역제곱 법칙 (margin_of_error.md) ====================
def fig_inverse_square_law():
    sigma = 7.0

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.6, 4.7))

    # --- 왼쪽: n 대 E ---
    E = np.linspace(0.45, 5.0, 800)
    n = (Z95 * sigma / E) ** 2
    axL.plot(E, n, color=BLUE, lw=2.2, zorder=3)

    marks = [4.0, 2.0, 1.0, 0.5]
    for e in marks:
        v = int(np.ceil((Z95 * sigma / e) ** 2))
        axL.plot([e], [v], marker="o", ms=7, color=BLUE, zorder=5)
        print(f"  E={e}: n={v}")

    # 계단: E 를 절반으로 줄일 때마다 n 이 네 배
    for e in (4.0, 2.0, 1.0):
        v0 = (Z95 * sigma / e) ** 2
        v1 = (Z95 * sigma / (e / 2)) ** 2
        axL.plot([e, e / 2], [v0, v0], color=MUTED, lw=1.1, ls=(0, (3, 3)),
                 zorder=1)
        axL.plot([e / 2, e / 2], [v0, v1], color=MUTED, lw=1.1, ls=(0, (3, 3)),
                 zorder=1)
        axL.annotate("", xy=(e / 2 - 0.02, v1), xytext=(e / 2 - 0.02, v0),
                     arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.5))
        axL.text(e / 2 - 0.13, (v0 + v1) / 2, "4배", fontsize=10.5,
                 color=ORANGE, ha="right", va="center")

    labels = {4.0: (0.30, 40), 2.0: (0.24, 95), 1.0: (0.22, 232),
              0.5: (0.20, 700)}
    for e in marks:
        v = int(np.ceil((Z95 * sigma / e) ** 2))
        dx, ly = labels[e]
        axL.text(e + dx, ly, f"$E={e:g}$ 이면 $n={v}$", fontsize=10.5,
                 color=INK, ha="left", va="center")

    axL.set_xlim(0.15, 5.1)
    axL.set_ylim(0, 830)
    axL.set_xlabel("원하는 오차한계 $E$ (cm)", fontsize=10.5, color=INK)
    axL.set_ylabel("필요한 표본크기 $n$", fontsize=10.5, color=INK)
    clean_axis(axL)
    axL.set_title(r"평균의 표본크기 ($\sigma=7$, 95%) — $E$ 를 반으로 줄이면",
                  fontsize=12, color=INK, pad=10)

    # --- 오른쪽: 같은 법칙을 예산 쪽에서 ---
    ns = np.arange(100, 3300, 5)
    Es = Z95 * np.sqrt(0.25 / ns)
    axR.plot(ns, 100 * Es, color=GREEN, lw=2.2, zorder=3)
    for nn in (400, 800, 1600, 3200):
        e = 100 * Z95 * np.sqrt(0.25 / nn)
        axR.plot([nn], [e], marker="o", ms=7, color=GREEN, zorder=5)
        axR.text(nn + 55, e + 0.16, f"$n={nn}$\n$E={e:.2f}$%p", fontsize=10.5,
                 color=INK, ha="left", va="bottom", linespacing=1.5)
        print(f"  n={nn}: E={e:.3f}%p")

    axR.annotate("", xy=(800, 3.46), xytext=(400, 4.90),
                 arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.6))
    axR.annotate("예산을 2배로 써도\n오차한계는 29%만 줄어든다",
                 xy=(600, 4.18), xytext=(1520, 5.25),
                 fontsize=10.5, color=ORANGE, ha="left", va="center",
                 linespacing=1.5,
                 arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2,
                                 shrinkA=4, shrinkB=6))

    axR.set_xlim(150, 3450)
    axR.set_ylim(0.6, 6.2)
    axR.set_xlabel("표본크기 $n$", fontsize=10.5, color=INK)
    axR.set_ylabel("달성되는 오차한계 (퍼센트포인트)", fontsize=10.5, color=INK)
    clean_axis(axR)
    axR.set_title(r"비율의 오차한계 ($p=0.5$, 95%) — 돈을 두 배 써도",
                  fontsize=12, color=INK, pad=10)

    fig.tight_layout(w_pad=2.6)
    save(fig, "inverse_square_law.png")


# === 그림 2. 계획값의 두 약한 고리 (sample_size.md) ==============
def fig_planning_inputs():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.6, 4.7))

    # --- 왼쪽: n 이 계획값 p* 에 어떻게 의존하나 ---
    E = 0.03
    p = np.linspace(0.005, 0.995, 800)
    n = Z95 ** 2 * p * (1 - p) / E ** 2
    axL.plot(p, n, color=BLUE, lw=2.2, zorder=3)
    axL.fill_between(p, 0, n, color=BLUE_F, zorder=0)

    for pp, col in ((0.5, ORANGE), (0.3, MUTED), (0.1, MUTED)):
        v = int(np.ceil(Z95 ** 2 * pp * (1 - pp) / E ** 2))
        axL.plot([pp], [v], marker="o", ms=7, color=col, zorder=5)
        axL.plot([pp, pp], [0, v], color=col, lw=1.1, ls=(0, (3, 3)), zorder=2)
        print(f"  p*={pp}: n={v}")

    axL.text(0.5, 1135, "보수적 선택 $p^*=0.5$ : 1068명", fontsize=10.5,
             color=ORANGE, ha="center", va="bottom")
    axL.text(0.298, 990, "$p^*=0.3$ : 897명", fontsize=10.5, color=INK,
             ha="right", va="center")
    axL.text(0.135, 320, "$p^*=0.1$ : 385명", fontsize=10.5, color=INK,
             ha="left", va="center")
    axL.annotate("", xy=(0.075, 300), xytext=(0.075, 1010),
                 arrowprops=dict(arrowstyle="<->", color=GREEN, lw=1.3))
    axL.text(0.062, 650, "최대 2.8배", fontsize=10.5, color=GREEN,
             ha="right", va="center", rotation=90)

    axL.set_xlim(0, 1)
    axL.set_ylim(0, 1300)
    axL.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    axL.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
    axL.set_xlabel(r"계획값 $p^*$", fontsize=10.5, color=INK)
    axL.set_ylabel("필요한 표본크기 $n$", fontsize=10.5, color=INK)
    clean_axis(axL)
    axL.set_title(r"비율: 계획값을 몰라도 $0.5$ 를 넣으면 안전하다 ($E=0.03$)",
                  fontsize=12, color=INK, pad=10)

    # --- 오른쪽: sigma 를 잘못 잡으면 ---
    sigma_plan, E_target = 15.0, 2.0
    n_plan = int(np.ceil((Z95 * sigma_plan / E_target) ** 2))
    ratio = np.linspace(0.6, 1.6, 400)
    achieved = Z95 * (sigma_plan * ratio) / np.sqrt(n_plan)
    axR.plot(ratio, achieved, color=BLUE, lw=2.2, zorder=3)
    axR.axhline(E_target, color=GREEN, lw=1.4, ls=(0, (5, 4)), zorder=2)
    axR.axvline(1.0, color=MUTED, lw=1.1, ls=(0, (3, 3)), zorder=1)
    axR.fill_between(ratio, E_target, achieved, where=achieved >= E_target,
                     color="#FBE3E3", zorder=0)

    print(f"  n_plan={n_plan}")
    for r in (0.8, 1.2, 1.5):
        v = Z95 * (sigma_plan * r) / np.sqrt(n_plan)
        axR.plot([r], [v], marker="o", ms=7, color=BLUE, zorder=5)
        print(f"  sigma ratio {r}: achieved E={v:.3f}")
    axR.text(1.235, 2.26, r"$\sigma$ 가 20% 크면" + "\n오차한계도 20% 커진다",
             fontsize=10.5, color=INK, ha="left", va="top", linespacing=1.5)
    axR.text(0.825, 1.36, r"$\sigma$ 가 20% 작으면" + "\n표본을 낭비한 것",
             fontsize=10.5, color=MUTED, ha="left", va="center",
             linespacing=1.5)
    axR.text(0.63, 2.05, f"목표 $E={E_target:g}$", fontsize=10.5, color=GREEN,
             ha="left", va="bottom")
    axR.text(1.02, 3.05, f"계획값 $\\sigma=15$ 로\n정한 $n={n_plan}$",
             fontsize=10.5, color=MUTED, ha="left", va="center",
             linespacing=1.5)

    axR.set_xlim(0.58, 1.62)
    axR.set_ylim(1.1, 3.35)
    axR.set_xticks([0.6, 0.8, 1.0, 1.2, 1.4, 1.6])
    axR.set_xlabel(r"참 $\sigma$ $\div$ 계획에 쓴 $\sigma$", fontsize=10.5,
                   color=INK)
    axR.set_ylabel("실제로 달성되는 오차한계", fontsize=10.5, color=INK)
    clean_axis(axR)
    axR.set_title(r"평균: 계획이 무너지는 곳은 언제나 $\sigma$ 다",
                  fontsize=12, color=INK, pad=10)

    fig.tight_layout(w_pad=2.6)
    save(fig, "planning_inputs.png")


# === 그림 3. 두 집단 설계 (two_group_sample_size.md) =============
def fig_two_group_design():
    z_a = norm.ppf(0.975)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.6, 4.7))

    # --- 왼쪽: n 은 표준화 효과크기만의 함수다 ---
    d = np.linspace(0.18, 1.05, 600)
    for beta, col, lab in ((0.20, BLUE, "검정력 80%"),
                           (0.10, PURPLE, "검정력 90%")):
        z_b = norm.ppf(1 - beta)
        n = 2 * (z_a + z_b) ** 2 / d ** 2
        axL.plot(d, n, color=col, lw=2.2, label=lab, zorder=3)

    for dd, col in ((0.2, MUTED), (0.5, GREEN), (0.8, MUTED)):
        n80 = int(np.ceil(2 * (z_a + norm.ppf(0.80)) ** 2 / dd ** 2))
        n90 = int(np.ceil(2 * (z_a + norm.ppf(0.90)) ** 2 / dd ** 2))
        print(f"  d={dd}: n80={n80} n90={n90}")
        axL.plot([dd], [n80], marker="o", ms=6, color=BLUE, zorder=5)
        axL.plot([dd], [n90], marker="o", ms=6, color=PURPLE, zorder=5)

    axL.text(0.225, 500, "작음 $d=0.2$\n80%: 393명\n90%: 526명", fontsize=10.5,
             color=INK, ha="left", va="center", linespacing=1.6)
    axL.text(0.545, 210, "중간 $d=0.5$\n80%: 63명\n90%: 85명", fontsize=10.5,
             color=GREEN, ha="left", va="center", linespacing=1.6)
    axL.text(0.845, 95, "큼 $d=0.8$\n80%: 25명\n90%: 33명", fontsize=10.5,
             color=INK, ha="left", va="center", linespacing=1.6)

    axL.legend(fontsize=10, frameon=False, loc="upper right")
    axL.set_xlim(0.15, 1.10)
    axL.set_ylim(0, 690)
    axL.set_xlabel(r"표준화 효과크기 $d=\delta/\sigma$", fontsize=10.5,
                   color=INK)
    axL.set_ylabel("집단당 표본크기 $n$", fontsize=10.5, color=INK)
    clean_axis(axL)
    axL.set_title(r"$\sigma$ 와 $\delta$ 는 따로 필요 없다 — 비 하나면 된다",
                  fontsize=12, color=INK, pad=10)

    # --- 오른쪽: 배정비 k 와 총 인원 ---
    sigma, delta, beta = 12.0, 5.0, 0.20
    base = (z_a + norm.ppf(1 - beta)) ** 2 * sigma ** 2 / delta ** 2
    ks = np.array([1, 1.5, 2, 2.5, 3, 4, 5], dtype=float)
    n1 = np.ceil((1 + 1 / ks) * base)
    n2 = np.ceil(ks * n1)
    tot = n1 + n2
    xs = np.arange(len(ks))
    axR.bar(xs - 0.19, n1, width=0.36, color=BLUE, alpha=0.35,
            edgecolor=BLUE, lw=1.4, label="처리군 $n_1$")
    axR.bar(xs + 0.19, n2, width=0.36, color=ORANGE, alpha=0.35,
            edgecolor=ORANGE, lw=1.4, label="대조군 $n_2$")
    axR.plot(xs, tot, color=INK, lw=1.8, marker="o", ms=6, zorder=5,
             label="총 인원")
    for i, t in enumerate(tot):
        axR.text(xs[i], t + 22, f"{int(t)}", fontsize=10.5, color=INK,
                 ha="center", va="bottom")
        print(f"  k={ks[i]:.1f}: n1={int(n1[i])} n2={int(n2[i])} tot={int(t)}")

    axR.annotate("1:1 이 총 인원을 최소로 한다",
                 xy=(0.0, tot[0] + 24), xytext=(2.15, 425),
                 fontsize=10.5, color=GREEN, ha="left", va="center",
                 arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.3,
                                 shrinkA=2, shrinkB=6))

    axR.legend(fontsize=10, frameon=False, loc="upper left",
               bbox_to_anchor=(0.0, 1.0))
    axR.set_xticks(xs)
    axR.set_xticklabels([f"1:{k:g}" for k in ks], fontsize=10.5, color=INK)
    axR.set_ylim(0, 560)
    axR.set_xlabel(r"배정비 $n_1:n_2$", fontsize=10.5, color=INK)
    axR.set_ylabel("인원", fontsize=10.5, color=INK)
    clean_axis(axR)
    axR.set_title(r"같은 검정력 ($\delta=5$, $\sigma=12$, 80%)을 사는 여러 방법",
                  fontsize=12, color=INK, pad=10)

    fig.tight_layout(w_pad=2.6)
    save(fig, "two_group_design.png")


if __name__ == "__main__":
    fig_inverse_square_law()
    fig_planning_inputs()
    fig_two_group_design()
