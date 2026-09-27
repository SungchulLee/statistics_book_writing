r"""9.4 '키/몸무게 가설검정' 쪽의 결과 요약 그림을 생성한다.

  ch09/two_sample_tests/img/height_results_summary.png
      왼쪽 : 개인 수준의 산포 — 남녀 키 분포가 얼마나 겹치는가
      오른쪽: 평균 수준의 결론 — 두 검정의 효과크기와 95% 신뢰구간

자료 출처:
  새로 자료를 만들거나 내려받지 않는다. 그림에 적힌 모든 수치는
  docs/ch09/two_sample_tests/height_weight_hypothesis_test.md 가
  본문과 코드 출력에 이미 보고한 값을 그대로 옮긴 것이다.

      일표본 t (예제 1, 연습문제 10)
          x-bar = 169.98,  s = 7.73,  SE = 0.49,  mu0 = 172
          t = -4.131,  p = 4.93e-05,  Cohen d = -0.261
          95% CI (169.02, 170.94)  ->  평균 - 172 로 옮기면 (-2.98, -1.06)
      이표본 Welch t (예제 3, 연습문제 10)
          차이 5.60 cm,  95% CI (4.29, 6.91)
          Welch t = 8.377,  df = 488.2,  p = 5.79e-16
          Cohen d = 0.749,  합동 SD 7.47 cm
          임의의 여성이 임의의 남성보다 클 확률 0.298

  왼쪽 그림의 두 곡선은 위 값 가운데 합동 SD 7.47 cm 와 차이 5.60 cm
  만으로 그린다. 두 집단의 표본평균 자체는 그 쪽이 보고하지 않으므로
  가로축을 '여성 평균 = 0' 인 상대 척도로 잡는다.

실행:  python3 scripts/make_ch09_final_heightweight.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       그림은 PNG로 커밋되므로 CI에서 다시 그리지 않는다.
"""

import numpy as np
from scipy import stats

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

OUT = "docs/ch09/two_sample_tests/img/"

# === 그 쪽이 보고한 수치 (하나도 새로 계산하지 않는다) ===
SD_POOL = 7.47        # 합동 SD (cm)
DIFF = 5.60           # 남 - 여 (cm)
P_TALLER = 0.298      # 임의의 여성이 임의의 남성보다 클 확률

ONE = dict(label="일표본\n평균 - 172 cm",
           est=-2.02, lo=-2.98, hi=-1.06,
           stat="$t = {-4.13}$,   $d = {-0.26}$\n"
                r"$p = 4.9 \times 10^{-5}$")
TWO = dict(label="이표본\n남 - 여",
           est=5.60, lo=4.29, hi=6.91,
           stat="Welch $t = 8.38$,   $d = 0.75$\n"
                r"$p = 5.8 \times 10^{-16}$")


def save(fig, path):
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def bare_axis(ax):
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=10, colors=INK, length=3)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)


# ==================================================================
# 개인의 산포와 평균의 정밀도를 나란히 놓는다
# ==================================================================
def height_results_summary():
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.9))

    # ---------- 왼쪽: 개인 수준 ----------
    ax = axes[0]
    x = np.linspace(-26, 32, 800)
    f = stats.norm.pdf(x, 0.0, SD_POOL)
    m = stats.norm.pdf(x, DIFF, SD_POOL)

    ax.fill_between(x, f, color=ORANGE_F, alpha=0.75, zorder=1)
    ax.fill_between(x, m, color=BLUE_F, alpha=0.65, zorder=2)
    ax.plot(x, f, color=ORANGE, linewidth=2.4, zorder=4, label="여성")
    ax.plot(x, m, color=BLUE, linewidth=2.4, zorder=4,
            label="남성  (평균이 5.60 cm 높다)")

    peak = stats.norm.pdf(0.0, 0.0, SD_POOL)
    for mu, c in [(0.0, ORANGE), (DIFF, BLUE)]:
        ax.plot([mu, mu], [0, peak], color=c, linewidth=1.3,
                linestyle=(0, (4, 3)), zorder=5)

    y_arrow = peak * 1.13
    ax.annotate("", xy=(DIFF, y_arrow), xytext=(0.0, y_arrow),
                arrowprops=dict(arrowstyle="<->", color=INK, linewidth=1.3))
    ax.text(DIFF / 2, y_arrow * 1.045, "5.60 cm", fontsize=11, color=INK,
            ha="center", va="bottom")

    ax.text(-25.2, peak * 1.30,
            f"임의의 여성이 임의의 남성보다\n클 확률은 여전히 {P_TALLER:.3f}",
            fontsize=11, color=INK, ha="left", va="top")
    ax.text(31, peak * 0.60, "Cohen $d = 0.749$\n합동 SD 7.47 cm",
            fontsize=11, color=INK, ha="right", va="top")

    ax.set_xlim(-26, 32)
    ax.set_ylim(0, peak * 1.40)
    ax.set_xlabel("키 (cm) — 여성 평균을 0 으로 잡은 상대 척도",
                  fontsize=11.5, color=INK)
    bare_axis(ax)
    ax.legend(fontsize=10.5, frameon=False, loc="upper right",
              bbox_to_anchor=(1.0, 1.0))
    ax.set_title("개인은 크게 겹친다", fontsize=13, pad=12, color=INK)

    # ---------- 오른쪽: 평균 수준 ----------
    ax = axes[1]
    XLO, XHI = -9.0, 11.0
    # 통계량은 각 행에서 비어 있는 반대쪽에 적어 영점선을 가로지르지 않게 한다
    rows = [(TWO, 1.0, BLUE, XLO + 0.4, "left", 5.60, "center"),
            (ONE, 0.0, ORANGE, XHI - 0.4, "right", -0.5, "right")]

    ax.axvline(0, color=MUTED, linewidth=1.4, linestyle=(0, (5, 4)), zorder=1)
    ax.text(0.35, 1.60, "차이 0  ($H_0$)", fontsize=11, color=INK,
            ha="left", va="bottom")

    for row, y, color, x_stat, ha_stat, x_val, ha_val in rows:
        ax.plot([row["lo"], row["hi"]], [y, y], color=color, linewidth=3.0,
                solid_capstyle="round", zorder=3)
        for b in (row["lo"], row["hi"]):
            ax.plot([b, b], [y - 0.11, y + 0.11], color=color, linewidth=2.2,
                    zorder=3)
        ax.plot([row["est"]], [y], marker="o", markersize=9, color=color,
                zorder=4)
        ax.text(x_val, y + 0.20,
                f"{row['est']:.2f} cm   [{row['lo']:.2f}, {row['hi']:.2f}]",
                fontsize=11.5, color=color, ha=ha_val, va="bottom")
        ax.text(x_stat, y, row["stat"], fontsize=10.5, color=INK,
                ha=ha_stat, va="center", linespacing=1.6)

    ax.set_yticks([r[1] for r in rows])
    ax.set_yticklabels([r[0]["label"] for r in rows], fontsize=11, color=INK)
    ax.set_ylim(-0.75, 1.85)
    ax.set_xlim(XLO, XHI)
    ax.set_xlabel("cm 로 잰 차이와 95% 신뢰구간", fontsize=11.5, color=INK)
    ax.tick_params(axis="x", labelsize=10, colors=INK, length=3)
    ax.tick_params(axis="y", length=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.set_title("평균의 차이는 0 에서 멀다", fontsize=13, pad=12, color=INK)

    fig.text(0.5, -0.045,
             "모든 수치는 이 쪽이 이미 보고한 값을 그대로 옮긴 것이다.",
             fontsize=9.5, color=MUTED, ha="center")
    fig.subplots_adjust(wspace=0.28)
    save(fig, OUT + "height_results_summary.png")


# ==================================================================
if __name__ == "__main__":
    height_results_summary()
