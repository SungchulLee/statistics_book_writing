r"""4.2 지수분포 쪽의 본문 그림 둘을 생성한다.

정의 1 과 정리 1 에 그림이 없어 식만으로 읽어야 했다. 둘 다 눈으로 보면
한 번에 잡히는 내용이다.

  ch04/continuous_distributions/img/exponential_density.png
      정의 1 — 밀도 f(x) = lam * exp(-lam x). 비율모수가 커지면 시작점이
      높아지고 더 빨리 줄어든다. 평균 1/lam 의 자리도 함께 찍는다.

  ch04/continuous_distributions/img/exponential_cdf_survival.png
      정리 1 — 왼쪽: 밀도 아래 넓이가 x0 을 기준으로 F(x0) 과 S(x0) 로
      갈리고 합이 1 이다. 오른쪽: 그 둘을 x 의 함수로 그린 것. 두 곡선은
      중앙값에서 만난다.

실행:  python3 scripts/make_ch04_exponential_figures.py   (저장소 최상위에서)
필요:  numpy, matplotlib — 문서 빌드에는 필요하지 않다.
       그림은 PNG로 커밋되므로 CI에서 다시 그리지 않는다.
"""

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# === 공통 설정 — 다른 장의 그림과 같은 팔레트 ===
plt.rcParams["font.sans-serif"] = ["NanumGothic", "Apple SD Gothic Neo", "Malgun Gothic"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

INK = "#37474F"
BLUE, BLUE_F = "#1565C0", "#DCEBFB"
ORANGE, ORANGE_F = "#E65100", "#FFE0B2"
GREEN = "#33691E"
MUTED = "#90A4AE"

OUT = "docs/ch04/continuous_distributions/img/"


def save(fig, path):
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def tidy(ax):
    ax.tick_params(labelsize=9, colors=INK, length=3)
    ax.spines[["top", "right"]].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)


# ==================================================================
# 정의 1 — 밀도. 비율모수가 모양을 어떻게 바꾸는가
# ==================================================================
def density():
    x = np.linspace(0, 6, 600)
    fig, ax = plt.subplots(figsize=(7.6, 3.4))

    for lam, color in ((0.5, GREEN), (1.0, BLUE), (2.0, ORANGE)):
        ax.plot(x, lam * np.exp(-lam * x), color=color, lw=2,
                label=f"λ = {lam}")
        ax.plot([0], [lam], "o", color=color, ms=5, zorder=3)
        # 평균 1/lam 의 자리에 짧은 눈금을 세운다.
        m = 1 / lam
        ax.plot([m, m], [0, lam * np.exp(-1)], color=color, lw=1,
                ls=(0, (3, 3)), alpha=0.8)
        ax.plot([m], [0], "v", color=color, ms=5, clip_on=False, zorder=3)

    ax.fill_between(x, np.exp(-x), color=BLUE_F, alpha=0.55, zorder=0)
    ax.annotate("x = 0 에서의 높이가 λ 다", xy=(0.05, 1.98), xytext=(1.0, 1.95),
                fontsize=10, color=INK,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1))
    ax.annotate("삼각형이 평균 1/λ 의 자리", xy=(0.5, 0.01), xytext=(1.3, 1.18),
                fontsize=10, color=INK,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1))
    ax.annotate("어느 λ 에서도 밀도 아래 넓이는 1", xy=(3.0, 0.03),
                xytext=(3.0, 0.62), fontsize=10, color=INK,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1))

    ax.set_xlim(0, 6)
    ax.set_ylim(0, 2.15)
    ax.set_xlabel("x", fontsize=10, color=INK)
    ax.set_ylabel("f(x)", fontsize=10, color=INK)
    ax.set_title("지수분포의 밀도 — 비율모수가 클수록 가파르게 시작해 빨리 줄어든다",
                 fontsize=11, color=INK, pad=10)
    ax.legend(frameon=False, fontsize=10, labelcolor=INK)
    tidy(ax)
    save(fig, OUT + "exponential_density.png")


# ==================================================================
# 정리 1 — 넓이가 F 와 S 로 갈린다
# ==================================================================
def cdf_survival():
    lam = 1.0
    x0 = 1.2
    x = np.linspace(0, 5, 600)
    f = lam * np.exp(-lam * x)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.4))

    # --- 왼쪽: 밀도 아래 넓이의 분할 ---
    ax = axes[0]
    left, right = x <= x0, x >= x0
    ax.fill_between(x[left], f[left], color=BLUE_F, zorder=0)
    ax.fill_between(x[right], f[right], color=ORANGE_F, zorder=0)
    ax.plot(x, f, color=INK, lw=1.8)
    ax.plot([x0, x0], [0, lam * np.exp(-lam * x0)], color=MUTED, lw=1,
            ls=(0, (3, 3)))

    ax.text(0.45, 0.22, "$F(x_0)$", fontsize=13, color=BLUE, ha="center")
    ax.text(2.1, 0.09, "$S(x_0)$", fontsize=13, color=ORANGE, ha="center")
    ax.text(x0, -0.085, "$x_0$", fontsize=11, color=INK, ha="center")
    ax.set_title("밀도 아래 넓이가 둘로 갈린다. 합은 1 이다",
                 fontsize=11, color=INK, pad=10)
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("x", fontsize=10, color=INK)
    ax.set_xticks([0, 1, 2, 3, 4, 5])
    tidy(ax)

    # --- 오른쪽: 두 함수를 x 의 함수로 ---
    ax = axes[1]
    F, S = 1 - np.exp(-lam * x), np.exp(-lam * x)
    ax.plot(x, F, color=BLUE, lw=2, label=r"$F(x) = 1 - e^{-\lambda x}$")
    ax.plot(x, S, color=ORANGE, lw=2, label=r"$S(x) = e^{-\lambda x}$")
    ax.axhline(0.5, color=MUTED, lw=1, ls=(0, (3, 3)))

    med = np.log(2) / lam
    ax.plot([med], [0.5], "o", color=INK, ms=5, zorder=3)
    ax.annotate("두 곡선이 만나는 자리가\n중앙값 (ln 2)/λ 다",
                xy=(med, 0.5), xytext=(2.15, 0.66), fontsize=10, color=INK,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1))

    ax.set_title("합이 1 이라 두 곡선은 서로를 뒤집은 꼴이다",
                 fontsize=11, color=INK, pad=10)
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("x", fontsize=10, color=INK)
    ax.set_xticks([0, 1, 2, 3, 4, 5])
    ax.legend(frameon=False, fontsize=10, labelcolor=INK, loc="center right")
    tidy(ax)

    fig.tight_layout()
    save(fig, OUT + "exponential_cdf_survival.png")


if __name__ == "__main__":
    density()
    cdf_survival()
