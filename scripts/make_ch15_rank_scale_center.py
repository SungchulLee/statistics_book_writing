r"""15.5절 「그림으로 보는 통계학」 — 순위 산포검정이 중심에 걸려 넘어진다.

플리그너–킬린보다 먼저 나온 순위 산포검정들(무드 1954, 안사리–브래들리 1960,
시겔–튜키 1960, 클로츠 1962)은 **두 표본을 합쳐 순위를 매긴 뒤 가운데에서 바깥으로
점수**를 준다. 퍼진 쪽이 바깥 순위를 더 많이 차지한다는 것이 착상이다.

그 점수가 **합친 자료의 가운데**를 기준으로 매겨진다는 것이 약점이다. 두 분포의
중심이 어긋나면 한 표본이 통째로 한쪽 끝으로 밀려가고, 그 순간 순위는 산포가 아니라
위치를 읽는다. 고장은 두 방향으로 난다.

  ① 표본 크기가 다르면  — 분산이 같아도 기각한다 (거짓 양성)
  ② 표본 크기가 같으면  — 분산이 달라도 못 잡는다 (검정력 붕괴)

②가 특히 깔끔하다. 두 표본이 완전히 갈라지면 안사리–브래들리 점수합이 두 집단에서
**정확히 같아져** 통계량이 귀무 기댓값에 붙는다. 검정이 눈을 감는 것이다.

1976 년 플리그너와 킬린이 한 일은 점수를 새로 만든 것이 아니라 **기준을 옮긴**
것이다 — 합친 자료의 가운데가 아니라 각 집단의 중앙값에서 잰 |편차|에 순위를 매긴다.

만드는 파일:

  ch15/robust_tests/img/rank_scale_center.png

실행:  python3 scripts/make_ch15_rank_scale_center.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다. 몇 분 걸린다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import warnings

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch15/robust_tests/img/"

INK = "#37474F"
AMBER = "#B07C2C"
TEAL = "#00838F"
RED = "#D32F2F"
BLUE = "#1565C0"
MUTED = "#90A4AE"

SEED = 1976
REPS = 6_000
DELTAS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
NAMES = ["안사리–브래들리 ($1960$)", "무드 ($1954$)",
         "플리그너–킬린 ($1976$)", "브라운–포사이드"]


def rates(n1, n2, sigma2, delta, rng):
    """네 검정의 기각률 — x ~ N(0,1), y ~ N(delta, sigma2^2)."""
    hit = np.zeros(4)
    for _ in range(REPS):
        x = rng.normal(0, 1, n1)
        y = rng.normal(delta, sigma2, n2)
        hit += [stats.ansari(x, y).pvalue < 0.05,
                stats.mood(x, y).pvalue < 0.05,
                stats.fligner(x, y).pvalue < 0.05,
                stats.levene(x, y, center="median").pvalue < 0.05]
    return hit / REPS


def panel(ax, data, title, ylab, note):
    for row, (nm, col, mk) in enumerate(zip(
            NAMES, [AMBER, RED, TEAL, BLUE], ["o", "s", "D", "^"])):
        ax.plot(DELTAS, data[:, row] * 100, marker=mk, ms=7, lw=2.5,
                color=col, label=nm)
    ax.set_xlabel("두 집단의 중심 차이 $\\delta$  (표준편차 단위)", fontsize=11.5)
    ax.set_ylabel(ylab, fontsize=11.5)
    ax.set_title(title, fontsize=12.8, color=INK)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.02, -0.235, note, transform=ax.transAxes, fontsize=10.3,
            color=INK)


def main():
    rng = np.random.default_rng(SEED)

    # 한글은 고정폭에서 두 칸을 차지하므로 f-string 자리맞춤이 보이는 열과 어긋난다
    print("① 분산이 같은데 중심만 어긋날 때 — n = (10, 30)")
    print("   델타    안사리      무드        FK        BF")
    a = []
    for d in DELTAS:
        r = rates(10, 30, 1.0, d, rng)
        a.append(r)
        print(f"{d:>7.1f}" + "".join(f"{v:>10.4f}" for v in r))
    a = np.array(a)

    print("\n② 분산이 네 배인데 중심이 어긋날 때 — n = (20, 20), 시그마_y = 2")
    print("   델타    안사리      무드        FK        BF")
    b = []
    for d in DELTAS:
        r = rates(20, 20, 2.0, d, rng)
        b.append(r)
        print(f"{d:>7.1f}" + "".join(f"{v:>10.4f}" for v in r))
    b = np.array(b)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.1), constrained_layout=True)

    panel(axes[0], a, "① 분산이 같은데 중심만 어긋날 때",
          "거짓 양성률 (%)",
          "집단 크기 $n = (10,\\ 30)$ · 두 분산이 같으므로 기각은 모두 거짓이다")
    axes[0].axhline(5, color=MUTED, lw=1.6, ls="--")
    axes[0].annotate(f"${a[-1,0]*100:.0f}\\%$", xy=(3, a[-1, 0] * 100),
                     xytext=(2.3, a[-1, 0] * 100 - 13), fontsize=11.5, color=RED,
                     arrowprops=dict(arrowstyle="->", color=RED, lw=1.3))
    axes[0].annotate("명목 $5\\%$", xy=(0.1, 5), xytext=(0.1, 13),
                     fontsize=10.8, color=INK,
                     arrowprops=dict(arrowstyle="->", color=INK, lw=1.1))
    axes[0].legend(fontsize=10.2, framealpha=0.93, loc="center left")

    panel(axes[1], b, "② 분산이 네 배인데 중심이 어긋날 때",
          "검정력 (%)",
          "집단 크기 $n = (20,\\ 20)$ · $\\sigma_y = 2\\sigma_x$ 이므로 기각이 옳다")
    axes[1].annotate(f"${b[-1,0]*100:.1f}\\%$", xy=(3, b[-1, 0] * 100),
                     xytext=(2.25, b[-1, 0] * 100 + 13), fontsize=11.5, color=RED,
                     arrowprops=dict(arrowstyle="->", color=RED, lw=1.3))
    axes[1].legend(fontsize=10.2, framealpha=0.93, loc="center left")

    fig.suptitle("합친 순위로 산포를 재면 중심이 끼어든다", fontsize=14.5)
    fig.text(0.5, -0.10,
             f"정규자료 · 각 점이 {REPS:,} 번이다. 네 검정 모두 명목 $5\\%$ 이고 "
             "$\\delta = 0$ 에서는 넷이 모두 제자리에 있다.",
             ha="center", fontsize=10.4, color=INK)

    path = OUT + "rank_scale_center.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
