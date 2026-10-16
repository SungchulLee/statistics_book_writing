r"""5.1절 — 좋은 추정량에 바라는 것 둘을 과녁 넷으로.

불편성(탄착군의 중심이 과녁 한가운데인가)과 정밀도(탄착군의 폭이 좁은가)는
서로 다른 요구이며 어느 한쪽이 다른 쪽을 함의하지 않는다. 그래서 네 조합이
모두 실제로 일어난다.

  왼쪽 위   불편 + 좁다  → 쓸 만하다
  왼쪽 아래 불편 + 넓다  → 불편이지만 한 발은 멀리 간다
  오른쪽 위 치우침 + 좁다 → 틀린 곳에 자신 있게 모인다
  오른쪽 아래 치우침 + 넓다 → 둘 다 나쁘다

탄착점은 과녁 중심 (0, 0) 을 모수 theta 로 두고 이변량 정규로 뿌린다.
치우친 쪽은 중심을 BIAS 만큼 옮기고, 넓은 쪽은 표준편차를 키운다.

만드는 파일:

  ch05/foundations/img/unbiased_vs_precise.png

실행:  python3 scripts/make_ch05_unbiased_vs_precise.py   (저장소 최상위에서)
필요:  numpy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch05/foundations/img/"

INK = "#37474F"
BLUE, BLUE_F = "#1565C0", "#DCEBFB"
RED = "#D32F2F"
GREEN = "#33691E"
MUTED = "#90A4AE"

SEED = 2026
SHOTS = 40                 # 과녁마다 찍는 추정값의 개수
TIGHT, WIDE = 0.30, 0.78   # 탄착군의 표준편차
BIAS = 0.95                # 치우친 쪽이 중심에서 밀려난 거리
RINGS = (1.45, 0.95, 0.45)

# (제목, 중심이 맞는가, 폭이 좁은가, 한 줄 평)
PANELS = [
    ("불편 · 폭이 좁다", True, True, "①과 ② 둘 다 — 이것을 바란다"),
    ("치우침 · 폭이 좁다", False, True, "틀린 곳에 모인다"),
    ("불편 · 폭이 넓다", True, False, "불편이지만 한 발은 멀리 간다"),
    ("치우침 · 폭이 넓다", False, False, "둘 다 나쁘다"),
]


def draw_target(ax, rng, title, unbiased, tight, caption):
    """과녁 하나에 탄착점 SHOTS 개를 뿌리고 중심을 표시한다."""
    cx, cy = (0.0, 0.0) if unbiased else (BIAS * 0.72, BIAS * 0.66)
    sd = TIGHT if tight else WIDE

    for r, fc in zip(RINGS, ("#F2F6F9", BLUE_F, "#BBD8F3")):
        ax.add_patch(plt.Circle((0, 0), r, fc=fc, ec=BLUE, lw=1.3, zorder=1))
    ax.plot(0, 0, marker="+", ms=13, mew=2.2, color=RED, zorder=4)

    shots = rng.normal((cx, cy), sd, (SHOTS, 2))
    ax.plot(shots[:, 0], shots[:, 1], linestyle="none", marker="x", ms=6.5,
            mew=1.6, color=GREEN if unbiased else "#8D6E63", zorder=3)

    # 탄착군의 중심 — 불편이면 과녁 중심과 겹친다
    ax.plot(shots[:, 0].mean(), shots[:, 1].mean(), marker="o", ms=9,
            mfc="white", mec=INK, mew=2.0, zorder=5)

    ok = unbiased and tight
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-2.0, 2.0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=12.5, color=INK if ok else MUTED,
                 fontweight="bold" if ok else "normal", pad=6)
    ax.text(0, -1.88, caption, ha="center", fontsize=10.5,
            color=GREEN if ok else MUTED)


def main():
    rng = np.random.default_rng(SEED)

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 9.6), constrained_layout=True)
    for ax, panel in zip(axes.ravel(), PANELS):
        draw_target(ax, rng, *panel)

    fig.suptitle("불편성과 정밀도는 서로 다른 요구다", fontsize=15)
    fig.text(0.5, -0.025,
             "빨간 +는 모수 $\\theta$, ×는 표본마다 나온 추정값, "
             "흰 ●는 그 전부의 평균이다.\n"
             "①은 흰 ●가 빨간 +에 놓이는가를, ②는 ×들이 얼마나 모여 있는가를 묻는다.",
             ha="center", fontsize=11, color=INK)

    path = OUT + "unbiased_vs_precise.png"
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
