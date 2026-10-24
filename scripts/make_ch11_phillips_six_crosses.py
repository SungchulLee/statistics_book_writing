r"""11.3절 「그림으로 보는 통계학」 — 필립스 곡선은 여섯 점에 그어졌다.

$1958$ 년 필립스가 『Economica』에 실은 그림 1 은 $1861$–$1913$ 년의 해마다 한 점,
곧 쉰세 점을 찍은 산점도다. 그런데 거기 그려진 곡선은 그 쉰세 점에 맞춘 것이 아니다.
실업률이 $0$–$2$, $2$–$3$, $3$–$4$, $4$–$5$, $5$–$7$, $7$–$11$ 퍼센트에 든 해들끼리
묶어 평균을 낸 **여섯 점**(그의 그림에서 ✕ 표)에 맞춘 것이고, 그나마도

  b, c   $0$–$5$ 퍼센트 구간의 평균점 **넷**에 최소제곱
  a      나머지 두 평균점에 **시행착오**로

라는 두 단계였다. 논문이 적은 결과는 $y + 0.900 = 9.638\,x^{-1.394}$ 이다.

이 스크립트는 둘을 그린다.

  ① 발표된 그 식의 곡선과, 그것을 정한 여섯 구간. 어느 네 구간이 최소제곱을
     받았고 어느 두 구간이 시행착오를 받았는지 색으로 가른다
  ② 구간 평균이 흩어짐을 지우는 것을 **되살려 본 것.** 필립스 자신이 적은 두
     가지(경기순환 6.5 회 · 평균 주기 8 년쯤, 그리고 실업이 줄는 해의 임금상승이
     그 수준의 평균보다 높고 느는 해는 낮다)를 넣어 쉰세 해를 만들고, 날 점과
     구간 평균 여섯을 나란히 놓는다. 곡선에서 잰 RMSE 가 평균점 쪽에서 몇 분의
     일로 줄어드는지가 이 판의 수다

만드는 파일:

  ch11/assumptions/img/phillips_six_crosses.png

실행:  python3 scripts/make_ch13_phillips_six_crosses.py   (저장소 최상위에서)
필요:  numpy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch11/assumptions/img/"

INK = "#37474F"
AMBER = "#B07C2C"
TEAL = "#00838F"
RED = "#D32F2F"
BLUE = "#1565C0"
MUTED = "#90A4AE"

# 논문이 적은 적합 결과:  y + 0.900 = 9.638 x^(-1.394)
A, B, C = 0.900, 9.638, -1.394
EDGES = (0.0, 2.0, 3.0, 4.0, 5.0, 7.0, 11.0)   # 논문의 여섯 구간 (위쪽 끝 포함)

YEARS = 53          # 1861-1913
CYCLES = 6.5        # 「6.5 회의 꽤 규칙적인 경기순환」
SEED = 1958


def curve(x):
    return B * x ** C - A


def make_years(rng):
    """필립스가 적은 두 가지를 넣어 쉰세 해를 되살린다.

    ① 경기순환이 6.5 회, 평균 주기 8 년쯤
    ② 같은 실업 수준에서, 실업이 **줄는** 해의 임금상승률은 그 수준의 평균보다
       높고 **느는** 해는 낮다 — 그래서 점들이 고리를 그린다
    """
    t = np.arange(YEARS, dtype=float)
    phase = 2 * np.pi * CYCLES * t / YEARS
    # 순환의 폭이 회마다 달라 어떤 회는 10 퍼센트를 넘는다
    amp = 3.1 + 2.3 * np.sin(2 * np.pi * t / YEARS + 0.7)
    u = 4.7 + amp * np.sin(phase) + rng.normal(0, 0.22, YEARS)
    u = np.clip(u, 0.9, 10.9)

    du = np.gradient(u)                      # 느는 해 > 0, 줄는 해 < 0
    y = curve(u) - 1.15 * du + rng.normal(0, 0.55, YEARS)
    return u, y, du


def bin_means(u, y):
    xs, ys, ns = [], [], []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        m = (u > lo) & (u <= hi)
        if m.sum():
            xs.append(u[m].mean()); ys.append(y[m].mean()); ns.append(int(m.sum()))
    return np.array(xs), np.array(ys), ns


def main():
    rng = np.random.default_rng(SEED)
    u, y, du = make_years(rng)
    cx, cy, cn = bin_means(u, y)

    rmse_raw = float(np.sqrt(np.mean((y - curve(u)) ** 2)))
    rmse_bin = float(np.sqrt(np.mean((cy - curve(cx)) ** 2)))

    # 한글은 고정폭에서 두 칸을 차지하므로 자리맞춤은 공백을 직접 적는다
    print(f"발표된 식   y + {A} = {B} x^({C})")
    print("   x = 1.5, 2.5, 3.5, 4.5, 6, 9 에서 곡선값 "
          + ", ".join(f"{curve(x):+.2f}" for x in (1.5, 2.5, 3.5, 4.5, 6.0, 9.0)))
    print(f"\n되살린 쉰세 해 — 실업률 {u.min():.2f}~{u.max():.2f}%, "
          f"임금변화 {y.min():+.2f}~{y.max():+.2f}%")
    print("  구간      해의 수   평균 실업률   평균 임금변화   곡선값")
    for (lo, hi), x_, y_, n_ in zip(zip(EDGES[:-1], EDGES[1:]), cx, cy, cn):
        print(f"  {lo:4.0f}-{hi:<4.0f}  {n_:5d}      {x_:8.2f}      "
              f"{y_:+9.2f}     {curve(x_):+7.2f}")
    print(f"\n곡선에서 잰 RMSE — 날 점 {rmse_raw:.3f}  구간 평균 {rmse_bin:.3f}  "
          f"(평균이 {rmse_raw / rmse_bin:.1f} 분의 일로 줄인다)")

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.9), constrained_layout=True)

    # ── ① 발표된 곡선과 그것을 정한 여섯 구간 ──
    ax = axes[0]
    xs = np.linspace(1.0, 11.4, 400)
    ax.axhline(0, color=MUTED, lw=1.0, ls=":")
    ax.axvspan(0.0, 5.0, color=TEAL, alpha=0.10)
    ax.axvspan(5.0, 11.0, color=AMBER, alpha=0.13)
    for e in EDGES[1:-1]:
        ax.axvline(e, color=MUTED, lw=1.0, ls="--", alpha=0.8)
    ax.plot(xs, curve(xs), color=INK, lw=3.0)

    ax.text(2.5, 9.0, "평균점 넷에 최소제곱 — $b,\\ c$", fontsize=11.5, color=TEAL,
            ha="center", fontweight="bold")
    ax.text(8.0, 4.2, "남은 평균점 둘에\n시행착오 — $a$", fontsize=11.5, color=AMBER,
            ha="center", fontweight="bold")
    ax.annotate("$y + 0.900 = 9.638\\,x^{-1.394}$", xy=(4.0, curve(4.0)),
                xytext=(5.4, 7.6), fontsize=12.2, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.3))
    ax.set_xlim(0, 11.4)
    ax.set_ylim(-3.2, 11.0)       # 두 판의 세로 눈금을 맞춘다
    ax.set_xticks([0, 2, 3, 4, 5, 7, 11])
    ax.set_xlabel("실업률 ($\\%$)", fontsize=11.5)
    ax.set_ylabel("임금상승률 ($\\%$/년)", fontsize=11.5)
    ax.set_title("① 발표된 곡선과 그것을 정한 여섯 구간", fontsize=12.8, color=INK)
    ax.grid(alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)

    # ── ② 구간 평균이 흩어짐을 지운다 ──
    ax = axes[1]
    ax.axhline(0, color=MUTED, lw=1.0, ls=":")
    ax.plot(xs, curve(xs), color=INK, lw=2.4, zorder=2,
            label="①의 그 곡선")
    up, down = du > 0, du <= 0
    ax.plot(u[down], y[down], "o", ms=7, color=BLUE, alpha=0.75, zorder=3,
            label="실업이 줄던 해")
    ax.plot(u[up], y[up], "o", ms=7, mfc="white", mec=BLUE, mew=1.6, zorder=3,
            label="실업이 늘던 해")
    ax.plot(cx, cy, "X", ms=15, color=RED, zorder=5, mec="white", mew=1.2,
            label="여섯 구간의 평균")
    for e in EDGES[1:-1]:
        ax.axvline(e, color=MUTED, lw=0.9, ls="--", alpha=0.6)
    ax.text(0.97, 0.56,
            f"곡선에서 잰 RMSE\n날 점 {rmse_raw:.2f}  →  평균 {rmse_bin:.2f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=11,
            color=RED,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=RED, alpha=0.92))
    ax.set_xlim(0, 11.4)
    ax.set_ylim(-3.2, 11.0)
    ax.set_xticks([0, 2, 3, 4, 5, 7, 11])
    ax.set_xlabel("실업률 ($\\%$)", fontsize=11.5)
    ax.set_title("② 되살려 본 것 — 쉰세 해와 그 평균 여섯", fontsize=12.8,
                 color=INK)
    ax.grid(alpha=0.22)
    ax.legend(fontsize=9.8, framealpha=0.93, loc="upper right",
              bbox_to_anchor=(1.0, 0.98))
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("가장 유명한 곡선은 쉰세 점이 아니라 여섯 점에 그어졌다",
                 fontsize=14.5)
    fig.text(0.5, -0.045,
             "왼쪽 식은 논문이 적은 그대로다. 오른쪽은 논문이 적은 두 가지"
             "(순환 $6.5$ 회 · 줄 때 높고 늘 때 낮음)를 넣어 되살린 것이며 "
             "필립스의 자료가 아니다.",
             ha="center", fontsize=10.4, color=INK)

    os.makedirs(OUT, exist_ok=True)
    path = OUT + "phillips_six_crosses.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
