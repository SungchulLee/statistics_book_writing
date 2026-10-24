r"""11.7절 「그림으로 보는 통계학」 — 척도가 우승자를 고른다.

품목 $200$ 개의 하루 수요를 예측한다고 하자. 절반은 **하루 열 개쯤** 나가는 작은
품목이고 절반은 **하루 천 개쯤** 나가는 큰 품목이다. 예측기 둘이 있다.

  가: 작은 품목을 잘 맞히고 큰 품목에서 크게 틀린다
  나: 작은 품목에서 조금 틀리고 큰 품목을 잘 맞힌다

이 스크립트가 재는 값(씨앗 고정, 품목 200 개):

  척도      가        나     이기는 쪽
  MAPE    0.0769   0.2208      가
  sMAPE   0.0781   0.2646      가
  MAE    37.0843   8.6245      나
  MASE    0.0753   0.0175      나

  작은 품목에서의 평균 절대오차 — 가 0.780   나 4.171
  큰 품목에서의 평균 절대오차   — 가 73.388  나 13.078

**같은 두 예측기, 같은 자료인데 척도를 바꾸면 우승자가 바뀐다.** 상대오차를 쓰는
MAPE·sMAPE 는 분모가 작은 품목의 오차를 무겁게 보고, 절대오차를 기준에 견주는
MASE 는 큰 품목의 오차를 무겁게 본다. 어느 쪽도 틀리지 않았고, 둘은 **다른 물음**에
답한다.

만드는 파일:

  ch11/performance/img/metric_decides_winner.png

실행:  python3 scripts/make_ch13_metric_decides_winner.py   (저장소 최상위에서)
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

OUT = "docs/ch11/performance/img/"

INK = "#37474F"
AMBER = "#B07C2C"
TEAL = "#00838F"
RED = "#D32F2F"
BLUE = "#1565C0"
MUTED = "#90A4AE"

M = 100            # 작은 품목 수 = 큰 품목 수
SEED = 7


def mape(y, f):
    return float(np.mean(np.abs(y - f) / y))


def smape(y, f):
    return float(np.mean(2 * np.abs(y - f) / (np.abs(y) + np.abs(f))))


def mae(y, f):
    return float(np.mean(np.abs(y - f)))


def mase(y, f):
    """소박한 기준은 이 쪽의 관례대로 표본평균 예측이다."""
    return mae(y, f) / mae(y, np.full_like(y, y.mean()))


def main():
    rng = np.random.default_rng(SEED)
    small = rng.normal(10, 1.0, M)          # 하루 열 개쯤 나가는 품목
    big = rng.normal(1000, 60.0, M)         # 하루 천 개쯤 나가는 품목
    y = np.concatenate([small, big])
    is_small = np.arange(2 * M) < M

    # 가 — 작은 품목을 잘 맞히고 큰 품목에서 크게 틀린다
    a = np.concatenate([small + rng.normal(0, 1.0, M),
                        big + rng.normal(0, 100.0, M)])
    # 나 — 작은 품목에서 조금 틀리고 큰 품목을 잘 맞힌다
    b = np.concatenate([small + rng.normal(0, 5.0, M),
                        big + rng.normal(0, 20.0, M)])

    names = ("MAPE", "sMAPE", "MAE", "MASE")
    funcs = (mape, smape, mae, mase)
    va = [f(y, a) for f in funcs]
    vb = [f(y, b) for f in funcs]

    # 한글은 고정폭에서 두 칸을 차지하므로 자리맞춤은 공백을 직접 적는다
    print(f"품목 {2 * M} 개 — 작은 품목 {M} 개(하루 10 개쯤), "
          f"큰 품목 {M} 개(하루 1000 개쯤)")
    print("  척도        가        나   이기는 쪽")
    for nm, x, z in zip(names, va, vb):
        print(f"  {nm:6s} {x:9.4f} {z:9.4f}      {'가' if x < z else '나'}")
    print(f"\n  작은 품목에서의 평균 절대오차 — 가 {mae(small, a[is_small]):.3f}  "
          f"나 {mae(small, b[is_small]):.3f}")
    print(f"  큰 품목에서의 평균 절대오차   — 가 {mae(big, a[~is_small]):.3f}  "
          f"나 {mae(big, b[~is_small]):.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.9), constrained_layout=True)

    # ── ① 오차가 어디에 쌓였는가 ──
    ax = axes[0]
    xs = np.arange(2)
    w = 0.34
    ea = [mae(small, a[is_small]), mae(big, a[~is_small])]
    eb = [mae(small, b[is_small]), mae(big, b[~is_small])]
    ax.bar(xs - w / 2, ea, width=w * 0.92, color=AMBER, alpha=0.9,
           label="예측기 가")
    ax.bar(xs + w / 2, eb, width=w * 0.92, color=TEAL, alpha=0.9,
           label="예측기 나")
    for i, (u, v) in enumerate(zip(ea, eb)):
        ax.text(i - w / 2, u * 1.25, f"${u:.2f}$", ha="center", fontsize=11.2,
                color=AMBER, fontweight="bold")
        ax.text(i + w / 2, v * 1.25, f"${v:.2f}$", ha="center", fontsize=11.2,
                color=TEAL, fontweight="bold")
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"작은 품목 ${M}$ 개\n(하루 $10$ 개쯤)",
                        f"큰 품목 ${M}$ 개\n(하루 $1000$ 개쯤)"], fontsize=11)
    ax.set_ylim(0.3, 500)
    ax.set_ylabel("평균 절대오차 (로그 눈금)", fontsize=11.5)
    ax.set_title("① 둘은 서로 다른 데서 틀린다", fontsize=12.8, color=INK)
    ax.grid(alpha=0.22, axis="y", which="both")
    ax.legend(fontsize=10.4, framealpha=0.93, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)

    # ── ② 척도마다 우승자가 다르다 ──
    ax = axes[1]
    xs = np.arange(4)
    ra = [1.0, 1.0, 1.0, 1.0]
    rb = [z / x for x, z in zip(va, vb)]
    ax.axhline(1.0, color=INK, lw=1.6)
    ax.bar(xs, rb, width=0.5,
           color=[TEAL if v < 1 else AMBER for v in rb], alpha=0.9)
    top = max(rb) * 1.42
    for i, v in enumerate(rb):
        col = TEAL if v < 1 else AMBER
        ax.text(i, v + top * 0.025, f"${v:.2f}$\n"
                f"{'나가 이긴다' if v < 1 else '가가 이긴다'}",
                ha="center", va="bottom", fontsize=11.4, color=col,
                fontweight="bold", linespacing=1.45)
    ax.text(3.54, 1.04, "예측기 가", ha="right", va="bottom", fontsize=10.8,
            color=INK)
    ax.set_xticks(xs)
    ax.set_xticklabels(names, fontsize=11.6)
    ax.set_ylim(0, top)
    ax.set_xlim(-0.6, 3.6)
    ax.set_ylabel("예측기 가를 $1$ 로 둔 나의 값", fontsize=11.5)
    ax.set_title("② 척도를 바꾸면 우승자가 바뀐다", fontsize=12.8, color=INK)
    ax.grid(alpha=0.22, axis="y")
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("어느 쪽이 더 나은 예측기인가 — 물음이 덜 적혔다", fontsize=14.5)
    fig.text(0.5, -0.05,
             f"품목 ${2 * M}$ 개의 하루 수요를 예측한 것이다. "
             f"네 척도는 모두 같은 두 예측값에서 계산했다. "
             f"모든 수는 이 스크립트가 계산한 값이다.",
             ha="center", fontsize=10.4, color=INK)

    os.makedirs(OUT, exist_ok=True)
    path = OUT + "metric_decides_winner.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
