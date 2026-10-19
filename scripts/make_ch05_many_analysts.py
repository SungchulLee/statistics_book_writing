r"""5.1절 「그림으로 보는 통계학」 — 자료는 그대로인데 답이 달라진다.

이 쪽은 5 장 전체의 출발점을 적는다. 「달라지게 만든 것은 모집단이 아니라
누가 표본에 뽑혔는가이다.」 맞는 말이다. 그런데 그것이 **유일한** 흔들림의
원천은 아니다.

2018 년에 실버잔과 동료들이 연구팀 29 개(분석자 61 명)에게 **똑같은 자료**를
주었다. 프로축구 선수들의 자료였고 물음은 하나였다 — 심판이 피부색이
어두운 선수에게 레드카드를 더 주는가. 같은 파일, 같은 물음이다.

답으로 돌아온 오즈비가 0.89 에서 2.93 사이에 퍼져 있었다. 29 팀 가운데
20 팀이 유의한 양의 효과를 찾았고 9 팀은 찾지 못했다. 서로의 분석을
평가하게 했더니 모두 그럴듯하다는 평을 받았다.

표본은 한 번도 다시 뽑히지 않았다. 바뀐 것은 **분석자**다.

이 그림은 그 일을 기계로 다시 만든다. 자료를 딱 한 벌 만들고, 공변량
여섯 개 가운데 무엇을 보정할지만 달리하는 분석자 64 명에게 준다. 그리고
따로, 분석은 하나로 고정한 채 자료를 64 벌 새로 뽑는다.

  같은 자료, 다른 분석자    추정값 -0.007 ~ +0.823   표준편차 0.2336
  새 자료,  같은 분석      추정값 +0.227 ~ +0.432   표준편차 0.0469

분석자 사이가 표본 사이보다 5 배 넓다. 참값은 0.30 이다.

정직하게 덧붙일 것이 있다. 이 모의실험에서는 어느 쪽이 옳은지 **알 수**
있다 — 교란변수를 모두 보정한 쪽이다. 실제 연구에서는 그것을 모른다.
실버잔의 29 팀이 모두 그럴듯했던 까닭이 그것이다.

만드는 파일:

  ch05/foundations/img/many_analysts.png

실행:  python3 scripts/make_ch05_many_analysts.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import itertools

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch05/foundations/img/"

INK = "#37474F"
MUTED = "#90A4AE"
GREY = "#78909C"
RED = "#D32F2F"
TEAL = "#00838F"
AMBER = "#B07C2C"
BLUE = "#1565C0"

N, K, BETA, SEED = 500, 6, 0.30, 2018
GAMMA = np.array([0.8, -0.6, 0.5, 0.4, -0.3, 0.2])


def make(rng):
    """z1~z3 이 x 와 y 양쪽에 걸린 교란변수인 자료 한 벌."""
    z = rng.normal(size=(N, K))
    x = 0.6 * z[:, :3].sum(1) / np.sqrt(3) + np.sqrt(1 - 0.36) * rng.normal(size=N)
    y = BETA * x + z @ GAMMA + rng.normal(size=N)
    return x, z, y


def fit(x, z, y, cols):
    """x 의 계수와 그 표준오차, 양쪽 p 값."""
    parts = [np.ones(N), x] + ([z[:, list(cols)]] if cols else [])
    X = np.column_stack(parts)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b
    s2 = r @ r / (N - X.shape[1])
    se = float(np.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1]))
    t = b[1] / se
    return float(b[1]), se, float(2 * stats.t.sf(abs(t), N - X.shape[1]))


def main():
    rng = np.random.default_rng(SEED)
    x, z, y = make(rng)                         # 자료는 딱 한 번만 만든다

    subsets = [c for r in range(K + 1) for c in itertools.combinations(range(K), r)]
    res = np.array([fit(x, z, y, c) for c in subsets])
    bb, ses, ps = res[:, 0], res[:, 1], res[:, 2]
    full_idx = subsets.index(tuple(range(K)))

    print(f"    분석자 {len(subsets)} 명이 같은 자료를 본다")
    print(f"      추정값 {bb.min():+.3f} ~ {bb.max():+.3f}   표준편차 {bb.std():.4f}")
    print(f"      p < 0.05 인 분석 {int((ps<0.05).sum())}/{len(subsets)}")
    print(f"      교란을 모두 보정한 분석 = {bb[full_idx]:+.3f}   (참값 {BETA})")

    samp = np.array([fit(*make(rng), tuple(range(K)))[0] for _ in subsets])
    print(f"\n    같은 분석을 새 자료 {len(samp)} 벌에 돌린다")
    print(f"      추정값 {samp.min():+.3f} ~ {samp.max():+.3f}   "
          f"표준편차 {samp.std():.4f}")
    ratio = bb.std() / samp.std()
    print(f"\n    분석자 사이 / 표본 사이 = {ratio:.2f} 배")

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2), constrained_layout=True)

    # ── 왼쪽: 두 흔들림을 같은 자 위에 ──
    ax = axes[0]
    jit = rng.uniform(-0.17, 0.17, len(bb))
    ax.scatter(bb, 1 + jit, s=52, color=RED, alpha=0.72, edgecolor="white",
               linewidth=0.8)
    ax.scatter(samp, 0 + rng.uniform(-0.17, 0.17, len(samp)), s=52, color=TEAL,
               alpha=0.72, edgecolor="white", linewidth=0.8)
    ax.axvline(BETA, color=INK, lw=2.2, ls="--")
    ax.text(BETA + 0.015, 1.52, f"참값 $\\beta = {BETA}$", fontsize=11.2, color=INK)
    for yy, v, col, lab in (
            (1, bb, RED, f"같은 자료 한 벌, 분석자 {len(bb)} 명"),
            (0, samp, TEAL, f"새 자료 {len(samp)} 벌, 분석 하나")):
        ax.plot([v.min(), v.max()], [yy - 0.30, yy - 0.30], lw=3.0, color=col,
                solid_capstyle="butt")
        ax.text(v.min() - 0.012, yy - 0.30, f"{v.min():+.2f}", ha="right",
                va="center", fontsize=10.2, color=col)
        ax.text(v.max() + 0.012, yy - 0.30, f"{v.max():+.2f}", ha="left",
                va="center", fontsize=10.2, color=col)
        ax.text(0.015, yy + 0.33, lab, transform=ax.get_yaxis_transform(),
                fontsize=11.4, color=col, clip_on=False)
    ax.annotate(f"분석자 사이가 {ratio:.1f} 배 넓다",
                xy=(0.72, 1.0), xytext=(0.60, 0.46), fontsize=11.4, color=INK,
                ha="center", arrowprops=dict(arrowstyle="->", color=INK, lw=1.5))
    ax.set_yticks([])
    ax.set_ylim(-0.6, 1.72)
    ax.set_xlim(-0.12, 0.95)
    ax.set_xlabel("$x$ 의 계수 추정값", fontsize=11.5)
    ax.set_title("흔들림의 원천이 둘이다", fontsize=12.8, color=INK, pad=8)
    ax.grid(alpha=0.22, axis="x")
    ax.spines[["top", "right", "left"]].set_visible(False)

    # ── 오른쪽: 분석자 64 명의 추정값과 신뢰구간 ──
    ax = axes[1]
    order = np.argsort(bb)
    lo = bb[order] - 1.96 * ses[order]
    hi = bb[order] + 1.96 * ses[order]
    sig = ps[order] < 0.05
    pos = np.arange(len(bb))
    ax.hlines(pos, lo, hi, color=[RED if s else GREY for s in sig], lw=1.6,
              alpha=0.85)
    ax.scatter(bb[order], pos, s=26,
               color=[RED if s else GREY for s in sig], zorder=4)
    ax.axvline(BETA, color=INK, lw=2.2, ls="--")
    ax.axvline(0, color=MUTED, lw=1.6)
    rank_full = int(np.where(order == full_idx)[0][0])
    ax.scatter([bb[full_idx]], [rank_full], s=130, marker="*", color=BLUE,
               zorder=6)
    ax.annotate("교란을 모두 보정한 분석", xy=(bb[full_idx], rank_full),
                xytext=(0.56, 14), fontsize=10.8, color=BLUE, ha="center",
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.4))
    ax.text(0.02, 0.97,
            f"$p < 0.05$ 인 분석  ${int(sig.sum())}/{len(bb)}$\n"
            f"그렇지 않은 분석  ${int((~sig).sum())}/{len(bb)}$",
            transform=ax.transAxes, va="top", fontsize=11, color=INK,
            bbox=dict(boxstyle="round,pad=0.42", fc="white", ec=GREY, alpha=0.95))
    ax.legend(handles=[
        Line2D([], [], color=RED, lw=2.2, marker="o", ms=5,
               label="유의하다고 보고할 분석"),
        Line2D([], [], color=GREY, lw=2.2, marker="o", ms=5,
               label="유의하지 않다고 보고할 분석")],
        fontsize=10.2, framealpha=0.93, loc="lower right")
    ax.set_yticks([])
    ax.set_xlabel("$x$ 의 계수 추정값과 $95\\%$ 신뢰구간", fontsize=11.5)
    ax.set_ylabel(f"분석자 {len(bb)} 명 (추정값 순)", fontsize=11.5)
    ax.set_title("같은 자료에서 나온 답들", fontsize=12.8, color=INK, pad=8)
    ax.grid(alpha=0.22, axis="x")
    ax.spines[["top", "right", "left"]].set_visible(False)

    fig.suptitle("표본을 다시 뽑지 않아도 답은 움직인다", fontsize=14.5)
    fig.text(0.5, -0.05,
             f"자료 한 벌($n = {N}$)을 만들어 두고, 공변량 여섯 개 가운데 무엇을 "
             "보정할지만 달리하는 분석자 $64$ 명에게 주었다. 이 모의실험에서는 "
             "어느 쪽이 옳은지 알 수 있다 — 실제 연구에서는 모른다.",
             ha="center", fontsize=10.4, color=INK)

    path = OUT + "many_analysts.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
