r"""11.3절 「그림으로 보는 통계학」 — 같은 표의 두 행이 같은 뜻이 아니다.

회귀표의 행들은 생김새가 똑같다. 계수, 표준오차, $t$, $p$, 신뢰구간이 나란하고
**어느 행이 믿을 만하고 어느 행이 아닌지는 표 어디에도 적혀 있지 않다.**

성과 $Y$ 를 설명하는 모형에 관심 변수 **연수 시간** $X$ 와 교란요인 **교육 수준**
$Z$ 를 넣었다고 하자. 참 구조는

    Z -> X (0.8)      X -> Y (1.0)      Z -> Y (0.5, 직접)

이다. 교육은 연수를 늘리고, 연수는 성과를 올리고, 교육은 성과를 따로도 올린다.
그러면 **교육의 총효과는 $0.5 + 0.8 \times 1.0 = 1.3$** 이다.

이 스크립트가 재는 값($n = 2000$, 되풀이 $2000$ 번, 평균):

  [배경 U 가 없을 때]        참 총효과 — 연수 1.0, 교육 1.3
    연수 X 행      +0.9999 (SD 0.0227)   <- 맞는다
    교육 Z 행      +0.5008 (SD 0.0291)   <- 직접효과뿐, 총효과의 38%
    교육만 넣은 단순회귀 +1.3004 (SD 0.0314)   <- 이쪽이 교육의 총효과다

  [관측되지 않은 배경 U 가 교육과 성과에 함께 작용할 때]
    연수 X 행      +0.9990 (SD 0.0394)   <- 여전히 맞는다
    교육 Z 행      -0.4987 (SD 0.0428)   <- 부호가 뒤집혔다
    교육만 넣은 단순회귀 +0.3008 (SD 0.0318)   <- 이것도 틀렸다

곧 **같은 표에서 한 행은 옳고 다른 행은 부호까지 틀릴 수 있다.** 그런데 그 표를
한 번 뽑아 보면

    연수 X   +0.9291  SE 0.0383  t = +24.24
    교육 Z   -0.4352  SE 0.0399  t = -10.90

로 **두 행 모두 $t$ 가 열을 훌쩍 넘는다.** 웨스트라이크와 그린랜드가 $2013$ 년에
「표 2 의 오류」라는 이름을 붙인 것이 이것이다.

만드는 파일:

  ch11/testing_coefficients/img/table2_fallacy.png

실행:  python3 scripts/make_ch13_table2_fallacy.py   (저장소 최상위에서)
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

OUT = "docs/ch11/testing_coefficients/img/"

INK = "#37474F"
AMBER = "#B07C2C"
TEAL = "#00838F"
RED = "#D32F2F"
BLUE = "#1565C0"
MUTED = "#90A4AE"

N = 2000
REPS = 2000
A, B, C = 0.8, 1.0, 0.5        # Z→X, X→Y, Z→Y(직접)
GU_Z, GU_Y = 1.0, -2.0         # U→Z, U→Y
SEED = 13


def draw(i, with_u):
    """한 번의 모의실험 — 두 계수와, 교육만 넣은 단순회귀의 계수."""
    rng = np.random.default_rng([SEED, i])
    U = rng.normal(size=N) if with_u else np.zeros(N)
    Z = GU_Z * U + rng.normal(size=N)
    X = A * Z + rng.normal(size=N)
    Y = B * X + C * Z + GU_Y * U + rng.normal(size=N)
    M = np.column_stack([np.ones(N), X, Z])
    b = np.linalg.lstsq(M, Y, rcond=None)[0]
    bz = np.linalg.lstsq(np.column_stack([np.ones(N), Z]), Y,
                         rcond=None)[0][1]
    return b[1], b[2], bz


def node(ax, xy, label, color, dashed=False):
    ax.add_patch(plt.Circle(xy, 0.15, facecolor="white", edgecolor=color,
                            lw=2.4, ls=":" if dashed else "-", zorder=4))
    ax.text(xy[0], xy[1], label, ha="center", va="center", fontsize=11.6,
            color=color, fontweight="bold", zorder=5)


def arrow(ax, p, q, label, color, dashed=False, off=(0, 0), rad=0.0):
    d = np.array(q) - np.array(p)
    u = d / np.linalg.norm(d)
    p2, q2 = np.array(p) + 0.17 * u, np.array(q) - 0.17 * u
    ax.annotate("", xy=q2, xytext=p2,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=2.0,
                                ls=":" if dashed else "-",
                                connectionstyle=f"arc3,rad={rad}",
                                shrinkA=0, shrinkB=0))
    m = (p2 + q2) / 2
    ax.text(m[0] + off[0], m[1] + off[1], label, ha="center", va="center",
            fontsize=11.2, color=color, fontweight="bold")


def main():
    rows = {}
    # 한글은 고정폭에서 두 칸을 차지하므로 자리맞춤은 공백을 직접 적는다
    print(f"n = {N}, 되풀이 {REPS} 번 · 참 총효과 — 연수 {B}, 교육 {C + A * B}")
    for with_u in (False, True):
        r = np.array([draw(i, with_u) for i in range(REPS)])
        rows[with_u] = r
        print(f"  [배경 U {'있다' if with_u else '없다'}]")
        for k, nm in ((0, "연수 X 행    "), (1, "교육 Z 행    "),
                      (2, "교육 단순회귀")):
            print(f"    {nm} 평균 {r[:, k].mean():+8.4f}  "
                  f"SD {r[:, k].std():.4f}")

    # 한 번의 표가 실제로 어떻게 생겼는지 — 두 행의 t 와 p
    rng = np.random.default_rng([SEED, 0])
    U = rng.normal(size=N)
    Z = GU_Z * U + rng.normal(size=N)
    X = A * Z + rng.normal(size=N)
    Y = B * X + C * Z + GU_Y * U + rng.normal(size=N)
    M = np.column_stack([np.ones(N), X, Z])
    b = np.linalg.lstsq(M, Y, rcond=None)[0]
    e = Y - M @ b
    se = np.sqrt((e ** 2).sum() / (N - 3) * np.diag(np.linalg.inv(M.T @ M)))
    print("  [배경 U 가 있을 때 한 번 뽑은 표]")
    print("    행        계수       SE        t")
    for nm, i in (("연수 X", 1), ("교육 Z", 2)):
        print(f"    {nm} {b[i]:+10.4f} {se[i]:8.4f} {b[i] / se[i]:+9.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.0), constrained_layout=True)

    # ── ① 그림 하나로 그린 구조 ──
    ax = axes[0]
    node(ax, (0.5, 0.78), "교육 $Z$", AMBER)
    node(ax, (0.5, 0.26), "연수 $X$", BLUE)
    node(ax, (1.45, 0.52), "성과 $Y$", INK)
    node(ax, (-0.42, 0.78), "배경 $U$", MUTED, dashed=True)
    arrow(ax, (0.5, 0.78), (0.5, 0.26), f"${A}$", AMBER, off=(-0.1, 0))
    arrow(ax, (0.5, 0.26), (1.45, 0.52), f"${B}$", BLUE, off=(0.02, -0.08))
    arrow(ax, (0.5, 0.78), (1.45, 0.52), f"${C}$", AMBER, off=(-0.02, 0.09))
    arrow(ax, (-0.42, 0.78), (0.5, 0.78), f"${GU_Z}$", MUTED, dashed=True,
          off=(0, 0.08))
    # 굽은 점선이라 꼬리표를 호 위에 직접 얹는다 (중점에 두면 다른 화살표에 붙는다)
    arrow(ax, (-0.42, 0.78), (1.45, 0.52), f"${GU_Y}$", MUTED, dashed=True,
          off=(0.56, 0.27), rad=-0.42)
    ax.text(-0.42, 0.56, "관측되지 않는다", ha="center", fontsize=10.2,
            color=MUTED)
    ax.text(0.52, 1.03,
            f"교육의 총효과 $= {C} + {A} \\times {B} = {C + A * B}$",
            ha="center", fontsize=12.2, color=AMBER, fontweight="bold")
    ax.text(0.52, -0.03,
            "표의 교육 행이 주는 것은 곧은 화살표 하나뿐이다",
            ha="center", fontsize=11.2, color=INK)
    ax.set_xlim(-0.95, 1.95)
    ax.set_ylim(-0.17, 1.17)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("① 표에는 없고 그림에만 있는 것", fontsize=12.8, color=INK)

    # ── ② 같은 표의 두 행 ──
    ax = axes[1]
    labels = ["연수 $X$ 행\n(관심 변수)", "교육 $Z$ 행\n(교란요인으로 넣었다)"]
    xs = np.arange(2)
    w = 0.33
    for j, (with_u, col, tag) in enumerate(
            ((False, TEAL, "배경 $U$ 가 없을 때"),
             (True, RED, "배경 $U$ 가 있을 때"))):
        vals = [rows[with_u][:, 0].mean(), rows[with_u][:, 1].mean()]
        ax.bar(xs + (j - 0.5) * w, vals, width=w * 0.92, color=col, alpha=0.88,
               label=tag)
        for i, v in enumerate(vals):
            ax.text(i + (j - 0.5) * w, v + (0.07 if v > 0 else -0.16),
                    f"${v:+.3f}$", ha="center", fontsize=11.4, color=col,
                    fontweight="bold")
    ax.axhline(0, color=INK, lw=1.0)
    for i, (tv, nm) in enumerate(((B, "참 총효과 $1.0$"),
                                  (C + A * B, "참 총효과 $1.3$"))):
        ax.plot([i - 0.52, i + 0.52], [tv, tv], "--", lw=2.2, color=INK)
        ax.text(i + 0.53, tv, nm, ha="left", va="center", fontsize=10.8,
                color=INK)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(-0.95, 1.75)
    ax.set_xlim(-0.6, 1.95)
    ax.set_ylabel("추정된 계수의 평균", fontsize=11.5)
    ax.set_title("② 왼쪽 행은 맞고 오른쪽 행은 틀린다", fontsize=12.8,
                 color=INK)
    ax.grid(alpha=0.22, axis="y")
    ax.legend(fontsize=10.2, framealpha=0.93, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("한 모형이 옳게 재는 것은 한 변수뿐이다", fontsize=14.5)
    fig.text(0.5, -0.04,
             f"관측 ${N}$ 개짜리 자료를 ${REPS}$ 번 만들어 평균한 값이다. "
             f"두 행의 $t$ 는 모두 열을 넘고 $p$ 는 모두 $0$ 에 붙는다 — "
             f"어느 행이 틀렸는지는 표가 알려 주지 않는다. "
             f"모든 수는 이 스크립트가 계산한 값이다.",
             ha="center", fontsize=10.4, color=INK)

    os.makedirs(OUT, exist_ok=True)
    path = OUT + "table2_fallacy.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
