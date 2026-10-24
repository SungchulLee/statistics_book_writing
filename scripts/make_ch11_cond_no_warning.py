r"""11.9절 「그림으로 보는 통계학」 — 요약표 맨 아래 경고가 실제로 재는 것.

`statsmodels` 의 OLS 요약표는 조건수가 $1000$ 을 넘으면 각주를 붙인다.

    The condition number is large, ... This might indicate that there are
    strong multicollinearity or other numerical problems.

그 경고가 정말 공선성을 재는지 확인한다. 네 자료를 만들어 조건수와 VIF 를 나란히
본다.

  설명변수 하나, x ~ U(0, 10)          조건수 작고 VIF = 1        경고 없다
  **같은 자료에 1000 을 더하기만**       조건수 3.5e5, VIF = 1      경고 뜬다
  중심화해 되돌리면                     조건수 2.9                 경고 없다
  **상관 0.99 인 두 변수(평균 0)**       조건수 15, VIF = 55.6      경고 없다

곧 이 경고는 **공선성이 아니라 설계행렬의 눈금과 중심**에 반응한다. 공선성이
있을 수 없는 자료(설명변수 하나)에서도 뜨고, VIF 가 $55$ 인 자료에서는 조용하다.

그리는 것:

  ① 네 자료의 조건수와 VIF
  ② 설명변수 하나짜리 자료에 상수 $c$ 를 더해 가며 본 조건수. VIF 는 줄곧 $1$ 인데
     조건수만 자라 어느 지점에서 $1000$ 을 넘는다

만드는 파일:

  ch11/package_usage/img/cond_no_warning.png

실행:  python3 scripts/make_ch13_cond_no_warning.py   (저장소 최상위에서)
필요:  numpy, statsmodels, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import warnings

import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch11/package_usage/img/"

INK = "#37474F"
AMBER = "#B07C2C"
TEAL = "#00838F"
RED = "#D32F2F"
BLUE = "#1565C0"
MUTED = "#90A4AE"

THRESH = 1000        # statsmodels 가 각주를 붙이는 문턱
N = 100
SEED = 13


def fit_stats(X, y):
    Xc = sm.add_constant(X)
    res = sm.OLS(y, Xc).fit()
    vifs = [variance_inflation_factor(Xc, j) for j in range(1, Xc.shape[1])]
    return float(res.condition_number), float(max(vifs))


def main():
    rng = np.random.default_rng(SEED)
    x = rng.uniform(0, 10, N)
    y = 2 + 1.5 * x + rng.normal(0, 1, N)
    z1 = rng.normal(size=N)
    z2 = 0.99 * z1 + np.sqrt(1 - 0.99 ** 2) * rng.normal(size=N)

    cases = [
        ("설명변수 하나\n$x \\sim U(0,10)$", x, y),
        ("같은 자료에\n$1000$ 을 더했다", x + 1000, y),
        ("중심화해\n되돌렸다", x - x.mean(), y),
        ("상관 $0.99$ 인 둘\n(평균 $0$)", np.column_stack([z1, z2]), y),
    ]
    rows = []
    # 한글은 고정폭에서 두 칸을 차지하므로 자리맞춤은 공백을 직접 적는다
    print("자료                        조건수        최대 VIF   경고")
    for name, X, yy in cases:
        c, v = fit_stats(X, yy)
        rows.append((name, c, v))
        flat = name.replace("\n", " ")
        print(f"{flat:26s} {c:11.4g} {v:10.2f}   "
              f"{'뜬다' if c > THRESH else '없다'}")

    cs = np.linspace(0, 2000, 400)
    curve = np.array([fit_stats(x + c, y)[0] for c in cs])
    cross = cs[np.argmax(curve > THRESH)]
    print(f"\n상수를 더해 가면 조건수가 {THRESH} 를 넘는 자리: c = {cross:.0f}")
    print(f"  (그때 VIF 는 {fit_stats(x + cross, y)[1]:.2f} 로 그대로다)")

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), constrained_layout=True)

    # ── ① 네 자료 ──
    ax = axes[0]
    xs = np.arange(len(rows))
    conds = [r[1] for r in rows]
    cols = [RED if c > THRESH else TEAL for c in conds]
    ax.bar(xs, conds, width=0.56, color=cols, alpha=0.88)
    ax.set_yscale("log")
    ax.axhline(THRESH, color=INK, lw=1.8, ls="--")
    ax.text(-0.45, THRESH * 1.5, f"경고가 붙는 문턱 ${THRESH}$", ha="left",
            fontsize=10.8, color=INK)
    for i, (name, c, v) in enumerate(rows):
        # 큰 수는 e 표기 대신 10 의 거듭제곱으로 — mathtext 가 e 를 기울여 찍는다
        exp = int(np.floor(np.log10(c)))
        lab = f"${c:.1f}$" if c < 1000 else f"${c / 10 ** exp:.1f} \\times 10^{{{exp}}}$"
        ax.text(i, c * 2.0, lab, ha="center",
                fontsize=11.4, color=cols[i], fontweight="bold")
        ax.text(i, 0.35, f"VIF ${v:.2f}$", ha="center", fontsize=11,
                color=INK)
    ax.set_xticks(xs)
    ax.set_xticklabels([r[0] for r in rows], fontsize=10.4)
    ax.set_ylim(0.2, 1e8)
    ax.set_ylabel("조건수 (Cond. No.)", fontsize=11.5)
    ax.set_title("① 경고는 공선성을 재지 않는다", fontsize=12.8, color=INK)
    ax.grid(alpha=0.22, axis="y")
    ax.spines[["top", "right"]].set_visible(False)

    # ── ② 상수를 더해 가며 ──
    ax = axes[1]
    ax.plot(cs, curve, lw=2.8, color=BLUE)
    ax.axhline(THRESH, color=INK, lw=1.8, ls="--")
    ax.axvline(cross, color=RED, lw=1.6, ls=":")
    ax.plot([cross], [THRESH], "o", ms=11, color=RED, mec="white", mew=1.5,
            zorder=5)
    ax.annotate(f"$c = {cross:.0f}$ 에서 경고가 켜진다\n(VIF 는 줄곧 $1.00$)",
                xy=(cross, THRESH), xytext=(cross + 160, 90),
                fontsize=11.4, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))
    ax.set_yscale("log")
    ax.set_xlim(-30, 2030)
    ax.set_ylim(2, 1e6)
    ax.set_xlabel("설명변수에 더한 상수 $c$  (자료의 모양은 그대로다)",
                  fontsize=11.5)
    ax.set_ylabel("조건수", fontsize=11.5)
    ax.set_title("② 상수를 더하는 것만으로", fontsize=12.8, color=INK)
    ax.grid(alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("요약표 맨 아래 경고가 실제로 재는 것", fontsize=14.5)
    fig.text(0.5, -0.05,
             f"관측 ${N}$ 개로 statsmodels 의 OLS 를 적합해 그 결과가 돌려준 "
             f"조건수와 VIF 를 그대로 적었다. 모든 수는 이 스크립트가 계산한 값이다.",
             ha="center", fontsize=10.4, color=INK)

    os.makedirs(OUT, exist_ok=True)
    path = OUT + "cond_no_warning.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
