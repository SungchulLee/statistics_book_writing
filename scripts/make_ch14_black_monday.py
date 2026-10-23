r"""14.3절 「그림으로 보는 통계학」 — $1987$ 년 $10$ 월 $19$ 일을 두 모형에 물어본다.

그날 다우지수가 $508$ 포인트, $22.6\%$ 떨어졌고 S&P $500$ 은 $20.47\%$ 떨어졌다.
둘 다 각 지수 역사상 하루 최대 낙폭이다.

일간 수익률이 정규분포라고 두면 그런 하루는 얼마나 드문가. 일간 표준편차를
$1\%$ 로 보면 $-20.47\%$ 는 $-20.5$ 시그마이고, 확률이 $2.0 \times 10^{-93}$ 이다.
거래일을 한 해 $252$ 일로 치면 **$2.0 \times 10^{90}$ 년에 한 번**, 곧 우주 나이
($138$ 억 년)의 $1.4 \times 10^{80}$ 배를 기다려야 한다.

같은 하루를 꼬리가 두꺼운 $t$ 분포로 보면 수가 전혀 달라진다. 표준편차를 똑같이
$1\%$ 로 맞춘 $t(3)$ 에서는 **$161$ 년에 한 번**이다.

그리는 것:

  ① 같은 하루를 두고 모형마다 말하는 기다림(년). 로그 눈금이고 우주의 나이와
     $100$ 년을 가로선으로 긋는다
  ② 모형마다 말하는 「백 년에 한 번」의 하루 낙폭. 정규는 $-3.95\%$ 인데
     실제로 일어난 날은 $-20.47\%$ 였다

$t$ 분포는 모두 표준편차가 $1\%$ 가 되도록 $\sqrt{\nu/(\nu-2)}$ 로 나눠 맞춘다.

만드는 파일:

  ch14/graphical_methods/img/black_monday.png

실행:  python3 scripts/make_ch14_black_monday.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch14/graphical_methods/img/"

INK = "#37474F"
AMBER = "#B07C2C"
TEAL = "#00838F"
RED = "#D32F2F"
BLUE = "#1565C0"
MUTED = "#90A4AE"

DROP = -20.47        # 그날 S&P 500 의 낙폭 (퍼센트)
SD = 1.0             # 일간 표준편차를 1 퍼센트로 둔다
DAYS = 252           # 한 해 거래일
UNIVERSE = 1.38e10   # 우주의 나이 (년)
NUS = (3, 4, 5, 6, 10)


def t_scale(nu):
    """표준편차가 1 이 되도록 t 분포를 줄이는 인자."""
    return np.sqrt(nu / (nu - 2))


def wait_years(p):
    return 1.0 / (p * DAYS)


def main():
    p_norm = stats.norm.cdf(DROP / SD)
    # 한글은 고정폭에서 두 칸을 차지하므로 자리맞춤은 공백을 직접 적는다
    print(f"그날의 낙폭 {DROP}% — 일간 표준편차를 {SD}% 로 두면 "
          f"{DROP/SD:.1f} 시그마")
    print(f"  정규  확률 {p_norm:.3e}   기다림 {wait_years(p_norm):.3e} 년 "
          f"(우주 나이의 {wait_years(p_norm)/UNIVERSE:.2e} 배)")
    ps = {}
    for nu in NUS:
        ps[nu] = float(stats.t.cdf(DROP / SD * t_scale(nu), nu))
        print(f"  t({nu:2d}) 확률 {ps[nu]:.3e}   기다림 {wait_years(ps[nu]):12.1f} 년")

    q100 = 1.0 / (100 * DAYS)
    cent = {"정규": float(stats.norm.ppf(q100) * SD)}
    for nu in (5, 4, 3):
        cent[f"$t({nu})$"] = float(stats.t.ppf(q100, nu) / t_scale(nu) * SD)
    print("\n모형마다 말하는 「백 년에 한 번」의 하루 낙폭")
    for k, v in cent.items():
        print(f"  {k:7s} {v:7.2f}%")

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.7), constrained_layout=True)

    # ── ① 기다림 ──
    ax = axes[0]
    names = ["정규"] + [f"$t({nu})$" for nu in NUS]
    waits = [wait_years(p_norm)] + [wait_years(ps[nu]) for nu in NUS]
    cols = [RED] + [TEAL] * len(NUS)
    ax.bar(np.arange(len(names)), waits, width=0.6, color=cols, alpha=0.88)
    ax.set_yscale("log")
    ax.axhline(UNIVERSE, color=INK, lw=1.6, ls="--")
    ax.text(len(names) - 0.4, UNIVERSE * 2.5, "우주의 나이 ($138$ 억 년)",
            ha="right", fontsize=10.6, color=INK)
    ax.axhline(100, color=MUTED, lw=1.4, ls=":")
    ax.text(-0.45, 160, "$100$ 년", ha="left", fontsize=10.6, color=MUTED)
    for i, w in enumerate(waits):
        txt = f"$10^{{{np.log10(w):.0f}}}$ 년" if w > 1e6 else f"${w:,.0f}$ 년"
        ax.text(i, w * 2.2, txt.replace(",", "{,}"), ha="center", fontsize=10.2,
                color=cols[i], fontweight="bold")
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, fontsize=11.5)
    ax.set_ylim(10, 1e105)
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda v, _: f"$10^{{{int(np.log10(v))}}}$" if v > 0 else ""))
    ax.set_ylabel("그런 하루를 한 번 보려면 (년)", fontsize=11.5)
    ax.set_xlabel("일간 수익률에 씌운 분포 (표준편차는 모두 $1\\%$)", fontsize=11.5)
    ax.set_title("① 같은 하루, 전혀 다른 대답", fontsize=12.8, color=INK)
    ax.grid(alpha=0.22, axis="y")
    ax.spines[["top", "right"]].set_visible(False)

    # ── ② 백 년에 한 번 ──
    ax = axes[1]
    ks = list(cent.keys())
    vs = [cent[k] for k in ks]
    ax.barh(np.arange(len(ks)), vs, height=0.55,
            color=[RED, TEAL, TEAL, TEAL], alpha=0.88)
    for i, v in enumerate(vs):
        ax.text(v - 0.5, i, f"${v:.2f}\\%$", ha="right", va="center",
                fontsize=11.5, color=INK, fontweight="bold")
    ax.axvline(DROP, color=INK, lw=2.2, ls="--")
    ax.text(DROP + 0.5, 1.45,
            f"실제로 일어난 하루\n$1987$ 년 $10$ 월 $19$ 일\n${DROP}\\%$",
            ha="left", va="center", fontsize=11.4, color=INK, fontweight="bold")
    ax.set_yticks(np.arange(len(ks)))
    ax.set_yticklabels(ks, fontsize=11.5)
    ax.set_xlim(-23, 1.5)
    ax.set_xlabel("하루 낙폭 ($\\%$)", fontsize=11.5)
    ax.set_title("② 「백 년에 한 번」이라 부르는 날", fontsize=12.8, color=INK)
    ax.grid(alpha=0.22, axis="x")
    ax.spines[["top", "right", "left"]].set_visible(False)

    fig.suptitle("정규분포는 그날을 설명하지 못한다", fontsize=14.5)
    fig.text(0.5, -0.05,
             f"S&P $500$ 의 그날 낙폭 ${DROP}\\%$ 만 바깥에서 가져온 수이고, "
             f"나머지는 모두 이 스크립트가 계산한 값이다. 거래일은 한 해 "
             f"${DAYS}$ 일로 쳤다.",
             ha="center", fontsize=10.4, color=INK)

    os.makedirs(OUT, exist_ok=True)
    path = OUT + "black_monday.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
