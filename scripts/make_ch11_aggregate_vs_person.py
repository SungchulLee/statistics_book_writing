r"""11.5절 「그림으로 보는 통계학」 — 분포를 완벽히 맞히면서 절반을 틀리는 모형.

성적처럼 **등급**으로 나오는 예측에서, 모형이 내놓은 등급의 **분포**를 목표 분포에
맞추어 놓으면 집계 수준의 오차는 $0$ 이 된다. 그런데 그것은 개인이 맞았다는 말이
전혀 아니다.

이 스크립트는 그 둘을 나란히 잰다. 잠재 능력 $L$ 에서 참 등급을 만들고, 모형은
상관 $\rho$ 인 $\hat L$ 로 예측한 뒤 **분위수로 맞추어** 등급 분포를 목표와 똑같이
만든다. 그러면

  분포의 전변동거리        $0.00000$   (어떤 $\rho$ 에서도)
  정확히 맞춘 비율         $\rho = 0.9$ 에서 $0.555$
  $\pm 1$ 등급 안          $0.972$
  평균 절대 등급오차       $0.474$

가 된다. **집계로 재면 만점이고 개인으로 재면 절반이 틀린다.**

그리는 것:

  ① $\rho = 0.9$ 인 한 모형의 네 가지 성적표
  ② $\rho$ 를 바꿔 가며 — 분포 오차는 늘 $0$ 이고 개인 정확도만 움직인다

만드는 파일:

  ch11/performance/img/aggregate_vs_person.png

실행:  python3 scripts/make_ch13_aggregate_vs_person.py   (저장소 최상위에서)
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

N = 200_000
GRADES = ("A*", "A", "B", "C", "D", "E")
TARGET = np.array([0.08, 0.20, 0.26, 0.24, 0.14, 0.08])   # 목표 등급 분포
RHOS = (0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.98)
RHO_MAIN = 0.90
SEED = 2020


def to_grade(v, cuts):
    return np.digitize(v, np.quantile(v, cuts))


def measure(rho, lat, cuts, rng):
    true_g = to_grade(lat, cuts)
    pred = rho * lat + np.sqrt(1 - rho ** 2) * rng.standard_normal(len(lat))
    pred_g = to_grade(pred, cuts)      # 분위수로 맞추므로 분포가 목표와 같다
    n = len(lat)
    tv = 0.5 * np.abs(np.bincount(pred_g, minlength=6) / n
                      - np.bincount(true_g, minlength=6) / n).sum()
    return dict(tv=tv,
                exact=float((pred_g == true_g).mean()),
                near=float((np.abs(pred_g - true_g) <= 1).mean()),
                mae=float(np.abs(pred_g - true_g).mean()))


def main():
    rng = np.random.default_rng(SEED)
    lat = rng.standard_normal(N)
    cuts = np.cumsum(TARGET)[:-1]

    # 잡음 씨앗은 lat 을 뽑은 씨앗과 겹치지 않게 둔다 (겹치면 pred 가 lat 과 같아진다)
    rows = {rho: measure(rho, lat, cuts, np.random.default_rng([SEED, i + 1]))
            for i, rho in enumerate(RHOS)}
    m = rows[RHO_MAIN]
    # 한글은 고정폭에서 두 칸을 차지하므로 자리맞춤은 공백을 직접 적는다
    print(f"학생 {N:,} 명, 등급 여섯, 목표 분포 {np.round(TARGET, 2)}")
    print("  rho   분포 전변동거리   정확히 맞춤   ±1 등급   평균 절대 등급오차")
    for rho in RHOS:
        r = rows[rho]
        print(f" {rho:5.2f} {r['tv']:16.5f} {r['exact']:13.3f} "
              f"{r['near']:9.3f} {r['mae']:18.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.7), constrained_layout=True)

    # ── ① 한 모형, 네 가지 성적표 ──
    ax = axes[0]
    labels = ["분포가 목표와\n얼마나 다른가",
              "개인의 등급을\n정확히 맞힌 비율",
              "$\\pm 1$ 등급\n안에 든 비율",
              "평균 절대\n등급오차"]
    vals = [m["tv"], m["exact"], m["near"], m["mae"]]
    shown = [f"${m['tv']:.3f}$", f"${m['exact']:.3f}$", f"${m['near']:.3f}$",
             f"${m['mae']:.3f}$"]
    cols = [TEAL, RED, AMBER, RED]
    ax.bar(np.arange(4), vals, width=0.58, color=cols, alpha=0.88)
    for i, (v, s) in enumerate(zip(vals, shown)):
        ax.text(i, v + 0.025, s, ha="center", fontsize=13, color=cols[i],
                fontweight="bold")
    ax.text(0, 0.17, "완벽하다", ha="center", fontsize=11.5, color=TEAL,
            fontweight="bold")
    ax.text(1, 0.69, "절반 가까이\n틀린다", ha="center", fontsize=11.5, color=RED,
            fontweight="bold")
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(labels, fontsize=10.6)
    ax.set_ylim(0, 1.18)
    ax.set_title(f"① 같은 모형의 네 가지 성적표 ($\\rho = {RHO_MAIN}$)",
                 fontsize=12.8, color=INK)
    ax.grid(alpha=0.22, axis="y")
    ax.spines[["top", "right"]].set_visible(False)

    # ── ② 모형이 좋아져도 ──
    ax = axes[1]
    xs = np.array(RHOS)
    ax.plot(xs, [rows[r]["near"] for r in RHOS], "s--", ms=7, lw=2.2,
            color=AMBER, label="$\\pm 1$ 등급 안")
    ax.plot(xs, [rows[r]["exact"] for r in RHOS], "o-", ms=8, lw=2.6,
            color=RED, label="정확히 맞힌 비율")
    ax.plot(xs, [rows[r]["tv"] for r in RHOS], "D-", ms=7, lw=2.6,
            color=TEAL, label="분포가 목표와 다른 정도")
    ax.axvline(RHO_MAIN, color=MUTED, lw=1.2, ls=":")
    ax.annotate(f"분포 오차는 어떤 $\\rho$ 에서도 $0$ 이다",
                xy=(0.86, 0.0), xytext=(0.72, 0.17), fontsize=11.2, color=TEAL,
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.4))
    ax.set_xlabel("모형의 예측력 $\\rho$ (잠재 능력과의 상관)", fontsize=11.5)
    ax.set_ylabel("비율", fontsize=11.5)
    ax.set_ylim(-0.03, 1.05)
    ax.set_title("② 모형을 아무리 키워도 갈라진다", fontsize=12.8, color=INK)
    ax.grid(alpha=0.22)
    ax.legend(fontsize=10.4, framealpha=0.93, loc="center left")
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("집계로 재면 만점, 개인으로 재면 절반", fontsize=14.5)
    fig.text(0.5, -0.05,
             f"학생 ${N // 1000}$ 천 명을 만들어 등급 여섯으로 자른 것이다. "
             f"모형의 예측은 분위수로 맞추어 등급 분포가 목표와 정확히 같게 했다. "
             f"모든 수는 이 스크립트가 계산한 값이다.",
             ha="center", fontsize=10.4, color=INK)

    os.makedirs(OUT, exist_ok=True)
    path = OUT + "aggregate_vs_person.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
