r"""11.8절 「그림으로 보는 통계학」 — 갈릴레오의 두 자료.

파도바 시절의 갈릴레오가 공을 경사면에 굴려 책상 끝에서 튀어나간 **수평거리**를
놓는 높이별로 쟀다. 처음에는 경사면만 두고 일곱 번(자료 1), 다음에는 경사면 끝에
**수평 선반**을 덧대 다섯 번(자료 2) 쟀다. 길이 단위는 푼티(punti)다.

이 쪽이 하는 일(차수를 교차검증으로 고르기)을 그 일곱 점에 그대로 해 보면

  차수     1        2        3        4        5
  R^2   0.9264   0.9903   0.9994   0.9998   0.9970
  LOOCV  45.00    27.94     7.36     5.68     7.56   (RMSE, punti)

로 **차수 4** 가 이긴다. 그런데 그 4차식은 $h = 1214.8$ 에서 꺾여 **내려간다** —
더 높은 데서 놓으면 더 가까이 떨어진다는 말이다. 2차식은 더 이르게 $h = 1030.5$,
곧 자료의 맨 끝($1000$)에서 겨우 $30$ 푼티 위에서 꺾인다. 꺾이지 않는 것은 3차뿐이다.

선반을 덧댄 자료 2 에서는 모수 **하나**짜리 물리 모형 $d = c\sqrt h$ 가

  모수 1 개  R^2 0.9967  LOOCV RMSE 16.9   (c = 47.0858)
  모수 3 개(2차)  R^2 0.9961  LOOCV RMSE 101.3

로 이긴다. 선반이 없던 자료 1 에서는 같은 모형의 $R^2$ 가 $0.826$ 으로 떨어지고
잔차가 $+52.8, +53.9, +48.3, +26.4, +4.7, -32.1, -59.9$ 로 **굽어 있다.**
선반을 덧댄 까닭이 바로 그것이다.

자료는 드레이크 『Galileo at Work』($1978$)에서 온 것으로, JSE 의 자료 보관소에
실린 표를 그대로 적어 넣었다.

만드는 파일:

  ch11/model_selection/img/galileo_degree.png

실행:  python3 scripts/make_ch13_galileo_degree.py   (저장소 최상위에서)
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

OUT = "docs/ch11/model_selection/img/"

INK = "#37474F"
AMBER = "#B07C2C"
TEAL = "#00838F"
RED = "#D32F2F"
BLUE = "#1565C0"
MUTED = "#90A4AE"

# 갈릴레오의 두 자료 — 놓는 높이 H 와 수평거리 D, 단위는 푼티
H1 = np.array([100, 200, 300, 450, 600, 800, 1000.])
D1 = np.array([253, 337, 395, 451, 495, 534, 573.])
H2 = np.array([300, 600, 800, 828, 1000.])
D2 = np.array([800, 1172, 1328, 1340, 1500.])


def loocv(X, y):
    """모자행렬 지름길로 LOOCV RMSE 와 R^2 를 한 번에."""
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    r = y - X @ b
    h = np.diag(X @ np.linalg.pinv(X.T @ X) @ X.T)
    rmse = float(np.sqrt(np.mean((r / (1 - h)) ** 2)))
    r2 = float(1 - (r ** 2).sum() / ((y - y.mean()) ** 2).sum())
    return rmse, r2, b


def poly(H, D, d):
    return loocv(np.vander(H, d + 1, increasing=True), D)


def turning_point(b):
    """다항식이 꺾여 내려가기 시작하는 자리. 없으면 None."""
    roots = np.polynomial.Polynomial(b).deriv().roots()
    real = sorted(r.real for r in roots if abs(r.imag) < 1e-8)
    return real[-1] if real else None


def main():
    # 한글은 고정폭에서 두 칸을 차지하므로 자리맞춤은 공백을 직접 적는다
    print("자료 1 — 경사면만, 관측 7 개")
    print("  차수   모수      R^2   LOOCV RMSE   꺾이는 자리")
    best_d, best_cv = None, np.inf
    for d in range(1, 6):
        cv, r2, b = poly(H1, D1, d)
        tp = turning_point(b)
        print(f"  {d:4d} {d + 1:6d} {r2:10.6f} {cv:10.3f}   "
              f"{'없다' if tp is None else f'{tp:.1f}'}")
        if cv < best_cv:
            best_d, best_cv = d, cv
    print(f"  LOOCV 가 고르는 차수 = {best_d}")

    cv_s1, r2_s1, b_s1 = loocv(np.sqrt(H1)[:, None], D1)
    print(f"  d = c*sqrt(h) 는 R^2 {r2_s1:.4f} 로 떨어진다 (c = {b_s1[0]:.4f})")
    print(f"    잔차 {np.round(D1 - b_s1[0] * np.sqrt(H1), 1)}")

    print("\n자료 2 — 선반을 덧댔다, 관측 5 개")
    print("  모형                모수      R^2   LOOCV RMSE")
    cv2_1, r2_1, b2_1 = poly(H2, D2, 1)
    cv2_2, r2_2, b2_2 = poly(H2, D2, 2)
    cv_s2, r2_s2, b_s2 = loocv(np.sqrt(H2)[:, None], D2)
    print(f"  1차 다항식          {2:6d} {r2_1:10.6f} {cv2_1:10.3f}")
    print(f"  2차 다항식          {3:6d} {r2_2:10.6f} {cv2_2:10.3f}")
    print(f"  d = c*sqrt(h)       {1:6d} {r2_s2:10.6f} {cv_s2:10.3f}"
          f"   c = {b_s2[0]:.4f}")
    res2 = D2 - b_s2[0] * np.sqrt(H2)
    print(f"    잔차 {np.round(res2, 1)}  최대 절댓값 {np.abs(res2).max():.1f}")

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.9), constrained_layout=True)

    # ── ① 자료 1 — 고른 차수가 내려간다 ──
    ax = axes[0]
    g1 = np.linspace(60, 1500, 400)
    ax.axvspan(1000, 1520, color=MUTED, alpha=0.13)
    ax.text(1250, 300, "자료가\n끝나는 곳", ha="center", fontsize=10.6,
            color=INK)
    for d, col, ls in ((2, AMBER, "--"), (3, TEAL, "-."), (4, RED, "-")):
        cv, r2, b = poly(H1, D1, d)
        tp = turning_point(b)
        lab = f"{d}차  LOOCV {cv:.2f}"
        if tp is not None:
            lab += f"  ($h = {tp:.0f}$ 에서 꺾인다)"
        ax.plot(g1, np.polynomial.Polynomial(b)(g1), ls, lw=2.3, color=col,
                label=lab)
        if tp is not None and tp < 1500:
            ax.plot([tp], [np.polynomial.Polynomial(b)(tp)], "v", ms=11,
                    color=col, mec="white", mew=1.2, zorder=5)
    ax.plot(H1, D1, "o", ms=10, color=INK, mec="white", mew=1.4, zorder=6,
            label="갈릴레오의 관측 $7$ 개")
    ax.set_xlim(40, 1520)
    ax.set_ylim(180, 700)
    ax.set_xlabel("놓는 높이 $h$ (푼티)", fontsize=11.5)
    ax.set_ylabel("수평거리 $d$ (푼티)", fontsize=11.5)
    ax.set_title("① 교차검증이 고른 $4$ 차식은 내려간다", fontsize=12.8,
                 color=INK)
    ax.grid(alpha=0.22)
    ax.legend(fontsize=9.6, framealpha=0.94, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)

    # ── ② 자료 2 — 모수 하나 ──
    ax = axes[1]
    g2 = np.linspace(0, 1150, 400)
    ax.plot(g2, np.polynomial.Polynomial(b2_2)(g2), "--", lw=2.3, color=AMBER,
            label=f"$2$ 차식 — 모수 $3$ 개, LOOCV {cv2_2:.1f}")
    ax.plot(g2, b_s2[0] * np.sqrt(g2), lw=2.8, color=TEAL,
            label=f"$d = c\\sqrt{{h}}$ — 모수 $1$ 개, LOOCV {cv_s2:.1f}")
    ax.plot(H2, D2, "o", ms=10, color=INK, mec="white", mew=1.4, zorder=6,
            label="선반을 덧댄 관측 $5$ 개")
    ax.annotate(f"$c = {b_s2[0]:.2f}$ 하나로\n최대 어긋남 "
                f"${np.abs(res2).max():.0f}$ 푼티",
                xy=(600, b_s2[0] * np.sqrt(600)), xytext=(660, 700),
                fontsize=11.2, color=TEAL,
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.5))
    # 높이 0 에서 두 모형이 갈라지는 자리 — 2차식은 굴리지 않아도 날아간다고 말한다
    ax.plot([0], [b2_2[0]], "X", ms=14, color=AMBER, mec="white", mew=1.4,
            zorder=6)
    ax.plot([0], [0], "o", ms=9, color=TEAL, mec="white", mew=1.2, zorder=6)
    ax.annotate(f"굴리지 않아도\n${b2_2[0]:.0f}$ 푼티를 날아간다",
                xy=(0, b2_2[0]), xytext=(250, 185), fontsize=11.2, color=AMBER,
                arrowprops=dict(arrowstyle="->", color=AMBER, lw=1.5))
    ax.set_xlim(-45, 1150)
    ax.set_ylim(-80, 1700)
    ax.set_xlabel("놓는 높이 $h$ (푼티)", fontsize=11.5)
    ax.set_ylabel("수평거리 $d$ (푼티)", fontsize=11.5)
    ax.set_title("② 선반을 덧대자 모수 하나로 맞는다", fontsize=12.8, color=INK)
    ax.grid(alpha=0.22)
    ax.legend(fontsize=9.6, framealpha=0.94, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("차수를 고르는 일은 「다항식이 맞다」를 전제한다", fontsize=14.5)
    fig.text(0.5, -0.05,
             "갈릴레오가 파도바에서 잰 두 자료다(드레이크 『Galileo at Work』, "
             "$1978$ 에 실린 표). $R^2$ 과 LOOCV 값은 모두 이 스크립트가 "
             "계산한 값이다.",
             ha="center", fontsize=10.4, color=INK)

    os.makedirs(OUT, exist_ok=True)
    path = OUT + "galileo_degree.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
