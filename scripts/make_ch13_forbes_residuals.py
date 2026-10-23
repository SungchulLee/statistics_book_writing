r"""13.3절 「그림으로 보는 통계학」 — 포브스의 열일곱 점.

$1857$ 년 에든버러의 물리학자 제임스 D. 포브스가 알프스의 $17$ 곳에서 **물의
끓는점**(화씨)과 **기압**(수은주 인치)을 함께 쟀다. 끓는점으로 고도를 재는
방법을 다듬으려던 것이다.

그 열일곱 점으로 회귀표를 만들면 $R^2 = 0.994$ 가 나온다. 기압을 로그로 바꾸어
다시 적합해도 $R^2 = 0.995$ 다. **표만 보아서는 두 모형을 가를 수 없다.**
가르는 것은 잔차그림이고, 거기에 한 가지 함정이 더 있다.

이 스크립트가 재는 값:

  원 눈금   pres ~ bp          R^2 0.99443
  로그 눈금 100 log10(pres)~bp R^2 0.99496

  잔차에 이차항을 넣어 굽음을 검정하면
    원 눈금  12 번 빼고 t = +7.87 (p < 0.0001) · 전부 쓰면 t = +1.55 (p = 0.143)
    로그 눈금 12 번 빼고 t = -1.04 (p = 0.318)  · 전부 쓰면 t = -1.33 (p = 0.204)

곧 **이상점 하나가 굽음을 가린다.** $12$ 번을 넣은 채로 검정하면 원 눈금에서도
굽음이 유의하지 않다.

자료는 R 의 `MASS::forbes`(= `alr4::Forbes`)로 전해지는 포브스의 관측 열일곱 개를
그대로 적어 넣었다. 로그 눈금 $100\log_{10}(\text{pres})$ 는 `alr4` 가 `lpres` 라는
이름으로 함께 싣는 바로 그 열이다.

만드는 파일:

  ch13/testing_coefficients/img/forbes_residuals.png

실행:  python3 scripts/make_ch13_forbes_residuals.py   (저장소 최상위에서)
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

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch13/testing_coefficients/img/"

INK = "#37474F"
AMBER = "#B07C2C"
TEAL = "#00838F"
RED = "#D32F2F"
BLUE = "#1565C0"
MUTED = "#90A4AE"

# 포브스(1857)의 관측 열일곱 — 끓는점(화씨)과 기압(수은주 인치)
BP = np.array([194.5, 194.3, 197.9, 198.4, 199.4, 199.9, 200.9, 201.1, 201.4,
               201.3, 203.6, 204.6, 209.5, 208.6, 210.7, 211.9, 212.2])
PRES = np.array([20.79, 20.79, 22.40, 22.67, 23.15, 23.35, 23.89, 23.99, 24.02,
                 24.01, 25.14, 26.57, 28.49, 27.76, 29.04, 29.88, 30.06])
ODD = 11          # 0 부터 셌을 때 12 번 관측


def ols(x, y):
    X = np.column_stack([np.ones(len(x)), x])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    r = y - X @ b
    r2 = 1 - (r ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return b, r, float(r2)


def quad_t(x, r):
    """잔차 r 을 x 의 이차식에 맞춰 굽음을 재고, 그릴 계수까지 돌려준다."""
    xc = x - x.mean()
    X = np.column_stack([np.ones(len(x)), xc, xc ** 2])
    b = np.linalg.lstsq(X, r, rcond=None)[0]
    e = r - X @ b
    n, p = X.shape
    s2 = (e ** 2).sum() / (n - p)
    se = np.sqrt(s2 * np.linalg.inv(X.T @ X)[2, 2])
    t = b[2] / se
    return float(t), float(2 * (1 - stats.t.cdf(abs(t), n - p))), b


def main():
    keep = np.ones(len(BP), bool)
    keep[ODD] = False
    lpres = 100 * np.log10(PRES)

    scales = [("원 눈금 — 기압 그대로", PRES, RED),
              ("로그 눈금 — $100\\log_{10}$(기압)", lpres, TEAL)]

    # 한글은 고정폭에서 두 칸을 차지하므로 자리맞춤은 공백을 직접 적는다
    print("포브스의 관측 17 개")
    info = []
    for name, y, col in scales:
        b, r, r2 = ols(BP, y)
        t_all, p_all, _ = quad_t(BP, r)            # 잔차의 굽음 (전부)
        t_keep, p_keep, bq = quad_t(BP[keep], r[keep])   # 12 번을 뺀 잔차의 굽음
        info.append((name, y, col, b, r, r2, t_all, p_all, t_keep, p_keep, bq))
        flat = name.replace("$", "").replace("\\log_{10}", "log10")
        print(f"  {flat:34s} R^2 {r2:.5f}")
        print(f"      굽음 검정 — 전부 t {t_all:+5.2f} (p {p_all:.4f}) · "
              f"12 번 빼고 t {t_keep:+5.2f} (p {p_keep:.4f})")
    print(f"  12 번 관측의 잔차 — 원 {info[0][4][ODD]:+.3f}  "
          f"로그 {info[1][4][ODD]:+.3f}  "
          f"(다른 점들의 최대 절댓값 {np.abs(info[0][4][keep]).max():.3f} / "
          f"{np.abs(info[1][4][keep]).max():.3f})")

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.7), constrained_layout=True)
    grid = np.linspace(BP.min() - 0.6, BP.max() + 0.6, 200)

    notes = ["가운데가 꺼지고 양끝이 들린다", "굽음이 사라졌다"]

    for ax, (name, y, col, b, r, r2, t_all, p_all, t_keep, p_keep, bq), note in zip(
            axes, info, notes):
        ax.axhline(0, color=MUTED, lw=1.2, ls=":")
        ax.plot(BP[keep], r[keep], "o", ms=9, color=col, alpha=0.85,
                label="잔차")
        ax.plot([BP[ODD]], [r[ODD]], "X", ms=16, color=INK, mec="white",
                mew=1.4, label=f"$12$ 번 관측 (잔차 ${r[ODD]:+.2f}$)")
        gc = grid - BP[keep].mean()
        ax.plot(grid, bq[0] + bq[1] * gc + bq[2] * gc ** 2, lw=2.2, ls="--",
                color=col, alpha=0.9,
                label=f"$12$ 번을 뺀 이차 맞춤  $t = {t_keep:+.2f}$")
        # 12 번의 잔차가 다른 점들보다 훨씬 커서, 아래위로 자리를 넉넉히 둔다
        lo, hi = min(r.min(), 0.0), r.max()
        span = hi - lo
        ax.set_ylim(lo - 0.46 * span, hi + 0.20 * span)
        ax.text(0.98, 0.035, note, transform=ax.transAxes, ha="right",
                fontsize=11.6, color=col, fontweight="bold")
        ax.set_xlabel("끓는점 (화씨)", fontsize=11.5)
        ax.set_ylabel("잔차", fontsize=11.5)
        ax.set_title(f"{name}   $R^2 = {r2:.5f}$", fontsize=12.4, color=INK)
        ax.grid(alpha=0.22)
        ax.legend(fontsize=9.6, framealpha=0.93, loc="lower left")
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("$R^2$ 은 둘을 가르지 못하고, 잔차그림은 가른다", fontsize=14.5)
    fig.text(0.5, -0.05,
             "포브스가 $1857$ 년에 알프스 $17$ 곳에서 잰 값이다"
             "(R 의 MASS::forbes 로 전해지는 그 표). "
             "$R^2$ 과 $t$ 는 모두 이 스크립트가 계산한 값이다.",
             ha="center", fontsize=10.4, color=INK)

    os.makedirs(OUT, exist_ok=True)
    path = OUT + "forbes_residuals.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
