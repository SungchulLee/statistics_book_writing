r"""13.1절 다중선형회귀 — Advertising 자료의 회귀평면.

설명변수가 둘이면 적합값 $\hat y = \hat\beta_0 + \hat\beta_1 x_1 + \hat\beta_2 x_2$ 가
$(x_1, x_2, y)$ 공간에서 **평면**을 이룬다. 그 평면을 이 쪽이 쓰는 자료로 직접 그린다.

쪽의 보기 $4$ 와 같은 모형이다 — 훈련 $140$ 행에 `Sales ~ TV + Radio`.

이 스크립트가 재는 값:

  적합식  Sales = 2.9881 + 0.047058 TV + 0.177991 Radio
  훈련 R^2 0.8936,  잔차 표준오차 s = 1.7313

만드는 파일:

  ch13/linear_regression/img/advertising_plane.png

실행:  python3 scripts/make_ch13_advertising_plane.py   (저장소 최상위에서)
필요:  numpy, pandas, matplotlib — 문서 빌드에는 필요하지 않다.
       자료는 쪽의 보기들과 같은 주소에서 내려받는다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch13/linear_regression/img/"
URL = ("https://raw.githubusercontent.com/justmarkham/scikit-learn-videos/"
       "8545c74961398def7724501648fd504dbf061b41/data/Advertising.csv")

INK = "#37474F"
AMBER = "#B07C2C"
TEAL = "#00838F"
RED = "#D32F2F"
BLUE = "#1565C0"
MUTED = "#90A4AE"

NTRAIN = 140
EVERY = 5          # 몇 점마다 잔차선을 그을 것인가


def main():
    df = pd.read_csv(URL, usecols=[1, 2, 3, 4]).iloc[:NTRAIN]
    tv, radio, sales = df.TV.values, df.Radio.values, df.Sales.values
    X = np.column_stack([np.ones(len(df)), tv, radio])
    b = np.linalg.lstsq(X, sales, rcond=None)[0]
    fit = X @ b
    e = sales - fit
    r2 = 1 - (e ** 2).sum() / ((sales - sales.mean()) ** 2).sum()
    s = np.sqrt((e ** 2).sum() / (len(df) - 3))

    # 한글은 고정폭에서 두 칸을 차지하므로 자리맞춤은 공백을 직접 적는다
    print(f"훈련 {len(df)} 행 — Sales ~ TV + Radio")
    print(f"  적합식  Sales = {b[0]:.4f} + {b[1]:.6f} TV + {b[2]:.6f} Radio")
    print(f"  훈련 R^2 {r2:.4f},  잔차 표준오차 s = {s:.4f}")

    fig = plt.figure(figsize=(8.6, 6.4))
    ax = fig.add_subplot(111, projection="3d")

    g1 = np.linspace(tv.min(), tv.max(), 24)
    g2 = np.linspace(radio.min(), radio.max(), 24)
    G1, G2 = np.meshgrid(g1, g2)
    ax.plot_surface(G1, G2, b[0] + b[1] * G1 + b[2] * G2, alpha=0.32,
                    color=TEAL, edgecolor="none")

    above = e >= 0
    ax.scatter(tv[above], radio[above], sales[above], s=26, color=INK,
               depthshade=False, label="평면 위의 점")
    ax.scatter(tv[~above], radio[~above], sales[~above], s=26, color=AMBER,
               depthshade=False, label="평면 아래의 점")
    for i in range(0, len(df), EVERY):
        ax.plot([tv[i], tv[i]], [radio[i], radio[i]], [sales[i], fit[i]],
                color=RED, lw=1.1, alpha=0.85)

    ax.set_xlabel("TV", fontsize=11, labelpad=6)
    ax.set_ylabel("Radio", fontsize=11, labelpad=6)
    ax.set_zlabel("Sales", fontsize=11, labelpad=4)
    ax.view_init(elev=18, azim=-128)
    ax.set_title(f"$\\widehat{{\\text{{Sales}}}} = {b[0]:.3f} + "
                 f"{b[1]:.4f}\\,\\text{{TV}} + {b[2]:.4f}\\,\\text{{Radio}}$"
                 f"   ($R^2 = {r2:.4f}$)", fontsize=12.6, color=INK, pad=2)
    ax.legend(fontsize=10, framealpha=0.92, loc="upper left")
    fig.text(0.5, 0.035,
             f"훈련 {len(df)} 행이다. 붉은 선분은 {EVERY} 점마다 하나씩 그은 "
             f"잔차이고 모두 Sales 축과 나란하다. 수는 직접 계산한 값이다.",
             ha="center", fontsize=10.2, color=INK)

    os.makedirs(OUT, exist_ok=True)
    path = OUT + "advertising_plane.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
