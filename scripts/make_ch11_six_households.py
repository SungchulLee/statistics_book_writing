r"""11.6절 「그림으로 보는 통계학」 — 여섯 가구에 사는 칠천사백예순 명.

이 쪽은 California Housing 자료에 설명변수 셋(MedInc, AveRooms, AveOccup)으로
회귀를 적합하고 쿡 거리를 잰다. 연습문제 $6$ 이 「$D_i > 1$ 을 넘는 점은 세 개뿐」
이라고 적어 두었는데, **그 셋이 무엇인지**는 적혀 있지 않다. 이 스크립트가 그것을
찾는다.

자료는 $1990$ 년 인구조사의 **블록 그룹**(보통 $600$–$3000$ 명) 단위다. `AveRooms`
와 `AveOccup` 은 분모가 **가구 수**인 비(ratio)여서, 가구가 몇 안 되는 구역에서는
값이 터진다.

이 스크립트가 재는 값:

  쿡 거리 상위 셋 (설명변수 3 개, n = 20,640, k = 4)
    번호 1914   D 7.108  h 0.1673   AveRooms 141.91   인구    30   가구  11
    번호 19006  D 6.987  h 0.6911   AveOccup 1243.33  인구  7460   가구   6
    번호 1979   D 1.176  h 0.1425   AveRooms 132.53   인구    36   가구  15

  그 셋을 빼고 다시 적합하면
    절편      0.6069 ->  0.6993   (+15.2%)
    MedInc    0.4347 ->  0.4444   ( +2.2%)
    AveRooms -0.0383 -> -0.0608   (-58.6%)
    AveOccup -0.0042 -> -0.0071   (-70.0%)

곧 **관측 $20{,}640$ 개 가운데 셋이 두 계수를 $60$–$70\%$ 움직인다.**

만드는 파일:

  ch11/diagnostics/img/six_households.png

실행:  python3 scripts/make_ch13_six_households.py   (저장소 최상위에서)
필요:  numpy, scikit-learn, matplotlib — 문서 빌드에는 필요하지 않다.
       자료는 scikit-learn 이 내려받아 캐시한다. PNG는 커밋되므로 CI에서
       다시 그리지 않는다.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np
from sklearn.datasets import fetch_california_housing

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False
# 로그 축의 음수 지수는 mathtext 를 거치므로 unicode_minus 로 막지 못한다
plt.rcParams["mathtext.fontset"] = "dejavusans"

OUT = "docs/ch11/diagnostics/img/"

INK = "#37474F"
AMBER = "#B07C2C"
TEAL = "#00838F"
RED = "#D32F2F"
BLUE = "#1565C0"
MUTED = "#90A4AE"

FEATURES = ["MedInc", "AveRooms", "AveOccup"]
NAMES = ("절편", "MedInc", "AveRooms", "AveOccup")


def fit(X, y):
    XtXi = np.linalg.inv(X.T @ X)
    return XtXi @ X.T @ y, XtXi


def main():
    d = fetch_california_housing(as_frame=True).frame
    X = np.column_stack([np.ones(len(d))] + [d[f].values for f in FEATURES])
    y = d["MedHouseVal"].values
    n, k = X.shape

    b, XtXi = fit(X, y)
    r = y - X @ b
    h = np.einsum("ij,jk,ik->i", X, XtXi, X)
    s2 = (r ** 2).sum() / (n - k)
    D = r ** 2 * h / (k * s2 * (1 - h) ** 2)

    big = np.argsort(D)[-3:][::-1]
    # 한글은 고정폭에서 두 칸을 차지하므로 자리맞춤은 공백을 직접 적는다
    print(f"n = {n}, k = {k}   쿡 거리 평균 {D.mean() * n:.2f}/n  "
          f"최대 {D.max():.3f}   D > 1 인 점 {int((D > 1).sum())} 개")
    print("  번호      D       h    AveRooms  AveOccup     인구    가구")
    for i in big:
        row = d.iloc[i]
        hh = row.Population / row.AveOccup
        print(f"{i:7d} {D[i]:7.3f} {h[i]:7.4f} {row.AveRooms:9.2f} "
              f"{row.AveOccup:9.2f} {row.Population:8.0f} {hh:7.0f}")

    keep = D <= 1
    b2, _ = fit(X[keep], y[keep])
    print("\n  그 셋을 빼고 다시 적합하면")
    for j, nm in enumerate(NAMES):
        print(f"    {nm:9s} {b[j]:8.4f} -> {b2[j]:8.4f}   "
              f"({100 * (b2[j] - b[j]) / abs(b[j]):+6.1f}%)")

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.9), constrained_layout=True)

    # ── ① 두 비(ratio)가 터진 자리 ──
    ax = axes[0]
    ax.plot(d.AveRooms, d.AveOccup, ".", ms=2.4, color=MUTED, alpha=0.5,
            label=f"블록 그룹 ${n:,}$ 개".replace(",", "{,}"))
    for i, col in zip(big, (RED, BLUE, AMBER)):
        row = d.iloc[i]
        ax.plot([row.AveRooms], [row.AveOccup], "X", ms=16, color=col,
                mec="white", mew=1.5, zorder=5)
    r1, r2, r3 = (d.iloc[i] for i in big)
    ax.annotate(f"가구 ${r1.Population / r1.AveOccup:.0f}$ 호에\n"
                f"방 ${r1.AveRooms:.0f}$ 개꼴",
                xy=(r1.AveRooms, r1.AveOccup), xytext=(18, 0.95),
                fontsize=11, color=RED, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))
    ax.annotate(f"가구 ${r2.Population / r2.AveOccup:.0f}$ 호에\n"
                f"사람 ${r2.Population:.0f}$ 명",
                xy=(r2.AveRooms, r2.AveOccup), xytext=(1.3, 160),
                fontsize=11, color=BLUE, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.5))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("AveRooms — 가구당 방 수 (로그 눈금)", fontsize=11.5)
    ax.set_ylabel("AveOccup — 가구당 사람 수 (로그 눈금)", fontsize=11.5)
    ax.set_title("① 쿡 거리가 $1$ 을 넘는 점 셋", fontsize=12.8, color=INK)
    ax.grid(alpha=0.22, which="both")
    ax.legend(fontsize=10, framealpha=0.93, loc="upper right", markerscale=5)
    ax.spines[["top", "right"]].set_visible(False)

    # ── ② 셋을 빼면 계수가 얼마나 움직이는가 ──
    ax = axes[1]
    ch = [100 * (b2[j] - b[j]) / abs(b[j]) for j in range(k)]
    cols = [RED if abs(c) > 20 else MUTED for c in ch]
    ax.barh(np.arange(k), ch, height=0.5, color=cols, alpha=0.9)
    for j, c in enumerate(ch):
        # 음수 막대의 꼬리표도 0 의 오른쪽에 두어 왼쪽의 계수 글자와 겹치지 않게 한다
        ax.text(c + 2.5 if c > 0 else 2.5, j, f"${c:+.1f}\\%$", va="center",
                ha="left", fontsize=11.8, color=cols[j], fontweight="bold")
        ax.text(-80, j, f"${b[j]:.4f} \\to {b2[j]:.4f}$", va="center",
                ha="left", fontsize=10.6, color=INK)
    ax.axvline(0, color=INK, lw=1.2)
    ax.set_yticks(np.arange(k))
    ax.set_yticklabels(NAMES, fontsize=11.6)
    ax.invert_yaxis()
    ax.set_xlim(-84, 46)
    ax.set_xlabel("세 점을 빼고 다시 적합했을 때 계수의 변화", fontsize=11.5)
    ax.set_title(f"② 관측 ${n:,}$ 개 가운데 셋이 한 일".replace(",", "{,}"),
                 fontsize=12.8, color=INK)
    ax.grid(alpha=0.22, axis="x")
    ax.spines[["top", "right", "left"]].set_visible(False)

    fig.suptitle("가구 수로 나눈 열이 분모에서 터질 때", fontsize=14.5)
    fig.text(0.5, -0.05,
             "$1990$ 년 인구조사의 블록 그룹 자료에 "
             "MedInc·AveRooms·AveOccup 으로 적합한 것이다. "
             "모든 수는 이 스크립트가 계산한 값이다.",
             ha="center", fontsize=10.4, color=INK)

    os.makedirs(OUT, exist_ok=True)
    path = OUT + "six_households.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
