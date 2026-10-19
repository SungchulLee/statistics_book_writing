r"""13.5절 「그림으로 보는 통계학」 — 하나씩 빼서는 못 찾는 것.

쿡 거리는 **한 점을 뺐을 때** 적합이 얼마나 움직이는지를 잰다. 그래서
나쁜 점이 하나면 잘 찾아내지만, 여럿이 **뭉쳐 있으면** 서로 가려 준다.
하나를 빼도 남은 것들이 그 자리를 지키고 있어 적합이 꿈쩍하지 않기
때문이다. 이것을 가림(masking)이라 한다.

깨끗한 점 24 개에 멀리 떨어진 자리의 나쁜 점 6 개를 섞으면

  기울기   참 +1.0 인데 전체 적합은 -0.24
  쿡 거리  나쁜 점들의 최댓값이 0.070 으로 문턱 4/n = 0.133 아래다
           게다가 자료 전체에서 가장 큰 쿡 거리는 **깨끗한 점**의 것이다

만드는 파일:

  ch13/diagnostics/img/masking.png

실행:  python3 scripts/make_ch13_masking.py   (저장소 최상위에서)
필요:  numpy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch13/diagnostics/img/"

INK = "#37474F"
AMBER = "#B07C2C"
TEAL = "#00838F"
RED = "#D32F2F"
MUTED = "#90A4AE"

N0, NBAD, X0, DROP, SEED = 24, 6, 8.0, 9.0, 7


def main():
    rng = np.random.default_rng(SEED)
    xc = np.sort(rng.uniform(0, 3, N0))
    yc = 1 + xc + rng.normal(0, 0.3, N0)
    xb = X0 + rng.normal(0, 0.15, NBAD)
    yb = 1 + xb - DROP + rng.normal(0, 0.3, NBAD)
    x = np.concatenate([xc, xb])
    y = np.concatenate([yc, yb])
    n = len(x)

    A = np.column_stack([np.ones(n), x])
    p = 2
    b_all = np.linalg.lstsq(A, y, rcond=None)[0]
    h = np.diag(A @ np.linalg.pinv(A))
    e = y - A @ b_all
    s2 = float((e ** 2).sum() / (n - p))
    D = e ** 2 * h / (p * s2 * (1 - h) ** 2)
    thr = 4 / n
    b_clean = np.linalg.lstsq(np.column_stack([np.ones(N0), xc]), yc, rcond=None)[0]

    print(f"    전체 적합    절편 {b_all[0]:+.3f}  기울기 {b_all[1]:+.3f}")
    print(f"    여섯을 빼면   절편 {b_clean[0]:+.3f}  기울기 {b_clean[1]:+.3f}  (참 +1.000)")
    print(f"    쿡 거리 문턱 4/n = {thr:.4f}")
    print(f"      나쁜 점 여섯의 최대  {D[N0:].max():.4f}")
    print(f"      깨끗한 점의 최대     {D[:N0].max():.4f}   "
          f"(= 자료 전체의 최대, {int(np.argmax(D)) + 1} 번)")
    print(f"      문턱을 넘은 점의 수  {(D > thr).sum()} 개")

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.9), constrained_layout=True)

    ax = axes[0]
    g = np.linspace(-0.3, 9.0, 50)
    ax.scatter(xc, yc, s=36, color=INK, label=f"깨끗한 점 {N0} 개", zorder=4)
    ax.scatter(xb, yb, s=80, facecolors="none", edgecolors=RED, linewidths=2.2,
               label=f"뭉쳐 있는 나쁜 점 {NBAD} 개", zorder=5)
    ax.plot(g, b_all[0] + b_all[1] * g, lw=3.0, color=RED,
            label=f"전부 쓴 적합  기울기 {b_all[1]:+.2f}")
    ax.plot(g, b_clean[0] + b_clean[1] * g, lw=3.0, color=TEAL, ls="--",
            label=f"여섯을 뺀 적합  기울기 {b_clean[1]:+.2f}")
    ax.set_xlabel("$x$", fontsize=12)
    ax.set_ylabel("$y$", fontsize=12)
    ax.set_title("여섯 점이 기울기의 부호를 바꿨다",
                 fontsize=12.8, color=INK, pad=8)
    ax.legend(fontsize=9.6, framealpha=0.93, loc="lower left")
    ax.grid(alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    cols = [INK] * N0 + [RED] * NBAD
    ax.bar(np.arange(1, n + 1), D, color=cols, width=0.72)
    ax.axhline(thr, color=AMBER, lw=2.4, ls="--")
    ax.text(n, thr + 0.004, f"흔히 쓰는 문턱 $4/n$ = {thr:.3f}",
            ha="right", fontsize=10.6, color=AMBER)
    top = int(np.argmax(D))
    ax.annotate("자료 전체에서 가장 큰 쿡 거리가\n**깨끗한** 점의 것이다".replace("**", ""),
                xy=(top + 1, D[top]), xytext=(n * 0.42, D.max() * 0.92),
                fontsize=10.6, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.4))
    ax.set_xlabel("관측 번호  (붉은 쪽이 나쁜 점 여섯)", fontsize=11.5)
    ax.set_ylabel("쿡 거리 (하나씩 뺐을 때)", fontsize=11.5)
    ax.set_title("문턱을 넘는 점이 하나도 없다", fontsize=12.8, color=INK, pad=8)
    ax.set_ylim(0, thr * 1.35)
    ax.grid(alpha=0.22, axis="y")
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("하나씩 빼서 찾는 진단은 뭉친 것을 못 본다", fontsize=14.5)
    fig.text(0.5, -0.05,
             f"깨끗한 점 {N0} 개에 뭉쳐 있는 나쁜 점 {NBAD} 개를 섞었다. "
             "하나를 빼도 남은 다섯이 그 자리를 지키므로 적합이 꿈쩍하지 않고, "
             "그래서 쿡 거리가 작게 나온다.",
             ha="center", fontsize=10.4, color=INK)

    path = OUT + "masking.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
