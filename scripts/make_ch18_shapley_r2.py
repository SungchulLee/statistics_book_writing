r"""18.4절 「AI로 보는 통계학」 — 기여를 공리로 나누면 순서가 달라진다.

regularization_path 쪽은 라쏘의 **진입 순서**를 변수 중요도로 읽는 것을
경계한다. 잡음변수 x6·x7·x12 가 참 변수 넷 바로 뒤에 들어오기 때문이다.

이 그림은 같은 자료에 **섀플리값**을 대어 본다. 특징 집합 S 만 넣고
적합했을 때의 R^2 을 v(S) 로 두고, 모든 S 에 대해 증가분을 가중평균한다.

  왼쪽   가중치 w(s) = s!(n-s-1)!/n! 를 크기 s 마다 다 더하면 꼭 1/n 이다.
         크기가 s 인 S 가 C(n-1, s) 개이고 w(s) = 1/(n C(n-1,s)) 이므로.
         층이 n 개이니 전체가 1 이 되고, 그래서 기여의 합이 R^2 을 정확히
         복원한다.
  오른쪽 그 섀플리 R^2 기여를 변수별로 그린 것. 참 변수 넷이 거의 전부를
         가져가고 잡음 여덟은 바닥에 붙는다 — 라쏘 진입 순서와 다르다.

자료는 그 쪽의 모의 설정을 그대로 쓴다(씨앗 0, n = 200, 참 계수
3, -2, 1.5, 1 과 잡음 여덟).

만드는 파일:

  ch18/tuning/img/shapley_vs_entry.png

실행:  python3 scripts/make_ch18_shapley_r2.py   (저장소 최상위에서)
필요:  numpy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import itertools
from math import comb, factorial

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch18/tuning/img/"

INK = "#37474F"
AMBER = "#B07C2C"
TEAL = "#00838F"
RED = "#D32F2F"
MUTED = "#90A4AE"

P = 12
# 그 쪽 본문이 적어 둔 라쏘 진입 순서
ENTRY = ["x1", "x2", "x3", "x4", "x6", "x7", "x12", "x8", "x11", "x10", "x5"]


def design():
    """regularization_path 쪽과 글자 그대로 같은 자료."""
    rng = np.random.default_rng(0)
    n = 200
    X = rng.normal(0, 1, (n, P))
    beta = np.array([3.0, -2.0, 1.5, 1.0] + [0.0] * 8)
    y = X @ beta + rng.normal(0, 1, n)
    return X, y


def all_r2(X, y):
    """2^12 개 부분집합마다 OLS 의 R^2 을 비트마스크로 모아 둔다."""
    tss = ((y - y.mean()) ** 2).sum()
    out = np.zeros(1 << P)
    ones = np.ones((len(y), 1))
    for mask in range(1 << P):
        cols = [j for j in range(P) if mask >> j & 1]
        A = np.hstack([ones, X[:, cols]]) if cols else ones
        resid = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
        out[mask] = 1.0 - (resid ** 2).sum() / tss
    return out


def shapley(r2):
    """phi_j = sum_S w(|S|) [v(S+j) - v(S)]  —  S 는 j 를 담지 않는다."""
    w = [factorial(s) * factorial(P - s - 1) / factorial(P) for s in range(P)]
    phi = np.zeros(P)
    for j in range(P):
        rest = [k for k in range(P) if k != j]
        for s in range(P):
            for S in itertools.combinations(rest, s):
                m = sum(1 << k for k in S)
                phi[j] += w[s] * (r2[m | (1 << j)] - r2[m])
    return phi


def main():
    X, y = design()
    r2 = all_r2(X, y)
    phi = shapley(r2)

    print(f"    전체 R^2 = {r2[-1]:.6f}")
    print(f"    섀플리 기여의 합 = {phi.sum():.6f}  (차 {abs(phi.sum()-r2[-1]):.2e})")
    order = np.argsort(phi)[::-1]
    print("    섀플리 순위 :", [f"x{j+1}" for j in order][:6])
    print("    라쏘 진입순 :", ENTRY[:6])

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.9), constrained_layout=True)

    # ── 왼쪽: 층마다 무게의 합이 1/n ──
    ax = axes[0]
    sizes = np.arange(P)
    layer = np.array([comb(P - 1, s) * factorial(s) * factorial(P - s - 1)
                      / factorial(P) for s in sizes])
    ax.bar(sizes, layer, color=TEAL, width=0.7)
    ax.axhline(1 / P, color=RED, lw=2.2, ls="--")
    ax.text((P - 1) / 2, 1 / P * 1.28, f"층마다 1/n = {1/P:.4f}",
            ha="center", fontsize=11.5, color=RED)
    for s in (0, 5, 11):
        ax.text(s, layer[s] + 0.004, f"{comb(P-1,s)} 개", ha="center",
                fontsize=9.5, color=INK)
    ax.set_xlabel("연합의 크기 s", fontsize=11.5)
    ax.set_ylabel("그 층의 무게를 다 더한 값", fontsize=11.5)
    ax.set_title("층이 몇 개짜리든 몫이 똑같다", fontsize=12.8, color=INK, pad=8)
    ax.set_ylim(0, 1 / P * 1.45)
    ax.spines[["top", "right"]].set_visible(False)

    # ── 오른쪽: 섀플리 R^2 기여 ──
    ax = axes[1]
    names = [f"x{j+1}" for j in range(P)]
    cols = [AMBER if j < 4 else MUTED for j in range(P)]
    ax.bar(names, phi, color=cols, width=0.68)
    for j, nm in enumerate(names):
        if nm in ENTRY[:7]:
            ax.text(j, phi[j] + 0.012, str(ENTRY.index(nm) + 1), ha="center",
                    fontsize=10.5, color=RED, fontweight="bold")
    ax.set_ylabel("섀플리 $R^2$ 기여", fontsize=11.5)
    ax.set_title("붉은 수가 라쏘 진입 순서다", fontsize=12.8, color=INK, pad=8)
    ax.grid(alpha=0.22, axis="y")
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.98, 0.93, f"참 변수 넷(황갈색)이 $R^2$ 의 {phi[:4].sum()/phi.sum()*100:.1f}%",
            transform=ax.transAxes, ha="right", fontsize=10.6, color=AMBER)

    fig.suptitle("라쏘가 다섯째로 부른 변수는 기여가 사실상 $0$ 이다", fontsize=14.5)
    fig.text(0.5, -0.045,
             "오른쪽에서 5·6·7 번으로 들어온 x6·x7·x12 의 막대가 바닥에 붙어 있다. "
             "진입 순서는 상관 구조가 정하고, 섀플리 기여는 설명한 몫이 정한다.",
             ha="center", fontsize=10.5, color=INK)

    path = OUT + "shapley_vs_entry.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")
    for j in range(P):
        print(f"      x{j+1:<3} 섀플리 {phi[j]:+.5f}")


if __name__ == "__main__":
    main()
