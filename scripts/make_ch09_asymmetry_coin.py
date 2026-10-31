"""9.1절 귀무가설과 대립가설 — 두 가설은 대칭이 아니다.

들머리의 동전을 그린다. 열 번 던져 여덟 번 앞면이 나왔다. 공정한 동전
아래에서도 0.0547 로 생기고, 앞면이 참으로 0.7 인 동전 아래에서는
0.3828 로 생긴다. 같은 자료가 어느 쪽도 반박하지 못한다.

쪽의 hypothesis_types 는 모수축 위에 H0 와 H1 을 놓는 그림이고
lady_tasting_tea 는 상자의 것이다. 여기서는 두 분포를 맞댄다.
"""
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
from scipy import stats

OUT = "docs/ch09/foundations/img/"
N, K = 10, 8
k = np.arange(N + 1)
fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.4), sharey=True)
for ax, (p, name, col) in zip(axes, [(0.5, "공정한 동전", "tab:blue"),
                                     (0.7, "앞면이 0.7 인 동전", "tab:orange")]):
    pmf = stats.binom.pmf(k, N, p)
    tail = stats.binom.sf(K - 1, N, p)
    ax.bar(k, pmf, width=0.78, color="0.78")
    ax.bar(k[k >= K], pmf[k >= K], width=0.78, color=col, alpha=0.9)
    ax.axvline(K - 0.5, color="0.2", ls="--", lw=1.5)
    ax.annotate(f"여덟 번 이상이 나올 확률\n{tail:.4f}",
                xy=(K + 0.6, pmf[K:].max() * 0.6),
                xytext=(3.3, 0.205), fontsize=12, color=col,
                fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=col, lw=1.8))
    ax.set_xticks(k)
    ax.set_xlabel("열 번 가운데 앞면이 나온 수", fontsize=11)
    ax.set_title(f"{name}이라면", fontsize=12.5, color=col)
    ax.set_ylim(0, 0.33)
    ax.grid(axis="y", alpha=0.2)
axes[0].set_ylabel("확률", fontsize=11)
axes[0].text(0.03, 0.97, "기울었다고 말하기에 모자란다",
             transform=axes[0].transAxes, va="top", fontsize=11.5,
             fontweight="bold", color="tab:blue")
axes[1].text(0.03, 0.97, "셋에 하나가 넘게 생기는 일이다",
             transform=axes[1].transAxes, va="top", fontsize=11.5,
             fontweight="bold", color="tab:orange")
fig.suptitle("같은 자료 — 앞면 여덟 번 — 가 어느 쪽도 반박하지 못한다",
             fontsize=14, y=1.0)
plt.tight_layout()
path = OUT + "asymmetry_coin.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
print("saved", path)
for p in (0.5, 0.7):
    print(f"  p={p}: P(X>=8) = {stats.binom.sf(7, 10, p):.4f}")
print(f"  공정한 동전에서 양쪽을 다 세면 {2*stats.binom.sf(7,10,0.5):.4f}")
