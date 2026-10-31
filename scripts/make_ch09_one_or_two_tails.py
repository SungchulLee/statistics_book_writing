"""9.1절 검정통계량과 p 값 — 같은 자료, 두 개의 p.

동전을 100 번 던져 앞면이 60 번 나왔다. 「이만큼 극단적인 일」을 한쪽으로
세면 0.0284 이고 양쪽으로 세면 0.0569 다. 자료는 하나인데 수가 두 배
다르고, 어느 쪽을 세는지는 자료가 아니라 **물음**이 정한다.

쪽의 pvalue_tail_and_uniform 은 연속인 귀무분포에서 꼬리넓이와 균등성을
보이고 coin_toss_simulation_100 은 Bin(30, 0.5) 의 히스토그램이다.
여기서는 같은 관측값에 자를 두 번 대는 것만 본다.
"""
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
from scipy import stats

OUT = "docs/ch09/foundations/img/"
N, K = 100, 60
k = np.arange(N + 1)
pmf = stats.binom.pmf(k, N, 0.5)
one = stats.binom.sf(K - 1, N, 0.5)
two = one + stats.binom.cdf(N - K, N, 0.5)

fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.4), sharey=True)
SETS = [(k >= K, f"한쪽만 센다 — 「앞면이 더 많이 나왔나」", one, "tab:blue"),
        ((k >= K) | (k <= N - K), "양쪽을 센다 — 「한쪽으로 치우쳤나」", two,
         "tab:red")]
for ax, (mask, name, val, col) in zip(axes, SETS):
    ax.bar(k, pmf, width=1.0, color="0.80")
    ax.bar(k[mask], pmf[mask], width=1.0, color=col, alpha=0.9)
    ax.axvline(K, color="0.15", lw=1.8)
    ax.text(K + 1.2, pmf.max() * 0.95, f"관측 {K}", fontsize=11,
            color="0.15", fontweight="bold")
    ax.text(0.03, 0.80, f"$p$ = {val:.4f}", transform=ax.transAxes,
            fontsize=16, fontweight="bold", color=col)
    ax.set_xlim(30, 70)
    ax.set_xlabel("백 번 가운데 앞면이 나온 수", fontsize=11)
    ax.set_title(name, fontsize=12.5, color=col)
    ax.grid(axis="y", alpha=0.2)
axes[0].set_ylabel("공정한 동전 아래의 확률", fontsize=11)
fig.suptitle("자료는 하나인데 $p$ 가 두 배 다르다 — 무엇을 「극단」으로 "
             "칠지가 정해 주기 때문이다", fontsize=13.5, y=1.0)
plt.tight_layout()
path = OUT + "one_or_two_tails.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
print("saved", path)
print(f"  n={N}, 관측 {K}: 단측 {one:.6f} · 양측 {two:.6f} · 비 {two/one:.4f}")
