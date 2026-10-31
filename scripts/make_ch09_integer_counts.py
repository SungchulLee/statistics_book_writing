"""9.2절 p 에 대한 z 검정 — 성공 횟수가 정수라는 사실.

들머리의 보기를 그린다. 동전을 다섯 번 던져 H0: p = 0.5 를 alpha = 0.05 로
검정한다. 가능한 결과가 여섯뿐이고 그 여섯의 정확 양측 p 값 가운데 가장
작은 값이 이미 0.0625 라 0.05 아래로 내려가는 것이 하나도 없다 ---
정확검정의 기각역이 비어 실제 크기가 0 이다. 같은 자리에서 정규근사는
양 끝을 기각해 실제 크기가 0.0625 로 명목을 넘는다.

쪽의 discrete_level_sawtooth 는 n 을 훑어 톱니를 보이고
binomial_normal_condition 은 np 가 작을 때의 근사를 다룬다. 여기서는
n = 5 한 자리에서 여섯 결과를 하나씩 들여다본다.
"""
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
from scipy import stats

OUT = "docs/ch09/one_sample_tests/img/"
N, P0, A = 5, 0.5, 0.05
k = np.arange(N + 1)
pmf = stats.binom.pmf(k, N, P0)
exact = np.array([min(1.0, 2 * min(stats.binom.cdf(kk, N, P0),
                                   stats.binom.sf(kk - 1, N, P0)))
                  for kk in k])
se = np.sqrt(P0 * (1 - P0) / N)
z = (k / N - P0) / se
approx = 2 * stats.norm.sf(np.abs(z))
size_exact = pmf[exact <= A].sum()
size_approx = pmf[approx <= A].sum()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 4.6), sharey=True)
for ax, (vals, name, size, col) in zip(
        (ax1, ax2),
        [(exact, "정확검정 (이항)", size_exact, "tab:blue"),
         (approx, "정규근사", size_approx, "tab:red")]):
    bars = ax.bar(k, vals, width=0.68,
                  color=[col if v <= A else "0.72" for v in vals], alpha=0.9)
    for kk, v in zip(k, vals):
        ax.text(kk, v + 0.022, f"{v:.4f}", ha="center", fontsize=10.5,
                fontweight="bold", color=col if v <= A else "0.3")
    ax.axhline(A, color="0.15", ls="--", lw=2.0)
    ax.text(5.45, A + 0.02, f"$\\alpha$ = {A}", ha="right", fontsize=11,
            color="0.15", fontweight="bold")
    ax.set_xticks(k)
    ax.set_xlabel("다섯 번 가운데 앞면이 나온 수", fontsize=11)
    ax.set_ylim(0, 1.18)
    ax.set_title(f"{name} — 실제 크기 {size:.4f}", fontsize=12.5, color=col)
    ax.grid(axis="y", alpha=0.2)
ax1.set_ylabel("양측 $p$ 값", fontsize=11)
ax1.text(2.5, 0.92, "$0.05$ 선 아래로 내려오는 것이\n**하나도 없다**"
         .replace("**", ""), ha="center", fontsize=12, color="tab:blue",
         fontweight="bold")
ax2.text(2.5, 0.92, "양 끝 둘이 선 아래로 내려온다", ha="center",
         fontsize=12, color="tab:red", fontweight="bold")
fig.suptitle("가능한 결과가 여섯뿐이면 약속한 $0.05$ 를 맞출 수가 없다",
             fontsize=14, y=1.0)
plt.tight_layout()
path = OUT + "integer_counts.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
print("saved", path)
for kk, pm, e, a in zip(k, pmf, exact, approx):
    print(f"  k={kk}: 확률 {pm:.5f} · 정확 p {e:.4f} · 근사 p {a:.4f}")
print(f"  실제 크기 — 정확검정 {size_exact:.4f} · 정규근사 {size_approx:.4f}")
