"""9.6절 분석의 자유 — 보고된 p 값이 0.05 바로 아래에 쌓인다.

효과가 없는 자료에서 정직하게 한 번만 검정하면 p 는 [0,1] 에 고르게
떨어진다. 그런데 결과변수를 둘 재어 두고 더 나은 쪽을 쓰면서 표본을
열 명씩 늘려 가다 p < 0.05 가 되는 순간 멈추면, 보고되는 p 가 0.05 **바로
아래**에 쌓인다. 효과는 끝까지 0 이고 어느 검정도 틀리게 계산되지 않았다.

쪽의 researcher_dof 는 자유를 하나씩 누적했을 때의 거짓양성률을 막대로
그리고 optional_stopping 은 엿보며 멈추는 일을 다룬다. 여기서는 그 결과
**보고되는 p 값의 모양**을 본다.
"""
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
from scipy import stats

OUT = "docs/ch09/multiple_testing/img/"
R, N0, STEP, LOOKS, A = 20_000, 20, 10, 5, 0.05
rng = np.random.default_rng(2011)

honest = np.empty(R)
hacked = np.empty(R)
for r in range(R):
    a = rng.standard_normal(N0 + STEP * (LOOKS - 1))
    b = rng.standard_normal(N0 + STEP * (LOOKS - 1))
    y2 = rng.standard_normal(N0 + STEP * (LOOKS - 1))   # 둘째 결과변수
    honest[r] = stats.ttest_ind(a[:N0], b[:N0]).pvalue
    best = 1.0
    for i in range(LOOKS):
        n = N0 + STEP * i
        p1 = stats.ttest_ind(a[:n], b[:n]).pvalue
        p2 = stats.ttest_ind(a[:n], y2[:n]).pvalue
        best = min(p1, p2)
        if best < A:
            break
    hacked[r] = best

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 4.5), sharey=True)
bins = np.linspace(0, 1, 41)
for ax, (v, name, col) in zip((ax1, ax2),
                              [(honest, "정직하게 한 번만 검정하면", "0.62"),
                               (hacked, "자유 둘을 쓰면", "tab:red")]):
    cnt, _, patches = ax.hist(v, bins=bins, color=col, alpha=0.85)
    for pat, left in zip(patches, bins[:-1]):
        if left < A:
            pat.set_facecolor("tab:red" if col != "0.62" else "0.3")
    ax.axvline(A, color="0.15", ls="--", lw=2.0)
    ax.text(A + 0.015, cnt.max() * 0.96, f"$\\alpha$ = {A}", fontsize=11,
            color="0.15", fontweight="bold")
    ax.set_xlabel("보고된 $p$ 값", fontsize=11)
    ax.set_title(f"{name} — $p < {A}$ 가 {(v < A).mean():.4f}",
                 fontsize=12.5, color="0.25" if col == "0.62" else col)
    ax.grid(axis="y", alpha=0.2)
ax1.set_ylabel(f"연구 수 (되풀이 {R:,})", fontsize=11)
ax2.annotate("$0.05$ 아래에 쌓인다", xy=(0.03, 2300), xytext=(0.30, 1900),
             fontsize=13, color="tab:red", fontweight="bold",
             arrowprops=dict(arrowstyle="->", color="tab:red", lw=2.0))
fig.suptitle("효과는 끝까지 $0$ 이고 어느 검정도 틀리게 계산되지 않았다",
             fontsize=14, y=1.0)
plt.tight_layout()
path = OUT + "p_curve_piles.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
print("saved", path)
print(f"  정직: p<0.05 비율 {(honest < A).mean():.4f}")
print(f"  자유 둘: p<0.05 비율 {(hacked < A).mean():.4f}")
print(f"  자유 둘에서 0.04~0.05 에 든 비율 "
      f"{np.mean((hacked >= 0.04) & (hacked < A)):.4f}")
