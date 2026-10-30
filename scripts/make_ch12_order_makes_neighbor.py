"""12.4절 독립성 — 늘어놓은 순서 하나가 통계량을 바꾼다.

더빈-왓슨은 **이웃한** 잔차의 차를 재므로, 어떤 순서로 늘어놓느냐가 곧
「이웃」의 정의다. 자기상관이 0.8 인 오차로 만든 자료인데도 자료프레임의
집단순으로 재면 1.8443 이라 「독립이다」로 읽히고, 측정 순서로 되돌리면
0.7927 이다.

견줄 기준도 자료의 짜임이 정한다 — 집단별로 묶인 순서에서는 이 통계량의
중심조차 2 가 아니라 2.0667 이다.

쪽의 코드(seed 7, N = 60, rho = 0.8)를 그대로 다시 돌린다.
"""

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

import numpy as np
import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.stattools import durbin_watson

OUT = "docs/ch12/assumptions/img/"

rng_ar = np.random.default_rng(7)
N, rho_true = 60, 0.8
grp = np.repeat(["A", "B", "C"], 20)
mu = {"A": 10.0, "B": 10.8, "C": 12.0}
order = rng_ar.permutation(N)
eps, x = np.empty(N), 0.0
for t in range(N):
    x = rho_true * x + rng_ar.normal(0, 1)
    eps[t] = x
y = np.empty(N)
for t in range(N):
    y[order[t]] = mu[grp[order[t]]] + eps[t]
df = pd.DataFrame({"group": grp, "response": y})
df["t"] = np.argsort(order)

r = ols("response ~ C(group)", data=df).fit().resid.values
r_time = r[np.argsort(df["t"].values)]
d_grp, d_time = durbin_watson(r), durbin_watson(r_time)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11.6, 6.4), sharey=True)

for ax, (rr, d, name, col) in zip(
        [ax1, ax2],
        [(r, d_grp, "자료프레임의 집단순 — 「독립이다」로 읽힌다", "tab:blue"),
         (r_time, d_time, "측정 순서로 되돌리면 — 자기상관이 드러난다", "tab:red")]):
    ax.plot(np.arange(N), rr, "-o", color=col, lw=1.4, ms=4.2)
    ax.axhline(0, color="0.6", lw=0.9)
    ax.set_ylabel("잔차")
    ax.set_title(f"{name}      d = {d:.4f}", fontsize=11.5, color=col)
    ax.grid(alpha=0.2)

for i, b in enumerate([20, 40]):
    ax1.axvline(b - 0.5, color="0.55", ls="--", lw=1.2)
for i, lab in enumerate(["A", "B", "C"]):
    ax1.text(10 + 20 * i, ax1.get_ylim()[1] * 0.92, f"집단 {lab}", ha="center",
             fontsize=10.5, color="0.4")

ax2.set_xlabel("늘어놓은 자리")
fig.suptitle(f"같은 잔차 {N} 개 · 참 자기상관 {rho_true} — 다른 것은 순서 하나뿐이다",
             fontsize=13, y=1.0)

plt.tight_layout()
path = OUT + "order_makes_neighbor.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
print("saved", path)

print(f"참 자기상관 rho = {rho_true}")
print(f"  집단순 d = {d_grp:.4f}   (쪽: 1.8443)")
print(f"  측정순 d = {d_time:.4f}   (쪽: 0.7927)   기대 2(1-rho) = {2 * (1 - rho_true):.1f}")

