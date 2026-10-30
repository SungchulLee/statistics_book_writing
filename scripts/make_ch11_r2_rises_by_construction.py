"""11.6절 회귀 진단 — 잔차가 큰 점을 빼면 R² 는 반드시 오른다.

King County 98105 구역 313 건에 다섯 설명변수로 회귀하면 R² = 0.7954 다.
쿡 거리가 4/n 을 넘는 스무 건을 빼고 다시 적합하면 0.8415 로 오르는데,
그 상승은 모형이 나아진 증거가 아니다 — R² = 1 - SSE/SST 에서 **잔차가 큰
점을 골라 뺐으므로** 분자가 분모보다 빠르게 준다.

그것을 보이는 길은 대조군이다. 같은 수를 **무작위로** 빼면 R² 가 꿈쩍하지
않는다. 왼쪽이 그 두 곡선이고, 오른쪽이 SSE 와 SST 가 각각 얼마나 주는지다.
"""

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

import numpy as np
import pandas as pd
import statsmodels.api as sm

OUT = "docs/ch11/diagnostics/img/"
URL = ("https://raw.githubusercontent.com/gedeck/"
       "practical-statistics-for-data-scientists/"
       "8a6d3bb6468e979c861d4b37215e1413702dfdfa/data/house_sales.csv")
PRED = ['SqFtTotLiving', 'SqFtLot', 'Bathrooms', 'Bedrooms', 'BldgGrade']
MAXDROP, RANDREP = 40, 200
rng = np.random.default_rng(11)

house = pd.read_csv(URL, sep='\t')
h = house.loc[house.ZipCode == 98105, :]
X = h[PRED].assign(const=1).to_numpy(float)
y = h['AdjSalePrice'].to_numpy(float)
n = len(y)

fit = sm.OLS(y, X).fit()
D = fit.get_influence().cooks_distance[0]
thr = 4 / n
n_over = int((D > thr).sum())


def refit(keep):
    r = sm.OLS(y[keep], X[keep]).fit()
    yy = y[keep]
    sse = float(((yy - r.fittedvalues) ** 2).sum())
    sst = float(((yy - yy.mean()) ** 2).sum())
    return r.rsquared, sse, sst


order = np.argsort(-D)                      # 쿡 거리가 큰 순서
r2_cook, sse_c, sst_c = [], [], []
for m in range(MAXDROP + 1):
    keep = np.ones(n, bool); keep[order[:m]] = False
    a, b, c = refit(keep)
    r2_cook.append(a); sse_c.append(b); sst_c.append(c)

r2_rand = []
for m in range(MAXDROP + 1):
    vals = []
    for _ in range(RANDREP):
        keep = np.ones(n, bool)
        keep[rng.choice(n, m, replace=False)] = False
        vals.append(refit(keep)[0])
    r2_rand.append(np.mean(vals))

ms = np.arange(MAXDROP + 1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.5))

# ── (좌) 고른 제거 대 무작위 제거 ───────────────────────────────────
ax1.plot(ms, r2_cook, "-", color="tab:red", lw=2.6, label="쿡 거리가 큰 순서로")
ax1.plot(ms, r2_rand, "-", color="tab:blue", lw=2.6,
         label=f"무작위로 ({RANDREP} 번 평균)")
ax1.axvline(n_over, color="0.5", ls="--", lw=1.4)
ax1.scatter([0, n_over], [r2_cook[0], r2_cook[n_over]], s=110, c="tab:red",
            zorder=5, edgecolor="white", lw=1.6)
ax1.annotate(f"{r2_cook[0]:.4f}", (0, r2_cook[0]), textcoords="offset points",
             xytext=(8, -16), fontsize=11, fontweight="bold", color="tab:red")
ax1.annotate(f"{r2_cook[n_over]:.4f}", (n_over, r2_cook[n_over]),
             textcoords="offset points", xytext=(8, -4), fontsize=11,
             fontweight="bold", color="tab:red")
ax1.annotate(f"무작위로 같은 {n_over} 건을 빼면 {r2_rand[n_over]:.4f}\n— 꿈쩍하지 않는다",
             (n_over + 8, r2_rand[n_over + 8]), textcoords="offset points",
             xytext=(10, -30), ha="left", fontsize=10.5, color="tab:blue",
             arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1.2))
ax1.text(n_over - 0.5, min(r2_cook) + 0.002, f"쿡 거리 > 4/n 인 {n_over} 건",
         fontsize=9.5, color="0.4", rotation=90, va="bottom", ha="right")
# 처음 몇 건에서는 오히려 내려간다 — 가장 비싼 집이 SST 도 크게 쥐고 있기 때문이다
dip = int(np.argmin(r2_cook[:8]))
ax1.annotate(f"처음 {dip} 건에서는 오히려 내려간다", (dip, r2_cook[dip]),
             textcoords="offset points", xytext=(26, -4), ha="left",
             fontsize=10, color="0.35",
             arrowprops=dict(arrowstyle="->", color="0.5", lw=1.1))

ax1.set_xlabel("뺀 관측의 수")
ax1.set_ylabel("다시 적합한 R²")
ax1.set_title(f"King County 98105 구역 {n} 건 — 무엇을 빼느냐가 전부다",
              fontsize=11.5)
ax1.legend(fontsize=10, loc="lower right", framealpha=0.95)
ax1.grid(alpha=0.25)

# ── (우) 분자와 분모가 주는 속도 ────────────────────────────────────
ax2.plot(ms, np.array(sse_c) / sse_c[0], "-", color="tab:red", lw=2.6,
         label="SSE (분자)")
ax2.plot(ms, np.array(sst_c) / sst_c[0], "-", color="0.45", lw=2.6,
         label="SST (분모)")
ax2.axvline(n_over, color="0.5", ls="--", lw=1.4)
for arr, c, lab in [(sse_c, "tab:red", "SSE"), (sst_c, "0.45", "SST")]:
    v = arr[n_over] / arr[0]
    ax2.scatter([n_over], [v], s=100, c=c, zorder=5, edgecolor="white", lw=1.5)
    ax2.annotate(f"{lab} 는 {(1 - v) * 100:.0f}% 줄었다", (n_over, v),
                 textcoords="offset points", xytext=(10, 0), fontsize=10.5,
                 color=c)
ax2.set_xlabel("뺀 관측의 수 (쿡 거리가 큰 순서로)")
ax2.set_ylabel("처음을 1 로 둔 비")
ax2.set_title("R² = 1 - SSE/SST — 분자가 분모보다 빨리 준다", fontsize=11.5)
ax2.set_ylim(0, 1.05)
ax2.legend(fontsize=10, loc="lower left", framealpha=0.95)
ax2.grid(alpha=0.25)

plt.tight_layout()
path = OUT + "r2_rises_by_construction.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
print("saved", path)

print(f"n = {n}, 쿡 거리 > 4/n = {thr:.5f} 인 관측 {n_over} 건  (쪽: 스무 건)")
print(f"  처음           R² = {r2_cook[0]:.4f}   (쪽: 0.7954)")
print(f"  쿡 순서로 {n_over} 건 제거  R² = {r2_cook[n_over]:.4f}   (쪽: 0.8415)")
print(f"  무작위로  {n_over} 건 제거  R² = {r2_rand[n_over]:.4f}   ← 꿈쩍하지 않는다")
print(f"  SSE {sse_c[n_over] / sse_c[0]:.4f} 배 · SST {sst_c[n_over] / sst_c[0]:.4f} 배")
print(f"  쿡 거리 최댓값 {D.max():.4f}  (쪽: 0.5608)")
