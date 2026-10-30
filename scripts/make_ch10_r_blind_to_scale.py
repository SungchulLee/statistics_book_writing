"""10.2절 회귀직선과 상관 — 상관은 순서만 보고 눈금은 보지 않는다.

예측을 통째로 4 만큼 올려 놓아도 순서는 그대로이니 상관이 정확히 1 인데,
결정계수는 -1.0 이다. 「표적의 평균을 그냥 내놓는 것보다 두 배 나쁘다」는 뜻이다.

쪽의 코드(seed 12, n = 200)를 그대로 다시 돌려 네 줄을 확인한다.
"""

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

import numpy as np
from scipy import stats
from sklearn.metrics import r2_score

OUT = "docs/ch10/visualization/img/"

rng = np.random.default_rng(12)
n = 200
y = rng.normal(10, 3, n)
x = y + rng.normal(0, 2, n)
b, a = np.polyfit(x, y, 1)

CASES = [("OLS 적합값\n(훈련 자료)", a + b * x, "tab:blue"),
         ("평행이동만\ny + 4", y + 4, "tab:red"),
         ("축소된 예측\n0.5y + 5", 0.5 * y + 5, "tab:orange"),
         ("눈금이 어긋난 예측\n2y + 5", 2 * y + 5, "tab:purple")]

rows = [(lab, yh, c, stats.pearsonr(y, yh)[0], r2_score(y, yh))
        for lab, yh, c in CASES]

fig = plt.figure(figsize=(12.6, 6.6))
gs = fig.add_gridspec(2, 4, height_ratios=[1.25, 1], hspace=0.45, wspace=0.3)

# ── (위) 네 예측을 실제값에 맞대어 ───────────────────────────────────
lo = min(y.min(), min(yh.min() for _, yh, _ in CASES)) - 1
hi = max(y.max(), max(yh.max() for _, yh, _ in CASES)) + 1

for i, (lab, yh, c, r, r2) in enumerate(rows):
    ax = fig.add_subplot(gs[0, i])
    ax.plot([lo, hi], [lo, hi], "--", color="0.6", lw=1.4, zorder=1)
    ax.scatter(y, yh, s=9, c=c, alpha=0.55, zorder=2)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_title(f"{lab}\nr = {r:.4f}   R² = {r2:.4f}", fontsize=10,
                 color=c)
    ax.set_xlabel("실제값 y", fontsize=9)
    if i == 0:
        ax.set_ylabel("예측값", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.2)

fig.text(0.5, 0.955, "회색 대각선이 「예측 = 실제」다. 점들이 직선 위에 있으면 "
         "상관은 1 이지만, 그 직선이 대각선이 아니면 결정계수가 무너진다",
         ha="center", va="top", fontsize=11, color="0.3")

# ── (아래) r 과 R² 를 나란히 ─────────────────────────────────────────
ax = fig.add_subplot(gs[1, :])
yy = np.arange(len(rows))[::-1]
for k, (lab, _, c, r, r2) in zip(yy, rows):
    ax.scatter([r], [k + 0.14], s=140, c="tab:green", zorder=4,
               edgecolor="white", lw=1.4)
    ax.scatter([max(r2, -1.6)], [k - 0.14], s=140, c="tab:red", zorder=4,
               marker="D", edgecolor="white", lw=1.4)
    ax.text(r + 0.04, k + 0.14, f"r = {r:.4f}", va="center", ha="left",
            fontsize=10.5, color="tab:green")
    if r2 < -1.6:
        # 축 밖이라 왼쪽에 적으면 잘린다 — 잘린 표식 위에 적는다
        ax.text(-1.6, k - 0.14, f"  R² = {r2:.4f}  (축 밖)", va="bottom",
                ha="left", fontsize=10.5, color="tab:red")
    else:
        ax.text(r2 - 0.04, k - 0.14, f"R² = {r2:.4f}", va="center",
                ha="right", fontsize=10.5, color="tab:red")
    ax.hlines(k, max(r2, -1.6), r, color="0.8", lw=1.4, zorder=1)

ax.axvline(1.0, color="0.75", ls=":", lw=1.2)
ax.axvline(0, color="0.4", lw=1.2)
ax.text(0.03, -0.5, "0 — 표적의 평균을 그냥 내놓는 것과 같다", ha="left",
        va="bottom", fontsize=9.5, color="0.4")
ax.scatter([], [], s=140, c="tab:green", label="상관 r")
ax.scatter([], [], s=140, c="tab:red", marker="D", label="결정계수 R²")

ax.set_yticks(yy, [r[0].replace("\n", " ") for r in rows], fontsize=10)
ax.set_xlim(-1.75, 1.45)
ax.set_ylim(-0.6, len(rows) - 0.35)
ax.set_xlabel("값")
ax.legend(fontsize=10, loc="upper left", framealpha=0.95)
ax.grid(axis="x", alpha=0.25)

path = OUT + "r_blind_to_scale.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
print("saved", path)

print(f"{'예측':28} {'상관 r':>9} {'r^2':>9} {'r2_score':>11}")
for lab, yh, _, r, r2 in rows:
    print(f"{lab.replace(chr(10), ' '):28} {r:9.4f} {r ** 2:9.4f} {r2:11.4f}")
