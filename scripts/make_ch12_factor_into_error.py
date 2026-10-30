"""12.3절 웰치 이원배치 — 요인을 빼면 그 변동이 오차로 흘러든다.

온도 세 수준 x 비료 세 종류, 칸마다 하나씩 아홉 관측. 요인을 하나씩 떼어
웰치 분산분석을 두 번 돌리면 둘 다 유의하지 않은데, 같은 아홉 수를 3x3 으로
보고 두 요인을 함께 넣으면 온도가 F = 62 다.

달라진 것은 분모뿐이다 — 일원배치가 쓰던 흩어짐 안에 다른 요인의 효과가
통째로 들어 있었다. 그 몫을 제곱합으로 쪼개 쌓은 막대로 보인다.
"""

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

import numpy as np
import pandas as pd
from scipy import stats

OUT = "docs/ch12/anova_welch/img/"

df = pd.DataFrame({
    "Temperature": ["High"] * 3 + ["Low"] * 3 + ["Medium"] * 3,
    "Fertilizer": ["A", "B", "C"] * 3,
    "Growth": [12., 15, 14, 10, 13, 11, 14, 16, 15],
})
M = df.pivot(index="Temperature", columns="Fertilizer", values="Growth")
y = M.to_numpy(float)
a, b = y.shape
gm = y.mean()

ss_row = b * ((y.mean(axis=1) - gm) ** 2).sum()        # 온도
ss_col = a * ((y.mean(axis=0) - gm) ** 2).sum()        # 비료
ss_tot = ((y - gm) ** 2).sum()
ss_err = ss_tot - ss_row - ss_col                      # 잔차 (교호작용 자리가 없다)

# 일원배치 — 다른 요인을 무시하면 그 요인의 제곱합이 오차로 들어간다
ONE = [("온도만 넣은 일원배치", ss_row, ss_col, "비료의 효과"),
       ("비료만 넣은 일원배치", ss_col, ss_row, "온도의 효과")]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.4),
                               gridspec_kw={"width_ratios": [1.25, 1]})

# ── (좌) 수준 안의 흩어짐이 곧 다른 요인이다 ───────────────────────
# 쪽이 말하는 2.3333 은 합동 MSE 가 아니라 **온도 High 안의 세 값 12·15·14 의
# 분산**이다. 그 셋은 비료 A·B·C 를 하나씩 받았으므로 그 분산이 곧 비료 효과다.
lev = M.index.tolist()
within = M.var(axis=1, ddof=1).to_numpy()
mse2 = ss_err / 4

x0 = np.arange(len(lev))
ax1.bar(x0, within, width=0.5, color="tab:red", alpha=0.85,
        label="온도 수준 안의 분산 — 비료 효과가 들어 있다")
for xi, (lv, v) in zip(x0, zip(lev, within)):
    ax1.text(xi, v + 0.06, f"{v:.4f}", ha="center", va="bottom", fontsize=11,
             fontweight="bold", color="tab:red")
    ax1.text(xi, 0.08, "  ".join(f"{z:.0f}" for z in M.loc[lv]), ha="center",
             va="bottom", fontsize=9.5, color="white", fontweight="bold")

ax1.axhline(mse2, color="tab:blue", lw=2.4)
# 막대 안의 자료값과 겹치지 않게 선 아래 왼쪽에
ax1.text(-0.45, mse2 - 0.06, f"두 요인을 함께 넣은 뒤의 MSE = {mse2:.4f}",
         ha="left", va="top", fontsize=10.5, color="tab:blue", fontweight="bold")
ax1.annotate("", xy=(0.38, within[0]), xytext=(0.38, mse2),
             arrowprops=dict(arrowstyle="<->", color="0.3", lw=1.5))
ax1.text(0.45, (within[0] + mse2) / 2, f"{within[0] / mse2:.0f} 배", ha="left",
         va="center", fontsize=12, fontweight="bold", color="0.2")

ax1.set_xticks(x0, [f"{l}\n(비료 A·B·C 를 하나씩)" for l in lev], fontsize=10)
ax1.set_ylim(-0.22, 2.95)
ax1.set_ylabel("분산")
ax1.set_title("일원배치가 분모로 쓰는 흩어짐 안에 무엇이 있나", fontsize=11.5)
ax1.legend(fontsize=9.5, loc="upper center", framealpha=0.95)
ax1.grid(axis="y", alpha=0.25)

# ── (우) 그 분모가 F 를 정한다 ──────────────────────────────────────
f_row = (ss_row / 2) / mse2
f_col = (ss_col / 2) / mse2
labels = ["온도", "비료"]
f_two = [f_row, f_col]
def welch_f(groups):
    """쪽이 손으로 적은 웰치 F — 집단마다 자기 분산을 쓴다."""
    k = len(groups)
    n = np.array([len(g) for g in groups], float)
    m = np.array([g.mean() for g in groups])
    s2 = np.array([g.var(ddof=1) for g in groups])
    w = n / s2
    W = w.sum()
    Lam = (((1 - w / W) ** 2) / (n - 1)).sum()
    num = (w * (m - (w * m).sum() / W) ** 2).sum() / (k - 1)
    den = 1 + 2 * (k - 2) / (k**2 - 1) * Lam
    return num / den, (k**2 - 1) / (3 * Lam)


# 쪽이 쓴 것은 웰치 일원배치다 — 요인을 하나만 넣고 돌린다
f_one, d2_one = [], []
for fac in ("Temperature", "Fertilizer"):
    gs = [v.to_numpy(float) for _, v in df.groupby(fac)["Growth"]]
    F, d2 = welch_f(gs)
    f_one.append(F); d2_one.append(d2)

x = np.arange(2); w = 0.34
ax2.bar(x - w / 2, f_one, w, color="tab:red", alpha=0.85,
        label="요인을 하나만 (웰치 일원배치)")
ax2.bar(x + w / 2, f_two, w, color="tab:blue", alpha=0.85, label="두 요인을 함께")
for xi, (a1, a2) in zip(x, zip(f_one, f_two)):
    ax2.text(xi - w / 2, a1 + 1.5, f"F = {a1:.4f}", ha="center", va="bottom",
             fontsize=10, color="tab:red")
    ax2.text(xi + w / 2, a2 + 1.5, f"F = {a2:.2f}", ha="center", va="bottom",
             fontsize=11, fontweight="bold", color="tab:blue")
ax2.axhline(stats.f.ppf(0.95, 2, 4), color="0.4", ls="--", lw=1.4)
ax2.text(-0.45, stats.f.ppf(0.95, 2, 4) + 1.5,
         f"두 요인을 함께 넣었을 때의 유의 문턱 {stats.f.ppf(0.95, 2, 4):.2f} (df 2, 4)",
         ha="left", va="bottom", fontsize=9.5, color="0.4")
ax2.set_xticks(x, labels, fontsize=11.5)
ax2.set_ylim(0, 82)
ax2.set_ylabel("F 통계량")
ax2.set_title("분자는 그대로, 분모만 바뀐다", fontsize=11.5)
ax2.legend(fontsize=9.5, loc="upper right", framealpha=0.95)
ax2.grid(axis="y", alpha=0.25)

plt.tight_layout()
path = OUT + "factor_into_error.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
print("saved", path)

print(f"SS 온도 {ss_row:.4f} · SS 비료 {ss_col:.4f} · SS 오차 {ss_err:.4f} "
      f"· SS 전체 {ss_tot:.4f}")
print(f"이원배치  MSE = {ss_err:.4f}/4 = {mse2:.4f}   (쪽: 0.1667)")
print(f"온도 수준 안의 분산 {np.round(M.var(axis=1, ddof=1).to_numpy(), 4)}  "
      f"(쪽이 말하는 2.3333 은 High·Low 의 것)")
print(f"  합동 일원배치 MSE = {(ss_err + ss_col) / 6:.4f}  ·  2.3333/{mse2:.4f} = {2.3333333 / mse2:.1f} 배")
print(f"  이원 F: 온도 {f_row:.4f} (쪽: 62.0000) · 비료 {f_col:.4f} (쪽: 32.0000)")
print(f"  웰치 일원 F: 온도 {f_one[0]:.6f} (쪽: 5.173804) · 비료 {f_one[1]:.6f}")
print(f"    p: 온도 {stats.f(2, d2_one[0]).sf(f_one[0]):.6f} (쪽: 0.081821) · "
      f"비료 {stats.f(2, d2_one[1]).sf(f_one[1]):.6f} (쪽: 0.334719)")
print(f"  p: 온도 {stats.f(2, 4).sf(f_row):.6f} (쪽: 0.000977)")
