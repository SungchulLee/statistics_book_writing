"""14.3절 왜도 검정 — 표준오차로 나누기만 하면 꼬리를 과대평가한다.

표본왜도를 그 표준편차로 나눈 단순 표준화 Y 는 n = 300 에서 15.9548 인데,
다고스티노–피어슨 변환은 그것을 10.4038 로 35% 깎는다. 깎는 방향이 맞다 —
유한표본에서 g1 의 분포가 오른쪽으로 치우쳐 있어 단순 표준화가 큰 값을
과대평가하기 때문이다.

변환은 역쌍곡사인이라 큰 인수에서 로그처럼 자라고, 그래서 꼬리를 누른다.
모든 수가 닫힌 꼴이므로 쪽의 값과 자리까지 같아야 한다.
"""

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

import numpy as np
from scipy import stats

OUT = "docs/ch14/descriptive_statistics/img/"


def dagostino(n):
    """변환의 세 상수 — 쪽의 식 그대로."""
    b2 = 3 * (n**2 + 27 * n - 70) * (n + 1) * (n + 3) / (
        (n - 2) * (n + 5) * (n + 7) * (n + 9))
    w2 = -1 + np.sqrt(2 * (b2 - 1))
    delta = 1 / np.sqrt(0.5 * np.log(w2))
    alpha = np.sqrt(2 / (w2 - 1))
    return b2, w2, delta, alpha


N = 300
G1 = 2.233937                                   # 비보정 표본왜도
sd_g1 = np.sqrt(6 * (N - 2) / ((N + 1) * (N + 3)))
Y = G1 / sd_g1
b2, w2, delta, alpha = dagostino(N)
Z1 = delta * np.arcsinh(Y / alpha)

p_simple, p_trans = 2 * stats.norm.sf(Y), 2 * stats.norm.sf(Z1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.5),
                               gridspec_kw={"width_ratios": [1.35, 1]})

# ── (좌) 변환이 꼬리를 누른다 ───────────────────────────────────────
ys = np.linspace(0, 20, 400)
ax1.plot(ys, ys, "--", color="0.6", lw=1.6, label="누르지 않는다면 (Y = Z₁)")
ax1.plot(ys, delta * np.arcsinh(ys / alpha), "-", color="tab:blue", lw=2.6,
         label="다고스티노–피어슨 변환")
ax1.vlines(Y, 0, Z1, color="tab:red", ls=":", lw=1.5)
ax1.hlines(Z1, 0, Y, color="tab:red", ls=":", lw=1.5)
ax1.hlines(Y, 0, Y, color="0.7", ls=":", lw=1.2)
ax1.scatter([Y], [Z1], s=150, c="tab:red", zorder=5, edgecolor="white", lw=1.6)
ax1.scatter([Y], [Y], s=90, c="0.55", zorder=5, edgecolor="white", lw=1.4)
ax1.annotate(f"단순 표준화 {Y:.4f}", (Y, Y), textcoords="offset points",
             xytext=(-10, 6), ha="right", fontsize=10.5, color="0.35")
ax1.annotate(f"변환하면 {Z1:.4f}\n— {(1 - Z1 / Y) * 100:.0f}% 깎인다", (Y, Z1),
             textcoords="offset points", xytext=(-12, -4), ha="right",
             va="top", fontsize=11, color="tab:red", fontweight="bold")
ax1.axhline(1.959964, color="0.45", lw=1.1)
ax1.text(0.3, 2.35, "유의 문턱 1.96", fontsize=9.5, color="0.45")

ax1.set_xlim(0, 20); ax1.set_ylim(0, 20)
ax1.set_xlabel("단순 표준화 Y = g₁ ÷ SD(g₁)")
ax1.set_ylabel("변환한 Z₁")
ax1.set_title(f"역쌍곡사인이라 큰 값일수록 더 눌린다   (n = {N})", fontsize=11.5)
ax1.legend(fontsize=10, loc="upper left", framealpha=0.95)
ax1.grid(alpha=0.25)

# ── (우) 그 35% 가 p 에서는 얼마인가 ────────────────────────────────
items = [("단순 표준화", Y, p_simple, "0.45"), ("변환한 Z₁", Z1, p_trans, "tab:blue")]
y = np.arange(len(items))[::-1]
ax2.barh(y, [-np.log10(p) for _, _, p, _ in items], height=0.45,
         color=[c for *_, c in items], alpha=0.85)
for yy, (nm, z, p, c) in zip(y, items):
    ax2.text(-np.log10(p) + 1.2, yy, f"p = {p:.2e}", va="center", ha="left",
             fontsize=11.5, fontweight="bold", color=c)
    ax2.text(1.5, yy, f"Z = {z:.4f}", va="center", ha="left", fontsize=10,
             color="white", fontweight="bold")
ax2.axvline(-np.log10(0.05), color="tab:red", ls="--", lw=1.6)
ax2.text(-np.log10(0.05) + 0.6, len(items) - 0.4, "α = 0.05", ha="left",
         va="bottom", fontsize=10.5, color="tab:red")
ax2.set_yticks(y, [i[0] for i in items], fontsize=11)
ax2.set_xlim(0, 75)
ax2.set_ylim(-0.55, len(items) - 0.3)
ax2.set_xlabel("-log10(p)  — 길수록 작은 p")   # 아래첨자 0 과 U+2212 는 두부가 된다
ax2.set_title("둘 다 기각하지만 거리가 32 자릿수 다르다", fontsize=11.5)
ax2.grid(axis="x", alpha=0.25)

plt.tight_layout()
path = OUT + "transform_compresses.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
print("saved", path)
print(f"n = {N}, g1 = {G1}")
print(f"  SD(g1) = {sd_g1:.5f}   (쪽: 0.14002)")
print(f"  Y  = {Y:.4f}   (쪽: 15.9548)")
print(f"  b2 = {b2:.6f} · W² = {w2:.6f} · δ = {delta:.6f} · α = {alpha:.6f}")
print(f"     (쪽: 3.110831 · 1.054668 · 6.129875 · 6.048490)")
print(f"  Z1 = {Z1:.6f}   (쪽: 10.403829)")
print(f"  p: 단순 {p_simple:.2e} (쪽: 2.64e-57) · 변환 {p_trans:.2e} (쪽: 2.38e-25)")
