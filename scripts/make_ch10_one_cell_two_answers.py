"""10.6절 쌍 그림 — 한 칸이 두 가지 답을 들고 있다.

들머리의 마지막 두 요점을 그린다.

왼쪽: 꽃받침 길이-너비 칸에는 r = -0.1176 이 적히는데, 품종별로 따로
재면 +0.7425, +0.5259, +0.4572 로 셋 다 양수다. 품종을 섞어 재면 그
양수 셋이 음수 하나로 바뀐다. 격자의 칸 하나가 두 가지 답을 들고 있고,
어느 쪽을 읽을지는 **품종을 색으로 입혔는가**가 가른다.

오른쪽: 그래서 격자를 읽는 순서가 「수를 읽는 것」이 아니라 「가르는
칸을 찾는 것」이다. 네 변수 각각에 칼을 한 번만 대어 versicolor 와
virginica 를 갈라 보면 꽃잎 너비가 0.06, 꽃받침 너비가 0.37 이다 ---
동전 던지기가 0.5 이니 꽃받침 너비는 거의 쓸모가 없다. setosa 는 어느
변수로도 쉽게 갈리므로 어려운 짝만 본다.

쪽의 pairplots_37·_64·_85·_180 은 격자 전체(기본·추림·상관 표시·회귀선)를
그린다. 여기서는 그 넷이 건드리지 않는 것 --- 칸 하나 안에서 답이 갈리는
것과, 칸마다의 가르는 힘 --- 만 본다.
"""

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

import numpy as np
from scipy import stats
from sklearn.datasets import load_iris

OUT = "docs/ch10/visualization/img/"

d = load_iris()
X, y = d.data, d.target
KOR = ["꽃받침 길이", "꽃받침 너비", "꽃잎 길이", "꽃잎 너비"]
SPECIES = ["setosa", "versicolor", "virginica"]
COLS = ["tab:blue", "tab:orange", "tab:green"]
HARD = (1, 2)                      # 갈라내기 어려운 두 품종


def best_cut(v, a, b):
    """한 칼로 두 품종을 가를 때의 가장 낮은 오분류율과 그 자리."""
    va, vb = v[y == a], v[y == b]
    pts = np.unique(np.concatenate([va, vb]))
    cuts = np.concatenate([[pts[0] - 0.5], (pts[:-1] + pts[1:]) / 2,
                           [pts[-1] + 0.5]])
    best, where = 1.0, cuts[0]
    for c in cuts:
        for sg in (1, -1):
            err = ((sg * va > sg * c).sum() + (sg * vb <= sg * c).sum()) \
                / (len(va) + len(vb))
            if err < best:
                best, where = err, c
    return best, where


r_all = stats.pearsonr(X[:, 0], X[:, 1]).statistic
r_each = [stats.pearsonr(X[y == k, 0], X[y == k, 1]).statistic
          for k in range(3)]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 5.1),
                               gridspec_kw={"width_ratios": [1.1, 1]})

# ── (좌) 한 칸, 네 개의 직선 ───────────────────────────────────────
gx = np.linspace(X[:, 0].min() - 0.2, X[:, 0].max() + 0.2, 2)
sl, ic = np.polyfit(X[:, 0], X[:, 1], 1)
ax1.plot(gx, sl * gx + ic, "-", lw=3.4, color="0.25", zorder=4,
         label=f"품종을 섞어 재면  r = {r_all:+.4f}")
for k, (name, col) in enumerate(zip(SPECIES, COLS)):
    m = y == k
    ax1.scatter(X[m, 0], X[m, 1], s=34, color=col, alpha=0.65,
                edgecolors="white", linewidths=0.5)
    s, i = np.polyfit(X[m, 0], X[m, 1], 1)
    xx = np.linspace(X[m, 0].min(), X[m, 0].max(), 2)
    ax1.plot(xx, s * xx + i, "-", lw=2.6, color=col, zorder=3,
             label=f"{name}  r = {r_each[k]:+.4f}")
ax1.set_xlabel(f"{KOR[0]} (cm)", fontsize=11)
ax1.set_ylabel(f"{KOR[1]} (cm)", fontsize=11)
ax1.set_title("쌍 그림의 한 칸 — 꽃받침 길이와 너비", fontsize=12.5)
ax1.legend(fontsize=10, loc="upper right", framealpha=0.95)
ax1.grid(alpha=0.25)
ax1.text(0.02, 0.03, "품종을 색으로 입히기 전에는 회색 직선 하나만 보인다\n"
         "— 양수 셋이 음수 하나로 바뀐다", transform=ax1.transAxes,
         ha="left", va="bottom", fontsize=11, fontweight="bold",
         color="0.15",
         bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.94,
                   edgecolor="0.5", lw=1.0))

# ── (우) 칼을 한 번만 대면 ─────────────────────────────────────────
errs = [best_cut(X[:, j], *HARD) for j in range(4)]
order = np.argsort([e for e, _ in errs])
ypos = np.arange(4)[::-1]
for yy, j in zip(ypos, order):
    err, cut = errs[j]
    col = "tab:green" if err < 0.15 else "tab:red" if err > 0.3 else "0.45"
    ax2.barh(yy, 100 * err, height=0.52, color=col, alpha=0.85)
    ax2.text(100 * err + 1.2, yy, f"{100 * err:.0f}%", va="center",
             fontsize=12, fontweight="bold", color=col)
ax2.axvline(50, color="0.2", ls="--", lw=2.0)
ax2.text(49, 3.45, "동전 던지기", ha="right", va="top", fontsize=11,
         color="0.2", fontweight="bold")
ax2.set_yticks(ypos, [KOR[j] for j in order], fontsize=12)
ax2.set_xlim(0, 58)
ax2.set_ylim(-0.6, 3.6)
ax2.set_xlabel("한 칼로 가를 때 틀리는 비율 (%)", fontsize=11)
ax2.set_title(f"{SPECIES[HARD[0]]} 와 {SPECIES[HARD[1]]} 를 갈라 보면",
              fontsize=12.5)
ax2.grid(axis="x", alpha=0.25)
ax2.text(0.97, 0.76, "격자에서 찾아야 하는 것은 수가 아니라\n가르는 칸이다",
         transform=ax2.transAxes,
         ha="right", va="top", fontsize=11, fontweight="bold",
         color="0.15",
         bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.94,
                   edgecolor="0.5", lw=1.0))

fig.suptitle("같은 칸이 두 가지 답을 들고 있다 — 품종을 입히면 부호가 뒤집힌다",
             fontsize=14, y=1.0)
plt.tight_layout()
path = OUT + "one_cell_two_answers.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
print("saved", path)

print(f"\n붓꽃 {len(X)} 송이 · {KOR[0]}와 {KOR[1]}")
print(f"  품종을 섞어 재면  r = {r_all:+.4f}")
for k, name in enumerate(SPECIES):
    print(f"  {name:<11s}      r = {r_each[k]:+.4f}")
print(f"\n{KOR[2]}: {SPECIES[0]} 최대 {X[y == 0, 2].max():.1f} · "
      f"나머지 최소 {X[y != 0, 2].min():.1f} — 그 사이가 완전히 빈다")
print(f"\n{SPECIES[HARD[0]]} 와 {SPECIES[HARD[1]]} 를 한 칼로 가를 때")
for j in order:
    err, cut = errs[j]
    print(f"  {KOR[j]:<8s} 자른 자리 {cut:.2f} cm  →  틀리는 비율 "
          f"{100 * err:.0f}%")
