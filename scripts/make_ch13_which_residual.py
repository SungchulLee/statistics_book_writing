"""13.3절 동질성 잔차 — 잔차를 잘못 고르면 칸이 안 보인다.

표준화 잔차는 분모를 sqrt(E) 로 두는데, 그것은 H0 아래 칸 도수의 분산을
E 로 어림한 것이다. 실제 분산은 그보다 작으므로(주변합이 고정되어 있다)
표준화 잔차는 **체계적으로 작게** 나온다. 조정 잔차가 그 몫을 되돌린다.

같은 칸이 선을 넘고 못 넘는 일이 생기는 자리를 본페로니 임계값과 함께 본다.
"""

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

import numpy as np
from scipy import stats

OUT = "docs/ch13/test/img/"

observed = np.array([[25., 30, 20, 25],
                     [18., 22, 35, 25],
                     [30., 25, 15, 30]])
ROWS, COLS = ["행 1", "행 2", "행 3"], ["열 A", "열 B", "열 C", "열 D"]

row_tot = observed.sum(axis=1, keepdims=True)
col_tot = observed.sum(axis=0, keepdims=True)
tot = observed.sum()
expected = (row_tot @ col_tot) / tot

std = (observed - expected) / np.sqrt(expected)
adj = (observed - expected) / np.sqrt(
    expected * (1 - row_tot / tot) * (1 - col_tot / tot))

ncell = observed.size
crit_b = stats.norm.ppf(1 - 0.05 / (2 * ncell))      # 본페로니
crit_2 = 2.0

labels = [f"{r}·{c}" for r in ROWS for c in COLS]
s, a = std.ravel(), adj.ravel()
order = np.argsort(-np.abs(a))
labels = [labels[i] for i in order]; s, a = s[order], a[order]

fig, ax = plt.subplots(figsize=(11.8, 4.8))
x = np.arange(ncell); w = 0.36
ax.bar(x - w / 2, np.abs(s), w, color="tab:blue", alpha=0.85, label="표준화 잔차")
ax.bar(x + w / 2, np.abs(a), w, color="tab:red", alpha=0.85, label="조정 잔차")

ax.axhline(crit_2, color="0.45", ls=":", lw=1.6)
ax.text(ncell - 0.4, crit_2 - 0.05, "흔히 쓰는 문턱 2", ha="right", va="top",
        fontsize=10, color="0.45")
ax.axhline(crit_b, color="black", ls="--", lw=1.8)
# 왼쪽은 큰 막대와 설명이 쓰므로 오른쪽 빈자리에
ax.text(ncell - 0.4, crit_b + 0.04,
        f"본페로니 임계값 {crit_b:.4f}  (칸 {ncell} 개)", ha="right", va="bottom",
        fontsize=10.5, fontweight="bold")

# 선을 넘고 못 넘는 것이 갈리는 칸
flip2 = [i for i in range(ncell) if (abs(s[i]) < crit_2) != (abs(a[i]) < crit_2)]
for i in flip2:
    ax.annotate("같은 칸인데\n문턱 2 를 하나만 넘는다", (x[i], max(abs(s[i]), abs(a[i]))),
                textcoords="offset points", xytext=(58, 30), ha="left",
                fontsize=10, color="0.25",
                arrowprops=dict(arrowstyle="->", color="0.45", lw=1.2))

ax.set_xticks(x, labels, fontsize=9.5, rotation=45, ha="right")
ax.set_ylabel("잔차의 절댓값")
ax.set_ylim(0, max(crit_b, np.abs(a).max()) * 1.32)
ax.set_title("같은 표, 두 잔차 — 조정 잔차가 늘 더 크다", fontsize=12)
ax.legend(fontsize=10.5, loc="upper right", framealpha=0.95)
ax.grid(axis="y", alpha=0.25)

plt.tight_layout()
path = OUT + "which_residual.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
print("saved", path)

print(f"칸 {ncell} 개 · 본페로니 임계값 {crit_b:.4f}  (쪽: 2.8653)")
print(f"  조정/표준화 의 비  최소 {np.min(np.abs(a / s)):.4f} · "
      f"최대 {np.max(np.abs(a / s)):.4f}  — 늘 1 보다 크다")
print(f"  문턱 2 를 하나만 넘는 칸: {[labels[i] for i in flip2] or '없음'}")
for lab, ss, aa in zip(labels, s, a):
    print(f"    {lab}  표준화 {ss:+.4f}  조정 {aa:+.4f}")
