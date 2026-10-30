"""10.3절 켄달의 타우 — 쌍을 세는 식은 하나인데 분모가 수를 바꾼다.

ROC 곡선 아래 면적은 「무작위로 고른 양성 하나와 음성 하나에서 양성의 점수가
더 높을 확률」이므로 그것도 쌍을 세는 일이다. 같은 자료에서 AUC · Somers D ·
타우-b 가 셋 다 다른 수로 나오는데, 셋을 가르는 것은 세는 방식이 아니라
**무엇으로 나누는가**다.

쪽의 코드(seed 12, n = 300)를 그대로 다시 돌려 수를 확인한다.
(이 쪽의 기존 그림은 타우와 r_s 의 눈금 비교라 겹치지 않는다.)
"""

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

OUT = "docs/ch10/correlation/img/"

rng = np.random.default_rng(12)
n = 300
y = rng.integers(0, 2, n)
s = rng.normal(y * 0.9, 1.0)

auc = roc_auc_score(y, s)
pos, neg = s[y == 1], s[y == 0]
d = pos[:, None] - neg[None, :]
conc, disc, tie = int((d > 0).sum()), int((d < 0).sum()), int((d == 0).sum())
npairs = d.size
somers = (conc - disc) / npairs
tau_b = stats.kendalltau(y, s)[0]

# 타우-b 가 작은 까닭은 양성x음성 쌍의 동점이 아니라 (그것은 0 이다)
# **같은 부류끼리의 쌍**이다. 그 쌍은 y 가 동점이라 타우-b 의 분모 보정에 들어간다.
npos, nneg = int(y.sum()), int(n - y.sum())
all_pairs = n * (n - 1) // 2
same_class = all_pairs - npairs                 # y 가 동점인 쌍
den_auc = npairs                                # 양성x음성 쌍의 수
den_tau = np.sqrt((all_pairs - same_class) * (all_pairs - 0))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.4),
                               gridspec_kw={"width_ratios": [1, 1.12]})

# ── (좌) 모든 쌍을 가른다 ────────────────────────────────────────────
parts = [("일치 (양성 점수가 높다)", conc, "tab:blue"),
         ("불일치 (음성이 높다)", disc, "tab:red"),
         ("같은 부류끼리 — y 가 동점", same_class, "0.78")]
left = 0
for lab, v, c in parts:
    ax1.barh([0], [v], left=left, height=0.46, color=c, alpha=0.88, label=lab)
    ax1.text(left + v / 2, 0, f"{v:,}\n{v / all_pairs * 100:.1f}%",
             ha="center", va="center", fontsize=10.5,
             color="white" if c != "0.78" else "0.25", fontweight="bold")
    left += v

ax1.annotate("", xy=(0, 0.42), xytext=(npairs, 0.42),
             arrowprops=dict(arrowstyle="<->", color="tab:green", lw=1.6))
ax1.text(npairs / 2, 0.46, f"양성 × 음성  {npairs:,}\nAUC 와 Somers D 의 분모",
         ha="center", va="bottom", fontsize=9.5, color="tab:green")
ax1.annotate("", xy=(0, -0.42), xytext=(den_tau, -0.42),
             arrowprops=dict(arrowstyle="<->", color="tab:orange", lw=1.6))
ax1.text(den_tau / 2, -0.46, f"{den_tau:,.0f}  —  타우-b 의 분모\n"
         f"(= √({npairs:,} × {all_pairs:,}), 더 크다)",
         ha="center", va="top", fontsize=9.5, color="tab:orange")

ax1.set_yticks([0], [f"모든 쌍\n{all_pairs:,} 개"], fontsize=11)
ax1.set_xlim(0, all_pairs * 1.02)
ax1.set_ylim(-1.0, 1.0)
ax1.set_xlabel("쌍의 수")
ax1.set_title(f"자료 하나를 쌍으로 세면   (양성 {npos} · 음성 {nneg})",
              fontsize=11.5)
ax1.legend(fontsize=8.5, loc="upper center", ncol=1, framealpha=0.95,
           bbox_to_anchor=(0.5, 1.0))

# ── (우) 같은 셈에서 나온 세 수 ──────────────────────────────────────
ITEMS = [("AUC", auc, f"(일치 + 동점/2) ÷ {den_auc:,}", "눈금 0~1, 우연이 0.5",
          "tab:green"),
         ("Somers D\n= 2·AUC - 1", somers, f"(일치 - 불일치) ÷ {den_auc:,}",
          "눈금 -1~1, 우연이 0", "tab:blue"),
         ("켄달 타우-b", tau_b, f"(일치 - 불일치) ÷ {den_tau:,.0f}",
          "분자는 같고 분모만 크다", "tab:orange")]
y2 = np.arange(len(ITEMS))[::-1]

for yy, (lab, v, denom, note, c) in zip(y2, ITEMS):
    ax2.hlines(yy, 0, v, color=c, lw=3, alpha=0.5)
    ax2.scatter([v], [yy], s=190, c=c, zorder=4, edgecolor="white", lw=1.8)
    ax2.text(v + 0.018, yy + 0.17, f"{v:.10f}", va="center", ha="left",
             fontsize=11.5, fontweight="bold", color=c)
    ax2.text(v + 0.018, yy - 0.12, f"{denom}\n{note}", va="top", ha="left",
             fontsize=9, color="0.45")

ax2.axvline(0, color="0.3", lw=1.2)
ax2.set_yticks(y2, [i[0] for i in ITEMS], fontsize=10.5)
ax2.set_xlim(-0.03, 1.22)
ax2.set_ylim(-0.6, len(ITEMS) - 0.35)
ax2.set_xlabel("값")
ax2.set_title("같은 쌍 세기에서 나온 세 수 — 분자는 하나, 분모가 셋",
              fontsize=11.5)
ax2.grid(axis="x", alpha=0.25)

plt.tight_layout()
path = OUT + "auc_denominators.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
print("saved", path)

print(f"모든 쌍 {all_pairs:,} = 양성x음성 {npairs:,} + 같은 부류끼리 {same_class:,}")
print(f"  양성x음성 {npairs:,} = 일치 {conc:,} + 불일치 {disc:,} + 동점 {tie:,}")
print(f"  분모:  AUC·Somers D 는 {den_auc:,}  ·  타우-b 는 {den_tau:,.1f}")
print(f"  손으로:  ({conc:,} - {disc:,}) / {den_tau:,.1f} = "
      f"{(conc - disc) / den_tau:.10f}")
print(f"  roc_auc_score          {auc:.10f}   (쪽: 0.7447026353)")
print(f"  2*AUC - 1 = Somers D   {somers:.10f}   (쪽: 0.4894052707)")
print(f"  kendalltau  타우-b      {tau_b:.10f}   (쪽: 0.3463625782)")
