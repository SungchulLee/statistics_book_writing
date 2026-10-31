"""9.1절 신뢰구간과 검정의 쌍대성 — 구간은 검정을 한 줄에 모아 둔 것이다.

들머리의 자료 하나(xbar = 52, 표준오차 2)로 mu_0 를 바꿔 가며 검정을
되풀이한다. 기각하지 못하는 mu_0 들을 모으면 그것이 곧 95% 신뢰구간
(48.080072, 55.919928) 이다 --- 한 자리도 어긋나지 않는다. 검정 하나는
그 자 위의 눈금 하나를 묻는 일이다.

쪽의 ci_test_duality 는 둘의 대응을 보이고 ci_not_p 는 연구 넷과 p 값
함수를 그린다. 여기서는 같은 자료에 검정을 수십 번 돌려 그 결과를
자처럼 늘어놓는다.
"""
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
from scipy import stats

OUT = "docs/ch09/foundations/img/"
XBAR, SE, MU0 = 52.0, 2.0, 50.0
Z = stats.norm.ppf(0.975)
LO, HI = XBAR - Z * SE, XBAR + Z * SE

mus = np.arange(44.0, 60.01, 0.4)
zs = (XBAR - mus) / SE
ps = 2 * stats.norm.sf(np.abs(zs))
rej = ps < 0.05

fig, ax = plt.subplots(figsize=(13.0, 4.6))
for m, p, r in zip(mus, ps, rej):
    col = "tab:red" if r else "tab:blue"
    ax.plot([m], [1.0], "o" if not r else "X", ms=10 if r else 9, color=col,
            mew=2.5 if r else 0)
ax.plot([LO, HI], [0.42, 0.42], "-", lw=9.0, color="tab:blue",
        solid_capstyle="butt", alpha=0.85)
for x in (LO, HI):
    ax.plot([x, x], [0.30, 0.54], "-", lw=3.5, color="tab:blue")
    ax.text(x, 0.21, f"{x:.6f}", ha="center", va="top", fontsize=11,
            color="tab:blue", fontweight="bold")
ax.text((LO + HI) / 2, 0.42, "$95\\%$ 신뢰구간", ha="center", va="center",
        fontsize=12.5, color="white", fontweight="bold")
ax.plot([MU0], [1.0], "o", ms=18, mfc="none", mec="0.15", mew=2.6, zorder=5)
ax.annotate(f"$\\mu_0 = {MU0:.0f}$ 에 대한 검정 하나\n$z = 1$, $p = "
            f"{2*stats.norm.sf(1):.6f}$", xy=(MU0, 1.08), xytext=(45.1, 1.62),
            fontsize=11.5, color="0.15", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="0.35", lw=1.6))
ax.plot([], [], "o", ms=9, color="tab:blue", label="기각하지 못한다")
ax.plot([], [], "X", ms=10, color="tab:red", mew=2.5, label="기각한다")
ax.axvline(XBAR, color="0.55", ls="--", lw=1.4)
ax.text(XBAR, 1.78, f"$\\bar x = {XBAR:.0f}$", ha="center", fontsize=11,
        color="0.45")
ax.set_xlim(43.2, 60.8)
ax.set_ylim(0.0, 2.0)
ax.set_yticks([0.42, 1.0], ["구간", "검정"], fontsize=12)
ax.set_xlabel("$\\mu_0$ — 검정이 묻는 값", fontsize=11.5)
ax.set_title("같은 자료로 $\\mu_0$ 를 바꿔 가며 검정한 결과", fontsize=13)
ax.legend(fontsize=10.5, loc="upper right", framealpha=0.95)
ax.grid(axis="x", alpha=0.25)
fig.suptitle("기각하지 못한 값들을 모으면 그것이 곧 신뢰구간이다",
             fontsize=14, y=1.0)
plt.tight_layout()
path = OUT + "ruler_of_tests.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
print("saved", path)
keep = mus[~rej]
print(f"  구간 ({LO:.6f}, {HI:.6f})")
print(f"  기각하지 못한 mu_0 의 범위 [{keep.min():.1f}, {keep.max():.1f}] "
      f"(0.4 간격으로 훑은 것)")
print(f"  mu_0 = {MU0}: z = {(XBAR-MU0)/SE:.4f}, p = "
      f"{2*stats.norm.sf(abs((XBAR-MU0)/SE)):.6f}")
