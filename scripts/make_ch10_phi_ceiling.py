"""11.1절 특수한 상관 — 0.301 을 1 에 대어 읽으면 안 된다.

쪽의 보기 2 자료다. 운동하는 90 명 중 60 명(67%)과 운동하지 않는 110 명
중 40 명(36%)이 잘 잔다고 답했다. 거의 두 배 차이인데 파이 계수는
0.301 이다. 그런데 이 표의 주변합이 천장을 정해 버려서 파이가 아무리
커도 0.9045 를 넘지 못한다. 그러니 0.301 은 1 분의 0.3 이 아니라
**갈 수 있는 데까지의 3 분의 1** 이다.

쪽의 binary_ceilings 는 천장이 생기는 꼴을 곡선으로, phi_vs_q 는 오즈비와
견준다. 여기서는 그 천장에 대어 읽는 일만 본다.
"""
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
from matplotlib.patches import Rectangle

OUT = "docs/ch11/correlation/img/"
O = np.array([[60.0, 30.0], [40.0, 70.0]])      # 행 = 운동/비운동
r1, r2 = O.sum(1)
c1, c2 = O.sum(0)
n = O.sum()


def phi(t):
    return (t[0, 0] * t[1, 1] - t[0, 1] * t[1, 0]) / \
        np.sqrt(t.sum(1)[0] * t.sum(1)[1] * t.sum(0)[0] * t.sum(0)[1])


PHI = phi(O)
a_max = min(r1, c1)
MX = np.array([[a_max, r1 - a_max], [c1 - a_max, n - r1 - c1 + a_max]])
PHI_MAX = phi(MX)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 4.6),
                               gridspec_kw={"width_ratios": [1, 1.2]})

p1, p2 = O[0, 0] / r1, O[1, 0] / r2
ax1.bar([0, 1], [100 * p1, 100 * p2], width=0.46,
        color=["tab:blue", "0.65"], alpha=0.88)
for x, p, k, tot in ((0, p1, O[0, 0], r1), (1, p2, O[1, 0], r2)):
    ax1.text(x, 100 * p + 2.0, f"{100*p:.0f}%", ha="center", fontsize=15,
             fontweight="bold", color="0.15")
    ax1.text(x, 4, f"{k:.0f} / {tot:.0f}", ha="center", fontsize=12,
             color="white", fontweight="bold")
ax1.annotate("", xy=(1.34, 100 * p2), xytext=(1.34, 100 * p1),
             arrowprops=dict(arrowstyle="<->", color="tab:red", lw=2.0))
ax1.text(1.40, 100 * (p1 + p2) / 2, f"{p1/p2:.2f} 배", ha="left",
         va="center", fontsize=13, color="tab:red", fontweight="bold")
ax1.set_xticks([0, 1], ["운동한다", "운동하지 않는다"], fontsize=12)
ax1.set_xlim(-0.55, 2.0)
ax1.set_ylim(0, 82)
ax1.set_ylabel("잘 잔다고 답한 비율 (%)", fontsize=11)
ax1.set_title("표에 보이는 차이", fontsize=12.5)
ax1.grid(axis="y", alpha=0.25)

for i, (hi, name, col) in enumerate(
        [(1.0, "$0$ 부터 $1$ 까지의 자로 재면", "0.6"),
         (PHI_MAX, f"이 표가 갈 수 있는 데까지의 자로 재면", "tab:blue")]):
    y = 1 - i
    ax2.add_patch(Rectangle((0, y - 0.16), hi, 0.32, facecolor=col,
                            alpha=0.28, edgecolor=col, lw=1.6))
    ax2.plot([PHI, PHI], [y - 0.22, y + 0.22], "-", lw=4.0, color="tab:red")
    ax2.text(PHI, y + 0.29, f"$\\varphi$ = {PHI:.4f}", ha="center",
             fontsize=12, color="tab:red", fontweight="bold")
    ax2.text(hi + 0.02, y, f"{hi:.4f}", ha="left", va="center", fontsize=12,
             color=col if col != "0.6" else "0.35", fontweight="bold")
    ax2.text(0, y - 0.30, name, ha="left", va="top", fontsize=11.5,
             color="0.25")
    ax2.text(PHI / 2, y, f"{PHI/hi:.0%}", ha="center", va="center",
             fontsize=13, color="0.15", fontweight="bold")
ax2.set_xlim(-0.03, 1.14)
ax2.set_ylim(-0.6, 1.6)
ax2.set_yticks([])
ax2.set_xlabel("파이 계수", fontsize=11)
ax2.set_title("같은 $0.301$ 인데 어느 자에 대느냐가 다르다", fontsize=12.5)
ax2.grid(axis="x", alpha=0.25)

fig.suptitle("$0.301$ 은 「약한 연관」이 아니라 "
             "갈 수 있는 데까지의 $3$ 분의 $1$ 이다", fontsize=14, y=1.0)
plt.tight_layout()
path = OUT + "phi_ceiling.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
print("saved", path)
print(f"  표 {O.astype(int).tolist()}  행합 {r1:.0f}/{r2:.0f}  "
      f"열합 {c1:.0f}/{c2:.0f}")
print(f"  비율 {p1:.4f} 대 {p2:.4f}  ({p1/p2:.4f} 배)")
print(f"  파이 {PHI:.6f} · 같은 주변합에서의 최대 {PHI_MAX:.6f} "
      f"(그 표 {MX.astype(int).tolist()})")
print(f"  천장 대비 {PHI/PHI_MAX:.4f}")
