"""9.5절 두 오류 — 손잡이가 하나뿐이다.

들머리의 경보 장치를 그린다. H0: N(0,1), H1: N(delta,1) 에서 문턱을 옮기면
두 잘못이 함께 움직인다. 1000 번 돌린 결과를 문턱 셋에서 개수로 센다 ---
예민하게 두면 헛경보가 늘고 둔감하게 두면 놓침이 는다.

쪽의 alpha_beta_tradeoff 는 문턱을 연속으로 옮긴 곡선이고
type12_error_visualization 은 두 분포 위의 두 넓이를 보이며
petrov_threshold 는 대가와 사전확률로 최적 문턱을 푼다. 여기서는 같은
일을 **세어서** 본다 --- 1000 번에 몇 번인지를.
"""
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
from scipy import stats

OUT = "docs/ch09/errors_and_power/img/"
DELTA, R = 2.5, 1000
CUTS = [(1.0, "예민하게"), (1.645, "들머리의 자리"), (2.5, "둔감하게")]
rng = np.random.default_rng(1983)
quiet = rng.standard_normal(R)              # 아무 일도 없는 밤
real = rng.standard_normal(R) + DELTA       # 진짜 일이 있는 밤

fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.4), sharey=True)
rows = []
for ax, (c, name) in zip(axes, CUTS):
    fp = int((quiet > c).sum())
    fn = int((real <= c).sum())
    a, b = stats.norm.sf(c), stats.norm.cdf(c - DELTA)
    rows.append((c, name, fp, fn, a, b))
    ax.bar([0, 1], [fp, fn], width=0.5, color=["tab:red", "tab:blue"],
           alpha=0.88)
    for x, v, col in ((0, fp, "tab:red"), (1, fn, "tab:blue")):
        ax.text(x, v + 9, f"{v} 번", ha="center", fontsize=13,
                fontweight="bold", color=col)
    ax.set_xticks([0, 1], ["없는데 울린다", "있는데 안 울린다"], fontsize=11)
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0, 420)
    ax.set_title(f"문턱 {c} — {name}\n"
                 f"$\\alpha$ = {a:.3f},  $\\beta$ = {b:.3f}", fontsize=12)
    ax.grid(axis="y", alpha=0.25)
axes[0].set_ylabel(f"{R:,} 번 가운데", fontsize=11)
axes[1].annotate("", xy=(1.0, 330), xytext=(0.0, 330),
                 arrowprops=dict(arrowstyle="->", color="0.3", lw=2.0))
axes[1].text(0.5, 345, "한쪽을 줄이면 다른 쪽이 는다", ha="center",
             fontsize=11.5, color="0.2", fontweight="bold")
fig.suptitle("손잡이는 하나뿐이고, 그것이 두 잘못을 함께 정한다",
             fontsize=14, y=1.0)
plt.tight_layout()
path = OUT + "one_knob_two_errors.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
print("saved", path)
for c, name, fp, fn, a, b in rows:
    print(f"  문턱 {c:<6} {name:<10s} 헛경보 {fp:>3d} · 놓침 {fn:>3d} "
          f"(이론 alpha {a:.4f}, beta {b:.4f})")
