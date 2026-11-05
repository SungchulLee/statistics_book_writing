"""11.6절 상관 타원 — 작은 상관은 모양으로 보이지 않는다.

들머리의 셋째 문단을 그린다. 「모양이 눈에 들어오려면 꽤 큰 상관이
필요하다. |r| = 0.3 에서 이 쪽의 부호화가 내놓는 축비는 1.464 로, 단축이
장축의 68% 나 되어 원과 구별되지 않는다.」

쪽의 부호화는 너비 1.01, 높이 0.99 - |r| 이라 축비가 1.01/(0.99 - |r|) 이고,
등밀도 타원의 축비는 sqrt((1+|r|)/(1-|r|)) 이다. 두 곡선은 큰 |r| 에서
크게 갈리는데(0.97 에서 50 대 8) **작은 |r| 에서는 둘 다 1 에 붙어 있다.**

왼쪽이 그 두 곡선이고, 눈으로 원과 구별되지 않는 영역(축비 1.5 미만)을
띠로 칠했다. 그 띠가 |r| = 0.32 와 0.38 까지 올라온다. 오른쪽은 같은
수를 타원으로 직접 그린 것이다 --- 앞의 세 칸이 두 줄 모두 동그라미다.

쪽의 correlation_ellipses_82 는 업종 자료의 타원 격자를, _140 과 _188 은
신뢰타원을 그린다. 여기서는 그 셋이 건드리지 않는 것 --- 두 부호화를
맞대어 어디서부터 읽히는지 --- 만 본다.
"""

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

import numpy as np
from matplotlib.patches import Ellipse

OUT = "docs/ch11/visualization/img/"

DELTA = 0.01
SHOW = [0.0, 0.15, 0.3, 0.5, 0.8, 0.97]      # 쪽의 표에 이 가운데 넷이 있다
LIMIT = 1.5                                   # 이보다 납작해야 눈에 들어온다


def ratio_code(r):
    """쪽의 부호화 — 너비 1+delta, 높이 1-|r|-delta."""
    return (1 + DELTA) / np.abs(1 - DELTA - np.abs(r))


def ratio_true(r):
    """같은 상관을 가진 등밀도 타원의 장축 대 단축."""
    return np.sqrt((1 + np.abs(r)) / (1 - np.abs(r)))


fig = plt.figure(figsize=(13.6, 5.2))
gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.25], height_ratios=[1, 1],
                      hspace=0.42, wspace=0.22)
ax1 = fig.add_subplot(gs[:, 0])
axT = fig.add_subplot(gs[0, 1])
axB = fig.add_subplot(gs[1, 1])

# ── (좌) 두 부호화의 축비 ──────────────────────────────────────────
rr = np.linspace(0, 0.97, 400)
ax1.plot(rr, ratio_code(rr), "-", lw=2.8, color="tab:red",
         label="이 쪽의 부호화  $1.01/(0.99-|r|)$")
ax1.plot(rr, ratio_true(rr), "-", lw=2.8, color="tab:blue",
         label="등밀도 타원  $\\sqrt{(1+|r|)/(1-|r|)}$")
ax1.axhspan(1, LIMIT, color="0.6", alpha=0.22)
ax1.axhline(LIMIT, color="0.35", ls="--", lw=1.6)
r_code = 1 - DELTA - (1 + DELTA) / LIMIT
r_true = (LIMIT ** 2 - 1) / (LIMIT ** 2 + 1)
for x, col in ((r_code, "tab:red"), (r_true, "tab:blue")):
    ax1.plot([x, x], [1, LIMIT], "-", lw=1.8, color=col)
    ax1.plot([x], [LIMIT], "o", ms=9, color=col, zorder=5)
ax1.text(0.96, 1.07, f"축비 {LIMIT} 미만 — 원과 다름없다", ha="right",
         va="bottom", fontsize=10.5, color="0.2", fontweight="bold")
for v in (0.3, 0.8):
    ax1.plot([v], [ratio_code(v)], "o", ms=9, color="tab:red", zorder=5)
    ax1.plot([v], [ratio_true(v)], "o", ms=9, color="tab:blue", zorder=5)
ax1.annotate(f"$|r| = 0.3$ 에서 {ratio_code(0.3):.3f} 대 "
             f"{ratio_true(0.3):.3f}\n— 둘 다 원이나 다름없다",
             xy=(0.3, ratio_code(0.3)), xytext=(0.36, 4.3), fontsize=10.5,
             color="0.2", fontweight="bold",
             arrowprops=dict(arrowstyle="->", color="0.35", lw=1.6))
ax1.annotate(f"$|r| = 0.8$ 에서야 {ratio_code(0.8):.2f} 대 "
             f"{ratio_true(0.8):.2f} 로 갈린다", xy=(0.8, ratio_true(0.8)),
             xytext=(0.30, 8.6), fontsize=10.5, color="0.2",
             fontweight="bold",
             arrowprops=dict(arrowstyle="->", color="0.35", lw=1.6))
ax1.set_xlim(0, 0.98)
ax1.set_ylim(1, 11)
ax1.set_xlabel("|r|", fontsize=11)
ax1.set_ylabel("타원의 장축 대 단축", fontsize=11)
ax1.set_title("두 부호화 모두 작은 상관에서는 거의 원이다", fontsize=12.5)
ax1.legend(fontsize=10, loc="upper left", framealpha=0.95)
ax1.grid(alpha=0.25)

# ── (우) 그 수를 타원으로 직접 그린다 ──────────────────────────────
for ax, fn, name, col in ((axT, ratio_code, "이 쪽의 부호화", "tab:red"),
                          (axB, ratio_true, "등밀도 타원", "tab:blue")):
    for i, v in enumerate(SHOW):
        q = fn(v)
        # 넓이를 비슷하게 맞추되 장축에 상한을 두어 칸 밖으로 나가지 않게 한다.
        w = min(0.90, 0.84 * np.sqrt(q))
        h = w / q
        ax.add_patch(Ellipse((i, 0), width=w, height=h, angle=45,
                             facecolor=col, alpha=0.35, edgecolor=col,
                             lw=2.0))
        ax.text(i, -0.72, f"|r| = {v:g}\n비 {q:.2f}", ha="center", va="top",
                fontsize=10, color="0.25")
    ax.set_xlim(-0.6, len(SHOW) - 0.4)
    ax.set_ylim(-1.25, 0.72)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(name, fontsize=12, color=col)
    ax.add_patch(plt.Rectangle((-0.52, -0.58), 3.04, 1.16, fill=False,
                               ls="--", lw=1.6, edgecolor="0.45"))
axT.text(1.0, 0.68, "이 셋이 뭉개진다", ha="center", va="bottom",
         fontsize=11, color="0.3", fontweight="bold")

fig.suptitle("타원 그림의 가장 큰 한계 — 약한 상관이 모두 동그라미가 된다",
             fontsize=14, y=0.995)
path = OUT + "ellipse_hides_small_r.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
print("saved", path)

print(f"\n  |r|      이 쪽의 부호화    등밀도 타원     단축/장축(부호화)")
for v in (0.0, 0.3, 0.5, 0.8, 0.97):
    print(f"  {v:.2f}      {ratio_code(v):8.3f}      {ratio_true(v):8.3f}"
          f"        {1/ratio_code(v):.3f}")
print(f"\n축비가 {LIMIT} 에 이르는 자리")
print(f"  이 쪽의 부호화: |r| = {r_code:.4f}")
print(f"  등밀도 타원:    |r| = {r_true:.4f}")
