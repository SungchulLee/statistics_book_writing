"""9.3절 두 비율의 z 검정 — 같은 자료에 표준오차가 둘이고 순서가 뒤집힌다.

합동(귀무가설을 참이라 놓고 둘을 묶어 만든 것)과 비합동(둘을 따로 둔 것)은
넷째 자리에서 갈린다. 어느 쪽이 큰지는 고정되어 있지 않고 **두 집단 크기의
균형**이 정한다 — 쪽의 두 보기가 서로 반대다.

두 자료의 척도가 26 배 차이나므로 패널을 나누고 각각 크게 확대한다.
비(합동/비합동) 하나로 그리면 1.0 근처가 뭉개져 아무것도 보이지 않는다.
"""

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

import numpy as np

OUT = "docs/ch09/two_sample_tests/img/"


def ses(x1, n1, x2, n2):
    p1, p2 = x1 / n1, x2 / n2
    pb = (x1 + x2) / (n1 + n2)
    return (np.sqrt(pb * (1 - pb) * (1 / n1 + 1 / n2)),
            np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2))


CASES = [("화이자 시험", 8, 17411, 162, 17511, "백신군", "위약군", "{:.8f}"),
         ("A/B 자료",   45, 500,    58, 480,   "A 안",   "B 안",   "{:.7f}")]

fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6))

for ax, (name, x1, n1, x2, n2, g1, g2, fmt) in zip(axes, CASES):
    pooled, unpooled = ses(x1, n1, x2, n2)
    p1, p2 = x1 / n1, x2 / n2
    vals = [pooled, unpooled]
    cols = ["tab:purple", "tab:green"]

    # 넷째 자리를 보이려고 축을 자르므로 막대는 쓰지 않는다 — 길이가 거짓말을 한다
    ax.scatter([0, 1], vals, s=260, c=cols, zorder=3, edgecolor="white", lw=2)
    for i, v in enumerate(vals):
        ax.annotate(fmt.format(v), (i, v), textcoords="offset points",
                    xytext=(0, 15), ha="center", fontsize=12.5,
                    fontweight="bold", color=cols[i])

    # 넷째 자리의 차이를 보이려면 바닥을 끌어올려야 한다
    lo, hi = min(vals), max(vals)
    pad = (hi - lo) * 1.9
    ax.set_ylim(lo - pad, hi + pad * 1.5)

    bigger = "합동" if pooled > unpooled else "비합동"
    gap = abs(pooled - unpooled) / min(vals) * 100
    ax.annotate("", xy=(0.5, hi), xytext=(0.5, lo),
                arrowprops=dict(arrowstyle="<->", color="0.35", lw=1.4))
    # 값 꼬리표와 겹치지 않게 화살표 왼쪽에 둔다
    ax.text(0.44, (hi + lo) / 2, f"{bigger}이\n{gap:.2f}% 크다", ha="right",
            va="center", fontsize=11, fontweight="bold", color="0.2")

    # 무엇이 그 순서를 정했는가
    small = g1 if n1 < n2 else g2
    low = g1 if p1 < p2 else g2
    same = (n1 < n2) == (p1 < p2)
    ax.set_xticks([0, 1], ["합동 표준오차\n(귀무가설을 믿는다)",
                           "비합동 표준오차\n(믿지 않는다)"], fontsize=10.5)
    ax.set_title(f"{name}   {g1} {n1:,} 명 · {g2} {n2:,} 명\n"
                 f"작은 집단이 {small}, 비율이 낮은 쪽이 {low} "
                 f"— {'같다' if same else '다르다'} → {bigger}이 크다",
                 fontsize=10.5)
    ax.grid(axis="y", alpha=0.25)
    ax.set_xlim(-0.55, 1.55)

axes[0].set_ylabel("표준오차")
fig.suptitle("같은 가설에 분모가 둘 — 어느 쪽이 큰지는 집단 크기의 균형이 정한다",
             fontsize=12.5, y=1.02)

plt.tight_layout()
path = OUT + "pooled_se_crossover.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
print("saved", path)

for name, x1, n1, x2, n2, *_ in CASES:
    pl, un = ses(x1, n1, x2, n2)
    print(f"{name:10s} 합동 {pl:.8f}  비합동 {un:.8f}  "
          f"{'합동' if pl > un else '비합동'}이 "
          f"{abs(pl - un) / min(pl, un) * 100:.2f}% 크다")
