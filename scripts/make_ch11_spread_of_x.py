"""11.1절 최소제곱 — 판정식을 떨어뜨리는 것은 옮긴 일이 아니라 좁혀 모은 일이다.

연립방정식의 판정식은 n*sum((x-xbar)^2) 로 x 의 퍼짐만 본다.
  x = 1..6        퍼짐 17.5 → 판정식 105.00
  x = 100..105    퍼짐 17.5 → 판정식 105.00   (통째로 옮겨도 꿈쩍하지 않는다)
  x = 100..101    퍼짐  0.7 → 판정식   4.20   (좁혀 모으면 25 분의 1)

같은 참 기울기에서 잡음만 다시 뽑아 적합선을 여러 개 겹치면 그 차이가 보인다.
기울기의 표준오차는 sigma/sqrt(sum((x-xbar)^2)) 이므로 셋째 판에서 5 배가 된다.
"""

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

import numpy as np

OUT = "docs/ch11/estimation_inference/img/"
BETA0, BETA1, SIGMA = 1.0, 2.0, 1.0
REP = 400
rng = np.random.default_rng(11)

CASES = [("x 가 1 에서 6",      np.array([1., 2, 3, 4, 5, 6]),      "tab:blue"),
         ("x 를 통째로 옮김\n(100 에서 105)", np.arange(100., 106.), "tab:green"),
         ("x 를 좁혀 모음\n(100 에서 101)", np.linspace(100., 101., 6), "tab:red")]

fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.5))

for ax, (name, x, col) in zip(axes, CASES):
    n = len(x)
    sxx = ((x - x.mean()) ** 2).sum()
    det = n * sxx
    se = SIGMA / np.sqrt(sxx)                 # 기울기의 이론 표준오차

    slopes = []
    xs = np.linspace(x.min() - 0.08 * (x.ptp() or 1), x.max() + 0.08 * (x.ptp() or 1), 2)
    for _ in range(REP):
        y = BETA0 + BETA1 * x + rng.normal(0, SIGMA, n)
        b, a = np.polyfit(x, y, 1)
        slopes.append(b)
        # 중심을 맞춰 겹쳐야 기울기의 흔들림만 보인다
        ax.plot(xs, y.mean() + b * (xs - x.mean()), "-", color=col,
                lw=0.7, alpha=0.16, zorder=2)
    slopes = np.array(slopes)

    ax.plot(xs, (BETA0 + BETA1 * x.mean()) + BETA1 * (xs - x.mean()), "--",
            color="black", lw=2.0, zorder=4, label=f"참 기울기 {BETA1}")
    ax.scatter(x, BETA0 + BETA1 * x, s=26, c="black", zorder=5)

    ax.set_title(f"{name}\n퍼짐 {sxx:.1f} · 판정식 {det:.2f}", fontsize=11)
    ax.set_xlabel("x")
    if ax is axes[0]:
        ax.set_ylabel("y")
    ax.text(0.03, 0.97,
            f"기울기의 표준오차\n이론 {se:.4f} · 뽑은 값 {slopes.std(ddof=1):.4f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=10,
            bbox=dict(fc="white", ec="0.8", alpha=0.92))
    ax.legend(fontsize=9.5, loc="lower right", framealpha=0.95)
    ax.grid(alpha=0.2)

fig.suptitle(f"같은 참 기울기에서 잡음만 다시 뽑아 적합선 {REP} 개를 겹치면", fontsize=13, y=1.04)
fig.text(0.5, 0.985, "세 판의 가로·세로 눈금이 다르다 — 견줄 것은 부채의 폭이 아니라 상자 안의 표준오차다",
         ha="center", va="top", fontsize=10.5, color="0.4")
plt.tight_layout()
path = OUT + "spread_of_x.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
print("saved", path)

base = None
for name, x, _ in CASES:
    sxx = ((x - x.mean()) ** 2).sum()
    se = SIGMA / np.sqrt(sxx)
    base = base or se
    print(f"{name.replace(chr(10), ' '):32s} 퍼짐 {sxx:>5.1f}  판정식 {len(x) * sxx:>7.2f}  "
          f"기울기 SE {se:.4f}  ({se / base:.1f}배)")
