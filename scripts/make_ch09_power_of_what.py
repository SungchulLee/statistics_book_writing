"""9.5절 검정력 — 「무엇의 검정력인가」가 수를 스무 배로 가른다.

STEP 1 의 보고된 차이 -12.44 퍼센트포인트와 신뢰구간에서 되짚은 공통
표준편차 9.91 로 주 결과변수 하나만 보면 집단당 열 명이면 0.80 이고
스물넷이면 0.99 다. 같은 0.99 를 설계 전체에 붙이려니 1,950 명이 들었다.

왼쪽은 그 검정력 곡선, 오른쪽은 세 수의 간격이다.
"""

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

import numpy as np
from matplotlib.ticker import FixedLocator, FixedFormatter
from scipy import stats

OUT = "docs/ch09/errors_and_power/img/"

DELTA, SD, ALPHA = 12.44, 9.91, 0.05
TOTAL_DESIGN = 1950                       # 1,306 + 655


def power(n):
    """집단당 n 일 때 양측 alpha 검정의 검정력."""
    z = stats.norm.ppf(1 - ALPHA / 2)
    ncp = DELTA / (SD * np.sqrt(2 / n))
    return stats.norm.cdf(ncp - z) + stats.norm.cdf(-ncp - z)


def n_for(pw):
    z = stats.norm.ppf(1 - ALPHA / 2) + stats.norm.ppf(pw)
    return int(np.ceil(2 * z**2 * SD**2 / DELTA**2))


n80, n99 = n_for(0.80), n_for(0.99)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.3),
                               gridspec_kw={"width_ratios": [1.2, 1]})

# ── (좌) 주 결과변수 하나의 검정력 곡선 ──────────────────────────────
ns = np.arange(3, 41)
ax1.plot(ns, [power(n) for n in ns], "-", color="tab:blue", lw=2.4)
for nn, pw, c in [(n80, 0.80, "tab:green"), (n99, 0.99, "tab:orange")]:
    ax1.vlines(nn, 0, power(nn), color=c, ls="--", lw=1.4)
    ax1.hlines(pw, 0, nn, color=c, ls="--", lw=1.4)
    ax1.scatter([nn], [power(nn)], s=110, c=c, zorder=5,
                edgecolor="white", lw=1.5)
    ax1.annotate(f"검정력 {pw:.2f} 에 집단당 {nn} 명", (nn, power(nn)),
                 textcoords="offset points", xytext=(10, -14), ha="left",
                 fontsize=10.5, color=c)

ax1.set_xlim(0, 41)
ax1.set_ylim(0, 1.05)
ax1.set_xlabel("집단당 인원")
ax1.set_ylabel("검정력")
ax1.set_title("주 결과변수 하나만 보면", fontsize=11.5)
ax1.grid(alpha=0.25)

# ── (우) 같은 0.99 를 무엇에 붙이는가 ────────────────────────────────
BARS = [(f"주 결과변수에\n검정력 0.80", n80, "tab:green"),
        (f"주 결과변수에\n검정력 0.99", n99, "tab:orange"),
        ("설계 전체에\n검정력 0.99", TOTAL_DESIGN, "tab:red")]
y = np.arange(len(BARS))[::-1]
vals = [b[1] for b in BARS]

ax2.set_xscale("log")
ax2.plot(vals, y, "-", color="0.8", lw=1.6, zorder=1)
ax2.scatter(vals, y, s=190, c=[b[2] for b in BARS], zorder=3,
            edgecolor="white", lw=1.6)
for yy, (lab, v, c) in zip(y, BARS):
    ax2.text(v * 1.18, yy + 0.06, f"{v:,}명", va="center", ha="left",
             fontsize=12.5, fontweight="bold", color=c)
    if v != vals[0]:
        r = v / vals[0]
        ax2.text(v * 1.18, yy - 0.17,
                 f"열 명의 {r:.1f}배" if r < 10 else f"열 명의 {r:.0f}배",
                 va="center", ha="left", fontsize=9.5, color="0.45")

ax2.set_yticks(y, [b[0] for b in BARS], fontsize=10.5)
ticks = [10, 30, 100, 300, 1000, 3000]
ax2.xaxis.set_major_locator(FixedLocator(ticks))
ax2.xaxis.set_major_formatter(FixedFormatter([f"{t:,}" for t in ticks]))
ax2.xaxis.set_minor_locator(FixedLocator([]))
ax2.set_xlim(6, 20000)
ax2.set_ylim(-0.55, len(BARS) - 0.45)
ax2.set_xlabel("인원 (로그 눈금)")
ax2.set_title("「무엇의」 검정력인가", fontsize=11.5)
ax2.grid(axis="x", alpha=0.25)

plt.tight_layout()
path = OUT + "power_of_what.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
print("saved", path)

print(f"차 {DELTA}, 공통 표준편차 {SD}, 양측 {ALPHA}")
print(f"  검정력 0.80 → 집단당 {n80} 명   (쪽: 열 명)")
print(f"  검정력 0.99 → 집단당 {n99} 명   (쪽: 스물넷)")
print(f"  설계 전체에 0.99 → {TOTAL_DESIGN:,} 명   "
      f"(열 명의 {TOTAL_DESIGN / n80:.0f}배)")
