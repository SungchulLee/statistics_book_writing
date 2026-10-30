"""8.5절 두 집단 표본크기 — 식이 낸 수와 실제로 모은 수.

SURMOUNT-1 이 논문에 적어 둔 다섯(차 11 퍼센트포인트 · 공통 표준편차 10% ·
탈락 25% · 양측 0.025 · 검정력 90% 초과)을 이 절의 식에 넣으면 집단당 21 명이고
탈락을 메우면 28 명이다. 실제 등록은 2,539 명이었다.

21 과 28 은 식으로 직접 계산한다 — 쪽이 적어 둔 수와 맞아야 한다.
600 · 2,400 · 2,539 는 논문이 밝힌 설계와 등록 수다.

가로축이 로그다. 로그 눈금의 마이너스는 mathtext 라 한글 글꼴로 덮이지 않으므로
눈금 꼬리표를 직접 적어 mathtext 를 아예 쓰지 않는다.
"""

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

import numpy as np
from matplotlib.ticker import FixedLocator, FixedFormatter
from scipy import stats

OUT = "docs/ch08/sample_size/img/"

# ── 식이 내는 수 ─────────────────────────────────────────────────────
DELTA, SD = 11.0, 10.0          # 차 11 퍼센트포인트, 공통 표준편차 10%
ALPHA, POWER = 0.025, 0.90      # 양측 유의수준이 0.025 다 — a/2 가 아니라 a 가 0.025
DROPOUT = 0.25

z_a = stats.norm.ppf(1 - ALPHA / 2)
z_b = stats.norm.ppf(POWER)
n_raw = 2 * (z_a + z_b) ** 2 * SD**2 / DELTA**2
n_formula = int(np.ceil(n_raw))
n_dropout = int(np.ceil(n_formula / (1 - DROPOUT)))

STEPS = [(n_formula, "식이 내는 최소", "집단당"),
         (n_dropout, f"탈락 {int(DROPOUT * 100)}% 를 메우면", "집단당"),
         (600,       "네 갈래로 나누어 계획", "한 갈래"),
         (2539,      "실제로 등록한 사람", "전체")]

vals = [s[0] for s in STEPS]
y = np.arange(len(STEPS))[::-1]
colors = ["tab:blue", "tab:blue", "tab:orange", "tab:red"]

fig, ax = plt.subplots(figsize=(11.2, 3.9))
ax.set_xscale("log")

# 로그 눈금에서 막대 길이는 비율을 거짓으로 말한다 — 점과 잇는 선으로 그린다
ax.plot(vals, y, "-", color="0.75", lw=1.6, zorder=1)
ax.scatter(vals, y, s=190, c=colors, zorder=3, edgecolor="white", lw=1.5)

for yy, (v, lab, unit), c in zip(y, STEPS, colors):
    ax.text(v * 1.13, yy + 0.07, f"{v:,}명", va="center", ha="left",
            fontsize=12.5, fontweight="bold", color=c)
    ax.text(v * 1.13, yy - 0.21, f"({unit})", va="center", ha="left",
            fontsize=9.5, color="0.45")

# 식이 낸 수의 몇 배인지 — 1.3 배를 「1배」로 적지 않도록 소수까지
for yy, v in zip(y[1:], vals[1:]):
    r = v / vals[0]
    ax.text(v * 0.88, yy, f"식의 {r:.1f}배" if r < 10 else f"식의 {r:.0f}배",
            va="center", ha="right", fontsize=10, color="0.4")

ax.set_yticks(y, [s[1] for s in STEPS], fontsize=11)
ax.set_xlim(11, 9000)
ax.set_ylim(-0.55, len(STEPS) - 0.45)

ticks = [20, 50, 100, 300, 1000, 3000]
ax.xaxis.set_major_locator(FixedLocator(ticks))
ax.xaxis.set_major_formatter(FixedFormatter([f"{t:,}" for t in ticks]))
ax.xaxis.set_minor_locator(FixedLocator([]))
ax.set_xlabel("사람 수 (로그 눈금)")
ax.set_title("SURMOUNT-1 — 식이 낸 수와 실제로 모은 수", fontsize=12)
ax.grid(axis="x", alpha=0.25)

plt.tight_layout()
path = OUT + "formula_vs_enrolled.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
print("saved", path)

print(f"식이 내는 집단당 최소   {n_raw:.3f} -> {n_formula}명  (쪽의 21 과 맞아야 한다)")
print(f"탈락 25% 를 메우면      {n_formula / (1 - DROPOUT):.3f} -> {n_dropout}명"
      f"  (쪽의 28 과 맞아야 한다)")
for v, lab, _ in STEPS[1:]:
    print(f"  {lab:24s} {v:>6,}명   식의 {v / n_formula:.1f}배")
