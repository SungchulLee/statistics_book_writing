"""9.2절 평균의 일표본 z 검정 — 세 규칙이 하루에 울리는 경보.

특징 2,000 개를 90 일 지켜보면 검정이 18 만 번이다. 그 규모에서 세 규칙이
각각 몇 번 울리고 그 가운데 몇이 거짓인지를 쪽의 모의 출력에서 가져온다.
왼쪽이 「얼마나 울리는가」, 오른쪽이 「울린 것 가운데 몇이 거짓인가」다.
"""

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

import numpy as np

OUT = "docs/ch09/one_sample_tests/img/"

# 규칙 · 하루 참경보 · 하루 거짓경보 · 거짓의 비율  (쪽의 모의 출력)
RULES = [("p < 0.05",      29.4, 94.2, 0.762),
         ("효과크기 0.05",   1.6,  0.7, 0.305),
         ("본페로니",        0.3,  0.0, 0.034)]
DRIFTED = 100          # 참으로 밀린 특징 수

names = [r[0] for r in RULES]
true_a = np.array([r[1] for r in RULES])
false_a = np.array([r[2] for r in RULES])
frac = np.array([r[3] for r in RULES])
y = np.arange(len(RULES))[::-1]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 3.5),
                               gridspec_kw={"width_ratios": [1.45, 1]})

# ── (좌) 하루에 몇 번 울리는가 ───────────────────────────────────────
ax1.barh(y, true_a, height=0.5, color="tab:blue", label="참경보")
ax1.barh(y, false_a, height=0.5, left=true_a, color="tab:red",
         alpha=0.85, label="거짓경보")

for yy, t, f in zip(y, true_a, false_a):
    tot = t + f
    ax1.text(tot + 2.5, yy, f"하루 {tot:.1f} 번", va="center", ha="left",
             fontsize=11, fontweight="bold")
    if t >= 8:
        ax1.text(t / 2, yy, f"{t:.1f}", va="center", ha="center",
                 fontsize=10, color="white", fontweight="bold")
    if f >= 8:
        ax1.text(t + f / 2, yy, f"{f:.1f}", va="center", ha="center",
                 fontsize=10, color="white", fontweight="bold")

ax1.set_yticks(y, names, fontsize=11)
ax1.set_xlim(0, 160)
ax1.set_ylim(-0.6, len(RULES) - 0.4)
ax1.set_xlabel("하루에 울리는 경보 수")
ax1.set_title("얼마나 울리는가", fontsize=11.5)
ax1.legend(fontsize=9.5, loc="lower right", framealpha=0.95)
ax1.grid(axis="x", alpha=0.25)

# ── (우) 울린 것 가운데 몇이 거짓인가 ────────────────────────────────
colors = ["tab:red" if f > 0.5 else "tab:orange" if f > 0.1 else "tab:green"
          for f in frac]
ax2.barh(y, frac, height=0.5, color=colors, alpha=0.85)
for yy, f, t in zip(y, frac, true_a):
    ax2.text(f + 0.02, yy, f"{f * 100:.1f}% 가 거짓", va="center",
             ha="left", fontsize=10.5, fontweight="bold")
    # 긴 줄은 막대 아래에 왼쪽 맞춤으로 — 오른쪽으로 흘러 넘치지 않게
    ax2.text(0.015, yy - 0.33, f"밀린 {DRIFTED} 개 가운데 하루 {t:.1f} 개를 잡는다",
             va="center", ha="left", fontsize=9, color="0.45")

ax2.set_yticks(y, [])
ax2.set_xlim(0, 1.0)
ax2.set_ylim(-0.6, len(RULES) - 0.4)
ax2.set_xlabel("울린 것 가운데 거짓의 비율")
ax2.set_title("울린 것이 맞는 말인가", fontsize=11.5)
ax2.grid(axis="x", alpha=0.25)

plt.tight_layout()
path = OUT + "drift_alarms.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
print("saved", path)

print(f"{'규칙':>12} {'참':>7} {'거짓':>7} {'합':>7} {'거짓비율':>9}")
for n, t, f, fr in RULES:
    print(f"{n:>12} {t:>7.1f} {f:>7.1f} {t + f:>7.1f} {fr:>9.3f}")
