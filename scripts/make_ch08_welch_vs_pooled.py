"""8.3절 μ₁-μ₂ 의 신뢰구간 — 좁은 구간의 값은 포함률로 치른다.

분산이 다른 자리에서 네 방법의 포함률과 기대 폭을 나란히 둔다.
수는 쪽 연습문제의 되풀이 200 만 회 모의 결과다(몬테카를로 표준오차 0.0002).
그 자리에서 다시 돌리면 몇 시간이 걸리므로 상수로 적고 출처를 주석에 남긴다.
"""

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

import numpy as np

OUT = "docs/ch08/two_sample_intervals/img/"
NOMINAL = 0.95

# 방법 · 포함률 · 기대 폭 · 분모에 무엇을 넣는가
ROWS = [("웰치 t",      0.9499, 2.3333, "집단마다 따로"),
        ("합동 t",      0.9399, 2.2012, "하나로 합침"),
        ("z-known",    0.9498, 2.1766, "참 분산을 안다고"),
        ("z-plugin",   0.9317, 2.1418, "표본분산을 꽂음")]

names = [r[0] for r in ROWS]
cov = np.array([r[1] for r in ROWS])
wid = np.array([r[2] for r in ROWS])
note = [r[3] for r in ROWS]
y = np.arange(len(ROWS))[::-1]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 3.9),
                               gridspec_kw={"width_ratios": [1.15, 1]})

# ── (좌) 포함률 ──────────────────────────────────────────────────────
colors = ["tab:blue" if c >= NOMINAL - 0.0005 else "tab:red" for c in cov]
ax1.hlines(y, NOMINAL, cov, color=colors, lw=2.6, alpha=0.55)
ax1.scatter(cov, y, s=115, c=colors, zorder=3)
ax1.axvline(NOMINAL, color="black", ls="--", lw=1.2)
ax1.text(NOMINAL + 0.0004, len(ROWS) - 0.42, "명목 0.95", ha="left",
         va="bottom", fontsize=9.5, bbox=dict(fc="white", ec="none", pad=1.5))

for yy, c in zip(y, cov):
    gap = (c - NOMINAL) * 100
    lab = f"{c:.4f}" if abs(gap) < 0.1 else f"{c:.4f}   ({gap:+.1f}%p)"
    ax1.text(c - 0.0012, yy, lab, va="center", ha="right", fontsize=10)

ax1.set_yticks(y, [f"{n}\n{t}" for n, t in zip(names, note)], fontsize=10)
ax1.set_xlim(0.9225, 0.9535)
ax1.set_ylim(-0.6, len(ROWS) - 0.1)
ax1.set_xlabel("실제 포함률   (되풀이 200 만 회)")
ax1.set_title("약속한 것", fontsize=11.5)
ax1.grid(axis="x", alpha=0.25)

# ── (우) 기대 폭 ─────────────────────────────────────────────────────
bars = ax2.barh(y, wid, height=0.52, color=colors, alpha=0.8)
base = wid[0]
for yy, w in zip(y, wid):
    ax2.text(w - 0.03, yy, f"{w:.4f}", va="center", ha="right",
             fontsize=10, color="white", fontweight="bold")
    if w != base:
        ax2.text(w + 0.03, yy, f"웰치보다 {(1 - w / base) * 100:.1f}% 좁다",
                 va="center", ha="left", fontsize=9.5, color="0.3")

ax2.set_yticks(y, [])
ax2.set_xlim(0, 2.95)
ax2.set_ylim(-0.6, len(ROWS) - 0.1)
ax2.set_xlabel("기대 폭")
ax2.set_title("그 대가로 얻은 것", fontsize=11.5)
ax2.grid(axis="x", alpha=0.25)

plt.tight_layout()
path = OUT + "welch_vs_pooled.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
print("saved", path)

print(f"{'방법':>10} {'포함률':>9} {'명목과의 차':>12} {'폭':>9} {'웰치 대비':>10}")
for n, c, w in zip(names, cov, wid):
    print(f"{n:>10} {c:>9.4f} {(c - NOMINAL) * 100:>11.2f}%p "
          f"{w:>9.4f} {(1 - w / base) * 100:>9.1f}%")
