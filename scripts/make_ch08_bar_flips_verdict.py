"""8.1절 오차막대 그림 — 막대의 정의만 바꾸면 판정이 뒤집힌다.

쪽의 보기 1 과 **같은 자료**(seed 42, 네 집단 가운데 A 와 B)를 쓴다. 점의
자리는 세 그림에서 모두 같고 달라지는 것은 막대의 정의뿐인데, 「막대가
겹치니 차이가 없다」는 습관대로 읽으면 결론이 반대로 나온다.

마지막 칸이 옳은 길이다 --- 겹침을 눈으로 재는 대신 **차이 자체의 구간**을
계산한다. 쪽의 1 절 그림은 네 집단에서 막대 길이가 n 에 따라 어떻게
변하는지를 보이는 것이고, 여기서는 A 와 B 둘만 떼어 판정을 본다.
"""

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

import numpy as np

OUT = "docs/ch08/foundations/img/"

# ── 쪽의 보기 1 과 같은 자료 ────────────────────────────────────────
rng = np.random.default_rng(42)
gs = ["A", "B", "C", "D"]
data = {g: rng.normal(m, s, n_) for g, m, s, n_ in
        zip(gs, [10, 12, 11, 13], [3, 3, 3, 3], [10, 30, 100, 300])}

g2 = ["A", "B"]
mean = np.array([data[g].mean() for g in g2])
sd = np.array([data[g].std(ddof=1) for g in g2])
n = np.array([len(data[g]) for g in g2])
se = sd / np.sqrt(n)
ci = 1.96 * se

KIND = [(sd, "±1 SD", "표준편차"), (se, "±1 SE", "표준오차"), (ci, "95% 신뢰구간", "신뢰구간")]

fig, axes = plt.subplots(1, 4, figsize=(14.2, 4.6),
                         gridspec_kw={"width_ratios": [1, 1, 1, 1.25]})

for ax, (err, name, _) in zip(axes[:3], KIND):
    lo, hi = mean - err, mean + err
    over = min(hi) - max(lo)                      # 양수면 겹친다
    for i, g in enumerate(g2):
        ax.errorbar(i, mean[i], yerr=err[i], fmt="o", ms=9, capsize=9,
                    lw=2.2, color="tab:blue")
        ax.text(i, hi[i] + 0.35, f"[{lo[i]:.2f}, {hi[i]:.2f}]", ha="center",
                fontsize=9.5, color="0.3")
    if over > 0:
        ax.axhspan(max(lo), min(hi), color="tab:red", alpha=0.18)
        verdict, vc = f"겹친다 ({over:.2f} 폭)\n→ 「차이가 없다」", "tab:red"
    else:
        ax.axhspan(min(hi), max(lo), color="tab:green", alpha=0.16)
        verdict, vc = f"떨어져 있다 ({-over:.2f})\n→ 「차이가 있다」", "tab:green"
    ax.text(0.5, 0.035, verdict, transform=ax.transAxes, ha="center",
            va="bottom", fontsize=11, fontweight="bold", color=vc,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      alpha=0.92, edgecolor=vc, lw=1.0))
    ax.set_xticks([0, 1], g2, fontsize=12)
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(4.6, 16.4)
    ax.set_title(f"막대 = {name}", fontsize=12.5)
    ax.grid(axis="y", alpha=0.2)
axes[0].set_ylabel("값", fontsize=11)
for ax in axes[1:3]:
    ax.set_yticklabels([])

# ── (넷째) 옳은 길 — 차이 자체의 구간 ──────────────────────────────
d = mean[0] - mean[1]
sed = np.hypot(se[0], se[1])
dlo, dhi = d - 1.96 * sed, d + 1.96 * sed
ax = axes[3]
ax.plot([dlo, dhi], [0, 0], "-", lw=4.0, color="tab:purple",
        solid_capstyle="butt")
ax.plot([d], [0], "o", ms=10, color="tab:purple")
for x in (dlo, dhi):
    ax.plot([x, x], [-0.1, 0.1], "-", lw=3.0, color="tab:purple")
ax.axvline(0, color="0.15", lw=2.0)
ax.text(0, 0.42, "차이 0", ha="center", fontsize=11, color="0.15",
        fontweight="bold")
ax.text(d, -0.32, f"{d:.4f}", ha="center", va="top", fontsize=11,
        fontweight="bold", color="tab:purple")
ax.text((dlo + dhi) / 2, 0.20, f"({dlo:.2f}, {dhi:.2f})", ha="center",
        fontsize=11, color="tab:purple")
ax.text(0.5, 0.035, "0 을 담지 않는다\n→ 차이가 있다", transform=ax.transAxes,
        ha="center", va="bottom", fontsize=11, fontweight="bold",
        color="tab:purple",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.92,
                  edgecolor="tab:purple", lw=1.0))
ax.set_xlim(-6.6, 1.4)
ax.set_ylim(-1.0, 1.0)
ax.set_yticks([])
ax.set_xlabel("A 의 평균 빼기 B 의 평균", fontsize=11)
ax.set_title("겹침을 눈으로 재는 대신\n차이 자체에 구간을 붙인다", fontsize=12.5)
ax.grid(axis="x", alpha=0.25)

fig.suptitle("자료는 하나다 — 달라진 것은 막대의 정의뿐이다", fontsize=14,
             y=1.0)
plt.tight_layout()
path = OUT + "bar_flips_verdict.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
print("saved", path)

for i, g in enumerate(g2):
    print(f"  {g}: n={n[i]:3d}  평균={mean[i]:.4f}  SD={sd[i]:.4f}  "
          f"SE={se[i]:.4f}  95%CI 반폭={ci[i]:.4f}")
for err, name, _ in KIND:
    lo, hi = mean - err, mean + err
    o = min(hi) - max(lo)
    print(f"  {name:>12s}: A [{lo[0]:.2f}, {hi[0]:.2f}]  B [{lo[1]:.2f}, "
          f"{hi[1]:.2f}]  {'겹침 ' + format(o, '.2f') if o > 0 else '떨어짐 ' + format(-o, '.2f')}")
print(f"\n  차이 {d:.4f}  표준오차 {sed:.4f}  95% 구간 ({dlo:.4f}, {dhi:.4f})")
