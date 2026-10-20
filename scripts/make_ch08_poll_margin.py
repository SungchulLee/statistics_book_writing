r"""여론조사의 $\pm 3.1\%$p 가 덮는 것과 덮지 못하는 것.

① 오차한계는 $1/\sqrt n$ 이라 $n = 1000$ 을 넘기면 돈값을 못 한다.
② 두 후보의 **격차**에 붙는 오차한계는 한 후보에 붙는 것보다 크다.
③ 무응답이 만드는 치우침은 $n$ 을 아무리 키워도 줄지 않는다.

만드는 파일: docs/ch08/sample_size/img/poll_margin.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

Z = 1.959964  # 95% 양측

# === ① 오차한계 =====================================================
# p = 0.5 가 p(1-p) 를 최대로 하므로 「가장 보수적인」 오차한계가 된다.


def moe(n, p=0.5):
    return Z * np.sqrt(p * (1 - p) / n)


# === ② 격차의 오차한계 ==============================================
# 한 조사 안의 두 지지율은 다항분포라 음의 상관을 갖는다.
#   Var(p1 - p2) = [p1(1-p1) + p2(1-p2) + 2 p1 p2] / n
#                = [p1 + p2 - (p1-p2)^2] / n


def moe_gap(p1, p2, n):
    return Z * np.sqrt((p1 + p2 - (p1 - p2) ** 2) / n)


P1, P2, N_POLL = 0.45, 0.43, 1000

# === 그림 ==========================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.0, 4.8))

ns = np.logspace(np.log10(200), np.log10(20000), 300)
ax1.semilogx(ns, moe(ns) * 100, color="#1f77b4", lw=2.4)
for n, c, dx, ha in [
    (1000, "#d62728", 1.25, "left"),
    (4000, "#ff7f0e", 1.25, "left"),
    (16000, "#7f7f7f", 0.80, "right"),
]:
    ax1.plot([n], [moe(n) * 100], "o", color=c, ms=8, zorder=5)
    ax1.annotate(
        f"$n = {n:,}$\n$\\pm {moe(n) * 100:.2f}\\%$p".replace(",", "{,}"),
        xy=(n, moe(n) * 100),
        xytext=(n * dx, moe(n) * 100 + 0.55),
        ha=ha,
        fontsize=9.5,
        color=c,
    )
ax1.set_xlabel("표본 크기 $n$")
ax1.set_ylabel("오차한계 ($\\%$p)")
ax1.set_title("① 반으로 줄이려면 네 배를 뽑아야 한다", fontsize=12)
ax1.set_xticks([1000, 10000])
ax1.set_xticklabels(["$1{,}000$", "$10{,}000$"])
ax1.minorticks_off()
ax1.set_ylim(0, 7.4)
ax1.grid(alpha=0.25)

# ② 한 조사 안의 세 구간
rows = [
    ("후보 A", P1 * 100, moe(N_POLL, P1) * 100, "#1f77b4"),
    ("후보 B", P2 * 100, moe(N_POLL, P2) * 100, "#2ca02c"),
    ("격차 (A $-$ B)", (P1 - P2) * 100, moe_gap(P1, P2, N_POLL) * 100, "#d62728"),
]
for i, (lab, c0, m, col) in enumerate(rows):
    y = 2 - i
    ax2.plot([c0 - m, c0 + m], [y, y], color=col, lw=3.4, solid_capstyle="round")
    ax2.plot([c0], [y], "o", color=col, ms=9)
    ax2.text(c0, y + 0.22, f"${c0:.0f} \\pm {m:.1f}$", ha="center", fontsize=10.5)
ax2.axvline(0, color="#555555", lw=1.2, ls="--")
ax2.text(0.6, -0.42, "$0$ — 격차 구간이 이것을 품는다", fontsize=9.5, color="#555555")
ax2.set_yticks([2, 1, 0])
ax2.set_yticklabels([r[0] for r in rows], fontsize=11)
ax2.set_ylim(-0.75, 2.6)
ax2.set_xlim(-10, 52)
ax2.set_xlabel("지지율 ($\\%$)   $n = 1{,}000$")
ax2.set_title("② 격차의 오차한계가 더 크다", fontsize=12)
ax2.grid(alpha=0.25, axis="x")

# ③ 치우침은 n 으로 줄지 않는다
ax3.loglog(ns, moe(ns) * 100, color="#1f77b4", lw=2.4, label="치우침 없음")
for d, c in [(0.005, "#2ca02c"), (0.01, "#ff7f0e"), (0.02, "#d62728")]:
    rmse = np.sqrt((Z * np.sqrt(0.25 / ns)) ** 2 + (d * Z) ** 2) * 100
    ax3.loglog(ns, rmse, color=c, lw=2.2, label=f"치우침 ${d * 100:.1f}\\%$p")
    ax3.axhline(d * Z * 100, color=c, lw=0.9, ls=":")
ax3.set_xlabel("표본 크기 $n$")
ax3.set_ylabel("전체 오차 ($\\%$p)")
ax3.set_title("③ 바닥이 있다", fontsize=12)
ax3.set_xticks([1000, 10000])
ax3.set_xticklabels(["$1{,}000$", "$10{,}000$"])
ax3.set_yticks([1, 2, 5])
ax3.set_yticklabels(["$1$", "$2$", "$5$"])
ax3.minorticks_off()
ax3.legend(loc="lower left", fontsize=9.5)
ax3.grid(alpha=0.25, which="both")

fig.tight_layout()
out = "docs/ch08/sample_size/img/poll_margin.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)

# === 본문에 적을 수 =================================================
print("\n① 오차한계 (p = 0.5, 95%)")
for n in (500, 1000, 1500, 2000, 4000, 10000, 16000):
    print(f"  n = {n:>6,}:  ±{moe(n) * 100:.2f}%p")
print(f"  1.96*sqrt(0.25/1000) = {moe(1000):.6f}  → 반올림해 ±3.1%p")

print("\n② 한 조사 안에서")
print(f"  후보 A {P1 * 100:.0f}% ± {moe(N_POLL, P1) * 100:.2f}%p")
print(f"  후보 B {P2 * 100:.0f}% ± {moe(N_POLL, P2) * 100:.2f}%p")
g = moe_gap(P1, P2, N_POLL)
print(f"  격차   {(P1 - P2) * 100:.0f}%p ± {g * 100:.2f}%p   → [{(P1 - P2 - g) * 100:+.1f}, {(P1 - P2 + g) * 100:+.1f}]")
print(f"  격차 오차한계 / 한 후보 오차한계 = {g / moe(N_POLL, P1):.2f} 배")
need = (Z / (P1 - P2)) ** 2 * (P1 + P2 - (P1 - P2) ** 2)
print(f"  2%p 격차를 유의하게 만들려면 n = {need:,.0f}")

print("\n③ 치우침이 있을 때 전체 오차가 멈추는 바닥")
for d in (0.005, 0.01, 0.02):
    n_eq = 0.25 / d**2
    print(
        f"  치우침 {d * 100:.1f}%p:  바닥 {d * Z * 100:.2f}%p,  "
        f"표집오차와 같아지는 n = {n_eq:,.0f}"
    )
