"""11.6절 열지도 — 진한 칸이 많다고 더 안전해지지 않는다.

들머리의 둘째 문단을 그린다. 등가중으로 p 종목을 담고 수익률을 표준화해
분산을 모두 1 로 맞추면 포트폴리오의 분산은

    1/p + (1 - 1/p) * 평균상관

이다. 평균 상관이 0.8 이면 다섯에서 쉰으로 늘려도 0.840 에서 0.804 로
내려갈 뿐이다 --- 종목을 열 배로 늘려 얻는 것이 0.036 이다. p 를 아무리
키워도 분산은 평균 상관 아래로 내려가지 못한다. 그것이 열지도가 온통
짙을 때 읽어야 하는 한계다.

오른쪽은 그 한계를 거꾸로 물은 것이다. 분산을 절반으로 줄이려면 몇
종목이 드는가. 답은 (1 - rho) / (0.5 - rho) 이고 rho 가 0.5 에 다가가면
발산한다. **평균 상관이 0.5 를 넘으면 종목을 무한히 늘려도 절반에 닿지
못한다.**

쪽의 heatmaps_15 는 ETF 상관 열지도를, _67·_113·_140·_158 은 그 열지도의
변주(값 표시·군집화·부분집합·회색조)를 그린다. 여기서는 그 다섯이
건드리지 않는 것 --- 색칠한 칸에서 읽어 내야 할 한계 --- 만 본다.
"""

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

import numpy as np

OUT = "docs/ch11/visualization/img/"

RHOS = [(0.0, "평균 상관 0.0 — 서로 무관하다"),
        (0.2, "0.2"),
        (0.5, "0.5"),
        (0.8, "0.8 — 열지도가 온통 짙을 때")]
COLS = ["tab:green", "tab:cyan", "tab:orange", "tab:red"]
RHO_PAGE = 0.8


def port_var(p, rho):
    """표준화한 p 종목을 등가중으로 담았을 때의 분산."""
    return 1 / p + (1 - 1 / p) * rho


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 4.9))

# ── (좌) 종목을 늘려도 내려가지 않는다 ──────────────────────────────
ps = np.arange(1, 201)
for (rho, name), col in zip(RHOS, COLS):
    ax1.plot(ps, port_var(ps, rho), "-", lw=2.6, color=col, label=name)
    ax1.axhline(rho, color=col, ls=":", lw=1.6)
v5, v50 = port_var(5, RHO_PAGE), port_var(50, RHO_PAGE)
ax1.plot([5, 50], [v5, v50], "o", ms=11, color="tab:red", zorder=5)
ax1.annotate(f"다섯에서 쉰으로 열 배 늘려도\n{v5:.3f} 에서 {v50:.3f} 로 "
             f"내려갈 뿐이다", xy=(50, v50), xytext=(5.4, 0.735), fontsize=11,
             color="tab:red", fontweight="bold",
             arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.8))
ax1.set_xscale("log")
ax1.set_xlim(1, 210)
ax1.set_xticks([1, 2, 5, 10, 20, 50, 100, 200],
               ["1", "2", "5", "10", "20", "50", "100", "200"])
ax1.set_xticks([], minor=True)
ax1.set_ylim(0, 1.03)
ax1.set_xlabel("담은 종목 수 (로그 눈금)", fontsize=11)
ax1.set_ylabel("포트폴리오의 분산 (한 종목이 1)", fontsize=11)
ax1.set_title("바닥을 정하는 것은 종목 수가 아니라 평균 상관이다",
              fontsize=12.5)
ax1.legend(fontsize=10, loc="center right", framealpha=0.95)
ax1.grid(alpha=0.25, which="major")

# ── (우) 절반으로 줄이려면 몇 종목이 드나 ──────────────────────────
rr = np.linspace(0, 0.49, 400)
need = (1 - rr) / (0.5 - rr)
ax2.plot(rr, need, "-", lw=2.8, color="tab:purple")
ax2.axvline(0.5, color="0.2", ls="--", lw=2.0)
ax2.axvspan(0.5, 0.95, color="tab:red", alpha=0.12)
ax2.text(0.72, 60, "평균 상관이 0.5 를 넘으면\n종목을 무한히 늘려도\n"
         "절반에 닿지 못한다", ha="center", va="center", fontsize=11.5,
         color="tab:red", fontweight="bold")
for v in (0.0, 0.2, 0.3, 0.45):
    n = (1 - v) / (0.5 - v)
    ax2.plot([v], [n], "o", ms=9, color="tab:purple", zorder=5)
    ax2.text(v, n + 4, f"{n:.1f} 종목", ha="center", va="bottom",
             fontsize=10.5, color="tab:purple", fontweight="bold")
ax2.set_xlim(0, 0.95)
ax2.set_ylim(0, 115)
ax2.set_xlabel("평균 상관", fontsize=11)
ax2.set_ylabel("분산을 절반으로 줄이는 데 드는 종목 수", fontsize=11)
ax2.set_title("같은 목표를 이루는 값이 상관에 따라 발산한다", fontsize=12.5)
ax2.grid(alpha=0.25)

fig.suptitle("열지도가 온통 짙으면 종목을 늘리는 일이 거의 아무것도 하지 못한다",
             fontsize=14, y=1.0)
plt.tight_layout()
path = OUT + "diversification_limit.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
print("saved", path)

print(f"\n평균 상관 {RHO_PAGE} 에서 종목을 늘리면")
for p in (1, 2, 5, 10, 20, 50, 100, 1000):
    print(f"  {p:>5d} 종목  →  분산 {port_var(p, RHO_PAGE):.4f}")
print(f"  다섯에서 쉰으로: {v5:.4f} → {v50:.4f}  (줄어든 양 {v5 - v50:.4f})")
print("\n분산을 절반으로 줄이는 데 드는 종목 수")
for v in (0.0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.49):
    print(f"  평균 상관 {v:.2f}  →  {(1 - v) / (0.5 - v):8.1f} 종목")
print("  평균 상관 0.50 이상 →  닿지 못한다")
