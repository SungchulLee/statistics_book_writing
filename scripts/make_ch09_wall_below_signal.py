"""9.6절 거짓발견율 — 문턱이 신호가 사는 자리 아래에 있다.

들머리의 두 보기를 그린다. 왼쪽은 유전자 2 만 개 가운데 참으로 관련 있는
것이 100 개 있는 자리다. 본페로니 문턱 0.05/20000 = 0.0000025 는 그
100 개의 p 값이 사는 자리보다 훨씬 왼쪽에 있어서, 진짜 신호 넷에 셋이
그 벽을 넘지 못한다. 오른쪽은 화합물 열 개를 선별하는 자리로, 본페로니는
하나만 넘기고 BH 는 둘을 넘긴다.

쪽의 bh_procedure 는 정렬한 p 값에 세 문턱을 긋는 그림이고 fdr_scaling 은
검정 수에 따른 검정력을 그린다. 여기서는 **신호가 사는 자리와 벽의
위치**를 맞댄다.
"""
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
from scipy import stats

OUT = "docs/ch09/multiple_testing/img/"
M, M1, DELTA, A = 20_000, 100, 4.0, 0.05
BONF = A / M

rng = np.random.default_rng(1995)
z1 = rng.standard_normal(M1) + DELTA            # 참으로 관련 있는 유전자
p1 = 2 * stats.norm.sf(np.abs(z1))
p0 = rng.random(M - M1)                         # 관련 없는 유전자
allp = np.sort(np.concatenate([p0, p1]))
k = np.arange(1, M + 1)
below = allp <= k / M * 0.10
kbh = int(k[below].max()) if below.any() else 0
bh_cut = allp[kbh - 1] if kbh else 0.0
caught_b = int((p1 <= BONF).sum())
caught_h = int((p1 <= bh_cut).sum())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 4.6),
                               gridspec_kw={"width_ratios": [1.25, 1]})

# ── (좌) 진짜 신호가 사는 자리와 두 벽 ────────────────────────────
bins = np.logspace(-10, 0, 46)
ax1.hist(p1, bins=bins, color="tab:green", alpha=0.75,
         label=f"참으로 관련 있는 유전자 {M1} 개의 $p$ 값")
ax1.axvline(BONF, color="tab:red", lw=2.4)
ax1.text(BONF * 0.65, 13.6, f"본페로니 문턱\n{BONF:.7f}", ha="right",
         fontsize=11, color="tab:red", fontweight="bold")
ax1.axvline(bh_cut, color="tab:blue", lw=2.4, ls="--")
ax1.text(bh_cut * 1.5, 13.6, f"BH 가 실제로 자른 자리\n{bh_cut:.6f}",
         ha="left", fontsize=11, color="tab:blue", fontweight="bold")
ax1.set_xscale("log")
ax1.set_xlim(1e-10, 1)
# 로그 눈금의 기본 이름표는 mathtext 라 한글 글꼴에 없는 U+2212 를 부른다
T1 = [1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1]
ax1.set_xticks(T1, ["0.0000000001", "0.00000001", "0.000001", "0.0001",
                    "0.01", "1"], fontsize=9)
ax1.set_xticks([], minor=True)
ax1.set_ylim(0, 16.5)
ax1.set_xlabel("$p$ 값 (로그 눈금)", fontsize=11)
ax1.set_ylabel("유전자 수", fontsize=11)
ax1.set_title(f"유전자 {M:,} 개를 한꺼번에 검정하면", fontsize=12.5)
ax1.legend(fontsize=10, loc="upper left", framealpha=0.95)
ax1.grid(axis="y", alpha=0.2)
ax1.text(0.98, 0.42,
         f"벽을 넘은 진짜 신호\n본페로니 {caught_b} 개 · BH {caught_h} 개"
         f"\n(전체 {M1} 개 가운데)",
         transform=ax1.transAxes, ha="right", fontsize=11.5,
         fontweight="bold", color="0.2",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.94,
                   edgecolor="0.5", lw=1.0))

# ── (우) 화합물 열 개 ─────────────────────────────────────────────
PS = np.array([0.005, 0.011, 0.063, 0.142, 0.201, 0.334, 0.489, 0.622,
               0.780, 0.951])
Q = 0.10
bonf10 = Q / 10
for i, pv in enumerate(PS):
    hit_b = pv <= bonf10
    hit_h = pv <= (i + 1) / 10 * Q
    col = "tab:blue" if hit_h else "0.68"
    ax2.plot([pv], [1], "o", ms=12 if hit_h else 8, color=col)
    if i < 3:
        ax2.text(pv, 1.14, f"{pv}", ha="center", fontsize=10.5, color=col,
                 fontweight="bold")
ax2.axvline(bonf10, color="tab:red", lw=2.4)
ax2.text(bonf10 * 0.82, 1.58, f"본페로니 문턱 {bonf10}", ha="right",
         fontsize=11, color="tab:red", fontweight="bold")
ax2.plot(np.arange(1, 11) / 10 * Q, np.full(10, 0.55), "-", lw=2.2,
         color="tab:blue")
for i in range(10):
    ax2.plot([(i + 1) / 10 * Q], [0.55], "|", ms=11, color="tab:blue")
ax2.text(Q * 1.08, 0.55, "BH 의 문턱은 순위마다 다르다", ha="left",
         va="center", fontsize=10.5, color="tab:blue")
ax2.text(0.0045, 0.30, "본페로니는 첫째 하나만,\nBH 는 둘을 넘긴다",
         fontsize=11.5, fontweight="bold", color="0.2")
ax2.set_xscale("log")
ax2.set_xlim(0.003, 1.3)
T2 = [0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
ax2.set_xticks(T2, [f"{t:g}" for t in T2])
ax2.set_xticks([], minor=True)
ax2.set_ylim(0.1, 1.9)
ax2.set_yticks([])
ax2.set_xlabel("$p$ 값 (로그 눈금)", fontsize=11)
ax2.set_title("화합물 열 개를 선별할 때", fontsize=12.5)
ax2.grid(axis="x", alpha=0.25)

fig.suptitle("거짓을 막으려다 진짜까지 놓친다", fontsize=14, y=1.0)
plt.tight_layout()
path = OUT + "wall_below_signal.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
print("saved", path)
print(f"  m = {M:,}, 참 효과 {M1} 개, 이동폭 {DELTA}")
print(f"  본페로니 문턱 {BONF:.7f} → 진짜 {caught_b} 개를 잡는다")
print(f"  BH(q=0.10) 가 자른 자리 {bh_cut:.6f}, 기각 {kbh} 개 중 진짜 "
      f"{caught_h} 개 · 거짓 {kbh - caught_h} 개")
print(f"  화합물 보기: 본페로니 문턱 {bonf10}, BH 문턱 "
      f"{np.round(np.arange(1,11)/10*Q, 3).tolist()}")
