"""9장 장 개요 — 「그렇게 보인다」와 「그렇게 말해도 된다」.

왼쪽은 들머리가 먼저 던지는 물음이다. 백신이 아무 쓸모가 없었다면 환자
170 명이 두 팔에 어떻게 갈렸을까. 두 팔의 사람 수가 17,411 과 17,511 로
거의 같으므로, 환자 하나가 백신군에 들 확률이 0.4986 이다. 거기서 나오는
분포와 실제로 나온 8 을 맞댄다.

오른쪽은 들머리의 둘째 요점이다. 이 시험은 「성공」의 뜻을 자료를 보기
전에 글로 적어 두었다 --- 효과가 30% 를 넘는다고 믿을 만하면 성공이고,
그 「믿을 만하다」의 선이 98.6% 다. 95 가 아니라 98.6 인 것은 시험 도중에
자료를 한 번 들여다본 값이다.

쪽의 p_hacking_demo 가 optional_stopping 으로 「엿보면서 멈추면」을 이미
그리므로 그 곡선은 여기서 그리지 않는다.
"""

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

import numpy as np
from scipy import stats

OUT = "docs/ch09/img/"

N_VAC, N_PLA = 17411, 17511
K_VAC, K_PLA = 8, 162
TOTAL = K_VAC + K_PLA
P0 = N_VAC / (N_VAC + N_PLA)        # 쓸모가 없었다면 환자 하나가 백신군일 확률

k = np.arange(TOTAL + 1)
pmf = stats.binom.pmf(k, TOTAL, P0)
p_two = 2 * stats.binom.cdf(K_VAC, TOTAL, P0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 4.7),
                               gridspec_kw={"width_ratios": [1.3, 1]})

# ── (좌) 쓸모가 없었다면 환자가 어떻게 갈렸을까 ─────────────────────
ax1.bar(k, pmf, width=1.0, color="tab:blue", alpha=0.75,
        label=f"백신이 쓸모없었다면 (n = {TOTAL} 명의 환자)")
ax1.axvline(K_VAC, color="tab:red", lw=2.4)
ax1.plot([K_VAC], [0], "v", ms=13, color="tab:red", clip_on=False, zorder=5)
ax1.annotate(f"실제로 나온 수 {K_VAC}",
             xy=(K_VAC, pmf.max() * 0.12), xytext=(26, pmf.max() * 0.60),
             fontsize=12, color="tab:red", fontweight="bold",
             arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.8))
ax1.annotate("", xy=(TOTAL / 2, pmf.max() * 1.06),
             xytext=(K_VAC, pmf.max() * 1.06),
             arrowprops=dict(arrowstyle="<->", color="0.35", lw=1.4))
ax1.text((K_VAC + TOTAL / 2) / 2, pmf.max() * 1.10,
         "이만큼 벗어났다", ha="center", va="bottom", fontsize=11,
         color="0.3", fontweight="bold")
ax1.text(0.985, 0.60, f"이만큼 치우칠 확률\n{p_two:.2e}", transform=ax1.transAxes,
         ha="right", va="top", fontsize=11.5, fontweight="bold",
         color="tab:red",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.94,
                   edgecolor="tab:red", lw=1.1))
ax1.set_xlim(-3, TOTAL + 3)
ax1.set_ylim(0, pmf.max() * 1.30)
ax1.set_xlabel("환자 170 명 가운데 백신군에서 나온 수", fontsize=11)
ax1.set_ylabel("확률", fontsize=11)
ax1.set_title("「그렇게 보인다」를 수로 옮기면", fontsize=12.5)
ax1.legend(fontsize=9.8, loc="upper left", framealpha=0.95)
ax1.grid(axis="y", alpha=0.2)

# ── (우) 자료를 보기 전에 적어 둔 선 ───────────────────────────────
VE, CLO, CHI = 95.0, 90.3, 97.6
BAR = 30.0
ax2.axvspan(0, BAR, color="0.88")
ax2.axvline(BAR, color="tab:red", lw=2.4)
ax2.text(BAR - 2, 0.86, "미리 그어 둔 선\n효과 30%", ha="right", va="top",
         fontsize=11, color="tab:red", fontweight="bold")
ax2.text(BAR / 2, 0.14, "성공이 아니다", ha="center", fontsize=10.5,
         color="0.45")
ax2.plot([CLO, CHI], [0.5, 0.5], "-", lw=7.0, color="tab:blue",
         solid_capstyle="butt")
for x in (CLO, CHI):
    ax2.plot([x, x], [0.44, 0.56], "-", lw=3.5, color="tab:blue")
ax2.plot([VE], [0.5], "o", ms=11, color="tab:blue")
ax2.text(CHI, 0.62, f"나온 효과 {VE:.1f}%", ha="right", fontsize=11.5,
         color="tab:blue", fontweight="bold")
ax2.text((CLO + CHI) / 2, 0.40, f"({CLO}, {CHI})", ha="center", va="top",
         fontsize=10.5, color="tab:blue")
ax2.text(50, 0.22, "논문이 쓴 베이즈 모형의 구간이고\n이 장의 방법이 아니다",
         ha="center", va="top", fontsize=9.8, color="0.45")
ax2.set_xlim(0, 100)
ax2.set_ylim(0, 1.0)
ax2.set_yticks([])
ax2.set_xlabel("백신 효과 (%)", fontsize=11)
ax2.set_title("규칙을 자료보다 먼저 적었다", fontsize=12.5)
ax2.grid(axis="x", alpha=0.25)

fig.suptitle("보이는 것을 말해도 되는지는 따로 재고, 그 자를 자료보다 먼저 적는다",
             fontsize=13.5, y=1.0)
plt.tight_layout()
path = OUT + "prespecified_rule.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
print("saved", path)

print(f"\n백신군 {N_VAC:,} 명 중 {K_VAC} 명 · 위약군 {N_PLA:,} 명 중 {K_PLA} 명")
print(f"  환자 {TOTAL} 명을 조건으로, 쓸모가 없었다면 백신군일 확률 "
      f"{P0:.6f}")
print(f"  그 분포의 평균 {TOTAL * P0:.2f} · 표준편차 "
      f"{np.sqrt(TOTAL * P0 * (1 - P0)):.2f}")
print(f"  {K_VAC} 이하가 나올 확률 {stats.binom.cdf(K_VAC, TOTAL, P0):.4e}")
print(f"  양쪽을 다 세면 {p_two:.4e}")
