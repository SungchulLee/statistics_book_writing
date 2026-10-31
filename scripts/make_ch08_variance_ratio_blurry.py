"""8.3절 분산비의 신뢰구간 — 흐리게 짚인다는 말의 크기.

들머리가 든 보기를 그린다. 참 분산비가 0.4444 로 두 배 넘게 다른데도
95% 구간이 (0.2396, 2.4903) 이라 참값과 1 을 **함께** 담는다. 상한이
하한의 10.394 배인 것은 자유도 쌍이 정한 값이고 자료와 무관하다.

오른쪽은 그 다음 물음이다. 그렇게 흐리게 짚이면, 같은 참값을 1 에서
갈라내려면 표본이 얼마나 커야 하는가.

쪽의 ci_variance_ratio_74 는 F 구간 100 개를 쌓아 포함률을 보이고,
상자의 tail_variance 는 서버 수에 따른 확률과 꼬리가 다른 두 분포와
치우친 자료에서의 포함률을 다룬다. 여기서는 그 둘이 건드리지 않는 것 ---
구간 하나의 넓이와, 그 넓이를 줄이는 데 드는 n --- 만 본다.
"""

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

import numpy as np
from scipy import stats

OUT = "docs/ch08/two_sample_intervals/img/"

N1, N2 = 15, 12
SIG1, SIG2 = 1.0, 1.5
THETA = SIG1 ** 2 / SIG2 ** 2          # 참 분산비 0.4444
LO, HI = 0.2396, 2.4903                # 쪽이 적은 보기의 구간
D1, D2 = N1 - 1, N2 - 1
F_LO = stats.f.ppf(0.025, D1, D2)
F_HI = stats.f.ppf(0.975, D1, D2)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 4.6),
                               gridspec_kw={"width_ratios": [1.2, 1]})

# ── (좌) 구간 하나가 참값과 1 을 함께 담는다 ────────────────────────
ax1.plot([LO, HI], [0, 0], "-", lw=8.0, color="tab:blue",
         solid_capstyle="butt")
for x in (LO, HI):
    ax1.plot([x, x], [-0.13, 0.13], "-", lw=4.0, color="tab:blue")
    ax1.text(x, -0.22, f"{x:.4f}", ha="center", va="top", fontsize=11,
             color="tab:blue", fontweight="bold")
ax1.plot([THETA], [0], "o", ms=13, color="tab:green", zorder=5)
ax1.text(THETA, 0.21, f"참 분산비 {THETA:.4f}\n(두 배 넘게 다르다)",
         ha="center", va="bottom", fontsize=11, color="tab:green",
         fontweight="bold")
ax1.plot([1.0], [0], "D", ms=11, color="tab:red", zorder=5)
ax1.text(1.0, -0.42, "「두 분산이 같다」\n= 1", ha="center", va="top",
         fontsize=11, color="tab:red", fontweight="bold")
ax1.annotate("", xy=(HI, 0.42), xytext=(LO, 0.42),
             arrowprops=dict(arrowstyle="<->", color="0.3", lw=1.6))
ax1.text(np.sqrt(LO * HI), 0.47, f"상한 ÷ 하한 = {HI / LO:.3f}", ha="center",
         va="bottom", fontsize=11.5, color="0.2", fontweight="bold")
ax1.set_xscale("log")
ax1.set_xlim(0.16, 4.2)
ax1.set_ylim(-0.95, 0.80)
ax1.set_yticks([])
TICKS = [0.2, 0.3, 0.5, 1.0, 2.0, 3.0, 4.0]
ax1.set_xticks(TICKS, [f"{t:g}" for t in TICKS])
ax1.set_xticks([], minor=True)
ax1.set_xlabel("분산의 비 (로그 눈금)", fontsize=11)
ax1.set_title(f"$n_1 = {N1}$, $n_2 = {N2}$ 에서 얻은 $95\\%$ 구간 하나",
              fontsize=12.5)
ax1.grid(axis="x", alpha=0.25)
ax1.text(0.5, 0.025, "참값도 담고 「같다」도 함께 담는다",
         transform=ax1.transAxes, ha="center", va="bottom", fontsize=11.5,
         fontweight="bold", color="0.2",
         bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.94,
                   edgecolor="0.5", lw=1.0))

# ── (우) 1 을 갈라내려면 몇 개가 드나 ──────────────────────────────
ns = np.arange(5, 401)
d = ns - 1
hi_c = stats.f.ppf(0.975, d, d)
lo_c = stats.f.ppf(0.025, d, d)
power = stats.f.sf(hi_c / THETA, d, d) + stats.f.cdf(lo_c / THETA, d, d)
need = int(ns[power >= 0.80][0])
ax2.plot(ns, 100 * power, color="tab:purple", lw=2.4)
ax2.axhline(80, color="0.4", ls="--", lw=1.5)
ax2.text(395, 81.5, "열에 여덟", ha="right", fontsize=10.5, color="0.4")
ax2.plot([need], [100 * power[ns == need][0]], "o", ms=9, color="tab:purple")
ax2.annotate(f"군마다 {need} 개는 되어야\n열에 여덟을 갈라낸다",
             xy=(need, 80), xytext=(need + 45, 52), fontsize=11,
             color="tab:purple", fontweight="bold",
             arrowprops=dict(arrowstyle="->", color="tab:purple", lw=1.6))
pw12 = stats.f.sf(stats.f.ppf(0.975, D1, D2) / THETA, D1, D2) + \
    stats.f.cdf(stats.f.ppf(0.025, D1, D2) / THETA, D1, D2)
ax2.plot([12], [100 * pw12], "o", ms=9, color="tab:blue")
ax2.annotate(f"왼쪽 그림의 자리 — 다만 거기는\n군이 {N1} 과 {N2} 로 달랐다  "
             f"{100 * pw12:.1f}%", xy=(12, 100 * pw12),
             xytext=(46, 100 * pw12 - 4), fontsize=10.5, color="tab:blue",
             fontweight="bold",
             arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1.4))
ax2.set_xlim(0, 400)
ax2.set_ylim(0, 100)
ax2.set_xlabel("군마다의 표본 크기", fontsize=11)
ax2.set_ylabel("구간이 1 을 배제할 확률 (%)", fontsize=11)
ax2.set_title(f"참 분산비가 {THETA:.4f} 일 때", fontsize=12.5)
ax2.grid(alpha=0.25)

fig.suptitle("분산의 비는 흐리게 짚인다 — 그 흐림의 크기", fontsize=14, y=1.0)
plt.tight_layout()
path = OUT + "variance_ratio_blurry.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
print("saved", path)

print(f"\n참 분산비 {THETA:.6f} · 쪽의 구간 ({LO}, {HI})")
print(f"  상한/하한 {HI/LO:.5f}  ·  임계값 비 "
      f"{F_HI/F_LO:.5f}  (자유도 {D1}, {D2})")
print(f"  그 구간이 참값과 1 을 모두 담는다: "
      f"{LO <= THETA <= HI} / {LO <= 1 <= HI}")
print(f"\n구간이 1 을 배제할 확률")
for n in (12, 30, 60, 100, need, 200):
    dd = n - 1
    pw = stats.f.sf(stats.f.ppf(0.975, dd, dd) / THETA, dd, dd) + \
        stats.f.cdf(stats.f.ppf(0.025, dd, dd) / THETA, dd, dd)
    print(f"  군마다 {n:>3d} 개 → {100*pw:.2f}%")
print(f"  열에 여덟을 넘는 첫 크기 = {need}")
print(f"  쪽의 보기 (n1={N1}, n2={N2}) 에서는 {100*pw12:.2f}%")
