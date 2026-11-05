"""11.3절 세 검정을 같은 자료에서 견주기 — p 값에 바닥이 생긴다.

들머리의 첫 요점을 그린다. 순열을 모두 훑어 구한 귀무분포는 가장 바깥
칸의 확률이 1/n! 이므로, 양측 p 값이 2/n! 보다 작아질 수 없다. n = 8 이면
2/40320 = 0.000049603 이고, 자료가 완전히 단조여도 그보다 작은 값은
나오지 않는다. 근사 분포에는 그 바닥이 없다 --- 스피어만의 t 근사는
완전히 단조인 자료에 p = 0 을 찍고, 바로 다음으로 단조인 자료에도
바닥 아래의 값을 찍는다.

오른쪽 칸은 그 바닥이 n 에 따라 어떻게 내려가는지다. n = 4 에서는 바닥이
0.0833 이라 **어떤 자료로도** 5% 에서 기각할 수 없고, n = 5 에서는 1% 에서
기각할 수 없다.

쪽의 correlation_tests_* 넷은 세 측도가 같은 자료에서 다른 수를 내는 것을
다룬다. 여기서는 계수가 아니라 p 값의 눈금만 본다.
"""

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

import math
from itertools import permutations

import numpy as np
from scipy import stats

OUT = "docs/ch11/correlation_test/img/"

N = 8
TOTAL = math.factorial(N)
PAIRS = N * (N - 1) // 2
FLOOR = 2 / TOTAL

idx = np.arange(N)
S_all = np.empty(TOTAL, dtype=np.int64)
D2_all = np.empty(TOTAL, dtype=np.int64)
for k, perm in enumerate(permutations(range(N))):
    p = np.array(perm)
    conc = 0
    for i in range(N - 1):
        conc += np.sign(p[i + 1:] - p[i]).sum()
    S_all[k] = conc
    D2_all[k] = ((p - idx) ** 2).sum()

tau_all = S_all / PAIRS
rs_all = 1 - 6 * D2_all / (N * (N * N - 1))


def ladder(stat):
    """|통계량| 이 그 값 이상일 확률 --- 곧 닿을 수 있는 양측 p 값."""
    a = np.round(np.abs(stat), 10)          # 같은 값이 부동소수로 갈리지 않게
    vals = np.unique(a)[::-1]
    return vals, np.array([(a >= v).mean() for v in vals])

tau_v, tau_p = ladder(tau_all)
rs_v, rs_p = ladder(rs_all)

# 스피어만의 t 근사 --- 바닥이 없다.
with np.errstate(divide="ignore", invalid="ignore"):
    t_stat = rs_v * np.sqrt((N - 2) / (1 - rs_v ** 2))
rs_approx = 2 * stats.t.sf(np.abs(t_stat), N - 2)
rs_approx[~np.isfinite(t_stat)] = 0.0

K = 8                                   # 가장 바깥 여덟 칸만 그린다
fig, (ax1, ax2, ax3) = plt.subplots(
    1, 3, figsize=(15.0, 4.7), gridspec_kw={"width_ratios": [1.15, 1.1, 1]})

# ── (좌) 귀무분포의 가장 바깥 칸은 순열 하나뿐이다 ──────────────────
vals, cnts = np.unique(S_all, return_counts=True)
cols = ["tab:red" if abs(v) == PAIRS else "0.68" for v in vals]
ax1.bar(vals / PAIRS, cnts / TOTAL, width=2 / PAIRS * 0.85, color=cols)
ax1.set_yscale("log")                 # 양 끝 칸은 가운데의 사천분의 일이다
ax1.set_ylim(1.2e-5, 0.35)
ax1.set_yticks([1e-4, 1e-3, 1e-2, 1e-1],
               ["0.0001", "0.001", "0.01", "0.1"])
ax1.set_yticks([], minor=True)
ax1.annotate(f"양 끝은 순열이 하나씩뿐이다\n각각 1/{TOTAL:,} = {1/TOTAL:.8f}",
             xy=(1.0, 1 / TOTAL), xytext=(0.62, 0.0016), ha="center",
             fontsize=10.5, color="tab:red", fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                       alpha=0.94, edgecolor="tab:red", lw=1.0),
             arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.6))
ax1.set_xlabel("τ", fontsize=11)
ax1.set_ylabel("확률 (로그 눈금)", fontsize=11)
ax1.set_title(f"$n = {N}$ 에서 $\\tau$ 의 귀무분포\n"
              f"(순열 {TOTAL:,} 개를 모두 셌다)", fontsize=12.5)
ax1.grid(axis="y", alpha=0.25)

# ── (중) 닿을 수 있는 p 값은 사다리다 ──────────────────────────────
x = np.arange(1, K + 1)
ax2.axhspan(1e-7, FLOOR, color="0.80", alpha=0.55)
ax2.axhline(FLOOR, color="0.25", lw=2.0, ls="--")
ax2.text(K + 0.35, FLOOR * 1.35, f"바닥 2/{TOTAL:,} = {FLOOR:.8f}", ha="right",
         fontsize=10.5, color="0.2", fontweight="bold")
ax2.text(K + 0.35, 2.2e-7, "어떤 자료도 닿을 수 없는 곳", ha="right",
         fontsize=10.5, color="0.35", fontweight="bold")
ax2.plot(x, tau_p[:K], "o-", ms=8, lw=1.6, color="tab:blue",
         label="켄달 τ — 순열을 모두 센 값")
ax2.plot(x, rs_p[:K], "s-", ms=7, lw=1.6, color="tab:green",
         label="스피어만 — 순열을 모두 센 값")
pos = rs_approx[:K] > 0
ax2.plot(x[pos], rs_approx[:K][pos], "^-", ms=8, lw=1.6, color="tab:red",
         label="스피어만 — t 근사")
ax2.annotate(f"완전히 단조인 자료에\n$p = 0$ 이 찍힌다", xy=(1, 1.6e-7),
             xytext=(1.9, 6e-7), fontsize=10.5, color="tab:red",
             fontweight="bold",
             arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.6))
ax2.plot([1], [1.6e-7], "v", ms=11, color="tab:red")
ax2.set_yscale("log")
ax2.set_ylim(1e-7, 2.0)
ax2.set_xlim(0.45, K + 0.55)
TICKS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
ax2.set_yticks(TICKS, ["0.000001", "0.00001", "0.0001", "0.001", "0.01",
                       "0.1", "1"])
ax2.set_yticks([], minor=True)
ax2.set_xticks(x)
ax2.set_xlabel("가장 바깥에서 몇 번째 칸인가", fontsize=11)
ax2.set_ylabel("양측 p 값", fontsize=11)
ax2.set_title("닿을 수 있는 $p$ 값은 사다리다", fontsize=12.5)
ax2.legend(fontsize=9.5, loc="upper left", framealpha=0.95)
ax2.grid(alpha=0.25, which="major")

# ── (우) 바닥은 n 이 정한다 ────────────────────────────────────────
ns = np.arange(4, 11)
floors = np.array([2 / math.factorial(int(n)) for n in ns])
ax3.plot(ns, floors, "o-", ms=9, lw=2.2, color="tab:purple")
for lv, nm, c in ((0.05, "5%", "tab:red"), (0.01, "1%", "tab:orange")):
    ax3.axhline(lv, color=c, ls="--", lw=1.6)
    ax3.text(3.68, lv * 0.60, nm, ha="left", va="top", fontsize=11, color=c,
             fontweight="bold")
ax3.plot([4], [floors[0]], "o", ms=14, mfc="none", mec="tab:red", mew=2.4)
ax3.plot([5], [floors[1]], "o", ms=14, mfc="none", mec="tab:orange", mew=2.4)
ax3.text(0.975, 0.975,
         f"$n = 4$ 의 바닥은 {floors[0]:.4f}\n"
         f"— 어떤 자료도 $5\\%$ 에서 기각할 수 없다\n"
         f"$n = 5$ 의 바닥은 {floors[1]:.4f}\n"
         f"— $1\\%$ 에서 기각할 수 없다",
         transform=ax3.transAxes, ha="right", va="top", fontsize=10.5,
         color="0.2", fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.94,
                   edgecolor="0.5", lw=1.0))
ax3.set_yscale("log")
ax3.set_ylim(2e-8, 1.4)
ax3.set_xlim(3.6, 10.4)
ax3.set_yticks([1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1],
               ["0.0000001", "0.000001", "0.00001", "0.0001", "0.001",
                "0.01", "0.1", "1"])
ax3.set_yticks([], minor=True)
ax3.set_xticks(ns)
ax3.set_xlabel("표본 크기 n", fontsize=11)
ax3.set_ylabel("닿을 수 있는 가장 작은 양측 p 값", fontsize=11)
ax3.set_title("바닥을 정하는 것은 $2/n!$ 이다", fontsize=12.5)
ax3.grid(alpha=0.25, which="major")

fig.suptitle("순열을 세어 구한 p 값에는 바닥이 있고, 근사 분포에는 없다",
             fontsize=14, y=1.0)
plt.tight_layout()
path = OUT + "p_value_floor.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
print("saved", path)

print(f"\nn = {N} · 순열 {TOTAL:,} 개 · 양측 p 의 바닥 2/{TOTAL} = "
      f"{FLOOR:.10f} = {FLOOR:.4e}")
print(f"  가장 바깥 칸의 확률 1/{TOTAL} = {1/TOTAL:.10f}")
print("\n가장 바깥 여덟 칸에서 닿을 수 있는 양측 p 값")
print("   칸   |τ|       켄달(정확)     스피어만(정확)   스피어만(t 근사)")
for i in range(K):
    print(f"  {i+1:>3d}  {tau_v[i]:.4f}    {tau_p[i]:.8f}     "
          f"{rs_p[i]:.8f}      {rs_approx[i]:.8f}"
          f"{'   <- 바닥 아래' if 0 < rs_approx[i] < FLOOR else ''}")
print("\n표본 크기에 따른 바닥 2/n!")
for n, f in zip(ns, floors):
    note = []
    if f > 0.05:
        note.append("5% 에서 기각 불가")
    elif f > 0.01:
        note.append("1% 에서 기각 불가")
    print(f"  n = {n:>2d}  →  {f:.10f}   {' · '.join(note)}")
