r"""켤레 사전분포가 지금도 쓰이는 자리 — 버리고도 정확한 갱신.

베타–베르누이 켤레에서 사후분포는 자료가 몇 개든 **두 수** $(\alpha+s,\ \beta+n-s)$
로 끝난다. 자료를 저장할 필요가 없고, 갱신이 근사가 아니라 정확하다.
그 성질이 온라인 의사결정 — 밴딧과 A/B 시험 — 에서 그대로 쓰인다.

① 상태의 크기 — 켤레는 $n$ 과 무관하게 두 수다.
② 격자로 같은 일을 하려면 — 정확해질수록 비싸진다. 켤레는 둘 다 이긴다.
③ 톰슨 표집 — 그 값싼 사후분포가 탐색과 활용을 알아서 맞춘다.

만드는 파일: docs/ch06/bayesian/img/conjugate_streaming.png
"""

import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from scipy import stats   # noqa: E402

RNG = np.random.default_rng(0)
NS = [10, 100, 1000, 10 ** 4, 10 ** 5, 10 ** 6]
KS = [11, 51, 201, 1001, 5001, 20001]
S_OBS, N_OBS = 37, 100
P_ARMS = np.array([0.30, 0.33, 0.36])
T, R = 3000, 400

A, B = 1.0 + S_OBS, 1.0 + N_OBS - S_OBS
EXACT_M = A / (A + B)
EXACT_Q = stats.beta(A, B).ppf([0.025, 0.975])
EXACT_W = float(EXACT_Q[1] - EXACT_Q[0])
print(f"성공 {S_OBS}/{N_OBS} 의 정확한 사후분포 — 평균 {EXACT_M:.6f}, "
      f"95% 구간 폭 {EXACT_W:.6f}")

print(f"\n② 격자로 같은 일을 하려면")
print(f"{'격자 칸수 K':>11}{'구간폭의 오차':>15}{'갱신 1 회 (us)':>16}")
ERR, TG = [], []
for K in KS:
    g = np.linspace(1e-9, 1 - 1e-9, K)
    lp = S_OBS * np.log(g) + (N_OBS - S_OBS) * np.log1p(-g)
    p = np.exp(lp - lp.max())
    p /= p.sum()
    c = np.cumsum(p)
    w = g[np.searchsorted(c, 0.975)] - g[np.searchsorted(c, 0.025)]
    ERR.append(abs(w - EXACT_W))
    t0 = time.perf_counter()
    for _ in range(500):
        _ = lp + np.log(g)
    TG.append((time.perf_counter() - t0) / 500 * 1e6)
    print(f"{K:>11,}{ERR[-1]:>15.2e}{TG[-1]:>16.2f}")
t0 = time.perf_counter()
aa = 1.0
for _ in range(500):
    aa += 1
TC = (time.perf_counter() - t0) / 500 * 1e6
print(f"   켤레 갱신 1 회 : {TC:.3f} us,  오차 0 (근사가 아니라 정확)")
print(f"   K = 1001 과 견주면 {TG[3] / TC:.0f} 배 빠르면서 더 정확하다.")

print(f"\n③ 톰슨 표집 (팔 셋, 성공확률 {list(P_ARMS)}, T = {T:,}, {R} 번)")


def bandit(kind, eps=0.1):
    reg = np.zeros(T)
    for _ in range(R):
        a = np.ones(3)
        b = np.ones(3)
        s = np.zeros(3)
        c = np.zeros(3)
        tot = 0.0
        for t in range(T):
            if kind == "톰슨 표집":
                k = int(np.argmax(RNG.beta(a, b)))
            else:
                hat = np.where(c > 0, s / np.maximum(c, 1), 1.0)
                k = (int(RNG.integers(3))
                     if kind == "엡실론 탐욕" and RNG.random() < eps
                     else int(np.argmax(hat)))
            x = RNG.random() < P_ARMS[k]
            a[k] += x
            b[k] += 1 - x
            s[k] += x
            c[k] += 1
            tot += P_ARMS.max() - P_ARMS[k]
            reg[t] += tot
    return reg / R


REG = {}
for kind in ("톰슨 표집", "엡실론 탐욕", "늘 최선만"):
    REG[kind] = bandit(kind)
    r = REG[kind]
    print(f"   {kind:>8} : 누적 후회  T=500 {r[499]:6.1f}   "
          f"T=1500 {r[1499]:6.1f}   T={T} {r[-1]:6.1f}")

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.6, 5.1))
C_C, C_G, C_D = "#2ca02c", "#d62728", "#1f77b4"

ax1.loglog(NS, [16] * len(NS), "o-", color=C_C, lw=2.8, ms=8,
           label="켤레 — 두 수")
ax1.loglog(NS, [8 * n for n in NS], "s-", color=C_D, lw=2.8, ms=8,
           label="자료를 그대로 들고 있기")
ax1.loglog(NS, [8 * 1001] * len(NS), "^--", color=C_G, lw=2.4, ms=8,
           label="격자 $K = 1001$")
ax1.set_xticks(NS)
ax1.set_xticklabels(["$10$", "$10^{2}$", "$10^{3}$", "$10^{4}$", "$10^{5}$",
                     "$10^{6}$"], fontsize=9.5)
ax1.set_yticks([1e1, 1e3, 1e5, 1e7])
ax1.set_yticklabels(["$10$", "$10^{3}$", "$10^{5}$", "$10^{7}$"])
ax1.minorticks_off()
ax1.set_xlabel("본 관측의 수 $n$")
ax1.set_ylabel("들고 있어야 하는 바이트")
ax1.set_title("① 켤레는 자료를 버려도 된다", fontsize=12)
ax1.legend(loc="upper left", fontsize=10)
ax1.grid(alpha=0.25)

ax2.loglog(TG, ERR, "o-", color=C_G, lw=2.6, ms=8, label="격자")
for k, K in enumerate(KS):
    ax2.annotate(f"$K={K:,}$".replace(",", "{,}"), (TG[k], ERR[k]),
                 xytext=(0.6, 0.5), textcoords="offset fontsize",
                 fontsize=8.5, color=C_G)
FLOOR = 4e-7
ax2.plot([TC], [FLOOR], "*", color=C_C, ms=24, zorder=4, label="켤레")
ax2.annotate(f"켤레 — ${TC:.3f}$ $\\mu$s, 오차 $0$\n(축 아래, 찍을 자리가 없다)",
             xy=(TC, FLOOR), xytext=(0.9, 2.0e-6), textcoords="data",
             fontsize=10.5, color=C_C,
             arrowprops=dict(arrowstyle="->", color=C_C, lw=1.1))
ax2.set_xlabel("갱신 한 번에 드는 시간 ($\\mu$s)")
ax2.set_ylabel("$95\\%$ 구간폭의 오차")
# 로그 눈금의 마이너스가 mathtext 라 두부가 된다 — 눈금을 직접 적는다
ax2.set_xticks([0.03, 0.1, 0.3, 1, 3, 10, 30])
ax2.set_xticklabels(["$0.03$", "$0.1$", "$0.3$", "$1$", "$3$", "$10$", "$30$"],
                    fontsize=9.5)
ax2.set_yticks([1e-6, 1e-5, 1e-4, 1e-3, 1e-2])
ax2.set_yticklabels(["$10^{-6}$", "$10^{-5}$", "$10^{-4}$", "$10^{-3}$",
                     "$10^{-2}$"])
ax2.minorticks_off()
ax2.set_xlim(0.02, 80)
ax2.set_ylim(FLOOR * 0.7, 4e-2)
ax2.set_title("② 격자는 정확해질수록 비싸진다", fontsize=12)
ax2.legend(loc="upper right", fontsize=10.5)
ax2.grid(alpha=0.25)

ts = np.arange(1, T + 1)
for kind, c, ls in (("톰슨 표집", C_C, "-"), ("엡실론 탐욕", C_D, "-"),
                    ("늘 최선만", C_G, "--")):
    ax3.plot(ts, REG[kind], ls, color=c, lw=2.6, label=kind)
for kind, c in (("톰슨 표집", C_C), ("엡실론 탐욕", C_D), ("늘 최선만", C_G)):
    ax3.text(T * 1.01, REG[kind][-1], f"${REG[kind][-1]:.0f}$", color=c,
             fontsize=10.5, va="center")
ax3.set_xlabel("걸음 수 $t$")
ax3.set_ylabel("누적 후회")
ax3.set_xlim(0, T * 1.13)
ax3.set_title("③ 값싼 사후분포가 하는 일", fontsize=12)
ax3.legend(loc="upper left", fontsize=10.5)
ax3.grid(alpha=0.25)

fig.tight_layout()
out = "docs/ch06/bayesian/img/conjugate_streaming.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
