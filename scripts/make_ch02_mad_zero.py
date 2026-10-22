r"""MAD 가 $0$ 이 되는 평범한 자료.

이 쪽의 $\S 3$–$\S 4$ 는 MAD 가 $0$ 으로 무너지는 일을 **절반이 넘는 오염**의
결과로 다룬다. 그런데 그 일은 공격받지 않아도 일어난다 — **같은 값이 절반을
넘기만 하면** 된다. $0$ 이 많이 섞인 지표, 눈금 위에 놓인 지표에서 흔하다.

① 동점의 비율이 $0.5$ 를 넘으면 $\mathrm{MAD} = 0$ 이 된다. 경계가 날카롭다.
② 그 자리에서 $\mathrm{MAD}$ 기반 이상점 규칙은 자료의 절반 가까이를 찍는다.
   사분위범위 규칙은 버틴다.
③ 눈금 위에 놓인 지표 — 흩어짐이 눈금의 절반 아래면 $\mathrm{MAD}$ 가 늘 $0$ 이고,
   **관측을 늘릴수록 더 그렇다.**

만드는 파일: docs/ch02/spread/img/mad_zero.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

RNG = np.random.default_rng(5)
C = 0.6745                       # 이 쪽 정리 1 의 상수
THR = 3.5                        # 흔히 쓰는 로버스트 z 문턱
P0 = [0.1, 0.2, 0.3, 0.4, 0.45, 0.48, 0.5, 0.52, 0.55, 0.6, 0.7, 0.8]
NS = [50, 200, 1000]
N_FLAG, R_FLAG = 200, 1500
RATIOS = [0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 2.0, 4.0]
NS_GRID = [10, 30, 100]
R_GRID = 4000

mad = lambda x: float(np.median(np.abs(x - np.median(x))))   # noqa: E731


def qn(x):
    """루소·크루의 Qn — 쌍별 거리의 아래쪽 사분위에 기댄 척도."""
    d = np.abs(x[:, None] - x[None, :])
    d = d[np.triu_indices(len(x), 1)]
    h = len(x) // 2 + 1
    k = h * (h - 1) // 2
    return 2.2219 * float(np.partition(d, k - 1)[k - 1])


def zero_inflated(n, p0):
    return np.where(RNG.random(n) < p0, 0.0, RNG.exponential(1.0, n))


def flag_rate(x, scale, med):
    """로버스트 z 로 찍히는 비율. 척도가 0 이면 중앙값 아닌 점이 모두 찍힌다."""
    if scale == 0.0:
        return float(np.mean(x != med))
    return float(np.mean(np.abs(C * (x - med) / scale) > THR))


# === ① MAD 가 0 이 될 확률 ===========================================
print(f"① 동점이 섞인 자료에서 MAD = 0 이 될 확률")
print(f"{'동점 비율':>9}" + "".join(f"{f'n={n}':>9}" for n in NS))
PZ = {n: [] for n in NS}
for p0 in P0:
    for n in NS:
        R = 3000 if n <= 200 else 800
        PZ[n].append(float(np.mean([mad(zero_inflated(n, p0)) == 0
                                    for _ in range(R)])))
    print(f"{p0:>9.2f}" + "".join(f"{PZ[n][-1]:>9.3f}" for n in NS))

# === ② 이상점으로 찍히는 비율 ========================================
print(f"\n② 이상점으로 찍히는 비율 (n = {N_FLAG}, {R_FLAG} 번)")
print(f"{'동점 비율':>9}{'MAD 로버스트 z':>15}{'사분위범위 1.5':>15}{'Qn 로버스트 z':>14}")
FL = {k: [] for k in ("mad", "iqr", "qn")}
for p0 in P0:
    a = {k: 0.0 for k in FL}
    for _ in range(R_FLAG):
        x = zero_inflated(N_FLAG, p0)
        med = float(np.median(x))
        a["mad"] += flag_rate(x, mad(x), med)
        a["qn"] += flag_rate(x, qn(x), med)
        lo, hi = np.percentile(x, [25, 75])
        r = hi - lo
        a["iqr"] += float(np.mean((x < lo - 1.5 * r) | (x > hi + 1.5 * r)))
    for k in FL:
        FL[k].append(a[k] / R_FLAG)
    print(f"{p0:>9.2f}{FL['mad'][-1]:>15.3f}{FL['iqr'][-1]:>15.3f}"
          f"{FL['qn'][-1]:>14.3f}")
i50 = P0.index(0.5)
print(f"   동점이 절반이면 MAD 규칙이 자료의 {100 * FL['mad'][i50]:.0f}% 를 "
      f"이상점이라 한다. 사분위범위 규칙은 {100 * FL['iqr'][i50]:.0f}% 다.")
print(f"   Qn 도 함께 무너진다 ({100 * FL['qn'][i50]:.0f}%) — "
      f"쌍별 거리의 아래쪽 사분위도 동점이 차지하기 때문이다.")

# === ③ 눈금 위에 놓인 지표 ===========================================
print(f"\n③ 눈금 위에 놓인 지표에서 MAD = 0 이 될 확률")
print(f"{'흩어짐/눈금':>11}" + "".join(f"{f'n={n}':>9}" for n in NS_GRID))
PG = {n: [] for n in NS_GRID}
for r in RATIOS:
    for n in NS_GRID:
        PG[n].append(float(np.mean(
            [mad(np.round(RNG.normal(0, r, n))) == 0 for _ in range(R_GRID)])))
    print(f"{r:>11.2f}" + "".join(f"{PG[n][-1]:>9.3f}" for n in NS_GRID))
print("   흩어짐이 눈금의 절반 아래면 거의 늘 0 이고, "
      "관측을 늘릴수록 더 그렇다.")

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.6, 5.0))
CN = {50: "#1f77b4", 200: "#d62728", 1000: "#2ca02c"}
CG = {10: "#1f77b4", 30: "#d62728", 100: "#2ca02c"}

for n in NS:
    ax1.plot(P0, PZ[n], "o-", color=CN[n], lw=2.6, ms=6, label=f"$n = {n}$")
ax1.axvline(0.5, color="black", lw=1.5, ls="--")
ax1.text(0.505, 0.06, "동점이 절반", fontsize=10.5)
ax1.set_xlabel("같은 값이 차지하는 비율")
ax1.set_ylabel("$\\mathrm{MAD} = 0$ 이 될 확률")
ax1.set_ylim(-0.03, 1.05)
ax1.set_title("① 경계가 날카롭다", fontsize=12)
ax1.legend(loc="upper left", fontsize=10)
ax1.grid(alpha=0.25)

for k, (nm, c) in [("mad", ("$\\mathrm{MAD}$ 로버스트 $z$", "#d62728")),
                   ("qn", ("$Q_n$ 로버스트 $z$", "#ff7f0e")),
                   ("iqr", ("사분위범위 $1.5$ 규칙", "#2ca02c"))]:
    ax2.plot(P0, FL[k], "o-", color=c, lw=2.6, ms=6, label=nm)
ax2.axvline(0.5, color="black", lw=1.5, ls="--")
ax2.annotate(f"동점이 절반이면\n자료의 ${100 * FL['mad'][i50]:.0f}\\%$ 가 "
             f"이상점이 된다",
             xy=(0.5, FL["mad"][i50]), xytext=(0.12, 0.72),
             textcoords="data", fontsize=10.5, color="#d62728",
             arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.1))
ax2.set_xlabel("같은 값이 차지하는 비율")
ax2.set_ylabel("이상점으로 찍히는 비율")
ax2.set_ylim(-0.03, 1.05)
ax2.set_title(f"② 규칙 셋 가운데 하나만 버틴다 ($n = {N_FLAG}$)", fontsize=12)
ax2.legend(loc="upper right", fontsize=9.5)
ax2.grid(alpha=0.25)

for n in NS_GRID:
    ax3.semilogx(RATIOS, PG[n], "o-", color=CG[n], lw=2.6, ms=6,
                 label=f"관측 ${n}$ 개")
ax3.axvline(0.5, color="black", lw=1.5, ls="--")
ax3.text(0.52, 0.5, "흩어짐 = 눈금의 절반", fontsize=10.5, rotation=90,
         va="center")
ax3.set_xticks(RATIOS)
ax3.set_xticklabels([f"${v}$" for v in RATIOS], fontsize=9.5)
ax3.minorticks_off()
ax3.set_xlabel("흩어짐 $\\div$ 눈금 간격")
ax3.set_ylabel("$\\mathrm{MAD} = 0$ 이 될 확률")
ax3.set_ylim(-0.03, 1.05)
ax3.set_title("③ 늘릴수록 나빠지는 쪽", fontsize=12)
ax3.legend(loc="lower left", fontsize=10)
ax3.grid(alpha=0.25)

fig.tight_layout()
out = "docs/ch02/spread/img/mad_zero.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
