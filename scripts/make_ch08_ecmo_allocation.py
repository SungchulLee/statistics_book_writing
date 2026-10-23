r"""$11$ 대 $1$ 로 나눈 열두 명 — 배분비가 표본 수에 매기는 값.

이 쪽은 두 집단을 견줄 때 필요한 표본 수를 구한다. 거의 모든 식이 **두 집단을 반씩**
나눈다고 놓는데, 왜 그래야 하는지는 따로 적지 않는다. $1985$ 년 미시간의 ECMO 시험이
$11$ 대 $1$ 로 나눈 적이 있어 그 값을 수로 볼 수 있다.

두 비율의 차의 분산은 $p_1q_1/n_1 + p_2q_2/n_2$ 다. 전체 $N$ 을 $r : 1$ 로 나누면
분산이 반반으로 나눌 때의 $(1+r)^2/(4r)$ 배가 된다 — $r = 11$ 이면 $3.27$ 배다.

① 전체 $12$ 명을 고정하고 배분비만 바꿀 때의 구간 반폭.
② 배분비가 요구하는 표본 수의 배수 $(1+r)^2/(4r)$.
③ 대조군 한 명으로 그 집단의 생존율에 대해 말할 수 있는 것.

만드는 파일: docs/ch08/sample_size/img/ecmo_allocation.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from scipy import stats   # noqa: E402

Z = stats.norm.ppf(0.975)
N_TOTAL = 12          # 미시간 시험에 들어간 영아 수
P_E, P_C = 0.90, 0.60  # 두 치료의 참 생존율로 가정한 값


def halfwidth(n1, n2, p1=P_E, p2=P_C):
    """두 비율 차의 95% Wald 반폭 (기대 분산으로 계산)."""
    v = p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2
    return Z * np.sqrt(v)


# === ① 전체 12 명을 어떻게 나누는가 ==================================
print(f"① 전체 {N_TOTAL} 명을 나누는 방법마다 (참 생존율 {P_E} 대 {P_C})")
# 한글은 고정폭에서 두 칸을 차지하므로 자리맞춤을 직접 적는다
print("   ECMO  대조   반폭    반반 대비")
SPLITS = [(n1, N_TOTAL - n1) for n1 in range(1, N_TOTAL)]
HW = [halfwidth(a, b) for a, b in SPLITS]
best = min(HW)
for (a, b), h in zip(SPLITS, HW):
    mark = "   <- 미시간" if a == 11 else ("   <- 가장 좁다" if h == best else "")
    print(f"   {a:>4}  {b:>4}   {h:>5.3f}   {h / best:>6.2f}{mark}")
h11 = halfwidth(11, 1)
h66 = halfwidth(6, 6)
print(f"   11 대 1 의 반폭 {h11:.3f} 은 6 대 6 의 {h11 / h66:.2f} 배다.")

# === ② 배분비가 매기는 값 ============================================
RS = np.array([1, 2, 3, 4, 5, 7, 9, 11, 15, 20], float)
INFL = (1 + RS) ** 2 / (4 * RS)
print("\n② 배분비 r : 1 이 요구하는 표본 수의 배수 (1+r)^2/(4r)")
print("      r    배수")
for r, f in zip(RS, INFL):
    print(f"   {int(r):>4}   {f:>5.2f}")
print(f"   r = 11 이면 {(1 + 11) ** 2 / 44:.2f} 배다. 곧 11 대 1 로 나눈 {N_TOTAL} 명은")
print(f"   반반으로 나눈 {N_TOTAL / ((1 + 11) ** 2 / 44):.1f} 명만큼의 정보를 준다.")

# === ③ 대조군 한 명 ==================================================
print("\n③ 대조군에서 아무도 살지 못했을 때 그 집단 생존율의 95% 정확 구간")
print("   대조군 수   구간 위끝")
NC = [1, 2, 3, 5, 10, 20, 50]
UPPER = []
for n in NC:
    # 0/n 일 때 Clopper-Pearson 위끝 = 1 - alpha^(1/n) 꼴
    hi = float(stats.beta.ppf(0.975, 1, n))
    UPPER.append(hi)
    print(f"   {n:>9}   {hi:>9.3f}")
print(f"   한 명으로는 생존율이 {UPPER[0]:.1%} 까지 열려 있다 — 사실상 아무 말도 못 한다.")

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.6, 5.2))
C_A, C_B, C_C = "#d62728", "#1f77b4", "#2ca02c"

xs = [a for a, _ in SPLITS]
ax1.plot(xs, HW, "o-", color=C_B, lw=2.6, ms=9)
i11 = xs.index(11)
i66 = xs.index(6)
ax1.plot([11], [HW[i11]], "o", color=C_A, ms=15, zorder=5)
ax1.annotate("미시간 $1985$\n$11$ 대 $1$", (11, HW[i11]),
             xytext=(-1.0, -2.6), textcoords="offset fontsize",
             fontsize=10.5, color=C_A, ha="right")
ax1.plot([6], [HW[i66]], "o", color=C_C, ms=15, zorder=5)
ax1.annotate("$6$ 대 $6$", (6, HW[i66]), xytext=(0.0, 1.1),
             textcoords="offset fontsize", fontsize=10.5, color=C_C,
             ha="center")
ax1.set_xlabel("ECMO 에 배정한 수 (나머지가 대조군, 전체 $12$)")
ax1.set_ylabel("두 생존율 차의 $95\\%$ 구간 반폭")
ax1.set_ylim(0.40, 1.06)
ax1.set_title("① 같은 열두 명을 어떻게 나누는가", fontsize=12)
ax1.grid(alpha=0.25)

ax2.plot(RS, INFL, "o-", color=C_B, lw=2.8, ms=10)
ax2.axhline(1.0, color=C_C, lw=2.0, ls="--")
ax2.text(13.5, 1.08, "반반이 바닥이다", fontsize=10.5, color=C_C)
ax2.plot([11], [(1 + 11) ** 2 / 44], "o", color=C_A, ms=15, zorder=5)
ax2.annotate(f"$r = 11$ 이면 ${(1 + 11) ** 2 / 44:.2f}$ 배", (11, (1 + 11) ** 2 / 44),
             xytext=(-0.6, 1.0), textcoords="offset fontsize",
             fontsize=10.5, color=C_A, ha="right")
ax2.set_xlabel("배분비 $r : 1$")
ax2.set_ylabel("필요한 표본 수의 배수")
ax2.set_title("② 치우쳐 나눌수록 더 많이 든다", fontsize=12)
ax2.grid(alpha=0.25)

ax3.plot(NC, UPPER, "o-", color=C_B, lw=2.8, ms=10)
ax3.plot([1], [UPPER[0]], "o", color=C_A, ms=15, zorder=5)
ax3.annotate(f"대조군 $1$ 명\n위끝 ${UPPER[0]:.3f}$", (1, UPPER[0]),
             xytext=(1.4, -2.0), textcoords="offset fontsize",
             fontsize=10.5, color=C_A)
ax3.set_xscale("log")
ax3.set_xlabel("대조군의 크기")
ax3.set_ylabel("생존율 $95\\%$ 구간의 위끝")
ax3.set_xticks([1, 10, 50])
ax3.set_xticklabels(["$1$", "$10$", "$50$"])
ax3.minorticks_off()
ax3.set_ylim(0, 1.05)
ax3.set_title("③ 아무도 살지 못했을 때 말할 수 있는 것", fontsize=12)
ax3.grid(alpha=0.25)

fig.tight_layout()
out = "docs/ch08/sample_size/img/ecmo_allocation.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
