r"""주변합에 조건을 걸면 눈금이 얼마나 헐거워지는가.

$n_1 = n_2 = 15$ 인 $2\times2$ 표에서 **모든 표를 전수로** 펼쳐 계산한다
($16 \times 16 = 256$ 개). 모의실험이 아니다.

① 실제 크기 — 참 $p$ 를 바꾸어 가며 피셔 검정이 실제로 기각하는 확률.
   이산성 때문에 $0.05$ 에 한참 못 미치고 톱니처럼 흔들린다.
   피셔의 $p$ 값 문턱을 **크기가 $0.05$ 를 넘지 않는 가장 큰 값**까지 올리면
   (주변합에 조건을 걸지 않고 눈금을 다시 매기는 것) 어디까지 올라오는가.
② 그렇게 되찾은 검정력.

만드는 파일: docs/ch10/practice/img/barnard_size_power.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from scipy import stats  # noqa: E402

N1 = N2 = 15
ALPHA = 0.05
XS = np.arange(N1 + 1)

# --- 표 256 개의 피셔 양측 p 값을 미리 구해 둔다 ----------------------
pf = np.empty((N1 + 1, N2 + 1))
for a in XS:
    for b in XS:
        pf[a, b] = stats.fisher_exact([[a, N1 - a], [b, N2 - b]],
                                      alternative="two-sided")[1]


def table_prob(p1, p2):
    """두 이항표본이 (a, b) 를 낼 확률의 16x16 행렬."""
    return np.outer(stats.binom.pmf(XS, N1, p1), stats.binom.pmf(XS, N2, p2))


def size_curve(gamma, ps):
    """참 p 를 ps 로 훑으며 실제 크기(1종 오류율)를 잰다."""
    rej = pf <= gamma + 1e-12
    return np.array([table_prob(p, p)[rej].sum() for p in ps])


PS = np.linspace(0.02, 0.98, 481)

# --- 크기가 0.05 를 넘지 않는 가장 큰 문턱을 찾는다 -------------------
levels = np.unique(pf)
levels = levels[levels <= 0.5]
ok = [g for g in levels if size_curve(g, PS).max() <= ALPHA + 1e-12]
GAMMA = max(ok)
print(f"① 피셔의 p 값 문턱을 {ALPHA} 에서 {GAMMA:.6f} 까지 올려도 크기가 넘지 않는다")

s_fisher = size_curve(ALPHA, PS)
s_raised = size_curve(GAMMA, PS)
print(f"   실제 크기의 최댓값   피셔 {s_fisher.max():.4f}   올린 뒤 {s_raised.max():.4f}")
print(f"   실제 크기의 평균     피셔 {s_fisher.mean():.4f}   올린 뒤 {s_raised.mean():.4f}")
for p in [0.1, 0.3, 0.5, 0.7]:
    i = int(np.argmin(np.abs(PS - p)))
    print(f"   p = {PS[i]:.2f}:  피셔 {s_fisher[i]:.4f}   올린 뒤 {s_raised[i]:.4f}")

# --- 검정력 ----------------------------------------------------------
def power(gamma, p1, p2):
    rej = pf <= gamma + 1e-12
    return table_prob(p1, p2)[rej].sum()


DELTAS = [(0.7, 0.3), (0.6, 0.2), (0.8, 0.4), (0.9, 0.5), (0.6, 0.3)]
print("\n② 검정력")
print("    p1    p2     피셔     올린 뒤      차")
for p1, p2 in DELTAS:
    a, b = power(ALPHA, p1, p2), power(GAMMA, p1, p2)
    print(f"  {p1:.1f}   {p2:.1f}   {a:.4f}   {b:.4f}   {b - a:+.4f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.0))

ax1.plot(PS, s_fisher, color="#1f77b4", lw=2.2, label="피셔의 정확검정")
ax1.plot(PS, s_raised, color="#d62728", lw=2.2,
         label=f"문턱을 ${GAMMA:.4f}$ 로 올린 것")
ax1.axhline(ALPHA, color="#333333", lw=1.8, ls="--", label="약속한 $0.05$")
ax1.set_xlim(0.02, 0.98)
ax1.set_ylim(0, 0.062)
ax1.set_xlabel("참 비율 $p$ (두 집단이 같다)")
ax1.set_ylabel("실제 기각률")
ax1.set_title(f"① 약속한 $0.05$ 를 실제로 쓰는가 ($n_1 = n_2 = {N1}$)", fontsize=12)
ax1.legend(loc="center", fontsize=10)
ax1.grid(alpha=0.25)
ax1.text(
    0.50, 0.0560,
    f"평균 기각률  피셔 ${s_fisher.mean():.4f}$   올린 뒤 ${s_raised.mean():.4f}$",
    ha="center", fontsize=10.5,
    bbox=dict(boxstyle="round,pad=0.4", fc="#fff8e7", ec="#cc9900", lw=1.0),
)

p2s = np.linspace(0.02, 0.60, 120)
for p1, col in [(0.7, "#1f77b4"), (0.9, "#2ca02c")]:
    ax2.plot(p2s, [power(ALPHA, p1, q) for q in p2s], color=col, lw=2.2, ls="--")
    ax2.plot(p2s, [power(GAMMA, p1, q) for q in p2s], color=col, lw=2.4,
             label=f"$p_1 = {p1}$")
ax2.set_xlim(p2s[0], p2s[-1])
ax2.set_ylim(0, 1.02)
ax2.set_xlabel("다른 집단의 비율 $p_2$")
ax2.set_ylabel("기각할 확률")
ax2.set_title("② 파선이 피셔, 실선이 문턱을 올린 것", fontsize=12)
ax2.legend(loc="upper right", fontsize=10.5)
ax2.grid(alpha=0.25)

fig.tight_layout()
out = "docs/ch10/practice/img/barnard_size_power.png"
fig.savefig(out, dpi=150, facecolor="white", bbox_inches="tight")
print("\nsaved", out)
