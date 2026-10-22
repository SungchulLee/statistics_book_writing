r"""버크슨이 $1938$ 년에 적은 $200{,}000$ 은 어느 크기의 이탈을 가리키는가.

① 검정력이 $1/2$ 이 되는 초과첨도 $\gamma_2^\ast$ 를 $n$ 마다 재면 $n^{-1/2}$ 로 줄어든다.
   그 곡선 위에서 $n = 200{,}000$ 을 읽는다.
② 그 임계 크기의 표본을 하나 그려 본다 — 섞여 든 점 몇 개가 기각을 만드는가.

오염정규 $(1-\varepsilon)\,N(0,1) + \varepsilon\,N(0,3^2)$ 를 대립으로 쓴다.
검정은 Jarque-Bera ($n$ 이 수십만이라 샤피로-윌크는 쓸 수 없다).

만드는 파일: docs/ch14/limitations/img/berkson_200000.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from matplotlib.ticker import FuncFormatter  # noqa: E402
from scipy import stats  # noqa: E402
from scipy.optimize import brentq  # noqa: E402

RNG = np.random.default_rng(20261022)
SIG = 3.0  # 오염 성분의 표준편차
NS = [100, 300, 1_000, 3_000, 10_000, 30_000, 100_000, 200_000]
CHI2_CRIT = stats.chi2.ppf(0.95, 2)
# gamma2(eps) 는 eps = 0.1 에서 최대 5.33 을 찍고 되돌아 내려온다.
# 따라서 역함수는 eps < 0.1 쪽 가지에서만 구한다.
G2_MAX = 4.45


def sci(v):
    """mathtext 로 안전한 과학 표기."""
    m, e = f"{v:.1e}".split("e")
    return f"{m} \\times 10^{{{int(e)}}}"


def gamma2(eps):
    """오염정규의 초과첨도 (정확한 값)."""
    m2 = 1 - eps + eps * SIG**2
    m4 = 3 * (1 - eps + eps * SIG**4)
    return m4 / m2**2 - 3


def eps_for(g2):
    """주어진 초과첨도를 내는 오염 비율."""
    return brentq(lambda e: gamma2(e) - g2, 1e-15, 0.1)


def jb(x, axis=-1):
    """Jarque-Bera 통계량. x 는 (rep, n) 꼴."""
    z = x - x.mean(axis=axis, keepdims=True)
    m2 = (z**2).mean(axis=axis)
    g1 = (z**3).mean(axis=axis) / m2**1.5
    g2 = (z**4).mean(axis=axis) / m2**2 - 3
    n = x.shape[axis]
    return n / 6.0 * (g1**2 + g2**2 / 4.0)


def power(n, eps, rep, chunk_cells=4_000_000):
    """오염정규에서 뽑은 표본을 JB 가 5%에서 기각하는 비율."""
    hits = 0
    done = 0
    step = max(1, chunk_cells // n)
    while done < rep:
        k = min(step, rep - done)
        x = RNG.standard_normal((k, n))
        x[RNG.random((k, n)) < eps] *= SIG
        hits += int((jb(x) > CHI2_CRIT).sum())
        done += k
    return hits / rep


# === ① 검정력이 1/2 이 되는 초과첨도 =================================
# 안내선: (n/24) g2^2 가 카이제곱 임계값 근처가 되는 자리 → g2 ~ 10/sqrt(n)
rows = []
for n in NS:
    rep = 3000 if n <= 1_000 else (1500 if n <= 30_000 else 600)
    guide = 10.4 / np.sqrt(n)
    grid = np.array([0.4, 0.6, 0.8, 1.0, 1.3, 1.7, 2.3, 3.2, 4.2]) * guide
    grid = grid[grid < G2_MAX]
    ps = np.array([power(n, eps_for(g), rep) for g in grid])
    # 검정력이 0.5 를 지나는 자리를 선형보간으로 찾는다
    if ps.max() < 0.5:
        raise RuntimeError(f"n = {n}: 눈금 안에서 검정력 0.5 에 닿지 않는다 ({ps.max():.3f})")
    i = int(np.argmax(ps >= 0.5))
    if i == 0:
        g_star = grid[0]
    else:
        t = (0.5 - ps[i - 1]) / (ps[i] - ps[i - 1])
        g_star = grid[i - 1] + t * (grid[i] - grid[i - 1])
    e_star = eps_for(g_star)
    rows.append((n, g_star, e_star, ps, grid))
    print(f"n = {n:>7}:  gamma2* = {g_star:.5f}   eps* = {e_star:.3e}   "
          f"섞인 점 {n * e_star:.1f} 개")

ns = np.array([r[0] for r in rows], dtype=float)
gs = np.array([r[1] for r in rows])
es = np.array([r[2] for r in rows])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 5.2))

ax1.loglog(ns, gs, "o-", color="#1f77b4", lw=2.2, ms=7, label="모의실험으로 잰 값")
ax1.loglog(ns, gs[-1] * np.sqrt(ns[-1] / ns), "--", color="#999999", lw=1.4,
           label="$1/\\sqrt{n}$ 에 비례하는 선 (견줌)")
ax1.axvline(200_000, color="#d62728", lw=1.3, ls=":")
ax1.plot([200_000], [gs[-1]], marker="*", color="#d62728", ms=18, zorder=5)
ax1.annotate(
    f"$n = 200{{,}}000$ 에서\n$\\gamma_2^\\ast = {gs[-1]:.3f}$",
    xy=(200_000, gs[-1]),
    xytext=(1_500, 0.022),
    fontsize=11.5,
    color="#d62728",
    arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.2),
)
ax1.set_xlabel("표본 크기 $n$")
ax1.set_ylabel("검정력이 절반이 되는 초과첨도 $\\gamma_2^\\ast$")
ax1.set_title("① 잡아낼 수 있는 가장 작은 이탈", fontsize=12)
ax1.legend(loc="lower left", fontsize=10)
ax1.grid(alpha=0.25, which="both")
ax1.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
ax1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
ax1.set_xticks([100, 1_000, 10_000, 100_000])
ax1.set_yticks([0.01, 0.1, 1.0])
ax1.minorticks_off()

# === ② 「천 개에 하나」짜리 표본 하나 ================================
# 임계 크기(26 개)보다 넉넉한, 말로 하기 쉬운 크기를 고른다: 천 개에 하나.
N2 = 200_000
EPS2 = 0.001
print(f"\n   견줌 — eps = {EPS2} (초과첨도 {gamma2(EPS2):.4f}) 에서의 검정력")
for n in [1_000, 10_000, 200_000]:
    print(f"     n = {n:>7}:  {power(n, EPS2, 2000 if n < 100_000 else 600):.3f}")
x = RNG.standard_normal(N2)
odd = RNG.random(N2) < EPS2
x[odd] *= SIG
stat = float(jb(x[None, :])[0])
pval = float(stats.chi2.sf(stat, 2))
n_odd = int(odd.sum())
print(f"\n② n = {N2}, eps = {EPS2:.3e} 로 뽑은 표본 하나")
print(f"   섞여 든 점 {n_odd} 개   JB = {stat:.3f}   p = {pval:.3e}")
print(f"   왜도 {stats.skew(x):+.5f}   초과첨도 {stats.kurtosis(x):+.5f}")
print(f"   같은 표본에서 섞인 점을 빼면: JB = {float(jb(x[~odd][None, :])[0]):.3f}")

order = np.argsort(x)
xs = x[order]
odd_s = odd[order]
pp = (np.arange(1, N2 + 1) - 0.5) / N2
zq = stats.norm.ppf(pp)
sub = np.arange(0, N2, 20)  # 선은 솎아서 그린다
ax2.plot(zq[sub], xs[sub], ".", color="#1f77b4", ms=2.6, label="정규인 점")
ax2.plot(zq[odd_s], xs[odd_s], "o", color="#d62728", ms=5.0, mfc="none", mew=1.3,
         label=f"섞여 든 점 {n_odd} 개")
lo, hi = zq[sub][0] * 1.02, zq[sub][-1] * 1.02
ax2.plot([lo, hi], [lo * x.std(ddof=1) + x.mean(), hi * x.std(ddof=1) + x.mean()],
         color="black", lw=1.1, ls="--", label="정규 직선")
ax2.set_xlabel("정규 분위수")
ax2.set_ylabel("표본 분위수")
ax2.set_title(f"② 천 개에 하나만 다른 자로 잰 표본 — $p = {sci(pval)}$", fontsize=12)
ax2.legend(loc="upper left", fontsize=10)
ax2.grid(alpha=0.25)
ax2.text(
    0.97, 0.06,
    f"$n = 200{{,}}000$\n왜도 ${stats.skew(x):+.4f}$\n"
    f"초과첨도 ${stats.kurtosis(x):+.4f}$",
    transform=ax2.transAxes, ha="right", va="bottom", fontsize=10.5,
    bbox=dict(boxstyle="round,pad=0.4", fc="#fff8e7", ec="#cc9900", lw=1.0),
)

fig.tight_layout()
out = "docs/ch14/limitations/img/berkson_200000.png"
fig.savefig(out, dpi=150, facecolor="white", bbox_inches="tight")
print("\nsaved", out)

# === 본문에 적을 수 =================================================
print("\n표 — n 별 임계 이탈 크기와 「섞인 점 몇 개」")
print("      n     gamma2*       eps*     섞인 점")
for n, g, e, _, _ in rows:
    print(f"{n:>7}   {g:9.5f}   {e:.3e}   {n * e:9.1f}")
