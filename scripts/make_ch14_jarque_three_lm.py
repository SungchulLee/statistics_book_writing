r"""$1980$ 년의 다섯 쪽은 **셋**을 함께 보려 했다 — 그 가운데 하나만 이름을 얻었다.

회귀잔차에 대한 세 LM 검정을 그대로 구현한다.
  정규성  : Jarque-Bera, $\chi^2_2$
  등분산  : $\hat u^2$ 을 $x$ 에 회귀한 $nR^2$, $\chi^2_1$
  계열독립: 시차 $1$ 자기상관의 $n r_1^2$, $\chi^2_1$

① 네 가지 자료에 셋을 모두 걸어 기각률 표를 만든다.
② 그런데 첫째 칸이 셋째 고장에도 반응한다 — 까닭을 잔차의 꼬리에서 본다.

만드는 파일: docs/ch14/descriptive_statistics/img/jarque_three_lm.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from matplotlib.ticker import FuncFormatter  # noqa: E402
from scipy import stats  # noqa: E402

RNG = np.random.default_rng(19800255)
N = 200
REP = 20_000
RHO = 0.6
HET = 1.0  # 이분산 세기: sd(u) = exp(HET * x / 2)
BETA = (1.0, 2.0)

X = np.linspace(-1.5, 1.5, N)  # 설명변수는 고정
XC = X - X.mean()
SXX = (XC**2).sum()


def errors(kind, rep):
    """네 가지 오차항. 모두 분산이 1 근처가 되도록 맞춰 둔다."""
    if kind == "정규":
        return RNG.standard_normal((rep, N))
    if kind == "치우친 오차":  # 중심화한 로그정규 (왜도 큼)
        u = RNG.lognormal(0.0, 0.75, (rep, N))
        return (u - np.exp(0.75**2 / 2)) / np.sqrt(
            (np.exp(0.75**2) - 1) * np.exp(0.75**2))
    if kind == "이분산":  # 척도가 x 에 따라 달라진다
        s = np.exp(HET * X / 2)
        return RNG.standard_normal((rep, N)) * (s / s.mean())
    if kind == "계열상관":  # AR(1)
        e = RNG.standard_normal((rep, N + 200)) * np.sqrt(1 - RHO**2)
        u = np.empty_like(e)
        u[:, 0] = e[:, 0] / np.sqrt(1 - RHO**2)
        for t in range(1, e.shape[1]):
            u[:, t] = RHO * u[:, t - 1] + e[:, t]
        return u[:, 200:]
    raise ValueError(kind)


def residuals(u):
    """y = b0 + b1 x + u 에 OLS 를 맞추고 잔차를 돌려준다."""
    y = BETA[0] + BETA[1] * X + u
    b1 = (y * XC).sum(1) / SXX
    b0 = y.mean(1)
    return y - (b0[:, None] + b1[:, None] * X)


def lm_normal(r):
    """Jarque-Bera. chi2_2."""
    m2 = (r**2).mean(1)
    g1 = (r**3).mean(1) / m2**1.5
    g2 = (r**4).mean(1) / m2**2 - 3
    return N / 6.0 * (g1**2 + g2**2 / 4.0)


def lm_hetero(r):
    """u^2 을 x 에 회귀한 n R^2. chi2_1."""
    s = r**2
    sc = s - s.mean(1, keepdims=True)
    r2 = ((sc * XC).sum(1) ** 2) / (SXX * (sc**2).sum(1))
    return N * r2


def lm_serial(r):
    """시차 1 자기상관의 n r_1^2. chi2_1."""
    num = (r[:, 1:] * r[:, :-1]).sum(1)
    den = (r**2).sum(1)
    return N * (num / den) ** 2


KINDS = ["정규", "치우친 오차", "이분산", "계열상관"]
TESTS = [
    ("정규성\n$JB \\sim \\chi^2_2$", lm_normal, stats.chi2.ppf(0.95, 2)),
    ("등분산\n$nR^2 \\sim \\chi^2_1$", lm_hetero, stats.chi2.ppf(0.95, 1)),
    ("계열독립\n$nr_1^2 \\sim \\chi^2_1$", lm_serial, stats.chi2.ppf(0.95, 1)),
]

grid = np.zeros((len(KINDS), len(TESTS)))
joint = {}
print(f"① n = {N}, 반복 {REP:,}, 유의수준 0.05 에서의 기각률")
print(f"{'자료':<12}{'정규성':>10}{'등분산':>10}{'계열독립':>10}")
for i, kind in enumerate(KINDS):
    r = residuals(errors(kind, REP))
    stats_i = []
    for j, (_, fn, c) in enumerate(TESTS):
        v = fn(r)
        stats_i.append(v)
        grid[i, j] = float((v > c).mean())
    joint[kind] = stats_i[0] + stats_i[1] + stats_i[2]
    print(f"{kind:<12}{grid[i,0]:>10.4f}{grid[i,1]:>10.4f}{grid[i,2]:>10.4f}")
    del r

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.3),
                               gridspec_kw={"width_ratios": [1.22, 1.0]})

im = ax1.imshow(grid, cmap="YlOrRd", vmin=0.0, vmax=1.0, aspect="auto")
ax1.set_xticks(range(len(TESTS)))
ax1.set_xticklabels([t[0] for t in TESTS], fontsize=10.5)
ax1.set_yticks(range(len(KINDS)))
ax1.set_yticklabels(KINDS, fontsize=11)
for i in range(len(KINDS)):
    for j in range(len(TESTS)):
        ax1.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center",
                 fontsize=13.5,
                 color="white" if grid[i, j] > 0.55 else "#222222")
ax1.set_title("① 어느 검정이 어느 고장에 반응하는가", fontsize=12)
ax1.set_ylabel("자료를 망가뜨린 방식")
fig.colorbar(im, ax=ax1, fraction=0.045, pad=0.03, label="기각률")

# === ② 이분산 오차는 왜 「비정규」로 읽히는가 =========================
# 척도가 섞인 정규는 정규가 아니다: u = s(x)Z 이면 E[u^4]/E[u^2]^2 = 3E[s^4]/E[s^2]^2.
s_scale = np.exp(HET * X / 2)
s_scale = s_scale / s_scale.mean()
g2_exact = 3 * (s_scale**4).mean() / ((s_scale**2).mean() ** 2) - 3
print(f"\n② 이분산 오차의 초과첨도 — 식으로 계산한 값 {g2_exact:.4f}")

pool = {}
for kind in ["정규", "이분산"]:
    u = errors(kind, 4_000)
    r = residuals(u).ravel()
    pool[kind] = r / r.std(ddof=1)
    print(f"   {kind} 잔차의 초과첨도 (모의실험) {stats.kurtosis(pool[kind]):+.4f}")

bins = np.linspace(-6, 6, 160)
ax2.hist(pool["정규"], bins=bins, density=True, histtype="step", lw=2.2,
         color="#1f77b4", label="정규 오차")
ax2.hist(pool["이분산"], bins=bins, density=True, histtype="step", lw=2.2,
         color="#d62728", label="이분산 오차")
ax2.plot(bins, stats.norm.pdf(bins), "k--", lw=1.4, label="$\\mathcal{N}(0,1)$")
ax2.set_yscale("log")
ax2.set_ylim(1e-4, 1.0)
ax2.set_xlim(-6, 6)
ax2.set_xlabel("표준화한 잔차")
ax2.set_ylabel("밀도 (로그 눈금)")
ax2.set_title("② 척도가 섞인 정규는 정규가 아니다", fontsize=12)
ax2.legend(loc="lower center", fontsize=10)
ax2.grid(alpha=0.25, which="both")
# 로그 눈금의 마이너스는 mathtext 라 한글 글꼴에서 두부가 된다
ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
ax2.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
ax2.text(
    0.5, 0.97,
    f"이분산 잔차의 초과첨도 ${stats.kurtosis(pool['이분산']):+.3f}$\n"
    f"(식으로는 ${g2_exact:.3f}$)",
    transform=ax2.transAxes, ha="center", va="top", fontsize=10.5,
    bbox=dict(boxstyle="round,pad=0.4", fc="#fff8e7", ec="#cc9900", lw=1.0),
)

fig.tight_layout()
out = "docs/ch14/descriptive_statistics/img/jarque_three_lm.png"
fig.savefig(out, dpi=150, facecolor="white", bbox_inches="tight")
print("\nsaved", out)

c95 = float(stats.chi2.ppf(0.95, 3))
print(f"\n③ 논문이 제안한 「셋을 더한」 결합검정 (chi2_3, 임계값 {c95:.4f})")
for kind in KINDS:
    v = joint[kind]
    print(f"   {kind:<12} 기각률 {float((v > c95).mean()):.4f}"
          f"   95 백분위 {float(np.quantile(v, 0.95)):.4f}")
