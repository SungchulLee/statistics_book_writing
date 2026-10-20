r"""분산을 **검정하는** 대신 **예측하는** 쪽으로.

고전적 처방은 등분산을 가정하고 그것을 검정한 뒤, 깨지면 로버스트 표준오차나
가중최소제곱으로 간다. 요즘 모형은 $\mu(x)$ 와 함께 $\sigma(x)$ 를 **출력**하고
가우스 음의 로그가능도를 최소화한다 — 그 최소화가 곧 가중최소제곱이다.

① 자료와 두 모형의 $\pm 2\sigma$ 띠.
② $x$ 구간별 $95\%$ 구간의 포함률.
③ 음의 로그가능도가 각 점에 주는 유효 가중치 $1/\sigma^2(x)$.

만드는 파일: docs/ch15/applications/img/learned_variance.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from scipy import stats  # noqa: E402

RNG = np.random.default_rng(20261020)
N = 400
DEG = 3


def true_mu(x):
    return 2.0 + 1.5 * x - 0.6 * x**2


def true_sd(x):
    """x 가 커질수록 흩어짐이 커진다."""
    return np.exp(-1.2 + 0.9 * x)


def sample(n, rng):
    x = rng.uniform(0, 3, n)
    return x, true_mu(x) + true_sd(x) * rng.standard_normal(n)


def design(x, deg=DEG):
    return np.vander(x, deg + 1, increasing=True)


def fit_homo(x, y):
    """등분산을 가정한 보통 최소제곱."""
    X = design(x)
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    r = y - X @ b
    s2 = r @ r / (len(x) - X.shape[1])
    return b, lambda t: np.full_like(t, np.sqrt(s2))


def fit_hetero(x, y, iters=30):
    """가우스 음의 로그가능도를 최소화한다 — mu 는 가중최소제곱,
    log sigma^2 은 잔차제곱의 로그에 대한 회귀로 번갈아 갱신한다."""
    X = design(x)
    Z = np.column_stack([np.ones_like(x), x])  # log sigma^2 = Z g
    g = np.array([0.0, 0.0])
    for _ in range(iters):
        w = np.exp(-(Z @ g))  # 1 / sigma^2
        XtW = X.T * w
        b = np.linalg.solve(XtW @ X, XtW @ y)
        r2 = (y - X @ b) ** 2
        # r^2/sigma^2 ~ chi^2_1 이므로 E[log r^2] = log sigma^2 + E[log chi^2_1]
        # 이고 E[log chi^2_1] = psi(1/2) + log 2 = -1.27036 이다. 그만큼 되돌린다.
        g = np.linalg.lstsq(Z, np.log(r2 + 1e-12) + 1.2703628455, rcond=None)[0]
    return b, (lambda t: np.exp(0.5 * (np.column_stack([np.ones_like(t), t]) @ g))), g


x, y = sample(N, RNG)
b_ho, sd_ho = fit_homo(x, y)
b_he, sd_he, g_he = fit_hetero(x, y)

# === ② 구간별 포함률 ===============================================
EDGES = np.linspace(0, 3, 7)
cov_ho, cov_he = [], []
for _ in range(600):
    xs, ys = sample(N, RNG)
    bh, sh = fit_homo(xs, ys)
    be, se, _ = fit_hetero(xs, ys)
    for arr, bb, ss in ((cov_ho, bh, sh), (cov_he, be, se)):
        mu = design(xs) @ bb
        hit = np.abs(ys - mu) <= 1.959964 * ss(xs)
        idx = np.clip(np.digitize(xs, EDGES) - 1, 0, len(EDGES) - 2)
        arr.append([hit[idx == k].mean() for k in range(len(EDGES) - 1)])
cov_ho = np.array(cov_ho).mean(axis=0)
cov_he = np.array(cov_he).mean(axis=0)
mids = (EDGES[:-1] + EDGES[1:]) / 2

# === 그림 ==========================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.2, 4.9))

t = np.linspace(0, 3, 300)
ax1.scatter(x, y, s=12, color="#9e9e9e", alpha=0.55, edgecolors="none")
ax1.plot(t, true_mu(t), color="black", lw=1.6, ls="--", label="참 평균")
for mu_b, sd_f, c, lab in (
    (b_ho, sd_ho, "#d62728", "등분산을 가정"),
    (b_he, sd_he, "#1f77b4", "분산도 함께 배움"),
):
    m = design(t) @ mu_b
    s = sd_f(t)
    ax1.plot(t, m + 2 * s, color=c, lw=2.0, label=f"{lab} ($\\pm 2\\sigma$)")
    ax1.plot(t, m - 2 * s, color=c, lw=2.0)
ax1.set_xlabel("$x$")
ax1.set_ylabel("$y$")
ax1.set_title("① 같은 자료, 두 가지 띠", fontsize=12)
ax1.legend(loc="lower left", fontsize=9.5)
ax1.grid(alpha=0.25)

w = 0.34
xs_ = np.arange(len(mids))
ax2.bar(xs_ - w / 2, cov_ho * 100, w, color="#d62728", label="등분산을 가정")
ax2.bar(xs_ + w / 2, cov_he * 100, w, color="#1f77b4", label="분산도 함께 배움")
ax2.axhline(95, color="black", lw=1.4, ls="--")
ax2.text(2.5, 101.6, "명목 $95\\%$", ha="center", fontsize=10)
for i, (a, b) in enumerate(zip(cov_ho, cov_he)):
    ax2.text(i - w / 2, a * 100 + 0.7, f"{a * 100:.0f}", ha="center", fontsize=9)
    ax2.text(i + w / 2, b * 100 + 0.7, f"{b * 100:.0f}", ha="center", fontsize=9)
ax2.set_xticks(xs_)
ax2.set_xticklabels([f"{e:.1f}–{f:.1f}" for e, f in zip(EDGES[:-1], EDGES[1:])],
                    fontsize=9)
ax2.set_xlabel("$x$ 구간")
ax2.set_ylabel("$95\\%$ 구간의 포함률 ($\\%$)")
ax2.set_title("② 평균은 맞는데 어디서도 안 맞는다", fontsize=12)
ax2.set_ylim(60, 103)
ax2.legend(loc="lower left", fontsize=9.5)
ax2.grid(alpha=0.25, axis="y")

wgt = 1.0 / sd_he(t) ** 2
ax3.semilogy(t, wgt / wgt.max(), color="#1f77b4", lw=2.6, label="$1/\\sigma^2(x)$ (상대)")
ax3.axhline(1.0, color="#d62728", lw=2.0, ls="--", label="등분산이 주는 가중치")
ax3.set_xlabel("$x$")
ax3.set_ylabel("한 점이 받는 상대 가중치")
ax3.set_title("③ 최소화가 곧 가중최소제곱이다", fontsize=12)
ax3.set_yticks([1e-3, 1e-2, 1e-1, 1])
ax3.set_yticklabels(["$10^{-3}$", "$10^{-2}$", "$10^{-1}$", "$1$"])
ax3.minorticks_off()
ax3.legend(loc="lower left", fontsize=10)
ax3.grid(alpha=0.25, which="major")
r = wgt.max() / wgt.min()
ax3.annotate(
    f"양 끝의 가중치 비 ${r:,.0f}$ 배".replace(",", "{,}"),
    xy=(3.0, wgt.min() / wgt.max()),
    xytext=(1.4, 3e-3),
    fontsize=10,
    arrowprops=dict(arrowstyle="->", color="#555555", lw=1.0),
)

fig.tight_layout()
out = "docs/ch15/applications/img/learned_variance.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)

# === 본문에 적을 수 =================================================
print(f"\n참 흩어짐: x=0 에서 {true_sd(0.0):.3f}, x=3 에서 {true_sd(3.0):.3f} "
      f"({true_sd(3.0) / true_sd(0.0):.1f} 배)")
print(f"배운 흩어짐: x=0 에서 {sd_he(np.array([0.0]))[0]:.3f}, "
      f"x=3 에서 {sd_he(np.array([3.0]))[0]:.3f}")
print(f"등분산 가정이 쓰는 하나의 값: {sd_ho(np.array([0.0]))[0]:.3f}")
print(f"\n{'x 구간':>10}{'등분산 가정':>12}{'분산도 배움':>12}")
for (e, f), a, bb in zip(zip(EDGES[:-1], EDGES[1:]), cov_ho, cov_he):
    print(f"{f'{e:.1f}–{f:.1f}':>10}{a * 100:>11.1f}%{bb * 100:>11.1f}%")
print(f"{'전체':>10}{cov_ho.mean() * 100:>11.1f}%{cov_he.mean() * 100:>11.1f}%")
print(f"\n③ 유효 가중치의 양 끝 비: {r:,.0f} 배  (= (sigma(3)/sigma(0))^2)")
print(f"   참값으로는 {(true_sd(3.0) / true_sd(0.0)) ** 2:,.0f} 배")
