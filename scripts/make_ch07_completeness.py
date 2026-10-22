r"""완비가 아니면 무엇이 남는가 — $\mathcal{U}(\theta, \theta+1)$ 의 경우.

$X_1,\dots,X_n \sim \mathcal{U}(\theta, \theta+1)$ 에서 $T = (X_{(1)}, X_{(n)})$ 은
충분하지만 **완비가 아니다.** 범위 $R = X_{(n)} - X_{(1)}$ 의 기댓값이 $\theta$ 와
무관한 $(n-1)/(n+1)$ 이므로

    g(T) = R - (n-1)/(n+1)

은 모든 $\theta$ 에서 기댓값이 $0$ 인데 항등적으로 $0$ 이 아니다.

① 그래서 불편추정량이 한 줄로 늘어선다 —
   $\hat\theta_a = a\,(X_{(1)} - \tfrac{1}{n+1}) + (1-a)\,(X_{(n)} - \tfrac{n}{n+1})$
   는 $a$ 가 무엇이든 불편이고, 분산만 달라진다.
② $g(T)$ 의 분포. $\theta$ 를 바꾸어도 같은 자리에 있고 평균이 $0$ 이다.

만드는 파일: docs/ch07/theory/img/completeness_fail.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

RNG = np.random.default_rng(19500101)
REP = 200_000
NS = [5, 20, 100]
AS = np.linspace(0, 1, 101)


def draw(n, theta, rep=REP):
    x = theta + RNG.random((rep, n))
    return x.min(1), x.max(1)


def var_theory(a, n):
    """Var(theta_hat_a) — 순서통계량의 분산과 공분산으로 적는다."""
    c = 1.0 / ((n + 1) ** 2 * (n + 2))
    V, C = n * c, c                      # Var(X_(1)) = Var(X_(n)) = nc,  Cov = c
    return a**2 * V + (1 - a) ** 2 * V + 2 * a * (1 - a) * C


print("① 모든 a 에서 불편인가, 그리고 분산은 어떻게 달라지는가 (theta = 3)")
print("    n       a     편향(모의)     분산(모의)      분산(식)")
sim = {}
for n in NS:
    lo, hi = draw(n, 3.0)
    est = {a: a * (lo - 1 / (n + 1)) + (1 - a) * (hi - n / (n + 1)) for a in AS}
    sim[n] = np.array([est[a].var(ddof=1) for a in AS])
    for a in [0.0, 0.25, 0.5, 1.0]:
        e = est[a]
        print(f"  {n:>3}   {a:5.2f}   {e.mean()-3.0:+.6f}   {e.var(ddof=1):.8f}   "
              f"{var_theory(a, n):.8f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.0))

# === ① 불편추정량이 한 줄로 =========================================
COL = ["#1f77b4", "#2ca02c", "#d9a441"]
for n, c in zip(NS, COL):
    v = np.array([var_theory(a, n) for a in AS])
    ax1.plot(AS, v / v.min(), color=c, lw=2.4, label=f"$n = {n}$")
    ax1.plot(sim[n] / v.min(), ".", color=c, ms=3, alpha=0.0)  # 그리지는 않는다
ax1.axvline(0.5, color="#d62728", lw=1.2, ls=":")
ax1.set_xlim(0, 1)
ax1.set_ylim(0.95, 2.1)
ax1.set_xlabel("$a$ — 아래끝에 얼마나 기대는가")
ax1.set_ylabel("분산 $\\div$ 가장 작은 분산")
ax1.set_title("① 이 선 위의 점이 모두 불편추정량이다", fontsize=12)
ax1.legend(loc="upper center", fontsize=10)
ax1.grid(alpha=0.25)
for a, lab, dy in [(0.0, "$X_{(n)}$ 만 쓴다", 0.06),
                   (0.5, "중앙범위", 0.06),
                   (1.0, "$X_{(1)}$ 만 쓴다", 0.06)]:
    v = var_theory(a, NS[1]) / np.array([var_theory(t, NS[1]) for t in AS]).min()
    ax1.plot([a], [v], "o", color="#d62728", ms=8, zorder=5)
    ax1.annotate(lab, xy=(a, v), xytext=(a if a < 0.9 else a - 0.03, v + dy),
                 fontsize=10.5, color="#d62728",
                 ha="left" if a < 0.5 else ("center" if a == 0.5 else "right"))

# === ② 0 을 불편추정하는, 0 이 아닌 함수 ==============================
n2 = 20
gs = {}
for theta, col, lab in [(0.0, "#1f77b4", "$\\theta = 0$"),
                        (5.0, "#d9a441", "$\\theta = 5$"),
                        (-100.0, None, None)]:
    lo, hi = draw(n2, theta)
    gs[theta] = (hi - lo) - (n2 - 1) / (n2 + 1)
for theta, col, lab in [(0.0, "#1f77b4", "$\\theta = 0$"),
                        (5.0, "#d9a441", "$\\theta = 5$")]:
    g = gs[theta]
    ax2.hist(g, bins=np.linspace(-0.6, 0.15, 90), density=True, histtype="step",
             lw=2.3, color=col, label=f"{lab}   평균 ${g.mean():+.5f}$")
ax2.axvline(0, color="#d62728", lw=1.6, ls="--")
ax2.set_xlim(-0.6, 0.15)
ax2.set_xlabel(f"$g(T) = R - \\dfrac{{n-1}}{{n+1}}$   ($n = {n2}$)")
ax2.set_ylabel("밀도")
ax2.set_title("② $\\theta$ 가 무엇이든 평균이 $0$ 인데, $0$ 이 아니다", fontsize=12)
ax2.legend(loc="upper left", fontsize=10.5)
ax2.grid(alpha=0.25)

fig.tight_layout()
out = "docs/ch07/theory/img/completeness_fail.png"
fig.savefig(out, dpi=150, facecolor="white", bbox_inches="tight")
print("\nsaved", out)

# === 본문에 적을 수 =================================================
print("\n② g(T) 의 평균 — theta 를 바꾸어 가며 (n = 20, 그림과 같은 표본)")
for theta in [0.0, 5.0, -100.0]:
    g = gs[theta]
    print(f"   theta = {theta:>7.1f}:  평균 {g.mean():+.6f}   표준편차 {g.std(ddof=1):.6f}")

print("\n③ 끝점 하나만 쓰는 것과 중앙범위의 분산비")
print("     n     끝점 분산      중앙범위 분산      비")
for n in [5, 20, 100, 1000]:
    v0, v5 = var_theory(0.0, n), var_theory(0.5, n)
    print(f"  {n:>4}   {v0:.8f}   {v5:.8f}   {v0/v5:.4f}")
print("   비는 2n/(n+1) 이므로 n 이 커지면 2 로 간다.")
