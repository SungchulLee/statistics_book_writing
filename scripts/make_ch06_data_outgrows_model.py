r"""자료가 모형을 앞지르는 지점 — 틀린 모수적 적합이 지는 표본크기.

이 쪽은 모형이 **옳을 때** 모수적 적합이 비모수적 경험 PMF 를 이긴다는 것을
$n = 200$ 에서 쟀다. 모형이 **살짝 틀렸을 때**는 그 승부가 $n$ 에 달려 있다.
모수적 쪽의 오차는 편향에서 바닥을 치고 멈추지만 경험 PMF 는 $1/\sqrt n$ 로
계속 내려가므로, 어딘가에서 반드시 뒤집힌다.

① 참분포가 음이항(과산포)일 때 $n$ 을 키우며 둘을 잰다 — 교차점이 생긴다.
② 모형이 더 틀릴수록 교차점이 앞당겨진다.
③ 추월하는 쪽은 가정을 버린 경험 PMF 가 아니라 가정을 넓힌 혼합이다.

만드는 파일: docs/ch06/mle/img/data_outgrows_model.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from scipy import stats, special   # noqa: E402

RNG = np.random.default_rng(0)
MU = 4.5                      # 이 쪽 보기 1 과 같은 평균
KGRID = np.arange(0, 31)      # PMF 를 재는 격자


def nb_params(mu, vm):
    """평균 mu, 분산/평균 vm 인 음이항의 numpy 꼴 (n, p)."""
    p = 1.0 / vm
    return mu * p / (1.0 - p), p


def true_pmf(vm):
    r, p = nb_params(MU, vm)
    return stats.nbinom.pmf(KGRID, r, p)


def draw(n, vm, rng):
    r, p = nb_params(MU, vm)
    return rng.negative_binomial(r, p, n)


def pmf_parametric(x):
    return stats.poisson.pmf(KGRID, x.mean())


def pmf_empirical(x):
    c = np.bincount(x, minlength=len(KGRID))[:len(KGRID)]
    return c / len(x)


def rmse(fit, true):
    return float(np.sqrt(np.mean((fit - true) ** 2)))


def em_mixture(x, K, iters=120):
    """포아송 K 성분 혼합의 EM. 같은 값끼리 묶어 돌린다."""
    if K == 1:
        return pmf_parametric(x)
    v, c = np.unique(x, return_counts=True)
    xb = x.mean()
    lam = xb * np.linspace(0.35, 1.9, K)
    w = np.full(K, 1.0 / K)
    for _ in range(iters):
        lp = (v[:, None] * np.log(lam) - lam
              - special.gammaln(v + 1.0)[:, None] + np.log(w))
        lp -= lp.max(axis=1, keepdims=True)
        r = np.exp(lp)
        r /= r.sum(axis=1, keepdims=True)
        rc = r * c[:, None]
        nk = rc.sum(axis=0)
        w = np.maximum(nk / nk.sum(), 1e-6)
        w /= w.sum()
        lam = np.maximum((rc * v[:, None]).sum(axis=0) / np.maximum(nk, 1e-9),
                         1e-3)
    return (w * stats.poisson.pmf(KGRID[:, None], lam)).sum(axis=1)


def reps_for(n):
    return 400 if n <= 1200 else (120 if n <= 25000 else 40)


def crossing(ns, par, emp):
    """경험 PMF 가 모수적 적합을 따라잡는 n (로그-로그 선형보간)."""
    d = np.log(np.array(emp)) - np.log(np.array(par))
    for i in range(len(d) - 1):
        if d[i] > 0 >= d[i + 1]:
            t = d[i] / (d[i] - d[i + 1])
            return float(np.exp(np.log(ns[i]) + t * (np.log(ns[i + 1])
                                                     - np.log(ns[i]))))
    return float("nan")


# === ①② 교차점 — 두 판이 같은 격자를 쓴다 ============================
VMS = np.array([1.1, 1.2, 1.35, 1.5, 1.8, 2.2, 2.8])
VM1 = 1.5                     # ①에 그리는 것
NS = np.array([20, 35, 60, 100, 180, 320, 600, 1200, 3000, 8000, 25000,
               80000, 300000])
print(f"①② 참분포 = 음이항 (평균 {MU}), 모수적 적합은 포아송 — 곧 틀린 모형")
CURVES, NSTARS, BIASES = {}, [], []
for vm in VMS:
    T = true_pmf(vm)
    par, emp, win = [], [], []
    for n in NS:
        R = reps_for(n)
        a = np.empty(R)
        b = np.empty(R)
        for i in range(R):
            x = draw(n, vm, RNG)
            a[i] = rmse(pmf_parametric(x), T)
            b[i] = rmse(pmf_empirical(x), T)
        par.append(a.mean())
        emp.append(b.mean())
        win.append(float((a < b).mean()))
    CURVES[vm] = (par, emp, win)
    NSTARS.append(crossing(NS, par, emp))
    BIASES.append(rmse(stats.poisson.pmf(KGRID, MU), T))

PAR1, EMP1, WIN1 = CURVES[VM1]
BIAS = rmse(stats.poisson.pmf(KGRID, MU), true_pmf(VM1))
NSTAR1 = NSTARS[list(VMS).index(VM1)]
print(f"\n① 분산/평균 = {VM1} 일 때")
print(f"{'n':>8}{'모수적 RMSE':>14}{'경험 PMF RMSE':>16}{'모수적이 이긴 비율':>20}")
for n, a, b, w in zip(NS, PAR1, EMP1, WIN1):
    print(f"{n:>8,}{a:>14.5f}{b:>16.5f}{w:>20.3f}")
print(f"   포아송 쪽의 편향 바닥 (n -> 무한) = {BIAS:.5f}")
print(f"   교차점 n* = {NSTAR1:,.0f}")

print(f"\n② 분산/평균을 바꾸며 교차점을 다시 잰다")
print(f"{'분산/평균':>9}{'편향 바닥':>12}{'교차점 n*':>14}{'n* x 편향^2':>14}")
for vm, bi, ns in zip(VMS, BIASES, NSTARS):
    print(f"{vm:>9.2f}{bi:>12.5f}{ns:>14,.0f}{ns * bi ** 2:>14.4f}")
print("   모형이 참에 가까울수록 교차점이 멀어진다 — 틀림이 작으면 "
      "그 틀림을 자료가 이기는 데 더 오래 걸린다.")
PROD = np.array(NSTARS) * np.array(BIASES) ** 2
print(f"   편향이 {BIASES[-1] / BIASES[0]:.1f} 배로 벌어지는 동안 n* x 편향^2 은 "
      f"{PROD.min():.4f}~{PROD.max():.4f} 에 머문다 — n* 는 편향의 -2 승이다.")

# === ③ 자료의 양이 쓸 수 있는 복잡도를 정한다 =========================
KS = [1, 2, 3, 4, 6]
N3 = [200, 2000, 50000]
VM3 = 1.5
T3 = true_pmf(VM3)
print(f"\n③ 같은 참분포(분산/평균 {VM3})에서 성분 수 K 를 늘리며")
print(f"{'n':>8}" + "".join(f"{f'K={k}':>10}" for k in KS) + f"{'경험 PMF':>11}"
      + f"{'가장 나은 쪽':>14}")
M3 = {}
for n in N3:
    R = 200 if n <= 2000 else 50
    acc = np.zeros(len(KS))
    emp = 0.0
    for i in range(R):
        x = draw(n, VM3, RNG)
        for j, k in enumerate(KS):
            acc[j] += rmse(em_mixture(x, k), T3)
        emp += rmse(pmf_empirical(x), T3)
    acc /= R
    emp /= R
    M3[n] = (acc, emp)
    allv = list(acc) + [emp]
    best = (KS + ["경험 PMF"])[int(np.argmin(allv))]
    print(f"{n:>8,}" + "".join(f"{v:>10.5f}" for v in acc) + f"{emp:>11.5f}"
          + f"{str(best):>14}")
print("   추월하는 쪽은 「가정을 버린」 경험 PMF 가 아니라 「가정을 넓힌」 혼합이다 — "
      "경험 PMF 는 어느 n 에서도 최선이 아니고, 자료가 늘수록 쓸모 있는 K 가 커진다.")

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.6, 5.2))
C_PAR, C_EMP, C_MIX = "#1f77b4", "#d62728", "#2ca02c"

ax1.loglog(NS, PAR1, "o-", color=C_PAR, lw=2.6, ms=8,
           label="모수적 (포아송 MLE)")
ax1.loglog(NS, EMP1, "s-", color=C_EMP, lw=2.6, ms=8, label="비모수적 (경험 PMF)")
ax1.axhline(BIAS, color=C_PAR, lw=1.3, ls="--", alpha=0.8)
ax1.text(4000, BIAS * 1.22, "편향이 만든 바닥", color=C_PAR,
         fontsize=10.5)
ax1.axvline(NSTAR1, color="#888888", lw=1.4, ls=":")
ax1.annotate(f"$n^* \\approx {NSTAR1:,.0f}$".replace(",", "{,}"),
             xy=(NSTAR1, EMP1[0] * 0.55), fontsize=11.5, color="#555555",
             ha="center")
ax1.set_xlabel("표본크기 $n$")
ax1.set_ylabel("참 PMF 까지의 RMSE")
ax1.set_xticks([1e2, 1e3, 1e4, 1e5])
ax1.set_xticklabels(["$10^{2}$", "$10^{3}$", "$10^{4}$", "$10^{5}$"])
ax1.set_yticks([1e-3, 1e-2])
ax1.set_yticklabels(["$10^{-3}$", "$10^{-2}$"])
ax1.minorticks_off()
ax1.set_title("① 틀린 모형은 어디선가 추월당한다", fontsize=12)
ax1.legend(loc="lower left", fontsize=10.5)
ax1.grid(alpha=0.25)

ax2.loglog(VMS - 1.0, NSTARS, "o-", color=C_MIX, lw=2.6, ms=9)
for vm, ns in zip(VMS, NSTARS):
    ax2.annotate(f"${ns:,.0f}$".replace(",", "{,}"), (vm - 1.0, ns),
                 xytext=(0.5, 0.45), textcoords="offset fontsize",
                 fontsize=9.5, color="#555555")
ax2.set_xlabel("모형이 틀린 정도 (참분포의 분산/평균 $-\\ 1$)")
ax2.set_ylabel("교차점 $n^*$")
ax2.set_xticks([0.1, 0.2, 0.5, 1.0, 2.0])
ax2.set_xticklabels(["$0.1$", "$0.2$", "$0.5$", "$1$", "$2$"])
ax2.set_yticks([1e2, 1e3, 1e4, 1e5])
ax2.set_yticklabels(["$10^{2}$", "$10^{3}$", "$10^{4}$", "$10^{5}$"])
ax2.minorticks_off()
ax2.set_xlim(0.075, 3.2)
ax2.set_title("② 조금 틀릴수록 더 오래 버틴다", fontsize=12)
ax2.grid(alpha=0.25)

xs = np.arange(len(KS) + 1)
wd = 0.26
COLN = ["#9ecae1", "#4292c6", "#08519c"]
for t, n in enumerate(N3):
    acc, emp = M3[n]
    ax3.bar(xs + (t - 1) * wd, list(acc) + [emp], wd, color=COLN[t],
            label=f"$n = {n:,}$".replace(",", "{,}"))
    allv = list(acc) + [emp]
    j = int(np.argmin(allv))
    ax3.plot(j + (t - 1) * wd, allv[j] * 1.32, "v", color=COLN[t], ms=10)
ax3.set_yscale("log")
ax3.set_xticks(xs)
ax3.set_xticklabels([f"$K={k}$" for k in KS] + ["경험\nPMF"], fontsize=10)
ax3.set_xlabel("가정한 포아송 성분의 수 (오른쪽 끝은 아무 가정도 없다)")
ax3.set_ylabel("참 PMF 까지의 RMSE")
ax3.set_yticks([1e-4, 1e-3, 1e-2])
ax3.set_yticklabels(["$10^{-4}$", "$10^{-3}$", "$10^{-2}$"])
ax3.minorticks_off()
ax3.set_ylim(8e-5, 2.4e-2)
ax3.set_title("③ 이기는 쪽은 가정을 넓힌 모형이다 (▼ 가 최선)", fontsize=12)
ax3.legend(loc="lower left", fontsize=10)
ax3.grid(alpha=0.25, axis="y")

fig.tight_layout()
out = "docs/ch06/mle/img/data_outgrows_model.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
