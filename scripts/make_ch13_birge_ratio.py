r"""Welch 의 가중평균 $w_i = n_i/s_i^2$ 는 도량형이 상수값을 정하는 방식 그대로다.

실험실 $k$ 곳이 같은 양을 재어 추정값 $\bar y_i$ 와 불확도 $u_i$ 를 보고한다. 그런데
저마다 **알아채지 못한 계통오차** $b_i \sim N(0, \tau^2)$ 가 있다 — 보고된 $u_i$ 에는
들어 있지 않은 몫이다.

① $\tau = 0$ 이면 역분산 가중평균이 단순평균을 크게 이긴다.
② $\tau$ 가 커지면 뒤집힌다 — 가장 과신한 실험실이 결과를 끌고 간다.
③ 그 징후를 버지 비 $R_B = \sqrt{\chi^2/(k-1)}$ 이 잡고, 구간을 $R_B$ 배 넓히면
   포함률이 돌아온다. CODATA 가 중력상수에 쓴 $3.9$ 가 그것이다.

만드는 파일: docs/ch13/anova_welch/img/birge_ratio.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from scipy import stats  # noqa: E402

RNG = np.random.default_rng(20261021)

K = 16                  # CODATA 2018 이 중력상수에 쓴 측정의 수
UMIN, UMAX = 1.0, 5.0   # 보고된 불확도의 범위 (정밀도가 5 배까지 다르다)
TAUS = np.concatenate([[0.0], np.logspace(-1, 1.1, 24)])
REPS = 20000
Z = stats.norm.isf(0.025)
CODATA_RB = 3.9         # 2018 년 조정에서 중력상수에 적용된 확장인자

# 실험실마다의 보고 불확도는 고정해 둔다 (조건을 같게 하려고)
U = np.exp(RNG.uniform(np.log(UMIN), np.log(UMAX), K))
U.sort()
W = 1.0 / U ** 2
print(f"실험실 {K} 곳의 보고 불확도 {np.round(U, 2)}")
print(f"가장 정밀한 곳이 가장 거친 곳보다 {U[-1] / U[0]:.1f} 배 정밀하다고 말한다")
print(f"가중치가 가장 큰 곳이 전체의 {100 * W[0] / W.sum():.1f}% 를 차지한다")

rows = []
for tau in TAUS:
    b = RNG.normal(0.0, tau, (REPS, K))          # 알아채지 못한 계통오차
    e = RNG.normal(0.0, 1.0, (REPS, K)) * U      # 보고된 불확도만큼의 오차
    y = b + e                                     # 참값은 0
    simple = y.mean(axis=1)
    wmean = (y * W).sum(axis=1) / W.sum()
    se = 1.0 / np.sqrt(W.sum())                   # 「내부」 불확도
    chi2 = (W * (y - wmean[:, None]) ** 2).sum(axis=1)
    rb = np.sqrt(chi2 / (K - 1))                  # 버지 비
    rows.append(dict(
        tau=tau,
        rmse_simple=float(np.sqrt((simple ** 2).mean())),
        rmse_w=float(np.sqrt((wmean ** 2).mean())),
        rb=float(rb.mean()),
        rb_lo=float(np.quantile(rb, 0.025)),
        rb_hi=float(np.quantile(rb, 0.975)),
        cov_naive=float((np.abs(wmean) < Z * se).mean()),
        cov_birge=float((np.abs(wmean) < Z * se * rb).mean()),
    ))

print(f"\n{'τ':>6}{'단순평균':>10}{'가중평균':>10}{'버지 비':>9}"
      f"{'포함률(그대로)':>14}{'포함률(R_B 배)':>15}")
for r in rows:
    if r["tau"] in (0.0,) or np.isclose(r["tau"], TAUS[6]) \
            or np.isclose(r["tau"], TAUS[12]) or np.isclose(r["tau"], TAUS[18]) \
            or r is rows[-1]:
        print(f"{r['tau']:>6.2f}{r['rmse_simple']:>10.3f}{r['rmse_w']:>10.3f}"
              f"{r['rb']:>9.2f}{r['cov_naive']:>14.3f}{r['cov_birge']:>15.3f}")

tau = np.array([r["tau"] for r in rows])
rs = np.array([r["rmse_simple"] for r in rows])
rw = np.array([r["rmse_w"] for r in rows])
rb = np.array([r["rb"] for r in rows])
rblo = np.array([r["rb_lo"] for r in rows])
rbhi = np.array([r["rb_hi"] for r in rows])
cn = np.array([r["cov_naive"] for r in rows])
cb = np.array([r["cov_birge"] for r in rows])

cross = tau[np.argmax(rw > rs)] if (rw > rs).any() else np.nan
print(f"\n가중평균이 단순평균보다 나빠지는 지점: τ ≈ {cross:.2f}")
print(f"  (τ = 0 에서는 {rs[0] / rw[0]:.2f} 배 유리하다)")
tau_codata = float(np.interp(CODATA_RB, rb, tau))
i_cod = int(np.argmin(np.abs(tau - tau_codata)))
print(f"버지 비가 {CODATA_RB} 가 되는 τ ≈ {tau_codata:.2f}  "
      f"(보고 불확도의 중앙값 {np.median(U):.2f} 의 {tau_codata / np.median(U):.1f} 배)")
print(f"  그 자리에서 포함률 {cn[i_cod]:.3f} -> {cb[i_cod]:.3f}")

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.4, 4.9))
x = np.maximum(tau, 0.08)      # 로그축에 0 을 올리려고 왼쪽 끝을 당겨 둔다

ax1.loglog(x, rs, "s--", color="#1f77b4", lw=2.3, ms=6, label="단순평균")
ax1.loglog(x, rw, "o-", color="#d62728", lw=2.7, ms=7,
           label="역분산 가중평균 (Welch 의 $\\tilde y$)")
if np.isfinite(cross):
    ax1.axvline(max(cross, 0.08), color="#333333", lw=1.4, ls=":")
    ax1.text(max(cross, 0.08) * 1.12, rs[0] * 0.62,
             f"여기서 뒤집힌다\n$\\tau \\approx {cross:.2f}$", fontsize=10)
ax1.set_xticks([0.1, 1, 10])
ax1.set_xticklabels(["$0.1$", "$1$", "$10$"])
ax1.set_yticks([0.2, 0.5, 1, 2])
ax1.set_yticklabels(["$0.2$", "$0.5$", "$1$", "$2$"])
ax1.minorticks_off()
ax1.set_xlabel("알아채지 못한 계통오차 $\\tau$")
ax1.set_ylabel("합의값의 오차 (RMSE)")
ax1.set_title("① 가중평균이 언제나 낫지는 않다", fontsize=12)
ax1.legend(loc="upper left", fontsize=10)
ax1.grid(alpha=0.25, which="major")

ax2.fill_between(x, rblo, rbhi, color="#1f77b4", alpha=0.18,
                 label="가운데 $95\\%$")
ax2.loglog(x, rb, "o-", color="#1f77b4", lw=2.7, ms=7, label="버지 비 $R_B$")
ax2.axhline(1.0, color="black", lw=1.3, ls=":")
ax2.text(0.085, 1.08, "서로 맞으면 $R_B \\approx 1$", fontsize=10)
ax2.axhline(CODATA_RB, color="#d62728", lw=2.0, ls="--",
            label=f"CODATA $2018$ 의 중력상수 (${CODATA_RB}$)")
ax2.plot(max(tau_codata, 0.08), CODATA_RB, "*", color="#d62728", ms=20)
ax2.set_xticks([0.1, 1, 10])
ax2.set_xticklabels(["$0.1$", "$1$", "$10$"])
ax2.set_yticks([1, 2, 5, 10])
ax2.set_yticklabels(["$1$", "$2$", "$5$", "$10$"])
ax2.minorticks_off()
ax2.set_xlabel("알아채지 못한 계통오차 $\\tau$")
ax2.set_ylabel("$R_B = \\sqrt{\\chi^2/(k-1)}$")
ax2.set_title("② 어긋남은 $F$ 꼴로 드러난다", fontsize=12)
ax2.legend(loc="upper left", fontsize=9.5)
ax2.grid(alpha=0.25, which="major")

ax3.semilogx(x, cn, "s--", color="#d62728", lw=2.3, ms=6,
             label="보고된 불확도 그대로")
ax3.semilogx(x, cb, "o-", color="#2ca02c", lw=2.7, ms=7,
             label="$R_B$ 배로 넓힌 뒤")
ax3.axhline(0.95, color="black", lw=1.3, ls=":")
ax3.text(0.085, 0.905, "명목 $95\\%$", fontsize=10)
ax3.axvline(max(tau_codata, 0.08), color="#d62728", lw=1.4, ls="-.")
ax3.annotate(f"$R_B = {CODATA_RB}$ 인 자리\n${cn[i_cod]:.2f} \\to {cb[i_cod]:.2f}$",
             xy=(max(tau_codata, 0.08), cn[i_cod]), xytext=(0.3, 0.42),
             fontsize=10.5,
             arrowprops=dict(arrowstyle="->", color="#555555", lw=1.0))
ax3.set_xticks([0.1, 1, 10])
ax3.set_xticklabels(["$0.1$", "$1$", "$10$"])
ax3.minorticks_off()
ax3.set_ylim(0, 1.05)
ax3.set_xlabel("알아채지 못한 계통오차 $\\tau$")
ax3.set_ylabel("합의 구간이 참값을 담을 확률")
ax3.set_title("③ 넓히면 돌아온다", fontsize=12)
ax3.legend(loc="lower left", fontsize=10)
ax3.grid(alpha=0.25)

fig.tight_layout()
out = "docs/ch13/anova_welch/img/birge_ratio.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
