r"""라플라스 사전분포 하나로 두 가지를 할 수는 없다.

이 쪽의 절 5 는 「희소 모형에 대한 불확실성 정량화(신용구간)가 필요할 때
베이즈 라쏘를 쓴다」고 권한다. 그 권고를 재어 본다.

카스티요·슈미트히버·판데르파르트(2015)가 적은 까닭은 이렇다. 라플라스 사전분포의
lambda 는 잡음 계수를 0 으로 밀려면 커야 하는데, 동시에 0 이 아닌 계수를 제대로
모형화하려면 크지 않아야 한다. 최빈값(라쏘)은 그 모순을 기하로 피해 가지만
사후분포 전체는 피하지 못한다.

절 5 의 깁스 표집기를 그대로 구현해서(파크·카셀라) 95 % 신용구간의 빈도주의
포함확률을 잰다. 비교 상대는 절 6 의 스파이크-앤-슬랩이다.

① 참값에 따른 포함확률 — 베이즈 라쏘 대 스파이크-앤-슬랩.
② 잡음 계수에서 사후분포가 0 에 얼마나 모이는가.
③ lambda 를 움직이며 — 잡음 쪽과 신호 쪽의 포함확률이 반대로 간다.

만드는 파일: docs/ch18/lasso/img/bayes_lasso_coverage.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"

N, P, SIG = 100, 20, 1.0
BETA = np.zeros(P)
VALS = [3.0, 2.0, 1.0, 0.5]          # 0 이 아닌 넷
BETA[:4] = VALS
NZ = np.arange(4)
ZERO = np.arange(4, P)
REP, SWEEP, BURN = 200, 1500, 500


def design(rng, n=N, p=P):
    X = rng.normal(size=(n, p))
    X -= X.mean(0)
    X /= np.linalg.norm(X, axis=0) / np.sqrt(n)
    return X


X = design(np.random.default_rng(3))
XtX = X.T @ X


def gibbs_blasso(y, lam, rng, sweep=SWEEP, burn=BURN):
    """파크·카셀라의 베이즈 라쏘. sigma^2 는 안다고 둔다."""
    Xty = X.T @ y
    tau2 = np.ones(P)
    out = np.empty((sweep - burn, P))
    for t in range(sweep):
        A = XtX + np.diag(1.0 / tau2)
        L = np.linalg.cholesky(A)
        m = np.linalg.solve(A, Xty)
        z = rng.normal(size=P)
        b = m + SIG * np.linalg.solve(L.T, z)
        ab = np.maximum(np.abs(b), 1e-8)
        inv_tau2 = rng.wald(np.sqrt(lam ** 2 * SIG ** 2 / ab ** 2), lam ** 2)
        tau2 = 1.0 / np.maximum(inv_tau2, 1e-10)
        if t >= burn:
            out[t - burn] = b
    return out


def gibbs_ss(y, rng, pi=0.2, g=4.0, sweep=SWEEP, burn=BURN):
    """절 6 의 스파이크-앤-슬랩. 0 에 점질량을 둔다."""
    b = np.zeros(P)
    r = y.copy()
    out = np.empty((sweep - burn, P))
    v = np.diag(XtX)
    for t in range(sweep):
        for j in range(P):
            if b[j] != 0.0:
                r = r + X[:, j] * b[j]
            zj = X[:, j] @ r
            prec = v[j] + 1.0 / g
            # 0 에 있을 때와 슬랩에 있을 때의 주변가능도 비
            logbf = (0.5 * np.log(1.0 / (g * prec))
                     + zj ** 2 / (2.0 * SIG ** 2 * prec))
            pj = 1.0 / (1.0 + (1 - pi) / pi * np.exp(-logbf))
            if rng.random() < pj:
                b[j] = (zj / prec
                        + rng.normal() * SIG / np.sqrt(prec))
                r = r - X[:, j] * b[j]
            else:
                b[j] = 0.0
        if t >= burn:
            out[t - burn] = b
    return out


def ci_cover(draws, truth, lo=2.5, hi=97.5):
    a = np.percentile(draws, lo, axis=0)
    c = np.percentile(draws, hi, axis=0)
    return (a <= truth) & (truth <= c), c - a


# === ①② 두 방법 =====================================================
LAM0 = 6.0                       # 뒤의 ③ 에서 고르는 범위의 가운데
COV_B = np.zeros(P); COV_S = np.zeros(P)
WID_B = np.zeros(P); WID_S = np.zeros(P)
MASS_B = np.zeros(len(ZERO)); MASS_S = np.zeros(len(ZERO))
rng = np.random.default_rng(11)
for _ in range(REP):
    y = X @ BETA + SIG * rng.normal(size=N)
    db = gibbs_blasso(y, LAM0, rng)
    ds = gibbs_ss(y, rng)
    cb, wb = ci_cover(db, BETA)
    cs, ws = ci_cover(ds, BETA)
    COV_B += cb; COV_S += cs; WID_B += wb; WID_S += ws
    MASS_B += (np.abs(db[:, ZERO]) < 0.05).mean(0)
    MASS_S += (ds[:, ZERO] == 0.0).mean(0)
COV_B /= REP; COV_S /= REP; WID_B /= REP; WID_S /= REP
MASS_B /= REP; MASS_S /= REP

# 한글은 고정폭에서 두 칸을 차지하므로 자리맞춤을 직접 적는다
print(f"n = {N}, p = {P}, 0 이 아닌 계수 {len(NZ)} 개 {VALS}, 되풀이 {REP}")
print(f"깁스 {SWEEP} 번 가운데 앞 {BURN} 을 버렸다. lambda = {LAM0}")
print("\n① 95 % 신용구간의 빈도주의 포함확률")
print("   참 beta    베이즈 라쏘   스파이크-앤-슬랩   라쏘 폭   S&S 폭")
for j in NZ:
    print(f"   {BETA[j]:>7.1f}   {COV_B[j]:>11.3f}   {COV_S[j]:>16.3f}   "
          f"{WID_B[j]:>7.3f}   {WID_S[j]:>6.3f}")
print(f"   {0.0:>7.1f}   {COV_B[ZERO].mean():>11.3f}   "
      f"{COV_S[ZERO].mean():>16.3f}   {WID_B[ZERO].mean():>7.3f}   "
      f"{WID_S[ZERO].mean():>6.3f}   (잡음 {len(ZERO)} 개 평균)")

print("\n② 신용구간의 폭 — 덮기만 하면 되는 것이 아니다")
print(f"   베이즈 라쏘  |beta| < 0.05 인 사후질량  {MASS_B.mean():.3f}")
print(f"   스파이크-앤-슬랩  beta = 0 인 사후질량   {MASS_S.mean():.3f}")
print("   라플라스 사후분포는 0 에 점질량이 없다 — 모일 수는 있어도 앉지는 못한다.")

# === ③ lambda 를 움직이며 ===========================================
LAMS = [1.0, 2.0, 4.0, 6.0, 10.0, 16.0, 25.0]
C_ZERO, C_BIG, W_BIG, W_ZERO = [], [], [], []
for lam in LAMS:
    cz = cb_ = 0.0
    wb_ = wz_ = 0.0
    rng = np.random.default_rng(11)
    for _ in range(REP // 2):
        y = X @ BETA + SIG * rng.normal(size=N)
        d = gibbs_blasso(y, lam, rng)
        c, w = ci_cover(d, BETA)
        cz += c[ZERO].mean()
        cb_ += c[NZ[:2]].mean()          # 큰 신호 둘 (3.0, 2.0)
        wb_ += w[NZ[:2]].mean()
        wz_ += w[ZERO].mean()
    W_ZERO.append(wz_ / (REP // 2))
    C_ZERO.append(cz / (REP // 2))
    C_BIG.append(cb_ / (REP // 2))
    W_BIG.append(wb_ / (REP // 2))
C_ZERO, C_BIG, W_BIG, W_ZERO = map(
    np.array, (C_ZERO, C_BIG, W_BIG, W_ZERO))
print("\n③ lambda 를 키우며 (되풀이 {} 번)".format(REP // 2))
print("   lambda   잡음 포함확률   잡음 구간 폭   큰 신호 포함확률")
for i, lam in enumerate(LAMS):
    print(f"   {lam:>6.1f}   {C_ZERO[i]:>13.3f}   {W_ZERO[i]:>12.3f}   "
          f"{C_BIG[i]:>16.3f}")
print(f"   lambda 가 작으면 포함확률은 둘 다 명목에 가깝지만 "
      f"(lambda = {LAMS[0]:g} 에서 {C_ZERO[0]:.3f} / {C_BIG[0]:.3f}),")
print(f"   잡음 구간이 {W_ZERO[0]:.3f} 로 넓어 **아무것도 0 으로 밀지 않는다.**")
print(f"   lambda 를 키워 그 폭을 {W_ZERO[-1]:.3f} 까지 좁히면 큰 신호의 "
      f"포함확률이 {C_BIG[-1]:.3f} 로 무너진다.")
print("   **희소하게 만드는 lambda 와 덮어 주는 lambda 가 다르다.**")

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.6, 5.2))
C_B, C_S = "#d62728", "#1f77b4"

xs = np.r_[0.0, VALS[::-1]]
cb_plot = np.r_[COV_B[ZERO].mean(), COV_B[NZ][::-1]]
cs_plot = np.r_[COV_S[ZERO].mean(), COV_S[NZ][::-1]]
ax1.plot(xs, cb_plot, "o-", color=C_B, lw=2.8, ms=10, label="베이즈 라쏘")
ax1.plot(xs, cs_plot, "s-", color=C_S, lw=2.8, ms=9,
         label="스파이크-앤-슬랩")
ax1.axhline(0.95, color="#333333", lw=2.2, ls="--")
ax1.text(1.55, 0.965, "명목 $0.95$", fontsize=10.5, color="#333333")
ax1.set_xlabel("참 $\\beta_j$")
ax1.set_ylabel("$95\\%$ 신용구간의 포함확률")
ax1.set_ylim(0, 1.05)
ax1.set_title("① 신용구간이 참값을 덮는가", fontsize=12)
ax1.legend(loc="lower left", fontsize=10.5)
ax1.grid(alpha=0.25)

wb_plot = np.r_[WID_B[ZERO].mean(), WID_B[NZ][::-1]]
ws_plot = np.r_[WID_S[ZERO].mean(), WID_S[NZ][::-1]]
ax2.plot(xs, wb_plot, "o-", color=C_B, lw=2.8, ms=10, label="베이즈 라쏘")
ax2.plot(xs, ws_plot, "s-", color=C_S, lw=2.8, ms=9, label="스파이크-앤-슬랩")
ax2.annotate(f"잡음에서 ${wb_plot[0]/ws_plot[0]:.0f}$ 배 넓다",
             (xs[0], wb_plot[0]), xytext=(0.7, -2.2),
             textcoords="offset fontsize", fontsize=11, color=C_B)
ax2.set_xlabel("참 $\\beta_j$")
ax2.set_ylabel("$95\\%$ 신용구간의 평균 폭")
ax2.set_ylim(0, max(wb_plot.max(), ws_plot.max()) * 1.25)
ax2.set_title("② 그 구간이 무엇을 말해 주는가", fontsize=12)
ax2.legend(loc="lower right", fontsize=10.5)
ax2.grid(alpha=0.25)

ax3.semilogx(LAMS, C_ZERO, "o-", color="#2ca02c", lw=2.8, ms=10,
             label="잡음 계수 ($\\beta_j = 0$)")
ax3.semilogx(LAMS, C_BIG, "s-", color=C_B, lw=2.8, ms=10,
             label="큰 신호 ($\\beta_j = 3,\\, 2$)")
ax3.axhline(0.95, color="#333333", lw=2.2, ls="--")
ax3.text(10.5, 0.965, "명목 $0.95$", fontsize=10.5, color="#333333")
ax3.set_xlabel("$\\lambda$")
ax3.set_ylabel("포함확률")
ax3.set_ylim(0, 1.05)
ax3.set_xticks(LAMS)
ax3.set_xticklabels([f"{v:g}" for v in LAMS])
ax3.minorticks_off()
ax3.set_title("③ 덮는 $\\lambda$ 와 희소하게 하는 $\\lambda$ 가 다르다",
              fontsize=12)
ax3.grid(alpha=0.25)
ax3b = ax3.twinx()
ax3b.semilogx(LAMS, W_ZERO, "^:", color="#9467bd", lw=2.4, ms=9,
              label="잡음 구간의 폭 (오른쪽 축)")
ax3b.set_ylabel("잡음 계수 신용구간의 폭", color="#9467bd")
ax3b.tick_params(axis="y", labelcolor="#9467bd")
ax3b.set_ylim(0, 0.52)
# twinx 가 x 축 눈금을 되돌리므로 여기서 다시 적는다
for _a in (ax3, ax3b):
    _a.set_xticks(LAMS)
    _a.set_xticklabels([f"{v:g}" for v in LAMS])
    _a.minorticks_off()
h1, l1 = ax3.get_legend_handles_labels()
h2, l2 = ax3b.get_legend_handles_labels()
ax3.legend(h1 + h2, l1 + l2, loc="lower left", fontsize=9.5)

fig.tight_layout()
out = "docs/ch18/lasso/img/bayes_lasso_coverage.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
