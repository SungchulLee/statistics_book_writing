r"""축소를 되돌리면 반대쪽으로 넘어간다 — 사후 라쏘의 선택 편향.

18 장은 「사후 라쏘 OLS 로 축소 편향을 제거한다」를 다섯 군데에 적는다(이 쪽
절 3·절 6, lasso.md, information_criteria.md, elastic_net.md). 축소를 되돌리는
것은 맞다. 그런데 되돌린 값은 **고른 뒤에** 다시 잰 값이다.

고르는 기준이 |beta_hat| 이 크다는 것이므로 약한 신호는 **우연히 크게 나온 때만**
뽑히고, 그 조건에서 다시 재면 그 우연이 그대로 남는다 — 8·9 장의 승자의 저주다.

이 쪽 절 5 의 절차 6 이 「더 간결한 모형을 위해 1-SE 규칙을 쓴다」고 권하므로
두 규칙을 함께 본다. 문턱이 높을수록 고르는 일이 선별적이 되어 저주가 세진다.

① 뽑힌 자료에서만 잰 E[beta_hat | 뽑힘] 대 참값.
② 상대 치우침 — 약한 신호에서 몇 퍼센트나 크게 나오는가.
③ 선택을 모른 체한 사후 라쏘 95 % 신뢰구간의 포함확률.

만드는 파일: docs/ch18/overview/img/post_lasso_coverage.png
          (18.3절 lasso 의 post_lasso_bias.png 와 다른 그림이다 — 이쪽은 포함확률을 본다)
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"

from scipy import stats                                       # noqa: E402
from sklearn.linear_model import LassoCV, lasso_path          # noqa: E402

RNG = np.random.default_rng(0)
N, P, SIG, REP = 100, 20, 1.0, 1500
VALS = np.round(np.linspace(0.0, 1.0, 10), 3)
BETA = np.zeros(P)
BETA[:10] = VALS                                   # 뒤 열 개는 참값이 0
NZ = np.arange(1, 10)                              # 0 보다 큰 아홉
NULL = np.arange(10, P)
RULES = ("min", "1se")


def draw(rng):
    X = rng.normal(size=(N, P))
    X -= X.mean(0)
    X /= np.linalg.norm(X, axis=0) / np.sqrt(N)
    y = X @ BETA + SIG * rng.normal(size=N)
    return X, y - y.mean()


acc = {r: dict(nsel=np.zeros(P), s_l=np.zeros(P), s_p=np.zeros(P),
               s_abs=np.zeros(P), cov=np.zeros(P)) for r in RULES}
sum_ols = np.zeros(P)
sum_abs_ols = np.zeros(P)

for _ in range(REP):
    X, y = draw(RNG)
    m = LassoCV(n_alphas=50, cv=5, max_iter=50_000,
                fit_intercept=False, random_state=0).fit(X, y)
    b_ols = np.linalg.lstsq(X, y, rcond=None)[0]
    sum_ols += b_ols
    sum_abs_ols += np.abs(b_ols)
    mse = m.mse_path_.mean(1)
    se = m.mse_path_.std(1) / np.sqrt(m.mse_path_.shape[1])
    i0 = int(np.argmin(mse))
    lam_1se = m.alphas_[mse <= mse[i0] + se[i0]].max()
    for rule in RULES:
        if rule == "min":
            b = m.coef_
        else:
            b = lasso_path(X, y, alphas=[lam_1se])[1][:, 0]
        S = np.flatnonzero(b)
        if S.size == 0 or S.size >= N:
            continue
        XS = X[:, S]
        G = np.linalg.inv(XS.T @ XS)
        bS = G @ XS.T @ y
        r = y - XS @ bS
        s2 = float(r @ r) / (N - S.size)
        t = stats.t.ppf(0.975, N - S.size) * np.sqrt(s2 * np.diag(G))
        a = acc[rule]
        a["nsel"][S] += 1
        a["s_l"][S] += b[S]
        a["s_p"][S] += bS
        a["s_abs"][S] += np.abs(bS)
        a["cov"][S] += (bS - t <= BETA[S]) & (BETA[S] <= bS + t)

OUT = {}
for rule in RULES:
    a = acc[rule]
    n = np.maximum(a["nsel"], 1)
    OUT[rule] = dict(
        psel=a["nsel"] / REP,
        lasso=np.where(a["nsel"] > 0, a["s_l"] / n, np.nan),
        post=np.where(a["nsel"] > 0, a["s_p"] / n, np.nan),
        absp=np.where(a["nsel"] > 0, a["s_abs"] / n, np.nan),
        cov=np.where(a["nsel"] > 0, a["cov"] / n, np.nan))
E_OLS, E_ABS_OLS = sum_ols / REP, sum_abs_ols / REP

# 한글은 고정폭에서 두 칸을 차지하므로 자리맞춤을 직접 적는다
print(f"n = {N}, p = {P}, sigma = {SIG}, 되풀이 {REP:,}")
print("lambda 는 되풀이마다 5 겹 교차검증으로 골랐다. 뒤 열 개의 참값은 0 이다.")
print(f"고르지 않은 OLS 는 거의 불편이다 — 가장 큰 치우침 "
      f"{np.max(np.abs(E_OLS - BETA)):.4f}\n")

for rule, lab in (("min", "lambda_min"), ("1se", "lambda_1SE")):
    o = OUT[rule]
    print(f"① [{lab}] 뽑힌 자료에서만 잰 추정값")
    print("   참 beta   선택확률    라쏘    사후 라쏘   치우침   상대 치우침")
    for j in range(10):
        rel = (o["post"][j] / BETA[j] - 1) * 100 if BETA[j] > 0 else float("nan")
        rs = "—" if BETA[j] == 0 else f"{rel:>+9.1f} %"
        print(f"   {BETA[j]:>7.3f}   {o['psel'][j]:>8.3f}   "
              f"{o['lasso'][j]:>6.3f}   {o['post'][j]:>8.3f}   "
              f"{o['post'][j]-BETA[j]:>+7.3f}   {rs}")
    print()

print("② 참값이 0 인 열 개 — 부호가 섞이므로 크기로 본다")
print("   규칙          선택확률   E[|post| | 뽑힘]   그냥 OLS 의 E[|beta_hat|]")
for rule, lab in (("min", "lambda_min"), ("1se", "lambda_1SE")):
    o = OUT[rule]
    print(f"   {lab:<12}  {o['psel'][NULL].mean():>8.3f}   "
          f"{np.nanmean(o['absp'][NULL]):>16.3f}   "
          f"{E_ABS_OLS[NULL].mean():>23.3f}")
print("   뽑혔다는 조건이 붙으면 크기가 부풀어 있다. 고르지 않았으면 그만큼이 아니다.")

print("\n③ 선택을 모른 체한 사후 라쏘 95 % 신뢰구간의 포함확률")
print("   참 beta   lambda_min   lambda_1SE")
for j in list(range(0, 10, 2)):
    print(f"   {BETA[j]:>7.3f}   {OUT['min']['cov'][j]:>10.3f}   "
          f"{OUT['1se']['cov'][j]:>10.3f}")
print(f"   {'0 (잡음)':>7}   {np.nanmean(OUT['min']['cov'][NULL]):>10.3f}   "
      f"{np.nanmean(OUT['1se']['cov'][NULL]):>10.3f}")
for rule, lab in (("min", "lambda_min"), ("1se", "lambda_1SE")):
    print(f"   {lab} 의 가장 나쁜 자리 {np.nanmin(OUT[rule]['cov']):.3f}")
print("   명목은 0.95 다.")

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.6, 5.2))
C_L, C_M, C_S = "#1f77b4", "#ff7f0e", "#d62728"
v = BETA[:10]

ax1.plot([0, 1.05], [0, 1.05], "--", color="#333333", lw=2.2, label="참값")
ax1.plot(v, OUT["1se"]["lasso"][:10], "o-", color=C_L, lw=2.6, ms=8,
         label="라쏘 ($\\lambda_{1SE}$)")
ax1.plot(v, OUT["min"]["post"][:10], "^-", color=C_M, lw=2.6, ms=8,
         label="사후 라쏘 ($\\lambda_{\\min}$)")
ax1.plot(v, OUT["1se"]["post"][:10], "s-", color=C_S, lw=2.8, ms=9,
         label="사후 라쏘 ($\\lambda_{1SE}$)")
ax1.set_xlabel("참 $\\beta_j$")
ax1.set_ylabel("$\\mathrm{E}[\\hat\\beta_j \\mid$ 뽑힘$]$")
ax1.set_xlim(-0.03, 1.05)
ax1.set_title("① 뽑힌 것만 모아 재면", fontsize=12)
ax1.legend(loc="upper left", fontsize=10)
ax1.grid(alpha=0.25)

for rule, c, mk, lab in (("min", C_M, "^", "$\\lambda_{\\min}$"),
                         ("1se", C_S, "s", "$\\lambda_{1SE}$")):
    rel = (OUT[rule]["post"][NZ] / BETA[NZ] - 1) * 100
    ax2.plot(BETA[NZ], rel, mk + "-", color=c, lw=2.8, ms=9, label=lab)
ax2.axhline(0.0, color="#333333", lw=2.2, ls="--")
ax2.set_xlabel("참 $\\beta_j$")
ax2.set_ylabel("사후 라쏘의 상대 치우침 (%)")
ax2.set_title("② 약한 신호일수록 부풀어 보인다", fontsize=12)
ax2.legend(loc="upper right", fontsize=11)
ax2.grid(alpha=0.25)

for rule, c, mk, lab in (("min", C_M, "^", "$\\lambda_{\\min}$"),
                         ("1se", C_S, "s", "$\\lambda_{1SE}$")):
    ax3.plot(v, OUT[rule]["cov"][:10], mk + "-", color=c, lw=2.8, ms=9,
             label=lab)
ax3.axhline(0.95, color="#333333", lw=2.2, ls="--")
ax3.text(0.42, 0.958, "명목 $0.95$", fontsize=10.5, color="#333333")
ax3.set_xlabel("참 $\\beta_j$")
ax3.set_ylabel("$95\\%$ 신뢰구간의 포함확률")
ax3.set_ylim(0.6, 1.02)
ax3.set_title("③ 그 구간을 그대로 보고하면", fontsize=12)
ax3.legend(loc="lower right", fontsize=11)
ax3.grid(alpha=0.25)

fig.tight_layout()
out = "docs/ch18/overview/img/post_lasso_coverage.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
