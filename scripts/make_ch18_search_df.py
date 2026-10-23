r"""고른 변수를 세는 것으로 자유도가 되는 것은 라쏘뿐이다.

이 쪽의 절 2 는 라쏘의 유효자유도가 0 이 아닌 계수의 개수와 같다고 적는다
(Zou, Hastie, Tibshirani 2007). 「고른 것을 세면 된다」는 말로 들리지만,
그것은 라쏘에만 해당하는 일이다.

자유도의 정의는 에프론의 공분산 꼴이다.

    df = (1/sigma^2) * sum_i Cov(yhat_i, y_i)

y 를 보고 변수를 고르면 그 고르는 행위 자체가 Cov(yhat_i, y_i) 를 키운다.
최량부분집합은 그 값을 치르고, 라쏘는 축소로 그만큼을 되돌려 준다.

① df 대 고른 변수의 개수 k — 최량부분집합은 45 도 선 위로 치솟고 라쏘는 붙어 간다.
② 얀손·피시언·헤이스티의 두 변수 보기 — 자유도가 주위 공간의 차원 2 를 넘는다.
③ df = k 로 믿고 AIC 를 쓰면 되풀이마다 어떤 k 를 고르는가.

만드는 파일: docs/ch18/tuning/img/search_df.png
"""

import itertools
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"

from sklearn.linear_model import Lasso   # noqa: E402

RNG = np.random.default_rng(0)
N, P, SIG, REP = 100, 10, 1.0, 4000
K_TRUE = 3


def design(rng, n=N, p=P, rho=0.3):
    """고정된 계획행렬. 열끼리 약하게 상관시킨다."""
    A = rng.normal(size=(n, p))
    C = rng.normal(size=(n, 1))
    X = np.sqrt(1 - rho) * A + np.sqrt(rho) * C
    X -= X.mean(0)
    X /= np.linalg.norm(X, axis=0) / np.sqrt(n)
    return X


X = design(RNG)
BETA = np.zeros(P)
BETA[:K_TRUE] = np.array([1.2, -1.0, 0.8])
MU = X @ BETA
MU -= MU.mean()

Y = MU[:, None] + SIG * RNG.normal(size=(N, REP))      # n x REP
Y -= Y.mean(0)


def df_of(YH, Y=Y, sig=SIG):
    """에프론의 공분산 자유도. 열마다 한 번의 되풀이다."""
    yc = Y - Y.mean(1, keepdims=True)
    hc = YH - YH.mean(1, keepdims=True)
    return float((hc * yc).sum() / (YH.shape[1] - 1) / sig ** 2)


# === ① 최량부분집합 ==================================================
SUBSETS = {k: list(itertools.combinations(range(P), k)) for k in range(P + 1)}
BEST_RSS = {k: np.full(REP, np.inf) for k in range(P + 1)}
BEST_YH = {k: np.zeros((N, REP)) for k in range(P + 1)}
for k in range(P + 1):
    for S in SUBSETS[k]:
        if k == 0:
            YH = np.zeros((N, REP))
        else:
            XS = X[:, list(S)]
            YH = XS @ np.linalg.solve(XS.T @ XS, XS.T @ Y)
        rss = ((Y - YH) ** 2).sum(0)
        m = rss < BEST_RSS[k]
        BEST_RSS[k][m] = rss[m]
        BEST_YH[k][:, m] = YH[:, m]

KS = np.arange(P + 1)
DF_BS = np.array([df_of(BEST_YH[k]) for k in KS])
RSS_BS = np.array([BEST_RSS[k].mean() for k in KS])

# 한글은 고정폭에서 두 칸을 차지하므로 자리맞춤을 직접 적는다
print(f"n = {N}, p = {P}, 참 변수 {K_TRUE} 개, 되풀이 {REP:,}")
print(f"자유도는 에프론의 공분산 꼴로 쟀다.\n")
print("① 최량부분집합 — 변수를 k 개 고르면 자유도는 얼마인가")
print("    k    참 df    k 와의 차    배수")
for k in KS:
    r = DF_BS[k] / k if k else float("nan")
    print(f"   {k:>2}   {DF_BS[k]:>6.2f}   {DF_BS[k]-k:>9.2f}   "
          f"{'—' if k == 0 else f'{r:>5.2f}'}")
print(f"   k = 1 에서 이미 {DF_BS[1]:.2f} 다 — 하나를 고르는 데 "
      f"{DF_BS[1]:.2f} 개 몫을 쓴 셈이다.")
print(f"   k = p = {P} 에서만 {DF_BS[P]:.2f} 로 k 와 맞는다 "
      "(고를 것이 없으니 탐색 비용이 0 이다).")

# === 라쏘 ============================================================
LAMS = np.logspace(np.log10(0.006), np.log10(0.9), 22)
DF_L, NNZ_L, RSS_L = [], [], []
for lam in LAMS:
    YH = np.zeros((N, REP))
    nnz = np.zeros(REP)
    las = Lasso(alpha=lam, fit_intercept=False, max_iter=50_000, tol=1e-9)
    for r in range(REP):
        las.fit(X, Y[:, r])
        YH[:, r] = X @ las.coef_
        nnz[r] = int((las.coef_ != 0).sum())
    DF_L.append(df_of(YH))
    NNZ_L.append(float(nnz.mean()))
    RSS_L.append(float(((Y - YH) ** 2).sum(0).mean()))
DF_L, NNZ_L, RSS_L = map(np.array, (DF_L, NNZ_L, RSS_L))

print("\n② 라쏘 — 0 이 아닌 계수의 평균 개수와 참 자유도")
print("   lambda    E|S|     참 df    차")
for i in (0, 4, 8, 12, 16, 21):
    print(f"   {LAMS[i]:>6.3f}   {NNZ_L[i]:>5.2f}   {DF_L[i]:>6.2f}   "
          f"{DF_L[i]-NNZ_L[i]:>+5.2f}")
print(f"   가장 크게 어긋난 곳도 {np.abs(DF_L-NNZ_L).max():.2f} 다. "
      "**세는 것으로 맞는다.**")

# === 능형 — 재는 법이 맞는지 =========================================
LAM_R = np.logspace(-2, 3, 18)
G = X.T @ X
DF_R, DF_R_TRUE = [], []
for lam in LAM_R:
    Hm = X @ np.linalg.solve(G + lam * np.eye(P), X.T)
    DF_R.append(df_of(Hm @ Y))
    DF_R_TRUE.append(float(np.trace(Hm)))
DF_R, DF_R_TRUE = np.array(DF_R), np.array(DF_R_TRUE)
print(f"\n③ 능형으로 재는 법을 검산 — 모의 df 와 닫힌 형태 tr(H)")
print(f"   가장 큰 차 {np.abs(DF_R-DF_R_TRUE).max():.4f} "
      f"(값은 {DF_R_TRUE.min():.2f} 에서 {DF_R_TRUE.max():.2f} 까지)")
print("   추정량 자체는 맞는다. ①의 치솟음은 재는 법의 탓이 아니다.")

# === 얀손·피시언·헤이스티의 두 변수 보기 ============================
# y ~ N(mu, I_2) 에서 두 일변량 모형 가운데 나은 것을 고른다.
# 자유로운 모수는 하나뿐이고 주위 공간은 2 차원인데 df 가 2 를 넘는다.
TS = np.linspace(0, 14, 29)
RJ = 400_000
rj = np.random.default_rng(7)
DF_J = []
for t in TS:
    Yj = np.array([t, t]) + rj.normal(size=(RJ, 2))
    p1 = np.abs(Yj[:, 0]) >= np.abs(Yj[:, 1])
    YHj = np.zeros_like(Yj)
    YHj[p1, 0] = Yj[p1, 0]
    YHj[~p1, 1] = Yj[~p1, 1]
    DF_J.append(sum(np.cov(YHj[:, j], Yj[:, j])[0, 1] for j in range(2)))
DF_J = np.array(DF_J)
print("\n④ 두 일변량 모형 가운데 나은 것 고르기 (주위 공간은 2 차원)")
print("   mu = (t, t)     df")
for t in (0.0, 2.0, 4.0, 7.0, 10.0, 14.0):
    i = int(np.argmin(np.abs(TS - t)))
    print(f"   t = {TS[i]:>5.1f}   {DF_J[i]:>6.3f}")
print(f"   모수는 하나뿐인데 df 가 {DF_J.max():.2f} 까지 간다. "
      "**2 를 넘는 것부터가 이상한 일이다.**")

# === ④ df = k 로 믿으면 어떤 k 를 고르는가 ===========================
AIC_NAIVE = RSS_BS / SIG ** 2 + 2 * KS
AIC_TRUE = RSS_BS / SIG ** 2 + 2 * DF_BS
k_naive, k_true = int(np.argmin(AIC_NAIVE)), int(np.argmin(AIC_TRUE))
print(f"\n⑤ 최량부분집합에 AIC 를 씌우면 (참 변수는 {K_TRUE} 개)")
print("    k      RSS    df=k 로 믿은 AIC    참 df 를 쓴 AIC")
for k in KS:
    mk = " ←" if k in (k_naive, k_true) else ""
    print(f"   {k:>2}   {RSS_BS[k]:>6.2f}   {AIC_NAIVE[k]:>15.2f}   "
          f"{AIC_TRUE[k]:>14.2f}{mk}")
print(f"   df = k 로 믿으면 k = {k_naive} 을, 참 df 를 쓰면 k = {k_true} 을 고른다.")

RSS_MAT = np.vstack([BEST_RSS[k] for k in KS])          # (p+1) x REP
SEL_N = KS[np.argmin(RSS_MAT / SIG ** 2 + 2 * KS[:, None], axis=0)]
SEL_T = KS[np.argmin(RSS_MAT / SIG ** 2 + 2 * DF_BS[:, None], axis=0)]
print("\n   되풀이마다 고른 k 의 분포")
print("    k      df=k 로 믿었을 때   참 df 를 썼을 때")
for k in KS:
    a, b = (SEL_N == k).mean(), (SEL_T == k).mean()
    if a or b:
        print(f"   {k:>2}   {a:>17.3f}   {b:>16.3f}")
print(f"   평균 크기 {SEL_N.mean():.2f} 대 {SEL_T.mean():.2f}, "
      f"참 크기를 맞힌 비율 {(SEL_N==K_TRUE).mean():.3f} 대 "
      f"{(SEL_T==K_TRUE).mean():.3f}")

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.6, 5.2))
C_B, C_L, C_R = "#d62728", "#2ca02c", "#1f77b4"

ax1.plot([0, P], [0, P], "--", color="#333333", lw=2.2,
         label="$\\mathrm{df} = $ 고른 개수")
ax1.plot(KS, DF_BS, "o-", color=C_B, lw=2.8, ms=8, label="최량부분집합")
ax1.plot(NNZ_L, DF_L, "s-", color=C_L, lw=2.8, ms=7, label="라쏘")
ax1.annotate(f"$k=1$ 인데 $\\mathrm{{df}}={DF_BS[1]:.1f}$", (1, DF_BS[1]),
             xytext=(1.2, -2.6), textcoords="offset fontsize", fontsize=10.5,
             color=C_B)
ax1.set_xlabel("$0$ 이 아닌 계수의 개수")
ax1.set_ylabel("참 자유도 $\\mathrm{df}$")
ax1.set_xlim(-0.4, P + 0.4)
ax1.set_title("① 고른 것을 세면 자유도가 되는가", fontsize=12)
ax1.legend(loc="upper left", fontsize=10.5)
ax1.grid(alpha=0.25)

ax2.plot(TS, DF_J, "o-", color="#9467bd", lw=2.8, ms=7)
ax2.axhline(2.0, color="#333333", lw=2.2, ls="--")
ax2.text(0.3, 2.2, "주위 공간의 차원 $= 2$", fontsize=10.5, color="#333333")
ax2.axhline(1.0, color=C_L, lw=2.0, ls=":")
ax2.text(0.3, 1.1, "자유로운 모수는 $1$ 개", fontsize=10.5, color=C_L)
ax2.set_xlabel("$\\mu = (t,\\, t)$ 의 $t$")
ax2.set_ylabel("$\\mathrm{df}$")
ax2.set_title("② 모수 하나짜리 절차의 자유도", fontsize=12)
ax2.grid(alpha=0.25)

w = 0.4
fn = np.array([(SEL_N == k).mean() for k in KS])
ft = np.array([(SEL_T == k).mean() for k in KS])
ax3.bar(KS - w / 2, fn, width=w, color="#9467bd",
        label=f"$\\mathrm{{df}} = k$ 로 믿은 AIC (평균 {SEL_N.mean():.2f})")
ax3.bar(KS + w / 2, ft, width=w, color=C_B,
        label=f"참 $\\mathrm{{df}}$ 를 쓴 AIC (평균 {SEL_T.mean():.2f})")
ax3.axvline(K_TRUE, color="#333333", lw=2.0, ls=":")
ax3.text(K_TRUE + 0.18, 0.95, f"참 변수 {K_TRUE} 개", fontsize=10.5,
         color="#333333", va="top")
ax3.set_xlabel("고른 변수의 개수 $k$")
ax3.set_ylabel("되풀이 가운데 그 $k$ 를 고른 비율")
ax3.set_xticks(KS)
ax3.set_xlim(1.4, P + 0.6)
ax3.set_ylim(0, 1.0)
ax3.set_title("③ 자유도를 틀리면 모형도 틀린다", fontsize=12)
ax3.legend(loc="upper right", fontsize=10)
ax3.grid(alpha=0.25, axis="y")

fig.tight_layout()
out = "docs/ch18/tuning/img/search_df.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
