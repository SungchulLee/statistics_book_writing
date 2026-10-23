r"""그룹 효과는 묶을 것을 가리지 않는다.

이 쪽의 절 6 은 그룹 효과의 이점만 적는다 — 경로 구성원이 함께 뽑히고, 상관된
지표가 비슷한 계수를 받는다. 그런데 묶는 기준은 **상관**뿐이고 「참인가」가
아니다. 참 변수와 상관된 잡음변수는 그 묶음에 함께 실려 들어온다.

설계: 참 집단 x1,x2,x3 (계수 1.5, 서로 상관 0.95).
      위성 잡음 x4,x5,x6 (계수 0, 참 집단과 상관 0.9).
      홀로 선 잡음 x7..x20 (계수 0, 상관 없음).
alpha 마다 lambda 를 교차검증으로 고른다.

① alpha 를 낮추면 참 집단 안의 계수가 모이는가 (이점).
② 뽑는 개수를 6 개로 맞추어 놓고, 그 여섯이 무엇인가 (기제를 떼어 본다).
③ 교차검증으로 고른 lambda 에서 실제로 받는 목록의 정밀도와 재현율.

만드는 파일: docs/ch18/elastic_net/img/grouping_cost.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"

from sklearn.linear_model import ElasticNetCV, enet_path   # noqa: E402

RNG = np.random.default_rng(0)
N, P, REP = 100, 20, 300
R_IN, R_SAT = 0.95, 0.90        # 집단 안 상관, 위성과의 상관
BETA = np.zeros(P)
BETA[:3] = 1.5                  # 참은 셋뿐
TRUE = np.array([0, 1, 2])
SAT = np.array([3, 4, 5])       # 위성 잡음 — 참과 상관되어 있다
LONE = np.arange(6, P)          # 홀로 선 잡음


def draw(rng, n=N):
    g = rng.normal(size=(n, 1))                       # 집단의 공통 요인
    Z = np.empty((n, P))
    for j in range(3):                                # 참 집단
        Z[:, j] = (np.sqrt(R_IN) * g[:, 0]
                   + np.sqrt(1 - R_IN) * rng.normal(size=n))
    for i, j in enumerate(SAT):                       # 위성 — 참 쪽 하나에 붙인다
        Z[:, j] = (np.sqrt(R_SAT) * Z[:, i]
                   + np.sqrt(1 - R_SAT) * rng.normal(size=n))
    Z[:, LONE] = rng.normal(size=(n, len(LONE)))
    Z -= Z.mean(0)
    Z /= Z.std(0)
    y = Z @ BETA + rng.normal(size=n)
    return Z, y - y.mean()


Z0, _ = draw(np.random.default_rng(1))
C = np.corrcoef(Z0.T)
print(f"n = {N}, p = {P}, 되풀이 {REP}")
print(f"참 변수는 x1,x2,x3 (계수 {BETA[0]}), 위성 잡음은 x4,x5,x6 (계수 0)")
print(f"   표본상관: 참끼리 {C[0,1]:.3f}, 참–위성 {C[0,3]:.3f}, "
      f"참–홀로선잡음 {C[0,6]:.3f}")

ALPHAS = [1.0, 0.9, 0.7, 0.5, 0.3, 0.15, 0.05]
SD_IN, P_SAT, P_LONE, P_TRUE, NFP = [], [], [], [], []
for a in ALPHAS:
    sd, ps, pl, pt, nf = [], [], [], [], []
    rng = np.random.default_rng(42)
    for _ in range(REP):
        Zm, y = draw(rng)
        m = ElasticNetCV(l1_ratio=a, cv=5, n_alphas=40, max_iter=20_000,
                         tol=1e-6, fit_intercept=False, random_state=0)
        m.fit(Zm, y)
        b = m.coef_
        sd.append(b[TRUE].std(ddof=1))
        ps.append((b[SAT] != 0).mean())
        pl.append((b[LONE] != 0).mean())
        pt.append((b[TRUE] != 0).mean())
        nf.append(int((b[SAT] != 0).sum() + (b[LONE] != 0).sum()))
    SD_IN.append(np.mean(sd)); P_SAT.append(np.mean(ps))
    P_LONE.append(np.mean(pl)); P_TRUE.append(np.mean(pt))
    NFP.append(np.mean(nf))
SD_IN, P_SAT, P_LONE = map(np.array, (SD_IN, P_SAT, P_LONE))
P_TRUE, NFP = np.array(P_TRUE), np.array(NFP)

# 한글은 고정폭에서 두 칸을 차지하므로 자리맞춤을 직접 적는다
print("\n① 이점과 대가를 한 표에")
print("   alpha   집단 안 표준편차   참 뽑힘   위성 뽑힘   홀로선 잡음   거짓양성 수")
for i, a in enumerate(ALPHAS):
    print(f"   {a:>5.2f}   {SD_IN[i]:>14.3f}   {P_TRUE[i]:>7.3f}   "
          f"{P_SAT[i]:>9.3f}   {P_LONE[i]:>11.3f}   {NFP[i]:>11.2f}")
print(f"   alpha 를 {ALPHAS[0]} 에서 {ALPHAS[-1]} 로 낮추면 집단 안 표준편차가 "
      f"{SD_IN[0]:.3f} -> {SD_IN[-1]:.3f} 로 준다 (이점).")
print(f"   같은 동안 위성 잡음이 뽑히는 비율은 "
      f"{P_SAT[0]:.3f} -> {P_SAT[-1]:.3f} 로 오른다 (대가).")
print(f"   홀로 선 잡음은 {P_LONE[0]:.3f} -> {P_LONE[-1]:.3f} 다 — "
      "상관되지 않은 것은 덜 끌려온다.")

# === 뽑는 개수를 맞추어 놓고 본다 ===================================
# alpha 를 낮추면 교차검증이 더 촘촘한 모형을 고른다. 그러면 「무엇이든」 더
# 뽑히므로 그룹 효과의 몫을 가려낼 수 없다. 뽑는 개수를 K 로 고정해 떼어 본다.
K_FIX = 6
COMP = {a: np.zeros(3) for a in ALPHAS}
for a in ALPHAS:
    cnt = np.zeros(3)
    rng = np.random.default_rng(42)
    for _ in range(REP):
        Zm, y = draw(rng)
        _, coefs, _ = enet_path(Zm, y, l1_ratio=a, n_alphas=120, eps=1e-3)
        nnz = (coefs != 0).sum(0)
        j = int(np.argmin(np.abs(nnz - K_FIX)))
        b = coefs[:, j]
        cnt += [(b[TRUE] != 0).sum(), (b[SAT] != 0).sum(),
                (b[LONE] != 0).sum()]
    COMP[a] = cnt / REP
print(f"\n② 뽑는 개수를 {K_FIX} 개로 맞추어 놓고, 그 여섯이 무엇인가")
print("   alpha   참 (셋 가운데)   위성 (셋 가운데)   홀로 선 잡음 (열넷 가운데)")
for a in ALPHAS:
    t, sa, lo = COMP[a]
    print(f"   {a:>5.2f}   {t:>14.2f}   {sa:>16.2f}   {lo:>23.2f}")
print("   개수를 묶어 놓으면 홀로 선 잡음은 밀려나고 **위성이 그 자리를 메운다** — "
      "이것이 그룹 효과의 몫이다.")

PREC = 3 * P_TRUE / np.maximum(3 * P_TRUE + NFP, 1e-9)
REC = P_TRUE
F1 = 2 * PREC * REC / np.maximum(PREC + REC, 1e-9)
i_best = int(np.argmax(F1))
print("\n③ 교차검증으로 고른 lambda 에서 받는 목록의 품질")
print("   alpha   정밀도   재현율      F1")
for i, a in enumerate(ALPHAS):
    mk = " ←" if i == i_best else ""
    print(f"   {a:>5.2f}   {PREC[i]:>6.3f}   {REC[i]:>6.3f}   {F1[i]:>6.3f}{mk}")
print(f"   재현율은 {REC.min():.3f}–{REC.max():.3f} 로 어디서나 거의 1 이다. "
      "참 셋은 alpha 와 무관하게 늘 뽑힌다.")
print(f"   그래서 F1 은 정밀도가 좌우하고, 가장 큰 곳이 "
      f"alpha = {ALPHAS[i_best]} 다 — **이 설계에서는 순수 라쏘다.**")
print("   그룹 효과가 갚아 주는 것은 목록이 아니라 계수의 안정성(①)이다.")

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.6, 5.2))
C_G, C_S, C_L = "#2ca02c", "#d62728", "#1f77b4"
A = np.array(ALPHAS)

ax1.plot(A, SD_IN, "o-", color=C_G, lw=2.8, ms=9)
ax1.invert_xaxis()
ax1.set_xlabel("$\\alpha$ (오른쪽으로 갈수록 능형 쪽)")
ax1.set_ylabel("참 집단 안 $\\hat\\beta$ 의 표준편차")
ax1.set_title("① 이점 — 상관된 것끼리 모인다", fontsize=12)
ax1.grid(alpha=0.25)

w = 0.26
xs = np.arange(len(ALPHAS))
TT = np.array([COMP[a][0] for a in ALPHAS])
SS = np.array([COMP[a][1] for a in ALPHAS])
LL = np.array([COMP[a][2] for a in ALPHAS])
ax2.bar(xs - w, TT, width=w, color=C_G, label="참 ($3$ 개 가운데)")
ax2.bar(xs, SS, width=w, color=C_S, label="위성 잡음 ($3$ 개 가운데)")
ax2.bar(xs + w, LL, width=w, color=C_L, label="홀로 선 잡음 ($14$ 개 가운데)")
ax2.set_xticks(xs)
ax2.set_xticklabels([f"{a:g}" for a in ALPHAS])
ax2.set_xlabel("$\\alpha$ (오른쪽으로 갈수록 능형 쪽)")
ax2.set_ylabel(f"뽑힌 ${K_FIX}$ 개 가운데 평균 몇 개")
ax2.set_title(f"② 개수를 ${K_FIX}$ 로 묶어 놓고 보면", fontsize=12)
ax2.legend(loc="upper center", fontsize=10)
ax2.set_ylim(0, 3.75)
ax2.grid(alpha=0.25, axis="y")

ax3.plot(A, PREC, "s-", color=C_S, lw=2.8, ms=9, label="정밀도")
ax3.plot(A, REC, "o-", color=C_G, lw=2.8, ms=9, label="재현율")
ax3.plot(A, F1, "d-", color="#9467bd", lw=3.0, ms=10, label="$F_1$")
ax3.plot([A[i_best]], [F1[i_best]], "d", color="#9467bd", ms=20,
         mfc="none", mew=3)
ax3.invert_xaxis()
ax3.set_xlabel("$\\alpha$ (오른쪽으로 갈수록 능형 쪽)")
ax3.set_ylabel("값")
ax3.set_ylim(0, 1.05)
ax3.set_title("③ 목록만 놓고 보면 그룹 효과는 손해다", fontsize=12)
ax3.legend(loc="upper right", fontsize=10.5)
ax3.grid(alpha=0.25)

fig.tight_layout()
out = "docs/ch18/elastic_net/img/grouping_cost.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
