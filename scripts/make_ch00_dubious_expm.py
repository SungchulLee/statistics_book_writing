r"""쪽에 적힌 공식이 「미심쩍은 열아홉 가지」 가운데 하나다.

이 쪽의 절 2 는 e^A = P diag(e^lambda) P^{-1} 을 적는다. 몰러와 밴 론은 이것을
방법 14 「고유벡터」로 세고, 행렬이 결손(defective)이면 **아무 경고 없이 완전히
실패한다**고 적었다.

① A(eps) = [[1+eps, 1], [0, 1-eps]] 에서 eps 를 0 으로 보내며 — 고유벡터 공식의
   상대오차와 cond(V).
② 같은 고윳값을 가진 대칭행렬에서는 오차가 기계정밀도에 머문다.
③ 무작위 행렬에서 상대오차 대 cond(V) — u * cond(V) 선 위에 얹힌다.

만드는 파일: docs/ch00/linalg_regression/square_matrices/img/dubious_expm.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"

from scipy.linalg import expm   # noqa: E402

RNG = np.random.default_rng(0)
U = np.finfo(float).eps / 2          # 단위 반올림 ~ 1.11e-16


def eig_expm(A):
    """이 쪽의 절 2 가 적는 공식 그대로 — 몰러·밴 론의 방법 14."""
    lam, V = np.linalg.eig(A)
    return V @ np.diag(np.exp(lam)) @ np.linalg.inv(V)


def cond_V(A):
    V = np.linalg.eig(A)[1]
    return float(np.linalg.cond(V))


def relerr(Ahat, Atrue):
    return float(np.linalg.norm(Ahat - Atrue, 2) / np.linalg.norm(Atrue, 2))


def A_eps(e):
    return np.array([[1.0 + e, 1.0], [0.0, 1.0 - e]])


# 같은 고윳값 {1+eps, 1-eps} 를 가진 대칭행렬
TH = 0.7
Q = np.array([[np.cos(TH), -np.sin(TH)], [np.sin(TH), np.cos(TH)]])


def A_sym(e):
    return Q @ np.diag([1.0 + e, 1.0 - e]) @ Q.T


EPS = np.logspace(-1, -16, 46)
E_NON = np.array([relerr(eig_expm(A_eps(e)), expm(A_eps(e))) for e in EPS])
E_SYM = np.array([relerr(eig_expm(A_sym(e)), expm(A_sym(e))) for e in EPS])
K_NON = np.array([cond_V(A_eps(e)) for e in EPS])
K_SYM = np.array([cond_V(A_sym(e)) for e in EPS])

# 한글은 고정폭에서 두 칸을 차지하므로 자리맞춤을 직접 적는다
print("① A(eps) = [[1+eps, 1], [0, 1-eps]] — 고윳값은 1+eps 와 1-eps 다")
print("      eps      cond(V)     비대칭 오차    대칭 오차")
for e in (1e-1, 1e-3, 1e-5, 1e-7, 1e-9, 1e-11, 1e-13, 1e-15):
    i = int(np.argmin(np.abs(EPS - e)))
    print(f"   {EPS[i]:>8.1e}   {K_NON[i]:>10.3e}   {E_NON[i]:>11.3e}   "
          f"{E_SYM[i]:>10.3e}")
i_last = len(EPS) - 1
print(f"   오차는 cond(V) 를 따라 커진다 — eps = {EPS[i_last]:.1e} 에서 "
      f"cond(V) = {K_NON[i_last]:.2e}, 오차 {E_NON[i_last]:.2e} 다.")
print(f"   같은 고윳값의 대칭행렬에서는 끝까지 {E_SYM.max():.2e} 를 넘지 않는다.")

# === ② 아예 결손인 자리 =============================================
D = np.array([[1.0, 1.0], [0.0, 1.0]])      # 몰러·밴 론이 든 결손 행렬
lam, V = np.linalg.eig(D)
print("\n② eps = 0 — 결손행렬 [[1, 1], [0, 1]]")
print(f"   numpy 가 내놓는 고윳값   {lam}")
print(f"   고유벡터행렬 V 의 두 열  {V[:, 0]} 과 {V[:, 1]}")
print(f"   cond(V) = {np.linalg.cond(V):.3e}  ← 사실상 무한")
GOT, TRUE = eig_expm(D), expm(D)
print(f"   공식이 내놓는 것   {GOT.ravel().round(6)}")
print(f"   참값               {TRUE.ravel().round(6)}")
print(f"   상대오차 {relerr(GOT, TRUE):.3e}")
print("   **예외도 경고도 없다. 그럴듯한 수가 그냥 나온다.**")

# === ③ 무작위 행렬 ==================================================
NREP, N = 400, 6
KS, ES, SYMFLAG = [], [], []
for _ in range(NREP):
    lam = RNG.uniform(-1.0, 1.0, N)
    # 고유벡터를 점점 평행하게 — 쏠린 분포에서 P 를 뽑는다
    P = RNG.normal(size=(N, N)) + RNG.uniform(0, 1) * np.ones((N, N))
    A = P @ np.diag(lam) @ np.linalg.inv(P)
    KS.append(cond_V(A))
    ES.append(max(relerr(eig_expm(A), expm(A)), 1e-18))
    SYMFLAG.append(False)
    S = RNG.normal(size=(N, N))
    S = (S + S.T) / 2
    KS.append(cond_V(S))
    ES.append(max(relerr(eig_expm(S), expm(S)), 1e-18))
    SYMFLAG.append(True)
KS, ES, SYMFLAG = np.array(KS), np.array(ES), np.array(SYMFLAG)
print(f"\n③ {N}x{N} 행렬 {NREP}+{NREP} 개")
print("   갈래          cond(V) 중앙값   상대오차 중앙값   최악 오차")
for name, m in (("대칭 아님", ~SYMFLAG), ("대칭", SYMFLAG)):
    print(f"   {name:<10}   {np.median(KS[m]):>13.3e}   "
          f"{np.median(ES[m]):>14.3e}   {ES[m].max():>9.3e}")
print(f"   대칭 쪽은 cond(V) 가 {KS[SYMFLAG].max():.4f} 를 넘지 않는다 — "
      "직교로 잡히기 때문이다.")

# === 그림 ============================================================
def logticks(ax, which, lo, hi, step=3):
    """로그 축의 음수 지수는 mathtext 를 거치므로 unicode_minus 로 막지 못한다.
    눈금과 글자를 직접 적는다."""
    e = list(range(lo, hi + 1, step))
    t = [10.0 ** k for k in e]
    lab = [f"$10^{{{k}}}$" for k in e]
    if which == "x":
        ax.set_xticks(t); ax.set_xticklabels(lab)
    else:
        ax.set_yticks(t); ax.set_yticklabels(lab)
    ax.minorticks_off()


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.6, 5.2))
C_N, C_S = "#d62728", "#2ca02c"

ax1.loglog(EPS, np.maximum(E_NON, 1e-18), "o-", color=C_N, lw=2.6, ms=5,
           label="대칭 아님")
ax1.loglog(EPS, np.maximum(E_SYM, 1e-18), "s-", color=C_S, lw=2.6, ms=5,
           label="같은 고윳값의 대칭행렬")
ax1.loglog(EPS, U * K_NON, "--", color="#333333", lw=2.0,
           label="$u \\cdot \\mathrm{cond}(V)$")
ax1.invert_xaxis()
ax1.set_xlabel("$\\varepsilon$ (오른쪽으로 갈수록 결손에 가깝다)")
# matplotlib mathtext 는 \lVert 를 모른다 — \| 를 쓴다
ax1.set_ylabel("$\\|\\hat{E}-e^{A}\\|_2 \\,/\\, \\|e^{A}\\|_2$")
ax1.set_title("① 고유벡터 공식의 상대오차", fontsize=12)
ax1.legend(loc="lower left", fontsize=10)
ax1.grid(alpha=0.25)
ax1.set_xlim(2e-1, 5e-17)
ax1.set_ylim(1e-17, 5e0)
logticks(ax1, "x", -16, -1)
logticks(ax1, "y", -16, 0, 4)

ax2.loglog(EPS, K_NON, "o-", color=C_N, lw=2.6, ms=5, label="대칭 아님")
ax2.loglog(EPS, K_SYM, "s-", color=C_S, lw=2.6, ms=5, label="대칭")
ax2.invert_xaxis()
ax2.set_xlabel("$\\varepsilon$")
ax2.set_ylabel("$\\mathrm{cond}(V)$")
ax2.set_title("② 고유벡터행렬의 조건수", fontsize=12)
ax2.legend(loc="lower left", fontsize=10)
ax2.grid(alpha=0.25)
ax2.set_xlim(2e-1, 5e-17)
ax2.set_ylim(2e-1, 5e16)
logticks(ax2, "x", -16, -1)
logticks(ax2, "y", 0, 16, 4)

ax3.loglog(KS[~SYMFLAG], ES[~SYMFLAG], "o", color=C_N, ms=5, alpha=0.45,
           label="대칭 아님")
ax3.loglog(KS[SYMFLAG], ES[SYMFLAG], "s", color=C_S, ms=5, alpha=0.6,
           label="대칭")
kk = np.logspace(0, np.log10(KS.max()), 50)
ax3.loglog(kk, U * kk, "--", color="#333333", lw=2.2,
           label="$u \\cdot \\mathrm{cond}(V)$")
ax3.set_xlabel("$\\mathrm{cond}(V)$")
ax3.set_ylabel("상대오차")
ax3.set_title(f"③ 무작위 ${N}\\times{N}$ 행렬 {2*NREP} 개", fontsize=12)
ax3.legend(loc="upper left", fontsize=10)
ax3.grid(alpha=0.25)
ax3.set_ylim(1e-17, 1e-8)
logticks(ax3, "x", 0, 3, 1)
logticks(ax3, "y", -16, -8, 2)

fig.tight_layout()
out = "docs/ch00/linalg_regression/square_matrices/img/dubious_expm.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
