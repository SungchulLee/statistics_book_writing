r"""이 쪽의 분산 공식이 서지 않는 자리 — 보간 문턱과 두 번째 내리막.

이 쪽은 OLS 추정량의 표집분포를 세우고 Var(beta_hat) = sigma^2 (X'X)^{-1} 을 얻는다.
그 식에는 조건이 하나 숨어 있다 — **X'X 가 역행렬을 가져야 한다**, 곧 p < n 이어야
한다. 요즘 모형은 그 조건을 예사로 넘긴다.

p 를 1 에서 300 까지 키우며(n 은 60 으로 고정) 시험오차를 잰다. p <= n 이면 OLS,
p > n 이면 최소노름 해(유사역행렬)를 쓴다.

① 시험오차 — p = n 에서 치솟고 그 너머에서 다시 내려간다.
② 왜 치솟는가 — X'X 의 가장 작은 고윳값이 0 으로 간다.
③ 왜 다시 내려가는가 — 최소노름 해가 계수의 크기를 묶는다.

만드는 파일: docs/ch00/linalg_regression/linalg_statistics/img/double_descent.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"

RNG = np.random.default_rng(0)
N_TR, N_TE = 60, 4000
D = 320                      # 숨은 특성의 전체 수
SIGMA = 0.5
REP = 40

# 참 계수 — 뒤로 갈수록 작아진다 (대부분의 신호가 앞쪽에 있다)
BETA = RNG.normal(0, 1, D) / np.sqrt(np.arange(1, D + 1))
PS = sorted(set([1, 2, 3, 5, 8, 12, 18, 25, 35, 45, 52, 56, 58, 59, 60,
                 61, 62, 64, 68, 75, 85, 100, 120, 150, 190, 240, 320]))


def one_run(seed):
    r = np.random.default_rng(seed)
    Xtr = r.normal(0, 1, (N_TR, D))
    Xte = r.normal(0, 1, (N_TE, D))
    ytr = Xtr @ BETA + r.normal(0, SIGMA, N_TR)
    yte = Xte @ BETA + r.normal(0, SIGMA, N_TE)
    err, nrm, lmin = [], [], []
    for p in PS:
        A, B = Xtr[:, :p], Xte[:, :p]
        # p <= n 이면 OLS, p > n 이면 최소노름 해. lstsq 가 둘 다 준다.
        beta = np.linalg.lstsq(A, ytr, rcond=None)[0]
        err.append(float(np.mean((B @ beta - yte) ** 2)))
        nrm.append(float(np.linalg.norm(beta)))
        ev = np.linalg.eigvalsh(A.T @ A) if p <= N_TR else np.linalg.eigvalsh(A @ A.T)
        lmin.append(float(max(ev.min(), 1e-12)))
    return np.array(err), np.array(nrm), np.array(lmin)


E, Nm, Lm = [], [], []
for s in range(REP):
    e, nn, ll = one_run(100 + s)
    E.append(e); Nm.append(nn); Lm.append(ll)
E = np.median(np.array(E), axis=0)
Nm = np.median(np.array(Nm), axis=0)
Lm = np.median(np.array(Lm), axis=0)

idx_n = PS.index(N_TR)
best_lo = int(np.argmin(E[:idx_n]))
best_hi = idx_n + 1 + int(np.argmin(E[idx_n + 1:]))
print(f"훈련 {N_TR} 개, 시험 {N_TE} 개, 숨은 특성 {D} 개, sigma = {SIGMA}, "
      f"되풀이 {REP}")
# 한글은 고정폭에서 두 칸을 차지하므로 자리맞춤을 직접 적는다
print("\n     p    p/n    시험오차   beta 노름   X'X 최소고윳값")
for i, p in enumerate(PS):
    mark = ""
    if p == N_TR:
        mark = "   <- 보간 문턱 (p = n)"
    elif i == best_lo:
        mark = "   <- 고전적 최적"
    elif i == best_hi:
        mark = "   <- 문턱 너머 최적"
    print(f"   {p:>4}  {p/N_TR:>5.2f}  {E[i]:>10.3f}  {Nm[i]:>10.3f}  "
          f"{Lm[i]:>14.3e}{mark}")

print(f"\n   고전적 최적은 p = {PS[best_lo]} 에서 시험오차 {E[best_lo]:.3f} 다.")
print(f"   보간 문턱 p = n = {N_TR} 에서 {E[idx_n]:.1f} 로 치솟는다 "
      f"({E[idx_n]/E[best_lo]:.0f} 배).")
print(f"   그런데 문턱을 넘어가면 다시 내려가 p = {PS[best_hi]} 에서 "
      f"{E[best_hi]:.3f} 가 된다.")
if E[best_hi] < E[best_lo]:
    print(f"   **문턱 너머가 고전적 최적보다 낫다** "
          f"({E[best_hi]:.3f} < {E[best_lo]:.3f}).")
else:
    print(f"   여기서는 고전적 최적이 아직 더 낫다 "
          f"({E[best_lo]:.3f} < {E[best_hi]:.3f}) — 두 번째 내리막은 분명하지만")
    print(f"   첫 골짜기만큼 내려오지는 않았다.")

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.6, 5.2))
C_A, C_B, C_C = "#d62728", "#1f77b4", "#2ca02c"
x = np.array(PS) / N_TR

ax1.semilogy(x, E, "o-", color=C_B, lw=2.6, ms=7)
ax1.axvline(1.0, color=C_A, lw=2.2, ls="--")
ax1.text(1.06, E.max() * 0.45, "보간 문턱\n$p = n$", fontsize=10.5, color=C_A)
ax1.plot([x[best_lo]], [E[best_lo]], "o", color=C_C, ms=14, zorder=5)
ax1.annotate("고전적 최적", (x[best_lo], E[best_lo]), xytext=(0.3, 1.1),
             textcoords="offset fontsize", fontsize=10.5, color=C_C)
ax1.plot([x[best_hi]], [E[best_hi]], "o", color="#9467bd", ms=14, zorder=5)
ax1.annotate("문턱 너머", (x[best_hi], E[best_hi]), xytext=(0.2, 1.0),
             textcoords="offset fontsize", fontsize=10.5, color="#9467bd")
ax1.set_xlabel("$p / n$  (계수 수 / 표본 수)")
ax1.set_ylabel("시험 평균제곱오차")
ax1.set_title("① 봉우리 하나와 내리막 둘", fontsize=12)
ax1.grid(alpha=0.25)

ax2.semilogy(x, Lm, "o-", color=C_A, lw=2.6, ms=7)
ax2.axvline(1.0, color=C_A, lw=2.2, ls="--")
ax2.set_xlabel("$p / n$")
ax2.set_ylabel("$X^T X$ 의 가장 작은 고윳값")
# 로그 눈금의 음수 지수는 mathtext 라 한글 글꼴에서 두부가 된다. 직접 적는다
ax2.set_yticks([0.01, 0.1, 1, 10, 100])
ax2.set_yticklabels(["$0.01$", "$0.1$", "$1$", "$10$", "$100$"])
ax2.minorticks_off()
ax2.set_title("② 역행렬이 서지 못하는 자리", fontsize=12)
ax2.grid(alpha=0.25)

ax3.plot(x, Nm, "o-", color="#9467bd", lw=2.6, ms=7)
ax3.axvline(1.0, color=C_A, lw=2.2, ls="--")
ax3.text(1.06, Nm.max() * 0.82, "보간 문턱", fontsize=10.5, color=C_A)
ax3.set_xlabel("$p / n$")
ax3.set_ylabel("계수 벡터의 길이  $\\|\\hat\\beta\\|$")
ax3.set_title("③ 계수의 크기", fontsize=12)
ax3.grid(alpha=0.25)

fig.tight_layout()
out = "docs/ch00/linalg_regression/linalg_statistics/img/double_descent.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
