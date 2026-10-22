r"""「자료 증강이 정칙화다」가 정확히 참이 되는 한 곳.

설명변수에 $N(0,\tau^2)$ 잡음을 더한 복제본으로 최소제곱을 풀면,
증강 복제본 수 $K \to \infty$ 에서

$$
\mathbb E\bigl[(X+E)^\top (X+E)\bigr] = X^\top X + n\tau^2 I
$$

이므로 해가 **능형 해 $(\lambda = n\tau^2)$ 로 간다.** 이 쪽의 연습문제가 보인
증강설계 $[\,X;\ \sqrt{\lambda_2}\,I\,]$ 가 결정적인 가짜 관측이라면,
잡음 증강은 **무작위 가짜 관측**이다.

① $K$ 가 커지면 증강 해가 능형 해로 간다.
② 되찾은 유효 $\lambda$ 가 $n\tau^2$ 과 맞는가.
③ 값 — 유한한 $K$ 는 능형보다 나쁘다. 그리고 **반응변수**에 잡음을
   더하면 아무 일도 일어나지 않는다.

만드는 파일: docs/ch18/overview/img/noise_augmentation.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

RNG = np.random.default_rng(4)
N, P, SIG, NTE = 120, 30, 1.0, 8000
TAU = 0.12          # n*tau^2 가 이 문제의 최적 lambda 근처가 되도록
KS = [1, 2, 5, 10, 20, 50, 100, 500, 2000]
TAUS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8]

V = np.linalg.qr(RNG.normal(0, 1, (P, P)))[0]
dd = np.logspace(0.4, -0.9, P)
X = RNG.normal(0, 1, (N, P)) @ V.T @ np.diag(dd) @ V
beta = RNG.normal(0, 1, P)
y = X @ beta + RNG.normal(0, SIG, N)
Xte = RNG.normal(0, 1, (NTE, P)) @ V.T @ np.diag(dd) @ V
yte = Xte @ beta + RNG.normal(0, SIG, NTE)

G, c = X.T @ X, X.T @ y
ridge = lambda lam: np.linalg.solve(G + lam * np.eye(P), c)   # noqa: E731
rmse = lambda b: float(np.sqrt(np.mean((Xte @ b - yte) ** 2)))  # noqa: E731
rel = lambda a, r: float(np.linalg.norm(a - r) / np.linalg.norm(r))  # noqa


def aug_x(tau, K, seed=0):
    """설명변수에 잡음을 더한 복제본 K 개로 푼 최소제곱."""
    r = np.random.default_rng(seed)
    A = np.zeros((P, P))
    v = np.zeros(P)
    for _ in range(K):
        Xk = X + r.normal(0, tau, (N, P))
        A += Xk.T @ Xk
        v += Xk.T @ y
    return np.linalg.solve(A / K, v / K)


def aug_y(sd, K, seed=1):
    """반응변수에 잡음을 더한 복제본 K 개 — 대조군."""
    r = np.random.default_rng(seed)
    v = np.zeros(P)
    for _ in range(K):
        v += X.T @ (y + r.normal(0, sd, N))
    return np.linalg.solve(G, v / K)


LAM = N * TAU ** 2
RG = ridge(LAM)
OLS = np.linalg.solve(G, c)
print(f"n = {N}, p = {P}, tau = {TAU}  ->  n*tau^2 = {LAM}")
print(f"\n① 복제본 수 K 에 따라")
print(f"{'K':>6}{'능형과의 상대거리':>16}{'시험 RMSE':>11}")
d_k, e_k = [], []
for K in KS:
    a = aug_x(TAU, K)
    d_k.append(rel(a, RG))
    e_k.append(rmse(a))
    print(f"{K:>6}{d_k[-1]:>16.4f}{e_k[-1]:>11.4f}")
print(f"   능형 (lambda = {LAM:g})  시험 RMSE {rmse(RG):.4f}")
print(f"   최소제곱             시험 RMSE {rmse(OLS):.4f}")

print(f"\n② 되찾은 유효 lambda (K = 2000)")
print(f"{'tau':>7}{'n*tau^2':>10}{'맞춘 lambda':>13}{'비':>7}")
LGRID = np.logspace(-3, 3.5, 500)
fit_l = []
for tau in TAUS:
    a = aug_x(tau, 2000, seed=7)
    j = int(np.argmin([np.linalg.norm(ridge(l) - a) for l in LGRID]))
    fit_l.append(float(LGRID[j]))
    print(f"{tau:>7.2f}{N * tau ** 2:>10.3f}{fit_l[-1]:>13.3f}"
          f"{fit_l[-1] / (N * tau ** 2):>7.3f}")

ay = aug_y(SIG, 2000)
print(f"\n③ 반응변수에 잡음을 더하면 (K = 2000)")
print(f"   최소제곱과의 상대거리 {rel(ay, OLS):.5f}  "
      f"— 0 이면 정칙화가 전혀 일어나지 않은 것이다")
print(f"   시험 RMSE {rmse(ay):.4f} (최소제곱 {rmse(OLS):.4f})")

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.6, 5.0))
C_A, C_R, C_O = "#d62728", "#1f77b4", "#8c564b"

ax1.loglog(KS, d_k, "o-", color=C_A, lw=2.6, ms=7)
ax1.loglog(KS, [d_k[0] / np.sqrt(k) for k in KS], ":", color="#555555",
           lw=1.8, label="$K^{-1/2}$ 기울기")
ax1.set_xticks([1, 10, 100, 1000])
ax1.set_xticklabels(["$1$", "$10$", "$10^{2}$", "$10^{3}$"], fontsize=9.5)
ax1.set_yticks([0.003, 0.01, 0.03, 0.1, 0.3])
ax1.set_yticklabels(["$0.003$", "$0.01$", "$0.03$", "$0.1$", "$0.3$"])
ax1.minorticks_off()
ax1.set_xlabel("증강 복제본 수 $K$")
ax1.set_ylabel(f"능형 해 $(\\lambda = {LAM:g})$ 와의 상대거리")
ax1.set_title("① 잡음을 더해 푼 것이 능형으로 간다", fontsize=12)
ax1.legend(loc="lower left", fontsize=10)
ax1.grid(alpha=0.25)

ax2.loglog([N * t ** 2 for t in TAUS], fit_l, "o", color=C_A, ms=11,
           zorder=4, label="맞춘 유효 $\\lambda$")
lo, hi = 0.3, 250
ax2.loglog([lo, hi], [lo, hi], "-", color="#555555", lw=1.8,
           label="$\\lambda = n\\tau^2$")
for k, (t, f) in enumerate(zip(TAUS, fit_l)):
    off = (0.8, 0.5) if k == 0 else (0.7, -1.1)   # 첫 점은 왼쪽 아래 모서리에 붙는다
    ax2.annotate(f"$\\tau = {t}$", (N * t ** 2, f), xytext=off,
                 textcoords="offset fontsize", fontsize=9.5)
ax2.set_xticks([0.5, 2, 8, 50, 200])
ax2.set_xticklabels(["$0.5$", "$2$", "$8$", "$50$", "$200$"], fontsize=9.5)
ax2.set_yticks([0.5, 2, 8, 50, 200])
ax2.set_yticklabels(["$0.5$", "$2$", "$8$", "$50$", "$200$"], fontsize=9.5)
ax2.minorticks_off()
ax2.set_xlim(lo, hi)
ax2.set_ylim(lo, hi)
ax2.set_xlabel("$n\\tau^2$")
ax2.set_ylabel("증강 해에 맞춘 $\\lambda$")
ax2.set_title("② 눈금이 정확히 $n\\tau^2$ 이다", fontsize=12)
ax2.legend(loc="upper left", fontsize=10)
ax2.grid(alpha=0.25)

ax3.semilogx(KS, e_k, "o-", color=C_A, lw=2.6, ms=7,
             label="설명변수에 잡음 ($K$ 개)")
ax3.axhline(rmse(RG), color=C_R, lw=2.0, ls="--",
            label=f"능형 $(\\lambda = {LAM:g})$")
ax3.axhline(rmse(OLS), color=C_O, lw=2.0, ls=":", label="최소제곱")
ax3.plot([2000], [rmse(ay)], "s", color=C_O, ms=12, zorder=4)
ax3.annotate("반응변수에 잡음을 더하면\n최소제곱에 그대로 머문다",
             xy=(2000, rmse(ay)), xytext=(2.4, rmse(OLS) * 0.986),
             textcoords="data", fontsize=10.5, color=C_O, ha="left",
             arrowprops=dict(arrowstyle="->", color=C_O, lw=1.1))
ax3.annotate(f"$K = 1$ 이면 오히려 나쁘다", xy=(1, e_k[0]),
             xytext=(3.2, e_k[0] * 0.999), textcoords="data", fontsize=10.5,
             color=C_A, ha="left",
             arrowprops=dict(arrowstyle="->", color=C_A, lw=1.1))
ax3.set_xticks([1, 10, 100, 1000])
ax3.set_xticklabels(["$1$", "$10$", "$10^{2}$", "$10^{3}$"], fontsize=9.5)
ax3.minorticks_off()
ax3.set_xlabel("증강 복제본 수 $K$")
ax3.set_ylabel("시험 RMSE")
ax3.set_title("③ 공짜는 아니다", fontsize=12)
ax3.legend(loc="lower right", fontsize=9.5, framealpha=0.95)
ax3.grid(alpha=0.25)

fig.tight_layout()
out = "docs/ch18/overview/img/noise_augmentation.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
