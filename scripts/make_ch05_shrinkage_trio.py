r"""분모를 고르는 일은 수축을 고르는 일이다 — 같은 거래가 나오는 세 자리.

이 쪽은 $\sum(X_i-\bar X)^2$ 를 무엇으로 나눌지 묻고 $n+1$ 을 답으로 얻는다.
$\lambda S^2$ 로 다시 쓰면 그 답이 **수축계수**가 된다. 같은 꼴의 물음이 분류
확률을 추정하는 자리에도 그대로 나온다 — 그리고 거기서는 **라벨 평활**이라 불린다.

① 분모를 고르는 일 = $\lambda$ 를 고르는 일. 최적 $\lambda$ 는 $1$ 보다 작다.
② 확률벡터에서도 같다 — 더해 주는 $\alpha$ 의 최적값이 $0$ 이 아니다.
③ 섞이는 몫은 $1/n$ 로 사라진다 — 라벨 평활은 그것을 못 박는 일이다.

만드는 파일: docs/ch05/variance/img/shrinkage_trio.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

RNG = np.random.default_rng(0)

# === ① 분모 = 수축계수 ===============================================
N1, REP = 10, 400_000
s2 = RNG.standard_normal((REP, N1)).var(axis=1, ddof=1)      # sigma^2 = 1
LAM = np.linspace(0.5, 1.25, 151)
MSE = [float(np.mean((l * s2 - 1.0) ** 2)) for l in LAM]
best = float(LAM[int(np.argmin(MSE))])
print(f"① n = {N1} 에서 lambda * S^2 의 MSE (반복 {REP:,})")
for nm, l in (("불편 (c = n-1)", 1.0), ("MLE (c = n)", (N1 - 1) / N1),
              ("최적 (c = n+1)", (N1 - 1) / (N1 + 1))):
    m = float(np.mean((l * s2 - 1.0) ** 2))
    print(f"   {nm:>16} : lambda = {l:.4f},  MSE = {m:.5f}")
print(f"   모의로 찾은 최솟값 lambda = {best:.4f}, "
      f"식이 주는 (n-1)/(n+1) = {(N1 - 1) / (N1 + 1):.4f}")
print("   **분모를 고르는 일은 S^2 을 얼마나 줄일지 고르는 일이다.**")

# === ② 확률벡터에서 같은 물음 ========================================
K2, N2, REP2 = 10, 20, 200_000
P2 = RNG.dirichlet(np.ones(K2) * 0.7)
ALPHAS = np.linspace(0, 3, 61)
cnt = RNG.multinomial(N2, P2, REP2)
MSE2 = []
for a in ALPHAS:
    ph = (cnt + a) / (N2 + K2 * a)
    MSE2.append(float(np.mean(((ph - P2) ** 2).sum(axis=1))))
a_best = float(ALPHAS[int(np.argmin(MSE2))])
print(f"\n② class {K2} 개, n = {N2} 에서 (x + alpha)/(n + K alpha) 의 MSE")
print(f"{'alpha':>8}{'MSE':>12}")
for a in (0.0, 0.5, a_best, 1.0, 2.0):
    i = int(np.argmin(np.abs(ALPHAS - a)))
    tag = "  <- 최솟값" if abs(ALPHAS[i] - a_best) < 1e-9 else ""
    print(f"{ALPHAS[i]:>8.2f}{MSE2[i]:>12.6f}{tag}")
print(f"   최적 alpha = {a_best:.2f} 로 **0 이 아니다.** "
      f"MLE(alpha = 0)보다 MSE 가 {100 * (1 - min(MSE2) / MSE2[0]):.1f}% 낮다.")

# === ③ n 이 커지면 ==================================================
NS = [5, 10, 20, 50, 100, 200, 500]
KS = [2, 10, 100]
REP3 = 120_000
PFIX = {k: RNG.dirichlet(np.ones(k) * 0.7) for k in KS}   # K 마다 p 를 고정한다
print(f"\n③ n 을 키우며 — p 는 K 마다 **고정**해 둔다")
print(f"{'n':>6}" + "".join(f"{f'K={k}: a*  수축':>20}" for k in KS)
      + f"{'분모 쪽 수축':>14}")
AB = {k: [] for k in KS}
RS = {k: [] for k in KS}
SHR = []
aa = np.linspace(0, 20, 101)   # K = 2 의 최적값이 끝에 걸리지 않게 넓게 둔다
for n in NS:
    cells = []
    for k in KS:
        p = PFIX[k]
        c = RNG.multinomial(n, p, REP3)
        m = [float(np.mean((((c + a) / (n + k * a) - p) ** 2).sum(1)))
             for a in aa]
        astar = float(aa[int(np.argmin(m))])
        AB[k].append(astar)
        RS[k].append(k * astar / (n + k * astar))
        cells.append(f"{astar:>8.2f}{RS[k][-1]:>12.4f}")
    SHR.append(1 - (n - 1) / (n + 1))
    print(f"{n:>6}" + "".join(f"{c:>20}" for c in cells)
          + f"{SHR[-1]:>14.4f}")
print("   최적 alpha 자체는 0 으로 가지 않는다 — p 가 정하는 값 둘레에 머문다.")
print("   그런데 **실제로 섞이는 몫** K alpha/(n + K alpha) 는 n 이 K alpha 를 "
      "넘어서면 1/n 로 사라진다. 분모 쪽 2/(n+1) 과 같은 속도다.")
print("   K = 100 쪽이 늦게 꺾이는 것은 K alpha = 80 이라 n 이 그것을 넘는 데 "
      "시간이 걸리기 때문이다.")
print("   라벨 평활 eps 를 K class 에 고르게 뿌리는 일이 바로 그 '섞이는 몫'을 "
      "eps 로 못 박는 것이다 — n 이 늘어도 줄지 않게.")

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.6, 5.2))
C1, C2, C3 = "#1f77b4", "#d62728", "#2ca02c"

ax1.plot(LAM, MSE, color=C1, lw=2.8)
for nm, l, c in (("불편\n$c=n-1$", 1.0, "#333333"),
                 ("MLE\n$c=n$", (N1 - 1) / N1, C3),
                 ("최적\n$c=n+1$", (N1 - 1) / (N1 + 1), C2)):
    m = float(np.mean((l * s2 - 1.0) ** 2))
    ax1.plot([l], [m], "o", color=c, ms=11, zorder=5)
    ax1.annotate(nm, (l, m), xytext=(0, 1.0), textcoords="offset fontsize",
                 ha="center", fontsize=10, color=c)
ax1.set_xlabel("수축계수 $\\lambda$  (추정량은 $\\lambda S^2$)")
ax1.set_ylabel("평균제곱오차")
ax1.set_ylim(min(MSE) * 0.93, min(MSE) * 1.9)
ax1.set_title(f"① 분모를 고르는 일 = $\\lambda$ 를 고르는 일 ($n = {N1}$)",
              fontsize=12)
ax1.grid(alpha=0.25)

ax2.plot(ALPHAS, MSE2, color=C1, lw=2.8)
ax2.plot([0], [MSE2[0]], "o", color="#333333", ms=11, zorder=5)
ax2.annotate("MLE\n$\\alpha = 0$", (0, MSE2[0]), xytext=(0.6, 0.2),
             textcoords="offset fontsize", fontsize=10, color="#333333")
ax2.plot([a_best], [min(MSE2)], "o", color=C2, ms=11, zorder=5)
ax2.annotate(f"최적 $\\alpha = {a_best:.2f}$", (a_best, min(MSE2)),
             xytext=(0.5, -1.6), textcoords="offset fontsize", fontsize=10,
             color=C2)
ax2.set_xlabel("더해 주는 $\\alpha$")
ax2.set_ylabel("$\\sum_j(\\hat p_j - p_j)^2$ 의 평균")
ax2.set_title(f"② 확률벡터에서도 같다 ($K = {K2}$, $n = {N2}$)", fontsize=12)
ax2.grid(alpha=0.25)

for i, k in enumerate(KS):
    ax3.loglog(NS, RS[k], "o-", color=["#9ecae1", "#4292c6", "#08519c"][i],
               lw=2.4, ms=8, label=f"$K = {k}$ 에서 섞이는 몫")
ax3.loglog(NS, SHR, "s--", color=C2, lw=2.4, ms=8,
           label="분모 쪽 $1-\\lambda^* = 2/(n+1)$")
ax3.set_xlabel("표본크기 $n$")
ax3.set_ylabel("실제로 섞이는 몫")
ax3.set_xticks([10, 100])
ax3.set_xticklabels(["$10$", "$10^{2}$"])
ax3.set_yticks([0.01, 0.1, 1])
ax3.set_yticklabels(["$0.01$", "$0.1$", "$1$"])
ax3.minorticks_off()
ax3.set_title("③ 둘 다 $1/n$ 로 사라진다", fontsize=12)
ax3.legend(loc="lower left", fontsize=9.5)
ax3.grid(alpha=0.25)

fig.tight_layout()
out = "docs/ch05/variance/img/shrinkage_trio.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
