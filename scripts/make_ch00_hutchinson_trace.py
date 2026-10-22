r"""행렬을 만들지 않고 대각합을 잰다 — $E[\mathbf{z}^\top\mathbf{A}\mathbf{z}]
= \operatorname{tr}(\mathbf{A})$ 를 거꾸로 읽기.

이 쪽 $\S 5$ 는 평균 $\mathbf{0}$·공분산 $\mathbf{I}$ 인 $\mathbf{z}$ 에 대해
$E[\mathbf{z}^\top\mathbf{A}\mathbf{z}] = \operatorname{tr}(\mathbf{A})$ 라고 적는다.
오른쪽에서 왼쪽으로 읽으면 **추정량**이 된다. $\mathbf{A}$ 를 볼 수 없어도
$\mathbf{A}\mathbf{z}$ 만 계산할 수 있으면 대각합을 불편으로 잴 수 있다.

① 탐침 수 $m$ 을 늘릴 때의 오차 — 가우스 탐침과 라데마허(±1) 탐침.
② 분산 공식을 확인한다. 가우스는 $2\lVert A\rVert_F^2$, 라데마허는
   $2(\lVert A\rVert_F^2 - \sum a_{ii}^2)$ — 대각이 무거울수록 ±1 이 이긴다.
③ 상대오차는 $n$ 과 무관하다. $\mathbf{A} = \mathbf{B}\mathbf{B}^\top$ 를
   만들지 않고 $\mathbf{B}^\top\mathbf{z}$ 만 써서 잰다.

만드는 파일: docs/ch00/linalg_regression/square_matrices/img/hutchinson_trace.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

RNG = np.random.default_rng(0)


def sym_psd(n, rng, k=None):
    """고윳값이 퍼져 있는 대칭 양정부호 행렬."""
    k = k or n
    b = rng.standard_normal((n, k)) / np.sqrt(k)
    return b @ b.T


def probes(kind, shape, rng):
    if kind == "가우스":
        return rng.standard_normal(shape)
    return rng.integers(0, 2, shape) * 2.0 - 1.0          # 라데마허 ±1


# === ① 탐침 수에 따른 오차 ===========================================
N1, REP1 = 400, 400
A1 = sym_psd(N1, RNG)
TR1 = float(np.trace(A1))
FRO1 = float((A1 ** 2).sum())
DIAG1 = float((np.diag(A1) ** 2).sum())
VAR1 = {"가우스": 2 * FRO1, "라데마허": 2 * (FRO1 - DIAG1)}
MS = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000])
print(f"① n = {N1}, tr(A) = {TR1:.4f}, "
      f"||A||_F^2 = {FRO1:.4f}, sum a_ii^2 = {DIAG1:.4f}")
print(f"   한 번의 분산 — 가우스 {VAR1['가우스']:.4f}, "
      f"라데마허 {VAR1['라데마허']:.4f}")
print(f"{'m':>7}{'가우스 RMSE':>14}{'이론':>12}{'라데마허 RMSE':>16}{'이론':>12}")
RM = {"가우스": [], "라데마허": []}
for m in MS:
    for kind in RM:
        z = probes(kind, (REP1, m, N1), RNG)
        est = np.einsum("rmi,ij,rmj->rm", z, A1, z, optimize=True).mean(axis=1)
        RM[kind].append(float(np.sqrt(np.mean((est - TR1) ** 2))))
    print(f"{m:>7}{RM['가우스'][-1]:>14.4f}"
          f"{np.sqrt(VAR1['가우스'] / m):>12.4f}"
          f"{RM['라데마허'][-1]:>16.4f}"
          f"{np.sqrt(VAR1['라데마허'] / m):>12.4f}")
print("   둘 다 1/sqrt(m) 로 줄고 라데마허 쪽이 꾸준히 낮다.")

# === ② 분산 공식 ======================================================
N2, REP2 = 120, 200_000
print(f"\n② 분산 공식 확인 (n = {N2}, {REP2:,} 번)")
print(f"{'대각이 차지하는 몫':>18}{'가우스 모의':>13}{'2|A|F^2':>11}"
      f"{'라데마허 모의':>15}{'2(|A|F^2-sum)':>15}")
TS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
G_SIM, G_TH, R_SIM, R_TH, SHARE = [], [], [], [], []
base = sym_psd(N2, RNG)
dpart = np.diag(np.abs(RNG.standard_normal(N2)) + 0.3)
for t in TS:
    a = (1 - t) * base + t * dpart
    a = 0.5 * (a + a.T)
    fro = float((a ** 2).sum())
    dg = float((np.diag(a) ** 2).sum())
    gt, rt = 2 * fro, 2 * (fro - dg)
    gs = rs = 0.0
    for kind in ("가우스", "라데마허"):
        z = probes(kind, (REP2, N2), RNG)
        q = np.einsum("ri,ij,rj->r", z, a, z, optimize=True)
        if kind == "가우스":
            gs = float(q.var(ddof=1))
        else:
            rs = float(q.var(ddof=1))
    SHARE.append(dg / fro)
    G_SIM.append(gs)
    G_TH.append(gt)
    R_SIM.append(rs)
    R_TH.append(rt)
    print(f"{dg / fro:>18.4f}{gs:>13.2f}{gt:>11.2f}{rs:>15.2f}{rt:>15.2f}")
err_g = max(abs(a / b - 1) for a, b in zip(G_SIM, G_TH))
# 대각행렬이면 라데마허 분산이 정확히 0 이라 상대차를 잴 수 없다 — 빼고 센다
pos = [(a, b) for a, b in zip(R_SIM, R_TH) if b > 1e-6]
err_r = max(abs(a / b - 1) for a, b in pos)
print(f"   식과 모의의 상대차가 가우스 {100 * err_g:.2f}%, "
      f"라데마허 {100 * err_r:.2f}% 안이다 (분산이 0 인 맨 아래 줄은 뺐다).")
print(f"   그 맨 아래 줄 — 대각행렬에서 라데마허 분산은 모의값도 "
      f"{R_SIM[-1]:.2e} 로 0 이다.")
print("   대각이 무거울수록 라데마허가 유리하고, 대각행렬이면 분산이 0 이다.")

# === ③ n 과 무관하다 ==================================================
NS3, K3, REP3 = [200, 2000, 20000], 60, 300
MS3 = np.array([1, 2, 5, 10, 20, 50, 100, 200])
print(f"\n③ A = B B^T (B 는 n x {K3}) 를 만들지 않고 B^T z 만 쓴다")
print(f"{'n':>8}{'tr(A)':>12}{'|A|F/tr(A)':>13}" +
      "".join(f"{f'm={m}':>10}" for m in MS3))
REL = {}
for n in NS3:
    r = np.random.default_rng(100 + n)
    B = r.standard_normal((n, K3)) / np.sqrt(K3)
    tr = float((B ** 2).sum())
    G = B.T @ B
    fro = float(np.sqrt((G ** 2).sum()))
    rel = []
    for m in MS3:
        z = probes("라데마허", (REP3, m, n), r)
        bz = np.einsum("rmi,ik->rmk", z, B, optimize=True)
        est = (bz ** 2).sum(axis=2).mean(axis=1)
        rel.append(float(np.sqrt(np.mean((est / tr - 1) ** 2))))
    REL[n] = rel
    print(f"{n:>8}{tr:>12.3f}{fro / tr:>13.4f}"
          + "".join(f"{v:>10.4f}" for v in rel))
print(f"   세 줄이 거의 겹친다 — 상대오차는 n 이 아니라 "
      f"||A||_F/tr(A) 와 m 이 정한다.")

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.6, 5.2))
C_G, C_R = "#d62728", "#1f77b4"

for kind, c in (("가우스", C_G), ("라데마허", C_R)):
    ax1.loglog(MS, RM[kind], "o", color=c, ms=8, label=f"{kind} 탐침 — 모의")
    ax1.loglog(MS, np.sqrt(VAR1[kind] / MS), "-", color=c, lw=2.0, alpha=0.8,
               label=f"{kind} — $\\sqrt{{\\mathrm{{Var}}/m}}$")
ax1.set_xlabel("탐침의 수 $m$")
ax1.set_ylabel("$\\operatorname{tr}(\\mathbf{A})$ 추정의 RMSE")
ax1.set_xticks([1, 10, 100, 1000])
ax1.set_xticklabels(["$1$", "$10$", "$10^{2}$", "$10^{3}$"])
ax1.minorticks_off()
ax1.annotate(f"참값 $\\operatorname{{tr}}(\\mathbf{{A}}) = {TR1:.1f}$",
             xy=(0.03, 0.06), xycoords="axes fraction", fontsize=11,
             color="#333333")
ax1.set_title(f"① 탐침을 늘리면 ($n = {N1}$)", fontsize=12)
ax1.legend(loc="lower left", fontsize=9.5, bbox_to_anchor=(0.0, 0.14))
ax1.grid(alpha=0.25)

ax2.plot(SHARE, G_SIM, "o", color=C_G, ms=10, label="가우스 — 모의")
ax2.plot(SHARE, G_TH, "-", color=C_G, lw=2.2,
         label="$2\\|A\\|_F^2$")   # mathtext 에 \lVert 가 없다
ax2.plot(SHARE, R_SIM, "s", color=C_R, ms=9, label="라데마허 — 모의")
ax2.plot(SHARE, R_TH, "-", color=C_R, lw=2.2,
         label="$2(\\|A\\|_F^2-\\sum a_{ii}^2)$")
ax2.set_xlabel("$\\sum a_{ii}^2 \\,/\\, \\|A\\|_F^2$ "
               "— 대각이 차지하는 몫")
ax2.set_ylabel("$\\mathbf{z}^\\top\\mathbf{A}\\mathbf{z}$ 한 번의 분산")
ax2.set_title("② 분산은 식 그대로다", fontsize=12)
ax2.legend(loc="upper right", fontsize=9.5)
ax2.grid(alpha=0.25)

COL3 = {200: "#9ecae1", 2000: "#4292c6", 20000: "#08519c"}
for n in NS3:
    ax3.loglog(MS3, REL[n], "o-", color=COL3[n], lw=2.4, ms=8,
               label=f"$n = {n:,}$".replace(",", "{,}"))
ax3.set_xlabel("탐침의 수 $m$")
ax3.set_ylabel("상대오차 (RMSE / 참 대각합)")
ax3.set_xticks([1, 10, 100])
ax3.set_xticklabels(["$1$", "$10$", "$10^{2}$"])
ax3.set_yticks([1e-2, 1e-1])
ax3.set_yticklabels(["$1\\%$", "$10\\%$"])
ax3.minorticks_off()
ax3.annotate("$n$ 이 백 배가 되어도\n세 줄이 겹친다",
             xy=(0.05, 0.08), xycoords="axes fraction", fontsize=11,
             color="#333333")
ax3.set_title("③ 크기와 상관이 없다", fontsize=12)
ax3.legend(loc="upper right", fontsize=10.5)
ax3.grid(alpha=0.25)

fig.tight_layout()
out = "docs/ch00/linalg_regression/square_matrices/img/hutchinson_trace.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
