r"""같은 $\lambda$ 를 적어 넣어도 최적화기가 다르면 다른 추정량이 나온다.

벌점을 **손실에 적는 것**(Adam + L2)과 **갱신식에 적는 것**(AdamW)은 보통의
경사하강에서는 같은 일이지만 적응형 최적화기에서는 다르다. 얼마나 다른지 잰다.

① 능형 해와의 거리 — Adam + L2 는 능형에 닿고 AdamW 는 닿지 않는다.
② 까닭 — 아담의 걸음은 기울기 크기와 무관하게 $\eta$ 이므로 감쇠 $\eta\lambda_w w$
   와 균형을 이루는 자리가 $\lambda_w |w_j| = 1$ 이다. 좌표가 거기로 모인다.
③ 회전 등변성 — 능형과 Adam + L2 는 좌표계를 돌려도 같은 답을 내고
   AdamW 는 다른 답을 낸다.

만드는 파일: docs/ch18/tuning/img/adamw_ridge.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

RNG = np.random.default_rng(5)
N, P, B = 4000, 30, 64
T_DET, T_STO, ETA = 20000, 60000, 1e-2
LAMS = [0.1, 0.3, 1.0, 3.0]

dd = np.logspace(1.0, -1.0, P)
V = np.linalg.qr(RNG.normal(0, 1, (P, P)))[0]
X = RNG.normal(0, 1, (N, P)) @ V.T @ np.diag(dd) @ V
beta = RNG.normal(0, 1, P)
y = X @ beta + RNG.normal(0, 0.5, N)
G, c = X.T @ X / N, X.T @ y / N


def adam(G, c, l2=0.0, wd=0.0, T=T_DET, eta=ETA, Xs=None, ys=None,
         b1=0.9, b2=0.999, eps=1e-8):
    """Adam. l2 는 손실에 벌점을 더하고, wd 는 갱신식에서 직접 깎는다."""
    q = len(c)
    w = np.zeros(q)
    m = np.zeros(q)
    v = np.zeros(q)
    r = np.random.default_rng(0)
    for t in range(1, T + 1):
        if Xs is None:
            g = G @ w - c
        else:
            i = r.integers(0, len(ys), B)
            g = Xs[i].T @ (Xs[i] @ w - ys[i]) / B
        g = g + l2 * w
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        w = (w - eta * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)
             - eta * wd * w)
    return w


rel = lambda a, r: float(np.linalg.norm(a - r) / np.linalg.norm(r))  # noqa

# === ① 능형 해와의 거리 ==============================================
print("① 능형 해와의 상대거리")
print(f"{'lambda':>8}{'Adam+L2 (전배치)':>18}{'AdamW (전배치)':>16}"
      f"{'Adam+L2 (미니배치)':>20}{'AdamW (미니배치)':>18}")
res = {k: [] for k in ["l2d", "wdd", "l2s", "wds"]}
sols = {}
for lam in LAMS:
    rg = np.linalg.solve(G + lam * np.eye(P), c)
    a1 = adam(G, c, l2=lam)
    a2 = adam(G, c, wd=lam)
    s1 = adam(G, c, l2=lam, T=T_STO, Xs=X, ys=y)
    s2 = adam(G, c, wd=lam, T=T_STO, Xs=X, ys=y)
    sols[lam] = (rg, a1, a2)
    for k, vv in zip(["l2d", "wdd", "l2s", "wds"],
                     [rel(a1, rg), rel(a2, rg), rel(s1, rg), rel(s2, rg)]):
        res[k].append(vv)
    print(f"{lam:>8}{res['l2d'][-1]:>18.4f}{res['wdd'][-1]:>16.4f}"
          f"{res['l2s'][-1]:>20.4f}{res['wds'][-1]:>18.4f}")
print("   Adam + L2 의 고정점 조건은 기울기의 이동평균이 0 이 되는 것,")
print("   곧 grad L + lambda*w = 0 이다. 그것이 능형의 정상조건 그 자체다.")

# === ② AdamW 의 고정점 ===============================================
print(f"\n② AdamW 고정점에서 |w_j| 가 모이는 자리 (전배치)")
print(f"{'lambda_w':>10}{'|w_j| 중앙값':>14}{'1/lambda_w':>12}{'사분위폭':>10}")
for lam in LAMS:
    w = sols[lam][2]
    q1, q3 = np.percentile(np.abs(w), [25, 75])
    print(f"{lam:>10}{np.median(np.abs(w)):>14.4f}{1 / lam:>12.4f}"
          f"{q3 - q1:>10.4f}")
w3 = sols[3.0][2]
near = int((np.abs(np.abs(w3) * 3.0 - 1.0) < 0.01).sum())
print(f"   lambda_w = 3 에서는 좌표 {P} 개 가운데 {near} 개가 "
      f"1/lambda_w = {1 / 3:.4f} 의 1% 안에 든다 "
      f"(사분위폭 {np.percentile(np.abs(w3), 75) - np.percentile(np.abs(w3), 25):.4f}).")

# === ③ 회전 등변성 ===================================================
LAM = 1.0
Q = np.linalg.qr(RNG.normal(0, 1, (P, P)))[0]
Xr = X @ Q
Gr, cr = Xr.T @ Xr / N, Xr.T @ y / N
rg = np.linalg.solve(G + LAM * np.eye(P), c)
rg_r = Q @ np.linalg.solve(Gr + LAM * np.eye(P), cr)
rot = {"능형": rel(rg_r, rg)}
for nm, kw in [("Adam + L2", dict(l2=LAM)), ("AdamW", dict(wd=LAM))]:
    a = adam(G, c, **kw)
    ar = Q @ adam(Gr, cr, **kw)
    rot[nm] = rel(ar, a)
print(f"\n③ 좌표계를 돌렸다 되돌리면 (lambda = {LAM})")
for k, v in rot.items():
    print(f"   {k:<10} 원래 해와의 거리 {v:.2e}" if v < 1e-6
          else f"   {k:<10} 원래 해와의 거리 {v:.3f}")

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.6, 5.0))
C_L2, C_WD, C_R = "#2ca02c", "#d62728", "#1f77b4"
xs = np.arange(len(LAMS))

w = 0.2
ax1.bar(xs - 1.5 * w, res["l2d"], w, color=C_L2, label="Adam + L2 (전배치)")
ax1.bar(xs - 0.5 * w, res["l2s"], w, color=C_L2, alpha=0.5,
        label="Adam + L2 (미니배치)")
ax1.bar(xs + 0.5 * w, res["wdd"], w, color=C_WD, label="AdamW (전배치)")
ax1.bar(xs + 1.5 * w, res["wds"], w, color=C_WD, alpha=0.5,
        label="AdamW (미니배치)")
for i in range(len(LAMS)):
    ax1.text(xs[i] - w, res["l2s"][i] + 0.012, f"${res['l2s'][i]:.3f}$",
             ha="center", fontsize=9)
    ax1.text(xs[i] + w, res["wds"][i] + 0.012, f"${res['wds'][i]:.2f}$",
             ha="center", fontsize=9)
ax1.set_xticks(xs)
ax1.set_xticklabels([f"$\\lambda = {v}$" for v in LAMS], fontsize=10.5)
ax1.set_ylabel("능형 해와의 상대거리")
ax1.set_ylim(0, 0.82)
ax1.set_title("① 어느 쪽이 능형인가", fontsize=12)
ax1.legend(loc="upper left", fontsize=9)
ax1.grid(alpha=0.25, axis="y")

for k, lam in enumerate(LAMS):
    aw = np.abs(sols[lam][2])
    ax2.scatter(np.full(P, k) + RNG.uniform(-0.13, 0.13, P), aw, s=26,
                color=C_WD, alpha=0.7, zorder=3)
    ax2.plot([k - 0.32, k + 0.32], [1 / lam, 1 / lam], color="black", lw=2.0)
ax2.plot([], [], color="black", lw=2.0, label="$1/\\lambda_w$")
ax2.plot([], [], "o", color=C_WD, ms=6, label="AdamW 고정점의 $|w_j|$")
ax2.set_yscale("log")
ax2.set_yticks([0.1, 0.3, 1, 3, 10])
ax2.set_yticklabels(["$0.1$", "$0.3$", "$1$", "$3$", "$10$"])
ax2.minorticks_off()
ax2.set_xticks(xs)
ax2.set_xticklabels([f"$\\lambda_w = {v}$" for v in LAMS], fontsize=10.5)
ax2.set_ylabel("$|w_j|$")
ax2.set_title("② 좌표가 $1/\\lambda_w$ 로 모인다", fontsize=12)
ax2.legend(loc="lower left", fontsize=10)
ax2.grid(alpha=0.25, axis="y")

names = list(rot)
vals = [max(rot[k], 1e-16) for k in names]
cols = [C_R, C_L2, C_WD]
ax3.bar(np.arange(3), vals, 0.52, color=cols)
for i, v in enumerate(vals):
    lab = f"${rot[names[i]]:.3f}$" if rot[names[i]] > 1e-6 else "$0$ (기계오차)"
    ax3.text(i, max(v, 1e-16) * 2.2, lab, ha="center", fontsize=10.5)
ax3.set_yscale("log")
ax3.set_yticks([1e-15, 1e-12, 1e-9, 1e-6, 1e-3, 1e0])
ax3.set_yticklabels(["$10^{-15}$", "$10^{-12}$", "$10^{-9}$", "$10^{-6}$",
                     "$10^{-3}$", "$1$"])
ax3.minorticks_off()
ax3.set_xticks(np.arange(3))
ax3.set_xticklabels(names, fontsize=10.5)
ax3.set_ylim(1e-16, 1e3)
ax3.set_ylabel("돌렸다 되돌린 해와의 거리")
ax3.set_title(f"③ 좌표계를 돌려 보면 ($\\lambda = {LAM}$)", fontsize=12)
ax3.grid(alpha=0.25, axis="y")

fig.tight_layout()
out = "docs/ch18/tuning/img/adamw_ridge.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
