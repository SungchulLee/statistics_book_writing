r"""그람 행렬을 만드는 순간 조건수가 제곱된다 — 그리고 직교화의 두 판본.

이 쪽 $\S 4$ 는 $\mathbf{X}^\top\mathbf{X}$ 의 조건수를 다루고 $\S 7$ 은 그람–슈미트로
열을 직교화하라고 적는다. 둘을 붙여 놓으면 수치적으로 중요한 사실이 나온다 —
$\kappa(\mathbf{X}^\top\mathbf{X}) = \kappa(\mathbf{X})^2$ 이므로 **그람 행렬을 만드는
것 자체가 자릿수를 절반 버리는 일**이고, 그람–슈미트도 쓰는 판본에 따라 결과가 갈린다.

① 직교성을 얼마나 잃는가 — 고전 그람–슈미트 · 수정 그람–슈미트 · 하우스홀더.
② 그람 행렬의 조건수는 정확히 제곱이다.
③ 최소제곱해의 오차 — 정규방정식(그람 행렬)과 QR.

만드는 파일: docs/ch00/linalg_regression/square_matrices/img/gram_schmidt_conditioning.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

U_EPS = np.finfo(float).eps / 2          # 단위 반올림 u = 2^-53
M, N = 60, 12
KAPPAS = np.logspace(1, 13, 25)
RNG = np.random.default_rng(0)


def design(kappa, rng):
    """조건수가 정확히 kappa 인 m x n 계획행렬."""
    q1 = np.linalg.qr(rng.standard_normal((M, N)))[0]
    q2 = np.linalg.qr(rng.standard_normal((N, N)))[0]
    s = np.logspace(0, -np.log10(kappa), N)
    return (q1 * s) @ q2.T


def cgs(a):
    """고전 그람–슈미트 — 교과서에 적힌 그대로."""
    m, n = a.shape
    q = np.zeros((m, n))
    for k in range(n):
        v = a[:, k].copy()
        if k:
            v -= q[:, :k] @ (q[:, :k].T @ a[:, k])   # 원본 열에서 한 번에 뺀다
        q[:, k] = v / np.linalg.norm(v)
    return q


def mgs(a):
    """수정 그람–슈미트 — 뺄 때마다 갱신된 벡터를 쓴다."""
    v = a.copy()
    m, n = a.shape
    q = np.zeros((m, n))
    for k in range(n):
        q[:, k] = v[:, k] / np.linalg.norm(v[:, k])
        for j in range(k + 1, n):
            v[:, j] -= (q[:, k] @ v[:, j]) * q[:, k]
    return q


def loss(q):
    return float(np.linalg.norm(q.T @ q - np.eye(q.shape[1]), 2))


print(f"단위 반올림 u = {U_EPS:.3e}  (배정밀도)")
print(f"\n①② 계획행렬 {M} x {N}")
print(f"{'kappa(X)':>11}{'고전 GS':>12}{'수정 GS':>12}{'하우스홀더':>12}"
      f"{'kappa(X\'X)':>13}{'/ kappa^2':>11}")
L_C, L_M, L_H, K_G = [], [], [], []
for kap in KAPPAS:
    x = design(kap, RNG)
    L_C.append(loss(cgs(x)))
    L_M.append(loss(mgs(x)))
    L_H.append(loss(np.linalg.qr(x)[0]))
    kg = float(np.linalg.cond(x.T @ x))
    K_G.append(kg)
    print(f"{kap:>11.1e}{L_C[-1]:>12.2e}{L_M[-1]:>12.2e}{L_H[-1]:>12.2e}"
          f"{kg:>13.2e}{kg / kap ** 2:>11.3f}")
# kappa(X)^2 이 1/u 를 넘으면 kappa(X'X) 자체를 수로 잴 수 없다 — 그 앞까지만 센다
ok = [(k, kg) for k, kg in zip(KAPPAS, K_G) if k ** 2 < 1 / U_EPS]
good = [k for k, kg in ok]
rat = [kg / k ** 2 for k, kg in ok]
print(f"   kappa(X) 가 {max(good):.1e} 이하인 {len(ok)} 곳에서 "
      f"kappa(X'X)/kappa(X)^2 가 {min(rat):.3f}~{max(rat):.3f} 다 "
      f"— 정확히 제곱이다.")
print(f"   kappa(X) 가 {max(good):.1e} 를 넘으면 그람 행렬의 조건수가 "
      f"1/u = {1 / U_EPS:.1e} 를 넘어 수치적으로 특이해진다.")

# === ③ 최소제곱해의 오차 =============================================
print(f"\n③ y 를 X 의 열공간 안에 두어 참 beta 를 알고 푼다")
print(f"{'kappa(X)':>11}{'정규방정식':>14}{'QR':>12}{'kappa^2 u':>12}{'kappa u':>11}")
E_NE, E_QR = [], []
for kap in KAPPAS:
    x = design(kap, RNG)
    beta = RNG.standard_normal(N)
    y = x @ beta
    g = x.T @ x
    try:
        b_ne = np.linalg.solve(g, x.T @ y)
    except np.linalg.LinAlgError:
        b_ne = np.full(N, np.nan)
    q, r = np.linalg.qr(x)
    b_qr = np.linalg.solve(r, q.T @ y)
    nb = np.linalg.norm(beta)
    E_NE.append(float(np.linalg.norm(b_ne - beta) / nb))
    E_QR.append(float(np.linalg.norm(b_qr - beta) / nb))
    print(f"{kap:>11.1e}{E_NE[-1]:>14.2e}{E_QR[-1]:>12.2e}"
          f"{kap ** 2 * U_EPS:>12.2e}{kap * U_EPS:>11.2e}")
print("   정규방정식은 kappa^2 u 를 따라가고 QR 은 kappa u 를 따라간다 — "
      "자릿수가 두 배로 든다.")

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.6, 5.2))
C_C, C_M, C_H = "#d62728", "#1f77b4", "#2ca02c"

ax1.loglog(KAPPAS, np.maximum(L_C, 1e-18), "o-", color=C_C, lw=2.4, ms=7,
           label="고전 그람–슈미트")
ax1.loglog(KAPPAS, np.maximum(L_M, 1e-18), "s-", color=C_M, lw=2.4, ms=7,
           label="수정 그람–슈미트")
ax1.loglog(KAPPAS, np.maximum(L_H, 1e-18), "^-", color=C_H, lw=2.4, ms=7,
           label="하우스홀더 ($\\mathtt{numpy.linalg.qr}$)")
ax1.loglog(KAPPAS, U_EPS * KAPPAS ** 2, "--", color=C_C, lw=1.4, alpha=0.7)
ax1.loglog(KAPPAS, U_EPS * KAPPAS, "--", color=C_M, lw=1.4, alpha=0.7)
ax1.axhline(1.0, color="#666666", lw=1.2, ls=":")
ax1.text(1.3e1, 1.4, "직교성이 통째로 사라진 높이", fontsize=10,
         color="#555555")
ax1.text(1.5e3, 1.2e-6, "$u\\,\\kappa^2$", color=C_C, fontsize=11)
ax1.text(1.5e8, 4e-7, "$u\\,\\kappa$", color=C_M, fontsize=11)
ax1.set_xlabel("$\\kappa(\\mathbf{X})$")
ax1.set_ylabel("$\\|\\mathbf{I}-\\mathbf{Q}^\\top\\mathbf{Q}\\|_2$")
ax1.set_xticks([1e2, 1e6, 1e10])
ax1.set_xticklabels(["$10^{2}$", "$10^{6}$", "$10^{10}$"])
ax1.set_yticks([1e-16, 1e-12, 1e-8, 1e-4, 1])
ax1.set_yticklabels(["$10^{-16}$", "$10^{-12}$", "$10^{-8}$", "$10^{-4}$",
                     "$1$"])
ax1.minorticks_off()
ax1.set_ylim(1e-17, 50)
ax1.set_title("① 같은 식, 다른 차례, 다른 결과", fontsize=12)
ax1.legend(loc="lower right", fontsize=9.5)
ax1.grid(alpha=0.25)

ax2.loglog(KAPPAS, K_G, "o", color=C_M, ms=9, label="잰 값 $\\kappa$"
           "$(\\mathbf{X}^\\top\\mathbf{X})$")
ax2.loglog(KAPPAS, KAPPAS ** 2, "-", color="#666666", lw=2.0,
           label="$\\kappa(\\mathbf{X})^2$")
ax2.axhline(1 / U_EPS, color=C_C, lw=1.6, ls="--")
ax2.text(1.3e1, 1 / U_EPS / 1.6, "$1/u$ — 여기를 넘으면 수치적으로 특이하다",
         color=C_C, fontsize=10, va="top")
ax2.set_xlabel("$\\kappa(\\mathbf{X})$")
ax2.set_ylabel("$\\kappa(\\mathbf{X}^\\top\\mathbf{X})$")
ax2.set_xticks([1e2, 1e6, 1e10])
ax2.set_xticklabels(["$10^{2}$", "$10^{6}$", "$10^{10}$"])
ax2.set_yticks([1e4, 1e10, 1e16, 1e22])
ax2.set_yticklabels(["$10^{4}$", "$10^{10}$", "$10^{16}$", "$10^{22}$"])
ax2.minorticks_off()
ax2.set_title("② 만드는 순간 제곱된다", fontsize=12)
ax2.legend(loc="lower right", fontsize=10.5)
ax2.grid(alpha=0.25)

ax3.loglog(KAPPAS, np.maximum(E_NE, 1e-18), "o-", color=C_C, lw=2.4, ms=7,
           label="정규방정식 ($\\mathbf{X}^\\top\\mathbf{X}$ 를 만든다)")
ax3.loglog(KAPPAS, np.maximum(E_QR, 1e-18), "^-", color=C_H, lw=2.4, ms=7,
           label="QR ($\\mathbf{X}$ 를 그대로 둔다)")
ax3.loglog(KAPPAS, U_EPS * KAPPAS ** 2, "--", color=C_C, lw=1.4, alpha=0.7)
ax3.loglog(KAPPAS, U_EPS * KAPPAS, "--", color=C_H, lw=1.4, alpha=0.7)
ax3.axhline(1.0, color="#666666", lw=1.2, ls=":")
ax3.text(1.3e1, 1.4, "답이 아예 없는 높이", fontsize=10, color="#555555")
ax3.set_xlabel("$\\kappa(\\mathbf{X})$")
ax3.set_ylabel("$\\hat{\\beta}$ 의 상대오차")
ax3.set_xticks([1e2, 1e6, 1e10])
ax3.set_xticklabels(["$10^{2}$", "$10^{6}$", "$10^{10}$"])
ax3.set_yticks([1e-16, 1e-12, 1e-8, 1e-4, 1])
ax3.set_yticklabels(["$10^{-16}$", "$10^{-12}$", "$10^{-8}$", "$10^{-4}$",
                     "$1$"])
ax3.minorticks_off()
ax3.set_ylim(1e-17, 50)
ax3.set_title("③ 자릿수가 두 배로 든다", fontsize=12)
ax3.legend(loc="lower right", fontsize=9.5)
ax3.grid(alpha=0.25)

fig.tight_layout()
out = ("docs/ch00/linalg_regression/square_matrices/img/"
       "gram_schmidt_conditioning.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
