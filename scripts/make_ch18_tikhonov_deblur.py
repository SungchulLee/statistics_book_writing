r"""티호노프가 실제로 풀던 문제를 능형회귀의 식으로 푼다.

제1종 프레드홀름 적분방정식 — 흐려진 신호에서 원래 신호를 되찾는 문제다.
$\mathbf A$ 를 흐림 연산자라 하면 관측은 $\mathbf y = \mathbf A\mathbf f + \varepsilon$ 이고,
되찾는 식은 $(\mathbf A^\top\mathbf A + \lambda\mathbf I)^{-1}\mathbf A^\top\mathbf y$ 로
통계학의 능형회귀와 **글자 하나 다르지 않다.**

① 참 신호와 흐려진 관측. 곁들여 순진한 역변환이 어디까지 가는지.
② 세 가지 $\lambda$ 의 복원.
③ $\lambda$ 에 대한 복원오차.

만드는 파일: docs/ch18/motivation/img/tikhonov_deblur.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

RNG = np.random.default_rng(0)
M, W, SD = 200, 6.0, 1e-3

t = np.arange(M)
f = np.zeros(M)
f[40:70] = 1.0                                     # 네모 봉우리
f += 1.4 * np.exp(-0.5 * ((t - 120) / 6.0) ** 2)   # 좁은 봉우리
f += 0.7 * np.exp(-0.5 * ((t - 165) / 18.0) ** 2)  # 넓은 봉우리

A = np.exp(-0.5 * ((t[:, None] - t[None, :]) / W) ** 2)
A /= A.sum(axis=1, keepdims=True)
y = A @ f + RNG.normal(0, SD, M)

sv = np.linalg.svd(A, compute_uv=False)
KA = sv[0] / sv[-1]
err = lambda x: float(np.linalg.norm(x - f) / np.linalg.norm(f))   # noqa
tik = lambda lam: np.linalg.solve(A.T @ A + lam * np.eye(M), A.T @ y)  # noqa

naive = np.linalg.solve(A, y)
print(f"격자 {M} 칸, 흐림 폭 {W}, 잡음 sd {SD}")
print(f"조건수 kappa(A) = {KA:.3g},  kappa(A'A) = {KA ** 2:.3g}")
print(f"특잇값 최대 {sv[0]:.4g}, 최소 {sv[-1]:.4g}")
print(f"\n순진한 역변환 A^-1 y 의 상대오차 : {err(naive):.4g}")
print(f"   (복원값의 최대 절댓값 {np.abs(naive).max():.4g})")
print(f"아무것도 안 하고 y 를 쓰면        : {err(y):.4f}")

LAMS = np.logspace(-14, 1, 300)
E = np.array([err(tik(l)) for l in LAMS])
i = int(E.argmin())
L_OPT, E_OPT = float(LAMS[i]), float(E[i])
# 너무 작은 lambda 는 오차가 수십에 이르러 그림을 채워 버린다.
# 보이되 포화되지 않는 자리(오차 0.6 언저리)를 고른다.
L_LO = float(LAMS[int(np.abs(E[:i] - 0.6).argmin())])
L_HI = float(LAMS[np.abs(LAMS - 1e-1).argmin()])
print(f"티호노프 최적 lambda = {L_OPT:.3g},  상대오차 {E_OPT:.4f}")
print(f"   너무 작으면 lambda = {L_LO:.1g} -> {err(tik(L_LO)):.4f}")
print(f"   너무 크면   lambda = {L_HI:.1g} -> {err(tik(L_HI)):.4f}")
print(f"   최적 lambda 를 더한 뒤의 조건수 "
      f"{(sv[0] ** 2 + L_OPT) / (sv[-1] ** 2 + L_OPT):.3g} "
      f"(원래 {KA ** 2:.3g})")

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.6, 5.0))
C_T, C_Y, C_N = "#2ca02c", "#1f77b4", "#d62728"

ax1.plot(t, f, color=C_T, lw=2.6, label="참 신호 $f$")
ax1.plot(t, y, color=C_Y, lw=1.8, label="관측 $y = Af + \\varepsilon$")
ax1.set_xlabel("격자 칸")
ax1.set_ylabel("세기")
ax1.set_ylim(-0.35, 1.72)
ax1.set_title("① 흐려진 것을 되돌리고 싶다", fontsize=12)
ax1.legend(loc="upper left", fontsize=10)
ax1.grid(alpha=0.25)
ins = ax1.inset_axes([0.46, 0.52, 0.50, 0.42])
ins.plot(t, naive, color=C_N, lw=0.8)
ins.set_title("$A^{-1}y$ 를 그대로 풀면", fontsize=9, pad=2, color=C_N)
ins.tick_params(labelsize=7.5)
mx = np.abs(naive).max()
ex = int(np.floor(np.log10(mx)))
ins.set_yticks([-10 ** ex, 0, 10 ** ex])
ins.set_yticklabels([f"$-10^{{{ex}}}$", "$0$", f"$10^{{{ex}}}$"], fontsize=7.5)
ins.set_xticks([0, 100, 200])
ins.set_xticklabels(["$0$", "$100$", "$200$"], fontsize=7.5)

ax2.plot(t, f, color=C_T, lw=2.8, label="참 신호", zorder=1)
for lam, col, ls in [(L_LO, C_N, "-"), (L_OPT, "#111111", "-"),
                     (L_HI, "#ff7f0e", "--")]:
    ax2.plot(t, tik(lam), color=col, lw=1.8, ls=ls,
             label=f"$\\lambda = {lam:.0e}$".replace("e-0", "e-")
                   + f"  (오차 ${err(tik(lam)):.3f}$)")
ax2.set_xlabel("격자 칸")
ax2.set_ylabel("세기")
ax2.set_ylim(-0.6, 2.0)
ax2.set_title("② 같은 식에 $\\lambda\\mathbf{I}$ 하나 더하면", fontsize=12)
ax2.legend(loc="upper left", fontsize=9)
ax2.grid(alpha=0.25)

ax3.loglog(LAMS, E, color="#111111", lw=2.6)
ax3.plot([L_OPT], [E_OPT], "o", color="#111111", ms=12, zorder=4)
ax3.axhline(err(y), color=C_Y, lw=1.8, ls="--")
ax3.text(1e-13, err(y) * 0.62, "흐린 $y$ 를 그대로 답이라 하면", fontsize=10,
         color=C_Y)
ax3.annotate(f"$\\lambda = {L_OPT:.1e}$".replace("e-0", "e-")
             + f"\n오차 ${E_OPT:.3f}$",
             xy=(L_OPT, E_OPT), xytext=(4e-10, 2.6), textcoords="data",
             fontsize=10.5, arrowprops=dict(arrowstyle="->", lw=1.1))
ax3.set_xticks([1e-14, 1e-11, 1e-8, 1e-5, 1e-2, 1e1])
ax3.set_xticklabels(["$10^{-14}$", "$10^{-11}$", "$10^{-8}$", "$10^{-5}$",
                     "$10^{-2}$", "$10$"], fontsize=9.5)
ax3.set_yticks([0.1, 1.0, 10.0, 100.0])
ax3.set_yticklabels(["$0.1$", "$1$", "$10$", "$100$"])
ax3.minorticks_off()
ax3.set_xlabel("$\\lambda$")
ax3.set_ylabel("복원의 상대오차")
ax3.set_ylim(0.09, 300)
ax3.set_title("③ $\\lambda$ 를 고르는 일은 같은 일이다", fontsize=12)
ax3.grid(alpha=0.25)

out = "docs/ch18/motivation/img/tikhonov_deblur.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
