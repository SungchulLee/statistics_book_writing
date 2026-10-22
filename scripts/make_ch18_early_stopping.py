r"""경사하강을 일찍 멈추는 것이 능형과 같은 일을 한다 — 그러나 똑같지는 않다.

$X = UDV^\top$ 라 두고 $\beta^0 = 0$ 에서 학습률 $\eta$ 로 경사하강을 돌리면,
$j$ 번째 특이방향의 계수는 최소제곱값에 **수축인자**
$1 - (1 - \eta d_j^2)^t$ 를 곱한 것이 된다. 능형의 수축인자는
$d_j^2/(d_j^2+\lambda)$ 다. $\lambda = 1/(\eta t)$ 로 맞추면 둘이 같은 모양이 된다.

① 세 수축 프로파일 — 능형·조기 종료·주성분 절단. 조기 종료는 **그 사이**에 있다.
② 유효자유도에 대한 시험오차 — 둘 다 $U$ 자인데 최저점이 다르다.
③ 훈련을 오래 할수록 유효자유도가 올라간다. 모수의 수는 내내 그대로다.

만드는 파일: docs/ch18/ridge/img/early_stopping_ridge.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

RNG = np.random.default_rng(3)
N, P, NTE = 150, 80, 6000
SIG = 0.08

A = RNG.normal(0, 1, (N, P))
U, _, Vt = np.linalg.svd(A, full_matrices=False)
d = np.logspace(0.8, -1.4, P)                 # 특잇값을 넓게 퍼뜨린다
X = U @ np.diag(d) @ Vt
a = 1.0 / np.arange(1, P + 1)                 # 신호는 큰 특이방향에 실린다
a = a / np.linalg.norm(a)
beta = Vt.T @ a
y = X @ beta + RNG.normal(0, SIG, N)
Xte = RNG.normal(0, 1, (NTE, P)) @ Vt.T @ np.diag(d) @ Vt
yte = Xte @ beta + RNG.normal(0, SIG, NTE)

Uy = U.T @ y
ETA = 0.5 / d[0] ** 2
d2 = d ** 2


def fit(f):
    """수축인자 f 를 최소제곱 계수에 곱한 추정량."""
    return Vt.T @ (f * Uy / d)


def err(b):
    return float(np.sqrt(np.mean((Xte @ b - yte) ** 2)))


f_gd = lambda t: 1 - (1 - ETA * d2) ** t      # noqa: E731
f_rg = lambda lam: d2 / (d2 + lam)            # noqa: E731
f_pc = lambda k: (np.arange(P) < k).astype(float)   # noqa: E731

E_OLS = err(fit(np.ones(P)))
print(f"설계행렬 {X.shape},  조건수 kappa(X) = {d[0] / d[-1]:.1f}")
print(f"잡음 sd {SIG},  최소제곱 시험 RMSE {E_OLS:.4f}")
print(f"학습률 eta = {ETA:.5g}")

# === ② 두 U 자 =======================================================
TS = np.unique(np.round(np.logspace(0, 6.5, 500)).astype(int))
LAMS = np.logspace(-7, 3, 500)
e_gd = np.array([err(fit(f_gd(t))) for t in TS])
e_rg = np.array([err(fit(f_rg(l))) for l in LAMS])
df_gd = np.array([f_gd(t).sum() for t in TS])
df_rg = np.array([f_rg(l).sum() for l in LAMS])
i, j = int(e_gd.argmin()), int(e_rg.argmin())
T_STAR, LAM_STAR = int(TS[i]), float(LAMS[j])

print(f"\n② 가장 좋은 자리")
print(f"   조기 종료 : t = {T_STAR:,}  RMSE {e_gd[i]:.4f}  "
      f"유효자유도 {df_gd[i]:.1f}  (대응 lambda = {1 / (ETA * T_STAR):.4g})")
print(f"   능형      : lambda = {LAM_STAR:.4g}  RMSE {e_rg[j]:.4f}  "
      f"유효자유도 {df_rg[j]:.1f}")
print(f"   최소제곱 대비 — 조기 종료 {E_OLS / e_gd[i]:.2f} 배, "
      f"능형 {E_OLS / e_rg[j]:.2f} 배")

# === ① 같은 자유도에서 세 프로파일 ===================================
DF = df_gd[i]
lam_m = float(LAMS[np.abs(df_rg - DF).argmin()])
k_m = int(round(DF))
print(f"\n① 유효자유도를 {DF:.1f} 로 맞춘 세 수축 프로파일")
print(f"   능형 lambda = {lam_m:.4g},  조기 종료 t = {T_STAR},  "
      f"주성분 절단 k = {k_m}")
print(f"   같은 자유도에서의 시험 RMSE — "
      f"능형 {err(fit(f_rg(lam_m))):.4f}, "
      f"조기 종료 {err(fit(f_gd(T_STAR))):.4f}, "
      f"주성분 절단 {err(fit(f_pc(k_m))):.4f}")
print("   수축인자가 0.5 를 지나는 자리 (d^2 단위):")
for nm, f in [("능형", f_rg(lam_m)), ("조기 종료", f_gd(T_STAR))]:
    k0 = int(np.argmax(f < 0.5))
    print(f"      {nm:<8} d^2 = {d2[k0]:.4g}")
print("   그 자리에서 열 배 작은 방향에 남는 몫:")
for nm, f in [("능형", f_rg(lam_m)), ("조기 종료", f_gd(T_STAR))]:
    k0 = int(np.argmax(f < 0.5))
    k1 = int(np.abs(d2 - d2[k0] / 10).argmin())
    print(f"      {nm:<8} {f[k1]:.4f}")

# === ③ 시간이 자유도다 ===============================================
print(f"\n③ 훈련 걸음과 유효자유도 (모수는 내내 {P} 개)")
print(f"{'t':>9}{'1/(eta t)':>12}{'df_GD':>8}{'df_ridge':>10}")
for t in [1, 10, 100, 1000, 10000, 100000, 1000000]:
    lam = 1 / (ETA * t)
    print(f"{t:>9}{lam:>12.4g}{f_gd(t).sum():>8.1f}{f_rg(lam).sum():>10.1f}")

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.6, 5.0))
C_R, C_G, C_P = "#1f77b4", "#d62728", "#2ca02c"

ax1.semilogx(d2, f_rg(lam_m), "o-", color=C_R, lw=2.4, ms=5,
             label=f"능형 $\\lambda = {lam_m:.2f}$")
ax1.semilogx(d2, f_gd(T_STAR), "s-", color=C_G, lw=2.4, ms=5,
             label=f"조기 종료 $t = {T_STAR}$")
ax1.semilogx(d2, f_pc(k_m), color=C_P, lw=2.2, ls="--",
             label=f"주성분 절단 $k = {k_m}$")
ax1.axhline(0.5, color="#888888", lw=1.0, ls=":")
ax1.set_xticks([1e-3, 1e-2, 1e-1, 1e0, 1e1])
ax1.set_xticklabels(["$10^{-3}$", "$10^{-2}$", "$10^{-1}$", "$1$", "$10$"],
                    fontsize=9.5)
ax1.minorticks_off()
ax1.set_xlabel("$d_j^2$ — 그 방향을 자료가 얼마나 또렷이 보는가")
ax1.set_ylabel("최소제곱 계수에 남겨 두는 몫")
ax1.set_ylim(-0.04, 1.08)
ax1.set_title(f"① 셋 다 유효자유도 ${DF:.0f}$ 인데 모양이 다르다", fontsize=12)
ax1.legend(loc="upper left", fontsize=10)
ax1.grid(alpha=0.25)

ax2.plot(df_rg, e_rg, "-", color=C_R, lw=2.6, label="능형 ($\\lambda$ 를 쓸어서)")
ax2.plot(df_gd, e_gd, "-", color=C_G, lw=2.6,
         label="조기 종료 ($t$ 를 쓸어서)")
ax2.plot([df_rg[j]], [e_rg[j]], "o", color=C_R, ms=12, zorder=4)
ax2.plot([df_gd[i]], [e_gd[i]], "s", color=C_G, ms=12, zorder=4)
ax2.axhline(E_OLS, color="black", lw=1.3, ls="--")
ax2.text(P * 0.98, E_OLS * 1.04, "최소제곱", fontsize=10, ha="right")
ax2.annotate(f"${e_gd[i]:.3f}$", xy=(df_gd[i], e_gd[i]), xytext=(26, 0.21),
             textcoords="data", fontsize=10.5, color=C_G, ha="center",
             arrowprops=dict(arrowstyle="->", color=C_G, lw=1.1))
ax2.annotate(f"${e_rg[j]:.3f}$", xy=(df_rg[j], e_rg[j]), xytext=(52, 0.30),
             textcoords="data", fontsize=10.5, color=C_R, ha="center",
             arrowprops=dict(arrowstyle="->", color=C_R, lw=1.1))
ax2.set_xlabel("유효자유도")
ax2.set_ylabel("시험 RMSE")
ax2.set_ylim(0.2, 0.78)
ax2.set_xlim(0, P + 2)
ax2.set_title("② 같은 $U$ 자인데 바닥이 다르다", fontsize=12)
ax2.legend(loc="lower right", fontsize=10)
ax2.grid(alpha=0.25)

ax3.semilogx(TS, df_gd, "-", color=C_G, lw=2.8, label="조기 종료 $t$ 걸음 뒤")
ax3.semilogx(TS, [f_rg(1 / (ETA * t)).sum() for t in TS], "--", color=C_R,
             lw=2.2, label="능형 $\\lambda = 1/(\\eta t)$")
ax3.axhline(P, color="black", lw=1.3, ls=":")
ax3.text(1.4, P - 3.4, f"모수의 수 $p = {P}$ — 내내 그대로다", fontsize=10.5)
ax3.axvline(T_STAR, color="#555555", lw=1.2, ls=":")
ax3.annotate(f"가장 좋은 자리\n$t = {T_STAR}$, 자유도 ${DF:.0f}$",
             xy=(T_STAR, DF), xytext=(2.0e2, 8.0), textcoords="data",
             fontsize=10.5, color="#555555",
             arrowprops=dict(arrowstyle="->", color="#555555", lw=1.1))
ax3.set_xticks([1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6])
ax3.set_xticklabels(["$1$", "$10$", "$10^{2}$", "$10^{3}$", "$10^{4}$",
                     "$10^{5}$", "$10^{6}$"], fontsize=9.5)
ax3.minorticks_off()
ax3.set_xlabel("경사하강의 걸음 수 $t$")
ax3.set_ylabel("유효자유도")
ax3.set_ylim(0, P + 6)
ax3.set_title("③ 복잡도를 정하는 것은 훈련 시간이다", fontsize=12)
ax3.legend(loc="center right", fontsize=10)
ax3.grid(alpha=0.25)

fig.tight_layout()
out = "docs/ch18/ridge/img/early_stopping_ridge.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
