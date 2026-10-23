r"""고윳값이 같아도 하는 짓이 다르다 — 닮음이 지켜 주지 않는 것.

이 쪽의 절 2 는 고윳값이 닮음 불변량이라 하고, 절 3 은 대칭성이 불변량이 아니라고
한다. 그 둘을 겹치면 성가신 일이 하나 나온다.

대칭이 아닌 행렬에서는 **고윳값이 A^n 의 크기를 말해 주지 않는다.** 스펙트럼
반지름이 1 보다 작아 결국은 0 으로 가더라도, 가는 길에 몇 자릿수씩 커질 수 있다.
두 행렬이 같은 대각행렬에 닮아 있어도(= 고윳값이 같아도) 그렇다.

① 고윳값이 모두 0.9 인 두 행렬 — 대칭인 것과 아닌 것의 ||A^n||.
② 봉우리의 크기와 그 자리.
③ 대각화행렬 V 가 나빠질수록 봉우리가 커진다 — 한계는 kappa(V) * rho^n.

만드는 파일: docs/ch00/linalg_regression/square_matrices/img/nonnormal_transient.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"

RNG = np.random.default_rng(0)
RHO, K, NMAX = 0.9, 8, 260


def jordanish(k, rho, c):
    """대각에 rho, 바로 윗줄에 c. 고윳값은 모두 rho 다."""
    A = np.eye(k) * rho
    A[np.arange(k - 1), np.arange(1, k)] = c
    return A


def norms(A, nmax=NMAX):
    out, P = [], np.eye(A.shape[0])
    for _ in range(nmax + 1):
        out.append(float(np.linalg.norm(P, 2)))
        P = P @ A
    return np.array(out)


SYM = np.eye(K) * RHO                 # 대칭 (사실 대각). 고윳값 모두 0.9
NON = jordanish(K, RHO, 1.0)          # 대칭 아님. 고윳값 모두 0.9

print(f"크기 {K}x{K}, 두 행렬 모두 고윳값이 모두 {RHO} 다.")
print(f"   대칭 쪽 고윳값   {np.round(np.linalg.eigvals(SYM).real, 3)[:4]} ...")
print(f"   비대칭 쪽 고윳값 {np.round(np.linalg.eigvals(NON).real, 3)[:4]} ...")
print(f"   스펙트럼 반지름은 둘 다 {max(abs(np.linalg.eigvals(NON))):.3f} 다.")

ns = np.arange(NMAX + 1)
NS_SYM, NS_NON = norms(SYM), norms(NON)
pk = int(np.argmax(NS_NON))
# 한글은 고정폭에서 두 칸을 차지하므로 자리맞춤을 직접 적는다
print("\n①  n     대칭 ||A^n||     비대칭 ||A^n||        비")
for n in (0, 5, 10, 20, 40, 60, 100, 200, 260):
    print(f"   {n:>4}   {NS_SYM[n]:>13.3e}   {NS_NON[n]:>14.3e}   "
          f"{NS_NON[n]/NS_SYM[n]:>10.3e}")
print(f"\n② 비대칭 쪽은 n = {pk} 에서 {NS_NON[pk]:.4g} 까지 올라갔다가 내려온다.")
print(f"   같은 자리에서 대칭 쪽은 {NS_SYM[pk]:.4g} 다 — "
      f"{NS_NON[pk]/NS_SYM[pk]:.3g} 배 차이다.")
print(f"   **둘은 같은 대각행렬에 닮아 있다. 고윳값만 보면 구별할 수 없다.**")
below = np.where((NS_NON < 1.0) & (ns > pk))[0]
if below.size:
    print(f"   비대칭 쪽이 1 아래로 내려오는 것은 n = {below[0]} 부터다.")
else:
    print(f"   비대칭 쪽은 n = {NMAX} 에서도 아직 {NS_NON[-1]:.3g} 로 "
          f"1 위에 있다 — 결국 0 으로 가지만 한참 걸린다.")

# === ③ 대각화행렬이 나빠질수록 =====================================
CS = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
print("\n③ 윗줄의 c 를 키우며 (고윳값은 그대로 모두 0.9)")
print("      c     봉우리 ||A^n||   그 자리 n")
PEAK, PKN = [], []
for c in CS:
    v = norms(jordanish(K, RHO, c))
    PEAK.append(float(v.max()))
    PKN.append(int(np.argmax(v)))
    print(f"   {c:>4.2f}   {PEAK[-1]:>14.4g}   {PKN[-1]:>9}")
print("   c = 0 이면 대칭이고 봉우리가 1 이다 — 처음부터 줄기만 한다.")
print(f"   c 를 {CS[-1]} 로 키우면 봉우리가 {PEAK[-1]:.3g} 까지 간다.")
print("   **고윳값은 하나도 바뀌지 않았다.**")

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.6, 5.2))
C_S, C_N = "#2ca02c", "#d62728"

ax1.semilogy(ns, NS_SYM, lw=2.8, color=C_S, label="대칭 (대각)")
ax1.semilogy(ns, NS_NON, lw=2.8, color=C_N, label="대칭 아님")
ax1.axhline(1.0, color="#333333", lw=1.6, ls=":")
ax1.plot([pk], [NS_NON[pk]], "o", color="#333333", ms=13, zorder=5)
_m, _e = f"{NS_NON[pk]:.3e}".split("e")
ax1.annotate(f"봉우리 ${_m} \\times 10^{{{int(_e)}}}$\n($n = {pk}$)",
             (pk, NS_NON[pk]),
             xytext=(0.8, -1.4), textcoords="offset fontsize", fontsize=10.5,
             color="#333333")
# 로그 축의 음수 지수는 mathtext 를 거치므로 unicode_minus 로 막지 못한다 —
# 눈금과 글자를 직접 적는다
ax1.set_yticks([1e-12, 1e-9, 1e-6, 1e-3, 1e0, 1e3, 1e6])
ax1.set_yticklabels(["$10^{-12}$", "$10^{-9}$", "$10^{-6}$", "$10^{-3}$",
                     "$10^{0}$", "$10^{3}$", "$10^{6}$"])
ax1.minorticks_off()
ax1.set_ylim(1e-13, 3e7)
ax1.set_xlabel("$n$")
ax1.set_ylabel("$\\|A^n\\|_2$")
ax1.set_title(f"① 고윳값이 둘 다 모두 ${RHO}$ 인데", fontsize=12)
ax1.legend(loc="lower left", fontsize=10.5)
ax1.grid(alpha=0.25)

ev = np.linalg.eigvals(NON)
th = np.linspace(0, 2 * np.pi, 300)
ax2.plot(np.cos(th), np.sin(th), color="#888888", lw=1.6, ls="--",
         label="단위원")
ax2.plot(RHO * np.cos(th), RHO * np.sin(th), color="#bbbbbb", lw=1.2, ls=":")
ax2.plot(ev.real, ev.imag, "o", color=C_N, ms=16, alpha=0.5,
         label=f"고윳값 ({K} 개가 겹쳐 있다)")
ax2.plot(np.linalg.eigvals(SYM).real, np.linalg.eigvals(SYM).imag, "x",
         color=C_S, ms=13, mew=3, label="대칭 쪽 고윳값")
ax2.set_xlabel("실수부")
ax2.set_ylabel("허수부")
ax2.set_aspect("equal")
ax2.set_xlim(-1.25, 1.25)
ax2.set_ylim(-1.25, 1.25)
ax2.set_title("② 고윳값 그림은 둘을 구별하지 못한다", fontsize=12)
ax2.legend(loc="upper left", fontsize=9)
ax2.grid(alpha=0.25)

ax3.semilogy(CS, PEAK, "o-", color=C_N, lw=2.8, ms=11)
ax3.axhline(1.0, color=C_S, lw=2.2, ls="--")
ax3.text(1.4, 1.3, "대칭이면 봉우리가 $1$", fontsize=10.5, color=C_S)
ax3.set_xlabel("윗줄 성분 $c$")
ax3.set_ylabel("봉우리 $\\max_n \\|A^n\\|_2$")
ax3.set_title("③ 고윳값을 그대로 두고 $c$ 만 키우면", fontsize=12)
ax3.grid(alpha=0.25)

fig.tight_layout()
out = "docs/ch00/linalg_regression/square_matrices/img/nonnormal_transient.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
