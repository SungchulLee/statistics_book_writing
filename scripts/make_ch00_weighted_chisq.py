r"""멱등이 아니면 무엇이 되는가 — 고윳값이 섞인 이차형식.

이 쪽은 z'Az 가 카이제곱이 되는 **필요충분조건**이 A 의 멱등성(고윳값이 0 과 1
뿐)임을 보인다. 그러면 멱등이 아닐 때는 무엇이 되는가. 고윳값 lambda_i 로

    z'Az = sum_i lambda_i * (독립인 chi^2_1)

인 **가중 카이제곱합**이 된다. 요즘의 커널 검정이 사는 자리가 거기다.

① 두 가지 고윳값 모양 — 멱등(0/1)과 커널(완만히 줄어드는 값들).
② 그 둘이 내놓는 분포. 멱등은 chi^2_r 과 정확히 겹치고 커널은 어떤 chi^2 와도
   겹치지 않는다.
③ 분산 / 평균 = 2 tr(A^2) / tr(A) 가 2 인 것과 멱등인 것이 같은 말이다.

만드는 파일: docs/ch00/linalg_regression/linalg_statistics/img/weighted_chisq.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"

from scipy import stats   # noqa: E402

RNG = np.random.default_rng(0)
N, REP = 60, 200_000
R = 12                       # 멱등행렬의 계수


def kernel_eigs(n, gamma=0.5):
    """가우스 커널 그람행렬을 중심화한 것의 고윳값 — 완만히 줄어든다."""
    x = np.linspace(-3, 3, n)
    K = np.exp(-gamma * (x[:, None] - x[None, :]) ** 2)
    H = np.eye(n) - np.ones((n, n)) / n
    Kc = H @ K @ H
    ev = np.linalg.eigvalsh(Kc)[::-1]
    return np.maximum(ev, 0)


EIG = {
    "멱등 (0 과 1 뿐)": np.r_[np.ones(R), np.zeros(N - R)],
    "커널 (완만히 줄어듦)": kernel_eigs(N),
}

# 한글은 고정폭에서 두 칸을 차지하므로 자리맞춤을 직접 적는다
print(f"n = {N}, 되풀이 {REP:,}")
print("\n① 두 고윳값 모양")
print("   모양                 tr(A)    tr(A^2)   분산/평균   0 과 1 뿐인가")
SAMP = {}
for name, lam in EIG.items():
    tr1, tr2 = lam.sum(), (lam ** 2).sum()
    # z'Az = sum lambda_i * chi^2_1
    chi = RNG.chisquare(1, (REP, int((lam > 1e-12).sum())))
    SAMP[name] = chi @ lam[lam > 1e-12]
    only01 = np.allclose(lam, np.round(lam))
    print(f"   {name:<20} {tr1:>6.2f}   {tr2:>7.2f}   {2*tr2/tr1:>8.3f}   "
          f"{'예' if only01 else '아니다':>12}")
print("   멱등 쪽만 분산/평균이 정확히 2 다 — 카이제곱의 표식이다.")

# === ② 분포를 겹쳐 본다 =============================================
print("\n② 카이제곱과 맞춰 보기 (콜모고로프–스미르노프 거리)")
print("   모양                 chi^2_tr(A) 와의 거리   새터스웨이트와의 거리")
SATT = {}
for name, lam in EIG.items():
    s = SAMP[name]
    tr1, tr2 = lam.sum(), (lam ** 2).sum()
    # 그냥 chi^2_{tr(A)} 로 보았을 때
    d_naive = float(np.max(np.abs(
        np.sort(stats.chi2.cdf(np.sort(s), tr1)) - np.linspace(0, 1, len(s)))))
    # 새터스웨이트: c * chi^2_nu 로 평균과 분산을 맞춘다
    nu = tr1 ** 2 / tr2
    c = tr2 / tr1
    SATT[name] = (nu, c)
    d_satt = float(np.max(np.abs(
        np.sort(stats.chi2.cdf(np.sort(s) / c, nu)) - np.linspace(0, 1, len(s)))))
    print(f"   {name:<20} {d_naive:>19.4f}   {d_satt:>19.4f}")
    print(f"      → 새터스웨이트 자유도 {nu:.2f}, 배율 {c:.3f}")
print("   멱등 쪽은 둘 다 0 에 가깝다(그 경우 새터스웨이트가 원래 식으로 돌아간다).")
print("   커널 쪽은 그냥 카이제곱으로 보면 크게 어긋나고, 평균·분산을 맞추면 붙는다.")

# === ③ 멱등에서 얼마나 떨어졌는가 ===================================
TS = np.linspace(0, 1, 21)
base = EIG["커널 (완만히 줄어듦)"]
base = base / base.max()
print("\n③ 멱등(0/1)에서 커널 모양으로 옮겨 가며")
print("   섞은 정도 t   분산/평균")
RATIO = []
for t in TS:
    lam = (1 - t) * EIG["멱등 (0 과 1 뿐)"] + t * base * R / base.sum() * 1.0
    lam = np.maximum(lam, 0)
    RATIO.append(2 * (lam ** 2).sum() / lam.sum())
for i in (0, 5, 10, 15, 20):
    print(f"   {TS[i]:>11.2f}   {RATIO[i]:>9.3f}")
print("   t = 0 에서만 정확히 2 다. 조금만 섞여도 2 에서 벗어난다.")

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.6, 5.2))
C_I, C_K = "#2ca02c", "#d62728"

for (name, lam), c, mk in zip(EIG.items(), (C_I, C_K), ("o", "s")):
    ax1.plot(np.arange(1, N + 1), lam, mk + "-", color=c, lw=2.2, ms=5,
             label=name)
ax1.set_xlabel("고윳값 번호")
ax1.set_ylabel("$\\lambda_i$")
ax1.set_xlim(0, 30)
ax1.set_title("① 고윳값의 모양", fontsize=12)
ax1.legend(loc="upper right", fontsize=10.5)
ax1.grid(alpha=0.25)

name_k = "커널 (완만히 줄어듦)"
s = SAMP[name_k]
lam = EIG[name_k]
tr1 = lam.sum()
nu, c = SATT[name_k]
xs = np.linspace(0, np.percentile(s, 99.5), 400)
ax2.hist(s, bins=120, density=True, color="#cccccc", edgecolor="white",
         linewidth=0.3, label="가중 카이제곱합 (참)")
ax2.plot(xs, stats.chi2.pdf(xs, tr1), lw=2.8, color=C_K,
         label=f"$\\chi^2_{{{tr1:.1f}}}$ 로 본 것")
ax2.plot(xs, stats.chi2.pdf(xs / c, nu) / c, lw=2.8, color="#1f77b4", ls="--",
         label="평균·분산을 맞춘 것")
ax2.set_xlabel("$z^T A z$")
ax2.set_ylabel("밀도")
ax2.set_xlim(0, xs.max())
ax2.set_title("② 커널 모양일 때", fontsize=12)
ax2.legend(loc="upper right", fontsize=9.5)
ax2.grid(alpha=0.25)

ax3.plot(TS, RATIO, "o-", color="#9467bd", lw=2.8, ms=8)
ax3.axhline(2.0, color=C_I, lw=2.4, ls="--")
ax3.text(0.35, 2.05, "$\\chi^2$ 이면 반드시 $2$", fontsize=11, color=C_I)
ax3.plot([0], [RATIO[0]], "o", color=C_I, ms=14, zorder=5)
ax3.annotate("멱등", (0, RATIO[0]), xytext=(0.6, -1.6),
             textcoords="offset fontsize", fontsize=10.5, color=C_I)
ax3.set_xlabel("커널 쪽으로 섞은 정도 $t$")
ax3.set_ylabel("$2\\,\\mathrm{tr}(A^2)\\,/\\,\\mathrm{tr}(A)$")
ax3.set_title("③ 분산이 평균의 두 배인가", fontsize=12)
ax3.grid(alpha=0.25)

fig.tight_layout()
out = "docs/ch00/linalg_regression/linalg_statistics/img/weighted_chisq.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
