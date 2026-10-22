r"""표본 $n$ 개가 몇 개 값인가 — 중요도 표집의 유효표본수.

이 쪽 $\S 3$ 은 점근 상대효율을 「같은 정밀도를 내는 데 드는 표본 수의 비」로 읽는다.
다른 분포에서 뽑은 표본을 가중치로 고쳐 쓰는 **중요도 표집**에서는 그 비가 눈에
보이는 수로 나온다 — **유효표본수** $\mathrm{ESS} = (\sum w)^2/\sum w^2$ 다.

① 제안분포가 조금만 어긋나도 ESS 가 무너진다. 차원이 늘면 더 빨리.
② 분산의 비가 $n/\mathrm{ESS}$ 와 맞는다 — ARE 를 직접 잰 셈이다.
③ 가중치를 자르면 분산이 줄고 편향이 생긴다. MSE 에 바닥이 생긴다.

만드는 파일: docs/ch05/mean/img/importance_ess.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

RNG = np.random.default_rng(0)
N = 2000


def logw(x, delta):
    """목표 N(0, I) 과 제안 N(delta*1, I) 의 로그 가중치."""
    return (-0.5 * (x ** 2).sum(-1)
            + 0.5 * ((x - delta) ** 2).sum(-1))


def ess_frac(d, delta, rep=200):
    out = np.empty(rep)
    for i in range(rep):
        x = RNG.standard_normal((N, d)) + delta
        lw = logw(x, delta)
        w = np.exp(lw - lw.max())
        out[i] = w.sum() ** 2 / (w ** 2).sum() / N
    return float(out.mean())


# === ① ESS 가 무너진다 ===============================================
DELTAS = np.linspace(0, 1.6, 17)
DIMS = [1, 5, 20]
print(f"① 목표 N(0, I_d), 제안 N(delta·1, I_d), n = {N:,}")
print(f"{'delta':>8}" + "".join(f"{f'd={d}':>12}" for d in DIMS))
ESS = {d: [] for d in DIMS}
for dl in DELTAS:
    for d in DIMS:
        ESS[d].append(ess_frac(d, dl))
    if np.isclose(dl % 0.4, 0, atol=1e-9) or dl == DELTAS[-1]:
        print(f"{dl:>8.2f}" + "".join(f"{ESS[d][-1]:>12.4f}" for d in DIMS))
SHOW = [0.4, 0.8, 1.2]
print("   상자에 싣는 세 칸 (ESS/n)")
print(f"{'':>8}" + "".join(f"{f'delta={v}':>14}" for v in SHOW))
TAB = {}
for d in DIMS:
    TAB[d] = [ESS[d][int(np.argmin(np.abs(DELTAS - v)))] for v in SHOW]
    print(f"{f'd={d}':>8}" + "".join(f"{v:>14.4f}" for v in TAB[d])
          + f"   | n*ESS at 0.8 = {N * TAB[d][1]:.0f}")
for d in DIMS:
    i = int(np.argmax(np.array(ESS[d]) < 0.1))
    print(f"   d = {d:>2} : ESS 가 10% 아래로 떨어지는 delta = "
          f"{DELTAS[i] if ESS[d][i] < 0.1 else float('nan'):.2f}")
print("   가중치의 분산은 차원에 지수로 쌓인다 — 좌표마다 조금씩 어긋나기 때문이다.")

# === ② 분산의 비가 n/ESS 와 맞는가 ===================================
D2, REP2 = 5, 3000
print(f"\n② d = {D2} 에서 E[X_1] = 0 을 재는 두 방법의 분산 (반복 {REP2:,})")
print(f"{'delta':>8}{'직접 표집 분산':>16}{'중요도 분산':>14}{'비':>9}{'n/ESS':>9}")
VR, NE = [], []
for dl in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]:
    direct = np.empty(REP2)
    imp = np.empty(REP2)
    for i in range(REP2):
        z = RNG.standard_normal((N, D2))
        direct[i] = z[:, 0].mean()
        x = RNG.standard_normal((N, D2)) + dl
        lw = logw(x, dl)
        w = np.exp(lw)
        imp[i] = (w * x[:, 0]).sum() / w.sum()
    vr = float(imp.var() / direct.var())
    ne = 1.0 / ess_frac(D2, dl, rep=200)
    VR.append(vr)
    NE.append(ne)
    print(f"{dl:>8.2f}{direct.var():>16.3e}{imp.var():>14.3e}{vr:>9.2f}{ne:>9.2f}")
print("   비가 n/ESS 를 따라간다. 「표본 n 개가 몇 개 값인가」가 바로 이 수다.")

# === ③ 가중치를 자르면 ===============================================
D3, DL3, REP3 = 5, 1.0, 4000
CLIPS = np.r_[np.logspace(-0.3, 2.2, 14), np.inf]
print(f"\n③ d = {D3}, delta = {DL3} 에서 가중치를 c 로 자른다")
print(f"{'c':>9}{'편향':>12}{'표준편차':>12}{'MSE':>12}")
BI, SD, MS = [], [], []
for c in CLIPS:
    est = np.empty(REP3)
    for i in range(REP3):
        x = RNG.standard_normal((N, D3)) + DL3
        w = np.exp(logw(x, DL3))
        w = np.minimum(w, c)
        est[i] = (w * x[:, 0]).sum() / w.sum()
    b = float(est.mean())
    sd = float(est.std())
    BI.append(abs(b))
    SD.append(sd)
    MS.append(b ** 2 + sd ** 2)
    lab = "자르지 않음" if np.isinf(c) else f"{c:.2f}"
    print(f"{lab:>9}{b:>12.4f}{sd:>12.4f}{MS[-1]:>12.3e}")
best = int(np.argmin(MS))
print(f"   가장 낮은 MSE 는 c = "
      f"{'자르지 않음' if np.isinf(CLIPS[best]) else f'{CLIPS[best]:.2f}'} 에서 "
      f"{MS[best]:.3e} 이고, 자르지 않을 때의 {MS[best] / MS[-1]:.2f} 배다.")

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.6, 5.2))
COL = {1: "#9ecae1", 5: "#4292c6", 20: "#08519c"}

for d in DIMS:
    ax1.semilogy(DELTAS, ESS[d], "o-", color=COL[d], lw=2.4, ms=6,
                 label=f"$d = {d}$")
ax1.axhline(0.1, color="#d62728", lw=1.5, ls="--")
ax1.text(1.58, 0.125, "$\\mathrm{ESS}$ 가 $10\\%$", color="#d62728",
         fontsize=10.5, ha="right")
ax1.set_xlabel("제안분포가 어긋난 정도 $\\delta$ (좌표마다)")
ax1.set_ylabel("$\\mathrm{ESS}/n$")
ax1.set_yticks([1e-3, 1e-2, 1e-1, 1])
ax1.set_yticklabels(["$10^{-3}$", "$1\\%$", "$10\\%$", "$1$"])
ax1.minorticks_off()
ax1.set_ylim(5e-4, 1.6)
ax1.set_title("① 조금만 어긋나도 무너진다", fontsize=12)
ax1.legend(loc="lower left", fontsize=10.5)
ax1.grid(alpha=0.25)

dls = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
ax2.plot(dls, VR, "o", color="#d62728", ms=11, label="분산의 비 (직접 대비)")
ax2.plot(dls, NE, "-", color="#333333", lw=2.2, label="$n/\\mathrm{ESS}$")
ax2.set_yscale("log")
ax2.set_xlabel("제안분포가 어긋난 정도 $\\delta$")
ax2.set_ylabel("같은 정밀도에 드는 표본 수의 배수")
ax2.set_yticks([1, 10, 100])
ax2.set_yticklabels(["$1$", "$10$", "$10^{2}$"])
ax2.minorticks_off()
ax2.set_title(f"② 재어 보면 $n/\\mathrm{{ESS}}$ 다 ($d = {D2}$)", fontsize=12)
ax2.legend(loc="upper left", fontsize=10.5)
ax2.grid(alpha=0.25)

cs = np.array([1e3 if np.isinf(c) else c for c in CLIPS])
ax3.loglog(cs, np.array(SD) ** 2, "s-", color="#1f77b4", lw=2.2, ms=6,
           label="분산")
ax3.loglog(cs, np.array(BI) ** 2, "^-", color="#ff7f0e", lw=2.2, ms=6,
           label="편향$^2$")
ax3.loglog(cs, MS, "o-", color="#d62728", lw=2.8, ms=7, label="MSE")
ax3.plot([cs[best]], [MS[best]], "*", color="#d62728", ms=20, zorder=5)
ax3.axvline(1e3, color="#888888", lw=1.2, ls=":")
ax3.text(0.965, 0.04, "자르지 않음", transform=ax3.transAxes,
         fontsize=10, color="#555555", ha="right", rotation=90)
ax3.set_xlabel("가중치를 자르는 문턱 $c$")
ax3.set_ylabel("평균제곱오차와 그 두 조각")
ax3.set_xticks([1, 10, 100, 1000])
ax3.set_xticklabels(["$1$", "$10$", "$10^{2}$", "$\\infty$"])
# 로그 눈금의 마이너스는 mathtext 라 두부가 된다 — 눈금을 직접 적는다
ax3.set_yticks([1e-3, 1e-2, 1e-1])
ax3.set_yticklabels(["$10^{-3}$", "$10^{-2}$", "$10^{-1}$"])
ax3.minorticks_off()
ax3.set_title(f"③ 자르면 편향을 사고 분산을 판다", fontsize=12)
ax3.legend(loc="lower left", fontsize=10.5)
ax3.grid(alpha=0.25)

fig.tight_layout()
out = "docs/ch05/mean/img/importance_ess.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
