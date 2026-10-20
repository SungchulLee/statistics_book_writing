r"""질량을 모르는 혹을 찾을 때, 가능도비의 $\chi^2$ 근사는 쓸 수 없다.

질량 $m$ 은 **대립가설 아래에서만 존재하는 모수**다. 윌크스 정리의 전제가 깨지므로
$\max_m q(m)$ 의 귀무분포를 식으로 쓸 수 없고, 배경만인 모형에서 **가짜 실험을 만들어**
재는 수밖에 없다 — 모수 붓스트랩이다.

① 배경만인 가짜 실험의 국소 유의도 곡선 — 아무것도 없어도 어딘가는 올라간다.
② 고정 질량의 귀무분포와 최댓값의 귀무분포, 그리고 상향교차 외삽.
③ 탐색 창을 넓힐수록 같은 국소 유의도가 더 깎인다.

만드는 파일: docs/ch17/bootstrap/img/look_elsewhere.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from scipy import stats  # noqa: E402

RNG = np.random.default_rng(20261020)
LO, HI, DM = 110.0, 150.0, 0.25      # 탐색 창 (GeV) 과 구간 폭
RES = 1.5                             # 질량 분해능 (GeV)
N_BKG = 100_000                       # 창 안의 배경 사건 수
EDGE = 3.0                            # 가장자리에서 띄울 거리
TOYS, BATCH = 200_000, 20_000
Z0 = 1.0                              # 상향교차를 셀 문턱

edges = np.arange(LO, HI + DM / 2, DM)
mid = (edges[:-1] + edges[1:]) / 2
masses = np.arange(LO + EDGE, HI - EDGE + DM / 2, DM)

# --- 배경만인 모형 ---------------------------------------------------
shape = np.exp(-(mid - LO) / 30.0)
b = N_BKG * shape / shape.sum()

# --- 질량마다의 점수검정 ---------------------------------------------
# n_i ~ Poisson(b_i) 이고 신호 틀 s_i(m) 은 합이 1 인 가우스다.
# 점수 U(m) = sum_i s_i(m)(n_i - b_i)/b_i, 정보 I(m) = sum_i s_i(m)^2/b_i,
# Z(m) = U/sqrt(I) 는 귀무가설 아래에서 질량마다 표준정규다.
S = np.exp(-0.5 * ((mid[None, :] - masses[:, None]) / RES) ** 2)
S /= S.sum(axis=1, keepdims=True)
INFO = (S**2 / b[None, :]).sum(axis=1)
A = (S / b[None, :]) / np.sqrt(INFO)[:, None]   # Z = A @ (n - b)


def toy_Z(n_toys, rng, mu=0.0, m_sig=125.0):
    """가짜 실험 n_toys 판의 Z(m) 곡선을 만든다 (mu 는 넣을 신호 세기)."""
    lam = np.broadcast_to(b, (n_toys, len(b))).copy()
    if mu > 0:
        sig = np.exp(-0.5 * ((mid - m_sig) / RES) ** 2)
        lam = lam + mu * sig / sig.sum()
    n = rng.poisson(lam)
    return (n - b) @ A.T


def upcrossings(Z, level):
    """곡선이 level 을 아래에서 위로 넘은 횟수."""
    return np.sum((Z[:, :-1] < level) & (Z[:, 1:] >= level), axis=1)


# --- 직접 센 최댓값 분포와 상향교차 ----------------------------------
zmax, ncross = [], []
for _ in range(TOYS // BATCH):
    Z = toy_Z(BATCH, RNG)
    zmax.append(Z.max(axis=1))
    ncross.append(upcrossings(Z, Z0))
zmax = np.concatenate(zmax)
ncross = np.concatenate(ncross)
N1 = ncross.mean()
print(f"문턱 Z = {Z0} 의 평균 상향교차 수 ⟨N⟩ = {N1:.3f}  (가짜 실험 {TOYS:,} 판)")


def p_global(z, n1=N1, z0=Z0):
    """그로스–비텔스 외삽. q = Z^2 일 때
    P(max q > u) ≈ P(chi2_1 > u) + ⟨N_u0⟩ exp(-(u - u0)/2) 이다."""
    return stats.norm.sf(z) + n1 * np.exp(-(z**2 - z0**2) / 2)


def z_global(z_local, n1=N1):
    return stats.norm.isf(np.clip(p_global(z_local, n1), 1e-300, 1.0))


# --- ③ 창 너비에 따른 시행계수 ---------------------------------------
# 같은 가짜 실험에서 **부분 창**만 보고 상향교차를 세면, 너비에 따른
# 변화를 새 모의실험 없이 잴 수 있다.
widths, rates = [], []
for w in (4, 8, 12, 17, 23, 29, 34):
    k = int(w / DM)
    sel = slice(0, k + 1)
    Zs = []
    for _ in range(4):
        Zs.append(upcrossings(toy_Z(BATCH, RNG)[:, sel], Z0).mean())
    widths.append(masses[k] - masses[0])
    rates.append(np.mean(Zs))
widths, rates = np.array(widths), np.array(rates)
slope = np.linalg.lstsq(widths[:, None], rates, rcond=None)[0][0]
print(f"상향교차 밀도 {slope:.4f} 회/GeV  (분해능 {RES} GeV 기준)")

Z_LOCAL = 5.0
W_GRID = np.array([5, 10, 15, 20, 30, 50, 100, 200, 490.0])
ZG = np.array([z_global(Z_LOCAL, slope * w) for w in W_GRID])

# === 그림 ==========================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.4, 4.9))

Zb = toy_Z(3, np.random.default_rng(11))
for i in range(3):
    ax1.plot(masses, Zb[i], color="#9e9e9e", lw=1.3, alpha=0.9,
             label="배경만 (가짜 실험)" if i == 0 else None)
Zs1 = toy_Z(1, np.random.default_rng(5), mu=700, m_sig=126.0)[0]
ax1.plot(masses, Zs1, color="#d62728", lw=2.2, label="신호를 넣은 한 판")
ax1.axhline(Z0, color="#1f77b4", lw=1.4, ls="--")
ax1.text(LO + EDGE + 0.4, Z0 + 0.18, f"문턱 $Z = {Z0:.0f}$", fontsize=10,
         color="#1f77b4")
cr = np.where((Zb[0][:-1] < Z0) & (Zb[0][1:] >= Z0))[0]
ax1.plot(masses[cr + 1], np.full(len(cr), Z0), "o", color="#1f77b4", ms=6,
         zorder=5)
ax1.set_xlabel("혹을 찾는 질량 $m$ (GeV)")
ax1.set_ylabel("국소 유의도 $Z(m)$")
ax1.set_title("① 아무것도 없어도 어딘가는 올라간다", fontsize=12)
ax1.legend(loc="upper right", fontsize=9.5)
ax1.grid(alpha=0.25)

zs = np.linspace(0, 6.2, 300)
ax2.semilogy(zs, stats.norm.sf(zs), color="#9e9e9e", lw=2.2,
             label="질량을 **고정**했을 때 ($N(0,1)$)".replace("**", ""))
srt = np.sort(zmax)[::-1]
emp = np.arange(1, len(srt) + 1) / len(srt)
keep = emp >= 3 / len(srt)
ax2.semilogy(srt[keep], emp[keep], color="#d62728", lw=2.6,
             label="$\\max_m Z$ — 직접 센 값")
ax2.semilogy(zs, p_global(zs), color="#1f77b4", lw=1.8, ls="--",
             label="상향교차 외삽")
ax2.axvline(5.0, color="black", lw=1.3, ls=":")
ax2.text(4.90, 1.6e-8, "국소 $5\\sigma$", rotation=90, fontsize=10,
         ha="right", va="bottom")
ax2.annotate("가짜 실험 $20$ 만 판으로\n직접 셀 수 있는 한계",
             # srt 는 내림차순이므로 그려진 곡선의 오른쪽 끝은 [0] 이다
             xy=(srt[keep][0], emp[keep][0]), xytext=(3.30, 6e-8),
             fontsize=9.5, color="#d62728", ha="left",
             arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.0))
ax2.set_xlim(1, 6.2)
ax2.set_ylim(1e-8, 1.5)
ax2.set_xlabel("유의도 $Z$")
ax2.set_ylabel("그보다 클 확률")
ax2.set_title("② 최댓값의 꼬리는 훨씬 두껍다", fontsize=12)
ax2.set_yticks([1e-8, 1e-6, 1e-4, 1e-2, 1])
ax2.set_yticklabels(["$10^{-8}$", "$10^{-6}$", "$10^{-4}$", "$10^{-2}$", "$1$"])
ax2.minorticks_off()
ax2.legend(loc="upper right", fontsize=9)
ax2.grid(alpha=0.25, which="major")

ax3.semilogx(W_GRID, ZG, "o-", color="#d62728", lw=2.4, ms=6)
ax3.axhline(Z_LOCAL, color="#9e9e9e", lw=1.6, ls="--")
ax3.text(6, Z_LOCAL + 0.04, f"국소 ${Z_LOCAL:.1f}\\sigma$", fontsize=10,
         color="#555555")
for w, lab in ((15, "CMS $115$–$130$"), (35, "CMS $110$–$145$"),
               (490, "ATLAS $110$–$600$")):
    ax3.axvline(w, color="#1f77b4", lw=1.0, ls=":")
    ax3.text(w * 1.08, 3.55, lab, rotation=90, fontsize=9, color="#1f77b4",
             va="bottom")
ax3.set_xlabel("탐색한 질량 창의 너비 (GeV)")
ax3.set_ylabel("전역 유의도 $Z$")
ax3.set_title("③ 넓게 볼수록 같은 혹이 작아진다", fontsize=12)
ax3.set_xticks([5, 10, 30, 100, 490])
ax3.set_xticklabels(["$5$", "$10$", "$30$", "$100$", "$490$"])
ax3.minorticks_off()
ax3.set_ylim(3.4, 5.2)
ax3.grid(alpha=0.25, which="major")

fig.tight_layout()
out = "docs/ch17/bootstrap/img/look_elsewhere.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)

# === 본문에 적을 수 =================================================
print(f"\n창 {LO}–{HI} GeV, 분해능 {RES} GeV, 질량 눈금 {len(masses)} 개")
print(f"상향교차 수의 분포: 평균 {N1:.2f}, 중앙값 {np.median(ncross):.0f}, "
      f"최대 {ncross.max()}")

print(f"\n② 외삽이 직접 센 값과 맞는가 (가짜 실험 {TOYS:,} 판)")
print(f"{'Z':>5}{'직접 센 P(maxZ>z)':>20}{'외삽':>12}{'고정 질량':>12}{'시행계수':>10}")
for z in (2.0, 2.5, 3.0, 3.5, 4.0):
    emp_p = np.mean(zmax > z)
    print(f"{z:>5.1f}{emp_p:>20.5f}{p_global(z):>12.5f}"
          f"{stats.norm.sf(z):>12.2e}{emp_p / stats.norm.sf(z):>10.1f}")

print(f"\n③ 국소 {Z_LOCAL} σ 를 창 너비별로 전역으로 옮기면")
print(f"{'창 너비 (GeV)':>14}{'⟨N⟩':>9}{'전역 Z':>10}{'깎인 몫':>10}")
for w, zg in zip(W_GRID, ZG):
    print(f"{w:>14.0f}{slope * w:>9.2f}{zg:>10.2f}{Z_LOCAL - zg:>10.2f}")

print("\n   실제 보고된 짝과 나란히 (이 장난감의 분해능은 1.5 GeV 고정이다)")
print(f"{'':>26}{'보고된 전역':>12}{'이 장난감':>12}")
for lab, zl, w, rep in (("CMS  국소 5.0σ, 창 15 GeV", 5.0, 15, 4.6),
                        ("CMS  국소 5.0σ, 창 35 GeV", 5.0, 35, 4.5),
                        ("ATLAS 국소 5.9σ, 창 490 GeV", 5.9, 490, 5.1)):
    print(f"{lab:>26}{rep:>12.1f}{z_global(zl, slope * w):>12.2f}")
