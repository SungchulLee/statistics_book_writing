r"""16.4 Wilcoxon 순위합검정 쪽의 상자에 들어가는 그림을 만든다.

순위 자리에 정규점수를 넣으면(판 데르 바르던 검정) t 검정 대비 ARE 가 1 아래로
내려가지 않는다 — Chernoff-Savage(1958). 윌콕슨의 바닥이 108/125 인 것과 견준다.

만드는 파일:
  ch16/two_sample_nonparametric/img/normal_scores_floor.png

실행:  python3 scripts/make_ch16_normal_scores_floor.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""
import os

import numpy as np
from scipy import integrate, stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

INK = "#37474F"
BLUE = "#1565C0"
ORANGE = "#E65100"
GREEN = "#33691E"
MUTED = "#90A4AE"
RED = "#D32F2F"

OUT = "docs/ch16/two_sample_nonparametric/img/"
os.makedirs(OUT, exist_ok=True)

phi, Phi, Phinv = stats.norm.pdf, stats.norm.cdf, stats.norm.ppf
R5 = np.sqrt(5.0)


def are_wilcoxon(f, lo, hi):
    """12 * sigma^2 * (적분 f^2)^2 — 분산을 1 로 맞춘 밀도를 넣는다."""
    I = integrate.quad(lambda x: f(x) ** 2, lo, hi, limit=500)[0]
    return 12 * I ** 2


def are_vdw(f, F, lo, hi):
    """sigma^2 * (적분 f^2 / phi(Phi^{-1}(F)))^2."""
    def g(x):
        u = F(x)
        if u <= 1e-13 or u >= 1 - 1e-13:
            return 0.0
        d = phi(Phinv(u))
        return 0.0 if d < 1e-300 else f(x) ** 2 / d
    I = integrate.quad(g, lo, hi, limit=500)[0]
    return I ** 2


b = 1 / np.sqrt(2)
FAMS = [
    ("포물선", lambda x: 3 / (4 * R5) * (1 - x * x / 5) if abs(x) <= R5 else 0.0,
     lambda x: 0.0 if x < -R5 else (1.0 if x > R5 else
                                    0.5 + 3 / (4 * R5) * (x - x ** 3 / 15)),
     -R5, R5),
    ("정규", lambda x: phi(x), lambda x: Phi(x), -12, 12),
    ("로지스틱", None, None, None, None),          # 아래에서 채운다
    ("Laplace", lambda x: np.exp(-abs(x) / b) / (2 * b),
     lambda x: 0.5 * np.exp(x / b) if x < 0 else 1 - 0.5 * np.exp(-x / b),
     -40, 40),
    ("$t(5)$", None, None, None, None),
    ("$t(3)$", None, None, None, None),
]
s = np.sqrt(3) / np.pi
FAMS[2] = ("로지스틱",
           lambda x: np.exp(-x / s) / (s * (1 + np.exp(-x / s)) ** 2),
           lambda x: 1 / (1 + np.exp(-x / s)), -60, 60)
for i, nu in ((4, 5), (5, 3)):
    c = np.sqrt((nu - 2) / nu)
    FAMS[i] = (FAMS[i][0],
               (lambda n, cc: (lambda x: stats.t.pdf(x / cc, n) / cc))(nu, c),
               (lambda n, cc: (lambda x: stats.t.cdf(x / cc, n)))(nu, c), -90, 90)

print(f"{'분포':10s} {'ARE 윌콕슨':>12s} {'ARE 정규점수':>14s}")
names, aw, av = [], [], []
for nm, f, F, lo, hi in FAMS:
    w, v = are_wilcoxon(f, lo, hi), are_vdw(f, F, lo, hi)
    names.append(nm)
    aw.append(w)
    av.append(v)
    print(f"{nm:10s} {w:12.4f} {v:14.4f}")
print(f"(확인) 3/pi = {3 / np.pi:.4f},  108/125 = {108 / 125:.4f}")

# t 족을 따라가며
NUS = np.unique(np.round(np.logspace(np.log10(3), np.log10(300), 26)).astype(int))
tw, tv = [], []
for nu in NUS:
    c = np.sqrt((nu - 2) / nu)
    f = lambda x, n=nu, cc=c: stats.t.pdf(x / cc, n) / cc
    F = lambda x, n=nu, cc=c: stats.t.cdf(x / cc, n)
    tw.append(are_wilcoxon(f, -90, 90))
    tv.append(are_vdw(f, F, -90, 90))
print(f"\nt 족: nu = {NUS[0]} 에서 {tw[0]:.3f} / {tv[0]:.3f}, "
      f"nu = {NUS[-1]} 에서 {tw[-1]:.4f} / {tv[-1]:.4f}")

fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.8, 4.5))

# ---------------------------------------------------- 왼쪽: 여섯 분포
x = np.arange(len(names))
axL.bar(x - 0.19, aw, width=0.36, color=BLUE, edgecolor=INK, linewidth=0.5,
        zorder=3, label="윌콕슨 순위합")
axL.bar(x + 0.19, av, width=0.36, color=GREEN, edgecolor=INK, linewidth=0.5,
        zorder=3, label="정규점수 (판 데르 바르던)")
# 막대가 바닥부터 차 있어 빈 자리가 없다 — 기준선은 범례로 설명한다
axL.axhline(1.0, color=RED, lw=1.4, ls="--", zorder=4,
            label="$1$ — 정규점수의 바닥 (Chernoff-Savage)")
axL.axhline(108 / 125, color=BLUE, lw=1.3, ls=":", zorder=4,
            label="$108/125 = 0.864$ — 윌콕슨의 바닥")
for xi, v in zip(x, aw):
    axL.text(xi - 0.19, v + 0.03, f"{v:.3f}", ha="center", fontsize=8,
             color=BLUE)
for xi, v in zip(x, av):
    axL.text(xi + 0.19, v + 0.03, f"{v:.3f}", ha="center", fontsize=8,
             color=GREEN)
axL.set_xticks(x)
axL.set_xticklabels(names, fontsize=9.5)
axL.set_ylim(0, 2.75)
axL.set_ylabel("$t$ 검정 대비 ARE")
axL.set_title("바닥이 어디인가", color=INK, fontsize=12, pad=9)
axL.legend(loc="upper left", fontsize=8.5, framealpha=0.95)
for s_ in ("top", "right"):
    axL.spines[s_].set_visible(False)

# ---------------------------------------------------- 오른쪽: t 족을 따라
axR.plot(NUS, tv, "o-", color=GREEN, lw=2.2, ms=4, zorder=4,
         label="정규점수 (판 데르 바르던)")
axR.plot(NUS, tw, "s--", color=BLUE, lw=2.2, ms=4, zorder=3,
         label="윌콕슨 순위합")
axR.axhline(1.0, color=RED, lw=1.4, ls="--", zorder=2)
axR.axhline(3 / np.pi, color=MUTED, lw=1.1, ls=":", zorder=2)
axR.text(310, 1.015, "$1$", color=RED, fontsize=9.5, va="bottom")
axR.text(310, 3 / np.pi - 0.015, "$3/\\pi$", color=MUTED, fontsize=9.5,
         va="top")
axR.set_xscale("log")
show = [3, 5, 10, 20, 50, 100, 300]
axR.set_xticks(show)
axR.set_xticklabels([str(v) for v in show], fontsize=9)
axR.minorticks_off()
axR.set_xlim(3, 340)
axR.set_ylim(0.85, 2.0)
axR.set_xlabel("$t(\\nu)$ 의 자유도 $\\nu$ (가로축은 로그눈금, 오른쪽 끝이 정규)")
axR.set_ylabel("$t$ 검정 대비 ARE")
axR.set_title("꼬리를 얇게 해 가며 — 두 바닥이 갈린다", color=INK,
              fontsize=12, pad=9)
axR.legend(loc="upper right", fontsize=9, framealpha=0.95)
for s_ in ("top", "right"):
    axR.spines[s_].set_visible(False)

fig.tight_layout()
path = OUT + "normal_scores_floor.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
plt.close(fig)
print("saved", path)
