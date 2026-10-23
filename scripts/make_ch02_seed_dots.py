r"""2.5절 스트립·스웜 쪽의 「AI로 보는 통계학」 그림 — 시드마다 찍은 점.

만드는 파일:
  docs/ch02/visualization/img/seed_dots.png

실행:  python3 scripts/make_ch02_seed_dots.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
주의:  한글은 수식 $...$ 바깥에만 쓴다.
"""
import os

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

INK = "#37474F"
BLUE, BLUE_F = "#1565C0", "#DCEBFB"
ORANGE, ORANGE_F = "#E65100", "#FFE0B2"
GREEN = "#2E7D32"
RED = "#D32F2F"
MUTED = "#90A4AE"

OUT = "docs/ch02/visualization/img/"
os.makedirs(OUT, exist_ok=True)

rng = np.random.default_rng(7)
MU_A, MU_B, SD = 0.800, 0.805, 0.010   # 참값: ㄴ 가 0.5 %p 낫다
K = 5


def tidy(ax):
    ax.tick_params(labelsize=9, colors=INK, length=3)
    ax.spines[["top", "right"]].set_visible(False)
    for s in ax.spines.values():
        s.set_color(MUTED)


fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))

# ── ① 다섯 번 돌린 점 ───────────────────────────────────────────────────
ax = axes[0]
a = rng.normal(MU_A, SD, K)
b = rng.normal(MU_B, SD, K)
# 세로축을 퍼센트로 적는다 — 글상자의 수와 단위를 맞춘다
for x, v, col, fill, name in [(0, a * 100, BLUE, BLUE_F, "방법 ㄱ"),
                              (1, b * 100, ORANGE, ORANGE_F, "방법 ㄴ")]:
    jig = np.linspace(-0.09, 0.09, K)
    ax.scatter(x + jig, v, s=60, color=fill, edgecolor=col, lw=1.4, zorder=3)
    ax.plot([x - 0.2, x + 0.2], [v.mean()] * 2, color=col, lw=2.4, zorder=4)
    ax.text(x, 77.45, f"{name}\n평균 {v.mean():.2f}", ha="center",
            fontsize=9.5, color=col)
    ax.plot([x + 0.26], [v.max()], "v", ms=7, color=RED, zorder=5)
ax.text(1.32, b.max() * 100, " 보고되는 최고 점수", color=RED, fontsize=9,
        va="center")
for mu, col in [(MU_A, BLUE), (MU_B, ORANGE)]:
    ax.axhline(mu * 100, color=col, ls=":", lw=1.1)
ax.text(-0.42, MU_B * 100 + 0.12, "참값 (차이 0.5 %p)", fontsize=8.8, color=MUTED)
ax.set_xlim(-0.45, 1.75)
ax.set_ylim(77.2, 82.8)
ax.set_xticks([])
ax.set_ylabel("점수 (퍼센트)", fontsize=9.5, color=INK)
ax.set_title(f"① 시드를 바꿔 {K} 번 돌린 점수", fontsize=10.5, color=INK)
tidy(ax)

# ── ② 승자를 잘못 고를 확률 ─────────────────────────────────────────────
ax = axes[1]
ks = np.arange(2, 41)
err = stats.norm.cdf(0, MU_B - MU_A, SD * np.sqrt(2 / ks))
ax.plot(ks, err, color=BLUE, lw=2.0)
ax.axhline(0.05, color=MUTED, ls=":", lw=1.0)
for k in (3, 5, 10, 20):
    e = stats.norm.cdf(0, MU_B - MU_A, SD * np.sqrt(2 / k))
    ax.plot([k], [e], "o", ms=6, color=BLUE)
    ax.annotate(f"{k} 번 — {e*100:.1f} 퍼센트", xy=(k, e), xytext=(6, 6),
                textcoords="offset points", fontsize=8.8, color=BLUE)
ax.set_xlabel("방법마다 돌리는 횟수", fontsize=9.5, color=INK)
ax.set_ylabel("평균으로 견주어 진 쪽을 고를 확률", fontsize=9.5, color=INK)
ax.set_title("② 평균으로 견주어도 자주 뒤집힌다", fontsize=10.5, color=INK)
ax.set_ylim(0, 0.42)
tidy(ax)

# ── ③ 최고 점수의 거품 ──────────────────────────────────────────────────
ax = axes[2]
kk = np.arange(1, 41)
REP = 20000
infl = np.array([(rng.normal(MU_A, SD, (REP, k)).max(1).mean() - MU_A) * 100
                 for k in kk])
ax.plot(kk, infl, color=ORANGE, lw=2.0)
ax.axhline((MU_B - MU_A) * 100, color=GREEN, lw=1.6, ls="--")
ax.text(40, (MU_B - MU_A) * 100 + 0.04, "참 차이 0.50 %p", ha="right",
        fontsize=9, color=GREEN)
for k in (3, 5, 10, 20):
    ax.plot([k], [infl[k - 1]], "o", ms=6, color=ORANGE)
    ax.annotate(f"{k} 번 — +{infl[k-1]:.2f} %p", xy=(k, infl[k - 1]),
                xytext=(6, -12), textcoords="offset points", fontsize=8.8,
                color=ORANGE)
ax.set_xlabel("방법마다 돌리는 횟수", fontsize=9.5, color=INK)
ax.set_ylabel("최고 점수가 참 평균보다 높은 정도", fontsize=9.5, color=INK)
ax.set_title("③ 많이 돌릴수록 최고 점수는 더 부푼다", fontsize=10.5, color=INK)
ax.set_ylim(0, 2.2)
tidy(ax)

fig.tight_layout()
path = OUT + "seed_dots.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
plt.close(fig)
print("saved", path)
print(f"① 의 표본: ㄱ 평균 {a.mean()*100:.2f}, 최고 {a.max()*100:.2f} / "
      f"ㄴ 평균 {b.mean()*100:.2f}, 최고 {b.max()*100:.2f}")
for k in (3, 5, 10, 20):
    print(f"  {k:2d} 번: 뒤집힐 확률 {stats.norm.cdf(0, MU_B-MU_A, SD*np.sqrt(2/k)):.3f}, "
          f"최고 점수 거품 +{infl[k-1]:.2f} %p")
