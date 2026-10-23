r"""15.5 「Levene 검정 모의실험」 쪽의 상자에 들어가는 그림을 만든다.

브라운과 포사이드가 $1974$ 년에 한 비교를 다시 돌린다. 르빈 통계량의 **중심**을
평균(W0)·중앙값(W50)·$10\%$ 절사평균(W10) 셋으로 바꾸어 가며, 등분산인 자료에서
명목 $5\%$ 검정이 실제로 몇 퍼센트를 기각하는지 센다.

  요점은 「중앙값이 이긴다」가 아니라 **이기는 중심이 분포마다 다르다**는 것이다.
  대칭이고 꼬리가 보통이면 평균이, 치우치면 중앙값이, 꼬리가 두꺼우면 절사평균이
  명목값에 가깝다. 소프트웨어가 중앙값 하나를 기본값으로 고른 것은 그 가운데
  한 줄을 택한 것이다.

만드는 파일:
  ch15/robust_tests/img/levene_centering.png

실행:  python3 scripts/make_ch15_levene_centering.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.

주의: BLAS 스레드를 묶어 돌릴 것.
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
GREY, MUTED = "#90A4AE", "#B0BEC5"
COL = {"평균 (W0)": "#C62828",
       "중앙값 (W50)": "#1565C0",
       "10% 절사평균 (W10)": "#2E7D32"}

OUT = "docs/ch15/robust_tests/img/"
os.makedirs(OUT, exist_ok=True)

rng = np.random.default_rng(1974)
B, K, N = 20000, 3, 20
ALPHA = 0.05

CENTER = {"평균 (W0)": np.mean,
          "중앙값 (W50)": np.median,
          "10% 절사평균 (W10)": lambda a: stats.trim_mean(a, 0.1)}

# 등분산이므로 H0 가 참이다. 세 모집단은 브라운–포사이드가 든 세 가지 모양이다.
DIST = [("정규\n(대칭·보통 꼬리)", lambda m, n: rng.standard_normal((m, n))),
        ("카이제곱(4)\n(치우침)", lambda m, n: rng.chisquare(4, (m, n))),
        ("코시\n(두꺼운 꼬리)", lambda m, n: rng.standard_cauchy((m, n)))]


def size(groups_iter, center):
    """변환값에 일원분산분석을 돌려 기각 비율을 센다."""
    hit = 0
    for gs in groups_iter:
        z = [np.abs(g - center(g)) for g in gs]
        if stats.f_oneway(*z).pvalue < ALPHA:
            hit += 1
    return hit / B


rows = {}
for dname, gen in DIST:
    X = gen(B * K, N).reshape(B, K, N)
    rows[dname] = {cn: size((X[b] for b in range(B)), c)
                   for cn, c in CENTER.items()}

fig, ax = plt.subplots(figsize=(10.4, 4.8))
w, xs = 0.26, np.arange(len(DIST))
for t, (cname, col) in enumerate(COL.items()):
    vals = [rows[d][cname] for d, _ in DIST]
    pos = xs + (t - 1) * w
    ax.bar(pos, vals, w * 0.92, color=col, alpha=0.88, label=cname, zorder=3)
    for p, v in zip(pos, vals):
        ax.text(p, v + 0.006, f"{v:.3f}", ha="center", fontsize=9, color=INK)

ax.axhline(ALPHA, color=INK, lw=1.6, ls="--", zorder=4)
ax.text(len(DIST) - 0.42, ALPHA + 0.008, "명목 $0.05$", fontsize=10, color=INK)

# 분포마다 명목값에 가장 가까운 중심을 막대 위에 표시한다
for i, (d, _) in enumerate(DIST):
    best = min(COL, key=lambda c: abs(rows[d][c] - ALPHA))
    t = list(COL).index(best)
    ax.plot([xs[i] + (t - 1) * w], [rows[d][best] + 0.026], marker="v", ms=10,
            color=COL[best], zorder=5)

ax.set_xticks(xs)
ax.set_xticklabels([d for d, _ in DIST], fontsize=10)
ax.set_ylabel("등분산 자료를 기각한 비율")
ax.set_title("이기는 중심이 분포마다 다르다  (세모가 명목값에 가장 가까운 쪽)",
             color=INK, fontsize=12.5, pad=10)
ax.set_ylim(0, 0.32)
ax.legend(fontsize=9.5, framealpha=0.95, loc="upper left")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

fig.tight_layout()
path = OUT + "levene_centering.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
plt.close(fig)

print(f"집단 {K} 개, 집단당 n = {N}, 반복 {B:,}, 명목 {ALPHA}")
print(f"몬테카를로 표준오차 = {np.sqrt(ALPHA*(1-ALPHA)/B):.4f}\n")
# 한글은 고정폭에서 두 칸을 차지하므로 f-string 자리맞춤이 보이는 열과 어긋난다
print("  분포             평균 W0   중앙값 W50   10% 절사 W10")
for d, _ in DIST:
    lab = d.split("\n")[0]
    r = rows[d]
    print(f"  {lab:<12}"
          f"{r['평균 (W0)']:>9.4f}{r['중앙값 (W50)']:>11.4f}{r['10% 절사평균 (W10)']:>13.4f}")
print("\nsaved", path)
