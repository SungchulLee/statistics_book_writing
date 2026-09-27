r"""출판 편향 쪽의 깔때기 그림을 한글이 보이게 다시 그린다.

docs/ch01/classical/img/publication_bias_244.png 는 한글 폰트를 지정하지
않은 채로 만들어져 모든 한글이 두부(□)로 깨져 있었다. 같은 쪽 연습문제 7
의 코드를 그대로 쓰되 폰트만 지정해 다시 그린다. 시드와 계산이 같으므로
그림의 내용은 그 쪽이 보고한 수치와 그대로 맞는다.

실행:  python3 scripts/redraw_publication_bias_funnel.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       그림은 PNG로 커밋되므로 CI에서 다시 그리지 않는다.
"""

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"   # 한글 폰트
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch01/classical/img/publication_bias_244.png"

rng = np.random.default_rng(0)


def run_studies(K=400, d=0.3):
    """표본 크기가 제각각인 연구 K 편을 돌린다. (그 쪽 코드와 동일)"""
    ns = rng.integers(10, 300, K)
    eff = np.empty(K)
    se = np.empty(K)
    pv = np.empty(K)
    for i, n in enumerate(ns):
        c = rng.normal(0, 1, n)
        t = rng.normal(d, 1, n)
        sd = np.sqrt((c.var(ddof=1) + t.var(ddof=1)) / 2)
        eff[i] = (t.mean() - c.mean()) / sd
        se[i] = np.sqrt(2 / n)               # 코헨 d 의 근사 표준오차
        pv[i] = stats.ttest_ind(t, c)[1]
    return eff, se, pv


eff, se, pv = run_studies()
published = pv < 0.05

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)
for ax, mask, title in [
        (axes[0], np.ones(len(eff), bool), "수행된 모든 연구 — 대칭"),
        (axes[1], published, "유의한 연구만 — 왼쪽 아래가 비었다")]:
    ax.scatter(eff[mask], 1 / se[mask], s=14, alpha=0.6)
    ax.axvline(0.3, color="red", ls="--", lw=1.5, label="참 효과 0.3")
    ax.set_xlabel("효과크기 (코헨의 d)")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
axes[0].set_ylabel("정밀도 (1 / SE)")
fig.tight_layout()
fig.savefig(OUT, dpi=170, facecolor="white", bbox_inches="tight")
plt.close(fig)
print(f"saved {OUT}")

for label, mask in [("전체", np.ones(len(eff), bool)),
                    ("출판된 것만", published)]:
    print(f"  {label}: {mask.sum()}편, 평균 효과크기 {eff[mask].mean():.3f}")
