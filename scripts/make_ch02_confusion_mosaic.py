r"""2.5절 모자이크 그림 쪽의 「AI로 보는 통계학」 그림 — 혼동행렬을 모자이크로 읽는다.

만드는 파일:
  docs/ch02/visualization/img/confusion_mosaic.png

실행:  python3 scripts/make_ch02_confusion_mosaic.py   (저장소 최상위에서)
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
from matplotlib.patches import Rectangle

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

INK = "#37474F"
BLUE, BLUE_F = "#1565C0", "#BBDEFB"
ORANGE, ORANGE_F = "#E65100", "#FFCC80"
MUTED = "#90A4AE"
RED = "#D32F2F"

OUT = "docs/ch02/visualization/img/"
os.makedirs(OUT, exist_ok=True)

rng = np.random.default_rng(3)
n = 10_000
prev = 0.02
y = (rng.random(n) < prev).astype(int)          # 참값: 2 퍼센트만 양성

# 분류기 셋
pred = {}
pred["ㄱ 모두 음성"] = np.zeros(n, dtype=int)
pred["ㄴ 2 퍼센트를 무작위로"] = (rng.random(n) < prev).astype(int)
sens, spec = 0.60, 0.955
u = rng.random(n)
pred["ㄷ 진짜 탐지기"] = np.where(y == 1, u < sens, u > spec).astype(int)


def table(p):
    return np.array([[np.sum((p == i) & (y == j)) for j in (0, 1)] for i in (0, 1)])


fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.6))
rows = []
for ax, (name, p) in zip(axes, pred.items()):
    T = table(p).astype(float)           # 행 = 예측, 열 = 참값
    nn = T.sum()
    acc = (T[0, 0] + T[1, 1]) / nn
    # 가로 = 참값(음성/양성), 세로 = 그 안에서 양성이라 예측한 비율
    colw = T.sum(0) / nn
    x0 = 0.0
    for j in range(2):
        h = T[1, j] / T[:, j].sum()      # 그 열 안에서 양성 예측의 비율
        ax.add_patch(Rectangle((x0, 0), colw[j], 1 - h, facecolor=BLUE_F,
                               edgecolor="white", lw=1.2))
        ax.add_patch(Rectangle((x0, 1 - h), colw[j], h, facecolor=ORANGE_F,
                               edgecolor="white", lw=1.2))
        ax.plot([x0, x0 + colw[j]], [1 - h, 1 - h], color=RED, lw=2.4)
        ax.text(x0 + colw[j] / 2, -0.04,
                f"참 {'음성' if j == 0 else '양성'}\n{colw[j]*100:.1f} 퍼센트",
                ha="center", va="top", fontsize=8.8, color=INK)
        ax.annotate(f"양성이라 예측 {h*100:.1f} 퍼센트",
                    xy=(x0 + colw[j] / 2, 1 - h),
                    xytext=(x0 + colw[j] / 2, 1.11 if j == 0 else 1.04),
                    ha="center", fontsize=8.8, color=ORANGE,
                    arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.9))
        x0 += colw[j] + 0.012
    sub = T[:, T.sum(0) > 0]
    if (T.sum(1) > 0).sum() == 2:
        chi2, pv = stats.chi2_contingency(T, correction=False)[:2]
    else:
        chi2, pv = 0.0, 1.0
    v = np.sqrt(chi2 / nn) if chi2 else 0.0
    rows.append((name, acc, chi2, pv, v, T))
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.02, 1.20)
    ax.axis("off")
    tag = ("분할선이 바닥에 붙어 나란하다" if (T.sum(1) == 0).any()
           else ("분할선이 거의 나란하다 — 독립에 가깝다" if pv > 0.05
                 else "분할선이 크게 어긋난다 — 쓸모가 있다"))
    ax.set_title(f"{name}  (정확도 {acc*100:.2f} 퍼센트)\n{tag}", fontsize=10.5,
                 color=INK, linespacing=1.6)
    ax.title.set_color(INK)
    ax.text(0.5, 1.185, "", transform=ax.transAxes)

fig.text(0.5, 0.015, "가로 폭 = 참값의 비율,  빨간 선 = 그 안에서 양성이라 예측한 비율"
         "  (주황이 양성 예측)", ha="center", fontsize=9.5, color=MUTED)
fig.tight_layout(rect=(0, 0.05, 1, 1))
path = OUT + "confusion_mosaic.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
plt.close(fig)
print("saved", path)

print(f"n = {n:,},  참 양성 {y.sum():,} 개 ({100*y.mean():.2f} 퍼센트)")
print("분류기                   정확도     카이제곱        p      크래머 V")
for name, acc, chi2, pv, v, T in rows:
    print(f"{name:<22}{acc*100:8.2f}{chi2:12.2f}{pv:9.4f}{v:10.3f}   표 {T.ravel().astype(int)}")
