r"""1.4 「길이 편향 표집」 쪽의 상자에 들어가는 그림을 만든다.

비터리히의 각산정(1947)이 길이 편향을 **도구로 쓰는** 방식을 보인다.

  왼쪽  나무가 포함되는 조건은 r < d/k 이므로 포함원의 넓이가 pi d^2/k^2,
        곧 흉고단면적에 비례한다. 큰 나무가 더 자주 뽑히는 것이 결함이 아니라
        장치다 — 재려는 양이 바로 그 단면적의 합이기 때문이다.
  오른쪽 임의의 숲을 만들어 점 2000 곳에서 각산정을 돌리고, 센 나무 수에
        계수(BAF)를 곱한 값이 참 흉고단면적(헥타르당)을 맞히는지 본다.

만드는 파일:
  ch01/classical/img/bitterlich_angle_count.png

실행:  python3 scripts/make_ch01_bitterlich.py   (저장소 최상위에서)
필요:  numpy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

INK = "#37474F"
BLUE, BLUE_F = "#1565C0", "#DCEBFB"
ORANGE, ORANGE_F = "#E65100", "#FFE0B2"
GREEN = "#33691E"
MUTED = "#90A4AE"

OUT = "docs/ch01/classical/img/"
os.makedirs(OUT, exist_ok=True)

K = 1 / 50.0                 # 게이지의 폭:길이 비. BAF = 2500*k^2 = 1 m^2/ha
BAF = 2500 * K ** 2

# ------------------------------------------------ 왼쪽: 포함원이 단면적에 비례
fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.8, 4.6))

trees = [(0.20, "#8D6E63"), (0.40, "#6D4C41"), (0.60, "#4E342E")]
centers = [(-46.0, 0.0), (-6.0, 0.0), (48.0, 0.0)]
for (dd, col), (cx, cy) in zip(trees, centers):
    R = dd / K                                 # 포함 반지름
    axL.add_patch(Circle((cx, cy), R, facecolor=BLUE_F, edgecolor=BLUE,
                         lw=1.6, alpha=0.75, zorder=2))
    axL.plot([cx], [cy], "o", color=col, ms=4 + 14 * dd, zorder=4)
    axL.annotate(f"$d = {dd:.2f}$ m\n포함반지름 {R:.0f} m\n넓이 {np.pi*R*R/10000:.2f} ha",
                 (cx, cy - R), textcoords="offset points", xytext=(0, -13),
                 ha="center", va="top", fontsize=9, color=INK, zorder=5)
axL.set_xlim(-72, 88)
axL.set_ylim(-56, 42)
axL.set_aspect("equal")
axL.set_title("포함원의 넓이가 단면적에 비례한다", color=INK, fontsize=12, pad=9)
axL.text(-70, 34, "선 자리가 그 나무의 원 안에 들면 그 나무를 센다.\n"
                  "지름이 $3$ 배면 반지름도 $3$ 배, 넓이는 $9$ 배다 — $d^2$ 에 비례한다.",
         fontsize=9.5, color=INK)
axL.set_xticks([]); axL.set_yticks([])
for s in ("top", "right", "bottom", "left"):
    axL.spines[s].set_visible(False)

# ------------------------------------------------ 오른쪽: 모의 숲에서 맞는가
rng = np.random.default_rng(7)
SIDE = 260.0                                   # m. 숲 한 변
N_TREE = 4200
d = rng.lognormal(mean=np.log(0.28), sigma=0.45, size=N_TREE)
xy = rng.uniform(0, SIDE, size=(N_TREE, 2))
true_ba = (np.pi / 4 * d ** 2).sum() / (SIDE ** 2 / 10000.0)   # m^2/ha

# 숲을 원환으로 두르면 가장자리가 아예 없다. 가장자리를 피하려고 안쪽만
# 고르면 그 안쪽의 밀도가 전체 밀도와 미세하게 달라 추정이 치우친다.
pts = rng.uniform(0, SIDE, size=(2000, 2))
counts = np.empty(len(pts))
for i, p in enumerate(pts):
    delta = np.abs(xy - p)
    delta = np.minimum(delta, SIDE - delta)          # 원환 거리
    r = np.hypot(*delta.T)
    counts[i] = np.sum(r < d / K)
est = counts * BAF

axR.hist(est, bins=np.arange(-0.5, est.max() + 1.5) * BAF + 0.0,
         color=ORANGE_F, edgecolor=ORANGE, lw=0.6, zorder=3)
axR.axvline(true_ba, color=GREEN, lw=2.4, zorder=5)
axR.axvline(est.mean(), color=BLUE, lw=2.0, ls="--", zorder=5)
axR.text(true_ba, axR.get_ylim()[1] * 0.93, f"  참값 {true_ba:.2f}",
         color=GREEN, fontsize=10, va="top")
axR.text(est.mean(), axR.get_ylim()[1] * 0.80, f"  점 2000 곳의 평균 {est.mean():.2f}",
         color=BLUE, fontsize=10, va="top")
axR.set_xlabel("한 점에서 센 나무 수 $\\times$ 계수  (헥타르당 $m^2$)")
axR.set_ylabel("점의 수")
axR.set_title("자도 줄자도 쓰지 않고 맞힌다", color=INK, fontsize=12, pad=9)
for s in ("top", "right"):
    axR.spines[s].set_visible(False)

fig.tight_layout()
path = OUT + "bitterlich_angle_count.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
plt.close(fig)

print(f"게이지 비 k = 1/{1/K:.0f},  계수(BAF) = {BAF:.2f} m^2/ha")
print(f"나무 {N_TREE} 그루, 숲 {SIDE:.0f}x{SIDE:.0f} m (원환),  가장 굵은 나무 {d.max():.2f} m")
print(f"참 흉고단면적   {true_ba:.3f} m^2/ha")
print(f"각산정 평균     {est.mean():.3f} m^2/ha   (점 {len(pts)} 곳)")
print(f"한 점의 표준편차 {est.std():.3f},  평균의 표준오차 {est.std()/len(pts)**0.5:.3f}")
print("saved", path)
