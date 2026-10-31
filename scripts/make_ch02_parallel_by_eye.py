"""2.6절 모자이크 그림 — 나란함으로 독립을 판정할 수 있는가.

쪽의 §3 이 증명한 동치 「독립 <=> 모든 열의 분할선이 같은 높이」는 **참 확률**에
대한 것이다. 그림이 그리는 것은 표본의 상대도수이므로, 나란함을 눈으로 보고
독립을 판정하는 일은 양쪽으로 틀린다.

쪽의 보기 4 와 같은 설정(n = 400, X 의 주변확률 0.5)에서
  참 독립:  P(Y=예 | X) = 0.30  (두 열이 같다)
  참 종속:  P(Y=예 | X=예) = 0.40,  P(Y=예 | X=아니오) = 0.30
두 진실 아래 분할선 높이 차의 분포를 재고, 각 진실에서 「반대로 보이는」 표본을
하나씩 꺼내 나란히 둔다.
"""

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

import numpy as np
from matplotlib.patches import Rectangle
from scipy import stats

OUT = "docs/ch02/visualization/img/"

N = 400
PX = 0.5
P_IND = 0.30
P_DEP = (0.40, 0.30)
THR = 0.05          # 눈으로 「나란하다」고 보아 줄 느슨한 자
M = 200_000


def table(seed, p1, p0):
    """쪽의 보기 4 와 같은 식으로 한 표본을 뽑아 2x2 표를 만든다."""
    rng = np.random.default_rng(seed)
    x = rng.random(N) < PX
    y = rng.random(N) < np.where(x, p1, p0)
    n1 = int(x.sum())
    k1, k0 = int((y & x).sum()), int((y & ~x).sum())
    return np.array([[k1, n1 - k1], [k0, N - n1 - k0]], float)


def gap(O):
    return abs(O[0, 0] / O[0].sum() - O[1, 0] / O[1].sum())


def chi2p(O):
    c, p, _, _ = stats.chi2_contingency(O, correction=False)
    return c, p


def sim_gap(p1, p0, seed):
    """높이 차의 분포 — X 가 베르누이이므로 이항 셋으로 한 번에 뽑는다."""
    rng = np.random.default_rng(seed)
    n1 = rng.binomial(N, PX, M)
    n1 = np.clip(n1, 1, N - 1)
    k1 = rng.binomial(n1, p1)
    k0 = rng.binomial(N - n1, p0)
    return np.abs(k1 / n1 - k0 / (N - n1))


g_ind = sim_gap(P_IND, P_IND, 11)
g_dep = sim_gap(*P_DEP, 12)

look_par_but_dep = float(np.mean(g_dep < THR))      # 참 종속인데 나란해 보인다
look_off_but_ind = float(np.mean(g_ind >= THR))     # 참 독립인데 어긋나 보인다

O_ind = table(49, P_IND, P_IND)                     # 독립인데 어긋나 보이는 표본
O_dep = table(500122, *P_DEP)                       # 종속인데 나란해 보이는 표본
g1, (c1, p1v) = gap(O_ind), chi2p(O_ind)
g2, (c2, p2v) = gap(O_dep), chi2p(O_dep)


# ── 모자이크 한 장을 손으로 그린다 ─────────────────────────────────
def mosaic(ax, O, title, tcol, note):
    gapw = 0.015
    w = O.sum(axis=1) / O.sum()                     # 열 너비 = 주변확률
    hs = O[:, 0] / O.sum(axis=1)                    # 분할 높이 = 조건부확률
    x = 0.0
    for i, (wi, hi) in enumerate(zip(w, hs)):
        wi_ = wi - gapw
        ax.add_patch(Rectangle((x, 1 - hi), wi_, hi, facecolor="tab:orange",
                               alpha=0.80, edgecolor="white", lw=1.2))
        ax.add_patch(Rectangle((x, 0), wi_, 1 - hi, facecolor="0.80",
                               edgecolor="white", lw=1.2))
        ax.plot([x, x + wi_], [1 - hi, 1 - hi], color="0.15", lw=2.6)
        ax.text(x + wi_ / 2, 1 - hi + 0.045, f"{hi:.3f}", ha="center",
                fontsize=11, fontweight="bold", color="0.1")
        ax.text(x + wi_ / 2, -0.055, ["X = 예", "X = 아니오"][i], ha="center",
                va="top", fontsize=10.5, color="0.3")
        x += wi
    lo, hi2 = sorted([1 - hs[0], 1 - hs[1]])
    if hi2 - lo > 0.01:
        ax.add_patch(Rectangle((0, lo), 1 - gapw, hi2 - lo, facecolor="none",
                               edgecolor="tab:red", lw=1.6, ls="--"))
    ax.set_xlim(-0.02, 1.0)
    ax.set_ylim(-0.17, 1.14)
    ax.axis("off")
    ax.set_title(f"{title}\n{note}", fontsize=11.5, color=tcol,
                 fontweight="bold", pad=8)


fig = plt.figure(figsize=(13.2, 4.8))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.5], wspace=0.22)
ax1, ax2, ax3 = (fig.add_subplot(gs[0, i]) for i in range(3))

mosaic(ax1, O_ind, "참값은 독립이다", "tab:red",
       f"높이 차 {g1:.4f} · p = {p1v:.4f}")
mosaic(ax2, O_dep, "참값은 10 퍼센트포인트 차이다", "tab:blue",
       f"높이 차 {g2:.4f} · p = {p2v:.4f}")

bins = np.linspace(0, 0.30, 90)
ax3.hist(g_ind, bins=bins, density=True, color="tab:red", alpha=0.55,
         label="참 독립")
ax3.hist(g_dep, bins=bins, density=True, color="tab:blue", alpha=0.55,
         label="참 종속 (0.40 대 0.30)")
ax3.axvline(THR, color="0.15", lw=2.0)
BOX = dict(boxstyle="round,pad=0.28", facecolor="white", alpha=0.88,
           edgecolor="none")
ax3.text(0.0105, 8.4, f"참 종속인데\n이 왼쪽: {look_par_but_dep * 100:.1f}%",
         fontsize=10.5, color="tab:blue", fontweight="bold", va="top",
         bbox=BOX)
ax3.text(0.062, 14.6, f"참 독립인데 이 오른쪽: {look_off_but_ind * 100:.1f}%",
         fontsize=10.5, color="tab:red", fontweight="bold", va="top",
         bbox=BOX)
for g, col in [(g1, "tab:red"), (g2, "tab:blue")]:
    ax3.annotate("", xy=(g, 0), xytext=(g, -2.6),
                 arrowprops=dict(arrowstyle="->", color=col, lw=2.0))
ax3.text(g1, -3.1, "왼쪽 그림", ha="center", va="top", fontsize=9.5,
         color="tab:red")
ax3.text(g2, -3.1, "가운데 그림", ha="center", va="top", fontsize=9.5,
         color="tab:blue")
ax3.set_xlim(-0.004, 0.30)
ax3.set_ylim(-6.0, 19.5)
ax3.set_yticks([0, 5, 10, 15])
ax3.set_xlabel("두 열의 분할선 높이 차", fontsize=11)
ax3.set_ylabel("밀도", fontsize=11)
ax3.set_title(f"같은 n = {N} 에서 두 진실이 내는 높이 차\n"
              f"세로선은 눈으로 「나란하다」고 보아 주는 자 {THR}",
              fontsize=11.5)
ax3.legend(fontsize=10, loc="upper right", framealpha=0.95,
           bbox_to_anchor=(1.0, 0.86))
ax3.grid(alpha=0.2)

fig.suptitle("나란함은 참 확률의 성질이고, 그림은 표본의 상대도수를 그린다",
             fontsize=13.5, y=1.02)
fig.subplots_adjust(top=0.80, bottom=0.14, left=0.03, right=0.985)
path = OUT + "parallel_by_eye.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
print("saved", path)

print(f"\nn = {N} · 참 독립 {P_IND} · 참 종속 {P_DEP} · 모의 {M:,} 회")
print(f"독립 아래 높이 차: 중앙값 {np.median(g_ind):.4f} · 90 분위 "
      f"{np.quantile(g_ind, .9):.4f} · 최대 {g_ind.max():.4f}")
print(f"종속 아래 높이 차: 중앙값 {np.median(g_dep):.4f} · 10 분위 "
      f"{np.quantile(g_dep, .1):.4f} · 최소 {g_dep.min():.4f}")
print(f"\n「나란하다」를 높이 차 < {THR} 로 잡으면")
print(f"  참 종속인데 나란해 보인다: {look_par_but_dep:.4f}")
print(f"  참 독립인데 어긋나 보인다: {look_off_but_ind:.4f}")
print(f"\n왼쪽  (참 독립)  표 {O_ind.astype(int).tolist()}  높이 차 {g1:.4f}  "
      f"chi2 {c1:.4f}  p {p1v:.4f}")
print(f"가운데(참 종속)  표 {O_dep.astype(int).tolist()}  높이 차 {g2:.4f}  "
      f"chi2 {c2:.4f}  p {p2v:.4f}")
