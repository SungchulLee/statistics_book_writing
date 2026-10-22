r"""임베딩 산점도에서 눈이 읽는 것과 자료가 말하는 것.

고차원 벡터를 $t$-SNE 로 두 축에 내려 찍는 것은 기계학습에서 거의 표준
절차다. 그 그림에서 사람이 읽는 것은 셋이다 — **군집 사이의 거리**,
**군집의 크기**, **군집이 있다는 사실** 자체. 셋을 하나씩 잰다.

① 군집 사이 거리 — 같은 자료인데 perplexity 하나로 비가 네 배 바뀐다.
② 군집의 퍼짐 — 참 퍼짐이 여섯 배인 둘이 같은 크기로 그려진다.
③ 구조가 없는 자료에서도 군집이 보인다.

만드는 파일: docs/ch02/visualization/img/tsne_scatter.png
"""

import warnings

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans            # noqa: E402
from sklearn.manifold import TSNE             # noqa: E402
from sklearn.metrics import silhouette_score  # noqa: E402

RNG = np.random.default_rng(0)
D, N = 50, 300
PERS = [5, 10, 20, 30, 50, 75, 100, 150]
P_LO, P_HI = 5, 100
SIG_A, SIG_B = 1.0, 6.0
N_NOISE, K_NOISE, P_NOISE = 900, 5, 5

tsne = lambda X, p: TSNE(n_components=2, perplexity=p, init="pca",   # noqa
                         random_state=0).fit_transform(X)
cen = lambda X, lab, k: np.array([X[lab == i].mean(0) for i in range(k)])  # noqa


def ratio(Y, lab):
    """A–C 거리가 A–B 거리의 몇 배로 보이는가."""
    g = cen(Y, lab, 3)
    return (float(np.linalg.norm(g[0] - g[2]))
            / float(np.linalg.norm(g[0] - g[1])))


# === ① 군집 사이 거리 ================================================
mu = np.zeros((3, D))
mu[1, 0] = 6.0          # A 와 B 는 가깝고
mu[2, 1] = 30.0         # A 와 C 는 멀다
X = np.vstack([RNG.normal(mu[i], 1.0, (N, D)) for i in range(3)])
LAB = np.repeat([0, 1, 2], N)
c = cen(X, LAB, 3)
D_AB = float(np.linalg.norm(c[0] - c[1]))
D_AC = float(np.linalg.norm(c[0] - c[2]))
TRUE_R = D_AC / D_AB
print(f"① 세 군집의 참 중심거리 — A–B {D_AB:.2f}, A–C {D_AC:.2f} "
      f"(비 {TRUE_R:.2f})")
YS = {p: tsne(X, p) for p in PERS}
RS = {p: ratio(YS[p], LAB) for p in PERS}
print(f"{'perplexity':>11}{'그림 위의 비':>13}")
for p in PERS:
    print(f"{p:>11}{RS[p]:>13.2f}")
common = [p for p in PERS if p <= 50]      # 흔히 권하는 범위
print(f"   같은 자료인데 손잡이 하나로 {min(RS.values()):.2f} 에서 "
      f"{max(RS.values()):.2f} 까지 {max(RS.values()) / min(RS.values()):.1f} 배 "
      f"바뀐다.")
print(f"   흔히 쓰는 perplexity {common[0]}–{common[-1]} 에서는 "
      f"{min(RS[p] for p in common):.2f}–{max(RS[p] for p in common):.2f} 로 "
      f"참값 {TRUE_R:.2f} 의 절반에 못 미친다. 참값에 닿는 자리가 있기는 "
      f"하지만(perplexity {max(PERS)} 에서 {RS[max(PERS)]:.2f}),")
print(f"   그 자리를 고르려면 답을 이미 알아야 한다.")

# === ② 군집의 퍼짐 ==================================================
Z = np.vstack([RNG.normal(np.zeros(D), SIG_A, (N, D)),
               RNG.normal(np.r_[40.0, np.zeros(D - 1)], SIG_B, (N, D))])
L2 = np.repeat([0, 1], N)
sp = [float(Z[L2 == i].std(axis=0).mean()) for i in range(2)]
Y2 = tsne(Z, 30)
sy = [float(np.linalg.norm(Y2[L2 == i] - Y2[L2 == i].mean(0), axis=1).mean())
      for i in range(2)]
print(f"\n② 퍼짐 — 참 비 {sp[1] / sp[0]:.2f}  ->  그림 위의 비 "
      f"{sy[1] / sy[0]:.2f}")

# === ③ 구조가 없는 자료 ==============================================
W = RNG.normal(0, 1, (N_NOISE, D))
Y3 = tsne(W, P_NOISE)
s_raw = float(silhouette_score(
    W, KMeans(K_NOISE, n_init=10, random_state=0).fit_predict(W)))
k3 = KMeans(K_NOISE, n_init=10, random_state=0).fit_predict(Y3)
s_tsne = float(silhouette_score(Y3, k3))
print(f"\n③ 독립인 표준정규 {N_NOISE} 개 (군집이 없다)")
print(f"   {K_NOISE} 개로 나눈 실루엣 — 원자료 {s_raw:.3f}, "
      f"t-SNE 뒤 {s_tsne:.3f}")

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.6, 5.2))
CC = ["#1f77b4", "#2ca02c", "#d62728"]

ax1.plot(PERS, [RS[p] for p in PERS], "o-", color="#d62728", lw=2.6, ms=7)
ax1.axhline(TRUE_R, color="black", lw=1.8, ls="--")
ax1.text(5.4, TRUE_R * 1.04, f"참 거리비 ${TRUE_R:.2f}$", fontsize=10.5,
         ha="left")
ax1.set_xscale("log")
ax1.set_xticks(PERS)
ax1.set_xticklabels([f"${v}$" for v in PERS], fontsize=9)
ax1.minorticks_off()
ax1.set_xlabel("perplexity (손잡이 하나)")
ax1.set_ylabel("그림에서 읽히는 $A$–$C$ 대 $A$–$B$ 거리비")
ax1.set_ylim(0, TRUE_R * 1.62)
ax1.set_title("① 같은 자료, 네 배 다른 거리", fontsize=12)
ax1.grid(alpha=0.25)
for k, (p, box) in enumerate([(P_LO, [0.08, 0.68, 0.27, 0.27]),
                              (P_HI, [0.40, 0.68, 0.27, 0.27])]):
    ins = ax1.inset_axes(box)
    for i in range(3):
        ins.plot(YS[p][LAB == i, 0], YS[p][LAB == i, 1], ".", color=CC[i],
                 ms=1.6)
    ins.set_xticks([]); ins.set_yticks([])
    ins.set_title(f"perplexity ${p}$ → ${RS[p]:.2f}$", fontsize=8.5, pad=2)

for i, nm in enumerate([f"좁은 군집 ($\\sigma = {SIG_A:g}$)",
                        f"넓은 군집 ($\\sigma = {SIG_B:g}$)"]):
    ax2.plot(Y2[L2 == i, 0], Y2[L2 == i, 1], ".", color=CC[i], ms=4,
             label=nm)
ax2.set_xticks([]); ax2.set_yticks([])
ax2.set_xlabel("t-SNE 축에는 단위가 없다")
ax2.text(0.03, 0.97,
         f"참 퍼짐의 비 ${sp[1] / sp[0]:.2f}$\n"
         f"그림 위의 비 $\\mathbf{{{sy[1] / sy[0]:.2f}}}$",
         transform=ax2.transAxes, va="top", fontsize=11,
         bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc"))
ax2.set_title("② 여섯 배 넓은 군집이 같은 크기로", fontsize=12)
ax2.legend(loc="lower right", fontsize=10, markerscale=2.5)

for i in range(K_NOISE):
    ax3.plot(Y3[k3 == i, 0], Y3[k3 == i, 1], ".", ms=4,
             color=plt.cm.tab10(i))
ax3.set_xticks([]); ax3.set_yticks([])
ax3.set_xlabel(f"독립인 표준정규 ${D}$ 차원 벡터 ${N_NOISE}$ 개")
ax3.text(0.03, 0.97,
         f"$k$-평균 ${K_NOISE}$ 개의 실루엣\n"
         f"원자료   ${s_raw:.3f}$\n"
         f"t-SNE 뒤 $\\mathbf{{{s_tsne:.3f}}}$",
         transform=ax3.transAxes, va="top", fontsize=11,
         bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc"))
ax3.set_title("③ 군집이 없는 자료의 군집", fontsize=12)

fig.tight_layout()
out = "docs/ch02/visualization/img/tsne_scatter.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
