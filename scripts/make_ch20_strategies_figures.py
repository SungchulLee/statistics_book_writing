r"""ch20 다범주 전략 절의 개념 그림을 그린다.

만드는 파일:
  ch20/strategies/img/comparison_accuracy_vs_coherence.png
      정확도로는 갈리지 않고 확률의 정합성에서 갈린다
  ch20/strategies/img/ovo_condorcet_cycle.png
      쌍별 다수결에 생기는 순환과 그 결과인 동점

실행:  python3 scripts/make_ch20_strategies_figures.py   (저장소 최상위에서)
필요:  numpy, matplotlib, scikit-learn — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False
OUT = "docs/ch20/strategies/img/"
os.makedirs(OUT, exist_ok=True)

INK = "#37474F"
BLUE, BLUE_F = "#1565C0", "#DCEBFB"
ORANGE, ORANGE_F = "#E65100", "#FFE0B2"
GREEN, GREEN_F = "#33691E", "#C5E1A5"
PURPLE = "#6A1B9A"
MUTED = "#90A4AE"
RED = "#D32F2F"


# === 비교: 정확도 vs 확률의 정합성 =======================================
from sklearn.datasets import make_classification            # noqa: E402
from sklearn.linear_model import LogisticRegression         # noqa: E402
from sklearn.multiclass import OneVsRestClassifier          # noqa: E402

# --- 본문 보기 표의 5-겹 교차검증 결과 -----------------------------------
meth = ["OvR", "OvO", "소프트맥스"]
acc = np.array([0.6800, 0.6700, 0.6820])
sd = np.array([0.0105, 0.0138, 0.0068])
cols = [ORANGE, PURPLE, BLUE]

# --- 같은 설정의 자료에서 OvR 점수의 합 ----------------------------------
X, y = make_classification(n_samples=1000, n_features=10, n_informative=6,
                           n_redundant=2, n_classes=5, n_clusters_per_class=1,
                           random_state=0)
ovr = OneVsRestClassifier(LogisticRegression(max_iter=2000)).fit(X, y)
S = np.column_stack([e.predict_proba(X)[:, 1]
                     for e in ovr.estimators_]).sum(axis=1)
out_frac = np.mean((S < 0.9) | (S > 1.1))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.8))

# --- 왼쪽: 정확도는 갈리지 않는다 ----------------------------------------
yy = np.arange(3)[::-1]
for i in range(3):
    ax1.plot([acc[i] - sd[i], acc[i] + sd[i]], [yy[i], yy[i]], color=cols[i],
             lw=5, alpha=0.35, solid_capstyle="round")
    ax1.plot(acc[i], yy[i], "o", color=cols[i], ms=11, zorder=4)
    ax1.text(acc[i], yy[i] + 0.22, f"{acc[i]:.4f}", ha="center", va="bottom",
             fontsize=10, color=cols[i], fontweight="bold")
    ax1.text(acc[i] + sd[i] + 0.0022, yy[i] - 0.03,
             f"±{sd[i]:.4f}", ha="left", va="center", fontsize=9, color=MUTED)

lo = (acc - sd).max()
hi = (acc + sd).min()
ax1.axvspan(lo, hi, color=GREEN_F, alpha=0.5, zorder=0)
ax1.text((lo + hi) / 2, -0.62, "세 구간이 모두 겹치는 영역", ha="center",
         va="bottom", fontsize=9.5, color=GREEN, fontweight="bold")

ax1.set_yticks(yy)
ax1.set_yticklabels(meth, fontsize=11)
ax1.set_ylim(-0.75, 2.55)
ax1.set_xlim(0.650, 0.700)
ax1.set_xticks(np.arange(0.65, 0.701, 0.01))
ax1.set_xlabel("5-겹 교차검증 정확도 (가로막대는 ±1 표준편차)", fontsize=10.5)
ax1.set_title("정확도로는 세 전략을 가를 수 없다", fontsize=11.5, color=INK,
              pad=10)
ax1.grid(axis="x", alpha=0.25, ls=":")
ax1.set_axisbelow(True)

# --- 오른쪽: OvR 점수의 합은 1이 아니다 ----------------------------------
bins = np.linspace(0.2, 2.5, 48)
ax2.hist(S, bins=bins, color=ORANGE_F, edgecolor=ORANGE, linewidth=1.0,
         zorder=2)
ax2.axvline(1.0, color=BLUE, lw=2.4, zorder=3)
ax2.text(1.04, ax2.get_ylim()[1] * 0.96, "소프트맥스는 언제나 여기 한 점",
         ha="left", va="top", fontsize=10, color=BLUE, fontweight="bold",
         zorder=5,
         bbox=dict(facecolor="white", edgecolor="none", pad=1.5, alpha=0.9))

ymax = ax2.get_ylim()[1]
for v, nm, ha in ((S.min(), f"최소 {S.min():.3f}", "left"),
                  (S.max(), f"최대 {S.max():.3f}", "right")):
    ax2.axvline(v, color=RED, lw=1.4, ls="--", zorder=3)
    ax2.text(v + (0.03 if ha == "left" else -0.03), ymax * 0.52, nm, ha=ha,
             va="center", fontsize=9.5, color=RED, fontweight="bold",
             rotation=90, zorder=5)

ax2.text(0.24, ymax * 0.80,
         f"평균 {S.mean():.4f}\n그러나 {out_frac*100:.0f}%가\n[0.9, 1.1] 밖",
         ha="left", va="top", fontsize=10, color=INK, fontweight="bold",
         bbox=dict(facecolor="white", edgecolor=MUTED, pad=4.0, alpha=0.95),
         zorder=5)

ax2.set_xlim(0.2, 2.5)
ax2.set_xlabel("한 관측치에서 OvR 시그모이드 점수 5개의 합", fontsize=10.5)
ax2.set_ylabel("관측치 수", fontsize=10.5)
ax2.set_title("평균은 1이지만 개별 예측은 1이 아니다", fontsize=11.5,
              color=INK, pad=10)
ax2.grid(axis="y", alpha=0.25, ls=":")
ax2.set_axisbelow(True)

for ax in (ax1, ax2):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

fig.savefig(OUT + "comparison_accuracy_vs_coherence.png", dpi=170,
            facecolor="white", bbox_inches="tight")
plt.close(fig)

print("정확도 구간 겹침 =", f"[{lo:.4f}, {hi:.4f}]")
print("OvR 합: 평균", f"{S.mean():.4f}", " 최소", f"{S.min():.4f}",
      " 최대", f"{S.max():.4f}")
print("[0.9, 1.1] 밖 비율 =", f"{out_frac:.4f}")
print("wrote", OUT + "comparison_accuracy_vs_coherence.png")


# === OvO 의 순환: 콩도르세 역설 =========================================
from matplotlib.patches import Circle, FancyArrowPatch      # noqa: E402

POS = {"A": (0.0, 1.0), "B": (1.0, 0.0), "D": (0.0, -1.0), "C": (-1.0, 0.0)}
WINS = [("A", "B"), ("A", "C"), ("D", "A"),
        ("B", "C"), ("B", "D"), ("D", "C")]
CYCLE = {("A", "B"), ("B", "D"), ("D", "A")}
NAMES4 = ["A", "B", "C", "D"]
votes = {c: sum(1 for (w, l) in WINS if w == c) for c in NAMES4}
R = 0.20

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 5.2),
                               gridspec_kw={"width_ratios": [1.15, 1.0]})

# --- 왼쪽: 쌍별 승패 그래프 ----------------------------------------------
for (w, l) in WINS:
    p0 = np.array(POS[w], dtype=float)
    p1 = np.array(POS[l], dtype=float)
    d = p1 - p0
    d = d / np.linalg.norm(d)
    incycle = (w, l) in CYCLE
    rad = 0.0
    if {w, l} == {"D", "A"} or {w, l} == {"B", "C"}:
        rad = 0.30 if {w, l} == {"D", "A"} else -0.30
    ax1.add_patch(FancyArrowPatch(
        tuple(p0 + d * R * 1.12), tuple(p1 - d * R * 1.12),
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-|>", mutation_scale=20,
        lw=3.0 if incycle else 1.6,
        color=RED if incycle else MUTED,
        zorder=3 if incycle else 2))

for c in NAMES4:
    p = POS[c]
    on_cycle = c in ("A", "B", "D")
    ax1.add_patch(Circle(p, R, facecolor=ORANGE_F if on_cycle else BLUE_F,
                         edgecolor=ORANGE if on_cycle else BLUE, lw=2.0,
                         zorder=4))
    ax1.text(p[0], p[1] + 0.03, c, ha="center", va="center", fontsize=15,
             color=INK, fontweight="bold", zorder=5)
    ax1.text(p[0], p[1] - 0.085, f"{votes[c]}표", ha="center", va="center",
             fontsize=9.5, color=INK, zorder=5)

ax1.text(0.0, 1.46, "화살표는 쌍별 대결의 승자 → 패자", ha="center", va="center",
         fontsize=10, color=INK)
ax1.text(0.98, 1.12, "A ≻ B ≻ D ≻ A", ha="center", va="center", fontsize=12.5,
         color=RED, fontweight="bold",
         bbox=dict(facecolor="white", edgecolor=RED, pad=3.0, alpha=0.95))
ax1.text(0.0, -1.48, "굵은 붉은 화살표 세 개가 순환을 이룬다 — 한 줄로 세울 수 없다",
         ha="center", va="center", fontsize=10, color=RED, fontweight="bold")

ax1.set_xlim(-1.45, 1.45)
ax1.set_ylim(-1.68, 1.62)
ax1.set_aspect("equal")
ax1.axis("off")
ax1.set_title("네 범주의 쌍별 승패 관계", fontsize=11.5, color=INK, pad=6)

# --- 오른쪽: 표 집계 -----------------------------------------------------
vv = np.array([votes[c] for c in NAMES4], dtype=float)
xs = np.arange(4)
bar_c = [ORANGE if v == vv.max() else MUTED for v in vv]
bar_f = [ORANGE_F if v == vv.max() else "#ECEFF1" for v in vv]
ax2.bar(xs, vv, 0.56, color=bar_f, edgecolor=bar_c, linewidth=1.8, zorder=2)
for xi, v in zip(xs, vv):
    ax2.text(xi, v + 0.06, f"{v:.0f}", ha="center", va="bottom", fontsize=12,
             color=INK, fontweight="bold")

ax2.axhline(3, color=BLUE, lw=1.8, ls="--", zorder=3)
ax2.text(3.42, 3.06, "한 범주가 얻을 수 있는 최대 표  C - 1 = 3", ha="right",
         va="bottom", fontsize=9.5, color=BLUE, fontweight="bold")
ax2.axhline(2, color=RED, lw=1.4, ls=":", zorder=3)
ax2.annotate("세 범주가 2표로 동점\n투표만으로는 답이 없다", xy=(1.0, 2.0),
             xytext=(1.75, 1.12), fontsize=10, color=RED, fontweight="bold",
             ha="left", va="center",
             arrowprops=dict(arrowstyle="->", color=RED, lw=1.3))

ax2.set_xticks(xs)
ax2.set_xticklabels([f"범주 {c}" for c in NAMES4], fontsize=11)
ax2.set_xlim(-0.6, 3.6)
ax2.set_ylim(0, 3.6)
ax2.set_yticks([0, 1, 2, 3])
ax2.set_ylabel("받은 표 수", fontsize=10.5)
ax2.set_title("다수결의 결과", fontsize=11.5, color=INK, pad=10)
ax2.grid(axis="y", alpha=0.25, ls=":")
ax2.set_axisbelow(True)
for s in ("top", "right"):
    ax2.spines[s].set_visible(False)

fig.savefig(OUT + "ovo_condorcet_cycle.png", dpi=170, facecolor="white",
            bbox_inches="tight")
plt.close(fig)

print("표 집계 =", votes)
print("wrote", OUT + "ovo_condorcet_cycle.png")


# === OvR 의 모호한 영역 =================================================
from matplotlib.colors import ListedColormap                # noqa: E402


def ovr_fit(X, y, C=3):
    """범주마다 '이 범주인가 아닌가' 로지스틱 회귀를 따로 적합한다."""
    models = []
    for c in range(C):
        m = LogisticRegression(max_iter=2000)
        m.fit(X, (y == c).astype(int))
        models.append(m)
    return models


def ovr_scores(models, G):
    return np.column_stack([m.predict_proba(G)[:, 1] for m in models])


def make_blobs_at(centers, n_each, seed, sd=0.75):
    r = np.random.default_rng(seed)
    Xs, ys = [], []
    for k, c in enumerate(centers):
        Xs.append(r.normal(c, sd, size=(n_each, 2)))
        ys.append(np.full(n_each, k))
    return np.vstack(Xs), np.concatenate(ys)


TRI = [(-2.2, -1.5), (2.2, -1.5), (0.0, 2.3)]        # 삼각형 배치
LINE = [(-3.4, 0.0), (0.0, 0.0), (3.4, 0.0)]         # 일렬 배치

gx = np.linspace(-6.2, 6.2, 420)
gy = np.linspace(-4.6, 4.6, 320)
GX, GY = np.meshgrid(gx, gy)
G = np.column_stack([GX.ravel(), GY.ravel()])

ZONE_CMAP = ListedColormap([ORANGE_F, "#FFFFFF", "#F8C9C9"])
names3 = ["A", "B", "C"]
cols3 = [BLUE, ORANGE, GREEN]
fills3 = [BLUE_F, ORANGE_F, GREEN_F]

fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.9))
report = {}

for ax, centers, ttl, tag in (
        (axes[0], TRI, "삼각형으로 놓이면 자료가 있는 곳은 깨끗하다", "삼각형"),
        (axes[1], LINE, "한 줄로 늘어서면 가운데 B가 자기 영역을 잃는다", "일렬")):
    X, y = make_blobs_at(centers, 100, 7)
    models = ovr_fit(X, y)
    S = ovr_scores(models, G)
    cnt = (S > 0.5).sum(axis=1)
    zone = np.clip(cnt, 0, 2).reshape(GX.shape)

    ax.contourf(GX, GY, zone, levels=[-0.5, 0.5, 1.5, 2.5], cmap=ZONE_CMAP,
                zorder=0)
    for k in range(3):
        w = models[k].coef_[0]
        b0 = models[k].intercept_[0]
        if abs(w[1]) > 1e-6:
            ax.plot(gx, -(w[0] * gx + b0) / w[1], color=cols3[k], lw=1.8,
                    ls="--", zorder=2)
        ax.scatter(X[y == k, 0], X[y == k, 1], s=20, c=fills3[k],
                   edgecolors=cols3[k], linewidths=0.9, zorder=3)

    sb = ovr_scores(models, X)
    cb = (sb > 0.5).sum(axis=1)
    report[tag] = dict(
        none_data=float(np.mean(cb == 0)),
        multi_data=float(np.mean(cb >= 2)),
        maxB_on_B=float(sb[y == 1, 1].max()),
        meanB_on_B=float(sb[y == 1, 1].mean()),
        argmax_acc=float(np.mean(np.argmax(sb, axis=1) == y)))

    ax.set_xlim(-6.2, 6.2)
    ax.set_ylim(-4.6, 4.6)
    ax.set_xlabel("특성 1", fontsize=10.5)
    ax.set_title(ttl, fontsize=11.5, color=INK, pad=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

axes[0].set_ylabel("특성 2", fontsize=10.5)

for k, nm in enumerate(names3):
    axes[0].scatter([], [], s=28, c=fills3[k], edgecolors=cols3[k],
                    linewidths=1.0, label=f"범주 {nm}")
axes[0].legend(loc="upper left", fontsize=9.2, framealpha=0.95, ncol=3)

axes[0].text(-6.0, -4.35,
             "주황 = 아무도 0.5를 넘지 못함   흰색 = 정확히 하나   분홍 = 둘 이상",
             ha="left", va="bottom", fontsize=9.2, color=INK,
             bbox=dict(facecolor="white", edgecolor="none", pad=1.5, alpha=0.9))
axes[0].set_title(
    "삼각형으로 놓이면 자료가 있는 곳은 깨끗하다"
    f"  (B 자료 위 평균 점수 {report['삼각형']['meanB_on_B']:.3f})",
    fontsize=11.5, color=INK, pad=10)

axes[1].annotate(
    "B 자료 위에서도 B 분류기의 최대 점수가 "
    f"{report['일렬']['maxB_on_B']:.3f}\n가운데 범주는 아무도 자기 것이라 하지 않는다",
    xy=(0.0, 1.05), xytext=(0.0, 3.6), fontsize=10, color=RED,
    fontweight="bold", ha="center", va="center",
    bbox=dict(facecolor="white", edgecolor=RED, pad=3.0, alpha=0.95),
    arrowprops=dict(arrowstyle="->", color=RED, lw=1.3))
axes[1].text(6.0, -4.35,
             f"그래도 argmax 정확도는 {report['일렬']['argmax_acc']:.3f}",
             ha="right", va="bottom", fontsize=9.5, color=INK,
             fontweight="bold",
             bbox=dict(facecolor="white", edgecolor="none", pad=1.5, alpha=0.9))

fig.savefig(OUT + "ovr_ambiguous_regions.png", dpi=170, facecolor="white",
            bbox_inches="tight")
plt.close(fig)

for k, v in report.items():
    print(k, {a: round(b, 4) for a, b in v.items()})
print("wrote", OUT + "ovr_ambiguous_regions.png")
