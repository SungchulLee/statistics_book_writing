r"""ch20 최적화 절의 개념 그림을 그린다.

만드는 파일:
  ch20/optimization/img/cross_entropy_vs_squared.png   확신하며 틀릴 때 두 손실의 기울기
  ch20/optimization/img/regularization_lambda.png      lambda 가 정확도와 확신에 주는 영향
  ch20/optimization/img/softmax_decision_regions.png   소프트맥스 회귀의 결정경계는 직선

실행:  python3 scripts/make_ch20_optimization_figures.py   (저장소 최상위에서)
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
OUT = "docs/ch20/optimization/img/"
os.makedirs(OUT, exist_ok=True)

INK = "#37474F"
BLUE, BLUE_F = "#1565C0", "#DCEBFB"
ORANGE, ORANGE_F = "#E65100", "#FFE0B2"
GREEN, GREEN_F = "#33691E", "#C5E1A5"
PURPLE = "#6A1B9A"
MUTED = "#90A4AE"
RED = "#D32F2F"


# === 교차엔트로피 vs 제곱손실 ============================================
# 참 범주의 예측확률을 p 라 두면 (이항으로 단순화)
#   교차엔트로피  L = -log p,          |dL/dz| = 1 - p
#   제곱손실      L = (1 - p)^2,       |dL/dz| = 2 p (1 - p)^2
p = np.linspace(1e-3, 1 - 1e-3, 2000)
L_ce = -np.log(p)
L_se = (1 - p) ** 2
g_ce = 1 - p
g_se = 2 * p * (1 - p) ** 2

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.7))

# --- 왼쪽: 손실 자체 -----------------------------------------------------
ax1.axvspan(0, 0.1, color=ORANGE_F, alpha=0.5, zorder=0)
ax1.text(0.125, 4.88, "\u2190 확신하며 틀린 구간", ha="left", va="top", fontsize=9.5,
         color=ORANGE, fontweight="bold")

ax1.plot(p, L_ce, color=BLUE, lw=2.3, label="교차엔트로피")
ax1.plot(p, L_se, color=RED, lw=2.3, ls="--", label="제곱손실")
ax1.axhline(1.0, color=MUTED, lw=1.0, ls=":")
ax1.text(0.97, 1.08, "제곱손실의 상한 1", ha="right", va="bottom", fontsize=9,
         color=MUTED)

ax1.plot(0.01, -np.log(0.01), "o", color=BLUE, ms=6, zorder=5)
ax1.annotate("4.61", xy=(0.01, 4.605), xytext=(0.195, 3.90), fontsize=9.5,
             color=BLUE, fontweight="bold",
             arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.0))
ax1.plot(0.01, (1 - 0.01) ** 2, "o", color=RED, ms=6, zorder=5)
ax1.annotate("0.98", xy=(0.01, 0.98), xytext=(0.20, 1.75), fontsize=9.5,
             color=RED, fontweight="bold",
             arrowprops=dict(arrowstyle="->", color=RED, lw=1.0))

ax1.set_xlim(0, 1)
ax1.set_ylim(0, 5.0)
ax1.set_xlabel(r"참 범주에 부여한 확률 $\hat{p}_y$", fontsize=10.5)
ax1.set_ylabel("손실", fontsize=10.5)
ax1.set_title("벌점: 제곱손실은 아무리 틀려도 1을 못 넘는다", fontsize=11.5,
              color=INK, pad=10)
ax1.legend(loc="upper right", fontsize=10, framealpha=0.95)

# --- 오른쪽: 로짓에 대한 기울기의 크기 -----------------------------------
ax2.axvspan(0, 0.1, color=ORANGE_F, alpha=0.5, zorder=0)
ax2.text(0.125, 1.00, "\u2190 확신하며 틀린 구간", ha="left", va="top", fontsize=9.5,
         color=ORANGE, fontweight="bold")

ax2.plot(p, g_ce, color=BLUE, lw=2.3, label=r"교차엔트로피  $1-\hat{p}_y$")
ax2.plot(p, g_se, color=RED, lw=2.3, ls="--",
         label=r"제곱손실  $2\hat{p}_y(1-\hat{p}_y)^2$")

ax2.plot(0.01, 1 - 0.01, "o", color=BLUE, ms=6, zorder=5)
ax2.plot(0.01, 2 * 0.01 * 0.99 ** 2, "o", color=RED, ms=6, zorder=5)
ax2.annotate("0.990", xy=(0.01, 0.99), xytext=(0.075, 0.600), fontsize=9.5,
             color=BLUE, fontweight="bold",
             arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.0))
ax2.annotate("0.020 — 50배 작다", xy=(0.01, 0.0196), xytext=(0.145, 0.145),
             fontsize=9.5, color=RED, fontweight="bold",
             arrowprops=dict(arrowstyle="->", color=RED, lw=1.0))

ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1.02)
ax2.set_xlabel(r"참 범주에 부여한 확률 $\hat{p}_y$", fontsize=10.5)
ax2.set_ylabel("로짓에 대한 기울기의 크기", fontsize=10.5)
ax2.set_title("학습 신호: 가장 크게 틀린 곳에서 제곱손실은 멈춘다",
              fontsize=11.5, color=INK, pad=10)
ax2.legend(loc="upper right", fontsize=10, framealpha=0.95)

for ax in (ax1, ax2):
    ax.grid(alpha=0.25, ls=":")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

fig.savefig(OUT + "cross_entropy_vs_squared.png", dpi=170, facecolor="white",
            bbox_inches="tight")
plt.close(fig)

for v in (0.01, 0.1, 0.5, 0.9):
    print(f"p={v:.2f}  CE손실 {-np.log(v):.3f}  SE손실 {(1-v)**2:.3f}  "
          f"CE기울기 {1-v:.4f}  SE기울기 {2*v*(1-v)**2:.4f}  "
          f"비 {(1-v)/(2*v*(1-v)**2):.1f}")
print("wrote", OUT + "cross_entropy_vs_squared.png")


# === 정칙화 강도의 효과 (본문 예제 표를 그대로 그린다) ===================
lam_lab = ["0", "0.01", "0.1", "1.0", "10.0"]
xi = np.arange(len(lam_lab))
train = np.array([1.000, 0.967, 0.917, 0.817, 0.550])
test = np.array([0.730, 0.780, 0.780, 0.740, 0.510])
maxp = np.array([0.989, 0.834, 0.656, 0.437, 0.369])
best = test.max()
se = np.sqrt(best * (1 - best) / 100)          # 검정자료 100개의 표준오차
C = 3

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.7))

# --- 왼쪽: 훈련 정확도 vs 검정 정확도 ------------------------------------
ax1.fill_between([-0.4, 4.4], best - se, min(best + se, 1.0),
                 color=GREEN_F, alpha=0.55, zorder=0)
ax1.text(4.32, best + se - 0.008, f"최고 검정 정확도 ±1 표준오차 ({se:.3f})",
         ha="right", va="top", fontsize=9, color=GREEN)

ax1.plot(xi, train, "o-", color=ORANGE, lw=2.2, ms=7, label="훈련 정확도")
ax1.plot(xi, test, "s-", color=BLUE, lw=2.2, ms=7, label="검정 정확도")

for x, (a, b) in enumerate(zip(train, test)):
    ax1.text(x, a + 0.022, f"{a:.3f}", ha="center", va="bottom", fontsize=8.8,
             color=ORANGE, zorder=5,
             bbox=dict(facecolor="white", edgecolor="none", pad=0.8, alpha=0.9))
    ax1.text(x, b - 0.026, f"{b:.3f}", ha="center", va="top", fontsize=8.8,
             color=BLUE)

ax1.annotate("완전 암기(과적합)", xy=(0.05, 1.0), xytext=(0.5, 1.048),
             fontsize=9.5, color=INK, va="center", ha="left",
             arrowprops=dict(arrowstyle="->", color=INK, lw=1.0))
ax1.annotate("과소적합", xy=(4, 0.550), xytext=(3.15, 0.585),
             fontsize=9.5, color=INK, va="center", ha="right",
             arrowprops=dict(arrowstyle="->", color=INK, lw=1.0))

ax1.set_xticks(xi)
ax1.set_xticklabels(lam_lab)
ax1.set_xlim(-0.4, 4.4)
ax1.set_ylim(0.44, 1.08)
ax1.set_xlabel("정칙화 강도 (람다)", fontsize=10.5)
ax1.set_ylabel("정확도", fontsize=10.5)
ax1.set_title("훈련 정확도는 단조 감소, 검정 정확도는 봉우리를 그린다",
              fontsize=11.5, color=INK, pad=10)
ax1.legend(loc="lower left", fontsize=10, framealpha=0.95)

# --- 오른쪽: 확신의 수준 -------------------------------------------------
ax2.bar(xi, maxp, 0.55, color=BLUE_F, edgecolor=BLUE, linewidth=1.4, zorder=2)
for x, v in enumerate(maxp):
    ax2.text(x, v + 0.016, f"{v:.3f}", ha="center", va="bottom", fontsize=9.2,
             color=INK, zorder=4)

ax2.axhline(1 / C, color=RED, ls="--", lw=1.8, zorder=3)
ax2.annotate("균등분포의 최대확률\n1/3 = 0.333", xy=(3.5, 1 / C),
             xytext=(2.95, 0.72), fontsize=9.5, color=RED, fontweight="bold",
             ha="left", va="center",
             arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))

ax2.annotate("정칙화가 없으면\n거의 확률 1을 주장한다", xy=(0, 0.989),
             xytext=(0.7, 0.93), fontsize=9.5, color=INK, va="center",
             ha="left", arrowprops=dict(arrowstyle="->", color=INK, lw=1.0))

ax2.set_xticks(xi)
ax2.set_xticklabels(lam_lab)
ax2.set_xlim(-0.6, 4.6)
ax2.set_ylim(0, 1.12)
ax2.set_xlabel("정칙화 강도 (람다)", fontsize=10.5)
ax2.set_ylabel("최대 예측확률의 평균", fontsize=10.5)
ax2.set_title("같은 정확도라도 확신의 수준은 전혀 다르다", fontsize=11.5,
              color=INK, pad=10)

for ax in (ax1, ax2):
    ax.grid(axis="y", alpha=0.25, ls=":")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

fig.savefig(OUT + "regularization_lambda.png", dpi=170, facecolor="white",
            bbox_inches="tight")
plt.close(fig)

print("표준오차 =", f"{se:.4f}", " 구간 =",
      f"[{best-se:.3f}, {best+se:.3f}]")
print("구간 안에 드는 lambda =",
      [lam_lab[i] for i in range(len(test)) if abs(test[i] - best) <= se])
print("wrote", OUT + "regularization_lambda.png")


# === 소프트맥스 회귀의 결정경계는 직선이다 ==============================
# 본문 학습 루프를 그대로 쓰되, 그림으로 볼 수 있도록 꽃잎 특성 2개만 쓴다.
from sklearn.datasets import load_iris                     # noqa: E402
from sklearn.model_selection import train_test_split       # noqa: E402
from sklearn.preprocessing import StandardScaler           # noqa: E402


def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


def one_hot(y, C):
    Y = np.zeros((y.shape[0], C))
    Y[np.arange(y.shape[0]), y] = 1.0
    return Y


iris = load_iris()
X_all, y_all = iris.data[:, 2:4], iris.target        # 꽃잎 길이, 꽃잎 너비
Cc = 3
Xtr, Xte, ytr, yte = train_test_split(X_all, y_all, test_size=0.3,
                                      random_state=42)
sc = StandardScaler()
Xtr = sc.fit_transform(Xtr)
Xte = sc.transform(Xte)
Ytr = one_hot(ytr, Cc)

np.random.seed(0)
W = np.random.randn(2, Cc) * 0.01
b = np.zeros(Cc)
lr, epochs = 0.5, 200
for _ in range(epochs):
    Yh = softmax(Xtr @ W + b)
    dW = Xtr.T @ (Yh - Ytr) / Xtr.shape[0]
    db = np.mean(Yh - Ytr, axis=0)
    W -= lr * dW
    b -= lr * db

final_loss = -np.sum(Ytr * np.log(softmax(Xtr @ W + b) + 1e-12)) / Xtr.shape[0]
test_acc = np.mean(np.argmax(softmax(Xte @ W + b), axis=1) == yte)

# --- 세 로짓이 모두 같아지는 삼중점 --------------------------------------
A = np.array([W[:, 0] - W[:, 1], W[:, 1] - W[:, 2]])
rhs = -np.array([b[0] - b[1], b[1] - b[2]])
triple = np.linalg.solve(A, rhs)
p_triple = softmax(triple[None, :] @ W + b)[0]

XLO, XHI, YLO, YHI = -3.6, 2.8, -2.4, 3.6
gx = np.linspace(XLO, XHI, 600)
gy = np.linspace(YLO, YHI, 600)
GX, GY = np.meshgrid(gx, gy)
G = np.column_stack([GX.ravel(), GY.ravel()])
lab = np.argmax(G @ W + b, axis=1).reshape(GX.shape)

# --- 절단선 y = 0 을 따라가는 확률 프로파일 ------------------------------
cut_y = 0.0
xs = np.linspace(XLO, XHI, 1200)
Pcut = softmax(np.column_stack([xs, np.full_like(xs, cut_y)]) @ W + b)
cuts = {}
for (j, k) in ((0, 1), (1, 2)):
    w = W[:, j] - W[:, k]
    cuts[(j, k)] = -(b[j] - b[k] + w[1] * cut_y) / w[0]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 5.2))
names = ["setosa", "versicolor", "virginica"]
fills = [BLUE_F, ORANGE_F, GREEN_F]
edges = [BLUE, ORANGE, GREEN]

# --- 왼쪽: 결정영역과 자료 -----------------------------------------------
ax1.contourf(GX, GY, lab, levels=[-0.5, 0.5, 1.5, 2.5], colors=fills,
             alpha=0.8, zorder=0)
for (j, k) in ((0, 1), (1, 2), (0, 2)):
    w = W[:, j] - W[:, k]
    c = b[j] - b[k]
    ax1.plot(gx, -(w[0] * gx + c) / w[1], color=INK, lw=1.2, ls="--",
             alpha=0.8, zorder=2)
for k in range(Cc):
    m = ytr == k
    ax1.scatter(Xtr[m, 0], Xtr[m, 1], s=32, c=fills[k], edgecolors=edges[k],
                linewidths=1.2, zorder=3, label=names[k])

ax1.axhline(cut_y, color=PURPLE, lw=1.6, ls=":", zorder=4)
ax1.text(XHI - 0.1, cut_y + 0.1, "오른쪽 그림의 절단선", ha="right", va="bottom",
         fontsize=9.5, color=PURPLE, fontweight="bold", zorder=5,
         bbox=dict(facecolor="white", edgecolor="none", pad=1.2, alpha=0.88))

ax1.plot(*triple, "*", color=RED, ms=18, zorder=6,
         markeredgecolor="white", markeredgewidth=1.0)
ax1.annotate("삼중점 — 세 직선이 한 점에서 만난다", xy=triple,
             xytext=(-1.55, 3.25), fontsize=9.5, color=RED, fontweight="bold",
             ha="left", va="center", zorder=6,
             arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))

ax1.set_xlim(XLO, XHI)
ax1.set_ylim(YLO, YHI)
ax1.set_xlabel("표준화한 꽃잎 길이", fontsize=10.5)
ax1.set_ylabel("표준화한 꽃잎 너비", fontsize=10.5)
ax1.set_title(f"결정경계는 곧은 직선이다 (검정 정확도 {test_acc:.3f})",
              fontsize=11.5, color=INK, pad=10)
ax1.legend(loc="lower right", fontsize=9.5, framealpha=0.95)

# --- 오른쪽: 절단선 위의 확률 --------------------------------------------
for k in range(Cc):
    ax2.plot(xs, Pcut[:, k], color=edges[k], lw=2.3, label=names[k])

for (pair, xc) in cuts.items():
    ax2.axvline(xc, color=INK, lw=1.2, ls="--", alpha=0.8)
    pv = softmax(np.array([[xc, cut_y]]) @ W + b)[0]
    ax2.plot(xc, pv[pair[0]], "o", color=INK, ms=6, zorder=5)

ax2.axhline(1 / 3, color=MUTED, lw=1.0, ls=":")
ax2.text(XLO + 0.12, 1 / 3 + 0.015, "1/3", ha="left", va="bottom", fontsize=9,
         color=MUTED)

a, bb = cuts[(0, 1)], cuts[(1, 2)]
pv1 = softmax(np.array([[a, cut_y]]) @ W + b)[0]
pv2 = softmax(np.array([[bb, cut_y]]) @ W + b)[0]
ax2.annotate(f"교차점 {pv1[0]:.3f}", xy=(a, pv1[0]), xytext=(a - 0.25, 0.80),
             fontsize=9.5, color=INK, ha="right", va="center",
             arrowprops=dict(arrowstyle="->", color=INK, lw=1.0))
ax2.annotate(f"교차점 {pv2[1]:.3f}", xy=(bb, pv2[1]), xytext=(bb + 0.28, 0.80),
             fontsize=9.5, color=INK, ha="left", va="center",
             arrowprops=dict(arrowstyle="->", color=INK, lw=1.0))

ax2.set_xlim(XLO, XHI)
ax2.set_ylim(0, 1.06)
ax2.set_xlabel("표준화한 꽃잎 길이 (절단선 위)", fontsize=10.5)
ax2.set_ylabel("예측확률", fontsize=10.5)
ax2.set_title("확률은 부드럽게 바뀌지만 경계는 한 점에서 갈린다",
              fontsize=11.5, color=INK, pad=10)
ax2.legend(loc="center left", fontsize=9.5, framealpha=0.95)
ax2.grid(alpha=0.25, ls=":")
ax2.set_axisbelow(True)

for ax in (ax1, ax2):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

fig.savefig(OUT + "softmax_decision_regions.png", dpi=170, facecolor="white",
            bbox_inches="tight")
plt.close(fig)

print("2특성 최종 훈련손실 =", f"{final_loss:.4f}", " 검정 정확도 =",
      f"{test_acc:.4f}")
print("삼중점 =", np.round(triple, 3), " 확률 =", np.round(p_triple, 4))
print("절단선 교차 x =", {k: round(v, 3) for k, v in cuts.items()})
print("교차점 확률 =", np.round(pv1, 4), np.round(pv2, 4))
print("wrote", OUT + "softmax_decision_regions.png")
