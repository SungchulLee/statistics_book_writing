r"""선형성은 자료의 성질이 아니라 **표현**의 성질이다.

특징 $20$ 개 가운데 넷에만 비선형으로 기대는 분류 문제를 만든다.

$$y = \mathbf 1\!\left[x_1^2 + x_2^2 + 0.8\,x_3x_4 > \text{중앙값}\right]$$

여기에 은닉층 넷짜리 신경망을 학습시킨 뒤, **층마다** 두 가지 탐침을 올린다 —
로지스틱 회귀(선형)와 RBF 커널 SVM(비선형).

① 입력에서는 선형이 동전 던지기 수준인데 비선형은 잘 맞힌다. 층을 올라가면 선형이
   따라잡는다.
② 둘의 격차가 **「비선형으로 더 짜낼 수 있는 몫」**이고, 그것이 $0$ 으로 간다.
③ 그 대가로 과제와 무관한 정보가 버려진다.

만드는 파일: docs/ch12/assumptions/img/linear_probe.png
"""

import warnings

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression    # noqa: E402
from sklearn.neural_network import MLPClassifier       # noqa: E402
from sklearn.svm import SVC                            # noqa: E402

RNG = np.random.default_rng(7)
N, D = 3000, 20
HID, NLAYER = 32, 4

X = RNG.standard_normal((N, D))
s = X[:, 0] ** 2 + X[:, 1] ** 2 + 0.8 * X[:, 2] * X[:, 3]
y = (s > np.median(s)).astype(int)
# 과제와 무관한 정보 — 마지막 특징의 부호
B = (X[:, D - 1] > 0).astype(int)
i = int(N * 0.6)
tr, te = slice(0, i), slice(i, N)
print(f"표본 {N}, 특징 {D} 개 (그중 넷만 쓰인다), y 와 B 의 상관 "
      f"{abs(np.corrcoef(y, B)[0, 1]):.3f}")

net = MLPClassifier((HID,) * NLAYER, max_iter=2000, random_state=0,
                    alpha=1e-4).fit(X[tr], y[tr])
print(f"신경망 자체의 시험 정확도 {net.score(X[te], y[te]):.4f}")


def activations(Z):
    """입력과 은닉층들의 활성값을 차례로 돌려준다 (ReLU)."""
    out, a = [Z], Z
    for W, b in zip(net.coefs_[:-1], net.intercepts_[:-1]):
        a = np.maximum(a @ W + b, 0.0)
        out.append(a)
    return out


A = activations(X)
print(f"\n{'층':>4}{'차원':>6}{'선형 탐침':>11}{'비선형 탐침':>12}{'격차':>9}"
      f"{'무관한 정보':>12}")
lin, rbf, nui = [], [], []
for L, a in enumerate(A):
    lin.append(LogisticRegression(max_iter=5000)
               .fit(a[tr], y[tr]).score(a[te], y[te]))
    rbf.append(SVC().fit(a[tr], y[tr]).score(a[te], y[te]))
    nui.append(LogisticRegression(max_iter=5000)
               .fit(a[tr], B[tr]).score(a[te], B[te]))
    name = "입력" if L == 0 else f"은닉 {L}"
    print(f"{name:>4}{a.shape[1]:>6}{lin[-1]:>11.4f}{rbf[-1]:>12.4f}"
          f"{rbf[-1] - lin[-1]:>9.4f}{nui[-1]:>12.4f}")
lin, rbf, nui = map(np.array, (lin, rbf, nui))
gap = rbf - lin
print(f"\n격차 {gap[0]:.3f} -> {gap[-1]:.3f}")
print(f"무관한 정보의 선형 복원률 {nui[0]:.3f} -> {nui[-1]:.3f}")

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.4, 4.9))
LS = np.arange(len(A))
NAMES = ["입력"] + [f"은닉 ${k}$" for k in range(1, len(A))]

ax1.fill_between(LS, lin, rbf, color="#ff7f0e", alpha=0.25,
                 label="비선형으로 더 짜낼 수 있는 몫")
ax1.plot(LS, rbf, "s--", color="#d62728", lw=2.4, ms=8, label="비선형 탐침 (RBF)")
ax1.plot(LS, lin, "o-", color="#1f77b4", lw=2.8, ms=9, label="선형 탐침")
ax1.axhline(0.5, color="black", lw=1.3, ls=":")
ax1.text(2.0, 0.515, "동전 던지기", fontsize=10)
ax1.set_xticks(LS)
ax1.set_xticklabels(NAMES, fontsize=10)
ax1.set_ylim(0.42, 0.95)
ax1.set_ylabel("시험 정확도")
ax1.set_title("① 정보는 처음부터 있었다", fontsize=12)
ax1.legend(loc="lower right", fontsize=9.5)
ax1.grid(alpha=0.25)

ax2.bar(LS, gap, 0.6, color=["#d62728"] + ["#ff7f0e"] * (len(LS) - 1))
for L, g in zip(LS, gap):
    ax2.text(L, g + 0.008, f"${g:+.3f}$", ha="center", fontsize=10.5)
ax2.axhline(0, color="black", lw=1.2)
ax2.set_xticks(LS)
ax2.set_xticklabels(NAMES, fontsize=10)
ax2.set_ylim(-0.03, gap.max() * 1.22)
ax2.set_ylabel("비선형 탐침 $-$ 선형 탐침")
ax2.set_title("② 층이 하는 일은 펴는 것이다", fontsize=12)
ax2.grid(alpha=0.25, axis="y")

ax3.plot(LS, nui, "o-", color="#2ca02c", lw=2.8, ms=9)
ax3.axhline(0.5, color="black", lw=1.3, ls=":")
ax3.text(2.0, 0.515, "동전 던지기", fontsize=10)
ax3.annotate(f"${nui[0]:.3f} \\to {nui[-1]:.3f}$",
             xy=(len(LS) - 1, nui[-1]), xytext=(1.1, 0.70),
             fontsize=11.5, color="#2ca02c",
             arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.0))
ax3.set_xticks(LS)
ax3.set_xticklabels(NAMES, fontsize=10)
ax3.set_ylim(0.45, 1.03)
ax3.set_ylabel("과제와 무관한 정보의 선형 복원률")
ax3.set_title("③ 펴는 대신 버린다", fontsize=12)
ax3.grid(alpha=0.25)

fig.tight_layout()
out = "docs/ch12/assumptions/img/linear_probe.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
