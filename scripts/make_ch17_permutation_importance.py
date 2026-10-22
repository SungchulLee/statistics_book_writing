r"""17.2절 「AI로 보는 통계학」 — 섞어 보기 전에는 중요해 보였다.

Strobl 등($2007$)의 귀무 설계를 그대로 돌린다. $y$ 는 어떤 설명변수와도 **무관한**
동전던지기이고, 설명변수는 연속 하나와 범주 $2 \cdot 4 \cdot 10 \cdot 20$ 개짜리 넷이다.
불순도 중요도는 쪼갤 자리가 많은 변수를 체계적으로 치켜세우고, **열 하나를 섞어 보는**
중요도는 (따로 떼어 둔 자료에서 재면) 모두 $0$ 근처에 머문다.

① 두 중요도 — 같은 귀무 자료, 같은 숲.
② 섞는 자리가 어디냐 — 훈련자료에서 섞으면 순열 중요도도 속는다.
③ 순열 중요도가 못 하는 일 — 상관된 두 열은 서로를 가린다.

만드는 파일: docs/ch17/permutation/img/permutation_importance.png
실행:  python3 scripts/make_ch17_permutation_importance.py   (저장소 최상위에서)
필요:  scikit-learn. 문서 빌드에는 필요하지 않다.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance

OUT = "docs/ch17/permutation/img/permutation_importance.png"
RNG = np.random.default_rng(20261022)

# --- ①② 귀무 설계 -----------------------------------------------------
# Strobl 등(2007): n = 120, y ~ B(0.5), X1 ~ N(0,1), X2..X5 는 범주가
# 2, 4, 10, 20 개. y 는 어느 변수와도 무관하다. (여기서는 범주를 정수로
# 넣으므로 쪼갤 자리가 k-1 개다 — 범주 조합을 모두 보는 구현보다 적다.)
NAMES = ["$X_1$ 연속", "$X_2$ 2범주", "$X_3$ 4범주",
         "$X_4$ 10범주", "$X_5$ 20범주"]
CATS = [0, 2, 4, 10, 20]
N, NTEST, REPS = 120, 1_500, 80


def null_data(rng, n):
    cols = [rng.standard_normal(n)]
    for k in CATS[1:]:
        cols.append(rng.integers(0, k, n).astype(float))
    return np.column_stack(cols), rng.integers(0, 2, n)


imp_gini, imp_test, imp_train = [], [], []
for r in range(REPS):
    Xtr, ytr = null_data(RNG, N)
    Xte, yte = null_data(RNG, NTEST)
    rf = RandomForestClassifier(n_estimators=200, n_jobs=4,
                                random_state=int(RNG.integers(1e9)))
    rf.fit(Xtr, ytr)
    imp_gini.append(rf.feature_importances_)
    seed = int(RNG.integers(1e9))
    imp_test.append(permutation_importance(
        rf, Xte, yte, n_repeats=5, random_state=seed, n_jobs=1).importances_mean)
    imp_train.append(permutation_importance(
        rf, Xtr, ytr, n_repeats=5, random_state=seed, n_jobs=1).importances_mean)
    if (r + 1) % 20 == 0:
        print(f"  귀무 설계 {r + 1}/{REPS} 판")

imp_gini = np.array(imp_gini).mean(axis=0)
imp_test_m = np.array(imp_test).mean(axis=0)
imp_train_m = np.array(imp_train).mean(axis=0)
print("\n귀무 설계에서 받은 평균 중요도 (y 는 어느 변수와도 무관하다)")
for nm, g, t, s in zip(NAMES, imp_gini, imp_train_m, imp_test_m):
    print(f"  {nm:>12s}: 불순도 {g:.3f}   순열-훈련 {t:+.4f}   순열-테스트 {s:+.4f}")
print(f"불순도 중요도의 최대/최소 = {imp_gini.max() / imp_gini.min():.2f} 배")

# --- ③ 상관된 두 열은 서로를 가린다 ------------------------------------
N3, NT3, REPS3 = 600, 1_500, 25
with_copy, without_copy = [], []
for r in range(REPS3):
    x1 = RNG.standard_normal(N3 + NT3)
    x2 = x1 + 0.05 * RNG.standard_normal(N3 + NT3)       # 거의 같은 열
    x3 = RNG.standard_normal(N3 + NT3)                    # 잡음
    y = 2.0 * x1 + 0.5 * RNG.standard_normal(N3 + NT3)
    X = np.column_stack([x1, x2, x3])
    sl, st = slice(0, N3), slice(N3, None)
    seed = int(RNG.integers(1e9))

    rf = RandomForestRegressor(n_estimators=200, n_jobs=4,
                               random_state=seed).fit(X[sl], y[sl])
    with_copy.append(permutation_importance(
        rf, X[st], y[st], n_repeats=5, random_state=seed, n_jobs=1).importances_mean)

    rf2 = RandomForestRegressor(n_estimators=200, n_jobs=4,
                                random_state=seed).fit(X[sl][:, [0, 2]], y[sl])
    without_copy.append(permutation_importance(
        rf2, X[st][:, [0, 2]], y[st], n_repeats=5, random_state=seed, n_jobs=1).importances_mean)

with_copy = np.array(with_copy).mean(axis=0)
without_copy = np.array(without_copy).mean(axis=0)
print("\n상관된 복사본이 있을 때와 없을 때 (y = 2·x₁ + 잡음)")
print(f"  복사본 있음: x₁ {with_copy[0]:.3f}, 복사본 {with_copy[1]:.3f}, "
      f"잡음 {with_copy[2]:.3f}")
print(f"  복사본 없음: x₁ {without_copy[0]:.3f}, 잡음 {without_copy[1]:.3f}")
print(f"  x₁ 의 중요도가 {without_copy[0] / with_copy[0]:.1f} 배로 돌아온다")

# --- 그림 ---------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.6))
C1, C2, C3 = "#1f6f8b", "#c0532b", "#6a8a3a"
pos = np.arange(len(NAMES))

ax = axes[0]
w = 0.38
ax.bar(pos - w / 2, imp_gini, w, color=C2, label="불순도 중요도")
ax.bar(pos + w / 2, imp_test_m, w, color=C1, label="열 하나를 섞어 본 중요도")
ax.axhline(0, color="0.3", lw=1)
ax.set_xticks(pos)
ax.set_xticklabels(NAMES, fontsize=9)
ax.set_ylabel("평균 중요도")
ax.set_title("① y 는 다섯 열 모두와 무관하다", fontsize=11)
ax.legend(fontsize=9)
ax.text(0.03, 0.80, f"불순도: 최대 ÷ 최소 = {imp_gini.max()/imp_gini.min():.1f} 배\n"
        "섞어 본 쪽: 모두 0 근처",
        transform=ax.transAxes, fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.6", alpha=0.93))

ax = axes[1]
ax.bar(pos - w / 2, imp_train_m, w, color=C3, label="훈련자료에서 섞음")
ax.bar(pos + w / 2, imp_test_m, w, color=C1, label="떼어 둔 자료에서 섞음")
ax.axhline(0, color="0.3", lw=1)
ax.set_xticks(pos)
ax.set_xticklabels(NAMES, fontsize=9)
ax.set_ylabel("평균 중요도")
ax.set_title("② 섞는 자리가 어디냐", fontsize=11)
ax.legend(fontsize=9)

ax = axes[2]
lab = ["$x_1$", "$x_1$ 의 복사본", "잡음"]
ax.bar(np.arange(3) - w / 2, with_copy, w, color=C2, label="복사본을 함께 넣음")
ax.bar([0 + w / 2, 2 + w / 2], [without_copy[0], without_copy[1]], w,
       color=C1, label="복사본을 뺌")
ax.axhline(0, color="0.3", lw=1)
ax.set_xticks(np.arange(3))
ax.set_xticklabels(lab, fontsize=10)
ax.set_ylabel("평균 중요도")
ax.set_title("③ 상관된 두 열은 서로를 가린다", fontsize=11)
ax.legend(fontsize=9)
ax.annotate(f"{without_copy[0] / with_copy[0]:.1f} 배",
            xy=(w / 2, without_copy[0]), xytext=(0.55, without_copy[0] * 0.55),
            fontsize=10, color="0.25",
            arrowprops=dict(arrowstyle="->", color="0.45"))

fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"\n저장: {OUT}")
